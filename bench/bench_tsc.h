/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name(s) of the copyright holder(s) nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#ifndef BENCH_TSC_H
#define BENCH_TSC_H

#if defined(BENCH_TSC)
#error "Use BENCH_RDTSC or BENCH_RDTSCP instead of BENCH_TSC"
#endif

#if defined(BENCH_RDTSC) || defined(BENCH_RDTSCP)

/*
 * This optional timer uses the x86 time-stamp counter (TSC), a 64-bit counter
 * that increments continuously. On modern invariant-TSC processors it runs at
 * a constant reference rate, independent of turbo boost and power states.
 * Therefore, TSC ticks are timing ticks, not necessarily current core cycles.
 * Reading the TSC is much cheaper than calling clock_gettime() for every
 * benchmark sample.
 *
 * BENCH_RDTSC reads only the counter. BENCH_RDTSCP also reads IA32_TSC_AUX, an
 * OS-initialized tag that normally identifies the current logical processor.
 * Comparing this tag at the start and end lets the benchmark reject a sample
 * if the OS moved the thread while it was being timed.
 */
#if defined(BENCH_RDTSC) && defined(BENCH_RDTSCP)
#error "BENCH_RDTSC and BENCH_RDTSCP are mutually exclusive timing modes"
#endif

#if !defined(__i386__) && !defined(__x86_64__)
#error "BENCH_RDTSC/BENCH_RDTSCP requires an x86 processor"
#endif

#include <cpuid.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <x86intrin.h>

/*
 * Starting IA32_TSC_AUX value for the current sample. Its exact bit layout is
 * OS-defined; only equality with the ending value is required here.
 */
static unsigned bench_tsc_start_aux = 0;

/*
 * Read the TSC and form a timestamp boundary around the benchmarked operation.
 * x86 processors execute instructions out of order, and RDTSC itself is not a
 * serializing instruction. Without barriers, nearby work could execute on the
 * wrong side of a timestamp and make the measured interval too short.
 *
 * LFENCE before the read orders earlier work before the timestamp. LFENCE after
 * the read prevents later work from starting before the timestamp completes.
 * RDTSCP provides stronger ordering for earlier work than RDTSC and also reads
 * IA32_TSC_AUX, but it still needs the trailing fence to order later work. The
 * same fence pattern is used in both modes to keep their boundaries symmetric.
 */
static inline uint64_t bench_tsc_read(unsigned* aux)
{
    /* __rdtscp() requires storage for TSC_AUX, so provide scratch space if the
       caller only wants the timestamp. */
    unsigned aux_local;
    if (aux == NULL)
        aux = &aux_local;

#if defined(BENCH_RDTSCP)
    /* __rdtscp() returns TSC and writes IA32_TSC_AUX through aux. */
    _mm_lfence();
    uint64_t cycles = __rdtscp(aux);
    _mm_lfence();
    return cycles;
#else
    /* __rdtsc() returns TSC only, so migration cannot be checked in this mode. */
    (void)aux;
    _mm_lfence();
    uint64_t cycles = __rdtsc();
    _mm_lfence();
    return cycles;
#endif
}

/*
 * Return time from the best monotonic clock exposed by the platform. Prefer
 * CLOCK_MONOTONIC_RAW because it is not adjusted by NTP; otherwise use
 * CLOCK_MONOTONIC, which still cannot jump when calendar time changes. Either
 * clock is suitable as a reference when CPUID does not publish the TSC rate.
 */
#if defined(CLOCK_MONOTONIC_RAW) || defined(CLOCK_MONOTONIC)
static inline double bench_tsc_raw_seconds(void)
{
#if defined(CLOCK_MONOTONIC_RAW)
    const clockid_t clock_id = CLOCK_MONOTONIC_RAW;
    const char* clock_name = "clock_gettime(CLOCK_MONOTONIC_RAW)";
#else
    const clockid_t clock_id = CLOCK_MONOTONIC;
    const char* clock_name = "clock_gettime(CLOCK_MONOTONIC)";
#endif
    struct timespec ts;
    if (clock_gettime(clock_id, &ts) != 0)
    {
        /* Calibration cannot continue safely without a valid reference time. */
        perror(clock_name);
        exit(EXIT_FAILURE);
    }

    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}
#else
static inline double bench_tsc_raw_seconds(void)
{
    fprintf(stderr,
            "bench_tsc: no monotonic clock is available; cannot calibrate TSC\n");
    exit(EXIT_FAILURE);
}
#endif

/*
 * Determine how many TSC ticks occur per second, which is needed to convert a
 * TSC difference into elapsed seconds. This is the TSC reference frequency,
 * not the core's instantaneous operating frequency.
 *
 * CPUID is an x86 instruction that reports processor capabilities and fixed
 * configuration data. A CPUID "leaf" is a function number placed in EAX before
 * executing the instruction; results are returned in EAX, EBX, ECX, and EDX.
 * Leaf names are traditionally written in hexadecimal with an H suffix, so
 * CPUID.15H means leaf 0x15 and CPUID.16H means leaf 0x16.
 *
 * Prefer CPUID data because it avoids runtime measurement error. If the needed
 * leaves or fields are unavailable, measure TSC against CLOCK_MONOTONIC_RAW.
 * Cache the chosen rate so discovery runs once and all samples use the same
 * conversion factor.
 */
static inline double bench_tsc_frequency_hz(void)
{
    static double frequency_hz = 0.0;

    if (frequency_hz == 0.0)
    {
        unsigned int eax, ebx, ecx, edx;
        /*
         * CPUID leaf 0 reports the largest supported basic leaf. Check it
         * before asking for optional leaves 0x15 or 0x16.
         */
        unsigned int max_leaf = __get_cpuid_max(0, NULL);

        /*
         * CPUID leaf 0x15, subleaf 0, describes the ratio between the TSC and
         * the processor's fixed reference crystal clock:
         *
         *   EAX = denominator of the TSC/crystal ratio
         *   EBX = numerator of the TSC/crystal ratio
         *   ECX = reference crystal frequency in Hz
         *   EDX = reserved for this calculation
         *
         * Thus: TSC Hz = ECX * EBX / EAX. A zero ratio or frequency means the
         * processor did not enumerate enough information, so try leaf 0x16.
         */
        if (max_leaf >= 0x15 &&
            __get_cpuid_count(0x15, 0, &eax, &ebx, &ecx, &edx) &&
            eax != 0 && ebx != 0 && ecx != 0)
        {
            frequency_hz = (double)ecx * (double)ebx / (double)eax;
        }
        /*
         * CPUID leaf 0x16 reports nominal processor frequency information:
         *
         *   EAX = base frequency in MHz
         *   EBX = maximum frequency in MHz
         *   ECX = bus/reference frequency in MHz
         *
         * It does not report the current turbo or power-state frequency, nor
         * does it provide the explicit TSC ratio available in leaf 0x15. The
         * base frequency is therefore used only as a next-best TSC-rate
         * estimate. Multiplying by 1,000,000 converts MHz to Hz.
         */
        else if (max_leaf >= 0x16 &&
                 __get_cpuid(0x16, &eax, &ebx, &ecx, &edx) && eax != 0)
        {
            frequency_hz = (double)eax * 1.0e6;
        }
        else
        {
            /*
             * Last resort: read both clocks, wait for at least 50 ms of raw
             * elapsed time, then read the TSC again. Dividing elapsed TSC ticks
             * by elapsed seconds estimates ticks per second. The 50 ms window
             * makes the overhead and resolution of individual clock reads a
             * small fraction of the calibration interval. This busy wait is
             * acceptable because calibration happens only on the first call.
             */
            double start_seconds = bench_tsc_raw_seconds();
            uint64_t start_cycles = bench_tsc_read(NULL);
            double end_seconds;

            do
            {
                end_seconds = bench_tsc_raw_seconds();
            }
            while (end_seconds - start_seconds < 0.05);

            frequency_hz = (double)(bench_tsc_read(NULL) - start_cycles) /
                           (end_seconds - start_seconds);
        }
    }

    return frequency_hz;
}

/*
 * Cache the conversion factor from one TSC tick to seconds. Computing the
 * reciprocal once keeps floating-point division out of repeated timer calls;
 * converting a timestamp then requires only a multiplication.
 */
static inline double bench_tsc_seconds_per_tick(void)
{
    static double seconds_per_tick = 0.0;

    if (seconds_per_tick == 0.0)
        seconds_per_tick = 1.0 / bench_tsc_frequency_hz();

    return seconds_per_tick;
}

/*
 * Convert the current TSC value to seconds. The absolute value has no useful
 * epoch; callers subtract a starting value to obtain elapsed time. RDTSCP mode
 * also saves the starting processor tag for the migration check below.
 */
static inline double bench_tsc_clock(void)
{
    return (double)bench_tsc_read(&bench_tsc_start_aux) * bench_tsc_seconds_per_tick();
}

/*
 * Finish one timed sample. The benchmark repeats an operation and keeps its
 * minimum valid duration to reduce interference from preemption and interrupts.
 * time_min is the best duration so far; time_start came from bench_tsc_clock().
 */
static inline double bench_tsc_clock_min_diff(double time_min, double time_start)
{
    unsigned end_aux = 0;
    double time_diff = (double)bench_tsc_read(&end_aux) * bench_tsc_seconds_per_tick() - time_start;

#if defined(BENCH_RDTSCP)
    /*
     * A different TSC_AUX tag means the thread changed logical processors.
     * Reject that sample because counters may not be perfectly synchronized
     * across processors and migration adds scheduler/cache disturbance.
     */
    if (end_aux != bench_tsc_start_aux)
        return time_min;
#endif

    /* Accept only a positive duration that improves the current minimum. */
    if (time_diff > 1.0e-9 && time_diff < time_min)
        return time_diff;

    return time_min;
}

/* Make existing benchmark code use this timer without changing its call sites. */
#define bli_clock bench_tsc_clock
#define bli_clock_min_diff bench_tsc_clock_min_diff

#endif

#endif
