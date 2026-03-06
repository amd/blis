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




/*
 * Pthread-based test suite for BLIS thread control API
 * Tests the new global and thread-local thread control variants:
 *   - bli_thread_set_num_threads()       : Sets both global and thread-local
 *   - bli_thread_set_num_threads_local() : Sets thread-local only
 *   - bli_thread_get_num_threads()       : Gets effective thread count
 *   - bli_thread_reset()                 : Resets thread-local to global value
 *   - bli_thread_set_ways()              : Sets loop factorization
 *   - bli_thread_get_is_parallel()       : Checks if parallelism is enabled
 *
 * Compile: gcc test_thread_control_pthread.c -pthread -L../../lib/amdzen -lblis-mt \
 *          -I../../include/amdzen -Wl,-rpath,$(pwd)/../../lib/amdzen -o test_thread_control_pthread
 * Run: ./test_thread_control_pthread [test_number]
 */

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// AOCL BLIS threading functions
extern void bli_thread_set_num_threads(int num_threads);
extern void bli_thread_set_num_threads_local(int num_threads);
extern int bli_thread_get_num_threads(void);
extern void bli_thread_reset(void);
extern void bli_thread_set_ways(int jc, int pc, int ic, int jr, int ir);
extern int bli_thread_get_is_parallel(void);
extern int bli_thread_get_jc_nt(void);
extern int bli_thread_get_ic_nt(void);
extern int bli_thread_get_pc_nt(void);
extern int bli_thread_get_jr_nt(void);
extern int bli_thread_get_ir_nt(void);

// BLAS dgemm for actual operation test
extern void dgemm_(char* transa, char* transb, int* m, int* n, int* k,
                   double* alpha, double* a, int* lda, double* b, int* ldb,
                   double* beta, double* c, int* ldc);

#define MAX_THREADS 16
#define PASS "\033[32mPASS\033[0m"
#define FAIL "\033[31mFAIL\033[0m"

static int tests_passed = 0;
static int tests_failed = 0;

// Thread argument structure
typedef struct {
    int tid;
    int input_value;
    int output_value;
    int iterations;
    double* A;
    double* B;
    double* C;
    int n;
} thread_arg_t;

// Barrier for thread synchronization
static pthread_barrier_t barrier;

void print_separator(const char* title) {
    printf("\n");
    printf("========================================\n");
    printf("  %s\n", title);
    printf("========================================\n");
}

void check_result(const char* test_name, int condition) {
    if (condition) { printf("[%s] %s\n", PASS, test_name); tests_passed++; }
    else { printf("[%s] %s\n", FAIL, test_name); tests_failed++; }
}

void print_info(const char* msg) { printf("[INFO] %s\n", msg); }

// =============================================================================
// TEST 1: Environment variable inheritance
// =============================================================================
void* test_1_thread_func(void* arg) {
    thread_arg_t* targ = (thread_arg_t*)arg;
    targ->output_value = bli_thread_get_num_threads();
    return NULL;
}

void test_1_env_inheritance(void) {
    print_separator("TEST 1: Environment Variable Inheritance");

    int initial_nt = bli_thread_get_num_threads();
    printf("Initial bli_thread_get_num_threads() = %d\n", initial_nt);

    // Launch threads to check they see the same value
    pthread_t threads[4];
    thread_arg_t args[4];

    for (int i = 0; i < 4; i++) {
        args[i].tid = i;
        pthread_create(&threads[i], NULL, test_1_thread_func, &args[i]);
    }

    for (int i = 0; i < 4; i++) {
        pthread_join(threads[i], NULL);
    }

    printf("Threads launched: 4\n");
    for (int i = 0; i < 4; i++) {
        printf("  Thread %d sees: %d\n", i, args[i].output_value);
    }

    // All threads should see the same initial value
    int all_same = 1;
    for (int i = 1; i < 4; i++) {
        if (args[i].output_value != args[0].output_value) {
            all_same = 0;
            break;
        }
    }

    check_result("All threads see same initial thread count", all_same);
    check_result("Initial thread count > 0", initial_nt > 0);
}

// =============================================================================
// TEST 2: Global thread setting propagation
// =============================================================================
void* test_2_thread_func(void* arg) {
    thread_arg_t* targ = (thread_arg_t*)arg;
    targ->output_value = bli_thread_get_num_threads();
    return NULL;
}

void test_2_global_propagation(void) {
    print_separator("TEST 2: Global Setting Propagates to NEW Threads");

    const int EXPECTED_NT = 16;

    bli_thread_set_num_threads(EXPECTED_NT);
    printf("Set global to %d\n", EXPECTED_NT);

    pthread_t threads[4];
    thread_arg_t args[4];

    for (int i = 0; i < 4; i++) {
        args[i].tid = i;
        pthread_create(&threads[i], NULL, test_2_thread_func, &args[i]);
    }

    for (int i = 0; i < 4; i++) {
        pthread_join(threads[i], NULL);
    }

    for (int i = 0; i < 4; i++)
        printf("  Thread %d sees: %d\n", i, args[i].output_value);

    int all_correct = 1;
    for (int i = 0; i < 4; i++)
        if (args[i].output_value != EXPECTED_NT) all_correct = 0;

    check_result("Main thread sees correct value", bli_thread_get_num_threads() == EXPECTED_NT);
    check_result("All threads see global setting", all_correct);
    bli_thread_reset();
}

// =============================================================================
// TEST 3: Local only affects calling thread
// =============================================================================
void* test_3_thread_func(void* arg) {
    thread_arg_t* targ = (thread_arg_t*)arg;
    // Each child thread reads the BLIS thread count
    targ->output_value = bli_thread_get_num_threads();
    return NULL;
}

void test_3_local_only_affects_caller(void) {
    print_separator("TEST 3: Local Setting Only Affects Calling Thread");
    const int GLOBAL_NT = 8, LOCAL_NT = 24;

    // Set global first
    bli_thread_set_num_threads(GLOBAL_NT);

    // Now set local only for main thread
    bli_thread_set_num_threads_local(LOCAL_NT);

    check_result("Main thread sees local override", bli_thread_get_num_threads() == LOCAL_NT);

    // Launch NEW threads - they should see global, not main's local
    pthread_t threads[4];
    thread_arg_t args[4];

    for (int i = 0; i < 4; i++) {
        args[i].tid = i;
        pthread_create(&threads[i], NULL, test_3_thread_func, &args[i]);
    }

    for (int i = 0; i < 4; i++) {
        pthread_join(threads[i], NULL);
    }

    printf("Note: New pthreads should see global value\n");
    for (int i = 0; i < 4; i++) {
        const char* note = (args[i].output_value == LOCAL_NT) ? " (unexpected)" : "";
        printf("  Thread %d sees: %d%s\n", i, args[i].output_value, note);
    }

    int all_see_global = 1;
    for (int i = 0; i < 4; i++)
        if (args[i].output_value != GLOBAL_NT) all_see_global = 0;
    check_result("All new threads see global value", all_see_global);

    bli_thread_reset();
}

// =============================================================================
// TEST 4: Local override precedence
// =============================================================================
void test_4_local_precedence(void) {
    print_separator("TEST 4: Local Override Precedence and Reset");
    const int GLOBAL_NT = 16, LOCAL_NT = 32;

    bli_thread_set_num_threads(GLOBAL_NT);
    check_result("After global set", bli_thread_get_num_threads() == GLOBAL_NT);

    bli_thread_set_num_threads_local(LOCAL_NT);
    check_result("Local overrides global", bli_thread_get_num_threads() == LOCAL_NT);

    bli_thread_reset();
    check_result("Reset restores global", bli_thread_get_num_threads() == GLOBAL_NT);
}

// =============================================================================
// TEST 5: Per-thread local settings
// =============================================================================
void* test_5_thread_func(void* arg) {
    thread_arg_t* targ = (thread_arg_t*)arg;
    bli_thread_set_num_threads_local(targ->input_value);
    targ->output_value = bli_thread_get_num_threads();
    printf("Thread %d: set %d, gets %d\n", targ->tid, targ->input_value, targ->output_value);
    return NULL;
}

void test_5_per_thread_local(void) {
    print_separator("TEST 5: Per-Thread Local Settings");

    bli_thread_set_num_threads(1);
    int local_values[3] = {4, 12, 20};

    pthread_t threads[3];
    thread_arg_t args[3];

    for (int i = 0; i < 3; i++) {
        args[i].tid = i;
        args[i].input_value = local_values[i];
        pthread_create(&threads[i], NULL, test_5_thread_func, &args[i]);
    }

    for (int i = 0; i < 3; i++) {
        pthread_join(threads[i], NULL);
    }

    int all_correct = 1;
    for (int i = 0; i < 3; i++)
        if (args[i].output_value != local_values[i]) all_correct = 0;

    check_result("Each thread sees its own local setting", all_correct);
}

// =============================================================================
// TEST 6: Reset in child threads
// =============================================================================
void* test_6_thread_func(void* arg) {
    thread_arg_t* targ = (thread_arg_t*)arg;
    bli_thread_set_num_threads_local(100 + targ->tid);
    bli_thread_reset();
    targ->output_value = bli_thread_get_num_threads();
    return NULL;
}

void test_6_reset_in_children(void) {
    print_separator("TEST 6: Reset in Child Threads");
    const int GLOBAL_NT = 8;

    bli_thread_set_num_threads(GLOBAL_NT);

    pthread_t threads[3];
    thread_arg_t args[3];

    for (int i = 0; i < 3; i++) {
        args[i].tid = i;
        pthread_create(&threads[i], NULL, test_6_thread_func, &args[i]);
    }

    for (int i = 0; i < 3; i++) {
        pthread_join(threads[i], NULL);
    }

    int reset_works = 1;
    for (int i = 0; i < 3; i++) {
        printf("  Thread %d after reset: %d (expected %d)\n", i, args[i].output_value, GLOBAL_NT);
        if (args[i].output_value != GLOBAL_NT) reset_works = 0;
    }
    check_result("Reset restores global in all threads", reset_works);
}

// =============================================================================
// TEST 7: Thread hierarchy (nested threads via pthread)
// =============================================================================
typedef struct {
    int tid;
    int outer_local;
    int inner_values[2];
} test_7_outer_arg_t;

void* test_7_inner_thread_func(void* arg) {
    thread_arg_t* targ = (thread_arg_t*)arg;
    targ->output_value = bli_thread_get_num_threads();
    printf("    Inner thread %d: sees=%d\n", targ->tid, targ->output_value);
    return NULL;
}

void* test_7_outer_thread_func(void* arg) {
    test_7_outer_arg_t* targ = (test_7_outer_arg_t*)arg;
    int outer_values[3] = {2, 3, 4};

    bli_thread_set_num_threads_local(outer_values[targ->tid]);
    targ->outer_local = bli_thread_get_num_threads();
    printf("  Outer thread %d: local=%d\n", targ->tid, targ->outer_local);

    // Launch inner threads
    pthread_t inner_threads[2];
    thread_arg_t inner_args[2];

    for (int i = 0; i < 2; i++) {
        inner_args[i].tid = i;
        pthread_create(&inner_threads[i], NULL, test_7_inner_thread_func, &inner_args[i]);
    }

    for (int i = 0; i < 2; i++) {
        pthread_join(inner_threads[i], NULL);
        targ->inner_values[i] = inner_args[i].output_value;
    }

    return NULL;
}

void test_7_nested_threads(void) {
    print_separator("TEST 7: Nested Thread Hierarchy");

    const int GLOBAL_NT = 8;
    bli_thread_set_num_threads(GLOBAL_NT);
    int outer_expected[3] = {2, 3, 4};

    pthread_t threads[3];
    test_7_outer_arg_t args[3];

    for (int i = 0; i < 3; i++) {
        args[i].tid = i;
        pthread_create(&threads[i], NULL, test_7_outer_thread_func, &args[i]);
    }

    for (int i = 0; i < 3; i++) {
        pthread_join(threads[i], NULL);
    }

    // Verify outer threads saw their local values
    int outer_correct = 1;
    for (int i = 0; i < 3; i++) {
        if (args[i].outer_local != outer_expected[i]) outer_correct = 0;
    }
    check_result("Outer threads see their local values", outer_correct);

    // Document: Inner threads do NOT inherit parent's local - they see global
    print_info("Note: Inner threads don't inherit parent's local (expected TLS behavior)");
    int inner_valid = 1;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 2; j++) {
            // Inner threads should see global, not parent's local
            if (args[i].inner_values[j] == outer_expected[i]) {
                printf("  Unexpected: Inner[%d.%d] inherited parent local\n", i, j);
                inner_valid = 0;
            }
        }
    }
    check_result("Inner threads have independent TLS (don't inherit parent)", inner_valid);
}

// =============================================================================
// TEST 8: Edge cases
// =============================================================================
void test_8_edge_cases(void) {
    print_separator("TEST 8: Edge Cases");

    bli_thread_set_num_threads(0);
    check_result("Zero becomes 1", bli_thread_get_num_threads() == 1);

    bli_thread_set_num_threads_local(0);
    check_result("Local zero becomes 1", bli_thread_get_num_threads() == 1);

    bli_thread_set_num_threads(1000);
    check_result("Large value accepted", bli_thread_get_num_threads() == 1000);

    bli_thread_reset();
}

// =============================================================================
// TEST 9: bli_thread_set_ways() API
// =============================================================================
void test_9_set_ways(void) {
    print_separator("TEST 9: bli_thread_set_ways() API");

    bli_thread_set_ways(2, 1, 2, 2, 1);
    check_result("Ways (2*1*2*2*1=8)", bli_thread_get_num_threads() == 8);

    bli_thread_set_ways(4, 1, 4, 1, 1);
    check_result("Ways (4*1*4*1*1=16)", bli_thread_get_num_threads() == 16);
}

// =============================================================================
// TEST 10: bli_thread_get_is_parallel() API
// =============================================================================
void test_10_is_parallel(void) {
    print_separator("TEST 10: bli_thread_get_is_parallel() API");

    bli_thread_set_num_threads(1);
    bli_thread_reset();
    check_result("1 thread = not parallel", bli_thread_get_is_parallel() == 0);

    bli_thread_set_num_threads(4);
    check_result("4 threads = parallel", bli_thread_get_is_parallel() == 1);
}
// =============================================================================
// TEST 11: Concurrent global updates (stress test)
// =============================================================================
void* test_11_thread_func(void* arg) {
    thread_arg_t* targ = (thread_arg_t*)arg;
    int bad_count = 0;
    for (int i = 0; i < targ->iterations; i++) {
        bli_thread_set_num_threads(targ->tid + 1);
        int val = bli_thread_get_num_threads();
        // Value should be one of the expected values (1, 2, 3, or 4)
        if (val < 1 || val > 4) {
            bad_count++;
        }
    }
    targ->output_value = bad_count;
    return NULL;
}

void test_11_concurrent_global_updates(void) {
    print_separator("TEST 11: Concurrent Global Updates (Stress Test)");

    bli_thread_set_num_threads(1);

    pthread_t threads[4];
    thread_arg_t args[4];

    for (int i = 0; i < 4; i++) {
        args[i].tid = i;
        args[i].iterations = 100;
        pthread_create(&threads[i], NULL, test_11_thread_func, &args[i]);
    }

    for (int i = 0; i < 4; i++) {
        pthread_join(threads[i], NULL);
    }

    int total_bad = args[0].output_value + args[1].output_value + args[2].output_value + args[3].output_value;
    int final = bli_thread_get_num_threads();
    check_result("All values in expected range (1-4)", total_bad == 0);
    check_result("Final value valid", final >= 1 && final <= 4);
    print_info("Note: Run with -fsanitize=thread to detect actual data races");
}

// TEST 12: DGEMM with different thread settings
// =============================================================================
void test_12_dgemm_with_threads(void) {
    print_separator("TEST 12: DGEMM with Different Thread Settings");

    int n = 100;
    double alpha = 1.0, beta = 0.0;
    double *A = calloc(n * n, sizeof(double));
    double *B = calloc(n * n, sizeof(double));
    double *C = calloc(n * n, sizeof(double));

    for (int i = 0; i < n * n; i++) { A[i] = 1.0; B[i] = 1.0; }

    int thread_counts[] = {1, 2, 4, 8};
    int all_correct = 1;

    for (int t = 0; t < 4; t++) {
        bli_thread_set_num_threads(thread_counts[t]);
        memset(C, 0, n * n * sizeof(double));
        dgemm_("N", "N", &n, &n, &n, &alpha, A, &n, B, &n, &beta, C, &n);

        int correct = (C[0] == (double)n);
        printf("DGEMM with %d threads: %s\n", thread_counts[t], correct ? "PASS" : "FAIL");
        if (!correct) all_correct = 0;
    }

    free(A); free(B); free(C);
    check_result("DGEMM correct with various threads", all_correct);
}

// =============================================================================
// TEST 13: Interleaved global and local settings
// =============================================================================
void test_13_interleaved_settings(void) {
    print_separator("TEST 13: Interleaved Global and Local Settings");

    bli_thread_reset();

    int seq[5] = {0};
    bli_thread_set_num_threads(4);   seq[0] = bli_thread_get_num_threads();
    bli_thread_set_num_threads(8);   seq[1] = bli_thread_get_num_threads();
    bli_thread_set_num_threads_local(12); seq[2] = bli_thread_get_num_threads();
    bli_thread_set_num_threads(16);  seq[3] = bli_thread_get_num_threads();
    bli_thread_reset();              seq[4] = bli_thread_get_num_threads();

    printf("Sequence: %d->%d->%d->%d->%d (expected 4->8->12->16->16)\n",
           seq[0], seq[1], seq[2], seq[3], seq[4]);

    int correct = (seq[0]==4 && seq[1]==8 && seq[2]==12 && seq[3]==16 && seq[4]==16);
    check_result("Sequence correct", correct);
}

// =============================================================================
// TEST 14: Thread count persists after thread exits
// =============================================================================
void* test_14_thread_func(void* arg) {
    // Just a dummy function
    (void)arg;
    return NULL;
}

void test_14_persistence_across_threads(void) {
    print_separator("TEST 14: Thread Count Persists After Thread Exits");

    bli_thread_set_num_threads_local(42);

    // Launch and join a thread
    pthread_t thread;
    pthread_create(&thread, NULL, test_14_thread_func, NULL);
    pthread_join(thread, NULL);

    check_result("tl_rntm persists", bli_thread_get_num_threads() == 42);
    bli_thread_reset();
}

// =============================================================================
// TEST 15: Parallel DGEMM with per-thread settings
// =============================================================================
void* test_15_thread_func(void* arg) {
    thread_arg_t* targ = (thread_arg_t*)arg;
    int n = targ->n;
    double alpha = 1.0, beta = 0.0;

    bli_thread_set_num_threads_local(targ->input_value);
    memset(targ->C, 0, n * n * sizeof(double));
    dgemm_("N", "N", &n, &n, &n, &alpha, targ->A, &n, targ->B, &n, &beta, targ->C, &n);

    targ->output_value = (targ->C[0] == (double)n) ? 1 : 0;
    printf("Thread %d: BLIS=%d, C[0]=%f\n", targ->tid, bli_thread_get_num_threads(), targ->C[0]);
    return NULL;
}

void test_15_parallel_dgemm_different_threads(void) {
    print_separator("TEST 15: Parallel DGEMM with Per-Thread Settings");

    int n = 100;
    double *A = calloc(n * n, sizeof(double));
    double *B = calloc(n * n, sizeof(double));
    double *C1 = calloc(n * n, sizeof(double));
    double *C2 = calloc(n * n, sizeof(double));

    for (int i = 0; i < n * n; i++) { A[i] = 1.0; B[i] = 1.0; }

    bli_thread_set_num_threads(1);

    pthread_t threads[2];
    thread_arg_t args[2];

    args[0].tid = 0; args[0].input_value = 2; args[0].A = A; args[0].B = B; args[0].C = C1; args[0].n = n;
    args[1].tid = 1; args[1].input_value = 4; args[1].A = A; args[1].B = B; args[1].C = C2; args[1].n = n;

    for (int i = 0; i < 2; i++) {
        pthread_create(&threads[i], NULL, test_15_thread_func, &args[i]);
    }

    for (int i = 0; i < 2; i++) {
        pthread_join(threads[i], NULL);
    }

    free(A); free(B); free(C1); free(C2);
    check_result("Parallel DGEMM correct", args[0].output_value && args[1].output_value);
}

// =============================================================================
// TEST 16: Thread reuse behavior with pthread (informational)
// =============================================================================
void* test_16_first_pass_func(void* arg) {
    thread_arg_t* targ = (thread_arg_t*)arg;
    bli_thread_set_num_threads_local(targ->tid + 1);
    targ->output_value = bli_thread_get_num_threads();
    return NULL;
}

void* test_16_second_pass_func(void* arg) {
    thread_arg_t* targ = (thread_arg_t*)arg;
    targ->output_value = bli_thread_get_num_threads();
    return NULL;
}

void test_16_thread_reuse_behavior(void) {
    print_separator("TEST 16: Thread Reuse Behavior (pthread)");

    bli_thread_set_num_threads(4);

    print_info("With pthreads, each pthread_create spawns a NEW thread");
    print_info("Unlike OMP, there's no thread pool reuse by default");

    // First pass - set local values
    pthread_t threads1[4];
    thread_arg_t args1[4];

    for (int i = 0; i < 4; i++) {
        args1[i].tid = i;
        pthread_create(&threads1[i], NULL, test_16_first_pass_func, &args1[i]);
    }

    for (int i = 0; i < 4; i++) {
        pthread_join(threads1[i], NULL);
    }

    printf("First pass values: %d, %d, %d, %d\n",
           args1[0].output_value, args1[1].output_value,
           args1[2].output_value, args1[3].output_value);

    bli_thread_set_num_threads(32);
    printf("Set global to 32\n");

    // Second pass - new threads should see global
    pthread_t threads2[4];
    thread_arg_t args2[4];

    for (int i = 0; i < 4; i++) {
        args2[i].tid = i;
        pthread_create(&threads2[i], NULL, test_16_second_pass_func, &args2[i]);
    }

    for (int i = 0; i < 4; i++) {
        pthread_join(threads2[i], NULL);
    }

    printf("Second pass values: %d, %d, %d, %d\n",
           args2[0].output_value, args2[1].output_value,
           args2[2].output_value, args2[3].output_value);

    int all_32 = 1;
    for (int i = 0; i < 4; i++)
        if (args2[i].output_value != 32) all_32 = 0;

    check_result("New pthreads see updated global", all_32);
}

// =============================================================================
// TEST 17: Reset synchronizes with global
// =============================================================================
void* test_17_thread_func(void* arg) {
    thread_arg_t* targ = (thread_arg_t*)arg;
    bli_thread_reset();
    targ->output_value = bli_thread_get_num_threads();
    return NULL;
}

void test_17_reset_to_sync_global(void) {
    print_separator("TEST 17: Use reset() to Sync Threads with Global");

    bli_thread_set_num_threads(64);
    printf("Set global to 64\n");

    pthread_t threads[4];
    thread_arg_t args[4];

    for (int i = 0; i < 4; i++) {
        args[i].tid = i;
        pthread_create(&threads[i], NULL, test_17_thread_func, &args[i]);
    }

    for (int i = 0; i < 4; i++) {
        pthread_join(threads[i], NULL);
    }

    printf("After reset(): %d, %d, %d, %d\n",
           args[0].output_value, args[1].output_value,
           args[2].output_value, args[3].output_value);

    int all_64 = 1;
    for (int i = 0; i < 4; i++)
        if (args[i].output_value != 64) all_64 = 0;

    check_result("All threads sync to 64 after reset()", all_64);
}

// TEST 18: Concurrent set_num_threads and reset (race condition test)
// =============================================================================
// This test targets the specific race condition between:
// - Thread A: bli_thread_set_num_threads() modifying global_rntm
// - Thread B: bli_thread_reset() reading global_rntm
// The fix adds mutex protection in bli_thread_init_rntm_from_global_rntm().

typedef struct {
    int thread_type;  // 0 = setter, 1 = resetter
    int iterations;
    int bad_count;
} test_18_arg_t;

void* test_18_setter_func(void* arg) {
    test_18_arg_t* targ = (test_18_arg_t*)arg;
    for (int i = 0; i < targ->iterations; i++) {
        // Alternate between values to create contention
        bli_thread_set_num_threads((i % 4) + 1);
    }
    return NULL;
}

void* test_18_resetter_func(void* arg) {
    test_18_arg_t* targ = (test_18_arg_t*)arg;
    int bad = 0;
    for (int i = 0; i < targ->iterations; i++) {
        bli_thread_reset();
        int val = bli_thread_get_num_threads();
        // Value should be valid (1-4 based on setter)
        if (val < 1 || val > 4) {
            bad++;
        }
    }
    targ->bad_count = bad;
    return NULL;
}

void test_18_concurrent_set_and_reset(void) {
    print_separator("TEST 18: Concurrent set_num_threads and reset");

    // Initialize to known state
    bli_thread_set_num_threads(1);

    pthread_t setters[2];
    pthread_t resetters[2];
    test_18_arg_t setter_args[2];
    test_18_arg_t resetter_args[2];

    // Create setter threads
    for (int i = 0; i < 2; i++) {
        setter_args[i].thread_type = 0;
        setter_args[i].iterations = 200;
        pthread_create(&setters[i], NULL, test_18_setter_func, &setter_args[i]);
    }

    // Create resetter threads
    for (int i = 0; i < 2; i++) {
        resetter_args[i].thread_type = 1;
        resetter_args[i].iterations = 200;
        resetter_args[i].bad_count = 0;
        pthread_create(&resetters[i], NULL, test_18_resetter_func, &resetter_args[i]);
    }

    // Wait for all threads
    for (int i = 0; i < 2; i++) {
        pthread_join(setters[i], NULL);
        pthread_join(resetters[i], NULL);
    }

    int total_bad = resetter_args[0].bad_count + resetter_args[1].bad_count;
    int final_val = bli_thread_get_num_threads();

    check_result("All reset values in valid range", total_bad == 0);
    check_result("Final value valid", final_val >= 1 && final_val <= 4);
    print_info("This test targets the set_num_threads/reset race condition");
    print_info("Run with -fsanitize=thread to verify no data races");
}

// =============================================================================
// TEST 19: set_num_threads then set_ways — get_num_threads returns product of ways
// =============================================================================
void test_19_set_nt_then_set_ways(void) {
    print_separator("TEST 19: set_num_threads then set_ways overrides nt");

    // Step 1: Set num_threads to 8
    bli_thread_set_num_threads(8);
    int nt1 = bli_thread_get_num_threads();
    printf("After set_num_threads(8): nt=%d\n", nt1);
    check_result("Step 1: nt=8", nt1 == 8);

    // Step 2: Set ways to 4x1x2x1x1 = 8 (same total, but via ways)
    bli_thread_set_ways(4, 1, 2, 1, 1);
    int nt2 = bli_thread_get_num_threads();
    int jc2 = bli_thread_get_jc_nt();
    int ic2 = bli_thread_get_ic_nt();
    printf("After set_ways(4,1,2,1,1): nt=%d, jc=%d, ic=%d\n", nt2, jc2, ic2);
    check_result("Step 2: nt=8 (from 4*1*2*1*1)", nt2 == 8);
    check_result("Step 2: jc=4", jc2 == 4);
    check_result("Step 2: ic=2", ic2 == 2);

    // Step 3: Set ways to a DIFFERENT total: 2x1x3x1x1 = 6
    // This is the key test - nt must NOT be stale 8
    bli_thread_set_ways(2, 1, 3, 1, 1);
    int nt3 = bli_thread_get_num_threads();
    int jc3 = bli_thread_get_jc_nt();
    int ic3 = bli_thread_get_ic_nt();
    printf("After set_ways(2,1,3,1,1): nt=%d, jc=%d, ic=%d\n", nt3, jc3, ic3);
    check_result("Step 3: nt=6 (from 2*1*3*1*1, not stale 8)", nt3 == 6);
    check_result("Step 3: jc=2", jc3 == 2);
    check_result("Step 3: ic=3", ic3 == 3);

    // Cleanup
    bli_thread_set_num_threads(1);
}

// =============================================================================
// TEST 20: set_ways propagates to new pthread via global_rntm
// =============================================================================
typedef struct {
    int nt;
    int jc;
    int ic;
} test_20_result_t;

void* test_20_child_func(void* arg) {
    test_20_result_t* res = (test_20_result_t*)arg;
    // New thread - tl_rntm initialized from global_rntm
    res->nt = bli_thread_get_num_threads();
    res->jc = bli_thread_get_jc_nt();
    res->ic = bli_thread_get_ic_nt();
    return NULL;
}

void test_20_set_ways_propagates_to_new_threads(void) {
    print_separator("TEST 20: set_ways propagates to new pthreads via global_rntm");

    // Step 1: set_num_threads(12), then set_ways(3,1,2,1,1) = 6
    bli_thread_set_num_threads(12);
    int nt1 = bli_thread_get_num_threads();
    printf("After set_num_threads(12): nt=%d\n", nt1);
    check_result("Step 1: nt=12", nt1 == 12);

    bli_thread_set_ways(3, 1, 2, 1, 1);
    int nt2 = bli_thread_get_num_threads();
    printf("After set_ways(3,1,2,1,1): nt=%d\n", nt2);
    check_result("Step 2: nt=6 (from 3*1*2*1*1)", nt2 == 6);

    // Step 3: Spawn a new pthread - it should inherit ways from global_rntm
    // (not the stale num_threads=12)
    test_20_result_t child_res = { -1, -1, -1 };
    pthread_t child;
    pthread_create(&child, NULL, test_20_child_func, &child_res);
    pthread_join(child, NULL);

    printf("Child thread: nt=%d, jc=%d, ic=%d\n",
           child_res.nt, child_res.jc, child_res.ic);
    check_result("Step 3: child nt=6 (from ways, not stale 12)", child_res.nt == 6);
    check_result("Step 3: child jc=3", child_res.jc == 3);
    check_result("Step 3: child ic=2", child_res.ic == 2);

    // Cleanup
    bli_thread_set_num_threads(1);
}

// =============================================================================
// Main
// =============================================================================
int main(int argc, char** argv) {
    printf("BLIS Thread Control API Test Suite (pthread version)\n");
    printf("=====================================================\n");

    if (argc == 1) {
        test_1_env_inheritance(); test_2_global_propagation();
        test_3_local_only_affects_caller(); test_4_local_precedence();
        test_5_per_thread_local(); test_6_reset_in_children();
        test_7_nested_threads(); test_8_edge_cases();
        test_9_set_ways(); test_10_is_parallel();
        test_11_concurrent_global_updates(); test_12_dgemm_with_threads();
        test_13_interleaved_settings(); test_14_persistence_across_threads();
        test_15_parallel_dgemm_different_threads(); test_16_thread_reuse_behavior();
        test_17_reset_to_sync_global();
        test_18_concurrent_set_and_reset();
        test_19_set_nt_then_set_ways();
        test_20_set_ways_propagates_to_new_threads();
    } else {
        int test_num = atoi(argv[1]);
        switch (test_num) {
            case 0:
                test_1_env_inheritance(); test_2_global_propagation();
                test_3_local_only_affects_caller(); test_4_local_precedence();
                test_5_per_thread_local(); test_6_reset_in_children();
                test_7_nested_threads(); test_8_edge_cases();
                test_9_set_ways(); test_10_is_parallel();
                test_11_concurrent_global_updates(); test_12_dgemm_with_threads();
                test_13_interleaved_settings(); test_14_persistence_across_threads();
                test_15_parallel_dgemm_different_threads(); test_16_thread_reuse_behavior();
                test_17_reset_to_sync_global();
                test_18_concurrent_set_and_reset();
                test_19_set_nt_then_set_ways();
                test_20_set_ways_propagates_to_new_threads();
                break;
            case 1: test_1_env_inheritance(); break;
            case 2: test_2_global_propagation(); break;
            case 3: test_3_local_only_affects_caller(); break;
            case 4: test_4_local_precedence(); break;
            case 5: test_5_per_thread_local(); break;
            case 6: test_6_reset_in_children(); break;
            case 7: test_7_nested_threads(); break;
            case 8: test_8_edge_cases(); break;
            case 9: test_9_set_ways(); break;
            case 10: test_10_is_parallel(); break;
            case 11: test_11_concurrent_global_updates(); break;
            case 12: test_12_dgemm_with_threads(); break;
            case 13: test_13_interleaved_settings(); break;
            case 14: test_14_persistence_across_threads(); break;
            case 15: test_15_parallel_dgemm_different_threads(); break;
            case 16: test_16_thread_reuse_behavior(); break;
            case 17: test_17_reset_to_sync_global(); break;
            case 18: test_18_concurrent_set_and_reset(); break;
            case 19: test_19_set_nt_then_set_ways(); break;
            case 20: test_20_set_ways_propagates_to_new_threads(); break;
            default: printf("Invalid test number: %d\n", test_num); return 1;
        }
    }

    printf("\n========================================\n");
    printf("  SUMMARY\n");
    printf("========================================\n");
    printf("Passed: %d\n", tests_passed);
    printf("Failed: %d\n", tests_failed);
    printf("Total:  %d\n", tests_passed + tests_failed);

    return tests_failed > 0 ? 1 : 0;
}
