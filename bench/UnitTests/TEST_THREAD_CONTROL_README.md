# BLIS Thread Control API Test Suite

## Overview

This test suite validates the new BLIS thread control API introduced in CPUPL-7578. The API provides granular control over thread counts with both global and thread-local variants.

**Two versions are provided:**
- `test_thread_control.c` - OpenMP-based (uses `#pragma omp parallel`)
- `test_thread_control_pthread.c` - pthread-based (uses `pthread_create`/`pthread_join`)

## API Under Test

| Function | Description |
|----------|-------------|
| `bli_thread_set_num_threads(n)` | Sets **both** `global_rntm` and calling thread's `tl_rntm`; **clears ways** and enables auto-factorization |
| `bli_thread_set_num_threads_local(n)` | Sets **only** calling thread's `tl_rntm`; **clears ways** and enables auto-factorization |
| `bli_thread_get_num_threads()` | Returns thread count from `tl_rntm` |
| `bli_thread_reset()` | Resets `tl_rntm` to match `global_rntm` |
| `bli_thread_set_ways(jc,pc,ic,jr,ir)` | Sets loop factorization (product = thread count) |
| `bli_thread_get_is_parallel()` | Returns 1 if thread count > 1, else 0 |
| `bli_thread_get_jc_nt()` | Returns JC loop parallelism from `tl_rntm` |
| `bli_thread_get_ic_nt()` | Returns IC loop parallelism from `tl_rntm` |
| `bli_thread_get_pc_nt()` | Returns PC loop parallelism from `tl_rntm` |
| `bli_thread_get_jr_nt()` | Returns JR loop parallelism from `tl_rntm` |
| `bli_thread_get_ir_nt()` | Returns IR loop parallelism from `tl_rntm` |

## Build & Run

### OpenMP Version

```bash
# Compile
gcc test_thread_control.c -fopenmp -L../../lib/amdzen -lblis-mt \
    -I../../include/amdzen -Wl,-rpath,$(pwd)/../../lib/amdzen \
    -o test_thread_control

# Run all tests (OMP_MAX_ACTIVE_LEVELS=2 required for nested parallel tests)
LD_LIBRARY_PATH=../../lib/amdzen OMP_MAX_ACTIVE_LEVELS=2 ./test_thread_control

# Run specific test (e.g., test 3)
LD_LIBRARY_PATH=../../lib/amdzen OMP_MAX_ACTIVE_LEVELS=2 ./test_thread_control 3
```

### pthread Version

```bash
# Compile
gcc test_thread_control_pthread.c -pthread -L../../lib/amdzen -lblis-mt \
    -I../../include/amdzen -Wl,-rpath,$(pwd)/../../lib/amdzen \
    -o test_thread_control_pthread

# Run all tests
LD_LIBRARY_PATH=../../lib/amdzen ./test_thread_control_pthread

# Run specific test (e.g., test 3)
LD_LIBRARY_PATH=../../lib/amdzen ./test_thread_control_pthread 3
```

---

## Test Cases

### TEST 1: Environment Variable Inheritance

**Purpose:** Verify that BLIS correctly inherits thread count from environment variables (e.g., `OMP_NUM_THREADS`) at initialization.

**Assertions:**
- Initial thread count > 0
- All threads see the same initial value

**Expected Behavior:** Before any explicit API calls, `bli_thread_get_num_threads()` returns the environment-configured value (or a sensible default).

---

### TEST 2: Global Setting Propagates to NEW Threads

**Purpose:** Verify that `bli_thread_set_num_threads()` sets the global thread count and new threads inherit it.

**Setup:**
1. Call `bli_thread_set_num_threads(16)`
2. Launch parallel region with 4 threads
3. Each thread calls `bli_thread_get_num_threads()`

**Assertions:**
- Main thread sees 16
- All child threads see 16

**Key Insight:** New threads read from `global_rntm` on first access, inheriting the globally-set value.

---

### TEST 3: Local Setting Only Affects Calling Thread

**Purpose:** Verify that `bli_thread_set_num_threads_local()` only modifies the calling thread's `tl_rntm`, not `global_rntm`.

**Setup:**
1. Set global to 8, reset thread pool to sync (OMP only)
2. Main thread calls `bli_thread_set_num_threads_local(24)`
3. Launch parallel region

**Assertions:**
- Main thread sees 24 (its local override)
- Other threads see 8 (from global)

**Note (OMP):** Thread 0 may reuse main thread's TLS and see 24. This is expected OMP behavior.
**Note (pthread):** New pthreads always see global value since there's no thread pool reuse.

---

### TEST 4: Local Override Precedence and Reset

**Purpose:** Verify the precedence of local settings and that `bli_thread_reset()` restores global value.

**Setup:**
1. `bli_thread_set_num_threads(16)` → expect 16
2. `bli_thread_set_num_threads_local(32)` → expect 32
3. `bli_thread_reset()` → expect 16

**Assertions:**
- After global set: value is 16
- After local override: value is 32
- After reset: value is 16

---

### TEST 5: Per-Thread Local Settings

**Purpose:** Verify that each thread can independently set its own local thread count.

**Setup:**
1. Launch 3 threads
2. Thread 0 sets local=4, Thread 1 sets local=12, Thread 2 sets local=20
3. Each thread reads back its setting

**Assertions:**
- Each thread sees its own local setting (4, 12, 20 respectively)

---

### TEST 6: Reset in Child Threads

**Purpose:** Verify that child threads can call `bli_thread_reset()` to sync with global.

**Setup:**
1. Set global to 8
2. Launch 3 threads, each sets a different local value
3. Each thread calls `bli_thread_reset()`
4. Each thread reads its value

**Assertions:**
- All threads see 8 after reset

---

### TEST 7: Nested Parallel Regions / Nested Threads

**Purpose:** Test thread-local settings in nested parallel regions (OMP) or nested thread hierarchies (pthread).

**Requirements (OMP):** `OMP_MAX_ACTIVE_LEVELS >= 2`

**Setup:**
1. Set global to 8
2. Launch outer region with 3 threads, each sets local={2,3,4}
3. Each outer thread launches inner threads

**Assertions:**
- Nested regions/threads complete without errors
- Inner threads inherit from `global_rntm` (both pthread and OMP) and do **not** inherit the outer thread's `tl_rntm`

---

### TEST 8: Edge Cases

**Purpose:** Test boundary conditions and edge cases.

**Assertions:**
- Setting 0 threads → becomes 1 (minimum)
- Setting 0 locally → becomes 1
- Large value (1000) is accepted

---

### TEST 9: bli_thread_set_ways() API

**Purpose:** Verify the `bli_thread_set_ways()` function correctly sets thread count as product of loop factors.

**Setup:**
1. `bli_thread_set_ways(2, 1, 2, 2, 1)` → product = 8
2. `bli_thread_set_ways(4, 1, 4, 1, 1)` → product = 16

**Assertions:**
- Thread count matches product of factors

---

### TEST 10: bli_thread_get_is_parallel() API

**Purpose:** Verify the `bli_thread_get_is_parallel()` function.

**Assertions:**
- With 1 thread: returns 0 (not parallel)
- With 4 threads: returns 1 (parallel)

---

### TEST 11: Concurrent Global Updates

**Purpose:** Stress test concurrent access to global settings.

**Setup:**
1. Launch 4 threads
2. Each thread repeatedly sets global and reads back (100 iterations)

**Assertions:**
- No crashes or race conditions
- Thread count remains valid (> 0)

---

### TEST 12: DGEMM with Different Thread Settings

**Purpose:** Verify BLIS operations work correctly with various thread counts.

**Setup:**
1. Run 100×100 DGEMM with thread counts: 1, 2, 4, 8
2. Verify correctness (C[0,0] should equal n=100 for identity-like test)

**Assertions:**
- DGEMM produces correct results at all thread counts

---

### TEST 13: Interleaved Global and Local Settings

**Purpose:** Verify correct behavior when mixing global and local settings.

**Setup:**
1. `bli_thread_set_num_threads(4)` → 4
2. `bli_thread_set_num_threads(8)` → 8
3. `bli_thread_set_num_threads_local(12)` → 12
4. `bli_thread_set_num_threads(16)` → 16 (global AND local both set)
5. `bli_thread_reset()` → 16

**Assertions:**
- Sequence matches expected: 4→8→12→16→16

**Note:** `bli_thread_set_num_threads()` sets both global and local, so after step 4 both are 16. Reset then keeps 16.

---

### TEST 14: Thread Count Persists Across Regions

**Purpose:** Verify that thread-local settings persist across multiple parallel regions or thread exits.

**Setup:**
1. Set local to 42
2. Enter and exit a parallel region (or join a thread)
3. Check main thread's value

**Assertions:**
- Main thread still sees 42

---

### TEST 15: Parallel DGEMM with Per-Thread Settings

**Purpose:** Test concurrent DGEMM operations with different per-thread BLIS thread settings.

**Setup:**
1. Launch 2 threads
2. Thread 0 sets BLIS=2, Thread 1 sets BLIS=4
3. Both run independent DGEMM operations

**Assertions:**
- Both DGEMM operations produce correct results

---

### TEST 16: Thread Pool/Reuse Behavior (Informational)

**Purpose:** Document thread reuse behavior and its interaction with BLIS thread-local storage.

**OMP Observation:** OMP may reuse threads from its pool. These threads retain their `tl_rntm` values from previous parallel regions. If you set global *after* the thread pool was created, reused threads will NOT automatically see the new value.

**pthread Observation:** Each `pthread_create()` spawns a fresh thread. New threads always see the current global value since there's no pool reuse.

**Solution (OMP):** Call `bli_thread_reset()` in threads to synchronize with the updated global.

---

### TEST 17: Use reset() to Sync Threads with Global

**Purpose:** Demonstrate the correct pattern for synchronizing threads with a new global setting.

**Setup:**
1. Set global to 64
2. Launch threads where each calls `bli_thread_reset()`
3. Each thread reads its value

**Assertions:**
- All threads see 64 after reset

**Key Takeaway:** When you need all threads to see a new global value, have them call `bli_thread_reset()`.

---

### TEST 19 (OMP): set_ways → set_num_threads → reset

**Purpose:** Verify that after `set_ways` then `set_num_threads`, a `reset()` correctly
restores the global value (which was updated by `set_num_threads`).

**Setup:**
1. `bli_thread_set_ways(2, 1, 4, 2, 1)` → 16 threads
2. `bli_thread_set_num_threads(8)` → clears ways, sets global+local to 8
3. `bli_thread_reset()` → restores from global

**Assertions:**
- After reset: num_threads=8, jc=-1 (global was cleared by set_num_threads)

---

### TEST 20 (OMP): set_ways → set_num_threads_local → reset

**Purpose:** Verify that `set_num_threads_local()` clears ways locally but does NOT
modify `global_rntm`, so `reset()` reverts to the original global state.

**Setup:**
1. `bli_thread_set_ways(2, 1, 4, 2, 1)` → 16 threads (tl_rntm only)
2. `bli_thread_set_num_threads_local(8)` → clears ways locally, sets local nt=8
3. `bli_thread_reset()` → restores tl_rntm from global_rntm

**Assertions:**
- After reset: nt ≠ 8 (local cleared), nt ≠ 16 (ways cleared), jc = -1
- Actual value depends on environment (e.g., `omp_get_max_threads()`)

**Key Insight:** Neither `set_ways()` nor `set_num_threads_local()` modifies `global_rntm`.
After reset, the thread reverts to whatever global state existed before these calls.

---

### TEST 21 (OMP): num_threads → set_ways → num_threads round-trip

**Purpose:** Verify that switching between `set_num_threads` and `set_ways` works
correctly in both directions.

**Setup:**
1. `bli_thread_set_num_threads(8)` → nt=8, jc=-1
2. `bli_thread_set_ways(2, 1, 2, 2, 1)` → nt=8 (from ways), jc=2
3. `bli_thread_set_num_threads(4)` → nt=4, jc=-1 (cleared)

**Assertions:**
- Each transition correctly updates num_threads and ways state
- set_num_threads always clears ways; set_ways always sets explicit factorization


## Summary

| Test | Description | Checks |
|------|-------------|--------|
| 1 | Env var inheritance | Initial value > 0, consistent across threads |
| 2 | Global propagation | New threads inherit global_rntm |
| 3 | Local-only setting | bli_thread_set_num_threads_local() doesn't affect others |
| 4 | Local precedence & reset | Local overrides global; reset restores |
| 5 | Per-thread locals | Each thread maintains independent local |
| 6 | Reset in children | Child threads can sync to global via reset() |
| 7 | Nested parallel/threads | Thread-local works in nested regions |
| 8 | Edge cases | Zero→1, large values accepted |
| 9 | set_ways() | Loop factorization sets thread count |
| 10 | is_parallel() | Returns 0 for 1 thread, 1 otherwise |
| 11 | Concurrent updates (stress) | Values in expected range; use TSan for race detection |
| 12 | DGEMM correctness | Operations work at various thread counts |
| 13 | Interleaved settings | Correct sequencing of global/local calls |
| 14 | Persistence | Local settings survive region boundaries |
| 15 | Parallel DGEMM | Per-thread settings in concurrent ops |
| 16 | Pool reuse | Documents thread pool behavior |
| 17 | Reset pattern | Demonstrates correct sync pattern |
| 18 (OMP) | Ways vs set_num_threads | set_num_threads clears ways and overrides |
| 18 (pthread) | Concurrent set/reset | Tests set_num_threads/reset race condition |
| 19 (OMP) | set_ways → set_nt → reset | Global correctly preserved through reset |
| 20 (OMP) | set_ways → local → reset | Local-only APIs don't modify global |
| 21 (OMP) | nt → ways → nt round-trip | Bidirectional switching works correctly |

## Important Notes

### OpenMP-Specific

1. **OMP_MAX_ACTIVE_LEVELS**: Set to ≥2 for nested parallel tests (TEST 7)
2. **Thread Pool Reuse**: OMP reuses threads from its pool; they retain their `tl_rntm` from previous regions. Call `bli_thread_reset()` to sync with new global values.
3. **Thread 0 Behavior**: May reuse main thread's TLS, seeing main thread's local value instead of global.

### pthread-Specific

1. **No Thread Pool**: Each `pthread_create()` spawns a fresh thread with no prior `tl_rntm` state.
2. **New Threads See Global**: Unlike OMP, newly created pthreads always inherit from `global_rntm`.
3. **No OMP_MAX_ACTIVE_LEVELS**: Nested threads work without special configuration.
4. **Simpler Behavior**: No thread reuse means Test 3 passes cleanly (all new threads see global).

### General

1. **Thread-Local Storage**: Each thread has its own `tl_rntm` initialized from `global_rntm` on first access.
2. **Zero Handling**: Setting 0 threads is automatically converted to 1.
3. **Reset Pattern**: When threads need to sync with an updated global, call `bli_thread_reset()` in each thread.

---

## Thread Sanitizer (TSan) Testing

### Rebuilding BLIS with Thread Sanitizer

**Important:** The BLIS library itself must be built with TSan instrumentation
for accurate race detection. If only the test code is instrumented, you will see:
```
FATAL: ThreadSanitizer: unexpected memory mapping
```

**Steps to rebuild BLIS with TSan:**
```bash
cd /path/to/blis
make clean
CFLAGS="-fsanitize=thread -g" LDFLAGS="-fsanitize=thread" ./configure --enable-threading=pthreads amdzen
make -j8
```

Note: Use `--enable-threading=pthreads` for TSan testing. OpenMP threading with
TSan may produce false positives due to TSan not fully understanding OpenMP semantics.


### Building Test Code with TSan

To detect data races in the BLIS thread control API, compile with `-fsanitize=thread`:

**OpenMP version:**
```bash
gcc test_thread_control.c -fopenmp -fsanitize=thread -g \
    -L../../lib/amdzen -lblis-mt -I../../include/amdzen \
    -Wl,-rpath,$(pwd)/../../lib/amdzen -o test_thread_control_tsan
```

**pthread version:**
```bash
gcc test_thread_control_pthread.c -pthread -fsanitize=thread -g \
    -L../../lib/amdzen -lblis-mt -I../../include/amdzen \
    -Wl,-rpath,$(pwd)/../../lib/amdzen -o test_thread_control_pthread_tsan
```

### Running with TSan

**Note:** On systems with ASLR, you may see this error:
```
FATAL: ThreadSanitizer: unexpected memory mapping
```

**Solution:** Disable ASLR for the run using `setarch`:
```bash
LD_LIBRARY_PATH=../../lib/amdzen setarch $(uname -m) -R ./test_thread_control_tsan 11
LD_LIBRARY_PATH=../../lib/amdzen setarch $(uname -m) -R ./test_thread_control_pthread_tsan
```

### OpenMP False Positive Issue

When running the OpenMP version with TSan, you may see a warning like:

```
WARNING: ThreadSanitizer: data race (pid=XXXXX)
  Read of size 4 at 0x7fffffffc54c by main thread:
    #0 test_11_concurrent_global_updates ...
  Previous atomic write of size 4 at 0x7fffffffc54c by thread T3:
    #0 test_11_concurrent_global_updates._omp_fn.0 ...
SUMMARY: ThreadSanitizer: data race ... in test_11_concurrent_global_updates
```

**This is a FALSE POSITIVE.** TSan does not fully understand OpenMP's `#pragma omp reduction` semantics. The OpenMP runtime correctly manages the reduction variable (`bad_values`) using atomic operations and proper synchronization, but TSan's analysis doesn't recognize this pattern.

**Why it happens:**
- The test uses `#pragma omp parallel reduction(+:bad_values)`
- Each thread has a private copy of `bad_values`
- At region end, OpenMP atomically combines all values
- TSan sees the main thread reading the final value and worker threads writing to their copies, flagging it as a race

**How to verify it's a false positive:**
1. Run the pthread version with TSan - it reports **no data races**
2. The pthread version tests the same BLIS APIs without OpenMP reduction
3. This confirms the BLIS thread control functions are thread-safe

### TSan Test Results Summary

| Version | Data Races Detected | Notes |
|---------|---------------------|-------|
| pthread | 0 | Clean - no races |
| OpenMP | 1 (false positive) | OMP reduction false positive |

**Recommendation:** Use the pthread version for TSan testing to avoid OpenMP false positives.

### TEST 18 (OMP): Ways vs set_num_threads Interaction

TEST 18 in the OpenMP version verifies that `bli_thread_set_num_threads()` correctly
overrides any prior ways configuration set via `bli_thread_set_ways()` or
`BLIS_JC_NT`/`BLIS_IC_NT` environment variables.

**Setup:**
1. `bli_thread_set_ways(2, 1, 4, 2, 1)` → 16 threads via ways
2. `bli_thread_set_num_threads(8)` → should override to 8

**Assertions:**
- After set_ways: jc=2, ic=4, num_threads=16
- After set_num_threads(8): num_threads=8, jc=-1 (cleared), ic=-1 (cleared)

**Key Insight:** `bli_thread_set_num_threads()` clears all ways to -1 and enables
auto-factorization, ensuring the new thread count takes effect regardless of prior
ways configuration.

### TEST 18 (pthread): Race Condition Stress Test

TEST 18 in the pthread version targets the race condition between:
- `bli_thread_set_num_threads()` - writes to global_rntm (mutex protected)
- `bli_thread_reset()` - reads from global_rntm (mutex protected)

This test runs:
- 2 setter threads: Each calls `bli_thread_set_num_threads()` 200 times
- 2 resetter threads: Each calls `bli_thread_reset()` 200 times

**Running TEST 18 with TSan:**
```bash
# After rebuilding BLIS with TSan (see above)
cd bench/UnitTests
gcc -pthread -fsanitize=thread -g test_thread_control_pthread.c \
    -I../../include/amdzen -L../../lib/amdzen -lblis-mt \
    -o test_thread_control_pthread_tsan.x

# Run with setarch to disable ASLR (required for TSan)
LD_LIBRARY_PATH=../../lib/amdzen setarch $(uname -m) -R ./test_thread_control_pthread_tsan.x 18
```

**Expected result:** No data races detected.

---

## OMP_MAX_ACTIVE_LEVELS Runtime Check

The OpenMP test suite now includes runtime checks for `omp_get_max_active_levels()` to handle different OpenMP configurations gracefully.

### Behavior Summary

| OMP_MAX_ACTIVE_LEVELS | Tests Run | Tests Skipped |
|-----------------------|-----------|---------------|
| ≥ 2 | 21 | 0 |
| 1 (default on some systems) | 17 | 4 (Tests 5, 6, 7, 17 skipped; Tests 2 and 3 run with reduced assertions but are still counted as run) |

### Tests Affected

The following tests require `OMP_MAX_ACTIVE_LEVELS >= 2` for proper thread spawning and TLS isolation:

| Test | Behavior when ACTIVE_LEVELS < 2 |
|------|--------------------------------|
| **Test 2** | Runs, but skips thread propagation assertion |
| **Test 3** | Runs, but skips thread isolation assertion |
| **Test 5** | Skipped entirely |
| **Test 6** | Skipped entirely |
| **Test 7** | Skipped entirely (existing behavior) |
| **Test 17** | Skipped entirely |

### Why This Matters

When `OMP_MAX_ACTIVE_LEVELS=1`:
- OpenMP may not spawn actual worker threads in parallel regions
- Thread-local storage may behave unexpectedly
- The OMP runtime may serialize parallel regions

**Recommendation:** Set `OMP_MAX_ACTIVE_LEVELS=2` (or higher) for full test coverage:
```bash
LD_LIBRARY_PATH=../../lib/amdzen OMP_MAX_ACTIVE_LEVELS=2 ./test_thread_control
```
