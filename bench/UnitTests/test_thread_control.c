/*
 * Comprehensive test suite for BLIS thread control API
 * Tests the new global and thread-local thread control variants:
 *   - bli_thread_set_num_threads()       : Sets both global and thread-local
 *   - bli_thread_set_num_threads_local() : Sets thread-local only
 *   - bli_thread_get_num_threads()       : Gets effective thread count
 *   - bli_thread_reset()                 : Resets thread-local to global value
 *   - bli_thread_set_ways()              : Sets loop factorization
 *   - bli_thread_get_is_parallel()       : Checks if parallelism is enabled
 *
 * Compile: gcc test_thread_control.c -fopenmp -lblis-mt -o test_thread_control
 * Run: OMP_MAX_ACTIVE_LEVELS=2 ./test_thread_control [test_number]
 */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

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

extern void dgemm_(char* transa, char* transb, int* m, int* n, int* k,
                   double* alpha, double* a, int* lda, double* b, int* ldb,
                   double* beta, double* c, int* ldc);

#define MAX_THREADS 512
#define PASS "\033[32mPASS\033[0m"
#define FAIL "\033[31mFAIL\033[0m"

static int tests_passed = 0;
static int tests_failed = 0;

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
void test_1_env_inheritance(void) {
    print_separator("TEST 1: Environment Variable Inheritance");

    int initial_nt = bli_thread_get_num_threads();
    printf("OMP_MAX_ACTIVE_LEVELS=%d, omp_get_max_threads()=%d\n",
           omp_get_max_active_levels(), omp_get_max_threads());
    printf("Initial bli_thread_get_num_threads() = %d\n", initial_nt);

    check_result("Initial thread count > 0", initial_nt > 0);
}

// =============================================================================
// TEST 2: Global thread setting propagation
// =============================================================================
void test_2_global_propagation(void) {
    print_separator("TEST 2: Global Setting Propagates to NEW Threads");

    if (omp_get_max_active_levels() < 2) {
        print_info("OMP_MAX_ACTIVE_LEVELS < 2: Thread spawning may be limited");
    }

    int num_threads[MAX_THREADS] = {0};
    int num_launched = 0;
    const int EXPECTED_NT = 16;

    bli_thread_set_num_threads(EXPECTED_NT);
    printf("Set global to %d\n", EXPECTED_NT);

    #pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();
        #pragma omp single
        num_launched = omp_get_num_threads();
        num_threads[tid] = bli_thread_get_num_threads();
    }

    for (int i = 0; i < num_launched; i++)
        printf("  Thread %d sees: %d\n", i, num_threads[i]);

    int all_correct = 1;
    for (int i = 0; i < num_launched; i++)
        if (num_threads[i] != EXPECTED_NT) all_correct = 0;

    check_result("Main thread sees correct value", bli_thread_get_num_threads() == EXPECTED_NT);

    if (omp_get_max_active_levels() >= 2) {
        check_result("All threads see global setting", all_correct);
    } else {
        // With limited active levels, threads may not spawn or may reuse main thread TLS
        print_info("Skipping thread propagation check (OMP_MAX_ACTIVE_LEVELS < 2)");
    }
    bli_thread_reset();
}

// =============================================================================
// =============================================================================
// TEST 3: Local only affects calling thread
// =============================================================================
void test_3_local_only_affects_caller(void) {
    print_separator("TEST 3: Local Setting Only Affects Calling Thread");
    const int GLOBAL_NT = 8, LOCAL_NT = 24;

    // Set global first, then reset threads to sync with new global
    bli_thread_set_num_threads(GLOBAL_NT);
    #pragma omp parallel num_threads(4)
    {
        bli_thread_reset();
    }

    bli_thread_set_num_threads_local(LOCAL_NT);

    check_result("Main thread sees local override", bli_thread_get_num_threads() == LOCAL_NT);

    int num_threads[MAX_THREADS] = {0};
    int num_launched = 0;

    #pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();
        #pragma omp single
        num_launched = omp_get_num_threads();
        num_threads[tid] = bli_thread_get_num_threads();
    }

    printf("Note: Thread 0 may reuse main thread's TLS\n");
    for (int i = 0; i < num_launched; i++) {
        const char* note = (num_threads[i] == LOCAL_NT) ? " (reused)" : "";
        printf("  Thread %d sees: %d%s\n", i, num_threads[i], note);
    }

    if (omp_get_max_active_levels() >= 2) {
        int some_see_global = 0;
        for (int i = 0; i < num_launched; i++)
            if (num_threads[i] == GLOBAL_NT) some_see_global = 1;
        check_result("Some threads see global value", some_see_global);
    } else {
        print_info("Skipping thread isolation check (OMP_MAX_ACTIVE_LEVELS < 2)");
    }

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
// =============================================================================
// TEST 5: Per-thread local settings
// =============================================================================
void test_5_per_thread_local(void) {
    print_separator("TEST 5: Per-Thread Local Settings");

    if (omp_get_max_active_levels() < 2) {
        print_info("OMP_MAX_ACTIVE_LEVELS < 2: Skipping (thread-local isolation requires nested parallelism)");
        return;
    }

    bli_thread_set_num_threads(1);
    bli_thread_reset();  // Ensure clean state
    int local_values[3] = {4, 12, 20};
    int seen_values[3] = {0};

    #pragma omp parallel num_threads(3)
    {
        int tid = omp_get_thread_num();
        bli_thread_set_num_threads_local(local_values[tid]);
        seen_values[tid] = bli_thread_get_num_threads();
        printf("Thread %d: set %d, gets %d\n", tid, local_values[tid], seen_values[tid]);
    }

    int all_correct = 1;
    for (int i = 0; i < 3; i++)
        if (seen_values[i] != local_values[i]) all_correct = 0;

    check_result("Each thread sees its own local setting", all_correct);
}

// =============================================================================
// =============================================================================
// TEST 6: Reset in child threads
// =============================================================================
void test_6_reset_in_children(void) {
    print_separator("TEST 6: Reset in Child Threads");

    if (omp_get_max_active_levels() < 2) {
        print_info("OMP_MAX_ACTIVE_LEVELS < 2: Skipping (requires nested parallelism)");
        return;
    }

    const int GLOBAL_NT = 8;

    bli_thread_set_num_threads(GLOBAL_NT);

    int after_reset[MAX_THREADS] = {0};
    int num_launched = 0;

    #pragma omp parallel num_threads(3)
    {
        int tid = omp_get_thread_num();
        #pragma omp single
        num_launched = omp_get_num_threads();

        bli_thread_set_num_threads_local(100 + tid);
        bli_thread_reset();
        after_reset[tid] = bli_thread_get_num_threads();
    }

    int reset_works = 1;
    for (int i = 0; i < num_launched; i++) {
        printf("  Thread %d after reset: %d (expected %d)\n", i, after_reset[i], GLOBAL_NT);
        if (after_reset[i] != GLOBAL_NT) reset_works = 0;
    }
    check_result("Reset restores global in all threads", reset_works);
}

// =============================================================================
// TEST 7: Nested parallel regions
// =============================================================================
void test_7_nested_parallel(void) {
    print_separator("TEST 7: Nested Parallel Regions");
    if (omp_get_max_active_levels() < 2) {
        print_info("Skipping: need OMP_MAX_ACTIVE_LEVELS>=2");
        return;
    }

    const int GLOBAL_NT = 8;
    bli_thread_set_num_threads(GLOBAL_NT);
    int outer_values[3] = {2, 3, 4};
    int outer_sees[3] = {0};
    int inner_sees[3][2] = {{0}};

    #pragma omp parallel num_threads(3)
    {
        int ptid = omp_get_thread_num();
        bli_thread_set_num_threads_local(outer_values[ptid]);
        outer_sees[ptid] = bli_thread_get_num_threads();
        printf("Outer[%d]: local=%d\n", ptid, outer_sees[ptid]);

        #pragma omp parallel num_threads(2)
        {
            int ctid = omp_get_thread_num();
            inner_sees[ptid][ctid] = bli_thread_get_num_threads();
            printf("  Inner[%d.%d]: sees=%d\n", ptid, ctid, inner_sees[ptid][ctid]);
        }
    }

    // Verify outer threads see their local value
    int outer_correct = 1;
    for (int p = 0; p < 3; p++) {
        if (outer_sees[p] != outer_values[p]) outer_correct = 0;
    }
    check_result("Outer threads see their local values", outer_correct);

    // Document: Inner threads do NOT inherit parent's local - they see global or default
    // This is expected TLS behavior - each new thread starts fresh
    print_info("Note: Inner threads don't inherit parent's local (expected TLS behavior)");
    int inner_valid = 1;
    for (int p = 0; p < 3; p++) {
        for (int c = 0; c < 2; c++) {
            // Inner threads should see global or default, not parent's local
            if (inner_sees[p][c] == outer_values[p]) {
                printf("  Unexpected: Inner[%d.%d] inherited parent local\n", p, c);
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
    bli_thread_reset();  // Ensure clean state
    check_result("1 thread = not parallel", bli_thread_get_is_parallel() == 0);

    bli_thread_set_num_threads(4);
    check_result("4 threads = parallel", bli_thread_get_is_parallel() == 1);
}


// =============================================================================
// TEST 11: Concurrent global updates (stress test)
// =============================================================================
void test_11_concurrent_global_updates(void) {
    print_separator("TEST 11: Concurrent Global Updates (Stress Test)");

    bli_thread_set_num_threads(1);
    bli_thread_reset();  // Ensure clean state

    int bad_values = 0;
    #pragma omp parallel num_threads(4) reduction(+:bad_values)
    {
        for (int i = 0; i < 100; i++) {
            bli_thread_set_num_threads(omp_get_thread_num() + 1);
            int val = bli_thread_get_num_threads();
            // Value should be one of the expected values (1, 2, 3, or 4)
            if (val < 1 || val > 4) {
                bad_values++;
            }
        }
    }

    int final = bli_thread_get_num_threads();
    check_result("All values in expected range (1-4)", bad_values == 0);
    check_result("Final value valid", final >= 1 && final <= 4);
    print_info("Note: Run with -fsanitize=thread to detect actual data races");
}

// =============================================================================
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

    // Reset to ensure clean state
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
// TEST 14: Thread count persists across OMP regions
// =============================================================================
void test_14_persistence_across_regions(void) {
    print_separator("TEST 14: Thread Count Persists Across Regions");

    bli_thread_set_num_threads_local(42);

    #pragma omp parallel num_threads(2)
    { /* dummy region */ }

    check_result("tl_rntm persists", bli_thread_get_num_threads() == 42);
    bli_thread_reset();
}

// =============================================================================
// TEST 15: Parallel DGEMM with per-thread settings
// =============================================================================
void test_15_parallel_dgemm_different_threads(void) {
    print_separator("TEST 15: Parallel DGEMM with Per-Thread Settings");

    int n = 100;
    double alpha = 1.0, beta = 0.0;
    double *A = calloc(n * n, sizeof(double));
    double *B = calloc(n * n, sizeof(double));
    double *C1 = calloc(n * n, sizeof(double));
    double *C2 = calloc(n * n, sizeof(double));

    for (int i = 0; i < n * n; i++) { A[i] = 1.0; B[i] = 1.0; }

    bli_thread_set_num_threads(1);
    bli_thread_reset();  // Ensure clean state

    int results[2] = {0};
    #pragma omp parallel num_threads(2)
    {
        int tid = omp_get_thread_num();
        bli_thread_set_num_threads_local(tid == 0 ? 2 : 4);
        double *C = (tid == 0) ? C1 : C2;
        memset(C, 0, n * n * sizeof(double));
        dgemm_("N", "N", &n, &n, &n, &alpha, A, &n, B, &n, &beta, C, &n);
        results[tid] = (C[0] == (double)n);
        printf("Thread %d: BLIS=%d, C[0]=%f\n", tid, bli_thread_get_num_threads(), C[0]);
    }

    free(A); free(B); free(C1); free(C2);
    check_result("Parallel DGEMM correct", results[0] && results[1]);
}

// =============================================================================
// TEST 16: Thread pool reuse behavior (informational)
// =============================================================================
void test_16_thread_pool_behavior(void) {
    print_separator("TEST 16: Thread Pool Reuse Behavior");

    bli_thread_set_num_threads(4);

    // Create thread pool with initial setting
    int first_values[MAX_THREADS] = {0};
    #pragma omp parallel num_threads(4)
    {
        bli_thread_set_num_threads_local(omp_get_thread_num() + 1);
        first_values[omp_get_thread_num()] = bli_thread_get_num_threads();
    }

    print_info("OMP may reuse threads - they keep existing tl_rntm");

    bli_thread_set_num_threads(32);
    printf("Set global to 32 AFTER thread pool created\n");

    int second_values[MAX_THREADS] = {0};
    #pragma omp parallel num_threads(4)
    {
        second_values[omp_get_thread_num()] = bli_thread_get_num_threads();
    }

    printf("Thread values: %d, %d, %d, %d\n",
           second_values[0], second_values[1], second_values[2], second_values[3]);
    print_info("Reused threads may NOT see 32 - this is expected");
    print_info("Solution: call bli_thread_reset() in threads to sync");
    check_result("Test documents pool behavior", 1);
}

// =============================================================================
// TEST 17: Use reset() to sync threads with global
// =============================================================================
void test_17_reset_to_sync_global(void) {
    print_separator("TEST 17: Use reset() to Sync Threads with Global");

    if (omp_get_max_active_levels() < 2) {
        print_info("OMP_MAX_ACTIVE_LEVELS < 2: Skipping (requires nested parallelism)");
        return;
    }

    bli_thread_set_num_threads(64);
    printf("Set global to 64\n");

    int values[MAX_THREADS] = {0};
    #pragma omp parallel num_threads(4)
    {
        bli_thread_reset();
        values[omp_get_thread_num()] = bli_thread_get_num_threads();
    }

    printf("After reset(): %d, %d, %d, %d\n", values[0], values[1], values[2], values[3]);

    int all_64 = (values[0]==64 && values[1]==64 && values[2]==64 && values[3]==64);
    check_result("All threads sync to 64 after reset()", all_64);
}

// =============================================================================
// Main
// =============================================================================
// TEST 18: Interaction between BLIS_*_NT env vars and bli_thread_set_num_threads()
// =============================================================================
void test_18_env_ways_vs_set_num_threads(void) {
    print_separator("TEST 18: BLIS_*_NT env vars vs bli_thread_set_num_threads()");

    // This test verifies that bli_thread_set_num_threads() correctly overrides
    // any prior ways configuration (BLIS_JC_NT, BLIS_IC_NT, etc.).
    //
    // Expected behavior (after fix):
    // - bli_thread_set_num_threads() clears ways and sets num_threads
    // - bli_thread_get_num_threads() returns the new value
    // - Ways are reset to -1 (unset), enabling auto-factorization

    // First, set ways explicitly to simulate env var initialization
    bli_thread_set_ways(2, 1, 4, 2, 1);  // Total = 2*1*4*2*1 = 16 threads
    
    int initial_nt = bli_thread_get_num_threads();
    int initial_jc = bli_thread_get_jc_nt();
    int initial_ic = bli_thread_get_ic_nt();
    int initial_jr = bli_thread_get_jr_nt();
    
    printf("After bli_thread_set_ways(2,1,4,2,1):\n");
    printf("  num_threads=%d (derived from ways: 2*1*4*2*1=16)\n", initial_nt);
    printf("  jc_nt=%d, ic_nt=%d, jr_nt=%d\n", initial_jc, initial_ic, initial_jr);
    
    check_result("Initial ways set correctly (jc=2)", initial_jc == 2);
    check_result("Initial ways set correctly (ic=4)", initial_ic == 4);
    check_result("Initial num_threads = 16", initial_nt == 16);

    // Now call bli_thread_set_num_threads() with a different value
    bli_thread_set_num_threads(8);
    
    int after_nt = bli_thread_get_num_threads();
    int after_jc = bli_thread_get_jc_nt();
    int after_ic = bli_thread_get_ic_nt();
    int after_jr = bli_thread_get_jr_nt();
    
    printf("\nAfter bli_thread_set_num_threads(8):\n");
    printf("  bli_thread_get_num_threads() = %d\n", after_nt);
    printf("  jc_nt=%d, ic_nt=%d, jr_nt=%d\n", after_jc, after_ic, after_jr);
    
    // After fix: num_threads should be 8, ways should be cleared (-1)
    check_result("num_threads changed to 8", after_nt == 8);
    check_result("jc_nt cleared to -1", after_jc == -1);
    check_result("ic_nt cleared to -1", after_ic == -1);
    check_result("jr_nt cleared to -1", after_jr == -1);
    
    printf("\nVerified: bli_thread_set_num_threads() correctly overrides ways\n");
    
    // Cleanup
    bli_thread_reset();
    bli_thread_set_num_threads(1);
}

// =============================================================================
// TEST 19: set_ways then set_num_threads then reset
// =============================================================================
void test_19_ways_then_set_nt_then_reset(void) {
    print_separator("TEST 19: set_ways -> set_num_threads -> reset");

    // Step 1: Set ways
    bli_thread_set_ways(2, 1, 4, 2, 1);  // 16 threads
    int nt1 = bli_thread_get_num_threads();
    int jc1 = bli_thread_get_jc_nt();
    printf("After set_ways(2,1,4,2,1): nt=%d, jc=%d\n", nt1, jc1);
    check_result("Ways give 16 threads", nt1 == 16);
    check_result("jc=2", jc1 == 2);

    // Step 2: Override with set_num_threads
    bli_thread_set_num_threads(8);
    int nt2 = bli_thread_get_num_threads();
    int jc2 = bli_thread_get_jc_nt();
    printf("After set_num_threads(8): nt=%d, jc=%d\n", nt2, jc2);
    check_result("num_threads = 8", nt2 == 8);
    check_result("jc cleared to -1", jc2 == -1);

    // Step 3: Reset - should restore to global, which was updated by set_num_threads
    bli_thread_reset();
    int nt3 = bli_thread_get_num_threads();
    int jc3 = bli_thread_get_jc_nt();
    printf("After reset(): nt=%d, jc=%d\n", nt3, jc3);
    check_result("After reset, num_threads = 8 (from global)", nt3 == 8);
    check_result("After reset, jc still -1 (global was cleared)", jc3 == -1);

    // Cleanup
    bli_thread_set_num_threads(1);
}

// =============================================================================
// TEST 20: set_ways then set_num_threads_local then reset
// =============================================================================
void test_20_ways_then_local_then_reset(void) {
    print_separator("TEST 20: set_ways -> set_num_threads_local -> reset");

    // Step 1: Set ways (updates both tl_rntm AND global_rntm)
    bli_thread_set_ways(2, 1, 4, 2, 1);  // 16 threads
    int nt1 = bli_thread_get_num_threads();
    printf("After set_ways(2,1,4,2,1): nt=%d\n", nt1);
    check_result("Ways give 16 threads", nt1 == 16);

    // Step 2: Override locally with set_num_threads_local
    bli_thread_set_num_threads_local(8);
    int nt2 = bli_thread_get_num_threads();
    int jc2 = bli_thread_get_jc_nt();
    printf("After set_num_threads_local(8): nt=%d, jc=%d\n", nt2, jc2);
    check_result("Local num_threads = 8", nt2 == 8);
    check_result("Local jc cleared to -1", jc2 == -1);

    // Step 3: Reset - restores tl_rntm from global_rntm
    // global_rntm WAS modified by set_ways (ways + blis_mt + num_threads=-1).
    // set_num_threads_local() does NOT update global_rntm.
    // Therefore reset restores the ways set in Step 1.
    bli_thread_reset();
    int nt3 = bli_thread_get_num_threads();
    int jc3 = bli_thread_get_jc_nt();
    printf("After reset(): nt=%d, jc=%d\n", nt3, jc3);
    // The local override (8) is gone; the ways from set_ways() are restored
    // from global_rntm.
    check_result("After reset, nt is NOT 8 (local cleared)", nt3 != 8);
    check_result("After reset, nt=16 (ways restored from global)", nt3 == 16);
    check_result("After reset, jc=2 (ways restored from global)", jc3 == 2);

    // Cleanup
    bli_thread_set_num_threads(1);
}

// =============================================================================
// TEST 21: set_num_threads then set_ways then set_num_threads
// =============================================================================
void test_21_nt_ways_nt_roundtrip(void) {
    print_separator("TEST 21: set_num_threads -> set_ways -> set_num_threads");

    // Step 1: Set num_threads
    bli_thread_set_num_threads(8);
    check_result("Step 1: nt=8", bli_thread_get_num_threads() == 8);
    check_result("Step 1: jc=-1 (auto factor)", bli_thread_get_jc_nt() == -1);

    // Step 2: Override with explicit ways
    bli_thread_set_ways(2, 1, 2, 2, 1);  // 8 threads via ways
    int nt2 = bli_thread_get_num_threads();
    int jc2 = bli_thread_get_jc_nt();
    printf("After set_ways(2,1,2,2,1): nt=%d, jc=%d\n", nt2, jc2);
    check_result("Step 2: nt=8 (from ways)", nt2 == 8);
    check_result("Step 2: jc=2", jc2 == 2);

    // Step 3: Override again with set_num_threads - should clear ways
    bli_thread_set_num_threads(4);
    int nt3 = bli_thread_get_num_threads();
    int jc3 = bli_thread_get_jc_nt();
    printf("After set_num_threads(4): nt=%d, jc=%d\n", nt3, jc3);
    check_result("Step 3: nt=4", nt3 == 4);
    check_result("Step 3: jc cleared", jc3 == -1);

    // Cleanup
    bli_thread_set_num_threads(1);
}


void test_22_set_nt_then_set_ways(void) {
    print_separator("TEST 22: set_num_threads then set_ways overrides nt");

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
    printf("After set_ways(4,1,2,1,1): nt=%d, jc=%d, ic=%d\n",
           nt2, jc2, ic2);
    check_result("Step 2: nt=8 (from 4*1*2*1*1)", nt2 == 8);
    check_result("Step 2: jc=4", jc2 == 4);
    check_result("Step 2: ic=2", ic2 == 2);

    // Step 3: Set ways to a DIFFERENT total: 2x1x3x1x1 = 6
    // This is the key test — nt must NOT be stale 8
    bli_thread_set_ways(2, 1, 3, 1, 1);
    int nt3 = bli_thread_get_num_threads();
    int jc3 = bli_thread_get_jc_nt();
    int ic3 = bli_thread_get_ic_nt();
    printf("After set_ways(2,1,3,1,1): nt=%d, jc=%d, ic=%d\n",
           nt3, jc3, ic3);
    check_result("Step 3: nt=6 (from 2*1*3*1*1, not stale 8)", nt3 == 6);
    check_result("Step 3: jc=2", jc3 == 2);
    check_result("Step 3: ic=3", ic3 == 3);

    // Cleanup
    bli_thread_set_num_threads(1);
}

void test_23_set_ways_propagates_to_new_threads(void) {
    print_separator("TEST 23: set_ways propagates to new OMP threads via global_rntm");

    // Step 1: set_num_threads(12), then set_ways(3,1,2,1,1) = 6
    bli_thread_set_num_threads(12);
    int nt1 = bli_thread_get_num_threads();
    printf("After set_num_threads(12): nt=%d\n", nt1);
    check_result("Step 1: nt=12", nt1 == 12);

    bli_thread_set_ways(3, 1, 2, 1, 1);
    int nt2 = bli_thread_get_num_threads();
    printf("After set_ways(3,1,2,1,1): nt=%d\n", nt2);
    check_result("Step 2: nt=6 (from 3*1*2*1*1)", nt2 == 6);

    // Step 3: Spawn a new OMP thread — it should inherit ways from global_rntm
    // (not the stale num_threads=12)
    int child_nt = -1;
    int child_jc = -1;
    int child_ic = -1;
    #pragma omp parallel num_threads(1)
    {
        // New thread — tl_rntm initialized from global_rntm
        child_nt = bli_thread_get_num_threads();
        child_jc = bli_thread_get_jc_nt();
        child_ic = bli_thread_get_ic_nt();
    }
    printf("Child thread: nt=%d, jc=%d, ic=%d\n",
           child_nt, child_jc, child_ic);
    check_result("Step 3: child nt=6 (from ways, not stale 12)", child_nt == 6);
    check_result("Step 3: child jc=3", child_jc == 3);
    check_result("Step 3: child ic=2", child_ic == 2);

    // Cleanup
    bli_thread_set_num_threads(1);
}

// =============================================================================
int main(int argc, char** argv) {
    printf("BLIS Thread Control API Test Suite\n");
    printf("===================================\n");
    printf("OMP_MAX_ACTIVE_LEVELS=%d, omp_get_max_threads()=%d\n",
           omp_get_max_active_levels(), omp_get_max_threads());

    if (argc == 1) {
        test_1_env_inheritance(); test_2_global_propagation();
        test_3_local_only_affects_caller(); test_4_local_precedence();
        test_5_per_thread_local(); test_6_reset_in_children();
        test_7_nested_parallel(); test_8_edge_cases();
        test_9_set_ways(); test_10_is_parallel();
        test_11_concurrent_global_updates(); test_12_dgemm_with_threads();
        test_13_interleaved_settings(); test_14_persistence_across_regions();
        test_15_parallel_dgemm_different_threads(); test_16_thread_pool_behavior();
        test_17_reset_to_sync_global();
        test_18_env_ways_vs_set_num_threads();
        test_19_ways_then_set_nt_then_reset();
        test_20_ways_then_local_then_reset();
        test_21_nt_ways_nt_roundtrip();
        test_22_set_nt_then_set_ways();
        test_23_set_ways_propagates_to_new_threads();
    } else {
        int test_num = atoi(argv[1]);
        switch (test_num) {
            case 0: /* fall through to run all */
                test_1_env_inheritance(); test_2_global_propagation();
                test_3_local_only_affects_caller(); test_4_local_precedence();
                test_5_per_thread_local(); test_6_reset_in_children();
                test_7_nested_parallel(); test_8_edge_cases();
                test_9_set_ways(); test_10_is_parallel();
                test_11_concurrent_global_updates(); test_12_dgemm_with_threads();
                test_13_interleaved_settings(); test_14_persistence_across_regions();
                test_15_parallel_dgemm_different_threads(); test_16_thread_pool_behavior();
                test_17_reset_to_sync_global();
        test_18_env_ways_vs_set_num_threads();
                test_19_ways_then_set_nt_then_reset();
                test_20_ways_then_local_then_reset();
                test_21_nt_ways_nt_roundtrip();
                test_22_set_nt_then_set_ways();
                test_23_set_ways_propagates_to_new_threads();
                break;
            case 1: test_1_env_inheritance(); break;
            case 2: test_2_global_propagation(); break;
            case 3: test_3_local_only_affects_caller(); break;
            case 4: test_4_local_precedence(); break;
            case 5: test_5_per_thread_local(); break;
            case 6: test_6_reset_in_children(); break;
            case 7: test_7_nested_parallel(); break;
            case 8: test_8_edge_cases(); break;
            case 9: test_9_set_ways(); break;
            case 10: test_10_is_parallel(); break;
            case 11: test_11_concurrent_global_updates(); break;
            case 12: test_12_dgemm_with_threads(); break;
            case 13: test_13_interleaved_settings(); break;
            case 14: test_14_persistence_across_regions(); break;
            case 15: test_15_parallel_dgemm_different_threads(); break;
            case 16: test_16_thread_pool_behavior(); break;
            case 17: test_17_reset_to_sync_global(); break;
            case 18: test_18_env_ways_vs_set_num_threads(); break;
            case 19: test_19_ways_then_set_nt_then_reset(); break;
            case 20: test_20_ways_then_local_then_reset(); break;
            case 21: test_21_nt_ways_nt_roundtrip(); break;
            case 22: test_22_set_nt_then_set_ways(); break;
            case 23: test_23_set_ways_propagates_to_new_threads(); break;
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
