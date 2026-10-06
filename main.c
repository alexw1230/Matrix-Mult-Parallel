/*
 * Usage: mpirun -np P ./mm -n N [-r reps] [-s seed] [-e] [-p]
 *                          [-o results.csv] [-w raw_times.csv]
 */
#include "kernel.h"
#include "mm_ring.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { T_TOTAL, T_SCATTER, T_COMP, T_COMM, T_GATHER, NT };

typedef struct {
    int N, reps, exact, print;
    long seed;
    const char *csv; /* summary file (one row per run)          */
    const char *raw; /* per-rep timings; NULL = don't write one */
} opts_t;

static void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: mpirun -np P %s -n N [options]\n"
            "  -n N     matrix size (required; P must divide N)\n"
            "  -r R     timed repetitions (default depends on N)\n"
            "  -s S     random seed (default 530)\n"
            "  -e       exact mode: integer-valued entries, must match serial exactly\n"
            "  -p       print A, B, C (only if N <= 16)\n"
            "  -o FILE  summary CSV, appended (default results.csv)\n"
            "  -w FILE  per-rep timing CSV, appended (default: not written)\n",
            prog);
}

/* Plain argv loop rather than getopt so every rank parses identically. */
static int parse_args(int argc, char **argv, opts_t *o)
{
    o->N = 0; o->reps = 0; o->exact = 0; o->print = 0;
    o->seed = 530; o->csv = "results.csv"; o->raw = NULL;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        int has_val = (i + 1 < argc);
        if      (!strcmp(a, "-n") && has_val) o->N    = atoi(argv[++i]);
        else if (!strcmp(a, "-r") && has_val) o->reps = atoi(argv[++i]);
        else if (!strcmp(a, "-s") && has_val) o->seed = atol(argv[++i]);
        else if (!strcmp(a, "-o") && has_val) o->csv  = argv[++i];
        else if (!strcmp(a, "-w") && has_val) o->raw  = argv[++i];
        else if (!strcmp(a, "-e")) o->exact = 1;
        else if (!strcmp(a, "-p")) o->print = 1;
        else return -1;
    }
    if (o->N <= 0 || o->reps < 0) return -1;
    return 0;
}

/* Small N runs in microseconds and needs many reps; large N needs few. */
static int default_reps(int N)
{
    if (N <= 64)   return 100;
    if (N <= 256)  return 30;
    if (N <= 1024) return 5;
    return 3;
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* Median of v[0], v[stride], ..., v[(n-1)*stride]. */
static double median(const double *v, int n, int stride)
{
    double *tmp = malloc((size_t)n * sizeof *tmp);
    for (int i = 0; i < n; i++) tmp[i] = v[(size_t)i * stride];
    qsort(tmp, (size_t)n, sizeof *tmp, cmp_double);
    double m = (n % 2) ? tmp[n / 2] : 0.5 * (tmp[n / 2 - 1] + tmp[n / 2]);
    free(tmp);
    return m;
}

static void print_matrix(const char *name, const double *M, int N)
{
    printf("%s =\n", name);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) printf(" %8.3f", M[(size_t)i * N + j]);
        printf("\n");
    }
}

/* Open for append; write the header first if the file is new or empty. */
static FILE *open_csv(const char *path, const char *header)
{
    FILE *f = fopen(path, "a");
    if (!f) { perror(path); return NULL; }
    fseek(f, 0, SEEK_END);
    if (ftell(f) == 0) fprintf(f, "%s\n", header);
    return f;
}

static double *must_alloc(size_t count, MPI_Comm comm)
{
    double *p = alloc_doubles(count);
    if (!p) {
        fprintf(stderr, "allocation of %zu doubles failed\n", count);
        MPI_Abort(comm, 2);
    }
    return p;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    /* ---- 1-D periodic process topology: the ring ---- */
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int dims[1] = { world_size }, periods[1] = { 1 };
    MPI_Comm ring;
    MPI_Cart_create(MPI_COMM_WORLD, 1, dims, periods, 1, &ring);

    int rank, P;
    MPI_Comm_rank(ring, &rank);
    MPI_Comm_size(ring, &P);

    /* ---- Arguments (every rank parses the same argv, so all agree) ---- */
    opts_t o;
    if (parse_args(argc, argv, &o) != 0) {
        if (rank == 0) usage(argv[0]);
        MPI_Comm_free(&ring);
        MPI_Finalize();
        return 1;
    }
    if (o.N % P != 0) {
        if (rank == 0) fprintf(stderr, "P = %d must divide N = %d\n", P, o.N);
        MPI_Comm_free(&ring);
        MPI_Finalize();
        return 1;
    }
    if (o.reps == 0) o.reps = default_reps(o.N);

    const int N = o.N;
    const int nb = N / P;
    const size_t NN = (size_t)N * N;
    const size_t strip = (size_t)nb * N;
    if (strip > INT_MAX) {
        if (rank == 0) fprintf(stderr, "strip of %zu doubles exceeds MPI int count\n", strip);
        MPI_Abort(ring, 1);
    }
    const int count = (int)strip;

    char host[MPI_MAX_PROCESSOR_NAME];
    int host_len;
    MPI_Get_processor_name(host, &host_len);

    /* ---- Root generates the data (independent of P, so runs are comparable) ---- */
    double *A = NULL, *BT = NULL, *CT = NULL, *C = NULL, *C_ref = NULL;
    if (rank == 0) {
        A     = must_alloc(NN, ring);
        BT    = must_alloc(NN, ring);
        CT    = must_alloc(NN, ring);
        C     = must_alloc(NN, ring);
        C_ref = must_alloc(NN, ring);
        double *B = must_alloc(NN, ring);

        rng_seed(o.seed);
        fill_random(A, NN, o.exact);
        fill_random(B, NN, o.exact);
        transpose(B, BT, N); /* column strips of B = row strips of B^T */
        if (o.print && N <= 16) { print_matrix("A", A, N); print_matrix("B", B, N); }
        free(B);
    }

    /* ---- Local strips on every rank ---- */
    double *A_cur  = must_alloc(strip, ring); /* rolls around the ring */
    double *BT_loc = must_alloc(strip, ring); /* stays put              */
    double *CT_loc = must_alloc(strip, ring); /* my column strip of C   */

    /* Root keeps every timed rep: times[rep*NT + phase], max over ranks. */
    double *times = (rank == 0) ? malloc((size_t)o.reps * NT * sizeof *times) : NULL;

    /* ---- Timed region; rep = -1 is an untimed warm-up ---- */
    for (int rep = -1; rep < o.reps; rep++) {
        double loc[NT], mx[NT];
        ring_times_t rt;

        MPI_Barrier(ring);
        double t0 = MPI_Wtime();

        MPI_Scatter(A,  count, MPI_DOUBLE, A_cur,  count, MPI_DOUBLE, 0, ring);
        MPI_Scatter(BT, count, MPI_DOUBLE, BT_loc, count, MPI_DOUBLE, 0, ring);
        double t1 = MPI_Wtime();

        mm_ring(A_cur, BT_loc, CT_loc, N, ring, &rt);
        double t2 = MPI_Wtime();

        MPI_Gather(CT_loc, count, MPI_DOUBLE, CT, count, MPI_DOUBLE, 0, ring);
        double t3 = MPI_Wtime();

        loc[T_TOTAL]   = t3 - t0;
        loc[T_SCATTER] = t1 - t0;
        loc[T_COMP]    = rt.comp;
        loc[T_COMM]    = rt.comm;
        loc[T_GATHER]  = t3 - t2;
        MPI_Reduce(loc, mx, NT, MPI_DOUBLE, MPI_MAX, 0, ring);

        if (rank == 0 && rep >= 0)
            memcpy(times + (size_t)rep * NT, mx, sizeof mx);
    }

    /* ---- Root: assemble C, run serial baseline, verify, report ---- */
    if (rank == 0) {
        transpose(CT, C, N);

        int serial_reps = (N >= 4096) ? 1 : (N >= 1024) ? 3 : o.reps;
        double *ts = malloc((size_t)serial_reps * sizeof *ts);
        if (N < 1024) serial_matmul_bt(A, BT, C_ref, N); /* warm-up */
        for (int r = 0; r < serial_reps; r++) {
            double s0 = MPI_Wtime();
            serial_matmul_bt(A, BT, C_ref, N);
            ts[r] = MPI_Wtime() - s0;
        }
        double T_serial = median(ts, serial_reps, 1);
        free(ts);

        double max_err = 0.0;
        for (size_t i = 0; i < NN; i++) {
            double e = fabs(C[i] - C_ref[i]);
            if (e > max_err) max_err = e;
        }
        /* Same kernel and summation order -> expect 0. The tolerance only
         * guards against compiler differences in the normal (float) mode. */
        double tol = o.exact ? 0.0 : 16.0 * N * DBL_EPSILON;
        int pass = (max_err <= tol);

        double med[NT];
        for (int k = 0; k < NT; k++) med[k] = median(times + k, o.reps, NT);
        double speedup = T_serial / med[T_TOTAL];
        double eff = speedup / P;

        if (o.print && N <= 16) print_matrix("C", C, N);

        printf("N=%d P=%d reps=%d mode=%s host=%s\n"
               "  T_total=%.6e  (scatter %.3e, comp %.3e, comm %.3e, gather %.3e)\n"
               "  T_serial=%.6e  speedup=%.3f  efficiency=%.3f  max_err=%.3e  %s\n",
               N, P, o.reps, o.exact ? "exact" : "float", host,
               med[T_TOTAL], med[T_SCATTER], med[T_COMP], med[T_COMM], med[T_GATHER],
               T_serial, speedup, eff, max_err, pass ? "PASS" : "FAIL");
        fflush(stdout);

        FILE *f = open_csv(o.csv,
            "N,P,reps,mode,T_total,T_scatter,T_comp,T_comm,T_gather,"
            "T_serial,speedup,efficiency,max_err,pass,host");
        if (f) {
            fprintf(f, "%d,%d,%d,%s,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.6f,%.6f,%.3e,%d,%s\n",
                    N, P, o.reps, o.exact ? "exact" : "float",
                    med[T_TOTAL], med[T_SCATTER], med[T_COMP], med[T_COMM], med[T_GATHER],
                    T_serial, speedup, eff, max_err, pass, host);
            fclose(f);
        }

        if (o.raw) {
            FILE *g = open_csv(o.raw, "N,P,rep,T_total,T_scatter,T_comp,T_comm,T_gather");
            if (g) {
                for (int r = 0; r < o.reps; r++) {
                    const double *t = times + (size_t)r * NT;
                    fprintf(g, "%d,%d,%d,%.9e,%.9e,%.9e,%.9e,%.9e\n", N, P, r,
                            t[T_TOTAL], t[T_SCATTER], t[T_COMP], t[T_COMM], t[T_GATHER]);
                }
                fclose(g);
            }
        }

        free(times);
        free(A); free(BT); free(CT); free(C); free(C_ref);
    }

    free(A_cur); free(BT_loc); free(CT_loc);
    MPI_Comm_free(&ring);
    MPI_Finalize();
    return 0;
}