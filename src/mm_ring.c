#include "mm_ring.h"
#include "kernel.h"

void mm_ring(double *A_cur, const double *BT_loc, double *CT_loc, int N,
             MPI_Comm ring, ring_times_t *t)
{
    int rank, P;
    MPI_Comm_rank(ring, &rank);
    MPI_Comm_size(ring, &P);

    const int nb = N / P;
    const int count = nb * N;
    int from_right, to_left;
    MPI_Cart_shift(ring, 0, -1, &from_right, &to_left);

    t->comp = 0.0;
    t->comm = 0.0;

    for (int s = 0; s < P; s++) {
        const int src = (rank + s) % P;

        double t0 = MPI_Wtime();
        for (int j = 0; j < nb; j++) {
            const double *b = BT_loc + (size_t)j * N;
            double *ct_row  = CT_loc + (size_t)j * N + (size_t)src * nb;
            for (int i = 0; i < nb; i++){
                ct_row[i] = dot(A_cur + (size_t)i * N, b, N);
            }
        }
        double t1 = MPI_Wtime();
        t->comp += t1 - t0;

        if (s < P - 1) {
            MPI_Sendrecv_replace(A_cur, count, MPI_DOUBLE,
                                 to_left, 0, from_right, 0,
                                 ring, MPI_STATUS_IGNORE);
            t->comm += MPI_Wtime() - t1;
        }
    }
}