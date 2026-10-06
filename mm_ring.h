#ifndef MM_RING_H
#define MM_RING_H

#include <mpi.h>

typedef struct {
    double comp;
    double comm;
} ring_times_t;

void mm_ring(double *A_cur, const double *BT_loc, double *CT_loc, int N, MPI_Comm ring, ring_times_t *t);

#endif