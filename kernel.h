/* kernel.h -- serial building blocks shared by the parallel and serial code.
 *
 * All matrices are N x N doubles stored row-major in one contiguous block:
 * element (i, j) lives at M[i*N + j].
 */
#ifndef KERNEL_H
#define KERNEL_H

#include <stddef.h>

//Dot prod of 2 vect of len n
double dot(const double *a, const double *b, int n);

//transpose of matrix
void transpose(const double *src, double *dst, int n);

//Serial Baseline
void serial_matmul_bt(const double *A, const double *BT, double *C, int n);

//Rng seeding
void rng_seed(long seed);

//Rand fill
void fill_random(double *M, size_t count, int exact);

//Memory allocation
double *alloc_doubles(size_t count);

#endif