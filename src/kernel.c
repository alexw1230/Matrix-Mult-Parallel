#define _XOPEN_SOURCE 700
#include "kernel.h"
#include <stdlib.h>

double dot(const double *a, const double *b, int n){
    double s = 0.0;
    for (int k = 0; k < n; k++){
        s += a[k] * b[k];
    }
    return s;
}

void transpose(const double *src, double *dst, int n){
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            dst[(size_t)j * n + i] = src[(size_t)i * n + j];
        }
    }
}

void serial_matmul_bt(const double *A, const double *BT, double *C, int n){
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            C[(size_t)i * n + j] = dot(A + (size_t)i * n, BT + (size_t)j * n, n);
        }
    }
}

void rng_seed(long seed){srand48(seed);}

void fill_random(double *M, size_t count, int exact){
    for (size_t i = 0; i < count; i++) {
        if (exact){
            M[i] = (double)((long)(drand48() * 21.0) - 10);
        } else {
            M[i] = 2.0 * drand48() - 1.0;      
        }
    }
}

double *alloc_doubles(size_t count){
    size_t bytes = count * sizeof(double);
    bytes = (bytes + 63) & ~(size_t)63;
    if (bytes == 0){
        bytes = 64;
    }
    return aligned_alloc(64, bytes);
}