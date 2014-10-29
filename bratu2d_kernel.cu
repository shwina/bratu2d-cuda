#include <cuda.h>
#include <math.h>
 
extern "C" {
    __global__ void temperature_update16x16(double* a_d, double* b_d,
                                            const int N_x, const int N_y,
                                            double alpha)
    {
        #define BDIMX 16
        #define BDIMY 16

        __shared__ double slice[BDIMX+2][BDIMY+2];

        double dx = (double)1.0/(N_x-1);
        double dy = (double)1.0/(N_y-1);

        int ix = blockIdx.x*blockDim.x + threadIdx.x;
        int iy = blockIdx.y*blockDim.y + threadIdx.y;
        int i2d = iy*N_x + ix;

        int tx = threadIdx.x + 1;
        int ty = threadIdx.y + 1;

        bool compute_if = ix > 0 && ix < (N_x-1) && iy > 0 && iy < (N_y-1);

        if (compute_if){
            if(threadIdx.x == 0){ // Halo left
                slice[ty][tx-1]     =   a_d[i2d - 1];
            }

            if(threadIdx.x == BDIMX-1){ // Halo right
                slice[ty][tx+1]     =   a_d[i2d + 1];
            }

            if(threadIdx.y == 0){ // Halo bottom
                slice[ty-1][tx]     =   a_d[i2d - N_x];
            }

            if(threadIdx.y == BDIMY-1){ // Halo top
                slice[ty+1][tx]     =   a_d[i2d + N_x];
            }
        }

        __syncthreads();

        slice[ty][tx] = a_d[i2d];

        __syncthreads();

        if (compute_if){        
            b_d[i2d] = (2*slice[ty][tx] - slice[ty][tx+1] - slice[ty][tx-1]) * (dy/dx) \
                      +(2*slice[ty][tx] - slice[ty+1][tx] - slice[ty-1][tx]) * (dx/dy) \
                      -alpha*exp(slice[ty][tx]);
        }

        __syncthreads();
    }
}
