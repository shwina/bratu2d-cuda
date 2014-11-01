from petsc4py import PETSc
from pycuda import autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np

module = SourceModule("""
        extern "C" {
            __global__ void bratu2d16x16(double* a_d, double* b_d,
                                         int N_x, int N_y,
                                         double alpha)
            {
                double dx = (double)1/((double)N_x-1);
                double dy = (double)1/((double)N_y-1);

                int ix = blockIdx.x*blockDim.x + threadIdx.x;
                int iy = blockIdx.y*blockDim.y + threadIdx.y;
                int i2d = iy*N_x + ix;

                bool compute_if = ix > 0 && ix < (N_x-1) && iy > 0 && iy < (N_y-1);

                b_d[i2d] = a_d[i2d];
                __syncthreads();

                if (compute_if){
                    b_d[i2d] = (2*a_d[i2d] - a_d[i2d+1] - a_d[i2d-1]) * (dy/dx) \
                              +(2*a_d[i2d] - a_d[i2d+N_x] - a_d[i2d-N_x]) * (dx/dy) \
                              -alpha*exp(a_d[i2d])*(dx*dy);
                }
                __syncthreads();
            }
        }
     """)
kernel = module.get_function('bratu2d16x16')
kernel.prepare([np.intp, np.intp, np.int32, np.int32, np.float64])

def bratu2d(x, f, nx, ny, alpha):
    kernel.prepared_call((nx/16, ny/16, 1), (16, 16, 1), x, f, nx, ny, alpha)
