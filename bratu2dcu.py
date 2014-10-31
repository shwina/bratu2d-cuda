from petsc4py import PETSc
from pycuda import autoinit
import pycuda.driver as cuda
import numpy as np

module = cuda.module_from_file('bratu2dcu.cubin')
kernel = module.get_function('bratu2d16x16')
kernel.prepare([np.intp, np.intp, np.int32, np.int32, np.float64])

def bratu2d(x, f, nx, ny, alpha):
    kernel.prepared_call((nx/16, ny/16, 1), (16, 16, 1), x, f, nx, ny, alpha)
