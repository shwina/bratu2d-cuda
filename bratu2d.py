import sys, petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
from pycuda import autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
import time
# this user class is an application
# context for the nonlinear problem
# at hand; it contains some parametes
# and knows how to compute residuals

class Bratu2D:

    def __init__(self, nx, ny, alpha, impl='python'):
        self.nx = nx # x grid size
        self.ny = ny # y grid size
        self.alpha = alpha
        if impl == 'python':
            from bratu2dnpy import bratu2d
            order = 'c'
        elif impl == 'fortran':
            from bratu2df90 import bratu2d
            order = 'f'
        elif impl == 'cuda':
            from bratu2dcu import bratu2d
            order = None
        else:
            raise ValueError('invalid implementation')
        self.compute = bratu2d
        self.order = order

    def evalFunction(self, snes, X, F):
        nx, ny = self.nx, self.ny
        alpha = self.alpha
        order = self.order
        if impl == 'cuda':
            x = X.getCUDAHandle()
            f = F.getCUDAHandle()
            self.compute(x, f, nx, ny, alpha)
            X.restoreCUDAHandle(0)
            F.restoreCUDAHandle(0)
        else:
            x = X[...].reshape(nx, ny, order=order)
            f = F[...].reshape(nx, ny, order=order)
            self.compute(alpha, x, f)

# convenience access to
# PETSc options database
OptDB = PETSc.Options()

nx = OptDB.getInt('nx', 128)
ny = OptDB.getInt('ny', nx)
alpha = OptDB.getReal('alpha', 6.8)
impl  = OptDB.getString('impl', 'python')

if impl in ['python', 'fortran']:
    vectype = 'seq'
else:
    vectype = 'cusp'

# create application context
# and PETSc nonlinear solver
appc = Bratu2D(nx, ny, alpha, impl)
snes = PETSc.SNES().create()

# register the function in charge of
# computing the nonlinear residual
f = PETSc.Vec()
f.create()
f.setType(vectype)
f.setSizes(nx*ny)
snes.setFunction(appc.evalFunction, f)

# configure the nonlinear solver
# to use a matrix-free Jacobian
snes.setUseMF(True)
snes.getKSP().setType('cg')
snes.setFromOptions()

# solve the nonlinear problem
b, x = None, f.duplicate()
x.set(0.0) # zero inital guess
t1 = time.clock()
snes.solve(b, x)
t2 = time.clock()
print 'Time for solve: ', t2-t1

if OptDB.getBool('plot_mpl', True):
    import matplotlib.pyplot as plt
    from numpy import mgrid
    X, Y =  mgrid[0:1:1j*nx,0:1:1j*ny]
    Z = x[...].reshape(nx,ny)
    plt.figure()
    plt.contourf(X,Y,Z)
    plt.colorbar()
    plt.plot(X.ravel(),Y.ravel(),'.k')
    plt.axis('equal')
    plt.show()
    plt.savefig('bratu2dsolution.png')
