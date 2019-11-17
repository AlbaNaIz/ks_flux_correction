from fenics import *
from time import time

mesh = UnitSquareMesh(500, 500)
V = FunctionSpace(mesh, 'CG', 1)
u = TrialFunction(V)
v = TestFunction(V)

# M = PETScMatrix()
# assemble(inner(u, v)*dx, tensor=M)

#print("Current linear algebra backend:",
#        parameters.linear_algebra_backend) 

print("\n1. Building of matrices ············································\n")

t0=time()
M = assemble(inner(u, v)*dx )
t1=time()
print("M0: (assembled matrix)", type(M), end="\n")
print(" ... time:", t1-t0, end="\n\n")

t0=time()
PETSC_Mat = as_backend_type(M)
t1=time()
print("M1: as_backend_type", type(PETSC_Mat), end="\n")
print(" ... time:", t1-t0, end="\n\n")

t0=time()
Mmat = PETSC_Mat.mat()
t1=time()
print("M2: as_backend_type().mat()", type(Mmat), end="\n")
print(" ... time:", t1-t0, end="\n\n")

t0=time()
Mcopy = Mmat.duplicate()
t1=time()
print("M3: as_backend_type().mat()", type(Mmat), end="\n")
print(" ... time:", t1-t0, end="\n")
# Mcopy.zeroEntries()

print("\n2. Access to matrix rows ············································\n")

t0=time()
print(M.getrow(3))
print(" M0... time:", time()-t0, end="\n\n")

t0=time()
print(Mmat.getRow(3))
print(" M2... time:", time()-t0, end="\n\n")

t0=time()
print(Mcopy.getRow(3))
print(" M3... time:", time()-t0, end="\n\n")
