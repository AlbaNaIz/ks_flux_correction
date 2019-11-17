from fenics import *
import numpy as np
import matplotlib.pyplot as plt

from keller_segel.abstract_scheme import KS_AbstractScheme


#==============================================================================
# class KS_Matrix_DefaultScheme:
#   Explicit building of linear system (matrix and rhs) for usual implementat.
#------------------------------------------------------------------------------

def save_coo_matrix(A, filename):
    "Save matrix A in a file, using COO format (i, j, value_ij)"
    from scipy.sparse import csr_matrix, coo_matrix
    Mmat = as_backend_type(A).mat()
    coo_mat = coo_matrix(csr_matrix(Mmat.getValuesCSR()[::-1]))
    np.savetxt(filename,
            np.c_[coo_mat.row, coo_mat.col, coo_mat.data],
            fmt=['%d', '%d', '%.16f'])

class KS_Matrix_DefaultScheme(KS_AbstractScheme):
    """
    Deafult Keller-Segel space/time scheme. Underlying FE matrices are
    explicitly build
    """

    def __init__( self, mesh, fe_order, dt, t_init=0.,
                  k0=1, k1=1, k2=1, k3=1, k4=1 ):
        super().__init__(mesh, fe_order, dt, t_init, k0, k1, k2, k3, k4)


    def build_fe_scheme(self):
        """
        Define variational equations and FE systems which define current scheme
        """
        u, ub = TrialFunction(self.Vh), TestFunction(self.Vh)
        v, vb = TrialFunction(self.Vh), TestFunction(self.Vh)
        #
        # Define variational formulation for u and v
        #
        dt, k2, k3 = self.dt, self.k2, self.k3

        # Mass matrix
        self.M = assemble( u*ub*dx )

        # Diffusion matrix
        self.L = assemble( dot(grad(u), grad(ub))*dx )

        # Matrix for the v-equation:
        self.Av = (1 + k3*dt)*self.M + k2*dt*self.L


    def solve(self):
        """Compute u and v"""

        dt, k0, k1, k4 = self.dt, self.k0, self.k1, self.k4

        #
        # 1. Compute v and gradient of v
        #
        b = self.M * (self.v0.vector() + k4*dt*self.u0.vector())
        solve ( self.Av, self.v.vector(), b )  # Solve A*v = b
        grad_v = project( grad(self.v), self.Wh )

        #
        # 2. Compute u
        #

        # 2.1 Assemble chemotaxis transport matrix
        u, ub = TrialFunction(self.Vh), TestFunction(self.Vh)
        v, vb = TrialFunction(self.Vh), TestFunction(self.Vh)
        K = assemble( u*dot(grad_v, grad(ub)) * dx )

        # 2.2 Define system and solve it
        A = self.M + k0*dt*self.L - k1*dt*K
        b = self.M * self.u0.vector()
        solve (A, self.u.vector(), b)  # Solve A*u = b
