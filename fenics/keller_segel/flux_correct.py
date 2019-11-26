from fenics import *
import numpy as np
import matplotlib.pyplot as plt

from keller_segel.matrix_default_scheme import (
    KS_Matrix_DefaultScheme, save_coo_matrix )

# Import numba high-performance functions
from keller_segel.numba_optimized import (
    index, compute_D_values, update_F_values )

#==============================================================================
# class KS_FluxCorrect_DefaultScheme:
#   - Keller-Segel with flux-correction (FC)
#   - Used Numba compiler for high performance (in FC updating of marices)
#------------------------------------------------------------------------------

#--- Class to ease accessing to matrix data ------------------------------------

class CSR_Matrix(object):
    """Container for a sparse matrix whose data is accesed using CSR
    storage format

    Current implementation assumes that the PETSC backend is being used
    by FEniCS."""
    def __init__(self, FEniCS_matrix=None):
        self.FEniCS_matrix = FEniCS_matrix
        if FEniCS_matrix != None:
           underlying_PETSc_matrix = as_backend_type(self.FEniCS_matrix).mat()
           self.update_internal_data(underlying_PETSc_matrix)

    def update_internal_data(self, PETSc_matrix):
        self.underlying_PETSc_matrix = PETSc_matrix
        self.I, self.C, self.V = self.underlying_PETSc_matrix.getValuesCSR()
        self.size = self.underlying_PETSc_matrix.size
        self.nrows = self.size[0]
        self.ncols = self.size[1]

    def duplicate(self):
        "Returns a new CSR_Matrix which is a copy of this"
        new_PETSc_matrix = self.underlying_PETSc_matrix.duplicate()
        new_CSR_matrix = CSR_Matrix()
        new_CSR_matrix.update_internal_data(new_PETSc_matrix)
        return new_CSR_matrix

    def get_values_CSR(self):
        "Get data arrays defining content of current matrix, using CSR format"
        return self.I, self.C, self.V

    def set_values(self, values):
        "Set content of this matrix, assuming same nozero rows and columns"
        self.V = values
        self.underlying_PETSc_matrix.setValuesCSR(self.I, self.C, self.V)
        self.underlying_PETSc_matrix.assemble()

    def build_tranpose_into_matrix(self, CSR_mat):
        """Build into a CSR_Matrix the transpose of this.
        warning: previous contents of CSR_mat are not deleted, memory leak?"""
        # Build a new PETSc matrix
        transpose_mat = self.underlying_PETSc_matrix.duplicate()
        # Copy transpose of underlying_matrix into transpose_mat
        self.underlying_PETSc_matrix.transpose(transpose_mat)
        # Assign transpose_mat as underlying PETSc matrix of CSR_mat
        CSR_mat.update_internal_data(transpose_mat)

    def to_FEniCS_matrix(self):
        """Build a FEniCS matrix from current values internal CSR values or
        from new values stored in 'values_array'
        """
        return PETScMatrix(self.underlying_PETSc_matrix)

#--- Main Flux Correction class ------------------------------------------------

class KS_FluxCorrect_DefaultScheme(KS_Matrix_DefaultScheme):

    """
    Deafult Keller-Segel space/time scheme. Underlying FE matrices are
    explicitly build and corrected
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
        mass_form = u*ub*dx
        self.M = assemble( mass_form )

        # Mass lumping matrix
        mass_action_form = action(mass_form, Constant(1))
        self.ML = assemble(mass_form)
        self.ML.zero()
        self.ML.set_diagonal(assemble(mass_action_form))

        # Diffusion matrix
        self.L = assemble( dot(grad(u), grad(ub))*dx )

        # Matrix for the v-equation:
        self.Av = (1 + k3*dt)*self.ML + k2*dt*self.L

        # Save matrices
        if self.check_parameter("save_matrices"):
            save_coo_matrix(self.M, "M.matrix.coo")
            save_coo_matrix(self.ML, "ML.matrix.coo")
            save_coo_matrix(self.L, "L.matrix.coo")


    def solve(self):
        """Compute u and v"""

        dt, k0, k1, k4 = self.dt, self.k0, self.k1, self.k4

        ##,-------------------------------------------------------------
        ##| 1. compute v and gradient of v
        ##`-------------------------------------------------------------
        b = self.ML * (self.v0.vector() + k4*dt*self.u0.vector())
        solve ( self.Av, self.v.vector(), b )  # Solve A*v = b
        grad_v = project( grad(self.v), self.Wh )

        ##,-------------------------------------------------------------
        ##| 2. Define matrices and compute low order solution, u
        ##`-------------------------------------------------------------

        #
        # 2.1 Assemble chemotaxis transport matrix
        #
        u, ub = TrialFunction(self.Vh), TestFunction(self.Vh)
        v, vb = TrialFunction(self.Vh), TestFunction(self.Vh)
        C = assemble( u*dot(grad_v, grad(ub)) * dx )

        # Add diffusion matrix (at RHS) => complete advect+diffusion operator
        self.K = k1*C - k0*self.L;

        #
        # 2.2 Define D. It is an artifficial diffusion matrix D=d_{ij} such that
        # k_{ij} + d_{ij} >= 0 for all i,j
        #

        # Build object to access the FEniCS matrix as K a CSR matrix
        K_CSR = CSR_Matrix(self.K)
        # Get arrays defining the storage of K in CSR sparse matrix format,
        I, C, K_vals = K_CSR.get_values_CSR()

        # Build a new CSR matrix for the target matrix D
        D_CSR = K_CSR.duplicate()
        # Temporarily, we use the matrix D_CSR for building the transpose of K
        K_CSR.build_tranpose_into_matrix(D_CSR)
        # Get values for the transpose of K
        _, _, T_vals = D_CSR.get_values_CSR()

        # Build array with values max(0, -K_ij, -K_ji) for each row i
        D_vals = compute_D_values(I, C, K_vals, T_vals)

        # Create the new matrix D, storing the computed array D_vals
        D_CSR.set_values(D_vals)
        self.D =  D_CSR.to_FEniCS_matrix()

        #
        # 2.3 Eliminate all negative off-diagonal coefficients of K by
        # adding artifficial diffusion
        #
        self.KL = self.D + self.K

        #
        # 2.4 Compute low order solution
        #
        A = self.ML - dt*self.KL
        b = self.ML * self.u0.vector()
        solve (A, self.u.vector(), b)  # Solve A*u = b

        if self.check_parameter("save_matrices"):
            save_coo_matrix(self.K, "K.matrix.coo")
            save_coo_matrix(self.D, "D.matrix.coo")
            save_coo_matrix(self.KL, "KL.matrix.coo")


        if self.check_parameter("only_low_order_solution"):
            print("Computed the low order solution (only!!)")
            return()

        ##,-------------------------------------------------------------
        ##| 3. Update u system to high order solution
        ##`-------------------------------------------------------------

        #
        # 3.1 Compute residuals, f_ij = (m_ij + d_ij)*(u_j-u_i)
        #
        self.F = self.M - self.ML - dt*self.D

        if self.check_parameter("save_matrices"):
            save_coo_matrix(self.F, "FF_previous.matrix.coo")

        # ····· Update residuals: F_ij=0 if F_ij*(u_j-u_i) > 0

        # Build object to access the FEniCS matrix as K a CSR matrix
        F_CSR = CSR_Matrix(self.F)
        # Get arrays defining the storage of K in CSR sparse matrix format,
        I, C, F_vals = K_CSR.get_values_CSR()
        # Update F_vals array with values F_ij=0 if F_ij*(u_j-u_i) > 0
        F_vals = update_F_values(I, C, F_vals, U)
        # Create the new matrix F, storing the computed array F_vals
        F_CSR.set_values(F_vals)
        # IS NECCESARY NEXT LINE?
        # self.F =  F_CSR.to_FEniCS_matrix()

        # result = np.empty(len(fVals))

        # n = len(I)
        # for i in range(n-1):
        #     k0, k1 = I[row], I[row+1] # Pointers to begin and end of current row
        #     u_ij = u[C[k0:k1]] - u[row]
        #     result[k0:k1] = np.where(np.sign(u_ij)==np.sign(fVals[k0:k1]),
        #                          0, fVals[k0:k1])

        # # copy values into matrix D
        # fMat.setValuesCSR(I, C, fVals)
        # fMat.assemble()
        # self.FF = PETScMatrix(fMat)

        # t0 = time()
        # self.FF = update_F_values(self.FF, self.u.vector())
        # print("  ...time (update_F_values):", time()-t0)

        # if self.check_parameter("save_matrices"):
        #     save_coo_matrix(self.FF, "FF.matrix.coo")

        #
        # 3.2.  Compute the +,- sums of antidifusive fluxes to node i
        #
        # n = len(u)
        # for i in range(n-1):
        #     k0, k1 = I[row], I[row+1] # Pointers to begin and end of current row
