# -*- coding: utf-8 -*-
from fenics import *
import numpy as np
import matplotlib.pyplot as plt
from numpy.testing import assert_equal

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
    storage format Current implementation assumes that the PETSC
    backend is being used by FEniCS. The CSR_Matrix wraps this PETSC
    backend.
    """
    def __init__(self, FEniCS_matrix=None):
        if FEniCS_matrix != None:
           underlying_PETSc_matrix = as_backend_type(FEniCS_matrix).mat()
           self.update_internal_data(underlying_PETSc_matrix)

    def update_internal_data(self, PETSc_matrix):
        self.underlying_PETSc_matrix = PETSc_matrix
        self.I, self.C, self.V = self.underlying_PETSc_matrix.getValuesCSR()
        self.size = self.underlying_PETSc_matrix.size
        self._nrows = self.size[0]
        self._ncols = self.size[1]

    def nrows(self):
        return self._nrows #len(I)-1

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

    def assert_sparsity_pattern(self, other_CSR_mat):
        """Assert the position of nz elements math position in other matrix"""
        assert_equal(self.I, other_CSR_mat.I)
        assert_equal(self.C, other_CSR_mat.C)

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
        self.ML = assemble(mass_form) # !!!!
        #print("type ML:", type(self.ML))
        self.ML.zero()
        self.ML.set_diagonal(assemble(mass_action_form))
        #print("type Mass action form:", type(mass_action_form))
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
        U = self.u.vector()
        A = self.ML - dt*self.KL
        b = self.ML * self.u0.vector()
        solve (A, U, b)  # Solve A*u = b

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
        # 3.1 Compute raw flux: f_ij = (m_ij*d/dt + d_ij)*(u_j-u_i)
        #
        M_CSR = CSR_Matrix(self.M)
        M_CSR.assert_sparsity_pattern(D_CSR) # Same spasity is implicitly assumed below
        F_CSR = M_CSR.duplicate()

        # Get arrays defining the storage of M & F in CSR sparse matrix format
        I, C, M_vals = M_CSR.get_values_CSR()
        _, _, F_vals = F_CSR.get_values_CSR()

        # Access to PETSc vector data via numpy. This allows optimized code
        u_numpy = self.u.vector().vec().getArray()
        u0_numpy = self.u0.vector().vec().getArray()

        # Compute F values
        n = M_CSR.nrows()
        for i in range(n):
            # a) Get pointers to begin and end of nz elements in row i
            i0, i1 = I[i], I[i+1]
            jColumns = C[i0:i1]
            i_diag = i0 + index( jColumns, i )  # Pointer to diagonal elment

            diff_u_i  = u_numpy[i]  - u_numpy[jColumns]
            diff_u0_i = u0_numpy[i] - u0_numpy[jColumns]
            F_vals[i0:i1] = ( M_vals[i0:i1] * (diff_u_i - diff_u0_i) / dt +
                              D_vals[i0:i1] *  diff_u_i )

            F_vals[i_diag] = 0
            # F_vals[i_diag] = np.sum(F_vals[i0:i1])

        self.F =  F_CSR.to_FEniCS_matrix()

        if self.check_parameter("save_matrices"):
            save_coo_matrix(self.F, "FF_previous.matrix.coo")

        # ····· Update residuals: F_ij=0 if F_ij*(u_j-u_i) > 0
        F_vals = update_F_values(I, C, F_vals, u_numpy)
        F_CSR.set_values(F_vals)
        # IS NECCESARY NEXT LINE?
        # self.F =  F_CSR.to_FEniCS_matrix()

        if self.check_parameter("save_matrices"):
            save_coo_matrix(self.F, "FF.matrix.coo")

        #
        # 3.2.  Compute the +,- sums of antidifusive fluxes to node i
        #
        n = len(u_numpy)
        Pplus = np.empty(n);  Pminus = np.empty(n)
        for i in range(n):
            # a) Get pointers to begin and end of nz elements in row i
            i0, i1 = I[i], I[i+1]
            i_diag = i0 + index( C[i0:i1], i )  # Pointer to diagonal elment

            F0, F1 = F_vals[i0:i_diag], F_vals[i_diag+1:i1]
            Pplus[i]  = np.sum( np.maximum(F0,0) ) + np.sum( np.maximum(F1,0) )
            Pminus[i] = np.sum( np.minimum(F0,0) ) + np.sum( np.minimum(F1,0) )

        Qplus = np.empty(n);  Qminus = np.empty(n)
        for i in range(n):
            # a) Get pointers to begin and end of nz elements in row i
            i0, i1 = I[i], I[i+1]
            i_diag = i0 + index( C[i0:i1], i )  # Pointer to diagonal elment
            j_cols =  C[i0:i1] # Nz columns in row i
            j_cols_but_i = j_cols[ j_cols!=i ]
            # print("i =", i, ":", j_cols_non_i)

            # b) Compute u[j] - u[i] for all columns j in row i
            U_ji = u_numpy[j_cols_but_i] - u_numpy[i]

            Qplus[i]  = np.maximum( np.max(U_ji), 0 )
            Qminus[i] = np.minimum( np.min(U_ji), 0 )

        Rplus = np.empty(n);  Rminus = np.empty(n)
        # Object to access the FEniCS matrix ML a CSR matrix
        ML_CSR = CSR_Matrix(self.ML)
        # Get arrays defining the storage of ML in CSR sparse matrix format,
        _, _, ML_vals = ML_CSR.get_values_CSR()
        j = 0
        N = len(ML_vals)
        ML_diagonal = np.zeros(n)
        fp_tol = 1.e-20
        for i in range(N):
            if abs(ML_vals[i])>0:
                ML_diagonal[j]=ML_vals[i]
                j=j+1
        print("ML_diagonal:", ML_diagonal)
        for i in range(n):
            if abs(Pplus[i]) < fp_tol:
                Rplus[i] = 0
            else:
                Rplus[i] = np.minimum(1, Qplus[i]*ML_diagonal[i]/(dt*Pplus[i]))
            if abs(Pminus[i]) < fp_tol:
                Rminus[i] = 0
            else:
                Rminus[i] = np.minimum(1, Qminus[i]*ML_diagonal[i]/(dt*Pminus[i]))

        print("#Pplus:",  Pplus);
        print("#Pminus:", Pminus);
        print("#Qplus:",  Qplus);
        print("#Qminus:", Qminus);
        print("#ML_vals:", ML_vals);
        print("#Rplus:",  Rplus);
        print("#Rminus:", Rminus);

        alpha_CSR = F_CSR.duplicate()
        # Get arrays defining the storage of ML in CSR sparse matrix format,
        I, C, alpha_vals = alpha_CSR.get_values_CSR()
        for i in range(n):
            i0, i1 = I[i], I[i+1]
            jColumns = C[i0:i1]
            alpha_vals[i0:i1] = np.where(
                F_vals[i0:i1]>0,
                np.minimum(Rplus[i],  Rminus[jColumns]),
                np.minimum(Rminus[i], Rplus[jColumns])
                )
        alpha_CSR.set_values(alpha_vals)
        self.alpha =  alpha_CSR.to_FEniCS_matrix()
        if self.check_parameter("save_matrices"):
            print("Saving alpha")
            save_coo_matrix(self.alpha, "alpha.matrix.coo")

        #########################

        barFunction = Function(self.Vh);
        barf = barFunction.vector()
        for i in range(n):
            i0, i1 = I[i], I[i+1]
            i_diag = i0 + index( C[i0:i1], i )  # Pointer to diagonal elment
            # Under- and super-diagonal values
            F0, F1 = F_vals[i0:i_diag], F_vals[i_diag+1:i1]
            alpha0, alpha1 = alpha_vals[i0:i_diag], alpha_vals[i_diag+1:i1]
            barf[i] = np.sum( F0*alpha0 ) + np.sum( F1*alpha1 )

        A = self.ML - dt*self.KL
        b = self.ML * self.u0.vector() - dt*barf
        print("Solving high order scheme")
        print("barf=", barf.vec().getArray())
        solve (A, self.u.vector(), b)  # Solve A*u = b
