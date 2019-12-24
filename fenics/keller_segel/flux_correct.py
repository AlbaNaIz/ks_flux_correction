# -*- coding: utf-8 -*-
from fenics import *
import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix, coo_matrix

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
        
        print("Low order solution:")
        print("max(u)=" , max(U.vec().getArray()))
        print("min(u)=" , min(U.vec().getArray()))
        
        if self.check_parameter("save_matrices"):
            save_coo_matrix(self.K, "K.matrix.coo")
            save_coo_matrix(self.D, "D.matrix.coo")
            save_coo_matrix(self.KL, "KL.matrix.coo")


        if self.check_parameter("only_low_order_solution"):
            print("Computed the low order solution (only!!)")
            return()
            
        ##,-------------------------------------------------------------
        ##| 3. Update u system to high order solution
        ##`------------------------------------------------------------
        
        #
        # 3.1 Getting M and D matrix as coo type
        #
        
        print("Getting M and D matrix as coo type")
        
        Aux = as_backend_type(self.M).mat()
        coo_M = coo_matrix(csr_matrix(Aux.getValuesCSR()[::-1]))
        
        nM = coo_M.row[-1] + 1
        
        Aux = as_backend_type(self.D).mat()
        coo_D = coo_matrix(csr_matrix(Aux.getValuesCSR()[::-1]))
        
        nD = coo_D.row[-1] + 1
        
        #
        # 3.2 Generating M and D matrices synchronizing indices to optimize the algorithm
        #
        
        print("Generating M and D matrices synchronizing indices to optimize the algorithm")
        
  
        IJM = list(range(len(coo_M.row)))

        for k in range(len(coo_M.row)):
            #if coo_M.row[k] < coo_M.col[k]:
            IJM[k] = (coo_M.row[k],coo_M.col[k])
        
        
        IJD = list(range(len(coo_D.row)))
        
        for k in range(len(coo_D.row)):
            #if coo_M.row[k] < coo_M.col[k]:
            IJD[k] = (coo_D.row[k],coo_D.col[k])
        
        
        IJF = sorted(list(dict.fromkeys(IJM+IJD)))
        
        M_sync = list(range(len(IJF)))
        D_sync = list(range(len(IJF)))
        
        kf = 0
        km = 0
        kd = 0
        
        for (i,j) in IJF:
        
            if (i,j) == (coo_M.row[km],coo_M.col[km]):
                M_sync[kf] = coo_M.data[km]
                km = km + 1
            else:
                M_sync[kf] = 0
                
            if (i,j) == (coo_D.row[kd],coo_D.col[kd]):
                D_sync[kf] = coo_D.data[kd]
                kd = kd + 1
            else:
                D_sync[kf] = 0
                
            kf = kf + 1
          
        #
        # 3.3 Computing residuals, f_ij = (m_ij + d_ij)*(u_i-u_j)
        #
        
        print("Computing residuals")
        
        u_m1 = U.vec().getArray()
        u_m0 = self.u0.vector().vec().getArray()
        
        F_dat = list(range(len(IJF)))
        F_row = list(range(len(IJF)))
        F_col = list(range(len(IJF)))
        
        r = 0
        s = 0
        
        for (i,j) in IJF:
            
            mij = M_sync[r]
            dij = D_sync[r]
            
            fij = mij/dt * ((u_m1[i]-u_m1[j]) - (u_m0[i]-u_m0[j])) + dij * (u_m1[i]-u_m1[j])
            
            if fij * (u_m1[j] - u_m1[i]) <= 0 and fij != 0:
                
                F_dat[s] = fij
                F_row[s] = i
                F_col[s] = j
                
                s = s + 1
            
            r = r + 1
        
        # Get just the s first values saved
        F_dat = F_dat[:s]
        F_row = F_row[:s]
        F_col = F_col[:s]
        
        # Build coo F matrix
        coo_F = coo_matrix((F_dat, (F_row, F_col)), shape=(nM,nM))
        
        # Save matrices
        if self.check_parameter("save_matrices"):
            np.savetxt("F.matrix.coo",
            np.c_[coo_F.row, coo_F.col, coo_F.data],
            fmt=['%d', '%d', '%.16f'])
        
        #
        # 3.4 Generating ML diagonal vector to optimize the algotithm
        #
        
        print("Generating ML diagonal vector to optimize the algotithm")
        
        # Object to access the FEniCS matrix ML a CSR matrix
        ML_CSR = CSR_Matrix(self.ML)
        
        # Get arrays defining the storage of ML in CSR sparse matrix format,
        _, _, ML_vals = ML_CSR.get_values_CSR()
        
        ML_diag = np.zeros(nM)
        
        j = 0
        for i in range(len(ML_vals)):
            if ML_vals[i]!=0:
                ML_diag[j]=ML_vals[i]
                j=j+1
        
        #
        # 3.5 Computing R+ and R-
        #
        
        print("Computing R+ and R-")
        
        Rp = np.empty(nM);  
        Rm = np.empty(nM)
        
        tol = 1.e-20
        
        k = 0
        
        for i in list(dict.fromkeys(F_row)):
                
            Pp = 0
            Pm = 0
            
            while k < len(F_row) and i == F_row[k]:
                
                if i != F_col[k]:
                    Pp = Pp + np.maximum(0,F_dat[k])
                    Pm = Pm + np.minimum(0,F_dat[k])
                
                k = k + 1
            
            Qp = np.maximum(np.max(u_m1-u_m1[i]),0)
            Qm = np.minimum(np.min(u_m1-u_m1[i]),0)
            
            if Pp < tol:
                Rp[i] = 0
            else: 
                Rp[i] = np.minimum(1, ML_diag[i]*Qp/(dt*Pp))
            
            if -Pm < tol:
                Rm[i] = 0
            else:
                Rm[i] = np.minimum(1, ML_diag[i]*Qm/(dt*Pm))
        
        #
        # 3.6 Compunting alpha
        #
        
        print("Computing alpha")
        
        alpha = list(range(len(F_dat)))
       
        for k in range(len(F_dat)):
            
            i = F_row[k]
            j = F_col[k]
            
            if F_dat[k] > 0: 
                alpha[k] = np.minimum(Rp[i],Rm[j])
            else:
                alpha[k] = np.minimum(Rm[i],Rp[j])
        
        #
        # 3.7 Computing f bar
        #
        
        print("Computing f bar")
        
        barFunction = Function(self.Vh);
        barf = barFunction.vector()
        
        k = 0
        
        for i in list(dict.fromkeys(F_row)):
                
            barf[i] = 0
            
            while k < len(F_row) and i == F_row[k]:
                
                if i != F_col[k]:
                    barf[i] = barf[i] + alpha[k]*F_dat[k]
                
                k = k + 1
               
        print("barf = ", barf.vec().getArray())
        
        #
        # 3.8 Solving high order scheme
        #
        
        print("Solving high order scheme")
        
        A = self.ML - dt*self.KL
        b = self.ML * self.u0.vector() - barf
        
        solve (A, self.u.vector(), b)  # Solve A*u = b
