from fenics import *
import numpy as np
import matplotlib.pyplot as plt
from time import time
from petsc4py import PETSc

# FEniCS: only warnings (default: 20, information of general interet)
set_log_level(30)

# Try to import numba (high performance python compiler!!). If numba is not
# found, define empty wrappers for functions njit and prange
import importlib
try:
    numba_loader = importlib.import_module("numba")
    from numba import njit, prange
except ImportError:
    print("Numba module (high performance python compiler) not found")
    #1. Define wrapper for prange
    def prange(arg): return range(arg)
    # 2. Define wrapper for the njit decorator,
    # https://stackoverflow.com/questions/653368/how-to-create-a-python-decorator-that-can-be-used-either-with-or-without-paramet
    from functools import wraps
    def doublewrap(f):
        @wraps(f)
        def new_dec(*args, **kwargs):
            if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
                # actual decorated function
                return f(args[0])
            else:
                # decorator arguments
                return lambda realf: f(realf, *args, **kwargs)
        return new_dec
    @doublewrap
    def njit(f, parallel=2):
        '''Just return f(), with no parallelism'''
        @wraps(f)
        def wrap(*args, **kwargs): return f(*args,**kwargs)
        return wrap



# Module used for defining abstract classes
from abc import ABC, abstractmethod

#==============================================================================
# class KS_AbstractScheme:
#   Define abstract behaviour of all Keller-Segel objects
#------------------------------------------------------------------------------

class KS_AbstractScheme(ABC):
    """
    Abstract class for Keller-Segel tests.
    """

    def __init__( self, mesh, fe_order, dt, t_init,
                  k0, k1, k2, k3, k4 ):
        self.mesh = mesh
        self.fe_order = fe_order
        self.dt = dt
        self.t_init = t_init
        self.t = t_init
        self.k0 = k0
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.k4 = k4

        #
        # Build FE spaces and FE functions
        #
        assert(fe_order>0)
        self.Vh = FunctionSpace(mesh, "Lagrange", fe_order)
        #
        # Space for interpolation of gradient(v)
        #
        self.Wh = VectorFunctionSpace(mesh, "DG", fe_order-1)
        self.grad_v = Function(self.Wh);
        #
        # Variables to store solution at previous time step
        #
        self.u0 = Function(self.Vh);
        self.v0 = Function(self.Vh);
        #
        # Prepare dictonary for defining custom parameters
        #
        self.parameters = {}

    def check_parameter(self, parameter_key, parameter_value=True):
        "Checks if a parameter exists and matches a given value"
        return ( parameter_key in self.parameters and
                 self.parameters[parameter_key] == parameter_value )

    def set_parameter(self, parameter_key, parameter_value=True):
        "Sets a value for a parameter exists"
        self.parameters[parameter_key] = parameter_value

    @abstractmethod
    def build_fe_scheme(self):
        """
        Define variational formulation and FE system(s) which define current
        scheme (each derived class must implement this method)"""
        pass

    @abstractmethod
    def solve(self):
        """Solve the FE equation system(s) at a given time step.
        (Each derived class must implement this method)"""
        pass


    def set_u(self, u_init):
        self.u0.assign( interpolate(u_init, self.Vh) )


    def set_v(self, v_init):
        self.v0.assign( interpolate(v_init, self.Vh) )


    def run(self, nt_steps, break_when_negative_u=False,
            plot_u=True):
        #
        # Build variational formulation and matrices
        #
        self.build_fe_scheme()
        #
        # Run time iterations
        #
        self.u, self.v = Function(self.Vh), Function(self.Vh)
        for iter in range(nt_steps):
            self.t = self.t + self.dt
            print(f"Time iteration {iter}, t={self.t:.2e}")
            #
            # Compute u and v
            #
            self.solve()
            #
            # Save solution (to be used in next iteration)
            #
            self.u0.assign(self.u)
            self.v0.assign(self.v)
            #
            # Print info
            #
            u_max, u_min = max(self.u.vector()), min(self.u.vector())
            v_max, v_min = max(self.v.vector()), min(self.u.vector())
            end_of_line = '\n' if(u_min>0) else " <<< positivity broken!!\n"
            print(f"  máx(u)={u_max:.5e}, min(u)={u_min:.5e}", end=end_of_line)
            print(f"  máx(v)={v_max:.5e}, min(v)={v_min:.5e}")
            integral_u = assemble(self.u*dx); print(f"  int(u)={integral_u:.5e}" )

            #
            # Plot
            #
            if plot_u and (iter % 10 == 0): # Only draw some fotograms
                plot(self.u, title=f"u, t={self.t:.2e}", mode="warp")
                plt.show()

            if(u_min<=0 and break_when_negative_u):
                break

        return {'iter': iter, 't': self.t, 'u': self.u, 'v': self.v}


#==============================================================================
# class KS_DefaultScheme:
#   Usual KS implementation
#------------------------------------------------------------------------------

class KS_DefaultScheme(KS_AbstractScheme):
    """Deafult Keller-Segel space/time scheme.

    Specifically, we define the scheme (1,1,1,0), using the notation
    of [Alba N.I.,
    TFG](https://rodin.uca.es/xmlui/bitstream/handle/10498/21139/TFG.pdf)
    """

    def __init__( self, mesh, fe_order, dt, t_init=0.,
                  k0=1, k1=1, k2=1, k3=1, k4=1 ):
        super().__init__(mesh, fe_order, dt, t_init, k0, k1, k2, k3, k4)


    def build_fe_scheme(self):
        """
        Create variational equations and FE systems which define current scheme
        """
        u, ub = TrialFunction(self.Vh), TestFunction(self.Vh)
        v, vb = TrialFunction(self.Vh), TestFunction(self.Vh)
        #
        # Define variational formulation for u and v
        #
        dt, k0, k1, k2, k3, k4 \
            = self.dt, self.k0, self.k1, self.k2, self.k3, self.k4

        self.a_u = u*ub * dx + dt*k0*dot(grad(u), grad(ub)) * dx \
                   - dt*k1*u*dot(self.grad_v, grad(ub)) * dx
        self.f_u = self.u0*ub * dx

        self.a_v = v*vb * dx + dt*k2*dot(grad(v), grad(vb)) * dx \
              + dt*k3*v*vb * dx
        self.f_v = (self.v0*vb + dt*k4*self.u0*vb) * dx


    def solve(self):
        """Compute u and v"""
        #
        # Compute v
        #
        solve ( self.a_v == self.f_v, self.v )
        #
        # Compute gradient of v
        self.grad_v.assign( project( grad(self.v), self.Wh ) )
        #
        # Compute u
        #
        solve ( self.a_u == self.f_u, self.u )


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


#==============================================================================
# class KS_FluxCorrect_DefaultScheme:
#   - Keller-Segel with flux-correction (FC)
#   - Used Numba compiler for high performance (in FC updating of marices)
#------------------------------------------------------------------------------

@njit
def index(array, item):
    # Find item in array, return its index
    for idx, val in np.ndenumerate(array):
        if val == item:
            break
    return idx[0]

# @njit(parallel=True)
@njit
def compute_D_values(I, C, kVals, dVals):
    # 1. To parallize: initialize an output with the desired length
    out = np.empty(len(kVals))
    # 2. Use prange (and avoid "race conditions"!)
    for row in prange(len(I)-1):
        k0, k1 = I[row], I[row+1] # Pointers to begin and end of current row
        k_diag = k0 + index( C[k0:k1], row )
        # # Compute max(-k_{ij}, -k_{ji}, 0 )
        out[k0:k1] = np.maximum( np.zeros(k1-k0),
                                 np.maximum(-kVals[k0:k1], -dVals[k0:k1]) )
        row_sum = np.sum(out[k0:k1]) - out[k_diag]
        out[k_diag] = -row_sum
    return out

def update_F_values(F, u):
    """
    Update residuals, F_{ij}, so that  F_ij=0 if F_ij*(u_j-u_i) > 0
    Arguments:
      * F: PETScMatrix containing preliminar fluxes
      * u: numpy array containing low orde solution
    Returns:
      PETScMatrx containing corrected fluxes
    """
    # create object from underlying matrix library and get values to numpy array
    fMat = as_backend_type(F).mat()
    I, C, fVals = fMat.getValuesCSR()

    # Let F_ij=0 if F_ij*(u_j-u_i) > 0
    result = np.empty(len(fVals))
    for row in range(len(I)-1):
        k0, k1 = I[row], I[row+1] # Pointers to begin and end of current row
        u_ij = u[C[k0:k1]] - u[row]
        result[k0:k1] = np.where(np.sign(u_ij)==np.sign(fVals[k0:k1]),
                                 0, fVals[k0:k1])

    # copy values into matrix D
    fMat.setValuesCSR(I, C, fVals)
    fMat.assemble()
    return PETScMatrix(fMat)

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
        return self.I, self.C, self.V

    def set_values(self, values):
        self.V = values
        self.underlying_PETSc_matrix.setValuesCSR(self.I, self.C, self.V)
        self.underlying_PETSc_matrix.assemble()

    def build_tranpose_into_matrix(self, CSR_mat):
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


class Old_CSR_Matrix(object):
    """
    Store a sparse matrix in CSR format.

    We assume that the PETSC backend is being used by FEniCS.
    """
    def __init__(self, fenics_matrix):
        # Store fenics matrix
        self.fenics_matrix = fenics_matrix
        # Store PETSC (petsc4py) matrix (we assume fenics uses this backend)
        self.underlying_matrix = as_backend_type(self.fenics_matrix).mat()
        # Save matrix internal data (rows, pointer to columns, values)
        self.I, self.C, self.V = self.underlying_matrix.getValuesCSR()
        self.size = self.underlying_matrix.size
        self.nrows = self.size[0]
        self.ncols = self.size[1]

    def get_row(self, row):
        "Return list of columns and list of values for all the non-zero columns in row i"
        # Get pointer to first column in a row
        k0 = self.I[row]
        # Pointer to end of colum in a row (if i is last row, k1=end of columns)
        k1 = self.I[row+1] if row < self.nrows else self.ncols

        return self.C[k0:k1], self.V[k0:k1]

    def row_begin_end(self, i_row):
        """Return pointers to data respective to begin and end
        of nonzero columns in i_row"""
        # Get pointer to first column in a row
        k0 = self.I[i_row]
        # Pointer to end of colum in a row (if i is last row, k1=end of columns)
        k1 = self.I[i_row + 1] if i_row < self.nrows else self.ncols

        return k0,k1

    def row_diagonal(self, i_row, row_begin, row_end):
        """Return pointer to data respective diagonal in i_row"""
        k_diag = self.I[i_row] + index( self.C[row_begin:row_end], i_row )

    def set_row_values(self, row, values):
        """Modify values for all non-zero columns in a row"""
        # Get pointer to first column in a row
        k0 = self.I[row]
        # Pointer to end of colum in a row (if i is last row, k1=end of columns)
        k1 = self.I[row+1] if row < self.nrows else self.ncols

        self.V[k0:k1] = values[:]

    def assemble_values(self):
        self.underlying_matrix.setValuesCSR(self.I, self.C, self.V)
        self.underlying_matrix.assemble()


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

    # def compute_artificial_diffusion_v0(self, K):
    #     """Define an artifficial diffusion matrix D such that
    #      k_{ij} + d_{ij} >= 0 for all i,j"""

    #     # create object from underlying matrix library
    #     kMat = as_backend_type(K).mat()
    #     dmat = kMat.duplicate()

    #     # copy transpose of kMat into dmat
    #     kMat.transpose(dmat)

    #     # get values from kMat and dmat
    #     I, C, kVals = kMat.getValuesCSR()
    #     _, _, dVals = dmat.getValuesCSR()

    #     # compute values for matrix D
    #     for row in range(len(I)-1):
    #         row_sum = 0
    #         k_diag = None # Pointer to diagonal in current row
    #         for k in range(I[row], I[row+1]): # For non zero columns in row
    #             col = C[k]
    #             if row == col:
    #                 k_diag = k # Diagonal position localized
    #             else:
    #                 dVals[k] = max(0, max(-kVals[k], -dVals[k])) # Compute diffusion
    #                 row_sum += dVals[k]
    #         assert k_diag!=None
    #         dVals[k_diag] = -row_sum
    #     # dvals = - reduce(numpy.minimum, (kvals, dvals, 0))
    #     # np.where (i-j==0, 0, dvals)   # Put 0 in diagonal positions

    #     # copy values into matrix D
    #     dmat.setValuesCSR(I, C, dVals)
    #     dmat.assemble()
    #     return PETScMatrix(dmat)

    # def compute_artificial_diffusion_v1(self, K):
    #     """Define an artifficial diffusion matrix D such that
    #      k_{ij} + d_{ij} >= 0 for all i,j"""

    #     # create object from underlying matrix library
    #     kMat = as_backend_type(K).mat()
    #     dmat = kMat.duplicate()

    #     # copy transpose of kMat into dmat
    #     kMat.transpose(dmat)

    #     # get values from kMat and dmat
    #     I, C, kVals = kMat.getValuesCSR()
    #     _, _, dVals = dmat.getValuesCSR()

    #     # compute values for matrix D
    #     # print(f"len(I)={len(I)}")
    #     t0 = time()
    #     dVals = compute_D_array(I, C, kVals, dVals)
    #     print("  ...time (compute_D_array):", time()-t0)

    #     # copy values into matrix D
    #     dmat.setValuesCSR(I, C, dVals)
    #     dmat.assemble()
    #     return PETScMatrix(dmat)

    def compute_artificial_diffusion(self, K):
        """Define an artifficial diffusion matrix D such that
         k_{ij} + d_{ij} >= 0 for all i,j"""

        # 1) Create object from underlying matrix library
        K_mat = CSR_Matrix(K)
        # And get arrays of values stored in K_mat
        K_I = K_mat.I; K_C = K_mat.C; K_vals = K_mat.V

        # 2) Build array of values for the transpose of K_mat
        K_petsc = K_mat.underlying_matrix
        T_mat = K_petsc.duplicate()
        # Copy transpose of K_mat into T_mat
        K_petsc.transpose(T_mat)
        # Get array of internal values of T_mat
        _, _, T_vals = T_mat.getValuesCSR()

        # 3) Compute values max(0, -K_ij, -K_ji) for each row i

        new_vals = np.zeros_like(K_vals)
        for i_row in range(K_mat.nrows):
            # Get pointers to non-zero elements
            k0, k1 = K_mat.row_begin_end(i_row)

            # Compute values before diagonal
            zeros = np.zeros(k1-k0)
            new_vals[k0:k1] = np.maximum( zeros,
                np.maximum(-K_vals[k0:k1], -T_vals[k0:k1]) )

            # Let $K_{ii} = - \sum_{j\neq i} K_{ij}
            k_diag = K_mat.row_diagonal(i_row, row_begin=k0, row_end=k1)
            new_vals[k_diag] = 0
            row_sum = np.sum(new_vals[k0:k1])
            new_vals[k_diag] = -row_sum

        #
        # 4) Store new values in K_mat
        #
        K_mat.V = new_vals
        K_mat.assemble_values()

        print(K_I, K_C, K_vals)
        return PETScMatrix(K_petsc)


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
        # self.D = self.compute_artificial_diffusion(self.K)

        # Build object to access to the FEniCS matrix as K a CSR matrix
        K_CSR = CSR_Matrix(self.K)
        # Get arrays defining the storage of K in CSR sparse matrix format,
        I, C, K_vals = K_CSR.get_values_CSR()

        # Build a new CSR matrix chich for the target matrix D
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

        ##,-------------------------------------------------------------
        ##| 3. Update u system to high order solution
        ##`-------------------------------------------------------------

        #
        # 3.1 Compute residuals, f_ij = (m_ij + d_ij)*(u_j-u_i)
        #
        self.FF = self.M - self.ML - dt*self.D

        if self.check_parameter("save_matrices"):
            save_coo_matrix(self.FF, "FF_previous.matrix.coo")

        ## Update residuals:  FF_ij=0 if FF_ij*(u_j-u_i) > 0

        # # Get underlying backend matrix. We assume backend is PETSC
        # # and type(fMat)==<class 'petsc4py.PETSc.Mat'>
        # # For a whole description of this class, see e.g.
        # # https://www.mcs.anl.gov/petsc/petsc4py-current/docs/apiref/petsc4py.PETSc.Mat-class.html
        # fMat = as_backend_type(self.FF).mat()

        # # Let F_ij=0 if F_ij*(u_j-u_i) > 0
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
