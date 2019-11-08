from fenics import *
import numpy as np
import matplotlib.pyplot as plt

set_log_level(30) # Only warnings (default: 20, information of general interet)

# Module used for defining abstract classes
from abc import ABC, abstractmethod

#==============================================================================

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
        # Construct variational formulation for u and v
        #
        self.build_fe_scheme()


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

def save_coo_matrix(A, filename):
    "Save matrix A in a file, using COO format (i, j, value_ij)"
    from scipy.sparse import csr_matrix, coo_matrix
    Mmat = as_backend_type(A).mat()
    coo_mat = coo_matrix(csr_matrix(Mmat.getValuesCSR()[::-1]))
    np.savetxt(filename,
            np.c_[coo_mat.row, coo_mat.col, coo_mat.data],
            fmt=['%d', '%d', '%.16f'])
    # x_coo = x.tocoo()
    # row = coo_mat.row
    # col = coo_mat.col
    # data = coo_mat.data
    # shape = coo_mat.shape
    # np.savez(filename, row=row, col=col, data=data, shape=shape)

class KS_MatrixDefaultScheme(KS_AbstractScheme):
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


class KS_FC_DefaultScheme(KS_AbstractScheme):

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
        self.Av = (1 + k3*dt)*self.M + k2*dt*self.L

        # Save matrices
        save_matrices = True
        if save_matrices:
            save_coo_matrix(self.M, "M.matrix.coo")
            save_coo_matrix(self.ML, "ML.matrix.coo")

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
