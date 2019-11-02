from fenics import *
import matplotlib.pyplot as plt

set_log_level(30) # Only warnings (default: 20, information of general interet)

# Module used for defining abstract classes
from abc import ABC, abstractmethod

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
        self.Uh = FunctionSpace(mesh, "Lagrange", fe_order)
        self.Vh = self.Uh
        #
        # Space for interpolation of gradient(v)
        #
        self.Wh = VectorFunctionSpace(mesh, "DG", fe_order-1)
        self.grad_v = Function(self.Wh);
        #
        # Variables to store solution at previous time step
        #
        self.u0 = Function(self.Uh);
        self.v0 = Function(self.Vh);
        #
        # Construct variational formulation for u and v
        #
        self.build_fe_scheme()


    @abstractmethod
    def build_fe_scheme(self):
        """
        Define variational formulation and FE system(s) which define current scheme
        (each derived concrete class must implement this method)"""
        pass


    @abstractmethod
    def solve(self):
        """Solve the FE equation system(s) at a given time step.
        (Each derived concrete class must implement this method)"""
        pass


    def set_u(self, u_init):
        self.u0.assign( interpolate(u_init, self.Uh) )


    def set_v(self, v_init):
        self.v0.assign( interpolate(v_init, self.Vh) )


    def run(self, nt_steps, break_when_negative_u=False):
        #
        # Run time iterations
        #
        self.u, self.v = Function(self.Uh), Function(self.Vh)
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
            if (iter % 10 == 0): # Only draw some fotograms
                plot(self.u, title=f"u, t={self.t:.2e}", mode="warp")
                plt.show()

            if(u_min<=0 and break_when_negative_u):
                break

        return {'iter': iter, 't': self.t, 'u': self.u, 'v': self.v}


class KS_DefaultScheme(KS_AbstractScheme):
    """
    Deafult Keller-Segel space/time scheme.

    Specifically, we define the scheme (1,1,1,0), using the notation
    of [Alba N.I., TFG]
    """

    def __init__( self, mesh, fe_order, dt, t_init=0.,
                  k0=1, k1=1, k2=1, k3=1, k4=1 ):
        super().__init__(mesh, fe_order, dt, t_init, k0, k1, k2, k3, k4)


    def build_fe_scheme(self):
        """
        Define variational equations and FE systems which define current scheme
        """
        u, ub = TrialFunction(self.Uh), TestFunction(self.Uh)
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
