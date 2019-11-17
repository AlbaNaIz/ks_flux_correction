from fenics import *
import numpy as np
import matplotlib.pyplot as plt

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
