from fenics import *
import numpy as np
import matplotlib.pyplot as plt

from keller_segel.abstract_scheme import KS_AbstractScheme

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
