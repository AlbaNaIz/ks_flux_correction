#
# Keller-Segel Test 0
# ===================
#
# We explore the suitability of FEniCS for solving Keller-Segel equations.
#
# We use a time scheme (1,1,1,0), notation from [Alba N.I.,
# TFG](https://rodin.uca.es/xmlui/bitstream/handle/10498/21139/TFG.pdf),
# and data from [Chertok-Kurganov: 2008]
#

from fenics import *
import matplotlib.pyplot as plt
set_log_level(30) # Only warnings (default: 20, information of general interet)

if( __name__ == "__main__" ):
    #
    # Build mesh and define data
    #
    nx = 200
    mesh = RectangleMesh(p0=Point(-2,-2), p1=Point(2,2), nx=nx, ny=nx)
    do_plot = False
    if do_plot: plot(mesh)
    # plt.show()
    dt = 1.e-4
    nt_steps = 200
    k0=1; k1=0.2; k2=1; k3=0.1; k4=1

    #
    # Build FE spaces and FE functions
    #
    fe_order = 1; assert(fe_order>0)
    Uh = FunctionSpace(mesh, "Lagrange", fe_order)
    Vh = Uh
    u, ub = TrialFunction(Uh), TestFunction(Uh)
    v, vb = TrialFunction(Vh), TestFunction(Vh)
    # Space for interpolation of gradient(v)
    Wh = VectorFunctionSpace(mesh, "DG", fe_order-1)
    grad_v = Function(Wh);
    #
    # Define initial conditions (Chertok-Kurganov test)
    #
    # u_init = Expression("1.15*exp(-(x[0]*x[0]+x[1]*x[1]))*(4-x[0]*x[0])**2*(4-x[1]*x[1])**2", degree=fe_order)
    # v_init = Expression("0.55*exp(-(x[0]*x[0]+x[1]*x[1]))*(4-x[0]*x[0])**2*(4-x[1]*x[1])**2", degree=fe_order)
    u_init = Expression("1.15*exp(-(x[0]*x[0]+x[1]*x[1]))*(4-x[0]*x[0])*(4-x[0]*x[0])*(4-x[1]*x[1])*(4-x[1]*x[1])", degree=fe_order)
    v_init = Expression("0.55*exp(-(x[0]*x[0]+x[1]*x[1]))*(4-x[0]*x[0])*(4-x[0]*x[0])*(4-x[1]*x[1])*(4-x[1]*x[1])", degree=fe_order)
    u0 = interpolate(u_init, Uh)
    v0 = interpolate(v_init, Vh)
    #
    # Define variational formulation for u and v
    #
    a_u = u*ub * dx + dt*k0*dot(grad(u), grad(ub)) * dx \
          - dt*k1*u*dot(grad_v, grad(ub)) * dx
    f_u = u0*ub * dx

    a_v = v*vb * dx + dt*k2*dot(grad(v), grad(vb)) * dx \
          + dt*k3*v*vb * dx
    f_v = (v0*vb + dt*k4*u0*vb) * dx

    #
    # Run time iterations
    #
    t=0
    u, v = Function(Uh), Function(Vh)
    for iter in range(nt_steps):
        t = t + dt
        print(f"Time iteration {iter}, t={t:.2e}")
        #
        # Compute v
        #
        solve ( a_v == f_v, v )
        grad_v.assign( project( grad(v), Wh ) )  # Compute gradient of v
        #
        # Compute u
        #
        solve ( a_u == f_u, u )
        #
        # Print info
        #
        u_max, u_min = max(u.vector()), min(u.vector())
        v_max, v_min = max(v.vector()), min(u.vector())
        end_of_line = '\n' if(u_min>0) else " <<< positivity broken!!\n"
        print(f"  máx(u)={u_max:.5e}, min(u)={u_min:.5e}", end=end_of_line)
        print(f"  máx(v)={v_max:.5e}, min(v)={v_min:.5e}")
        integral_u = assemble(u*dx); print(f"  int(u)={integral_u:.5e}" )
        #
        # Plot
        #
        if (iter % 10 == 0 and do_plot): # Only draw some fotograms
            plot(u, title=f"u, t={t:.2e}", mode="warp")
            plt.show()
        #
        # Prepare next iteration
        #
        u0.assign(u)
        v0.assign(v)
