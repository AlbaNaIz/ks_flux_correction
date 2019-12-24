# Define data
from fenics import RectangleMesh, Point, Expression

# Define default Chertok-Kurganov variables

nx = 200  # $h \approx 1/128$
fe_order = 1
dt = 1.e-4


# Functions to build mesh and initial conditions

def build_mesh():
    mesh = RectangleMesh(p0=Point(-2,-2), p1=Point(2,2),
                         nx=nx, ny=nx)
    return mesh

def reset(_nx=nx, _dt=dt, _fe_order=fe_order):
    global nx, mesh, u_init, v_init
    nx = _nx
    dt = _dt
    mesh = build_mesh()

    u_init = Expression("1.15*exp(-(x[0]*x[0]+x[1]*x[1]))*(4-x[0]*x[0])*(4-x[0]*x[0])*(4-x[1]*x[1])*(4-x[1]*x[1])", degree=fe_order)
    v_init = Expression("0.55*exp(-(x[0]*x[0]+x[1]*x[1]))*(4-x[0]*x[0])*(4-x[0]*x[0])*(4-x[1]*x[1])*(4-x[1]*x[1])", degree=fe_order)

# Do build mesh and initil conditions

reset(nx, fe_order)
