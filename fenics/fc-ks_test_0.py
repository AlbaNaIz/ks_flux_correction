#
# Flux-Correction Keller-Segel Test 0
# ===================================
#
# In this test we use FEniCS for explicitly building and solving the
# K-S FE equation system,that is, we explicitly buy FE matrices and
# FE RHS vectors. This work is done by the class KS_MatrixDefaultScheme
# in the module keller_segel.
#

from keller_segel import KS_MatrixDefaultScheme as KellerSegelScheme
from numpy.testing import assert_approx_equal

if( __name__ == "__main__" ):
    #
    # Read all the data from a parameters file
    #
    import data.chertok_kurganov as dat
    dat.reset(_nx = 60)
    #
    # Define test as a concrete Keller-Segel scheme
    #
    ks_test = KellerSegelScheme( dat.mesh, dat.fe_order, dat.dt )
    #
    # Define C-K initial conditions
    #
    ks_test.set_u( dat.u_init )
    ks_test.set_v( dat.v_init )
    #
    # Run time iterations
    #
    result = ks_test.run( nt_steps=100, break_when_negative_u=True )

    #
    # Make sure that results match what happended in our previous tests!
    #

    # 1) Positivity is broken at iteration 10
    assert result['iter']==10

    # 2) max(u) is over 3.3*10^3
    max_u = max(result['u'].vector())
    assert_approx_equal(max_u, 3.328079e+03)
