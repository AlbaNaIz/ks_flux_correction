#
# Flux-Correction Keller-Segel Test 1
# ===================================
#
# First step in K-S flux correction: Consistent FE FE matrices and FE
# RHS vectors are buit and then they are modified, following usual
# steps in FC theory.
#
# In this test, we compute:
#   * The low order solution.
#   * The fluxes
#
# We build a coarse mesh, in order to make possible
# the inspection of the intermediate matrices.
#

from keller_segel import KS_FluxCorrect_DefaultScheme
from keller_segel import plot, plt
from numpy.testing import assert_approx_equal

if( __name__ == "__main__" ):
    #
    # Read all the data from a parameters file
    #
    import data.chertok_kurganov as dat
    dat.reset(_nx = 5, _dt=1.e-4)
    # plot(dat.mesh)
    # plt.show()

    #
    # Define test as a concrete Keller-Segel scheme
    #
    ks_test = KS_FluxCorrect_DefaultScheme( dat.mesh, dat.fe_order, dat.dt )
    #
    # Define C-K initial conditions
    #
    ks_test.set_u( dat.u_init )
    ks_test.set_v( dat.v_init )
    #
    # Run time iterations
    #
    ks_test.set_parameter("save_matrices")
    result = ks_test.run( nt_steps=2,
                          break_when_negative_u=True, plot_u=False )

    #
    # Make sure that results match what happended in our previous tests!
    #

    # 1) Positivity at this iteration
    assert min(result['u'].vector()>0)
    assert min(result['v'].vector()>0)

    # 2) max(u) and max(v) match the results obtained in previous tests
    max_u = max(result['u'].vector())
    # assert_approx_equal(max_u, 135.53886011)
    max_v = max(result['v'].vector())
    # assert_approx_equal(max_v, 183.930644124)
    print("max_u=", max_u, "max_v=", max_v)
