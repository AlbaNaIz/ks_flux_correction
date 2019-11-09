#
# Flux-Correction Keller-Segel Test 1
# ===================================
#
# First step in K-S flux correction: Consistent FE FE matrices and FE
# RHS vectors are buit and then they are modified, following usual
# steps in FC theory.
#
# In this test, we build a coarse mesh, in order to make possible
# the inspection of the intermediate matrices.
#

from keller_segel import KS_FC_DefaultScheme
from keller_segel import plot, plt
from numpy.testing import assert_approx_equal

if( __name__ == "__main__" ):
    #
    # Read all the data from a parameters file
    #
    import data.chertok_kurganov as dat
    dat.reset(_nx = 5)
    # plot(dat.mesh)
    # plt.show()

    #
    # Define test as a concrete Keller-Segel scheme
    #
    ks_test = KS_FC_DefaultScheme( dat.mesh, dat.fe_order, dat.dt )
    #
    # Define C-K initial conditions
    #
    ks_test.set_u( dat.u_init )
    ks_test.set_v( dat.v_init )
    #
    # Run time iterations
    #
    ks_test.save_all_matrices(True) # Save all FE matrices to respective files
    print(ks_test.save_matrices)
    result = ks_test.run( nt_steps=1,
                          break_when_negative_u=True, plot_u=False )

    #
    # Make sure that results match what happended in our previous tests!
    #

    # # 1) Positivity is broken at iteration 10
    # assert result['iter']==10

    # 2) max(u) is over 3.32*10^3
    max_u = max(result['u'].vector())
    # assert_approx_equal(max_u, 3.328079e+03)
