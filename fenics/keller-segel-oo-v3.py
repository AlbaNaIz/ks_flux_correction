from keller_segel import KS_DefaultScheme as KellerSegelScheme

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
    # Make sure that, for this test, positivity is broken at iteration 10
    # At least, that is what happended in our previous tests!
    #
    assert result['iter']==10
