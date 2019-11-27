import numpy as np

# Import numba (high performance python compiler!!).
#
# If not found, define empty wrappers for functions 'njit' and 'prange'

import importlib
try:
    numba_loader = importlib.import_module("numba")
    from numba import njit, prange
except ImportError:
    print("Numba module (high performance python compiler) not found")
    #1. Define wrapper for prange
    def prange(arg): return range(arg)
    # 2. Define wrapper for the njit decorator,
    # https://stackoverflow.com/questions/653368/how-to-create-a-python-decorator-that-can-be-used-either-with-or-without-paramet
    from functools import wraps
    def doublewrap(f):
        @wraps(f)
        def new_dec(*args, **kwargs):
            if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
                # actual decorated function
                return f(args[0])
            else:
                # decorator arguments
                return lambda realf: f(realf, *args, **kwargs)
        return new_dec
    @doublewrap
    def njit(f, parallel=2):
        '''Just return f(), with no parallelism'''
        @wraps(f)
        def wrap(*args, **kwargs): return f(*args,**kwargs)
        return wrap


#------------------------------------------------------------------------------
#--- Numba high performance code used in flux correction schemes---------------
#------------------------------------------------------------------------------

@njit
def index(array, item):
    # Find item in array, return its index
    for idx, val in np.ndenumerate(array):
        if val == item:
            break
    return idx[0]


@njit(parallel=True)
def compute_D_values(I, C, kVals, dVals):
    # 1. To parallize: initialize an output with the desired length
    result = np.empty(len(kVals))
    # 2. Use prange (and avoid "race conditions"!)
    nrows = len(I)-1
    for i in prange(nrows):
        # a) Get pointers to begin and end of current row
        i0, i1 = I[i], I[i+1]

        # b) Compute valuos for current row: max(-k_{ij}, -k_{ji}, 0 )
        result[i0:i1] = np.maximum( np.zeros(i1-i0),
                                 np.maximum(-kVals[i0:i1], -dVals[i0:i1]) )

        # b) Update diagonal value
        k_diag = i0 + index( C[i0:i1], i )
        result[k_diag] = 0
        row_sum = np.sum(result[i0:i1])
        result[k_diag] = -row_sum
    return result


# @njit(parallel=True)
def update_F_values(I, C, F_vals, U):
    """Let F_ij=0 if F_ij*(u_j-u_i) > 0"""

    # We will return an array with the same size than F_vals
    tmp = np.empty(2)
    result = np.empty( len(F_vals) )

    # Access to rows and modify respective elements<
    nrows = len(I)-1
    for i in prange(nrows):
        # a) Get pointers to begin and end of nz elements in row i
        i0, i1 = I[i], I[i+1]

        # b) Compute u[j] - u[i] for all columns j in row i
        jColumns = C[i0:i1]
        U_ji = U[jColumns] - U[i]

        # c) Let F values = 0 if sign(F)==sign(u[j]-u[i])
        result[i0:i1] = np.where(
            np.sign( F_vals[i0:i1] ) == np.sign( U_ji ),
            0,
            F_vals[i0:i1]
        )
    return result
