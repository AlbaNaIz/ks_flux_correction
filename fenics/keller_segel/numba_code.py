import numpy as np

# Try to import numba (high performance python compiler!!). If numba is not
# found, define empty wrappers for functions njit and prange
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

# @njit(parallel=True)
@njit
def compute_D_values(I, C, kVals, dVals):
    # 1. To parallize: initialize an output with the desired length
    out = np.empty(len(kVals))
    # 2. Use prange (and avoid "race conditions"!)
    for row in prange(len(I)-1):
        k0, k1 = I[row], I[row+1] # Pointers to begin and end of current row
        k_diag = k0 + index( C[k0:k1], row )
        # # Compute max(-k_{ij}, -k_{ji}, 0 )
        out[k0:k1] = np.maximum( np.zeros(k1-k0),
                                 np.maximum(-kVals[k0:k1], -dVals[k0:k1]) )
        row_sum = np.sum(out[k0:k1]) - out[k_diag]
        out[k_diag] = -row_sum
    return out
