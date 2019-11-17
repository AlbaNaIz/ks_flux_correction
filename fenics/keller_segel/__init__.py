from fenics import *
import numpy as np
import matplotlib.pyplot as plt

# FEniCS: only warnings (default: 20, information of general interet)
set_log_level(30)

# Import classes stored in respective .py files
from keller_segel.abstract_scheme import KS_AbstractScheme
from keller_segel.default_scheme import KS_DefaultScheme
from keller_segel.matrix_default_scheme import KS_Matrix_DefaultScheme
from keller_segel.flux_correct import KS_FluxCorrect_DefaultScheme
