import sys
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from math_se.lobatto import *
from math_se.base_func import *
from mesh_se.mesh_gen_2d import *
from input_se.bdr_load import *
from input_se.material_load import *
from calculate_se.em_2d import *
from disp_se.disp_result_2d import *
from calculate_se.em_spectral import *

coords = [np.array([[0,0],[1,1]]),np.array([[0,1],[0,1]])]
km = emm_qua(coords, 1, basis = "GLL")
print(km)