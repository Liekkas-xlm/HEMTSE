import sys
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from math_se.lobatto import *
from math_se.base_func import *
from mesh_se.mesh_gen_2d import *
from input_se.bdr_load import *
from calculate_se.em_2d import *

ne = 4
ng = 5
coordinates = np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1]],dtype=np.float64)
c = np.array([[1,2,3],[4,1,3],[5,1,4],[5,2,1]],dtype=int)-1

gdm = gdm_assemble(ne,ng,coordinates,c)

print(gdm)