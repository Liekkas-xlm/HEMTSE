import sys
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from math_se.lobatto import *
from math_se.base_func import *
from mesh_se.mesh_gen_2d import *

filename = "../model/2DHEMT.mphtxt"

mesh = TriMesh(filename)

ne, ng, x, y, c = mesh.tri_import_comsol_mesh()

print("x = ",x.dtype)
print("y = ",y.dtype)
print("c = ",c.dtype)