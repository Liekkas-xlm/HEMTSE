import sys
import matplotlib.pyplot as plt
import numpy as np
import json

from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from input_se.bdr_load import *
from mesh_se.mesh_gen_2d import *
from calculate_se.em_2d import *



def solve():
    # 导入边界条件
    bdr_file = "../input_se/solid_heat_transfer.json"
    heat_transfer = SolidHeatTransfer(bdr_file)

    # 导入网格
    mesh_file = "../model/2DHEMT.mphtxt"
    mesh = TriMesh(mesh_file)
    ne, ng, coordinates, c = mesh.tri_import_comsol_mesh()

    pass

