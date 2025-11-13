import sys
import matplotlib.pyplot as plt
import numpy as np
import json
import math

from pathlib import Path

from scipy.sparse import bsr_matrix

sys.path.append(str(Path(__file__).resolve().parents[1]))

from input_se.bdr_load import *
from input_se.material_load import *
from mesh_se.mesh_gen_2d import *
from calculate_se.em_2d import *
from disp_se.disp_result_2d import *
from disp_se.output_data import *

class Solver:
    def __init__(self, bdr_file, mesh_file, material_file, t_start, t_stop, t_step, mesh_unit = "um"):
        self.heat_transfer = SolidHeatTransfer(bdr_file)
        self.mesh = TriMesh(mesh_file,mesh_unit)
        self.ne, self.ng, self.coordinates, self.edgs, self.c= self.mesh.tri_import_comsol_mesh()
        self.materials_list = read_material_json(material_file)
        self.t_start = t_start
        self.t_stop = t_stop
        self.t_step = t_step
        self.T = np.ones((self.ng,int((t_stop - t_start) / t_step)+1))

    def solve(self):
        temp_init = self.heat_transfer.initial_value["T"]
        self.T[:, 0] = self.T[:, 0] * temp_init

        k_coefficient = extract_material_coefficient(self.materials_list, "k_iso")
        KM = np.zeros((self.ng, self.ng))
        # 组装扩散矩阵 
        gdm_assemble(KM, self.ne, self.ng, self.coordinates, self.c, k_coefficient)

        Q = self.heat_transfer.heat_source["source"]["Q0_value"]
        BV = np.zeros((self.ng, 1))
        flux_bdr_index = np.asarray(self.heat_transfer.heat_flux["boundary"])-1

        bdr_marks = self.edgs[np.isin(self.edgs[:, 2], flux_bdr_index)]
        bdr_domain_mark = find_edge_domains(bdr_marks, self.c)
        bdr_marks = np.column_stack((bdr_marks, bdr_domain_mark))
        glm_elm(BV, self.ne, self.ng, self.coordinates, self.c)
        BV *= (-Q)
        #TODO 自动使用合适的k系数
        flux_coefficient = self.heat_transfer.heat_flux["flux"]["h"] * self.heat_transfer.heat_flux["flux"]["Text"] * k_coefficient[2]
        bv_assemble(BV,self.coordinates, bdr_marks, flux_coefficient * np.ones((self.ng,1)))

        rho_coefficient = extract_material_coefficient(self.materials_list, "rho")
        cp_coefficient = extract_material_coefficient(self.materials_list, "Cp")
        MM = np.zeros_like(KM)
        gmm_assemble(MM, self.ne, self.ng, self.coordinates, self.c, rho_coefficient * cp_coefficient)

        dirc_bdr = extract_dirc_bdr(self.edgs,self.heat_transfer.temperature["boundary"],self.heat_transfer.temperature["T0"])

        BM = np.zeros((self.ng, self.ng))
        bmm_assemble(BM, self.coordinates, bdr_marks, self.heat_transfer.heat_flux["flux"]["h"] * k_coefficient[2])

        for i in range(self.T.shape[1]-1):
            LM = MM/self.t_step + KM - BM
            RV = np.dot(MM/self.t_step, self.T[:,i]).reshape(-1,1) - BV
            print("RV", RV)
            dbr_add(dirc_bdr, LM, RV)
            self.T[:, i+1] = np.dot(np.linalg.inv(LM), RV).ravel()

bdr_file = "../model/solid_heat_transfer.json"
mesh_file = "../model/2DHEMT.mphtxt"
material_file = "../model/material.json"
t_start = 0
t_stop = 1
t_step = 1

temp_solver = Solver(bdr_file, mesh_file, material_file, t_start, t_stop, t_step)
temp_solver.solve()
#
# disp_colorbar(temp_solver.coordinates, temp_solver.T[:,1])
# plt.show()

data = np.column_stack((temp_solver.coordinates*1e6, temp_solver.T[:,1]))
format_thermal_data_from_matrix(data,'../model/temp_data.txt')
print("finish")