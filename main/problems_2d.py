import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from lobatto import *
from base_func import *
from assemble_matrix import *
from forward import *
from node_discr import *
from mesh_gen_2d import *
from node_discr2d import *
from em_2d import *
from disp_result_2d import *


def lapl3_di():
    """
    狄利克雷边界条件下的拉普拉斯方程求解案例
    """
    ndiv = 3
    ne, ng, x, y, p_org, c, efl, gfl = trgl3_condiv_disk(ndiv)

    p = p_org.copy()
    defx = 0.6
    for i in range(ng):
        p[i, 0] = p[i, 0] * (1.0 - defx * p[i, 1] ** 2)

    gfl_new = np.zeros((ng, 2))
    for i in range(ng):
        if gfl[i] != 0:
            gfl_new[i, 0] = gfl[i]
            gfl_new[i, 1] = p[i, 0] * np.sin(0.5 * np.pi * p[i, 1])

    km = np.zeros((ng, ng))

    for l in range(ne):
        j = c[l, 0]
        x1 = p[j, 0]
        y1 = p[j, 1]

        j = c[l, 1]
        x2 = p[j, 0]
        y2 = p[j, 1]

        j = c[l, 2]
        x3 = p[j, 0]
        y3 = p[j, 1]

        edm_elm = edm_tnt_lob(x1, y1, x2, y2, x3, y3)
        for i in range(3):
            il = c[l, i]
            for j in range(3):
                jl = c[l, j]
                km[il, jl] = km[il, jl] + edm_elm[i, j]

    # 计算右边矩阵
    bm = np.zeros((ng, 1))

    k_ass, b_ass = dirichlet_assemble(km, bm, ng, gfl_new)

    fai_e = np.linalg.solve(k_ass, b_ass)
    plot3_2dfem(ne, ng, p_org, c, fai_e)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(p_org[:, 0], p_org[:, 1], np.zeros_like(p[:, 0]))
    ax.plot_trisurf(
        p_org[:, 0], p_org[:, 1], fai_e.flatten(), cmap="viridis", edgecolor="none"
    )


lapl3_di()
plt.show()
