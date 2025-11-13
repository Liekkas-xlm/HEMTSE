import sys
from abc import abstractmethod

import numpy as np

from pathlib import Path

math_dir = Path(__file__).parent.parent / "math_se"
sys.path.append(str(math_dir))

from lobatto import *
from base_func import *

def edm_tnt_lob(x1, y1, x2, y2, x3, y3):
    """
    Evaluation of the element diffusion matrix for a three-node triangle.

    Parameters:
    x1, y1 : float
        Coordinates of the first node.
    x2, y2 : float
        Coordinates of the second node.
    x3, y3 : float
        Coordinates of the third node.

    Returns:
    edm : 3x3 numpy array
        The element diffusion matrix.
    """
    d32x = x3 - x2
    d32y = y3 - y2
    d13x = x1 - x3
    d13y = y1 - y3
    d21x = x2 - x1
    d21y = y2 - y1

    A = 0.5 * (d13x * d21y - d13y * d21x)  # element area
    A4 = 4.0 * A

    edm = np.zeros((3, 3))
    edm[0, 0] = (d32x ** 2 + d32y ** 2) / A4
    edm[0, 1] = (d32x * d13x + d32y * d13y) / A4
    edm[0, 2] = (d32x * d21x + d32y * d21y) / A4

    edm[1, 0] = (d13x * d32x + d13y * d32y) / A4
    edm[1, 1] = (d13x ** 2 + d13y ** 2) / A4
    edm[1, 2] = (d13x * d21x + d13y * d21y) / A4

    edm[2, 0] = (d21x * d32x + d21y * d32y) / A4
    edm[2, 1] = (d21x * d13x + d21y * d13y) / A4
    edm[2, 2] = (d21x ** 2 + d21y ** 2) / A4

    return edm

def gdm_assemble(gdm, ne, ng, coordinates, c, material_coefficient):
    """
    组装整体扩散矩阵,考虑Neuman和dirichket边界条件

    params:
    ne : 划分单元总数
    ng : 节点总数
    coordinates : 坐标点矩阵
    c : 节点矩阵
    material : 材料参数,数组类型,索引代表材料区域

    return:
    gdm : 整体扩散矩阵
    """
    for l in range(ne):
        domain_index = c[l, -1]

        j = c[l, 0]
        x1 = coordinates[j, 0]
        y1 = coordinates[j, 1]

        j = c[l, 1]
        x2 = coordinates[j, 0]
        y2 = coordinates[j, 1]

        j = c[l, 2]
        x3 = coordinates[j, 0]
        y3 = coordinates[j, 1]

        edm_elm = edm_tnt_lob(x1, y1, x2, y2, x3, y3)
        # print(edm_elm)
        for i in range(3):
            il = c[l, i]
            for j in range(3):
                jl = c[l, j]
                # print(gdm[il, jl])
                # print(material_coefficient[domain_index])
                gdm[il, jl] += edm_elm[i, j] * material_coefficient[domain_index]
    #np.savetxt('../model/G.txt',gdm)

def emm_tnt_lob(x1, y1, x2, y2, x3, y3):
    """
    计算三角形单元的质量矩阵

    参数:
    x1, y1: 第一个顶点的坐标
    x2, y2: 第二个顶点的坐标
    x3, y3: 第三个顶点的坐标

    返回:
    3x3的质量矩阵
    """
    # Evaluation of the element mass matrix
    # from the coordinates of the vertices
    # of a three-node triangle
    # A: element area

    # 计算各边的差值
    d23x = x2 - x3
    d23y = y2 - y3
    d31x = x3 - x1
    d31y = y3 - y1
    d12x = x1 - x2
    d12y = y1 - y2

    # 计算三角形面积 (使用行列式公式)
    A = 0.5 * (d31x * d12y - d31y * d12x)

    # 计算质量矩阵因子
    fc = A / 12.0
    fs = 2.0 * fc

    # 构建3x3质量矩阵
    emm = np.zeros((3, 3))
    emm[0, 0] = fs  # (1,1)
    emm[0, 1] = fc  # (1,2)
    emm[0, 2] = fc  # (1,3)

    emm[1, 0] = fc  # (2,1)
    emm[1, 1] = fs  # (2,2)
    emm[1, 2] = fc  # (2,3)

    emm[2, 0] = fc  # (3,1)
    emm[2, 1] = fc  # (3,2)
    emm[2, 2] = fs  # (3,3)

    return emm

def gmm_assemble(gmm, ne, ng, coordinates, c, material_coefficient):
    """
    组装整体扩散矩阵,考虑Neuman和dirichket边界条件

    params:
    ne : 划分单元总数
    coordinates : 坐标点矩阵
    c : 节点矩阵

    return:
    gmm : 整体质量矩阵
    """
    for l in range(ne):
        domain_index = c[l, -1]

        j = c[l, 0]
        x1 = coordinates[j, 0]
        y1 = coordinates[j, 1]

        j = c[l, 1]
        x2 = coordinates[j, 0]
        y2 = coordinates[j, 1]

        j = c[l, 2]
        x3 = coordinates[j, 0]
        y3 = coordinates[j, 1]

        emm_elm = emm_tnt_lob(x1, y1, x2, y2, x3, y3)
        for i in range(3):
            il = c[l, i]
            for j in range(3):
                jl = c[l, j]
                gmm[il, jl] = gmm[il, jl] + emm_elm[i, j] * material_coefficient[domain_index]


def elm_tnt_lob(x1,y1,x2,y2,x3,y3):
    """
    计算插值基函数的积分,即载荷积矩阵
    return: 一阶插值基函数的积分
    """
    # TODO : 完善高阶插值基函数
    A = 0.5 * abs((x2-x1)*(y3-y1)-(x3-x1)*(y2-y1))
    return 1/3*A

def glm_elm(glm, ne, ng, coordinates, c, material_coefficient = None):
    """
    计算载荷积分
    params:

    """
    for l in range(ne):
        domain_index = c[l, -1]
        j = c[l, 0]
        x1 = coordinates[j, 0]
        y1 = coordinates[j, 1]

        j = c[l, 1]
        x2 = coordinates[j, 0]
        y2 = coordinates[j, 1]

        j = c[l, 2]
        x3 = coordinates[j, 0]
        y3 = coordinates[j, 1]
        elm_elm = elm_tnt_lob(x1, y1, x2, y2, x3, y3)
        for i in range(3):
            il = c[l, i]
            if material_coefficient is None:
                glm[il,0] += elm_elm
            else:
                glm[il, 0] += elm_elm * material_coefficient[domain_index]



def dirichlet_assemble(km, bm, ng, gfl_difx):
    """
    狄利克雷边界条件下进行矩阵装配

    params:
    km  :   左边矩阵
    bm  :   右边矩阵
    gfl :   整体结点标记矩阵,包含狄利克雷边界条件

    returns:
    k_new   :   新的右边矩阵
    b_new   :   新的左边矩阵
    """
    k_new = km.copy()
    b_new = bm.copy()
    for i in range(ng):
        if gfl_difx[i, 0] != 0:
            b_new[:, :] = b_new[:, :] - gfl_difx[i, 1] * km[:, i : i + 1]
            k_new[:, i : i + 1] = np.zeros_like(k_new[:, i : i + 1])
            k_new[i : i + 1, :] = np.zeros_like(k_new[i : i + 1, :])

    for i in range(ng):
        if gfl_difx[i, 0] != 0:
            b_new[i, 0] = gfl_difx[i, 1]
            k_new[i, i] = 1

    return k_new, b_new

def bmm_assemble(BM,  coordinates, bdr_marks, flux_coefficient):
    """
    边界线积分, 这里先处理常量热对流形式的积分

    params:
    BM :
    coordinates : 坐标参数
    bdr_marks : 边界节点编号
    flux_coefficient : 常量对流值
    material_coefficient : 由材料得到的物理参数
    """
    # TODO 补上线性热通量,修改积分公式
    # for bdr_element in bdr_marks:
    #     delta_s = np.sqrt((coordinates[bdr_element[0], 0] - coordinates[bdr_element[1], 0]) ** 2
    #                       + (coordinates[bdr_element[0], 1] - coordinates[bdr_element[1], 1]) ** 2)
    #     for i in range(2):
    #         bli_vector[i] = bli_vector[i] + delta_s * 0.5 * flux_coefficient[bdr_element[2]]
    for bdr_element in bdr_marks:
        delta_s = np.sqrt((coordinates[bdr_element[0], 0] - coordinates[bdr_element[1], 0]) ** 2
                          + (coordinates[bdr_element[0], 1] - coordinates[bdr_element[1], 1]) ** 2)
        BM[bdr_element[0], bdr_element[0]] += delta_s * flux_coefficient / 3
        BM[bdr_element[0], bdr_element[1]] += delta_s * flux_coefficient / 6
        BM[bdr_element[1], bdr_element[1]] += delta_s * flux_coefficient / 3
        BM[bdr_element[1], bdr_element[0]] += delta_s * flux_coefficient / 6

def bv_assemble(bli_vector, coordinates, bdr_marks, flux_coefficient):
    """
    边界线积分, 这里先处理常量热对流形式的积分

    params:
    BM :
    coordinates : 坐标参数
    bdr_marks : 边界节点编号
    flux_coefficient : 常量对流值
    material_coefficient : 由材料得到的物理参数
    """
    # TODO 补上线性热通量,修改积分公式
    for bdr_element in bdr_marks:
        delta_s = np.sqrt((coordinates[bdr_element[0], 0] - coordinates[bdr_element[1], 0]) ** 2
                          + (coordinates[bdr_element[0], 1] - coordinates[bdr_element[1], 1]) ** 2)

        bli_vector[bdr_element[0]] += delta_s * (
                    1 / 3 * flux_coefficient[bdr_element[0]] + 1 / 6 * flux_coefficient[bdr_element[1]])
        bli_vector[bdr_element[1]] += delta_s * (
                    1 / 3 * flux_coefficient[bdr_element[1]] + 1 / 6 * flux_coefficient[bdr_element[0]])


def dbr_add(dirc_bdr, km, bm):
    """
    添加Dirclet边界条件, 其中 aT/at
    km T = bm

    params:
    dirc_bdr :  狄利克雷边界

    """
    for bdr_element in dirc_bdr:
        for i in range(2):
            bm[:, :] -= bdr_element[-1] * km[:, bdr_element[i]:bdr_element[i] + 1]
            km[:, bdr_element[i]:bdr_element[i] + 1] = 0
            km[bdr_element[i]:bdr_element[i] + 1, :] = 0
    for bdr_element in dirc_bdr:
        for i in range(2):
            bm[bdr_element[i], 0] = bdr_element[-1]
            km[bdr_element[i], bdr_element[i]] = 1









