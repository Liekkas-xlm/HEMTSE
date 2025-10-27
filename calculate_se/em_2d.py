import sys
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
    edm[0, 0] = (d32x**2 + d32y**2) / A4
    edm[0, 1] = (d32x * d13x + d32y * d13y) / A4
    edm[0, 2] = (d32x * d21x + d32y * d21y) / A4

    edm[1, 0] = (d13x * d32x + d13y * d32y) / A4
    edm[1, 1] = (d13x**2 + d13y**2) / A4
    edm[1, 2] = (d13x * d21x + d13y * d21y) / A4

    edm[2, 0] = (d21x * d32x + d21y * d32y) / A4
    edm[2, 1] = (d21x * d13x + d21y * d13y) / A4
    edm[2, 2] = (d21x**2 + d21y**2) / A4

    return edm


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
