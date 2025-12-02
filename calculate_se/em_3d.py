import sys
from abc import abstractmethod

import numpy as np

from pathlib import Path

math_dir = Path(__file__).parent.parent / "math_se"
sys.path.append(str(math_dir))

from lobatto import *
from base_func import *

def emm_t4(xy_points):
    """
    四面体单位扩散矩阵

    parmas:
    xy_points : xy_points是一个三行四列的矩阵,第一行是x坐标,第二行是y坐标,第三行是z坐标

    returns:
    四面体单位扩散矩阵
    """
    x21 = xy_points[0,1] - xy_points[0,0]
    x31 = xy_points[0,2] - xy_points[0,0]
    x41 = xy_points[0,3] - xy_points[0,0]
    x32 = xy_points[0,2] - xy_points[0,1]
    x42 = xy_points[0,3] - xy_points[0,1]

    y21 = xy_points[1,1] - xy_points[1,0]
    y31 = xy_points[1,2] - xy_points[1,0]
    y41 = xy_points[1,3] - xy_points[1,0]
    y32 = xy_points[1,2] - xy_points[1,1]
    y42 = xy_points[1,3] - xy_points[1,1]

    z21 = xy_points[2,1] - xy_points[2,0]
    z31 = xy_points[2,2] - xy_points[2,0]
    z41 = xy_points[2,3] - xy_points[2,0]
    z32 = xy_points[2,2] - xy_points[2,1]
    z42 = xy_points[2,3] - xy_points[2,1]

    # 雅可比矩阵
    jac = np.zeros((3,3))
    jac[0,0] = x21
    jac[0,1] = x31
    jac[0,2] = x41
    jac[1,0] = y21
    jac[1,1] = y31
    jac[1,2] = y41
    jac[2,0] = z21
    jac[2,1] = z31
    jac[2,2] = z41

    # # 计算体积乘以6
    # vol6 = np.linalg.det(jac) * 6
    #
    # 梯度解
    grad_fai = np.zeros(4)
    grad_fai[0] = np.asarray([[-y32 * z42 + z32 * y42], [x32 * z42 - z32 * x42], [-x32 * y42 + y32 * x42]]) / 6
    grad_fai[1] = np.asarray([[y31 * z41 - z31 * y41], [-x31 * z41 + z31 * x41], [x31 * y41 - y31 * x41]]) / 6
    grad_fai[2] = np.asarray([[-y21 * z41 + z21 * y41], [x21 * z41 - z21 * x41], [-x21 * y41 + y41 * x21]]) / 6
    grad_fai[3] = np.asarray([[y21 * z31 - z21 * y31], [-x21 * z31 + z21 * x31], [x21 * y31 - y21 * x31]]) / 6

    edm = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            edm[i,j] = np.dot(grad_fai[i], grad_fai[j].T)

    return edm