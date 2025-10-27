import sys
import numpy as np

from pathlib import Path

math_dir = Path(__file__).parent.parent / "math_se"
sys.path.append(str(math_dir))

from lobatto import *
from base_func import *


def emm_lob(m, k):
    """
    element mass matrix 计算无量纲单元质量矩阵,即插值基函数的内积
    无量纲质量矩阵的计算公式为(int表示积分) 插值基函数积分 = int(Pij(ζ),-1,1)/(m^2*(m+1)^2*Lm(ζi)*Lm(ζj)),其中
    Pij(ζ) = (ζ^2-1)/((ζ-ζ_i)*(ζ-ζ_j))*Lo_m-1(ζ)^2

    params:
    m   :   插值多项式阶数
    k   :   Lobatto积分点数,正常应该是为m+1

    returns:
    质量矩阵,如果k<m,则返回-1,表示计算错误
    """
    # 计算k阶Lobatto多项式零点进行的数值积分
    if k >= m:
        # 先计算m阶Lobatto插值多项式零点
        zeta_z, _ = lobatto_roots(m)
        zp, w = lobatto_roots(k)
        psi_ij = lagrange_func(zeta_z, zp)
        psi_ij_w = psi_ij * w.reshape(1, -1)
        return np.dot(psi_ij, psi_ij_w.T)

    return -1


def edm_lob(m):
    """
    element diffusion matrix 计算无量纲单元扩散矩阵,即插值基函数微分的内积

    params:
    m   :   插值多项式阶数

    returns:
    A   :   扩散矩阵
    """
    zeta_z, w = lobatto_roots(m)

    zeta_z_arr = zeta_z.reshape(1, -1)
    sub_zeta = zeta_z_arr.T - zeta_z_arr + np.eye(m + 1)
    neg_sub_zeta = 1 / sub_zeta
    p = np.prod(sub_zeta, axis=1).reshape(1, -1)
    p1 = -p / p.T * neg_sub_zeta
    sum_zeta = np.sum(neg_sub_zeta, axis=1).reshape(1, -1)
    dij = p1 + (sum_zeta * np.eye(m + 1))

    A = np.dot(dij * w.reshape(1, -1), dij.T)
    return A
