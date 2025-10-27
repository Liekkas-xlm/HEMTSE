import numpy as np
import matplotlib.pyplot as plt
import quadpy

from scipy.special import legendre
from scipy.optimize import newton
from numpy.polynomial import Legendre


def lagrange_func(roots, zp):
    """
    m阶拉格朗日多项式计算 : fai(ζ) = (ζ-ζ_1)...(ζ-ζ_i-1)(ζ-ζ_i+1)...(ζ-ζ_m+1)/[(ζ_i-ζ_1)...(ζ_i-ζ_i-1)(ζ_i-ζ_i+1)...(ζ_i-ζ_m+1)]

    params:
    roots   :   不包含zeta_i的插值基点
    zp      :   计算点ζ,行向量

    return:
    矩阵,每一个zp在第i个插值基函数下的值,行数与zeta_i长度相同,列数与zp长度相同
    """
    roots_new = np.array([np.setdiff1d(roots, roots[i]) for i in range(len(roots))])

    zeta_i_arr = np.array(roots).reshape(1, -1)
    zp_arr = np.array(zp).reshape(1, -1)

    p1 = np.prod(zeta_i_arr.T - roots_new, axis=1).reshape(-1, 1)

    # roots摊成一列,广播机制使每一个zp与roots_arr相减
    sub_matrix = zp_arr - roots_new.reshape(-1, 1)
    # 变成每len(zp)列对应第i个插值基函数下zp得到的值
    re_sub_matrix = sub_matrix.reshape(-1, len(zp) * len(roots))
    # 行内元素相乘,形状变成行数与zeta_i长度相同,列数与zp长度相同
    p2 = np.prod(re_sub_matrix, axis=0).reshape(len(roots), len(zp))

    return p2 / p1


def diff_lagrange():
    """
    拉格朗日插值基函数的微分
    """
    pass
