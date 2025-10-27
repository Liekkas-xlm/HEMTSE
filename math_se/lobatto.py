import numpy as np
import matplotlib.pyplot as plt
import quadpy

from scipy.special import legendre
from scipy.optimize import newton
from numpy.polynomial import Legendre


def lobatto_roots(n):
    """
    返回lobatto多项式零点(包含-1和1节点)以及Gauss-Lobatto积分权重

    params:
    n   :   Lobatto零点个数,例如当n=1时候有一个零点,返回零点值为1,以此类推

    returns:
    roots   :   Lobatto零点,包含-1和+1
    weights :   Gauss-Lobatto积分零点
    """

    # Lobatto点与 (n+1)阶Legendre多项式的导数零点相同(不包含-1和1这两个点)
    roots = quadpy.c1.gauss_lobatto(n + 1).points
    weights = quadpy.c1.gauss_lobatto(n + 1).weights
    return roots, weights


def int_lob(m, func):
    """
    用Gauss-Lobatto方法实现任意被积函数在[-1,1]的数值积分计算,函数是一个(2m-1)阶的多项式采用Lobatto积分得到的值是准确的

    params:
    m   :   使用m阶Lobatto多项式零点进行数值积分
    func:   积分函数

    returns:
    积分值
    """
    r, w = lobatto_roots(m)

    intergral = 0.0
    for i in range(m + 1):
        intergral = intergral + func(r[i]) * w[i]

    return intergral
