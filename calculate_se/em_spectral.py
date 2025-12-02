import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from math_se.chebyshev import *
from math_se.lobatto import *

def linear_map_interval(ab,v):
    a,b = ab
    v = np.asarray(v)
    v0 = v[0]
    v1 = v[-1]
    mapped = a + (v - v0) * (b - a)/(v1 - v0)
    return mapped

def edm_qua(coords, order, basis = "GLL", k = None):
    """
    计算四边形单元扩散矩阵

    params:
    coords : 单元坐标
    order : int型, 单元阶数
    basis : 插值基函数类型,GLL表示Gauss-Lobatto-Legendre,GLC表示Gauss-Lobatto-Chebyshev节点

    return:
    edm : 六面体单元扩散矩阵
    """

    # 1. 生成一维积分节点和权重
    #TODO 写切比雪夫积分权重函数
    if basis == "GLL":
        xi,w = lobatto_roots(order)
    elif basis == "GLC":
        xi,w = lobatto_roots(order)
    else:
        xi,w = lobatto_roots(order)

    x,y = coords[0],coords[1]
    # 2. 求解雅可比矩阵
    diff_mat = legendre_diff_mat(xi)
    x_xi = np.dot(diff_mat,x)
    x_eta = np.dot(x,diff_mat.T)
    y_xi = np.dot(diff_mat,y)
    y_eta = np.dot(y,diff_mat.T)

    # 3. 计算参数张量矩阵
    detJ = x_xi * y_eta - x_eta * y_xi
    g11 = (y_eta / detJ) ** 2 + (-y_xi / detJ) ** 2
    g12 = (y_eta / detJ) * (-x_eta / detJ) + (-y_xi / detJ) * (x_xi / detJ)
    g22 = (-x_eta / detJ) ** 2 + (x_xi / detJ) ** 2

    w = w.reshape(1,-1)
    w_mat = w * w.T
    w_detJ_diag = np.diag((w_mat * detJ).flatten())
    w_g11 = np.diag(g11.flatten()) * w_detJ_diag
    w_g12 = np.diag(g12.flatten()) * w_detJ_diag
    w_g22 = np.diag(g22.flatten()) * w_detJ_diag

    # 4. 参考导数在所有节点上编织成所有插值节点基函数的矩阵
    d_xi = np.kron(diff_mat, np.eye(order + 1))
    d_eta = np.kron(np.eye(order + 1), diff_mat)
    # 5. 计算单元扩散矩阵
    d_mat = d_xi.T @ w_g11 @ d_xi + d_xi.T @ w_g12 @ d_eta + d_eta.T @ w_g12 @ d_xi + d_eta.T @ w_g22 @ d_eta

    return d_mat

def emm_qua(coords, order, basis = "GLL"):
    """
    计算单位质量矩阵
    """
    # 1. 生成一维积分节点和权重
    # TODO 写切比雪夫积分权重函数
    if basis == "GLL":
        xi, w = lobatto_roots(order)
    elif basis == "GLC":
        xi, w = lobatto_roots(order)
    else:
        xi, w = lobatto_roots(order)

    x, y = coords[0], coords[1]
    # 2. 求解雅可比矩阵
    diff_mat = legendre_diff_mat(xi)
    x_xi = np.dot(diff_mat, x)
    x_eta = np.dot(x, diff_mat.T)
    y_xi = np.dot(diff_mat, y)
    y_eta = np.dot(y, diff_mat.T)

    # 3. 计算参数张量矩阵
    detJ = x_xi * y_eta - x_eta * y_xi
    g11 = (y_eta / detJ) ** 2 + (-y_xi / detJ) ** 2
    g12 = (y_eta / detJ) * (-x_eta / detJ) + (-y_xi / detJ) * (x_xi / detJ)
    g22 = (-x_eta / detJ) ** 2 + (x_xi / detJ) ** 2

    w = w.reshape(1, -1)
    w_mat = w * w.T
    m_mat = np.diag((w_mat * detJ).flatten())

    return m_mat


def chev_diff1_mat(n):
    """
    切比雪夫配置点一阶微分矩阵

    params:
    n : 切比雪夫多项式次数,也是零点个数

    return:
    D1 : 一阶微分矩阵
    """
    x = chev_roots(n)
    delta = np.ones(n+1)
    delta[0] = 2
    delta[-1] = 2

    # Initialize diff matrix
    D1 = np.zeros((n,n))

    for i in range(n+1):
        for j in range(n+1-i):
            if i == 0 and j == 0:
                D1[i,j] = (2 * n**2 +1)/6
            elif i == n and j == n:
                D1[i,j] = -D1[0,0]
            elif i == j:
                if abs(1 - x[i]**2) < 1e-12:
                    raise ValueError("abs(1 - x[i]**2) = 0")
                else:
                    D1[i,j] = -x[i] / (2 * (1 - x[i]**2))
            else:
                D1[i,j] = (delta[i] * (-1)**(j + i)) / (delta[j] * (x[i] - x[j]))

    for i in range(1, n + 1):
        for j in range(n - i + 1, n + 1):
            D1[i,j] = -D1[n - i,n - j]

    return D1

def chev_diff2_mat(n):
    """
    切比雪夫配置点二阶微分矩阵
    params:
    n : 切比雪夫多项式次数,也是零点个数

    return:
    D2 : 二阶微分矩阵
    """
    D1 = chev_diff1_mat(n)
    D2 = np.linalg.matrix_power(D1, 2)
    return D2

def chev_diffn_mat(n, power):
    """
    切比雪夫配置点n阶微分矩阵
    params:
    n : 切比雪夫多项式次数,也是零点个数
    power : 求解阶数

    return:
    Dn : n阶微分矩阵
    """
    D1 = chev_diff1_mat(n)
    Dn = np.linalg.matrix_power(D1, power)
    return Dn

