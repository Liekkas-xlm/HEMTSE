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


def sds_lob_sys(ne, n_p, ng, xe, alpha, beta, fs, p, gama, q, is_real=True):
    """
    装配计算矩阵

    params:
    ne      :   节点单元数
    ng      :   总点数
    n_p     :   Lobatto插值多项式阶数
    xe      :   单元节点
    alpha   :   方程已知参数
    beta    :   方程已知参数
    fs      :   源函数值,行向量
    p       :   狄利克雷边界条件
    gama    :   诺曼边界条件参数
    q       :   诺曼边界条件参数

    returns:
    km :   左边矩阵
    b   :   右边矩阵

    """
    if is_real:
        km = np.zeros((ng, ng))
        bm = np.zeros((ng, 1))
    else:
        km = np.zeros((ng, ng), dtype=complex)
        bm = np.zeros((ng, 1), dtype=complex)
    fs_arr = fs.reshape(-1, 1)

    row_counter = 0  #  行计数
    col_counter = 0  #  列计数
    for i in range(ne):
        m = n_p[i]

        dm = edm_lob(m)  # 无量纲扩散矩阵
        mm = emm_lob(m, m + 1)  # 无量纲质量矩阵

        row_start = row_counter
        row_end = row_counter + m + 1
        col_start = col_counter
        col_end = col_counter + m + 1

        he = xe[i + 1] - xe[i]

        Dm = dm * 2 / he  # 回到坐标变换之前的矩阵值
        Mm = mm * he / 2

        km[row_start:row_end, col_start:col_end] = (
            km[row_start:row_end, col_start:col_end] + alpha[i] * Dm + beta[i] * Mm
        )

        bm[row_start:row_end, :] = (
            bm[row_start:row_end, :]
            + np.sum(Mm, axis=1).reshape(-1, 1) * fs_arr[row_start:row_end, :]
        )

        row_counter = row_counter + m
        col_counter = col_counter + m

    # 修正右边矩阵
    bm = bm - p * km[:, 0:1]
    bm[0, 0] = p
    bm[-1, -1] = bm[-1, -1] + q

    #   修正左边矩阵第一行和第一列
    if is_real:
        first_vector = np.zeros((1, ng))
    else:
        first_vector = np.zeros((1, ng), dtype=complex)
    first_vector[0][0] = 1

    km[0:1, :] = first_vector
    km[:, 0:1] = first_vector.reshape(-1, 1)

    #   修正左边矩阵最后一个元素
    km[-1, -1] = km[-1, -1] + gama

    return km, bm
