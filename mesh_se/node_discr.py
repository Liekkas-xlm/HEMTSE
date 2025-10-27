import sys
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from lobatto import *


def ele_line_avag(x1, x2, n, ratio=1):
    """
    将x轴上的一个区间划分为N个单元的均匀一维渐变网络

    params  :
    x1,x2   :   [x1,x2]为求解区间
    n       :   划分单元数

    returns :
    xe      :   被划分出来的节点值,第一个是x1,最后一个是x2
    """
    step = (x2 - x1) / n
    xe = [x1 + i * step for i in range(n + 1)]
    return np.array(xe)


def ele_line_ratio1(x1, x2, n, ratio):
    """
    将x轴上的一个区间划分为N个单元的一维渐变网络,从而使单元长度以几何级数从左到右递增或递减


    params  :
    x1,x2   :   [x1,x2]为求解区间
    n       :   划分单元数
    ratio   :   比率,表示最后一个单元长度△xN比上第一个单元的长度△x1

    returns :
    xe      :   被划分出来的节点值,第一个是x1,最后一个是x2
    """
    xe = np.zeros(n + 1)
    xe[0] = x1

    if n == 1:
        xe[-1] = x2
        return xe
    alpha = 1.0
    factor = 1.0 / n

    if ratio != 1:
        texp = 1 / (n - 1)
        alpha = ratio**texp
        factor = (1.0 - alpha) / (1.0 - alpha**n)

    deltax = (x2 - x1) * factor

    for i in range(n):
        xe[i + 1] = xe[i] + deltax
        deltax = deltax * alpha

    return xe


def ele_line_ratio2(x1, x2, n, ratio):
    """
    将x轴上的一个区间划分为N(偶数)个单元的一维渐变网络,从而使单元长度从左端点到区间中点都以一个指定比值按几何级数进行递增或递减

    params  :
    x1,x2   :   [x1,x2]为求解区间
    n       :   划分单元数
    ratio   :   比率,表示中点处相连的两个单元的中任一单元长度比上第一个单元的长度

    returns :
    xe      :   被划分出来的节点值,第一个是x1,最后一个是x2
    """
    if n % 2 != 0:
        raise ValueError("n should be even.")

    xe = np.zeros(n + 1)
    xe[0] = x1
    xe[-1] = x2

    half = int(n / 2)
    xe[half] = (x2 - x1) / 2

    alpha = 1.0
    factor = 1.0 / n

    if ratio != 1:
        texp = 1 / (half - 1)
        alpha = ratio**texp
        factor = (1.0 - alpha) / 2 / (1.0 - alpha**half)

    deltax = (x2 - x1) * factor

    for i in range(half):
        xe[i + 1] = xe[i] + deltax
        xe[n - i - 1] = xe[n - i] - deltax
        deltax = deltax * alpha

    return xe


def ele_line_ratio3(x1, x2, n, ratio):
    """
    将x轴上的一个区间划分为N(奇数)个单元的一维渐变网络,从而使单元长度从左端点到区间中点都以一个指定比值按几何级数进行递增或递减

    params  :
    x1,x2   :   [x1,x2]为求解区间
    n       :   划分单元数
    ratio   :   比率,表示区间中点单元长度比上第一个单元的长度

    returns :
    xe      :   被划分出来的节点值,第一个是x1,最后一个是x2
    """
    if n % 2 == 0:
        raise ValueError("n should be odd.")

    xe = np.zeros(n + 1)
    xe[0] = x1
    xe[-1] = x2

    half = int((n - 1) / 2)

    alpha = 1.0
    factor = 1.0 / n

    if ratio != 1:
        texp = 1 / half
        alpha = ratio**texp
        factor = (1.0 - alpha) / ((1.0 - alpha ** (half + 1)) + (1.0 - alpha**half))

    deltax = (x2 - x1) * factor

    for i in range(half):
        xe[i + 1] = xe[i] + deltax
        xe[n - 1 - i] = xe[n - i] - deltax
        deltax = deltax * alpha

    return xe


def discr_lob(x1, x2, ne, ratio, n_p, ele_func="ele_line_avag"):
    """
    离散解域[x1,x2]成为Ne个Lobatto单元,生成np阶Lobatto单元插值节点

    params  :
    x1,x2   :   单元起始节点
    ne      :   划分单元数
    ration  :
    n_p      :   Lobatto单元阶数

    returns:
    xe      :   单元节点
    xen     :   xi个单元Lobatto插值节点, 列数是单元数, 行数是插值节点数, 每一个单元映射到[-1,1]之间
    xien    :   xi个Lobatto单元插值节点, 列数是单元数, 行数是插值节点数
    xg      :   唯一全局节点,即将所有插值节点列在一行
    c       :   连接性矩阵,行表示为第i个划分单元,列表示为在整个求解区间内的第j个节点
    ng      :   全局节点的数量
    """
    # 划分单元,返回单元节点
    xe = eval(ele_func)(x1, x2, ne, ratio)
    xen = np.zeros((ne, max(n_p) + 1))
    xien = np.zeros((ne, max(n_p) + 1))

    # 定义连接矩阵
    Ic = 1
    c = np.zeros((ne, max(n_p) + 1))
    xg = np.zeros(np.sum(n_p) + 1)

    for i in range(ne):
        xi, _ = lobatto_roots(n_p[i])

        Ic = Ic - 1
        for j in range(n_p[i] + 1):
            # 存储第xi个Lobatto插值单元节点
            xien[i][j] = xi[j]

            # 划分单元内实际插值节点与Lobatto插值节点一一对应
            xen[i][j] = (xi[j] + 1.0) * (xe[i + 1] - xe[i]) / 2 + xe[i]

            # 连接矩阵
            c[i][j] = Ic

            xg[Ic] = xen[i][j]
            Ic = Ic + 1

    ng = Ic

    return xe, xen, xien, xg, c, ng


# TODO 采用第二类切比雪夫节点集的单元插值点
def discr_che(x1, x2, ne, ratio, n_p, ele_func="ele_line_ratio1"):
    """
    离散解域[x1,x2]成为Ne个Chebyshev单元,生成np阶Chebyshev单元插值节点

    params  :
    x1,x2   :   单元起始节点
    ne      :   划分单元数
    ration  :
    n_p      :   Chebyshev单元阶数

    returns:
    xe      :   单元节点
    xen     :   xi个单元Chebyshev插值节点, 列数是单元数, 行数是插值节点数, 每一个单元映射到[-1,1]之间
    xien    :   xi个Chebyshev单元插值节点, 列数是单元数, 行数是插值节点数
    xg      :   唯一全局节点,即将所有插值节点列在一行
    c       :   连接性矩阵,行表示为第i个划分单元,列表示为在整个求解区间内的第j个节点
    ng      :   全局节点的数量
    """
    pass
