import sys
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

math_dir = Path(__file__).parent.parent / "math_se"
sys.path.append(str(math_dir))

from lobatto import *
from base_func import *


def fai_forward(x, xe, xg, fai_e, pm):
    """
    根据求解得到的节点值,利用插值函数求出要求的点的值

    params:
    x   :   求解位置坐标,向量
    xe  :   节点划分坐标
    xg  :   全部插值节点
    fai_e   :   插值节点值
    pm  :   Lobatto多项式阶数

    returns:
    fai :   坐标对应的函数值
    """

    fai = np.zeros_like(x, dtype=fai_e.dtype)
    x_sort = np.sort(x)

    start_index = 0
    end_index = 0
    # print(fai_e)
    for i in range(len(xe) - 1):
        index = np.searchsorted(x_sort, xe[i + 1], side="left")

        if index != 0:
            m = pm[i]
            xe_start_index = np.where(abs(xg - xe[i]) < 1e-10)[0][0]
            xe_end_index = np.where(abs(xg - xe[i + 1]) < 1e-10)[0][0]
            end_index = index + 1
            #   结点映射到[-1,1]之间
            zeta = 2 * (x_sort[start_index:end_index] - xe[i]) / (xe[i + 1] - xe[i]) - 1
            zeta_i, _ = lobatto_roots(m)
            inter = lagrange_func(zeta_i, zeta)
            # print("\nfai_e = ", fai_e[xe_start_index : xe_end_index + 1])
            # print("inter = ", inter)
            temp = inter * fai_e[xe_start_index : xe_end_index + 1].reshape(-1, 1)
            # print("temp = ", temp)
            fai[start_index:end_index] = np.sum(temp, axis=0)

            start_index = end_index

    return fai
