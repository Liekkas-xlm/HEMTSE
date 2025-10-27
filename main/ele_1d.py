import sys
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

math_dir = Path(__file__).parent.parent / "math_se"
sys.path.append(str(math_dir))
assemble_dir = Path(__file__).parent.parent / "calculate_se"
sys.path.append(str(assemble_dir))
solver_dir = Path(__file__).parent.parent / "solver"
sys.path.append(str(solver_dir))
node_dir = Path(__file__).parent.parent / "node"
sys.path.append(str(node_dir))

from lobatto import *
from base_func import *
from assemble_matrix import *
from forward import *
from node_discr import *


def analytical_solution(k0, m, x, theta, epsilon, miu):
    """金属衬底介质片平面波反射的解析解

    Args:
        x:要求的场点坐标
        theta:入射角
        m:将介质片划分的薄层数
        k0:自由空间的k
        epsilon:相对介电常常数
        miu:相对磁导率

    Returns:
        计算得到的反射功率
    """
    k_xm = k0 * np.sqrt(miu * epsilon - (np.sin(theta.reshape(-1, 1)) ** 2))
    k_xm = k_xm.T
    # 计算电场的反射功率
    R_me = -1 * np.ones(len(theta))
    R_mm = np.ones(len(theta))
    for i in range(m - 1):
        # 电场反射系数计算
        lambda_me = (miu[i] * k_xm[i + 1] - miu[i + 1] * k_xm[i]) / (
            miu[i] * k_xm[i + 1] + miu[i + 1] * k_xm[i]
        )
        R_me = (
            (lambda_me + R_me * np.exp(-2.0j * k_xm[i] * x[i + 1]))
            / (1 + lambda_me * R_me * np.exp(-2.0j * k_xm[i] * x[i + 1]))
        ) * np.exp(2.0j * k_xm[i + 1] * x[i + 1])

        # 磁场反射系数计算
        lambda_mm = (epsilon[i] * k_xm[i + 1] - epsilon[i + 1] * k_xm[i]) / (
            epsilon[i] * k_xm[i + 1] + epsilon[i + 1] * k_xm[i]
        )
        R_mm = (
            (lambda_mm + R_mm * np.exp(-2.0j * k_xm[i] * x[i + 1]))
            / (1 + lambda_mm * R_mm * np.exp(-2.0j * k_xm[i] * x[i + 1]))
        ) * np.exp(2.0j * k_xm[i + 1] * x[i + 1])
    return R_me, R_mm


def run():
    # 在此例子中,平面波的波速是光速,设入射波的波长为1mm
    lamda_wave = 1
    # 真空中波数k_0为2π/λ
    k0 = 2 * np.pi / lamda_wave
    # 介质片厚度为5λ
    thickness = 5 * lamda_wave
    # 划分薄层数
    m = 100

    x = np.arange(thickness / m, thickness + thickness / m, thickness / m)  # 场点坐标

    max_theta = np.pi / 2
    theta = np.arange(
        0, max_theta + max_theta / 90, max_theta / 90
    )  # 入射角度,90个角度
    # theta = np.array([0])

    epsilon = 4 + (2 - 0.1j) * ((1 - np.arange(1 / m, 1, 1 / m)) ** 2)
    miu = np.ones_like(epsilon) * (2 - 0.1j)
    epsilon = np.append(epsilon, 1)  # 相对介电常数
    miu = np.append(miu, 1)  # 相对磁导率

    # 解析解求解
    analytical_Rme, analytical_Rmm = analytical_solution(k0, m, x, theta, epsilon, miu)
    # print("解析解的反射系数: ", abs(analytical_Rme))

    m = 50
    pm = np.ones(m, dtype=int) * 3
    xe, xen, xien, xg, c, ng = discr_lob(0, thickness, m, 0, pm)
    max_theta = np.pi / 2
    theta = np.arange(
        0, max_theta + max_theta / 90, max_theta / 90
    )  # 入射角度,90个角度

    epsilon = 4 + (2 - 0.1j) * ((1 - np.arange(1 / m, 1 + 1 / m, 1 / m)) ** 2)
    miu = np.ones_like(epsilon) * (2 - 0.1j)

    E0 = 1  # 初始电场值,随便设一个
    H0 = 1

    alpha_e = 1 / miu
    f_e = np.zeros(ng)  # 源函数

    E_l = np.ones((theta.shape[0], m + 1)) * 1j
    numerical_Rme = np.ones(theta.shape[0]) * 1j

    for i in range(len(theta)):
        # print("求解第", i, "个角度")
        beta_e = -(k0**2) * (epsilon - 1 / miu * (np.sin(theta[i]) ** 2))
        gama_e = k0 * np.cos(theta[i]) * 1j
        q_e = (
            np.cos(theta[i])
            * 2j
            * k0
            * E0
            * np.exp(1j * k0 * np.cos(theta[i]) * thickness)
        )

        gdm, bm = sds_lob_sys(
            m, pm, ng, xe, alpha_e, beta_e, f_e, 0, gama_e, q_e, is_real=False
        )
        fai_e = np.linalg.solve(gdm, bm)

        E_l[i] = fai_forward(xe, xe, xg, fai_e, pm)

        # print(E_l[i])
        numerical_Rme[i] = E_l[i][-1] - E0 * np.exp(
            1j * k0 * thickness * np.cos(theta[i])
        )
        print("电场数值解的反射系数: ", abs(numerical_Rme[i]))

    plt.figure(figsize=(12, 5))
    # plt.subplot(1, 2, 1)
    plt.plot(
        theta / max_theta * 90,
        abs(analytical_Rme),
        label="Analytical Solution",
        color="blue",
        linestyle="-",
    )
    plt.plot(
        theta / max_theta * 90,
        abs(numerical_Rme),
        label="Numerical Solution",
        color="red",
        linestyle="--",
    )
    plt.legend(loc="upper right")
    plt.title("Electric field reflection coefficient")
    plt.xlim(xmin=0, xmax=90)
    plt.yticks(np.arange(0, 1.01, 0.1))  # 设置 y 轴刻度
    plt.ylim(ymin=0, ymax=1)


if __name__ == "__main__":
    run()
    plt.show()
