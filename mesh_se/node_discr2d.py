import numpy as np


def cen3(ne, ng, c):
    """
    共享同一个结点的单元编号,先通过对全部全局节点后对所有单元上的节点进行循环

    params:
    ne  :   单元总数
    ng  :   结点总数
    c   :   联系矩阵

    returns:
    cen(i,j)    :   cen(i,1)(i=1,...,Ng),是第i个全局结点的单元总数,
                    (j=2,...,cen(i,1)+1)是共享第i个整体结点的第j个单元的单元编号
    """
    cen = np.zeros((ng, ne))
    for i in range(ng):
        cen[i, 0] = 0
        Icount = 1

        for j in range(ne):
            for k in range(3):
                if c[j, k] == i:
                    # 共享结点单元数
                    cen[i, 0] = cen[i, 0] + 1
                    cen[i, Icount] = j
                    Icount = Icount + 1

    return cen


def ces3(ne, ng, c):
    """
    生成TNT单元体系中单元与单元边对应关系的联系矩阵ces的库函数

    params:
    ne  :   单元总数
    ng  :   结点总数
    c   :   联系矩阵

    returns:
    ces[i,j]    :   共享第i个单元的第j条边的单元编号,其中j=1,2,3与i=1,...,Ne,
                    单元边1室连接结点1和2的连线,单元边2是连接节点2和3的连线,单元边3是连接结点3和1的连线,
                    若ces[i,j]=0,则第i个单元的第j条边无相邻单元
    """
    ces = np.zeros((ne, 3))

    for i in range(ne):
        for j in range(3):
            for k in range(ne):
                if k != i:
                    for l in range(3):
                        if c[k, l] == c[i, j] and c[k, l + 1] == c[i, j + 1]:
                            ces[i, j] = k
                        if c[k, l] == c[i, j + 1] and c[k, l + 1] == c[i, j]:
                            ces[i, j] = k
    return ces


def gen_standard_tri_avag(m):
    """
    由任意一维网格生成的标准三角形单元上对应于完备的m阶多项展开式的插值结点配置方法

    parmas:
    m   :   多项式阶数m
    discr_method    :   主对角线划分方法, "avag"表示平均划分

    returns:
    p   :   内嵌式三角形坐标矩阵

    """
    vi = np.arange(m + 1) / m
    N = (m + 1) * (m + 2) / 2
    p = np.zeros((int(N), 3))

    # 嵌套三角形数量
    nt = int((m - 1) / 3) + 1

    edge_m = m
    node_counter = 0
    for i in range(nt):
        for j in range(edge_m):
            p[node_counter, 0] = vi[i + j]
            p[node_counter, 1] = vi[i]
            node_counter = node_counter + 1
        for j in range(edge_m):
            p[node_counter, 0] = vi[m - 2 * i - j]
            p[node_counter, 1] = vi[i + j]
            node_counter = node_counter + 1
        for j in range(edge_m):
            p[node_counter, 0] = vi[i]
            p[node_counter, 1] = vi[m - 2 * i - j]
            node_counter = node_counter + 1
        edge_m = edge_m - 3

    if N - node_counter != 0:
        p[-1, 0] = vi[nt]
        p[-1, 1] = vi[nt]
    return p
