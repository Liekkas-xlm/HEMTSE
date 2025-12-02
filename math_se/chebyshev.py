import numpy as np

def chev_roots(n):
    """
    返回chebyshev多项式零点(包含-1和1节点)以及Gauss-Chebyshev-Lobatto积分权重

    params:
    n   :   Chebyshev零点个数,例如当n=1时候有一个零点,返回零点值为1,以此类推

    returns:
    roots   :   Chebyshev零点,包含-1和+1
    weights :   Gauss-Chebyshev-Lobatto积分零点
    """
    j = np.arange(0,n + 1)
    roots = np.cos(np.pi * j / n)

    return roots

def chev_int_weights(n):
    """
    积分权重
    """
    pass