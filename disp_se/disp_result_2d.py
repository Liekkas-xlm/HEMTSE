import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def plot3_2dfem(ne, ng, p, c, f):
    """
    函数fx值可视化
    """
    # compute the maximum and minimum of the function f
    fmax = -100.0  # initialize
    fmin = 100.0  # initialize

    for i in range(ng):
        if f[i] > fmax:
            fmax = f[i]
        if f[i] < fmin:
            fmin = f[i]

    range_val = 1.2 * (fmax - fmin)
    shift = fmin

    # shift the color index in the range(0,1)
    # and plot patches

    fig, ax = plt.subplots()
    patches = []
    colors = []

    for l in range(ne):
        # Get vertices coordinates and color values
        vertices = []
        color_vals = []

        for idx in [0, 1, 2, 0]:  # Triangle + repeat first vertex to close
            j = c[l, idx]  # Convert to 0-based indexing
            x = p[j, 0]
            y = p[j, 1]
            color_val = (f[j] - shift) / range_val
            vertices.append([x, y])
            color_vals.append(color_val)

        patches.append(vertices)
        # Use average color for the patch
        colors.append(np.mean(color_vals[:3]))  # Use only first 3 vertices for average

    # Create polygon collection
    coll = PolyCollection(
        patches, array=np.array(colors), cmap=cm.viridis, edgecolors="none"
    )
    ax.add_collection(coll)
    ax.autoscale_view()
    # 设置X轴和Y轴比例相同（核心修改）
    ax.set_aspect("equal", adjustable="box")  # 确保X轴和Y轴单位长度相同[1,2](@ref)
    # Add colorbar
    fig.colorbar(coll, ax=ax)
