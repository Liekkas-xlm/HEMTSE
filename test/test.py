import sys
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from lobatto import *
from base_func import *
from assemble_matrix import *
from mesh_gen_2d import *
from node_discr2d import *

m = 19
N = int((m + 1) * (m + 2) / 2)
nt = int((m - 1) / 3) + 1

p = gen_standard_tri_avag(m)
edge_m = m

node_counter = 0
fig, ax = plt.subplots()
for i in range(nt):
    x = np.zeros(edge_m * 3)
    y = np.zeros(edge_m * 3)

    for j in range(3):
        for k in range(edge_m):
            x[j * edge_m + k] = p[node_counter, 0]
            y[j * edge_m + k] = p[node_counter, 1]
            node_counter = node_counter + 1
    ax.plot(np.append(x, x[0]), np.append(y, y[0]), "-o")
    edge_m = edge_m - 3
if N - node_counter != 0:
    ax.plot(p[-1, 0], p[-1, 1], "o")
ax.set_aspect("equal")  # 关键代码：设置等比例

fig, ax = plt.subplots()
ax.plot(p[:, 0], p[:, 1], "o")
ax.set_aspect("equal")  # 关键代码：设置等比例

plt.show()
