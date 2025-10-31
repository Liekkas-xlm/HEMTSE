import numpy as np
from scipy.spatial import Delaunay

class TriMesh():
    def __init__(self, mesh_filename,order = 1):
        self.mesh_filename = mesh_filename
        self.order = order

    def _extract_coordinates(self):
        # 提取网格顶点坐标
        try:
            with open(self.mesh_filename,'r') as f:
                lines = f.readlines()
        except:
            raise FileNotFoundError(f"Cannot Open Mesh File {self.mesh_filename}")

        vertex_lines = []
        reading_vertices = False

        for line in lines:
            line = line.strip()
            if not reading_vertices:
                if '# Mesh vertex coordinates' in line:
                    reading_vertices = True
                continue

            if line == '':
                break

            vertex_lines.append(line)

        # 解析为浮点矩阵
        vertex_matrix = np.zeros((len(vertex_lines),2),dtype=np.float64)
        for i,line in enumerate(vertex_lines):
            data = np.fromstring(line,sep=' ',dtype=np.float64)
            if data.size == 2:
                vertex_matrix[i,:] = data
            else:
                raise Exception(f"The format of {i+1} line is abnormal : {line}")

        return vertex_matrix

    def _extract_vtxs(self):
        # 提取几何模型顶点索引
        try:
            with open(self.mesh_filename, 'r') as f:
                lines = f.readlines()
        except:
            raise FileNotFoundError(f"Cannot Open Mesh File {self.mesh_filename}")

        found_type0 = False
        elements_count = -1
        reading_elements = False
        reading_indices = False
        elements = []
        indices = []

        # 遍历每一行
        for line_count,line in enumerate(lines,start=1):

            if elements_count == len(indices) and elements_count == len(elements):
                # 完成读取
                break

            line = line.strip()
            if not line:    # 跳过空行
                continue

            # 查找 “# Type #0”
            if not found_type0 and "# Type #0" in line:
                found_type0 = True
                continue

            if found_type0:
                # 提取元素数量
                if "# number of elements" in line:
                    tokens = line.split()
                    try:
                        elements_count = int(tokens[0])
                    except:
                        raise ValueError(f"Can not extract number of elements from {line}")

                # 读取元素状态
                if "# Elements" in line:
                    reading_elements = True
                    reading_indices = True
                    continue

                # 读取索引状态
                if "# Geometric entity indices" in line:
                    reading_elements = False
                    reading_indices = True
                    continue

                # 读取元素数据
                if reading_elements and not line.startswith("#"):
                    try:
                        elements.append(int(line))
                    except ValueError:
                        pass
                    continue

                # 读取元素索引
                if reading_indices and not line.startswith("#"):
                    try:
                        indices.append(int(line))
                    except ValueError:
                        pass
                    continue

        if not elements:
            raise Exception("Cannot find elements in mesh file.")
        if not indices:
            raise Exception("Cannot find indices in mesh file.")

        elements = np.array(elements)

        reordered = np.zeros_like(elements)

        if np.max(indices) > elements_count or np.min(indices) < 0:
            raise IndexError(f"Invalid indices: {np.min(indices)} - {np.max(indices)}")

        for i in range(elements_count):
            reordered[indices[i]] = elements[i]

        elements = reordered
        return elements

    def _extract_edgs(self):
        # 提取边界元素
        try:
            with open(self.mesh_filename, 'r') as f:
                lines = f.readlines()
        except:
            raise FileNotFoundError(f"Cannot Open Mesh File {self.mesh_filename}")

        elements_data = []
        indices = []
        reading_elements = False
        reading_indices = False
        found_type1 = False
        elements_count = -1

        for line_count,line in enumerate(lines,start=1):

            if elements_count == len(indices) and elements_count == len(elements_data):
                break

            line = line.strip()
            if not line: continue

            if not found_type1 and "# Type #1" in line:
                found_type1 = True
                continue

            if found_type1:
                # 提取元素数量
                if "# number of elements" in line:
                    tokens = line.split()
                    try:
                        elements_count = int(tokens[0])
                    except:
                        raise ValueError(f"Can not extract number of elements from {line}")
                    continue

                # 读取元素状态
                if "# Elements" in line:
                    reading_elements = True
                    reading_indices = False
                    continue

                # 读取索引状态
                if "# number of geometric entity indices" in line:
                    reading_elements = False
                    reading_indices = True
                    continue

                # 读取元素数据
                if reading_elements and not line.startswith("#"):
                    elements_data.append(line)
                    continue

                # 读取所属的边
                if reading_indices and not line.startswith("#"):
                    try:
                        indices.append(int(line))
                    except ValueError:
                        pass
                    continue

        # 解析为整数矩阵
        edgs_matrix = np.zeros((elements_count,3),dtype=int)
        for i,line in enumerate(elements_data):
            data = np.fromstring(line,sep=' ')
            if data.size == 2:
                edgs_matrix[i,0:2] = data
                edgs_matrix[i,2] = indices[i]
            else:
                raise Exception(f"The format of {i+1} line is abnormal : {line}")

        return edgs_matrix

    def _extract_tris(self):
        # 提取三角形元素
        try:
            with open(self.mesh_filename, 'r') as f:
                lines = f.readlines()
        except:
            raise FileNotFoundError(f"Cannot Open Mesh File {self.mesh_filename}")

        elements_data = []
        indices = []
        reading_elements = False
        reading_indices = False
        found_type2 = False
        elements_count = -1

        for line_count, line in enumerate(lines, start=1):

            if elements_count == len(indices) and elements_count == len(elements_data):
                break

            line = line.strip()
            if not line: continue

            if not found_type2 and "# Type #2" in line:
                found_type2 = True
                continue

            if found_type2:
                # 提取元素数量
                if "# number of elements" in line:
                    tokens = line.split()
                    try:
                        elements_count = int(tokens[0])
                    except:
                        raise ValueError(f"Can not extract number of elements from {line}")
                    continue

                # 读取元素状态
                if "# Elements" in line:
                    reading_elements = True
                    reading_indices = False
                    continue

                # 读取索引状态
                if "# number of geometric entity indices" in line:
                    reading_elements = False
                    reading_indices = True
                    continue

                # 读取元素数据
                if reading_elements and not line.startswith("#"):
                    elements_data.append(line)
                    continue

                # 读取所属的边
                if reading_indices and not line.startswith("#"):
                    try:
                        indices.append(int(line))
                    except ValueError:
                        pass
                    continue


        # 解析为整数矩阵
        tris_matrix = np.zeros((elements_count, 4), dtype=int)
        for i, line in enumerate(elements_data):
            data = np.fromstring(line, sep=' ')
            if data.size == 3:
                tris_matrix[i, 0:3] = data
                tris_matrix[i, 3] = indices[i]
            else:
                raise Exception(f"The format of {i + 1} line is abnormal : {line}")

        return tris_matrix

    def tri_import_comsol_mesh(self):
        """
        从COMSOL导入网格

        returns:
        ne  :   划分单元数
        ng  :   全局节点的数量
        x,y :   划分点坐标
        c   :   联系矩阵c(i,j)(i=1,...,Ne; j=1,2,3),表示第i个单元的第j个节点所对应的整体节点编号
                取值范围为1, ..., Ng
        efl :   假设第i个单元有m个插值结点,单元节点标记矩阵, efl(i,j)(i=1,...,Ne j=1,...,m)
                规定为j个节点为边界节点,则设为边界条数,若不为边界点则为0
        glf :   整体节点标记矩阵glf(i)(i=1,...,Ng),使glf(i)值与efl对应节点相同
        """
        coordinates = self._extract_coordinates()
        vtxs = self._extract_vtxs()
        edgs = self._extract_edgs()
        tris = self._extract_tris()

        ne = tris.shape[0]
        ng = coordinates.shape[0]

        c = tris[:,:3]

        return ne, ng, coordinates, c

def trgl3_condiv_disk(ndiv):
    """
    从四个硬编码单元出发,通过连续细分各级单元组实现对单位直径圆盘三角剖分

    params:
    ndiv    :   连续细分等级

    returns:
    ne  :   划分单元数
    ng  :   全局节点的数量
    x,y :   划分点坐标
    p   :   整体数组p(i,j)表示整体节点的x,y坐标,其中i = 1,...,Ng; j=1,2
    c   :   联系矩阵c(i,j)(i=1,...,Ne; j=1,2,3),表示第i个单元的第j个节点所对应的整体节点编号
            取值范围为1, ..., Ng
    efl :   假设第i个单元有m个插值结点,单元节点标记矩阵, efl(i,j)(i=1,...,Ne j=1,...,m)
            规定为j个节点为边界节点,则设为边界条数,若不为边界点则为0
    glf :   整体节点标记矩阵glf(i)(i=1,...,Ng),使glf(i)值与efl对应节点相同
    """
    ne = 4

    # 定义坐标和元素标识符
    x = np.zeros((4 ** (ndiv + 1), 6))
    y = np.zeros_like(x)
    efl = np.zeros_like(x)

    x[0:4, 0:3] = np.array(
        [
            [0.0, 1.0, 0.0],  # 第一个单元的x坐标
            [0.0, 0.0, -1.0],  # 第二个单元的x坐标
            [0.0, -1.0, 0.0],  # 第三个元素的x坐标
            [0.0, 0.0, 1.0],  # 第四个元素的x坐标
        ]
    )

    y[0:4, 0:3] = np.array(
        [
            [0.0, 0.0, 1.0],  # 第一个元素的y坐标
            [0.0, 1.0, 0.0],  # 第二个元素的y坐标
            [0.0, 0.0, -1.0],  # 第三个元素的y坐标
            [0.0, -1.0, 0.0],  # 第四个元素的y坐标
        ]
    )

    efl[0:4, 0:3] = np.array(
        [
            [0, 1, 1],  # 第一个元素的标识符
            [0, 1, 1],  # 第二个元素的标识符
            [0, 1, 1],  # 第三个元素的标识符
            [0, 1, 1],  # 第四个元素的标识符
        ]
    )
    xn = np.zeros((4 ** (ndiv + 1), 3))
    yn = np.zeros_like(xn)
    efln = np.zeros_like(efl)

    if ndiv > 0:
        for i in range(ndiv):
            # 计算每次细化过程中产生的新元素,每次迭代将生成四个元素。
            nm = 0
            for j in range(ne):
                # 计算坐标中点
                x[j, 3] = 0.5 * (x[j, 0] + x[j, 1])
                y[j, 3] = 0.5 * (y[j, 0] + y[j, 1])

                x[j, 4] = 0.5 * (x[j, 1] + x[j, 2])
                y[j, 4] = 0.5 * (y[j, 1] + y[j, 2])

                x[j, 5] = 0.5 * (x[j, 2] + x[j, 0])
                y[j, 5] = 0.5 * (y[j, 2] + y[j, 0])

                # 新生成的中点根据前两个点是否为边界点而标记
                efl[j, 3] = 0
                efl[j, 4] = 0
                efl[j, 5] = 0
                if efl[j, 0] == 1 and efl[j, 1] == 1:
                    efl[j, 3] = 1
                if efl[j, 1] == 1 and efl[j, 2] == 1:
                    efl[j, 4] = 1
                if efl[j, 2] == 1 and efl[j, 0] == 1:
                    efl[j, 5] = 1

                # 定义新的子单元
                xn[nm, 0] = x[j, 0]
                yn[nm, 0] = y[j, 0]
                efln[nm, 0] = efl[j, 0]
                xn[nm, 1] = x[j, 3]
                yn[nm, 1] = y[j, 3]
                efln[nm, 1] = efl[j, 3]
                xn[nm, 2] = x[j, 5]
                yn[nm, 2] = y[j, 5]
                efln[nm, 2] = efl[j, 5]
                nm = nm + 1  # 第二个子元素

                xn[nm, 0] = x[j, 3]
                yn[nm, 0] = y[j, 3]
                efln[nm, 0] = efl[j, 3]
                xn[nm, 1] = x[j, 1]
                yn[nm, 1] = y[j, 1]
                efln[nm, 1] = efl[j, 1]
                xn[nm, 2] = x[j, 4]
                yn[nm, 2] = y[j, 4]
                efln[nm, 2] = efl[j, 4]
                nm = nm + 1  # 第三个子元素

                xn[nm, 0] = x[j, 5]
                yn[nm, 0] = y[j, 5]
                efln[nm, 0] = efl[j, 5]
                xn[nm, 1] = x[j, 4]
                yn[nm, 1] = y[j, 4]
                efln[nm, 1] = efl[j, 4]
                xn[nm, 2] = x[j, 2]
                yn[nm, 2] = y[j, 2]
                efln[nm, 2] = efl[j, 2]
                nm = nm + 1  # 第四个子元素

                xn[nm, 0] = x[j, 3]
                yn[nm, 0] = y[j, 3]
                efln[nm, 0] = efl[j, 3]
                xn[nm, 1] = x[j, 4]
                yn[nm, 1] = y[j, 4]
                efln[nm, 1] = efl[j, 4]
                xn[nm, 2] = x[j, 5]
                yn[nm, 2] = y[j, 5]
                efln[nm, 2] = efl[j, 5]
                nm = nm + 1  # 下一个子元素

            ne = 4 * ne

            for k in range(ne):  # 重新定义新节点
                for l in range(3):  # 将其重新放入矩阵
                    x[k, l] = xn[k, l]
                    y[k, l] = yn[k, l]
                    efl[k, l] = efln[k, l]

                    # 处理边界结点,使其与原点距离相同
                    if efl[k, l] == 1:
                        rad = np.sqrt(x[k, l] ** 2 + y[k, l] ** 2)
                        x[k, l] = x[k, l] / rad
                        y[k, l] = y[k, l] / rad

    # 定义全局结点
    p = np.zeros((3, 2))
    glf = np.zeros(3)
    c = np.zeros((ne, 3), dtype=int)

    # 第一个元素的坐标和全局节点编号
    p[0, 0] = x[0, 0]  # MATLAB中的1,1对应Python中的0,0
    p[0, 1] = y[0, 0]
    glf[0] = efl[0, 0]

    p[1, 0] = x[0, 1]  # MATLAB中的1,2对应Python中的0,1
    p[1, 1] = y[0, 1]
    glf[1] = efl[0, 1]

    p[2, 0] = x[0, 2]  # MATLAB中的1,3对应Python中的0,2
    p[2, 1] = y[0, 2]
    glf[2] = efl[0, 2]

    # 第一个元素的全局节点编号
    c[0, 0] = 0  # 第一个节点是全局节点1
    c[0, 1] = 1  # 第二个节点是全局节点2
    c[0, 2] = 2  # 第三个节点是全局节点3

    ng = 3
    eps = 0.000001

    for i in range(1, ne):
        for j in range(3):
            iflag = 0  # 初始化
            for k in range(ng):
                if abs(x[i, j] - p[k, 0]) < eps:
                    if abs(y[i, j] - p[k, 1]) < eps:
                        iflag = 1
                        c[i, j] = k

            if iflag == 0:
                p = np.append(p, np.array([[x[i, j], y[i, j]]]), axis=0)
                glf = np.append(glf, efl[i, j])
                c[i, j] = ng
                ng = ng + 1

    return ne, ng, x, y, p, c, efl, glf



def trgl3_dvt_sqr():
    """
    基于正方向网格进行随机扰动产生的整体结点集

    returns:
    ne  :   划分单元数
    ng  :   全局节点的数量
    x,y :   划分点坐标
    p   :   整体数组p(i,j)表示整体节点的x,y坐标,其中i = 1,...,Ng; j=1,2
    c   :   联系矩阵c(i,j)(i=1,...,Ne; j=1,2,3),表示第i个单元的第j个节点所对应的整体节点编号
            取值范围为1, ..., Ng
    efl :   假设第i个单元有m个插值结点,单元节点标记矩阵, efl(i,j)(i=1,...,Ne j=1,...,m)
            规定为j个节点为边界节点,则设为边界条数,若不为边界点则为0
    glf :   整体节点标记矩阵glf(i)(i=1,...,Ng),使glf(i)值与efl对应节点相同
    """
    # 初始化参数
    X1 = -1.0
    X2 = 1.0
    Y1 = -1.0
    Y2 = 1.0
    N = 8
    M = 8

    # 准备
    Dx = (X2 - X1) / N
    Dy = (Y2 - Y1) / M

    # 初始化点的坐标和边界标志数组
    p = np.zeros((M + 1, N + 1, 2))
    glf = np.zeros((M + 1, N + 1), dtype=int)

    # 安排网格上的点，设置边界标志，并计算节点数(ng)
    ng = 0
    for j in range(1, N + 1):
        for i in range(1, M + 1):
            ng += 1
            p[i, j, 0] = X1 + (i - 1.0) * Dx
            p[i, j, 1] = Y1 + (j - 1.0) * Dy
            glf[i, j] = 0
            if i == 1 or i == M + 1 or j == 1 or j == N + 1:
                glf[i, j] = 1

    # 随机化内部节点
    Ic = N + 2
    for j in range(2, M):
        for i in range(2, N):
            Ic += 1
            p[i, j, 0] = p[i, j, 0] + (np.random.rand() - 1.0) * 0.5 * Dx
            p[i, j, 1] = p[i, j, 1] + (np.random.rand() - 1.0) * 0.4 * Dy

    # Delaunay triangulation
    xdel = p[:, :, 0].flatten()
    ydel = p[:, :, 1].flatten()
    c = Delaunay(np.column_stack((xdel, ydel)))

    # 提取元素数量
    sc = c.shape
    ne = sc[0]

    # 设置元素-节点边界标志
    efl = np.zeros((ne, 3))
    for i in range(ne):
        efl[i, 0] = glf[c.simplices[i, 0]]
        efl[i, 1] = glf[c.simplices[i, 1]]
        efl[i, 2] = glf[c.simplices[i, 2]]

    return ne, ng, p, c, efl, glf


def tri_tnt_sqr(ndiv):
    """
    正方形矩阵三角剖分
    """
    pass
