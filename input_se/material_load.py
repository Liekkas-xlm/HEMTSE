import json

import numpy as np


def read_material_json(filename):
    with open(filename) as json_file:
        data = json.load(json_file)

    material_list = []
    for material_name, props in data.items():
        entry = {"material": material_name}
        entry.update(props)
        material_list.append(entry)

    return material_list

def extract_material_coefficient(materials, key):
    """"
    从材料字典列表中提取所需要的材料参数,按照domain排序
    """
    sorted_materials = sorted(materials, key=lambda x: x["domain"])
    value_list = [mat[key] for mat in sorted_materials]
    return np.asarray(value_list)