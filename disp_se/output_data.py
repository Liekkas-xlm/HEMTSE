import numpy as np


def format_thermal_data_from_matrix(data_matrix: np.ndarray, output_file: str = None) -> str:
    """
    格式化热数据numpy矩阵：将零值简化为"0"，并移除小数点后多余的尾随零

    参数:
        data_matrix: 输入的numpy数组/矩阵，形状为 (n, 3)，包含X, Y, T三列数据
        output_file: 可选参数，如果指定路径则将格式化结果写入文件

    返回:
        格式化后的字符串

    示例:
        >>> data = np.array([[0.0, 0.16499999999999998, 392.92653985442945],
        ...                  [0.00833685476253557, 0.15754974452724302, 392.79059086834417],
        ...                  [0.0, 0.15, 392.318260917888]])
        >>> print(format_thermal_data_from_matrix(data))
        0                         0.16499999999999998       392.92653985442945
        0.00833685476253557       0.15754974452724302       392.79059086834417
        0                         0.15                      392.318260917888
    """

    def format_number(value: float) -> str:
        """格式化单个数值"""
        if np.isclose(value, 0.0, atol=1e-15):
            return "0"
        return f"{value:.17g}"

    # 验证输入矩阵形状
    if data_matrix.ndim != 2 or data_matrix.shape[1] != 3:
        raise ValueError(f"期望形状为(n, 3)的二维数组，但得到形状为{data_matrix.shape}")

    # 格式化每一行数据
    formatted_lines = []
    for row in data_matrix:
        x_str = format_number(row[0])
        y_str = format_number(row[1])
        t_str = format_number(row[2])
        formatted_lines.append(f"{x_str:<25}{y_str:<25}{t_str}")

    # 组合成完整字符串
    result = "\n".join(formatted_lines)

    # 如果指定了输出文件，则写入
    if output_file is not None:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result)

    return result