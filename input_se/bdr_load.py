import json
from typing import Dict, List, Any, Optional


class SolidHeatTransfer:
    """
    固体热传递模拟配置类
    """

    def __init__(self, json_file_path: str = None, data: Dict = None):
        """
        初始化SolidHeatTransfer类

        Args:
            json_file_path: JSON文件路径
            data: 直接传入的字典数据
        """
        if json_file_path:
            self.load_from_file(json_file_path)
        elif data:
            self._load_data(data)
        else:
            self._initialize_defaults()

    def _initialize_defaults(self):
        """初始化默认值"""
        self.solid = {
            "thermal conductivity": "From Material",
            "density": "From Material",
            "constant pressure heat capacity": "From Material"
        }
        self.initial_value = {"T": 300}
        self.thermal_insulation = {"boundary": []}
        self.heat_source = {"domain": []}
        self.temperature = {"boundary": [], "TG": 300}
        self.heat_flux = {
            "material type": "solid",
            "flux": {
                "flux type": "convective heat flux",
                "heat transfer coefficient": "user-define",
                "h": 10,
                "Text": 300
            }
        }

    def _load_data(self, data: Dict):
        """从字典数据加载配置"""
        self.solid = data.get("solid", {})
        self.initial_value = data.get("initial value", {})
        self.thermal_insulation = data.get("thermal insulation", {})
        self.heat_source = data.get("heat source", {})
        self.temperature = data.get("temperature", {})
        self.heat_flux = data.get("heat flux", {})

    def load_from_file(self, file_path: str):
        """
        从JSON文件加载配置

        Args:
            file_path: JSON文件路径
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                self._load_data(data)
        except FileNotFoundError:
            raise FileNotFoundError(f"文件 {file_path} 未找到")
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON文件格式错误: {e}")
        except Exception as e:
            raise RuntimeError(f"读取文件时发生错误: {e}")

    def save_to_file(self, file_path: str):
        """
        保存配置到JSON文件

        Args:
            file_path: 保存的文件路径
        """
        data = self.to_dict()
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, indent=4, ensure_ascii=False)
        except Exception as e:
            raise RuntimeError(f"保存文件时发生错误: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式

        Returns:
            字典格式的配置数据
        """
        return {
            "solid": self.solid,
            "initial value": self.initial_value,
            "thermal insulation": self.thermal_insulation,
            "heat source": self.heat_source,
            "temperature": self.temperature,
            "heat flux": self.heat_flux
        }

    # 属性访问方法
    @property
    def thermal_conductivity(self) -> str:
        """获取热导率设置"""
        return self.solid.get("thermal conductivity", "From Material")

    @thermal_conductivity.setter
    def thermal_conductivity(self, value: str):
        """设置热导率"""
        self.solid["thermal conductivity"] = value

    @property
    def density(self) -> str:
        """获取密度设置"""
        return self.solid.get("density", "From Material")

    @density.setter
    def density(self, value: str):
        """设置密度"""
        self.solid["density"] = value

    @property
    def heat_capacity(self) -> str:
        """获取热容量设置"""
        return self.solid.get("constant pressure heat capacity", "From Material")

    @heat_capacity.setter
    def heat_capacity(self, value: str):
        """设置热容量"""
        self.solid["constant pressure heat capacity"] = value

    @property
    def initial_temperature(self) -> float:
        """获取初始温度"""
        return self.initial_value.get("T", 300.0)

    @initial_temperature.setter
    def initial_temperature(self, value: float):
        """设置初始温度"""
        self.initial_value["T"] = value

    @property
    def insulation_boundaries(self) -> List[int]:
        """获取隔热边界列表"""
        return self.thermal_insulation.get("boundary", [])

    @insulation_boundaries.setter
    def insulation_boundaries(self, boundaries: List[int]):
        """设置隔热边界列表"""
        self.thermal_insulation["boundary"] = boundaries

    @property
    def heat_source_domains(self) -> List[int]:
        """获取热源域列表"""
        return self.heat_source.get("domain", [])

    @heat_source_domains.setter
    def heat_source_domains(self, domains: List[int]):
        """设置热源域列表"""
        self.heat_source["domain"] = domains

    @property
    def temperature_boundaries(self) -> List[int]:
        """获取温度边界列表"""
        return self.temperature.get("boundary", [])

    @temperature_boundaries.setter
    def temperature_boundaries(self, boundaries: List[int]):
        """设置温度边界列表"""
        self.temperature["boundary"] = boundaries

    @property
    def temperature_value(self) -> float:
        """获取温度边界值"""
        return self.temperature.get("TG", 300.0)

    @temperature_value.setter
    def temperature_value(self, value: float):
        """设置温度边界值"""
        self.temperature["TG"] = value

    @property
    def heat_transfer_coefficient(self) -> float:
        """获取传热系数"""
        return self.heat_flux.get("flux", {}).get("h", 10.0)

    @heat_transfer_coefficient.setter
    def heat_transfer_coefficient(self, value: float):
        """设置传热系数"""
        if "flux" not in self.heat_flux:
            self.heat_flux["flux"] = {}
        self.heat_flux["flux"]["h"] = value

    @property
    def external_temperature(self) -> float:
        """获取外部温度"""
        return self.heat_flux.get("flux", {}).get("Text", 300.0)

    @external_temperature.setter
    def external_temperature(self, value: float):
        """设置外部温度"""
        if "flux" not in self.heat_flux:
            self.heat_flux["flux"] = {}
        self.heat_flux["flux"]["Text"] = value

    def validate(self) -> bool:
        """
        验证配置数据的有效性

        Returns:
            bool: 配置是否有效
        """
        try:
            # 检查必要的字段
            if not isinstance(self.initial_temperature, (int, float)):
                return False

            if not all(isinstance(x, int) for x in self.insulation_boundaries):
                return False

            if not all(isinstance(x, int) for x in self.heat_source_domains):
                return False

            if not all(isinstance(x, int) for x in self.temperature_boundaries):
                return False

            if not isinstance(self.heat_transfer_coefficient, (int, float)):
                return False

            if not isinstance(self.external_temperature, (int, float)):
                return False

            return True

        except Exception:
            return False

    def __str__(self) -> str:
        """返回类的字符串表示"""
        return f"SolidHeatTransfer(initial_T={self.initial_temperature}K, " \
               f"insulation_boundaries={self.insulation_boundaries}, " \
               f"heat_sources={self.heat_source_domains})"

    def __repr__(self) -> str:
        """返回类的详细表示"""
        return f"SolidHeatTransfer({self.to_dict()})"


# 使用示例

