from .utils import generate_node_mappings
from .filter.displace import ProductionDisplacementMapNodeZV, DesaturateNodeZV, GrayToDisplacementMapNodeZV




# 导出节点
NODE_CLASS_MAPPINGS = {
    "ProductionDisplacementMapNodeZV": ProductionDisplacementMapNodeZV,
    "DesaturateNodeZV": DesaturateNodeZV,
    "GrayToDisplacementMapNodeZV": GrayToDisplacementMapNodeZV
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ProductionDisplacementMapNodeZV": "Displacement Map (Photoshop Style)",
    "DesaturateNodeZV": "Desaturate (Grayscale)",
    "GrayToDisplacementMapNodeZV": "Image to Displacement Map"
}