from .utils import generate_node_mappings
from .filter.displace import ProductionDisplacementMapNodeZV




# 导出节点
NODE_CLASS_MAPPINGS = {
    "ProductionDisplacementMapNodeZV": ProductionDisplacementMapNodeZV
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ProductionDisplacementMapNodeZV": "Displacement Map (Photoshop Style)"
}