import importlib
import logging

version_code = [0, 0, 1]
version_str = f"V{version_code[0]}.{version_code[1]}" + (f'.{version_code[2]}' if len(version_code) > 2 else '')
logging.info(f"### Loading: ComfyUI-ZVNodes ({version_str})")

node_list = [
    "image_nodes",
    "txt_nodes",
    "video_nodes",
    "json_nodes",
    "list_nodes",
    "layout_nodes",
    "filter_nodes",
    "ps_nodes",
    "api_nodes"
]

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
WEB_DIRECTORY = "./web"

for module_name in node_list:
    imported_module = importlib.import_module(".modules.{}".format(module_name), __name__)

    NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **imported_module.NODE_CLASS_MAPPINGS}
    NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS, **imported_module.NODE_DISPLAY_NAME_MAPPINGS}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]