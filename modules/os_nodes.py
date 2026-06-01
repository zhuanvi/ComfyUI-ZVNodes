from .utils import generate_node_mappings
import os, re

class SubFolderScannerZV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "recursive": ("BOOLEAN", {"default": False}),
                "natural_sort": ("BOOLEAN", {"default": True}),
                "abs_path": ("BOOLEAN", {"default": True}),
                "skip_empty": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("subfolders", "count")
    FUNCTION = "scan_subfolders"
    OUTPUT_NODE = False
    CATEGORY = "Utils/File"

    def scan_subfolders(self, folder_path, recursive=False, natural_sort=True, abs_path=True, skip_empty=False):
        if not folder_path or not os.path.isdir(folder_path):
            raise ValueError(f"无效的文件夹路径: {folder_path}")

        base_path = os.path.abspath(folder_path) if abs_path else folder_path
        subfolders = []

        if recursive:
            for root, dirs, files in os.walk(base_path):
                # os.walk 的 dirs 只是当前层的目录名，需要拼接完整路径
                for d in dirs:
                    full_path = os.path.join(root, d)
                    if skip_empty and not os.listdir(full_path):
                        continue
                    subfolders.append(full_path)
        else:
            for item in os.listdir(base_path):
                item_path = os.path.join(base_path, item)
                if os.path.isdir(item_path):
                    if skip_empty and not os.listdir(item_path):
                        continue
                    subfolders.append(item_path)

        # 自然排序：folder1, folder2, folder10 而非 folder1, folder10, folder2
        if natural_sort:
            def natural_key(s):
                return [int(text) if text.isdigit() else text.lower()
                        for text in re.split(r'([0-9]+)', os.path.basename(s))]
            subfolders.sort(key=natural_key)
        else:
            subfolders.sort()

        # 用换行符拼接，方便在 ComfyUI 文本节点中查看，也方便下游节点 split("\n") 解析
        result_text = "\n".join(subfolders) if subfolders else ""
        return (result_text, len(subfolders))

NODE_CONFIG = {
    "SubFolderScannerZV":{"class": SubFolderScannerZV, "name": "📁 子文件夹扫描器 (SubFolder Scanner)"},
}

NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = generate_node_mappings(NODE_CONFIG)