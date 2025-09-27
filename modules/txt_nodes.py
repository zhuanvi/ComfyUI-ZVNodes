import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from PIL.PngImagePlugin import PngInfo
import torch
import torchvision.transforms.functional as F
import numpy as np
import json
from datetime import datetime
import uuid
import folder_paths
import comfy.utils
from .utils import detect_encoding, generate_node_mappings

class TxtCounterNodeZV:
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {"default": "", "folder_picker": True}),
                "include_subfolders": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("txt_count",)
    FUNCTION = "count_txts"
    CATEGORY = "ZVNodes/txt"
    DESCRIPTION = "Count txts in a directory"

    def count_txts(self, directory, include_subfolders):
        if not os.path.isdir(directory):
            raise ValueError(f"Directory not found: {directory}")
        
        # 支持的txt扩展名
        extensions = [".txt"]
        
        count = 0
        # 更高效的文件遍历方式
        if include_subfolders:
            # 递归遍历所有文件和子目录
            for _, _, files in os.walk(directory):
                for file in files:
                    ext = os.path.splitext(file)[-1].lower()
                    if ext in extensions:
                        count += 1
        else:
            # 仅检查顶层目录
            for file in os.listdir(directory):
                if os.path.isfile(os.path.join(directory, file)):
                    ext = os.path.splitext(file)[-1].lower()
                    if ext in extensions:
                        count += 1
        
        return (count,)


class LoadTxtFromDirZV:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder": ("STRING", {"default": ""})
            },
            "optional": {
                "start_index": ("INT", {"default": 0, "min": 0, "step": 1}),
                "include_subfolders": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("txt", "directory", "name", "prefix")
    FUNCTION = "load_txt"
    CATEGORY = "ZVNodes/txt"
    DESCRIPTION = """Loads images from a folder into a batch, images are resized and loaded into a batch."""

    def load_txt(self, folder, start_index, include_subfolders=False):
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Folder '{folder} cannot be found.'")
        
        valid_extensions = [".txt"]
        txt_paths = []
        if include_subfolders:
            for root, _, files in os.walk(folder):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in valid_extensions):
                        txt_paths.append(os.path.join(root, file))
        else:
            for file in os.listdir(folder):
                if any(file.lower().endswith(ext) for ext in valid_extensions) and os.path.isfile(os.path.join(folder, file)):
                    txt_paths.append(os.path.join(folder, file))

        dir_files = sorted(txt_paths)

        if len(dir_files) == 0:
            raise FileNotFoundError(f"No files in directory '{folder}'.")

        # start at start_index
        if len(dir_files) > start_index:
            txt_path = dir_files[start_index]
        else:
            raise FileNotFoundError(f"No Enough files in directory '{folder}'.")
    
        encoding = detect_encoding(txt_path)
        with open(txt_path, "r", encoding=encoding) as f:
            txt = f.read()

               
        txt_dir, name = os.path.split(txt_path)
        prefix= os.path.splitext(name)[0].lower()
        txt_dir  = os.path.relpath(txt_dir, folder)

        return (txt, txt_dir, name, prefix)

class SaveTxtToPathZV:
    
    @classmethod
    def INPUT_TYPES(s):
        return {
                    "required": {
                        "text": ("STRING", {"forceInput": True, "tooltip": "string to save as .txt file"}), 
                        "folder": ("STRING", {"default": "."}),
                        "subfolder": ("STRING", {"default": "."}),
                        "prefix": ("STRING", {"default": "txt"}),
                        "file_extension": ((".txt", ), {"default": ".txt"}),
                        "storage_method": (["folder_based", "suffix_based"], {"default": "folder_based"}),
                        "num_padding": ("INT", {"default": 4, "min": 0, "step": 1}),
                        "overwrite": ("BOOLEAN", {"default": True}),
                    }
                }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    CATEGORY = "ZVNodes/txt"
    FUNCTION = "execute"
    OUTPUT_NODE = True

    def execute(
        self,
        text: str,
        folder: str,
        subfolder: str,
        prefix: str,
        file_extension: str,
        storage_method: str,
        num_padding: int,
        overwrite: bool,
    ):
        assert isinstance(text, str)
        assert isinstance(folder, str)
        assert isinstance(subfolder, str)
        assert isinstance(prefix, str)
        assert isinstance(file_extension, str)
        assert isinstance(storage_method, str)
        assert isinstance(num_padding, int)
        assert isinstance(overwrite, bool)

        txt_path = os.path.join(folder,subfolder,f"{prefix}{file_extension}")
        path: Path = Path(txt_path)
        txt_path_list = []
        results = []
        if not overwrite and path.exists():
            return (txt_path_list,)

        path.parent.mkdir(exist_ok=True, parents=True)
        
        if isinstance(text, str) or (isinstance(text, list) and len(text) == 1):
            # text is not list
            path.write_text(text, encoding="utf-8")
            results.append(txt_path)
        else:
            # batch has multiple texts
            for i, txt in enumerate(text):
                batch_name = str(i).zfill(num_padding)
                if storage_method == "suffix_based":
                    subpath = path.with_stem(f"{path.stem}_{batch_name}")
                else:
                    subpath = path.parent / batch_name / path.name
                    subpath.parent.mkdir(exist_ok=True, parents=True)
                    
                subpath.write_text(txt, encoding="UTF-8")
                txt_path_list.append(str(subpath))

        return (txt_path_list,)


NODE_CONFIG = {
    "LoadTxtFromDirZV": {"class": LoadTxtFromDirZV, "name": "Load One TXT (Directory)"},
    "SaveTxtToPathZV": {"class": SaveTxtToPathZV, "name": "Save TXT (Directory)"}
}

NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = generate_node_mappings(NODE_CONFIG)