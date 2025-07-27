import os
from pathlib import Path
from PIL import Image, ImageOps, ImageSequence
from PIL.PngImagePlugin import PngInfo
import torch
import torchvision.transforms.functional as F
import numpy as np
import json
from .utils import generate_node_mappings, calculate_file_hash
from pillow_heif import register_heif_opener
register_heif_opener()

def save_image(img: torch.Tensor, path, quality, prompt=None, extra_pnginfo: dict = None):
    path = str(path)

    if len(img.shape) != 3:
        raise ValueError(f"can't take image batch as input, got {img.shape[0]} images")

    img = img.permute(2, 0, 1)
    if img.shape[0] not in (3, 4):
        raise ValueError(
            f"image must have 3 or 4 channels, but got {img.shape[0]} channels"
        )

    img = img.clamp(0, 1)
    img = F.to_pil_image(img)

    metadata = PngInfo()

    if prompt is not None:
        metadata.add_text("prompt", json.dumps(prompt))

    if extra_pnginfo is not None:
        for k, v in extra_pnginfo.items():
            metadata.add_text(k, json.dumps(v))

    

    subfolder, filename = os.path.split(path)
    ext = os.path.splitext(filename)[-1].lower()
    # 根据格式保存图像
    save_params = {}
    if ext == ".png":
        save_params["pnginfo"]=metadata
        save_params["compress_level"] = 9 - min(9, max(0, quality // 10))
    elif ext == ".jpg":
        img = img.convert("RGB")
        save_params["quality"] = quality
        save_params["subsampling"] = 0
    elif ext == ".webp":
        save_params["quality"] = quality
    elif ext == ".tiff":
        save_params["compression"] = "tiff_deflate"
    elif ext == ".heic":
        save_params["quality"] = quality
    
    img.save(path, **save_params)

    return {"filename": filename, "subfolder": subfolder, "type": "output"}


class ImageCounterNodeZV:
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {"default": "", "folder_picker": True}),
                "include_subfolders": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("image_count",)
    FUNCTION = "count_images"
    CATEGORY = "ZVNodes/image"
    DESCRIPTION = "Count images in a directory"

    def count_images(self, directory, include_subfolders):
        if not os.path.isdir(directory):
            raise ValueError(f"Directory not found: {directory}")
        
        # 支持的图片扩展名
        extensions = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".heic", ".pjp", ".pjpeg", ".jfif"]
        
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


class LoadImageFromDirZV:
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

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "mask", "directory", "name", "prefix")
    FUNCTION = "load_images"
    CATEGORY = "ZVNodes/image"
    DESCRIPTION = """Loads images from a folder into a batch, images are resized and loaded into a batch."""

    def load_images(self, folder, start_index, include_subfolders=False):
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Folder '{folder} cannot be found.'")
        
        valid_extensions = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".heic", ".pjp", ".pjpeg", ".jfif"]
        image_paths = []
        if include_subfolders:
            for root, _, files in os.walk(folder):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in valid_extensions):
                        image_paths.append(os.path.join(root, file))
        else:
            for file in os.listdir(folder):
                if any(file.lower().endswith(ext) for ext in valid_extensions) and os.path.isfile(os.path.join(folder, file)):
                    image_paths.append(os.path.join(folder, file))

        dir_files = sorted(image_paths)

        if len(dir_files) == 0:
            raise FileNotFoundError(f"No files in directory '{folder}'.")

        # start at start_index
        if len(dir_files) > start_index:
            image_path = dir_files[start_index]
        else:
            raise FileNotFoundError(f"No Enough files in directory '{folder}'.")

        i = Image.open(image_path)
        width, height = i.size
        i = ImageOps.exif_transpose(i)
        
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
            if mask.shape != (height, width):
                mask = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0), 
                                                        size=(height, width), 
                                                        mode='bilinear', 
                                                        align_corners=False).squeeze()
        else:
            mask = torch.zeros((height, width), dtype=torch.float32, device="cpu")
        
        image_dir, imagename = os.path.split(image_path)
        image_prefix= os.path.splitext(imagename)[0].lower()
        image_dir  = os.path.relpath(image_dir, folder)

        return (image, mask, image_dir, imagename, image_prefix)

class SaveImageToPathZV:
    
    @classmethod
    def INPUT_TYPES(s):
        return {
                    "required": {
                        "folder": ("STRING", {"default": "."}),
                        "subfolder": ("STRING", {"default": "."}),
                        "prefix": ("STRING", {"default": "image"}),
                        "file_extension": ((".png", ".jpg", ".webp", ".tiff", ".bmp", ".heic"), {"default": ".png"}),
                        "quality": ("INT", {"default": 95, "min": 1, "max": 100}),
                        "image": ("IMAGE",),
                        "metadata": ("BOOLEAN", {"default": False}),
                        "storage_method": (["folder_based", "suffix_based"], {"default": "folder_based"}),
                        "num_padding": ("INT", {"default": 4, "min": 0, "step": 1}),
                        "overwrite": ("BOOLEAN", {"default": True}),
                    },
                    "optional": {
                        "caption_file_extension": ("STRING", {"default": ".txt", "tooltip": "The extension for the caption file."}),
                        "caption": ("STRING", {"forceInput": True, "tooltip": "string to save as .txt file"}), 
                    },
                    "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    CATEGORY = "ZVNodes/image"
    FUNCTION = "execute"
    OUTPUT_NODE = True

    def execute(
        self,
        folder: str,
        subfolder: str,
        prefix: str,
        file_extension: str,
        quality: int,
        image: torch.Tensor,
        metadata: bool,
        storage_method: str,
        num_padding: int,
        overwrite: bool,
        caption=None, 
        caption_file_extension=".txt",
        prompt=None,
        extra_pnginfo=None,
    ):
        assert isinstance(folder, str)
        assert isinstance(subfolder, str)
        assert isinstance(prefix, str)
        assert isinstance(file_extension, str)
        assert isinstance(quality, int)
        assert isinstance(image, torch.Tensor)
        assert isinstance(metadata, bool)
        assert isinstance(storage_method, str)
        assert isinstance(num_padding, int)
        assert isinstance(overwrite, bool)

        image_path = os.path.join(folder,subfolder,f"{prefix}{file_extension}")
        path: Path = Path(image_path)
        image_path_list = []
        results = []
        if not overwrite and path.exists():
            return (image_path_list,)

        path.parent.mkdir(exist_ok=True, parents=True)

        if metadata:
            _prompt=prompt
            _extra_pnginfo=extra_pnginfo
        else:
            _prompt=None
            _extra_pnginfo=None
        
        

        if image.shape[0] == 1:
            # batch has 1 image only
            results.append(
                save_image(
                    image[0],
                    path,
                    quality,
                    prompt=_prompt,
                    extra_pnginfo=_extra_pnginfo,
                )
            )
            if caption is not None:
                txt_path = path.parent / (path.stem+caption_file_extension)
                with txt_path.open('w', encoding="UTF-8") as f:
                    f.write(caption)
            image_path_list.append(str(path))
        else:
            # batch has multiple images
            for i, img in enumerate(image):
                batch_name = str(i).zfill(num_padding)
                if storage_method == "suffix_based":
                    subpath = path.with_stem(f"{path.stem}_{batch_name}")
                else:
                    subpath = path.parent / batch_name / path.name
                    subpath.parent.mkdir(exist_ok=True, parents=True)
                    
                results.append(
                    save_image(
                        img,
                        subpath,
                        quality,
                        prompt=_prompt,
                        extra_pnginfo=_extra_pnginfo,
                    )
                )
                if caption is not None:
                    txt_path = subpath.parent / (subpath.stem+caption_file_extension)
                    with txt_path.open('w', encoding="UTF-8") as f:
                        f.write(caption)
                image_path_list.append(str(subpath))

        return (image_path_list,)
    

NODE_CONFIG = {
    "LoadImageFromDirZV": {"class": LoadImageFromDirZV, "name": "Load One Image (Directory)"},
    "SaveImageToPathZV": {"class": SaveImageToPathZV, "name": "Save Image (Directory)"},
    "ImageCounterNodeZV":{"class": ImageCounterNodeZV, "name": "Count Image (Directory)"},
}

NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = generate_node_mappings(NODE_CONFIG)