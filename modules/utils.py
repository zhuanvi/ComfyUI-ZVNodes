import hashlib
import os
import torch
import numpy as np
import numpy.typing as npt
from collections.abc import Callable, Sequence
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
import chardet
import requests
import tempfile
import urllib3
import shutil
import folder_paths
import io
import uuid
import subprocess

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

def generate_node_mappings(node_config):
    node_class_mappings = {}
    node_display_name_mappings = {}

    for node_name, node_info in node_config.items():
        node_class_mappings[node_name] = node_info["class"]
        node_display_name_mappings[node_name] = node_info.get("name", node_info["class"].__name__)

    return node_class_mappings, node_display_name_mappings

def calculate_file_hash(filename: str):
    h = hashlib.sha256()
    h.update(filename.encode())
    h.update(str(os.path.getmtime(filename)).encode())
    return h.hexdigest()

def normalize_path(path):
    return path.replace('\\', '/')

# region TENSOR Utilities
def to_numpy(image: torch.Tensor) -> npt.NDArray[np.uint8]:
    """Converts a tensor to a ndarray with proper scaling and type conversion."""
    np_array = np.clip(255.0 * image.cpu().numpy(), 0, 255).astype(np.uint8)
    return np_array

def handle_batch(
    tensor: torch.Tensor,
    func: Callable[[torch.Tensor], Image.Image | npt.NDArray[np.uint8]],
) -> list[Image.Image] | list[npt.NDArray[np.uint8]]:
    """Handles batch processing for a given tensor and conversion function."""
    return [func(tensor[i]) for i in range(tensor.shape[0])]


def tensor2pil(tensor: torch.Tensor) -> list[Image.Image]:
    """Converts a batch of tensors to a list of PIL Images."""

    def single_tensor2pil(t: torch.Tensor) -> Image.Image:
        np_array = to_numpy(t)
        if np_array.ndim == 2:  # (H, W) for masks
            return Image.fromarray(np_array, mode="L")
        elif np_array.ndim == 3:  # (H, W, C) for RGB/RGBA
            if np_array.shape[2] == 3:
                return Image.fromarray(np_array, mode="RGB")
            elif np_array.shape[2] == 4:
                return Image.fromarray(np_array, mode="RGBA")
        raise ValueError(f"Invalid tensor shape: {t.shape}")

    return handle_batch(tensor, single_tensor2pil)


def pil2tensor(images: Image.Image | list[Image.Image]) -> torch.Tensor:
    """Converts a PIL Image or a list of PIL Images to a tensor."""

    def single_pil2tensor(image: Image.Image) -> torch.Tensor:
        np_image = np.array(image).astype(np.float32) / 255.0
        if np_image.ndim == 2:  # Grayscale
            return torch.from_numpy(np_image).unsqueeze(0)  # (1, H, W)
        else:  # RGB or RGBA
            return torch.from_numpy(np_image).unsqueeze(0)  # (1, H, W, C)

    if isinstance(images, Image.Image):
        return single_pil2tensor(images)
    else:
        return torch.cat([single_pil2tensor(img) for img in images], dim=0)

def _headers(api_key: Optional[str]) -> Dict[str, str]:
    """生成HTTP请求头"""
    key = (api_key or "").strip()
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0"
    }

def _tensor_to_pil(img_any: Any):
    """将张量转换为PIL图像"""
    try:
        from PIL import Image
    except Exception:
        return None
    
    try:
        if torch is not None and isinstance(img_any, torch.Tensor):
            t = img_any
            if t.dim() == 4:
                t = t[0]
            t = t.detach().cpu().clamp(0, 1)
            arr = (t.numpy() * 255).astype("uint8")
            if arr.shape[-1] == 3:
                return Image.fromarray(arr, "RGB")
            if arr.shape[-1] == 4:
                return Image.fromarray(arr, "RGBA")
        if np is not None and isinstance(img_any, np.ndarray):
            arr = img_any
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 1)
                arr = (arr * 255).astype(np.uint8)
            if arr.shape[-1] == 3:
                return Image.fromarray(arr, "RGB")
            if arr.shape[-1] == 4:
                return Image.fromarray(arr, "RGBA")
    except Exception:
        return None
    return None


class VideoAdapter:
    """视频适配器类"""
    def __init__(self, path: Optional[str]):
        self.path = path
        self.width, self.height, self.fps = self._get_video_details(path)

    def _get_video_details(self, path: Optional[str]):
        """获取视频详细信息"""
        try:
            if not path or not os.path.exists(path):
                return 1280, 720, 30
            try:
                import cv2
            except Exception:
                return 1280, 720, 30
                
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                return 1280, 720, 30
                
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            if not fps or fps == 0:
                fps = 30
            return width or 1280, height or 720, int(fps)
        except Exception:
            return 1280, 720, 30

    def __repr__(self):
        return f"<VideoAdapter path={self.path} {self.width}x{self.height}@{self.fps}>"

    def get_dimensions(self):
        """获取视频尺寸"""
        return self.width, self.height

    def save_to(self, output_path, **kwargs):
        """保存视频到指定路径"""
        try:
            if self.path and os.path.exists(self.path):
                shutil.copyfile(self.path, output_path)
                return True
            return False
        except Exception:
            return False

    def get_components(self):
        """获取视频组件信息"""
        return {
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "bit_rate": 0,
        }


def get_temp_video_path(prefix="temp", ext=".mp4"):
    """在 ComfyUI 的 temp 目录下生成唯一视频路径"""
    temp_dir = folder_paths.get_temp_directory()
    os.makedirs(temp_dir, exist_ok=True)
    filename = f"{prefix}_{uuid.uuid4().hex}{ext}"
    return os.path.join(temp_dir, filename)

def run_ffmpeg(cmd, desc="ffmpeg"):
    """执行 ffmpeg 命令并检查错误"""
    print(f"[{desc}] Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"[{desc}] stderr:\n{proc.stderr}")
        raise RuntimeError(f"ffmpeg error: {proc.stderr}")
    print(f"[{desc}] Done.")
    return proc.stdout

def save_tensor_images(image_tensor, directory, prefix="img"):
    """
    将 ComfyUI 的 IMAGE 张量 (B,H,W,C) 保存为 PNG 图片序列，返回文件路径列表。
    image_tensor 值范围为 [0,1]。
    """
    os.makedirs(directory, exist_ok=True)
    paths = []
    for i, img in enumerate(image_tensor):
        # 转为 0-255 uint8
        np_img = (img.cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(np_img)
        filepath = os.path.join(directory, f"{prefix}_{i:05d}.png")
        pil_img.save(filepath)
        paths.append(filepath)
    return paths