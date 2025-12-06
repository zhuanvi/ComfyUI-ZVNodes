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
import shutil


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

# def _tensor_to_pil(img_any: Any):
#     """将张量转换为PIL图像"""
#     try:
#         from PIL import Image
#     except Exception:
#         return None
    
#     try:
#         if torch is not None and isinstance(img_any, torch.Tensor):
#             t = img_any
#             if t.dim() == 4:
#                 t = t[0]
#             t = t.detach().cpu().clamp(0, 1)
#             arr = (t.numpy() * 255).astype("uint8")
#             if arr.shape[-1] == 3:
#                 return Image.fromarray(arr, "RGB")
#             if arr.shape[-1] == 4:
#                 return Image.fromarray(arr, "RGBA")
#         if np is not None and isinstance(img_any, np.ndarray):
#             arr = img_any
#             if arr.dtype != np.uint8:
#                 arr = np.clip(arr, 0, 1)
#                 arr = (arr * 255).astype(np.uint8)
#             if arr.shape[-1] == 3:
#                 return Image.fromarray(arr, "RGB")
#             if arr.shape[-1] == 4:
#                 return Image.fromarray(arr, "RGBA")
#     except Exception:
#         return None
#     return None

def image_to_temp_png(image: Any, max_side: int = 1024) -> Optional[str]:
    """将图像转换为临时PNG文件"""
    try:
        try:
            from PIL import Image
        except Exception:
            Image = None
            
        pil_img = None
        if Image is not None and hasattr(image, "save"):
            pil_img = image
        elif isinstance(image, dict):
            candidate = image.get("image") or (image.get("images")[0] if image.get("images") else None)
            if candidate is not None:
                if hasattr(candidate, "save"):
                    pil_img = candidate
                elif hasattr(candidate, "to_pil"):
                    pil_img = candidate.to_pil()
                else:
                    pil_img = tensor2pil(candidate)
        elif hasattr(image, "to_pil"):
            pil_img = image.to_pil()
        else:
            pil_img = tensor2pil(image)
            
        if pil_img is None:
            return None
            
        try:
            if hasattr(pil_img, "mode") and pil_img.mode not in ("RGB", "RGBA") and Image is not None:
                pil_img = pil_img.convert("RGB")
        except Exception:
            pass
            
        try:
            if Image is not None:
                w, h = pil_img.size
                if max(w, h) > max_side:
                    pil_img = pil_img.copy()
                    pil_img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        except Exception:
            pass
            
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        pil_img.save(tmp, "PNG")
        tmp.flush()
        tmp.close()
        return tmp.name
    except Exception:
        return None

def _upload_image_to_apimart_cdn(image: Any, api_key: Optional[str], max_side: int = 1024) -> Optional[str]:
    """上传图像到Apimart CDN"""
    temp_png = image_to_temp_png(image, max_side=max_side)
    if not temp_png:
        return None
        
    try:
        headers = _headers(api_key)
        token_res = requests.post(
            "https://grsai.dakka.com.cn/client/resource/newUploadTokenZH",
            headers=headers,
            json={"sux": "png"},
            timeout=30,
        )
        if token_res.status_code != 200:
            return None
            
        try:
            token_data = token_res.json().get("data") or {}
        except Exception:
            return None
            
        token = token_data.get("token")
        key = token_data.get("key")
        up_url = token_data.get("url")
        domain = token_data.get("domain")
        if not (token and key and up_url and domain):
            return None

        with open(temp_png, "rb") as f:
            up_resp = requests.post(up_url, data={"token": token, "key": key}, files={"file": f}, timeout=120)
        if up_resp.status_code not in (200, 201):
            return None

        public_url = f"{domain}/{key}"
        return public_url if public_url.startswith("http") else None
    except Exception:
        return None
    finally:
        try:
            if temp_png and os.path.exists(temp_png):
                os.unlink(temp_png)
        except Exception:
            pass


def image_to_temp_file(image: Any, max_side: int = 1024, format: str = "PNG") -> Optional[str]:
    """将图像转换为临时文件，支持指定格式"""
    try:
        try:
            from PIL import Image
        except Exception:
            Image = None
        pil_img = None
        if Image is not None and hasattr(image, "save"):
            pil_img = image
        elif isinstance(image, dict):
            candidate = image.get("image") or (image.get("images")[0] if image.get("images") else None)
            if candidate is not None:
                if hasattr(candidate, "save"):
                    pil_img = candidate
                elif hasattr(candidate, "to_pil"):
                    pil_img = candidate.to_pil()
                else:
                    pil_img = tensor2pil(candidate)
        elif hasattr(image, "to_pil"):
            pil_img = image.to_pil()
        else:
            pil_img = tensor2pil(image)
        if pil_img is None:
            return None
        try:
            if hasattr(pil_img, "mode") and pil_img.mode not in ("RGB", "RGBA") and Image is not None:
                pil_img = pil_img.convert("RGB")
        except Exception:
            pass
        try:
            if Image is not None:
                w, h = pil_img.size
                if max(w, h) > max_side:
                    pil_img = pil_img.copy()
                    pil_img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        except Exception:
            pass
        
        if format.upper() == "PNG":
            suffix = ".png"
            save_format = "PNG"
        elif format.upper() == "JPG" or format.upper() == "JPEG":
            suffix = ".jpg"
            save_format = "JPEG"
        else:
            suffix = ".png"
            save_format = "PNG"
            
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        pil_img.save(tmp, save_format)
        tmp.flush()
        tmp.close()
        return tmp.name
    except Exception:
        return None


def _video_to_temp_file(video: Any) -> Optional[str]:
    """将视频转换为临时文件"""
    try:
        if isinstance(video, str) and os.path.exists(video):
            ext = os.path.splitext(video)[1].lower() or ".mp4"
            tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
            tmp.close()
            shutil.copyfile(video, tmp.name)
            return tmp.name

        path = getattr(video, "path", None)
        if isinstance(path, str) and os.path.exists(path):
            ext = os.path.splitext(path)[1].lower() or ".mp4"
            tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
            tmp.close()
            shutil.copyfile(path, tmp.name)
            return tmp.name

        if hasattr(video, "save_to"):
            tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            tmp.close()
            ok = False
            try:
                ok = video.save_to(tmp.name)
            except Exception:
                ok = False
            if ok and os.path.exists(tmp.name):
                return tmp.name
            try:
                os.unlink(tmp.name)
            except Exception:
                pass
    except Exception:
        return None
    return None


def upload_video_to_apimart_cdn(video: Any, api_key: Optional[str]) -> Optional[str]:
    """上传视频到Apimart CDN"""
    temp_video = _video_to_temp_file(video)
    if not temp_video:
        return None
        
    try:
        headers = _headers(api_key)
        ext = (os.path.splitext(temp_video)[1].lower().lstrip(".")) or "mp4"
        token_res = requests.post(
            "https://grsai.dakka.com.cn/client/resource/newUploadTokenZH",
            headers=headers,
            json={"sux": ext},
            timeout=30,
        )
        if token_res.status_code != 200:
            return None
            
        try:
            token_data = token_res.json().get("data") or {}
        except Exception:
            return None
            
        token = token_data.get("token")
        key = token_data.get("key")
        up_url = token_data.get("url")
        domain = token_data.get("domain")
        if not (token and key and up_url and domain):
            return None
            
        with open(temp_video, "rb") as f:
            up_resp = requests.post(up_url, data={"token": token, "key": key}, files={"file": f}, timeout=300)
        if up_resp.status_code not in (200, 201):
            return None
            
        public_url = f"{domain}/{key}"
        return public_url if public_url.startswith("http") else None
    except Exception:
        return None
    finally:
        try:
            if temp_video and os.path.exists(temp_video):
                os.unlink(temp_video)
        except Exception:
            pass

def batch_upload_image_to_apimart_cdn(image: Any, api_key: Optional[str], max_side: int = 2048):
    if image is None:
        return None

    if len(image.shape) == 4:
        image_urls = []
        for i in range(image.shape[0]):
            img_tensor = image[i]
            url = _upload_image_to_apimart_cdn(img_tensor, api_key, max_side)
            if url:
                image_urls.append(url)
            else:
                return None
        
        return image_urls
    else:
        url = _upload_image_to_apimart_cdn(image, api_key, max_side)
        if url:
            return [url]
        else:
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
        return f"<VideoAdapterZV path={self.path} {self.width}x{self.height}@{self.fps}>"

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