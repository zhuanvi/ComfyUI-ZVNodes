import os
import json
import time
import requests
import shutil
import re
import base64
import io
import tempfile
from typing import Any, Dict, List, Optional, Tuple
from .utils import pil2tensor

try:
    import numpy as np
except Exception:
    np = None

try:
    import torch
except Exception:
    torch = None

try:
    from comfy.comfy_types import IO
    import folder_paths
except Exception:
    class IO:
        VIDEO = "VIDEO"
    class folder_paths:
        @staticmethod
        def get_output_directory():
            return os.path.join(os.getcwd(), "output")

# 配置常量
API_CONFIG = {
    "base_url": "https://api.apimart.ai/v1/videos/generations",
    "status_url": "https://api.apimart.ai/v1/tasks",
    "images_base_url": "https://api.apimart.ai/v1/images/generations",
    "api_key": os.getenv("SORA2_API_KEY", ""),
    "timeout": 30,
    "max_retries": 3,
    "retry_delay": 2,
    "poll_interval": 5,
    "max_poll_time": 300,
}

CATEGORY = "ZVNodes/apimart"
NODE_DIR = os.path.dirname(__file__)
TASK_FILE = os.path.join(NODE_DIR, "apimart_task_history.json")


# ==================== 文件操作函数 ====================

def _ensure_task_file():
    """确保任务文件存在"""
    if not os.path.exists(TASK_FILE):
        with open(TASK_FILE, "w", encoding="utf-8") as f:
            json.dump({"tasks": []}, f, ensure_ascii=False, indent=2)


def _read_task_queue() -> List[Dict[str, Any]]:
    """读取任务队列"""
    _ensure_task_file()
    with open(TASK_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("tasks", [])


def _trim_task_list(tasks: List[Dict[str, Any]], max_keep: int = 50) -> List[Dict[str, Any]]:
    """修剪任务列表，保留最新的N个"""
    try:
        sorted_tasks = sorted(
            tasks,
            key=lambda t: t.get("saved_at", t.get("submitted_at", "1970-01-01 00:00:00")),
            reverse=True,
        )
        return sorted_tasks[:max_keep]
    except Exception:
        return tasks[:max_keep]


def _write_task_queue(tasks: List[Dict[str, Any]]):
    """写入任务队列"""
    with open(TASK_FILE, "w", encoding="utf-8") as f:
        json.dump({"tasks": tasks, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")}, 
                 f, ensure_ascii=False, indent=2)


# ==================== API工具函数 ====================

def _headers(api_key: Optional[str]) -> Dict[str, str]:
    """生成HTTP请求头"""
    key = (api_key or "").strip() or API_CONFIG.get("api_key")
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0"
    }


def _headers_form(api_key: Optional[str]) -> Dict[str, str]:
    """生成表单上传的HTTP请求头"""
    key = (api_key or "").strip() or API_CONFIG.get("api_key")
    return {
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0",
    }


def _sanitize_url(s: str) -> Optional[str]:
    """清理URL字符串"""
    if not isinstance(s, str):
        return None
    s2 = s.strip().strip("`\"'")
    return s2 if s2.startswith("http") else None


def _extract_video_url(obj: Any) -> Optional[str]:
    """从API响应中提取视频URL"""
    if isinstance(obj, str):
        return _sanitize_url(obj)

    if isinstance(obj, dict):
        # 直接字段
        for key in ("video_url", "url", "result_url"):
            v = obj.get(key)
            if isinstance(v, str):
                sv = _sanitize_url(v)
                if sv:
                    return sv
            if isinstance(v, list) and v:
                for item in v:
                    sv = _sanitize_url(item) if isinstance(item, str) else _extract_video_url(item)
                    if sv:
                        return sv

        # 嵌套 output / results / artifacts
        output = obj.get("output")
        if isinstance(output, dict):
            v = output.get("url")
            if isinstance(v, str):
                sv = _sanitize_url(v)
                if sv:
                    return sv
            if isinstance(v, list) and v:
                for item in v:
                    sv = _sanitize_url(item) if isinstance(item, str) else _extract_video_url(item)
                    if sv:
                        return sv

        for path in ("results", "artifacts"):
            arr = obj.get(path)
            if isinstance(arr, list):
                for el in arr:
                    v = _extract_video_url(el)
                    if v:
                        return v

        # result.videos[0].url 或字符串列表
        result = obj.get("result") or obj.get("data") or obj.get("output")
        if isinstance(result, dict):
            videos = result.get("videos")
            if isinstance(videos, list) and videos:
                for first in videos:
                    if isinstance(first, str):
                        sv = _sanitize_url(first)
                        if sv:
                            return sv
                    if isinstance(first, dict):
                        v = first.get("url")
                        if isinstance(v, str):
                            sv = _sanitize_url(v)
                            if sv:
                                return sv
                        if isinstance(v, list) and v:
                            for item in v:
                                sv = _sanitize_url(item) if isinstance(item, str) else _extract_video_url(item)
                                if sv:
                                    return sv

        # data 列表包裹
        data = obj.get("data")
        if isinstance(data, list) and data:
            for el in data:
                v = _extract_video_url(el)
                if v:
                    return v

        # 兜底：正则匹配
        try:
            text = json.dumps(obj, ensure_ascii=False)
            m = re.search(r"https?://[^\s\"']+", text)
            if m:
                return _sanitize_url(m.group(0)) or m.group(0)
        except Exception:
            pass

    if isinstance(obj, list) and obj:
        for el in obj:
            v = _extract_video_url(el)
            if v:
                return v

    return None

def _extract_image_urls(obj: Any) -> List[str]:
    """从API响应中提取多个图像URL"""
    urls = []
    if isinstance(obj, str):
        url = _sanitize_url(obj)
        if url:
            urls.append(url)

    if isinstance(obj, dict):
        # 检查常见字段
        for key in ("image_url", "url", "image_urls"):
            v = obj.get(key)
            if isinstance(v, str):
                sv = _sanitize_url(v)
                if sv:
                    urls.append(sv)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, str):
                        sv = _sanitize_url(item)
                        if sv:
                            urls.append(sv)
                    elif isinstance(item, dict):
                        urls.extend(_extract_image_urls(item))

        # 检查 images 字段
        images = obj.get("images")
        if isinstance(images, list):
            for img in images:
                if isinstance(img, str):
                    sv = _sanitize_url(img)
                    if sv:
                        urls.append(sv)
                elif isinstance(img, dict):
                    img_url = img.get("url")
                    if isinstance(img_url, str):
                        sv = _sanitize_url(img_url)
                        if sv:
                            urls.append(sv)

        # 检查 data 字段
        data = obj.get("data")
        if data:
            urls.extend(_extract_image_urls(data))

        # 检查 output 字段
        output = obj.get("output")
        if output:
            urls.extend(_extract_image_urls(output))

        # 兜底：正则查找所有http链接
        try:
            text = json.dumps(obj, ensure_ascii=False)
            for match in re.findall(r"https?://[^\s\"']+", text):
                sv = _sanitize_url(match)
                if sv and sv not in urls:
                    urls.append(sv)
        except Exception:
            pass

    if isinstance(obj, list):
        for item in obj:
            urls.extend(_extract_image_urls(item))

    # 去重
    return list(dict.fromkeys(urls))  # 保持顺序

# ==================== 图像处理函数 ====================

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


def _image_to_data_url(image: Any) -> Optional[str]:
    """将图像转换为Data URL"""
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
                    pil_img = _tensor_to_pil(candidate)
        elif hasattr(image, "to_pil"):
            pil_img = image.to_pil()
        else:
            pil_img = _tensor_to_pil(image)

        if pil_img is None:
            return None

        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except Exception:
        return None


def _image_to_temp_png(image: Any, max_side: int = 1024) -> Optional[str]:
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
                    pil_img = _tensor_to_pil(candidate)
        elif hasattr(image, "to_pil"):
            pil_img = image.to_pil()
        else:
            pil_img = _tensor_to_pil(image)
            
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

def _image_to_temp_file(image: Any, max_side: int = 1024, format: str = "PNG") -> Optional[str]:
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
                    pil_img = _tensor_to_pil(candidate)
        elif hasattr(image, "to_pil"):
            pil_img = image.to_pil()
        else:
            pil_img = _tensor_to_pil(image)
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

# ==================== 上传函数 ====================

def _upload_image_and_get_url(image: Any, max_side: int = 1024) -> Optional[str]:
    """上传图像到外部图床并获取URL"""
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
                    pil_img = _tensor_to_pil(candidate)
        elif hasattr(image, "to_pil"):
            pil_img = image.to_pil()
        else:
            pil_img = _tensor_to_pil(image)

        if pil_img is None:
            return None

        try:
            if pil_img.mode not in ("RGB", "RGBA") and Image is not None:
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
        try:
            pil_img.save(tmp, "PNG")
            tmp.flush()
            tmp.close()

            # 自定义上传端点
            custom_url = os.getenv("APIMART_UPLOAD_URL", "").strip()
            if custom_url:
                field = os.getenv("APIMART_UPLOAD_FIELD", "file").strip() or "file"
                auth = os.getenv("APIMART_UPLOAD_AUTH", "").strip()
                headers = {}
                if auth:
                    headers["Authorization"] = auth
                with open(tmp.name, "rb") as f:
                    resp_c = requests.post(custom_url, headers=headers, files={field: f}, timeout=60)
                url_c = None
                try:
                    body_c = resp_c.json()
                    for k in ("url", "download_url", "link", "fileUrl"):
                        v = body_c.get(k)
                        if isinstance(v, str) and v.startswith("http"):
                            url_c = v
                            break
                    if not url_c:
                        data_c = body_c.get("data")
                        if isinstance(data_c, dict):
                            for k in ("url", "download_url", "link"):
                                v = data_c.get(k)
                                if isinstance(v, str) and v.startswith("http"):
                                    url_c = v
                                    break
                except Exception:
                    url_c = (resp_c.text or "").strip()
                if url_c and url_c.startswith("http") and resp_c.status_code in (200, 201):
                    return url_c

            # 0x0.st
            try:
                with open(tmp.name, "rb") as f:
                    resp = requests.post("https://0x0.st", files={"file": f}, timeout=60)
                url = (resp.text or "").strip()
                if resp.status_code == 200 and url.startswith("http"):
                    return url
            except Exception:
                pass

            # transfer.sh
            try:
                with open(tmp.name, "rb") as f:
                    fname = os.path.basename(tmp.name) or "image.png"
                    resp2 = requests.put(f"https://transfer.sh/{fname}", data=f.read(), timeout=60)
                url2 = (resp2.text or "").strip()
                if resp2.status_code in (200, 201) and url2.startswith("http"):
                    return url2
            except Exception:
                pass

            # catbox.moe
            try:
                with open(tmp.name, "rb") as f:
                    resp3 = requests.post(
                        "https://catbox.moe/user/api.php",
                        data={"reqtype": "fileupload"},
                        files={"fileToUpload": f},
                        timeout=60,
                    )
                url3 = (resp3.text or "").strip()
                if resp3.status_code == 200 and url3.startswith("http"):
                    return url3
            except Exception:
                pass

            return None
        finally:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass
    except Exception:
        return None


def _upload_image_to_apimart_cdn(image: Any, api_key: Optional[str], max_side: int = 1024) -> Optional[str]:
    """上传图像到Apimart CDN"""
    temp_png = _image_to_temp_png(image, max_side=max_side)
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


def _upload_video_to_apimart_cdn(video: Any, api_key: Optional[str]) -> Optional[str]:
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


# ==================== API调用函数 ====================

def _submit_generation(payload: Dict[str, Any], api_key: Optional[str]) -> Tuple[int, Dict[str, Any]]:
    """提交生成任务"""
    url = API_CONFIG.get("base_url")
    timeout = API_CONFIG.get("timeout", 30)
    resp = requests.post(url, headers=_headers(api_key), json=payload, timeout=timeout)
    status_code = resp.status_code
    try:
        body = resp.json()
    except Exception:
        body = {"raw": resp.text}
    return status_code, body


def _submit_generation_multipart(img_path: str, form_fields: Dict[str, Any], 
                                api_key: Optional[str], file_field: str = "image") -> Tuple[int, Dict[str, Any]]:
    """通过multipart表单提交生成任务"""
    url = API_CONFIG.get("base_url")
    timeout = API_CONFIG.get("timeout", 30)
    files = None
    try:
        files = {file_field: (os.path.basename(img_path) or "image.png", open(img_path, "rb"), "image/png")}
        resp = requests.post(url, headers=_headers_form(api_key), files=files, data=form_fields, timeout=timeout)
    finally:
        try:
            if files and files[file_field][1]:
                files[file_field][1].close()
        except Exception:
            pass
    status_code = resp.status_code
    try:
        body = resp.json()
    except Exception:
        body = {"raw": resp.text}
    return status_code, body

def _submit_image_generation(payload: Dict[str, Any], api_key: Optional[str]) -> Tuple[int, Dict[str, Any]]:
    """提交Seedream 4.0图像生成任务"""
    url = API_CONFIG.get("images_base_url")
    timeout = API_CONFIG.get("timeout", 30)
    resp = requests.post(url, headers=_headers(api_key), json=payload, timeout=timeout)
    status_code = resp.status_code
    try:
        body = resp.json()
    except Exception:
        body = {"raw": resp.text}
    return status_code, body

def _query_task(task_id: str, api_key: Optional[str]) -> Tuple[int, Dict[str, Any]]:
    """查询任务状态"""
    url = f"{API_CONFIG.get('status_url')}/{task_id}"
    timeout = API_CONFIG.get("timeout", 30)
    resp = requests.get(url, headers=_headers(api_key), timeout=timeout)
    status_code = resp.status_code
    try:
        body = resp.json()
    except Exception:
        body = {"raw": resp.text}
    return status_code, body


def _submit_remix_by_video_id(video_id: str, prompt: str, model: str, duration: int, 
                             api_key: Optional[str], aspect_ratio: str = "16:9") -> Tuple[int, Dict[str, Any]]:
    """通过视频ID提交Remix任务"""
    base = "https://api.apimart.ai/v1/videos"
    url = f"{base}/{video_id}/remix"
    timeout = API_CONFIG.get("timeout", 30)
    payload = {
        "model": model,
        "prompt": prompt,
        "duration": duration,
        "aspect_ratio": aspect_ratio,
    }
    resp = requests.post(url, headers=_headers(api_key), json=payload, timeout=timeout)
    code = resp.status_code
    try:
        body = resp.json()
    except Exception:
        body = {"raw": resp.text}
    return code, body


def _submit_remix_by_url(video_url: str, prompt: str, model: str, duration: int, 
                        api_key: Optional[str]) -> Tuple[int, Dict[str, Any]]:
    """通过视频URL提交Remix任务"""
    base = "https://api.apimart.ai/v1/videos/remix"
    timeout = API_CONFIG.get("timeout", 30)
    headers = _headers(api_key)
    
    for field in ("url", "video_url", "video_urls"):
        payload = {"model": model, "prompt": prompt, "duration": duration}
        if field == "video_urls":
            payload[field] = [video_url]
        else:
            payload[field] = video_url
            
        resp = requests.post(base, headers=headers, json=payload, timeout=timeout)
        code = resp.status_code
        try:
            body = resp.json()
        except Exception:
            body = {"raw": resp.text}
            
        task_id = None
        if isinstance(body, dict):
            data = body.get("data")
            if isinstance(data, list) and data:
                task_id = data[0].get("task_id")
            elif isinstance(data, dict):
                task_id = data.get("task_id")
            task_id = task_id or body.get("task_id")
            
        if code == 200 or task_id:
            return code, body
            
        if code not in (413, 400, 404):
            return code, body
            
    return code, body


# ==================== 任务管理函数 ====================

def _save_task(task_id: str):
    """保存任务ID到历史记录"""
    tasks = _read_task_queue()
    tasks.append({"task_id": task_id, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")})
    tasks = _trim_task_list(tasks)
    _write_task_queue(tasks)


def _pop_first_task() -> Optional[Dict[str, Any]]:
    """弹出第一个任务"""
    tasks = _read_task_queue()
    if not tasks:
        return None
    first = tasks.pop(0)
    _write_task_queue(tasks)
    return first


def _remove_task_by_id(task_id: str):
    """根据ID移除任务"""
    tasks = _read_task_queue()
    new_tasks = [t for t in tasks if t.get("task_id") != task_id]
    _write_task_queue(new_tasks)


# ==================== 下载函数 ====================

def _download_with_retries(url: str, target_path: str, max_retries: int = 3):
    """带重试的下载函数"""
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/123.0.0.0"}
    backoff = 5.0
    
    for attempt in range(1, max_retries + 1):
        try:
            with requests.get(url, headers=headers, stream=True, timeout=API_CONFIG.get("timeout", 30)) as r:
                if 400 <= r.status_code < 500:
                    return False, f"客户端错误 {r.status_code}，停止重试"
                r.raise_for_status()
                with open(target_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            return True, f"下载成功: {target_path}"
        except Exception as e:
            if attempt == max_retries:
                return False, f"下载失败: {e}"
            time.sleep(backoff)
            backoff *= 2

def _download_image(url: str, target_path: str) -> bool:
    """下载单个图像文件"""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/123.0.0.0"}
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(target_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return True
    except Exception as e:
        print(f"下载图像失败: {e}")
        return False


def _load_image_to_tensor(image_path: str):
    """将图像文件加载为torch.Tensor，确保格式正确"""
    try:
        from PIL import Image
        import torch
        
        # 打开图像
        pil_img = Image.open(image_path)

        return pil2tensor(pil_img)
        
    except Exception as e:
        print(f"加载图像失败: {e}")
        return None


# ==================== 重命名后的类 ====================

class VideoAdapterZV:
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


class ApimartText2VideoSubmitZV:
    """文生视频提交任务类"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "api_key": ("STRING", {"default": ""}),
                "aspect_ratio": ("STRING", {"default": "16:9", "choices": ["16:9", "9:16"]}),
                "duration": ("STRING", {"default": "10", "choices": ["10", "15"]}),
                "model": ("STRING", {"default": "sora-2", "choices": ["sora-2", "sora-2-pro"]}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("report", "task_id")
    CATEGORY = CATEGORY
    FUNCTION = "submit"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time_ns()

    def submit(self, prompt: str, api_key: str, aspect_ratio: str, duration: str, model: str):
        """提交文生视频任务"""
        payload = {
            "model": model,
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "duration": int(duration),
        }
        code, body = _submit_generation(payload, api_key)

        # 提取 task_id
        task_id = None
        if isinstance(body, dict):
            data = body.get("data")
            if isinstance(data, list) and data:
                task_id = data[0].get("task_id")
            elif isinstance(data, dict):
                task_id = data.get("task_id")
            task_id = task_id or body.get("task_id")

        report = f"HTTP {code} - 提交成功" if code == 200 else f"HTTP {code} - 提交失败"
        if task_id:
            _save_task(task_id)
            report += f" | task_id: {task_id} 已保存"
        else:
            report += " | 未返回 task_id"

        return (report, task_id or "")


class ApimartImage2VideoSubmitZV:
    """图生视频提交任务类"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "api_key": ("STRING", {"default": ""}),
                "aspect_ratio": ("STRING", {"default": "16:9", "choices": ["16:9", "9:16"]}),
                "duration": ("STRING", {"default": "10", "choices": ["10", "15"]}),
                "model": ("STRING", {"default": "sora-2", "choices": ["sora-2", "sora-2-pro"]}),
            },
            "optional": {}
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("report", "task_id")
    CATEGORY = CATEGORY
    FUNCTION = "submit"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time_ns()

    def submit(self, image, prompt: str, api_key: str, aspect_ratio: str, duration: str, model: str):
        """提交图生视频任务"""
        # 1) 优先：上传到Apimart CDN
        cdn_url = _upload_image_to_apimart_cdn(image, api_key, max_side=1024)
        print(f"已经上传图片到{cdn_url}!")
        if cdn_url:
            payload = {
                "model": model,
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "duration": int(duration),
                "image_urls": [cdn_url],
            }
            code, body = _submit_generation(payload, api_key)

            # 如果失败，尝试使用单一url字段
            if code == 413 or (code != 200 and isinstance(body, dict) and body.get("error")):
                payload_retry = {
                    "model": model,
                    "prompt": prompt,
                    "aspect_ratio": aspect_ratio,
                    "duration": int(duration),
                    "url": cdn_url,
                }
                code, body = _submit_generation(payload_retry, api_key)

            # 提取task_id
            task_id = None
            if isinstance(body, dict):
                data = body.get("data")
                if isinstance(data, list) and data:
                    task_id = data[0].get("task_id")
                elif isinstance(data, dict):
                    task_id = data.get("task_id")
                task_id = task_id or body.get("task_id")

            report = f"HTTP {code} - 提交成功 | 使用CDN外链" if code == 200 else f"HTTP {code} - 提交失败 | 使用CDN外链"
            if task_id:
                _save_task(task_id)
                report += f" | task_id: {task_id} 已保存"
            else:
                report += " | 未返回 task_id"

            return (report, task_id or "")

        # 2) 兜底：使用本地文件上传
        temp_png = _image_to_temp_png(image, max_side=1024)
        if not temp_png:
            return ("图片外链上传失败且本地PNG生成失败。请检查网络或更换图源。", "")

        form_fields = {
            "model": model,
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "duration": int(duration),
        }
        file_field = os.getenv("APIMART_FILE_FIELD", "image").strip() or "image"
        code, body = _submit_generation_multipart(temp_png, form_fields, api_key, file_field=file_field)

        # 提取task_id
        task_id = None
        if isinstance(body, dict):
            data = body.get("data")
            if isinstance(data, list) and data:
                task_id = data[0].get("task_id")
            elif isinstance(data, dict):
                task_id = data.get("task_id")
            task_id = task_id or body.get("task_id")

        # 清理临时文件
        try:
            if temp_png and os.path.exists(temp_png):
                os.unlink(temp_png)
        except Exception:
            pass

        report = f"HTTP {code} - 提交成功 | 使用本地文件上传({file_field})" if code == 200 else f"HTTP {code} - 提交失败 | 使用本地文件上传({file_field})"
        if task_id:
            _save_task(task_id)
            report += f" | task_id: {task_id} 已保存"
        else:
            err_hint = ""
            raw = body.get("raw") if isinstance(body, dict) else None
            if code in (413, 415, 422):
                err_hint = " | 接口可能不支持文件直传，或需指定特定字段名"
            elif code in (400, 404):
                err_hint = " | 字段或端点不匹配，请确认生成接口是否支持multipart"
            report += f" | 未返回task_id{err_hint}"

        return (report, task_id or "")


class Veo31Image2VideoSubmitZV:
    """Veo3.1图生视频提交任务类"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "api_key": ("STRING", {"default": ""}),
                "model": ("STRING", {"default": "veo3.1-fast", "choices": ["veo3.1-fast", "veo3.1-quality"]}),
            },
            "optional": {}
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("report", "task_id")
    CATEGORY = CATEGORY
    FUNCTION = "submit"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time_ns()

    def submit(self, image, prompt: str, api_key: str, model: str):
        """提交Veo3.1图生视频任务"""
        aspect_ratio = "16:9"
        duration = 8

        # 优先使用CDN外链
        cdn_url = _upload_image_to_apimart_cdn(image, api_key, max_side=1024)
        if cdn_url:
            payload = {
                "model": model,
                "prompt": prompt,
                "duration": duration,
                "aspect_ratio": aspect_ratio,
                "image_urls": [cdn_url],
            }
            code, body = _submit_generation(payload, api_key)

            # 非200时尝试使用单一url字段
            if code != 200 and isinstance(body, dict) and (body.get("error") or body.get("message")):
                payload_retry = {
                    "model": model,
                    "prompt": prompt,
                    "duration": duration,
                    "aspect_ratio": aspect_ratio,
                    "url": cdn_url,
                }
                code, body = _submit_generation(payload_retry, api_key)

            # 提取task_id
            task_id = None
            if isinstance(body, dict):
                data = body.get("data")
                if isinstance(data, list) and data:
                    task_id = data[0].get("task_id")
                elif isinstance(data, dict):
                    task_id = data.get("task_id")
                task_id = task_id or body.get("task_id")

            # 生成回执字符串
            try:
                receipt = json.dumps(body, ensure_ascii=False)
            except Exception:
                receipt = str(body)

            report = f"HTTP {code} | 模型: {model} | AR: {aspect_ratio} | 时长: {duration}s | 使用CDN外链 | 回执: {receipt}"
            if task_id:
                _save_task(task_id)
                report += f" | task_id: {task_id} 已保存"
            else:
                report += " | 未返回task_id"

            return (report, task_id or "")

        # 兜底：使用multipart上传
        temp_png = _image_to_temp_png(image, max_side=1024)
        if not temp_png:
            return ("图片外链上传失败且本地PNG生成失败。请检查网络或更换图源。", "")

        form_fields = {
            "model": model,
            "prompt": prompt,
            "duration": duration,
            "aspect_ratio": aspect_ratio,
        }
        file_field = os.getenv("APIMART_FILE_FIELD", "image").strip() or "image"
        code, body = _submit_generation_multipart(temp_png, form_fields, api_key, file_field=file_field)

        task_id = None
        if isinstance(body, dict):
            data = body.get("data")
            if isinstance(data, list) and data:
                task_id = data[0].get("task_id")
            elif isinstance(data, dict):
                task_id = data.get("task_id")
            task_id = task_id or body.get("task_id")

        try:
            receipt = json.dumps(body, ensure_ascii=False)
        except Exception:
            receipt = str(body)

        report = f"HTTP {code} | 模型: {model} | AR: {aspect_ratio} | 时长: {duration}s | 使用multipart上传 | 回执: {receipt}"
        if task_id:
            _save_task(task_id)
            report += f" | task_id: {task_id} 已保存"
        else:
            report += " | 未返回task_id"

        return (report, task_id or "")


class ApimartDownloadSavedTaskVideoZV:
    """下载已保存任务视频类"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
            },
            "optional": {
                "task_id": ("STRING", {"default": ""}),
                "max_retries": ("INT", {"default": 12, "min": 1, "max": 60}),
                "retry_interval": ("INT", {"default": 5, "min": 1, "max": 30}),
            }
        }

    RETURN_TYPES = (IO.VIDEO, "STRING")
    RETURN_NAMES = ("video", "report")
    CATEGORY = CATEGORY
    FUNCTION = "run"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time_ns()

    def run(self, api_key: str, task_id: str = "", max_retries: int = 12, retry_interval: int = 5):
        """下载已保存任务的视频"""
        manual_id = (task_id or "").strip()
        if manual_id:
            selected_task_id = manual_id
        else:
            tasks = _read_task_queue()
            if not tasks:
                return (VideoAdapterZV(None), "没有已保存的任务ID")
            selected_task_id = tasks[0].get("task_id")

        # 添加轮询机制处理401错误
        retry_count = 0
        code = 0
        body = None
        
        while retry_count < max_retries:
            # 查询任务状态
            code, body = _query_task(selected_task_id, api_key)
            if code == 200:
                # 成功获取响应，跳出轮询
                # 提取状态信息
                status = None
                if isinstance(body, dict):
                    status = body.get("status") or body.get("state")
                    data = body.get("data")
                    if isinstance(data, list) and data:
                        status = status or data[0].get("status") or data[0].get("state")
                    elif isinstance(data, dict):
                        status = status or data.get("status") or data.get("state")
                vurl = _extract_video_url(body)
                if not vurl:
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"HTTP 401 - 查询失败，等待 {retry_interval} 秒后重试 ({retry_count}/{max_retries})")
                        time.sleep(retry_interval)
                    else:
                        # 达到最大重试次数
                        return (VideoAdapterZV(None), f"任务未完成或无视频链接 | status={status} | task_id={selected_task_id}")
                else:
                    break
            elif code == 401:
                # HTTP 401错误，等待后继续轮询
                retry_count += 1
                if retry_count < max_retries:
                    print(f"HTTP 401 - 查询失败，等待 {retry_interval} 秒后重试 ({retry_count}/{max_retries})")
                    time.sleep(retry_interval)
                else:
                    # 达到最大重试次数
                    return (VideoAdapterZV(None), 
                           f"HTTP 401 - 查询失败，已达到最大重试次数 ({max_retries}) | task_id: {selected_task_id}")
            else:
                # 其他错误，直接返回
                return (VideoAdapterZV(None), f"HTTP {code} - 查询失败 | task_id: {selected_task_id}")

        # 检查是否成功获取数据
        if code != 200:
            return (VideoAdapterZV(None), f"HTTP {code} - 查询失败 | task_id: {selected_task_id}")

        # 下载到输出目录
        base = folder_paths.get_output_directory()
        out_name = f"apimart_{selected_task_id}.mp4"
        out_path = os.path.join(base, out_name)
        
        # 使用原有的下载重试机制
        ok, msg = _download_with_retries(vurl, out_path, API_CONFIG.get("max_retries", 3))
        
        if not ok:
            return (VideoAdapterZV(None), f"下载失败: {msg} | task_id={selected_task_id}")

        # 下载成功，移除任务
        adapter = VideoAdapterZV(out_path)
        
        # 生成报告
        report_parts = []
        if retry_count > 0:
            report_parts.append(f"重试 {retry_count} 次后成功")
        report_parts.append(f"下载成功 | {out_name} | task_id={selected_task_id}")
        report = " | ".join(report_parts)
        
        _remove_task_by_id(selected_task_id)
        return (adapter, report)

class ApimartRemixVideoSubmitZV:
    """视频Remix提交任务类"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (IO.VIDEO, {}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "api_key": ("STRING", {"default": ""}),
                "duration": ("STRING", {"default": "15", "choices": ["10", "15"]}),
                "model": ("STRING", {"default": "sora-2", "choices": ["sora-2", "sora-2-pro"]}),
            },
            "optional": {
                "video_id": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("task_id",)
    CATEGORY = CATEGORY
    FUNCTION = "submit"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time_ns()

    def submit(self, video, prompt: str, api_key: str, duration: str, model: str, video_id: str = ""):
        """提交视频Remix任务"""
        final_duration = int(duration)
        vid = (video_id or "").strip()

        # 优先：使用video_id
        if vid:
            code, body = _submit_remix_by_video_id(vid, prompt, model, final_duration, api_key, aspect_ratio="16:9")
            task_id = None
            if isinstance(body, dict):
                data = body.get("data")
                if isinstance(data, list) and data:
                    task_id = data[0].get("task_id")
                elif isinstance(data, dict):
                    task_id = data.get("task_id")
                task_id = task_id or body.get("task_id")
            return (task_id or "")

        # 兜底：上传视频到CDN
        cdn_url = _upload_video_to_apimart_cdn(video, api_key)
        if not cdn_url:
            return ("")

        code, body = _submit_remix_by_url(cdn_url, prompt, model, final_duration, api_key)
        task_id = None
        if isinstance(body, dict):
            data = body.get("data")
            if isinstance(data, list) and data:
                task_id = data[0].get("task_id")
            elif isinstance(data, dict):
                task_id = data.get("task_id")
            task_id = task_id or body.get("task_id")

        return (task_id or "")


class ApimartRemixByTaskIdSubmitZV:
    """通过TaskID视频Remix提交类"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "task_id": ("STRING", {"default": ""}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "api_key": ("STRING", {"default": ""}),
                "aspect_ratio": ("STRING", {"default": "16:9", "choices": ["16:9", "9:16"]}),
                "duration": ("STRING", {"default": "15", "choices": ["10", "15"]}),
                "model": ("STRING", {"default": "sora-2", "choices": ["sora-2", "sora-2-pro"]}),
            },
            "optional": {},
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("report", "task_id")
    CATEGORY = CATEGORY
    FUNCTION = "submit"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time_ns()

    def submit(self, task_id: str, prompt: str, api_key: str, aspect_ratio: str, duration: str, model: str):
        """通过TaskID提交视频Remix任务"""
        vid = (task_id or "").strip()
        if not vid:
            return ("未提供有效的task_id", "")

        final_duration = int(duration)
        code, body = _submit_remix_by_video_id(vid, prompt, model, final_duration, api_key, aspect_ratio=aspect_ratio)

        # 提取新任务ID
        new_task_id = None
        if isinstance(body, dict):
            data = body.get("data")
            if isinstance(data, list) and data:
                new_task_id = data[0].get("task_id")
            elif isinstance(data, dict):
                new_task_id = data.get("task_id")
            new_task_id = new_task_id or body.get("task_id")

        report = f"HTTP {code} - 提交成功" if code == 200 else f"HTTP {code} - 提交失败"
        if new_task_id:
            _save_task(new_task_id)
            report += f" | task_id: {new_task_id} 已保存"
        else:
            msg = body.get("message") if isinstance(body, dict) else None
            if msg:
                report += f" | {msg}"
            else:
                report += " | 未返回task_id"

        return (report, new_task_id or "")

class ApimartSeedream40ImageSubmitZV:
    """Seedream 4.0 图像生成提交节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "可爱的熊猫在竹林中玩耍"}),
                "api_key": ("STRING", {"default": ""}),
                "size": ("STRING", {"default": "1:1", "choices": ["1:1", "4:3", "3:4", "16:9", "9:16", "3:2", "2:3", "21:9"]}),
                "n": ("INT", {"default": 1, "min": 1, "max": 15, "step": 1}),
                "model": ("STRING", {"default": "doubao-seedance-4-0", "choices": ["doubao-seedance-4-0"]}),
                "optimize_prompt_options": ("STRING", {"default": "standard", "choices": ["standard", "fast"]}),
                "watermark": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image": ("IMAGE", {}),
                "sequential_image_generation": ("STRING", {"default": "disabled", "choices": ["disabled", "auto"]}),
                "max_images": ("INT", {"default": 3, "min": 1, "max": 15, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("report", "task_id")
    CATEGORY = CATEGORY
    FUNCTION = "submit"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time_ns()

    def submit(self, prompt: str, api_key: str, size: str, n: int, model: str,
               optimize_prompt_options: str, watermark: bool,
               image=None, sequential_image_generation: str = "disabled", max_images: int = 3):
        
        # 构建基础payload
        payload = {
            "model": model,
            "prompt": prompt[:1000],  # 限制1000字符
            "size": size,
            "n": n,
            "optimize_prompt_options": optimize_prompt_options,
            "watermark": watermark,
        }
        
        # 处理参考图像
        image_urls = []
        if image is not None:
            # 上传图像到CDN获取URL
            cdn_url = _upload_image_to_apimart_cdn(image, api_key, max_side=1024)
            if cdn_url:
                image_urls.append(cdn_url)
        
        if image_urls:
            payload["image_urls"] = image_urls
        
        # 处理组图生成
        if sequential_image_generation == "auto":
            payload["sequential_image_generation"] = "auto"
            payload["sequential_image_generation_options"] = {
                "max_images": max_images
            }
        
        # 提交生成任务
        code, body = _submit_image_generation(payload, api_key)
        
        # 提取task_id
        task_id = None
        if isinstance(body, dict):
            data = body.get("data")
            if isinstance(data, list) and data:
                task_id = data[0].get("task_id")
            elif isinstance(data, dict):
                task_id = data.get("task_id")
            task_id = task_id or body.get("task_id")
        
        report = f"HTTP {code} - 提交{'成功' if code == 200 else '失败'} | 模型: {model} | 尺寸: {size} | 数量: {n}"
        if task_id:
            _save_task(task_id)
            report += f" | task_id: {task_id} 已保存"
        else:
            error_msg = ""
            if isinstance(body, dict) and body.get("error"):
                error = body.get("error", {})
                error_msg = f" | 错误: {error.get('message', '未知错误')}"
            report += f" | 未返回 task_id{error_msg}"
        
        return (report, task_id or "")

class ApimartSeedream45ImageSubmitZV:
    """Seedream 4.5 图像生成提交节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "可爱的熊猫在竹林中玩耍"}),
                "api_key": ("STRING", {"default": ""}),
                "size": ("STRING", {"default": "1:1", "choices": ["1:1", "4:3", "3:4", "16:9", "9:16", "3:2", "2:3", "21:9", "2k", "4k"]}),
                "n": ("INT", {"default": 1, "min": 1, "max": 15, "step": 1}),
                "model": ("STRING", {"default": "doubao-seedance-4-5", "choices": ["doubao-seedance-4-5"]}),
                "optimize_prompt_options_mode": ("STRING", {"default": "standard", "choices": ["standard", "fast"]}),
                "watermark": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image": ("IMAGE", {}),
                "sequential_image_generation": ("STRING", {"default": "disabled", "choices": ["disabled", "auto"]}),
                "max_images": ("INT", {"default": 3, "min": 1, "max": 15, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("report", "task_id")
    CATEGORY = CATEGORY
    FUNCTION = "submit"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time_ns()

    def submit(self, prompt: str, api_key: str, size: str, n: int, model: str,
               optimize_prompt_options_mode: str, watermark: bool,
               image=None, sequential_image_generation: str = "disabled", max_images: int = 3):
        
        # 构建基础payload
        payload = {
            "model": model,
            "prompt": prompt,  # 4.5没有字符限制，但保留合理长度
            "size": size,
            "n": n,
            "optimize_prompt_options": {
                "mode": optimize_prompt_options_mode
            },
            "watermark": watermark,
        }
        
        # 处理参考图像
        image_urls = []
        if image is not None:
            # 上传图像到CDN获取URL
            cdn_url = _upload_image_to_apimart_cdn(image, api_key, max_side=1024)
            if cdn_url:
                image_urls.append(cdn_url)
        
        if image_urls:
            payload["image_urls"] = image_urls
        
        # 处理组图生成
        if sequential_image_generation == "auto":
            payload["sequential_image_generation"] = "auto"
            payload["sequential_image_generation_options"] = {
                "max_images": max_images
            }
        
        # 提交生成任务
        code, body = _submit_image_generation(payload, api_key)
        
        # 提取task_id
        task_id = None
        if isinstance(body, dict):
            data = body.get("data")
            if isinstance(data, list) and data:
                task_id = data[0].get("task_id")
            elif isinstance(data, dict):
                task_id = data.get("task_id")
            task_id = task_id or body.get("task_id")
        
        report = f"HTTP {code} - 提交{'成功' if code == 200 else '失败'} | 模型: {model} | 尺寸: {size} | 数量: {n}"
        if task_id:
            _save_task(task_id)
            report += f" | task_id: {task_id} 已保存"
        else:
            error_msg = ""
            if isinstance(body, dict) and body.get("error"):
                error = body.get("error", {})
                error_msg = f" | 错误: {error.get('message', '未知错误')}"
            report += f" | 未返回 task_id{error_msg}"
        
        return (report, task_id or "")
    
class ApimartNanoBananaProImageSubmitZV:
    """NanoBananaPro (Gemini-3-Pro-Image-preview) 图像生成提交节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "月光下的竹林小径"}),
                "api_key": ("STRING", {"default": ""}),
                "size": ("STRING", {"default": "1:1", "choices": ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]}),
                "resolution": ("STRING", {"default": "1K", "choices": ["1K", "2K", "4K"]}),
                "model": ("STRING", {"default": "gemini-3-pro-image-preview", "choices": ["gemini-3-pro-image-preview"]}),
            },
            "optional": {
                "image": ("IMAGE", {}),
                "mask": ("IMAGE", {}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("report", "task_id")
    CATEGORY = CATEGORY
    FUNCTION = "submit"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time_ns()

    def submit(self, prompt: str, api_key: str, size: str, resolution: str, model: str,
               image=None, mask=None):
        
        # 构建基础payload - NanoBananaPro固定n=1
        payload = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "n": 1,  # 固定为1
            "resolution": resolution,
        }
        
        # 处理参考图像
        image_urls = []
        if image is not None:
            # 上传图像到CDN获取URL
            cdn_url = _upload_image_to_apimart_cdn(image, api_key, max_side=1024)
            if cdn_url:
                image_urls.append(cdn_url)
        
        if image_urls:
            payload["image_urls"] = image_urls
        
        # 处理蒙版图像
        if mask is not None:
            # 蒙版图像必须是PNG格式
            mask_temp_file = _image_to_temp_file(mask, max_side=1024, format="PNG")
            if mask_temp_file:
                try:
                    # 上传蒙版图像到CDN
                    headers = _headers(api_key)
                    token_res = requests.post(
                        "https://grsai.dakka.com.cn/client/resource/newUploadTokenZH",
                        headers=headers,
                        json={"sux": "png"},
                        timeout=30,
                    )
                    if token_res.status_code == 200:
                        try:
                            token_data = token_res.json().get("data") or {}
                            token = token_data.get("token")
                            key = token_data.get("key")
                            up_url = token_data.get("url")
                            domain = token_data.get("domain")
                            
                            if token and key and up_url and domain:
                                with open(mask_temp_file, "rb") as f:
                                    up_resp = requests.post(up_url, data={"token": token, "key": key}, files={"file": f}, timeout=120)
                                if up_resp.status_code in (200, 201):
                                    mask_url = f"{domain}/{key}"
                                    if mask_url.startswith("http"):
                                        payload["mask_url"] = mask_url
                        except Exception:
                            pass
                finally:
                    try:
                        if mask_temp_file and os.path.exists(mask_temp_file):
                            os.unlink(mask_temp_file)
                    except Exception:
                        pass
        
        # 提交生成任务
        code, body = _submit_image_generation(payload, api_key)
        
        # 提取task_id
        task_id = None
        if isinstance(body, dict):
            data = body.get("data")
            if isinstance(data, list) and data:
                task_id = data[0].get("task_id")
            elif isinstance(data, dict):
                task_id = data.get("task_id")
            task_id = task_id or body.get("task_id")
        
        report = f"HTTP {code} - 提交{'成功' if code == 200 else '失败'} | 模型: {model} | 尺寸: {size} | 分辨率: {resolution}"
        if task_id:
            _save_task(task_id)
            report += f" | task_id: {task_id} 已保存"
        else:
            error_msg = ""
            if isinstance(body, dict) and body.get("error"):
                error = body.get("error", {})
                error_msg = f" | 错误: {error.get('message', '未知错误')}"
            report += f" | 未返回 task_id{error_msg}"
        
        return (report, task_id or "")

class ApimartDownloadSavedTaskImageZV:
    """下载已保存的图像生成任务结果"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
            },
            "optional": {
                "task_id": ("STRING", {"default": ""}),
                "max_retries": ("INT", {"default": 12, "min": 1, "max": 60}),
                "retry_interval": ("INT", {"default": 5, "min": 1, "max": 30}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "report")
    CATEGORY = CATEGORY
    FUNCTION = "run"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time_ns()

    def run(self, api_key: str, task_id: str = "", max_retries: int = 12, retry_interval: int = 5):
        manual_id = (task_id or "").strip()
        if manual_id:
            selected_task_id = manual_id
        else:
            tasks = _read_task_queue()
            if not tasks:
                # 返回空批次和报告
                try:
                    import torch
                    empty_batch = torch.zeros((0, 3, 512, 512))
                except Exception:
                    empty_batch = None
                return (empty_batch, "没有已保存的任务ID")
            selected_task_id = tasks[0].get("task_id")

        # 添加轮询机制处理401错误
        retry_count = 0
        while retry_count < max_retries:
            # 查询任务状态
            code, body = _query_task(selected_task_id, api_key)

            if code == 200:
                # 成功获取响应，跳出轮询
                image_urls = _extract_image_urls(body)
                if not image_urls:
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"HTTP 401 - 查询失败，等待 {retry_interval} 秒后重试 ({retry_count}/{max_retries})")
                        time.sleep(retry_interval)
                    else:
                        # 检查任务状态
                        status = None
                        if isinstance(body, dict):
                            status = body.get("status") or body.get("state")
                            data = body.get("data")
                            if isinstance(data, list) and data:
                                status = status or data[0].get("status") or data[0].get("state")
                            elif isinstance(data, dict):
                                status = status or data.get("status") or data.get("state")
                        
                        try:
                            import torch
                            empty_batch = torch.zeros((0, 3, 512, 512))
                        except Exception:
                            empty_batch = None
                        return (empty_batch, f"任务未完成或无图像链接 | status={status} | task_id={selected_task_id}")
                else:
                    break
            elif code == 401:
                # HTTP 401错误，等待后继续轮询
                retry_count += 1
                if retry_count < max_retries:
                    print(f"HTTP 401 - 查询失败，等待 {retry_interval} 秒后重试 ({retry_count}/{max_retries})")
                    time.sleep(retry_interval)
                else:
                    # 达到最大重试次数
                    try:
                        import torch
                        empty_batch = torch.zeros((0, 3, 512, 512))
                    except Exception:
                        empty_batch = None
                    return (empty_batch, f"HTTP 401 - 查询失败，已达到最大重试次数 ({max_retries}) | task_id: {selected_task_id}")
            else:
                # 其他错误，直接返回
                try:
                    import torch
                    empty_batch = torch.zeros((0, 3, 512, 512))
                except Exception:
                    empty_batch = None
                return (empty_batch, f"HTTP {code} - 查询失败 | task_id: {selected_task_id}")

        # 检查是否成功获取数据
        if code != 200:
            try:
                import torch
                empty_batch = torch.zeros((0, 3, 512, 512))
            except Exception:
                empty_batch = None
            return (empty_batch, f"HTTP {code} - 查询失败 | task_id: {selected_task_id}")
                    

        # 下载所有图像并加载为tensor
        downloaded_images = []
        download_dir = os.path.join(folder_paths.get_output_directory(), "apimart_images")
        os.makedirs(download_dir, exist_ok=True)
        
        success_count = 0
        failed_urls = []
        
        for i, url in enumerate(image_urls):
            try:
                # 生成文件名
                filename = f"apimart_{selected_task_id}_{i+1}.png"
                filepath = os.path.join(download_dir, filename)
                
                # 下载图像
                if _download_image(url, filepath):
                    # 加载图像为tensor
                    tensor = _load_image_to_tensor(filepath)
                    if tensor is not None:
                        downloaded_images.append(tensor)
                        success_count += 1
                    else:
                        failed_urls.append(f"图像{i+1}加载失败")
                else:
                    failed_urls.append(f"图像{i+1}下载失败")
            except Exception as e:
                failed_urls.append(f"图像{i+1}处理失败: {str(e)}")
        
        # 组合所有图像为一个批次
        if downloaded_images:
            try:
                import torch
                # 将所有tensor堆叠为一个批次
                batch = torch.cat(downloaded_images, dim=0)
            except Exception as e:
                try:
                    import torch
                    batch = torch.zeros((0, 3, 512, 512))
                except Exception:
                    batch = None
                return (batch, f"批次组合失败: {str(e)} | task_id={selected_task_id}")
        else:
            try:
                import torch
                batch = torch.zeros((0, 3, 512, 512))
            except Exception:
                batch = None
        
        # 生成报告
        report_parts = []
        if retry_count > 0:
            report_parts.append(f"重试 {retry_count} 次后成功")
        report_parts.append(f"下载完成 | task_id={selected_task_id}")
        report_parts.append(f"成功: {success_count}/{len(image_urls)}")
        
        if failed_urls:
            report_parts.append(f"失败: {', '.join(failed_urls[:3])}")
            if len(failed_urls) > 3:
                report_parts.append(f"...等{len(failed_urls)}个失败")
        
        report = " | ".join(report_parts)
        
        # 下载成功后删除该任务
        if success_count > 0:
            _remove_task_by_id(selected_task_id)
        
        return (batch, report)


# ==================== 节点注册 ====================

NODE_CLASS_MAPPINGS = {
    "ApimartText2VideoSubmitZV": ApimartText2VideoSubmitZV,
    "ApimartImage2VideoSubmitZV": ApimartImage2VideoSubmitZV,
    "ApimartDownloadSavedTaskVideoZV": ApimartDownloadSavedTaskVideoZV,
    "ApimartRemixVideoSubmitZV": ApimartRemixVideoSubmitZV,
    "ApimartRemixByTaskIdSubmitZV": ApimartRemixByTaskIdSubmitZV,
    "Veo31Image2VideoSubmitZV": Veo31Image2VideoSubmitZV,
    "ApimartSeedream40ImageSubmitZV": ApimartSeedream40ImageSubmitZV,
    "ApimartSeedream45ImageSubmitZV": ApimartSeedream45ImageSubmitZV,
    "ApimartNanoBananaProImageSubmitZV": ApimartNanoBananaProImageSubmitZV,
    "ApimartDownloadSavedTaskImageZV": ApimartDownloadSavedTaskImageZV,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApimartText2VideoSubmitZV": "文生视频提交任务 ZV",
    "ApimartImage2VideoSubmitZV": "图生视频提交任务 ZV",
    "ApimartDownloadSavedTaskVideoZV": "下载已保存任务视频 ZV",
    "ApimartRemixVideoSubmitZV": "视频 Remix 提交任务 ZV",
    "ApimartRemixByTaskIdSubmitZV": "通过 TaskID 视频 Remix 提交 ZV",
    "Veo31Image2VideoSubmitZV": "veo3.1 图生视频提交任务 ZV",
    "ApimartSeedream40ImageSubmitZV": "Seedream 4.0 图像生成 ZV",
    "ApimartSeedream45ImageSubmitZV": "Seedream 4.5 图像生成 ZV",
    "ApimartNanoBananaProImageSubmitZV": "NanoBananaPro 图像生成 ZV",
    "ApimartDownloadSavedTaskImageZV": "下载已保存任务图像 ZV",
}