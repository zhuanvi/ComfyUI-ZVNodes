import os
import json
import time
import requests
import shutil
import re
import base64
import io
import tempfile
import threading
import queue
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from comfy.comfy_types import IO
import folder_paths
from PIL import Image


# 分类名称
CATEGORY = "ZVNodes/apimart"

# 配置文件路径
NODE_DIR = os.path.dirname(__file__)
TASK_FILE = os.path.join(NODE_DIR, "apimart_task_history.json")
CONFIG_FILE = os.path.join(NODE_DIR, "apimart_config.json")

# API配置
API_CONFIG = {
    "video_base_url": "https://api.apimart.ai/v1/videos/generations",
    "image_base_url": "https://api.apimart.ai/v1/images/generations",
    "status_url": "https://api.apimart.ai/v1/tasks",
    "upload_token_url": "https://grsai.dakka.com.cn/client/resource/newUploadTokenZH",
    "api_key": os.getenv("APIMART_API_KEY", ""),
    "timeout": 30,
    "max_retries": 3,
    "retry_delay": 2,
    "poll_interval": 5,
    "max_poll_time": 300,
}


# ==================== 配置管理 ====================
def _load_config():
    """加载配置"""
    default_config = {
        "proxy": "",
        "download_chunk_size": 8192,
        "max_download_threads": 4,
        "download_timeout": 60,
        "enable_cache": True,
        "cache_dir": os.path.join(NODE_DIR, "cache"),
        "use_multipart_download": True
    }
    
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                user_config = json.load(f)
                default_config.update(user_config)
    except Exception:
        pass
    
    return default_config

def _save_config(config):
    """保存配置"""
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# 加载配置
CONFIG = _load_config()

# ==================== 代理管理 ====================
def _get_proxies(proxy_url: str = "") -> Dict[str, str]:
    """获取代理配置"""
    if not proxy_url:
        proxy_url = CONFIG.get("proxy", "")
    
    if not proxy_url:
        return {}
    
    # 确保代理URL格式正确
    proxy_url = proxy_url.strip()
    if not proxy_url.startswith(('http://', 'https://', 'socks5://')):
        proxy_url = f"http://{proxy_url}"
    
    return {
        "http": proxy_url,
        "https": proxy_url
    }

def _get_session(proxy_url: str = ""):
    """获取带代理的session"""
    session = requests.Session()
    proxies = _get_proxies(proxy_url)
    if proxies:
        session.proxies.update(proxies)
    
    # 设置默认headers
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Connection": "keep-alive",
    })
    
    # 设置超时和重试
    session.timeout = CONFIG.get("download_timeout", 60)
    
    return session


# ==================== 任务管理函数 ====================
def _ensure_task_file():
    """确保任务文件存在"""
    if not os.path.exists(TASK_FILE):
        with open(TASK_FILE, "w", encoding="utf-8") as f:
            json.dump({"tasks": []}, f, ensure_ascii=False, indent=2)

def _read_task_queue() -> List[Dict[str, Any]]:
    """读取任务队列"""
    _ensure_task_file()
    try:
        with open(TASK_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("tasks", [])
    except Exception:
        return []

def _trim_task_list(tasks: List[Dict[str, Any]], max_keep: int = 100) -> List[Dict[str, Any]]:
    """修剪任务列表，保留最新的任务"""
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
        json.dump({"tasks": tasks, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")}, f, ensure_ascii=False, indent=2)

def _save_task(task_id: str, task_type: str = "video", model: str = "", prompt: str = ""):
    """保存任务到历史记录"""
    tasks = _read_task_queue()
    tasks.append({
        "task_id": task_id,
        "task_type": task_type,
        "model": model,
        "prompt": prompt[:200],  # 只保存前200个字符
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "submitted"
    })
    tasks = _trim_task_list(tasks)
    _write_task_queue(tasks)

def _remove_task_by_id(task_id: str):
    """根据任务ID移除任务"""
    tasks = _read_task_queue()
    new_tasks = [t for t in tasks if t.get("task_id") != task_id]
    _write_task_queue(new_tasks)

def _update_task_status(task_id: str, status: str, result_url: str = ""):
    """更新任务状态"""
    tasks = _read_task_queue()
    for task in tasks:
        if task.get("task_id") == task_id:
            task["status"] = status
            if result_url:
                task["result_url"] = result_url
            task["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            break
    _write_task_queue(tasks)


# ==================== HTTP辅助函数 ====================
def _headers(api_key: Optional[str] = None) -> Dict[str, str]:
    """生成请求头"""
    key = (api_key or "").strip() or API_CONFIG.get("api_key")
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0"
    }

def _sanitize_url(s: str) -> Optional[str]:
    """清理URL字符串"""
    if not isinstance(s, str):
        return None
    s2 = s.strip().strip("`\"'")
    return s2 if s2.startswith("http") else None

def _extract_url_from_response(obj: Any, is_image: bool = False) -> Optional[str]:
    """从API响应中提取URL（支持视频和图片）"""
    if isinstance(obj, str):
        return _sanitize_url(obj)
    
    if isinstance(obj, dict):
        # 尝试常见字段
        for key in ("url", "video_url", "image_url", "result_url", "download_url"):
            v = obj.get(key)
            if isinstance(v, str):
                sv = _sanitize_url(v)
                if sv:
                    return sv
        
        # 尝试data字段
        data = obj.get("data")
        if isinstance(data, list) and data:
            for item in data:
                if isinstance(item, dict):
                    # 检查status是否为success/completed
                    status = item.get("status", "").lower()
                    if status in ("success", "completed", "finished"):
                        for key in ("url", "video_url", "image_url"):
                            v = item.get(key)
                            if isinstance(v, str):
                                sv = _sanitize_url(v)
                                if sv:
                                    return sv
        
        # 尝试output字段
        output = obj.get("output")
        if isinstance(output, dict):
            for key in ("url", "video_url", "image_url"):
                v = output.get(key)
                if isinstance(v, str):
                    sv = _sanitize_url(v)
                    if sv:
                        return sv
        
        # 尝试results字段
        results = obj.get("results")
        if isinstance(results, list) and results:
            for item in results:
                if isinstance(item, dict):
                    for key in ("url", "video_url", "image_url"):
                        v = item.get(key)
                        if isinstance(v, str):
                            sv = _sanitize_url(v)
                            if sv:
                                return sv
    
    if isinstance(obj, list) and obj:
        for item in obj:
            url = _extract_url_from_response(item, is_image)
            if url:
                return url
    
    return None

def _extract_task_status(obj: Any) -> Tuple[str, str]:
    """从API响应中提取任务状态和消息"""
    status = "unknown"
    message = ""
    
    if isinstance(obj, dict):
        # 尝试提取状态
        for key in ("status", "state"):
            if key in obj:
                status = str(obj[key]).lower()
                break
        
        # 尝试从data中提取状态
        data = obj.get("data")
        if isinstance(data, list) and data:
            for item in data:
                if isinstance(item, dict):
                    for key in ("status", "state"):
                        if key in item:
                            status = str(item[key]).lower()
                            break
        
        # 提取消息
        for key in ("message", "msg", "error"):
            if key in obj:
                message = str(obj[key])
                break
        
        # 如果是错误响应
        if "error" in obj:
            error_obj = obj["error"]
            if isinstance(error_obj, dict):
                message = error_obj.get("message", str(error_obj))
            else:
                message = str(error_obj)
    
    return status, message

def _query_task(task_id: str, api_key: Optional[str], proxy_url: str = "") -> Tuple[int, Dict[str, Any]]:
    """查询任务状态"""
    url = f"{API_CONFIG.get('status_url')}/{task_id}"
    timeout = API_CONFIG.get("timeout", 30)
    try:
        session = _get_session(proxy_url)
        resp = session.get(url, headers=_headers(api_key), timeout=timeout)
        status_code = resp.status_code
        try:
            body = resp.json()
        except Exception:
            body = {"raw": resp.text}
        return status_code, body
    except Exception as e:
        return 0, {"error": str(e), "raw": ""}


# ==================== 优化下载函数 ====================
def _download_file_simple(url: str, target_path: str, proxy_url: str = "", 
                          chunk_size: int = 8192, max_retries: int = 3) -> Tuple[bool, str]:
    """简单下载函数（带代理）"""
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    for attempt in range(1, max_retries + 1):
        try:
            session = _get_session(proxy_url)
            
            with session.get(url, stream=True, timeout=CONFIG.get("download_timeout", 60)) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                
                with open(target_path, 'wb') as f:
                    downloaded = 0
                    start_time = time.time()
                    
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            # 计算下载速度
                            elapsed = time.time() - start_time
                            if elapsed > 0:
                                speed = downloaded / elapsed / 1024  # KB/s
                                if attempt == 1 and downloaded % (chunk_size * 100) == 0:
                                    print(f"下载进度: {downloaded}/{total_size if total_size > 0 else '未知'} bytes, "
                                          f"速度: {speed:.2f} KB/s")
            
            return True, f"下载成功: {target_path}"
        except Exception as e:
            if attempt == max_retries:
                return False, f"下载失败: {e}"
            time.sleep(2 ** attempt)  # 指数退避
    
    return False, "下载失败"

def _download_file_multipart(url: str, target_path: str, proxy_url: str = "", 
                             num_threads: int = 4, chunk_size: int = 1024 * 1024) -> Tuple[bool, str]:
    """多线程分片下载函数（带代理）"""
    try:
        session = _get_session(proxy_url)
        
        # 获取文件大小
        response = session.head(url, timeout=10)
        response.raise_for_status()
        
        if 'Content-Length' not in response.headers:
            # 如果不支持分片下载，使用简单下载
            return _download_file_simple(url, target_path, proxy_url)
        
        file_size = int(response.headers['Content-Length'])
        
        # 计算每个线程下载的字节范围
        chunk_size = min(chunk_size, file_size // num_threads)
        chunks = []
        start = 0
        
        while start < file_size:
            end = min(start + chunk_size - 1, file_size - 1)
            chunks.append((start, end))
            start = end + 1
        
        # 如果chunks数量大于线程数，重新分配
        if len(chunks) > num_threads:
            chunk_size = file_size // num_threads
            chunks = []
            start = 0
            for i in range(num_threads):
                end = file_size - 1 if i == num_threads - 1 else start + chunk_size - 1
                chunks.append((start, end))
                start = end + 1
        
        # 创建下载队列
        download_queue = queue.Queue()
        for chunk_id, (start, end) in enumerate(chunks):
            download_queue.put((chunk_id, start, end))
        
        # 创建临时文件目录
        temp_dir = os.path.join(os.path.dirname(target_path), f"temp_{os.path.basename(target_path)}")
        os.makedirs(temp_dir, exist_ok=True)
        
        # 下载状态
        downloaded_chunks = [False] * len(chunks)
        lock = threading.Lock()
        
        def download_worker(worker_id):
            while True:
                try:
                    chunk_id, start, end = download_queue.get_nowait()
                except queue.Empty:
                    break
                
                chunk_file = os.path.join(temp_dir, f"chunk_{chunk_id}.tmp")
                
                try:
                    headers = {'Range': f'bytes={start}-{end}'}
                    session = _get_session(proxy_url)
                    
                    with session.get(url, headers=headers, stream=True, timeout=60) as r:
                        r.raise_for_status()
                        
                        with open(chunk_file, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                    
                    with lock:
                        downloaded_chunks[chunk_id] = True
                        completed = sum(downloaded_chunks)
                        print(f"线程 {worker_id}: 分片 {chunk_id} 下载完成 ({completed}/{len(chunks)})")
                
                except Exception as e:
                    print(f"线程 {worker_id}: 分片 {chunk_id} 下载失败: {e}")
                    # 重新放入队列重试
                    download_queue.put((chunk_id, start, end))
                
                finally:
                    download_queue.task_done()
        
        # 启动下载线程
        threads = []
        for i in range(min(num_threads, len(chunks))):
            thread = threading.Thread(target=download_worker, args=(i,))
            thread.daemon = True
            thread.start()
            threads.append(thread)
        
        # 等待所有分片下载完成
        download_queue.join()
        
        # 合并分片
        with open(target_path, 'wb') as outfile:
            for chunk_id in range(len(chunks)):
                chunk_file = os.path.join(temp_dir, f"chunk_{chunk_id}.tmp")
                if os.path.exists(chunk_file):
                    with open(chunk_file, 'rb') as infile:
                        outfile.write(infile.read())
                    os.remove(chunk_file)
        
        # 清理临时目录
        try:
            os.rmdir(temp_dir)
        except Exception:
            pass
        
        return True, f"多线程下载成功: {target_path}"
        
    except Exception as e:
        return False, f"多线程下载失败: {e}"

def _download_with_retries(url: str, target_path: str, proxy_url: str = "", 
                          max_retries: int = 3, use_multipart: bool = True) -> Tuple[bool, str]:
    """带重试的下载函数（支持代理和多线程）"""
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    # 检查是否支持多线程下载
    if use_multipart and CONFIG.get("use_multipart_download", True):
        try:
            # 先尝试多线程下载
            success, msg = _download_file_multipart(
                url, target_path, proxy_url,
                num_threads=CONFIG.get("max_download_threads", 4),
                chunk_size=CONFIG.get("download_chunk_size", 8192) * 128  # 扩大块大小
            )
            if success:
                return True, msg
        except Exception as e:
            print(f"多线程下载失败，回退到单线程: {e}")
    
    # 回退到简单下载
    backoff = 5.0
    
    for attempt in range(1, max_retries + 1):
        success, msg = _download_file_simple(
            url, target_path, proxy_url,
            chunk_size=CONFIG.get("download_chunk_size", 8192)
        )
        
        if success:
            return True, msg
        
        if attempt == max_retries:
            return False, f"下载失败（尝试{attempt}次）: {msg}"
        
        time.sleep(backoff)
        backoff *= 2
    
    return False, "下载失败"


# ==================== 图像处理函数 ====================
def _tensor_to_pil(img_any: Any):
    """将张量或数组转换为PIL图像"""
    if Image is None:
        return None
    
    try:
        # 如果是torch.Tensor
        if torch is not None and isinstance(img_any, torch.Tensor):
            t = img_any
            if t.dim() == 4:  # [B, H, W, C]
                t = t[0]
            t = t.detach().cpu().clamp(0, 1)
            arr = (t.numpy() * 255).astype("uint8")
            if arr.shape[-1] == 3:
                return Image.fromarray(arr, "RGB")
            elif arr.shape[-1] == 4:
                return Image.fromarray(arr, "RGBA")
        
        # 如果是numpy数组
        if np is not None and isinstance(img_any, np.ndarray):
            arr = img_any
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 1)
                arr = (arr * 255).astype(np.uint8)
            if arr.shape[-1] == 3:
                return Image.fromarray(arr, "RGB")
            elif arr.shape[-1] == 4:
                return Image.fromarray(arr, "RGBA")
        
        # 如果是PIL图像
        if hasattr(img_any, "save"):
            return img_any
            
        # 如果对象有to_pil方法
        if hasattr(img_any, "to_pil"):
            return img_any.to_pil()
            
        # 如果是字典结构
        if isinstance(img_any, dict):
            candidate = img_any.get("image") or (img_any.get("images")[0] if img_any.get("images") else None)
            if candidate:
                if hasattr(candidate, "save"):
                    return candidate
                elif hasattr(candidate, "to_pil"):
                    return candidate.to_pil()
                else:
                    return _tensor_to_pil(candidate)
                    
    except Exception as e:
        print(f"Error converting tensor to PIL: {e}")
    
    return None

def _image_to_temp_png(image: Any, max_side: int = 1024) -> Optional[str]:
    """将图像转换为临时PNG文件"""
    pil_img = _tensor_to_pil(image)
    if pil_img is None:
        return None
    
    try:
        # 转换为RGB
        if pil_img.mode not in ("RGB", "RGBA") and Image is not None:
            pil_img = pil_img.convert("RGB")
        
        # 调整大小
        if Image is not None:
            w, h = pil_img.size
            if max(w, h) > max_side:
                ratio = max_side / max(w, h)
                new_w = int(w * ratio)
                new_h = int(h * ratio)
                pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # 保存到临时文件
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        pil_img.save(tmp, "PNG", optimize=True)
        tmp.close()
        return tmp.name
    except Exception as e:
        print(f"Error creating temp PNG: {e}")
        return None

def _upload_image_to_cdn(image: Any, api_key: Optional[str], max_side: int = 1024) -> Optional[str]:
    """上传图像到CDN"""
    temp_png = _image_to_temp_png(image, max_side)
    if not temp_png:
        return None
    
    try:
        headers = _headers(api_key)
        session = _get_session()  # 使用默认session
        
        # 获取上传令牌
        token_res = session.post(
            API_CONFIG.get("upload_token_url"),
            headers=headers,
            json={"sux": "png"},
            timeout=30
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
        
        # 上传文件
        with open(temp_png, "rb") as f:
            up_resp = session.post(
                up_url,
                data={"token": token, "key": key},
                files={"file": f},
                timeout=120
            )
        
        if up_resp.status_code not in (200, 201):
            return None
        
        # 构造公开URL
        public_url = f"{domain}/{key}"
        return public_url if public_url.startswith("http") else None
    except Exception as e:
        print(f"Error uploading to CDN: {e}")
        return None
    finally:
        try:
            if temp_png and os.path.exists(temp_png):
                os.unlink(temp_png)
        except Exception:
            pass


# ==================== 视频处理函数 ====================
def _video_to_temp_file(video: Any) -> Optional[str]:
    """将视频转换为临时文件"""
    try:
        # 如果是路径字符串
        if isinstance(video, str) and os.path.exists(video):
            ext = os.path.splitext(video)[1].lower() or ".mp4"
            tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
            tmp.close()
            shutil.copyfile(video, tmp.name)
            return tmp.name
        
        # 如果有path属性
        path = getattr(video, "path", None)
        if isinstance(path, str) and os.path.exists(path):
            ext = os.path.splitext(path)[1].lower() or ".mp4"
            tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
            tmp.close()
            shutil.copyfile(path, tmp.name)
            return tmp.name
        
        # 如果有save_to方法
        if hasattr(video, "save_to"):
            tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            tmp.close()
            try:
                if video.save_to(tmp.name):
                    return tmp.name
            except Exception:
                pass
            try:
                os.unlink(tmp.name)
            except Exception:
                pass
    except Exception:
        pass
    return None

def _upload_video_to_cdn(video: Any, api_key: Optional[str]) -> Optional[str]:
    """上传视频到CDN"""
    temp_video = _video_to_temp_file(video)
    if not temp_video:
        return None
    
    try:
        headers = _headers(api_key)
        session = _get_session()
        ext = os.path.splitext(temp_video)[1].lower().lstrip(".") or "mp4"
        
        # 获取上传令牌
        token_res = session.post(
            API_CONFIG.get("upload_token_url"),
            headers=headers,
            json={"sux": ext},
            timeout=30
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
        
        # 上传文件
        with open(temp_video, "rb") as f:
            up_resp = session.post(
                up_url,
                data={"token": token, "key": key},
                files={"file": f},
                timeout=300
            )
        
        if up_resp.status_code not in (200, 201):
            return None
        
        # 构造公开URL
        public_url = f"{domain}/{key}"
        return public_url if public_url.startswith("http") else None
    except Exception as e:
        print(f"Error uploading video to CDN: {e}")
        return None
    finally:
        try:
            if temp_video and os.path.exists(temp_video):
                os.unlink(temp_video)
        except Exception:
            pass


# ==================== API调用函数 ====================
def _submit_image_generation(payload: Dict[str, Any], api_key: Optional[str]) -> Tuple[int, Dict[str, Any]]:
    """提交图像生成任务"""
    url = API_CONFIG.get("image_base_url")
    timeout = API_CONFIG.get("timeout", 30)
    try:
        session = _get_session()
        resp = session.post(url, headers=_headers(api_key), json=payload, timeout=timeout)
        status_code = resp.status_code
        try:
            body = resp.json()
        except Exception:
            body = {"raw": resp.text}
        return status_code, body
    except Exception as e:
        return 0, {"error": str(e), "raw": ""}

def _submit_video_generation(payload: Dict[str, Any], api_key: Optional[str]) -> Tuple[int, Dict[str, Any]]:
    """提交视频生成任务"""
    url = API_CONFIG.get("video_base_url")
    timeout = API_CONFIG.get("timeout", 30)
    try:
        session = _get_session()
        resp = session.post(url, headers=_headers(api_key), json=payload, timeout=timeout)
        status_code = resp.status_code
        try:
            body = resp.json()
        except Exception:
            body = {"raw": resp.text}
        return status_code, body
    except Exception as e:
        return 0, {"error": str(e), "raw": ""}


# ==================== VideoAdapter类 ====================
class VideoAdapterZV:
    def __init__(self, path: Optional[str]):
        self.path = path
        self.width, self.height, self.fps = self._get_video_details(path)
    
    def _get_video_details(self, path: Optional[str]):
        try:
            if not path or not os.path.exists(path):
                return 1280, 720, 30
            
            # 尝试使用OpenCV获取视频信息
            try:
                import cv2
                cap = cv2.VideoCapture(path)
                if cap.isOpened():
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    cap.release()
                    if fps and fps > 0:
                        return width or 1280, height or 720, int(fps)
            except Exception:
                pass
            
            return 1280, 720, 30
        except Exception:
            return 1280, 720, 30
    
    def __repr__(self):
        return f"<VideoAdapterZV path={self.path} {self.width}x{self.height}@{self.fps}>"
    
    def get_dimensions(self):
        return self.width, self.height
    
    def save_to(self, output_path, **kwargs):
        try:
            if self.path and os.path.exists(self.path):
                shutil.copyfile(self.path, output_path)
                return True
            return False
        except Exception:
            return False


# ==================== 新增：SeeDream 4.0 图像生成节点 ====================
class SeeDream40Text2ImageSubmitZV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A beautiful landscape"}),
                "api_key": ("STRING", {"default": ""}),
                "size": (["1:1", "16:9", "9:16", "4:3", "3:4"], {"default": "1:1"}),
                "n": ("INT", {"default": 1, "min": 1, "max": 4}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("report", "task_id")
    CATEGORY = CATEGORY
    FUNCTION = "submit"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time_ns()
    
    def submit(self, prompt: str, api_key: str, size: str, n: int):
        payload = {
            "model": "doubao-seedance-4-0",
            "prompt": prompt,
            "size": size,
            "n": n
        }
        
        code, body = _submit_image_generation(payload, api_key)
        
        # 提取task_id
        task_id = None
        if isinstance(body, dict):
            if body.get("code") == 200:
                data = body.get("data")
                if isinstance(data, list) and data:
                    task_id = data[0].get("task_id")
        
        report = f"SeeDream 4.0 - HTTP {code} - 提交{'成功' if task_id else '失败'}"
        if task_id:
            _save_task(task_id, "image", "doubao-seedance-4-0", prompt)
            report += f" | task_id: {task_id}"
        else:
            report += f" | 响应: {json.dumps(body, ensure_ascii=False)[:200]}"
        
        return (report, task_id or "")


# ==================== 新增：SeeDream 4.5 图像生成节点 ====================
class SeeDream45Text2ImageSubmitZV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A beautiful landscape"}),
                "api_key": ("STRING", {"default": ""}),
                "size": (["1:1", "16:9", "9:16", "4:3", "3:4"], {"default": "1:1"}),
                "n": ("INT", {"default": 1, "min": 1, "max": 4}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("report", "task_id")
    CATEGORY = CATEGORY
    FUNCTION = "submit"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time_ns()
    
    def submit(self, prompt: str, api_key: str, size: str, n: int):
        payload = {
            "model": "doubao-seedance-4-5",
            "prompt": prompt,
            "size": size,
            "n": n
        }
        
        code, body = _submit_image_generation(payload, api_key)
        
        # 提取task_id
        task_id = None
        if isinstance(body, dict):
            if body.get("code") == 200:
                data = body.get("data")
                if isinstance(data, list) and data:
                    task_id = data[0].get("task_id")
        
        report = f"SeeDream 4.5 - HTTP {code} - 提交{'成功' if task_id else '失败'}"
        if task_id:
            _save_task(task_id, "image", "doubao-seedance-4-5", prompt)
            report += f" | task_id: {task_id}"
        else:
            report += f" | 响应: {json.dumps(body, ensure_ascii=False)[:200]}"
        
        return (report, task_id or "")


# ==================== 新增：Nano Banana Pro 图像生成节点 ====================
class NanoBananaProText2ImageSubmitZV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A beautiful landscape"}),
                "api_key": ("STRING", {"default": ""}),
                "size": (["1:1", "16:9", "9:16", "4:3", "3:4"], {"default": "1:1"}),
                "n": ("INT", {"default": 1, "min": 1, "max": 4}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("report", "task_id")
    CATEGORY = CATEGORY
    FUNCTION = "submit"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time_ns()
    
    def submit(self, prompt: str, api_key: str, size: str, n: int):
        payload = {
            "model": "gemini-3-pro-image-preview",
            "prompt": prompt,
            "size": size,
            "n": n
        }
        
        code, body = _submit_image_generation(payload, api_key)
        
        # 提取task_id
        task_id = None
        if isinstance(body, dict):
            if body.get("code") == 200:
                data = body.get("data")
                if isinstance(data, list) and data:
                    task_id = data[0].get("task_id")
        
        report = f"Nano Banana Pro - HTTP {code} - 提交{'成功' if task_id else '失败'}"
        if task_id:
            _save_task(task_id, "image", "gemini-3-pro-image-preview", prompt)
            report += f" | task_id: {task_id}"
        else:
            report += f" | 响应: {json.dumps(body, ensure_ascii=False)[:200]}"
        
        return (report, task_id or "")

# ==================== 原有的视频生成节点（保持兼容） ====================
class ApimartText2VideoSubmitZV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "api_key": ("STRING", {"default": ""}),
                "aspect_ratio": (["16:9", "9:16", "1:1", "4:3", "3:4"], {"default": "16:9"}),
                "duration": (["5", "10", "15", "20", "30"], {"default": "10"}),
                "model": (["sora-2", "sora-2-pro", "veo3.1-fast", "veo3.1-quality"], {"default": "sora-2"}),
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
        payload = {
            "model": model,
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "duration": int(duration),
        }
        
        code, body = _submit_video_generation(payload, api_key)
        
        # 提取task_id
        task_id = None
        if isinstance(body, dict):
            data = body.get("data")
            if isinstance(data, list) and data:
                task_id = data[0].get("task_id")
            elif isinstance(data, dict):
                task_id = data.get("task_id")
            task_id = task_id or body.get("task_id")
        
        report = f"HTTP {code} - 提交{'成功' if code == 200 else '失败'}"
        if task_id:
            _save_task(task_id, "video", model, prompt)
            report += f" | task_id: {task_id}"
        else:
            report += f" | 响应: {json.dumps(body, ensure_ascii=False)[:200]}"
        
        return (report, task_id or "")


class ApimartImage2VideoSubmitZV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "api_key": ("STRING", {"default": ""}),
                "aspect_ratio": (["16:9", "9:16", "1:1", "4:3", "3:4"], {"default": "16:9"}),
                "duration": (["5", "10", "15", "20", "30"], {"default": "10"}),
                "model": (["sora-2", "sora-2-pro", "veo3.1-fast", "veo3.1-quality"], {"default": "sora-2"}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("report", "task_id")
    CATEGORY = CATEGORY
    FUNCTION = "submit"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time_ns()
    
    def submit(self, image, prompt: str, api_key: str, aspect_ratio: str, duration: str, model: str):
        # 上传图片到CDN
        image_url = _upload_image_to_cdn(image, api_key)
        
        if not image_url:
            return ("错误: 图片上传失败", "")
        
        payload = {
            "model": model,
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "duration": int(duration),
            "image_urls": [image_url],
        }
        
        code, body = _submit_video_generation(payload, api_key)
        
        # 如果使用image_urls失败，尝试使用url字段
        if code != 200:
            payload = {
                "model": model,
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "duration": int(duration),
                "url": image_url,
            }
            code, body = _submit_video_generation(payload, api_key)
        
        # 提取task_id
        task_id = None
        if isinstance(body, dict):
            data = body.get("data")
            if isinstance(data, list) and data:
                task_id = data[0].get("task_id")
            elif isinstance(data, dict):
                task_id = data.get("task_id")
            task_id = task_id or body.get("task_id")
        
        report = f"HTTP {code} - 提交{'成功' if code == 200 else '失败'}"
        if task_id:
            _save_task(task_id, "video", model, prompt)
            report += f" | task_id: {task_id}"
        else:
            report += f" | 响应: {json.dumps(body, ensure_ascii=False)[:200]}"
        
        return (report, task_id or "")


# ==================== 下载任务视频节点（独立实现） ====================
class ApimartDownloadSavedTaskVideoZV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
                "proxy_url": ("STRING", {"default": "", "multiline": False}),
                "enable_multipart_download": ("BOOLEAN", {"default": True}),
                "max_retries": ("INT", {"default": 3, "min": 1, "max": 10}),
                "download_timeout": ("INT", {"default": 60, "min": 10, "max": 300}),
            },
            "optional": {
                "task_id": ("STRING", {"default": ""}),
                "output_filename": ("STRING", {"default": ""}),
                "check_status": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING")
    RETURN_NAMES = ("video", "report", "file_path")
    CATEGORY = CATEGORY
    FUNCTION = "run"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time_ns()
    
    def run(self, api_key: str, proxy_url: str, enable_multipart_download: bool,
            max_retries: int, download_timeout: int, task_id: str = "",
            output_filename: str = "", check_status: bool = True):
        
        # 确定任务ID
        manual_id = (task_id or "").strip()
        if manual_id:
            selected_task_id = manual_id
        else:
            tasks = _read_task_queue()
            if not tasks:
                return (VideoAdapterZV(None), "没有已保存的任务ID", "")
            
            # 优先查找视频任务
            video_tasks = [t for t in tasks if t.get("task_type") in ("video", "video_remix")]
            if video_tasks:
                selected_task_id = video_tasks[0].get("task_id")
            else:
                # 如果没有视频任务，使用第一个任务
                selected_task_id = tasks[0].get("task_id")
        
        print(f"开始处理视频任务: {selected_task_id}")
        
        # 如果需要检查状态，先查询任务状态
        if check_status:
            code, body = _query_task(selected_task_id, api_key, proxy_url)
            
            if code != 200:
                return (VideoAdapterZV(None), f"查询任务状态失败: HTTP {code}", "")
            
            # 提取状态
            status, message = _extract_task_status(body)
            
            # 检查任务是否完成
            if status not in ("success", "completed", "finished", "succeeded"):
                return (VideoAdapterZV(None), 
                       f"任务未完成，状态: {status}{', 消息: ' + message if message else ''}", "")
        
        # 如果需要检查状态但已经完成，或者不检查状态，直接下载
        # 首先获取视频URL
        if check_status:
            # 如果已经检查过状态，从响应中提取URL
            code, body = _query_task(selected_task_id, api_key, proxy_url)
            if code != 200:
                return (VideoAdapterZV(None), f"获取任务信息失败: HTTP {code}", "")
            
            video_url = _extract_url_from_response(body)
        else:
            # 如果不检查状态，尝试从历史记录中获取URL
            tasks = _read_task_queue()
            video_url = None
            for task in tasks:
                if task.get("task_id") == selected_task_id and task.get("result_url"):
                    video_url = task.get("result_url")
                    break
            
            # 如果没有保存的URL，需要查询任务
            if not video_url:
                code, body = _query_task(selected_task_id, api_key, proxy_url)
                if code != 200:
                    return (VideoAdapterZV(None), f"获取任务信息失败: HTTP {code}", "")
                
                video_url = _extract_url_from_response(body)
        
        if not video_url:
            return (VideoAdapterZV(None), f"未找到视频URL，任务可能仍在处理中", "")
        
        print(f"找到视频URL: {video_url}")
        
        # 准备输出路径
        output_dir = folder_paths.get_output_directory()
        
        if output_filename and output_filename.strip():
            filename = output_filename.strip()
            if not filename.lower().endswith(('.mp4', '.mov', '.avi', '.webm', '.mkv')):
                filename += '.mp4'
        else:
            timestamp = int(time.time())
            filename = f"apimart_video_{selected_task_id}_{timestamp}.mp4"
        
        output_path = os.path.join(output_dir, filename)
        
        # 避免文件名冲突
        counter = 1
        while os.path.exists(output_path):
            base, ext = os.path.splitext(filename)
            new_filename = f"{base}_{counter}{ext}"
            output_path = os.path.join(output_dir, new_filename)
            counter += 1
        
        # 下载视频
        print(f"开始下载视频到: {output_path}")
        start_time = time.time()
        
        ok, msg = _download_with_retries(
            video_url, output_path, proxy_url,
            max_retries=max_retries,
            use_multipart=enable_multipart_download
        )
        
        download_time = time.time() - start_time
        
        if not ok:
            return (VideoAdapterZV(None), f"视频下载失败: {msg}", "")
        
        # 获取文件大小
        file_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
        file_size_mb = file_size / (1024 * 1024)
        
        # 更新任务状态
        _update_task_status(selected_task_id, "downloaded", video_url)
        
        # 创建视频适配器
        video_adapter = VideoAdapterZV(output_path)
        
        report = f"视频下载成功!\n"
        report += f"• 任务ID: {selected_task_id}\n"
        report += f"• 文件: {os.path.basename(output_path)}\n"
        report += f"• 大小: {file_size_mb:.2f} MB\n"
        report += f"• 下载时间: {download_time:.1f} 秒\n"
        report += f"• 平均速度: {file_size_mb/download_time:.2f} MB/s" if download_time > 0 else "• 速度: 计算中"
        
        if proxy_url:
            report += f"\n• 使用代理: {proxy_url}"
        
        return (video_adapter, report, output_path)


# ==================== 下载任务图像节点（独立实现） ====================
class ApimartDownloadSavedTaskImageZV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
                "proxy_url": ("STRING", {"default": "", "multiline": False}),
                "enable_multipart_download": ("BOOLEAN", {"default": False}),
                "image_quality": (["original", "high", "medium", "low"], {"default": "original"}),
                "max_retries": ("INT", {"default": 3, "min": 1, "max": 10}),
                "download_timeout": ("INT", {"default": 60, "min": 10, "max": 300}),
            },
            "optional": {
                "task_id": ("STRING", {"default": ""}),
                "output_filename": ("STRING", {"default": ""}),
                "output_format": (["png", "jpg", "webp"], {"default": "png"}),
                "jpeg_quality": ("INT", {"default": 95, "min": 1, "max": 100}),
                "check_status": ("BOOLEAN", {"default": True}),
                "resize_width": ("INT", {"default": 0, "min": 0, "max": 8192}),
                "resize_height": ("INT", {"default": 0, "min": 0, "max": 8192}),
                "preserve_aspect_ratio": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING")
    RETURN_NAMES = ("image", "mask", "report", "file_path")
    CATEGORY = CATEGORY
    FUNCTION = "run"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time_ns()
    
    def run(self, api_key: str, proxy_url: str, enable_multipart_download: bool,
            image_quality: str, max_retries: int, download_timeout: int,
            task_id: str = "", output_filename: str = "", output_format: str = "png",
            jpeg_quality: int = 95, check_status: bool = True,
            resize_width: int = 0, resize_height: int = 0, preserve_aspect_ratio: bool = True):
        
        # 确定任务ID
        manual_id = (task_id or "").strip()
        if manual_id:
            selected_task_id = manual_id
        else:
            tasks = _read_task_queue()
            if not tasks:
                # 创建空的返回
                empty_image, empty_mask = self._create_empty_output()
                return (empty_image, empty_mask, "没有已保存的任务ID", "")
            
            # 优先查找图像任务
            image_tasks = [t for t in tasks if t.get("task_type") == "image"]
            if image_tasks:
                selected_task_id = image_tasks[0].get("task_id")
            else:
                # 如果没有图像任务，使用第一个任务
                selected_task_id = tasks[0].get("task_id")
        
        print(f"开始处理图像任务: {selected_task_id}")
        
        # 如果需要检查状态，先查询任务状态
        if check_status:
            code, body = _query_task(selected_task_id, api_key, proxy_url)
            
            if code != 200:
                empty_image, empty_mask = self._create_empty_output()
                return (empty_image, empty_mask, f"查询任务状态失败: HTTP {code}", "")
            
            # 提取状态
            status, message = _extract_task_status(body)
            
            # 检查任务是否完成
            if status not in ("success", "completed", "finished", "succeeded"):
                empty_image, empty_mask = self._create_empty_output()
                return (empty_image, empty_mask, 
                       f"任务未完成，状态: {status}{', 消息: ' + message if message else ''}", "")
        
        # 获取图像URL
        if check_status:
            # 如果已经检查过状态，从响应中提取URL
            code, body = _query_task(selected_task_id, api_key, proxy_url)
            if code != 200:
                empty_image, empty_mask = self._create_empty_output()
                return (empty_image, empty_mask, f"获取任务信息失败: HTTP {code}", "")
            
            image_url = _extract_url_from_response(body, is_image=True)
        else:
            # 如果不检查状态，尝试从历史记录中获取URL
            tasks = _read_task_queue()
            image_url = None
            for task in tasks:
                if task.get("task_id") == selected_task_id and task.get("result_url"):
                    image_url = task.get("result_url")
                    break
            
            # 如果没有保存的URL，需要查询任务
            if not image_url:
                code, body = _query_task(selected_task_id, api_key, proxy_url)
                if code != 200:
                    empty_image, empty_mask = self._create_empty_output()
                    return (empty_image, empty_mask, f"获取任务信息失败: HTTP {code}", "")
                
                image_url = _extract_url_from_response(body, is_image=True)
        
        if not image_url:
            empty_image, empty_mask = self._create_empty_output()
            return (empty_image, empty_mask, f"未找到图像URL，任务可能仍在处理中", "")
        
        print(f"找到图像URL: {image_url}")
        
        # 创建临时文件下载
        import tempfile
        temp_dir = tempfile.mkdtemp(prefix="apimart_download_")
        temp_filename = f"temp_{selected_task_id}.{output_format}"
        temp_path = os.path.join(temp_dir, temp_filename)
        
        # 下载图像
        print(f"开始下载图像到临时文件: {temp_path}")
        start_time = time.time()
        
        ok, msg = _download_with_retries(
            image_url, temp_path, proxy_url,
            max_retries=max_retries,
            use_multipart=enable_multipart_download
        )
        
        download_time = time.time() - start_time
        
        if not ok:
            # 清理临时目录
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass
            
            empty_image, empty_mask = self._create_empty_output()
            return (empty_image, empty_mask, f"图像下载失败: {msg}", "")
        
        # 检查文件是否成功下载
        if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass
            
            empty_image, empty_mask = self._create_empty_output()
            return (empty_image, empty_mask, "下载的文件为空或不存在", "")
        
        # 处理图像
        processed_image, processed_mask, process_report = self._process_downloaded_image(
            temp_path, image_quality, output_format, jpeg_quality,
            resize_width, resize_height, preserve_aspect_ratio
        )
        
        if processed_image is None:
            # 清理临时目录
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass
            
            empty_image, empty_mask = self._create_empty_output()
            return (empty_image, empty_mask, f"图像处理失败: {process_report}", "")
        
        # 准备最终输出路径
        output_dir = folder_paths.get_output_directory()
        
        if output_filename and output_filename.strip():
            filename = output_filename.strip()
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                filename += f'.{output_format}'
        else:
            timestamp = int(time.time())
            filename = f"apimart_image_{selected_task_id}_{timestamp}.{output_format}"
        
        final_path = os.path.join(output_dir, filename)
        
        # 避免文件名冲突
        counter = 1
        while os.path.exists(final_path):
            base, ext = os.path.splitext(filename)
            new_filename = f"{base}_{counter}{ext}"
            final_path = os.path.join(output_dir, new_filename)
            counter += 1
        
        # 保存处理后的图像
        try:
            from PIL import Image as PILImage
            
            if torch is not None:
                img_array = (processed_image[0].cpu().numpy() * 255).astype(np.uint8)
            else:
                img_array = (processed_image[0] * 255).astype(np.uint8)
            
            pil_img = PILImage.fromarray(img_array, 'RGB')
            
            # 根据格式保存
            if output_format.lower() == 'jpg' or output_format.lower() == 'jpeg':
                pil_img.save(final_path, "JPEG", quality=jpeg_quality, optimize=True)
            elif output_format.lower() == 'webp':
                pil_img.save(final_path, "WEBP", quality=jpeg_quality)
            else:  # PNG
                pil_img.save(final_path, "PNG", optimize=True)
            
            print(f"图像已保存到: {final_path}")
            
        except Exception as e:
            print(f"保存图像失败: {e}")
            # 仍然返回处理后的图像，只是保存失败
        
        finally:
            # 清理临时目录
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass
        
        # 获取文件大小
        file_size = os.path.getsize(final_path) if os.path.exists(final_path) else 0
        file_size_mb = file_size / (1024 * 1024)
        
        # 更新任务状态
        _update_task_status(selected_task_id, "downloaded", image_url)
        
        # 生成报告
        report = f"图像下载成功!\n"
        report += f"• 任务ID: {selected_task_id}\n"
        report += f"• 文件: {os.path.basename(final_path)}\n"
        report += f"• 格式: {output_format.upper()}\n"
        report += f"• 大小: {file_size_mb:.2f} MB\n"
        report += f"• 下载时间: {download_time:.1f} 秒\n"
        
        if resize_width > 0 or resize_height > 0:
            if torch is not None:
                h, w = processed_image.shape[1:3]
            else:
                h, w = processed_image.shape[1:3]
            report += f"• 尺寸: {w}x{h}\n"
        
        if process_report:
            report += f"• 处理: {process_report}\n"
        
        if proxy_url:
            report += f"• 使用代理: {proxy_url}"
        
        return (processed_image, processed_mask, report, final_path)
    
    def _create_empty_output(self):
        """创建空的图像和掩码输出"""
        if torch is not None:
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
        else:
            empty_image = np.zeros((1, 64, 64, 3), dtype=np.float32)
            empty_mask = np.zeros((1, 64, 64), dtype=np.float32)
        return empty_image, empty_mask
    
    def _process_downloaded_image(self, image_path, image_quality, output_format, 
                                 jpeg_quality, resize_width, resize_height, 
                                 preserve_aspect_ratio):
        """处理下载的图像"""
        try:
            from PIL import Image as PILImage
            
            # 打开图像
            pil_img = PILImage.open(image_path)
            
            # 转换为RGB
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            
            # 获取原始尺寸
            orig_width, orig_height = pil_img.size
            
            # 根据质量设置调整尺寸
            target_width, target_height = orig_width, orig_height
            
            if image_quality != "original":
                if image_quality == "high":
                    max_size = 2048
                elif image_quality == "medium":
                    max_size = 1024
                elif image_quality == "low":
                    max_size = 512
                
                if max(orig_width, orig_height) > max_size:
                    ratio = max_size / max(orig_width, orig_height)
                    target_width = int(orig_width * ratio)
                    target_height = int(orig_height * ratio)
            
            # 应用手动调整尺寸
            if resize_width > 0 or resize_height > 0:
                if resize_width > 0 and resize_height > 0:
                    target_width, target_height = resize_width, resize_height
                elif resize_width > 0:
                    if preserve_aspect_ratio:
                        ratio = resize_width / target_width
                        target_height = int(target_height * ratio)
                    target_width = resize_width
                elif resize_height > 0:
                    if preserve_aspect_ratio:
                        ratio = resize_height / target_height
                        target_width = int(target_width * ratio)
                    target_height = resize_height
            
            # 如果需要调整大小
            if target_width != orig_width or target_height != orig_height:
                # 使用高质量重采样
                pil_img = pil_img.resize((target_width, target_height), 
                                        PILImage.Resampling.LANCZOS)
            
            # 转换为numpy数组
            img_array = np.array(pil_img).astype(np.float32) / 255.0
            
            # 添加批次维度
            img_array = img_array[np.newaxis, ...]
            
            # 创建掩码（全白掩码）
            mask_array = np.ones((1, img_array.shape[1], img_array.shape[2]), dtype=np.float32)
            
            # 转换为torch张量（如果可用）
            if torch is not None:
                image_tensor = torch.from_numpy(img_array)
                mask_tensor = torch.from_numpy(mask_array)
            else:
                image_tensor = img_array
                mask_tensor = mask_array
            
            # 生成处理报告
            report = f"尺寸: {target_width}x{target_height}"
            if target_width != orig_width or target_height != orig_height:
                report += f" (原始: {orig_width}x{orig_height})"
            
            return image_tensor, mask_tensor, report
            
        except Exception as e:
            print(f"图像处理错误: {e}")
            return None, None, str(e)


# ==================== 视频Remix节点 ====================
def _submit_remix_by_video_id(video_id: str, prompt: str, model: str, duration: int, 
                            api_key: Optional[str], aspect_ratio: str = "16:9") -> Tuple[int, Dict[str, Any]]:
    """通过视频ID提交Remix任务"""
    url = f"https://api.apimart.ai/v1/videos/{video_id}/remix"
    timeout = API_CONFIG.get("timeout", 30)
    
    payload = {
        "model": model,
        "prompt": prompt,
        "duration": duration,
        "aspect_ratio": aspect_ratio,
    }
    
    try:
        session = _get_session()
        resp = session.post(url, headers=_headers(api_key), json=payload, timeout=timeout)
        status_code = resp.status_code
        try:
            body = resp.json()
        except Exception:
            body = {"raw": resp.text}
        return status_code, body
    except Exception as e:
        return 0, {"error": str(e), "raw": ""}


class ApimartRemixByTaskIdSubmitZV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "task_id": ("STRING", {"default": ""}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "api_key": ("STRING", {"default": ""}),
                "aspect_ratio": (["16:9", "9:16", "1:1", "4:3", "3:4"], {"default": "16:9"}),
                "duration": (["5", "10", "15", "20", "30"], {"default": "15"}),
                "model": (["sora-2", "sora-2-pro", "veo3.1-fast", "veo3.1-quality"], {"default": "sora-2"}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("report", "task_id")
    CATEGORY = CATEGORY
    FUNCTION = "submit"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time_ns()
    
    def submit(self, task_id: str, prompt: str, api_key: str, aspect_ratio: str, duration: str, model: str):
        vid = (task_id or "").strip()
        if not vid:
            return ("错误: 未提供有效的task_id", "")
        
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
        
        report = f"HTTP {code} - 提交{'成功' if code == 200 else '失败'}"
        if new_task_id:
            _save_task(new_task_id, "video_remix", model, prompt)
            report += f" | task_id: {new_task_id} 已保存"
        else:
            # 若服务端返回消息，附加到回执里
            msg = body.get("message") if isinstance(body, dict) else None
            if msg:
                report += f" | {msg}"
            else:
                report += " | 未返回 task_id"
        
        return (report, new_task_id or "")


# ==================== 视频Remix通过视频节点 ====================
def _submit_remix_by_url(video_url: str, prompt: str, model: str, duration: int, 
                        api_key: Optional[str], aspect_ratio: str = "16:9") -> Tuple[int, Dict[str, Any]]:
    """通过视频URL提交Remix任务"""
    url = "https://api.apimart.ai/v1/videos/remix"
    timeout = API_CONFIG.get("timeout", 30)
    
    payload = {
        "model": model,
        "prompt": prompt,
        "duration": duration,
        "aspect_ratio": aspect_ratio,
        "url": video_url,
    }
    
    try:
        session = _get_session()
        resp = session.post(url, headers=_headers(api_key), json=payload, timeout=timeout)
        status_code = resp.status_code
        try:
            body = resp.json()
        except Exception:
            body = {"raw": resp.text}
        return status_code, body
    except Exception as e:
        return 0, {"error": str(e), "raw": ""}


class ApimartRemixVideoSubmitZV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (IO.VIDEO, {}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "api_key": ("STRING", {"default": ""}),
                "aspect_ratio": (["16:9", "9:16", "1:1", "4:3", "3:4"], {"default": "16:9"}),
                "duration": (["5", "10", "15", "20", "30"], {"default": "15"}),
                "model": (["sora-2", "sora-2-pro", "veo3.1-fast", "veo3.1-quality"], {"default": "sora-2"}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("report", "task_id")
    CATEGORY = CATEGORY
    FUNCTION = "submit"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time_ns()
    
    def submit(self, video, prompt: str, api_key: str, aspect_ratio: str, duration: str, model: str):
        final_duration = int(duration)
        
        # 先上传视频到CDN
        video_url = _upload_video_to_cdn(video, api_key)
        if not video_url:
            return ("错误: 视频上传失败", "")
        
        # 通过URL提交Remix
        code, body = _submit_remix_by_url(video_url, prompt, model, final_duration, api_key, aspect_ratio)
        
        # 提取task_id
        task_id = None
        if isinstance(body, dict):
            data = body.get("data")
            if isinstance(data, list) and data:
                task_id = data[0].get("task_id")
            elif isinstance(data, dict):
                task_id = data.get("task_id")
            task_id = task_id or body.get("task_id")
        
        report = f"HTTP {code} - 提交{'成功' if code == 200 else '失败'}"
        if task_id:
            _save_task(task_id, "video_remix", model, prompt)
            report += f" | task_id: {task_id} 已保存"
        else:
            report += " | 未返回 task_id"
        
        return (report, task_id or "")


# ==================== Veo3.1图生视频节点 ====================
class Veo31Image2VideoSubmitZV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "api_key": ("STRING", {"default": ""}),
                "model": (["veo3.1-fast", "veo3.1-quality"], {"default": "veo3.1-fast"}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("report", "task_id")
    CATEGORY = CATEGORY
    FUNCTION = "submit"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time_ns()
    
    def submit(self, image, prompt: str, api_key: str, model: str):
        aspect_ratio = "16:9"
        duration = 8
        
        # 上传图片到CDN
        image_url = _upload_image_to_cdn(image, api_key)
        if not image_url:
            return ("错误: 图片上传失败", "")
        
        payload = {
            "model": model,
            "prompt": prompt,
            "duration": duration,
            "aspect_ratio": aspect_ratio,
            "image_urls": [image_url],
        }
        
        code, body = _submit_video_generation(payload, api_key)
        
        # 如果使用image_urls失败，尝试使用url字段
        if code != 200:
            payload = {
                "model": model,
                "prompt": prompt,
                "duration": duration,
                "aspect_ratio": aspect_ratio,
                "url": image_url,
            }
            code, body = _submit_video_generation(payload, api_key)
        
        # 提取task_id
        task_id = None
        if isinstance(body, dict):
            data = body.get("data")
            if isinstance(data, list) and data:
                task_id = data[0].get("task_id")
            elif isinstance(data, dict):
                task_id = data.get("task_id")
            task_id = task_id or body.get("task_id")
        
        report = f"Veo3.1 - HTTP {code} - 提交{'成功' if code == 200 else '失败'}"
        if task_id:
            _save_task(task_id, "video", model, prompt)
            report += f" | task_id: {task_id}"
        else:
            report += f" | 响应: {json.dumps(body, ensure_ascii=False)[:200]}"
        
        return (report, task_id or "")


# ==================== 配置管理节点 ====================
class ApimartConfigManagerZV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "proxy_url": ("STRING", {"default": "", "multiline": False}),
                "download_chunk_size": ("INT", {"default": 8192, "min": 1024, "max": 65536}),
                "max_download_threads": ("INT", {"default": 4, "min": 1, "max": 16}),
                "download_timeout": ("INT", {"default": 60, "min": 10, "max": 300}),
                "enable_multipart_download": ("BOOLEAN", {"default": True}),
                "save_to_config": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    CATEGORY = CATEGORY
    FUNCTION = "update_config"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time_ns()
    
    def update_config(self, proxy_url: str, download_chunk_size: int, max_download_threads: int,
                     download_timeout: int, enable_multipart_download: bool, save_to_config: bool):
        # 更新配置
        CONFIG.update({
            "proxy": proxy_url,
            "download_chunk_size": download_chunk_size,
            "max_download_threads": max_download_threads,
            "download_timeout": download_timeout,
            "use_multipart_download": enable_multipart_download,
        })
        
        # 如果用户选择保存到配置文件
        if save_to_config:
            _save_config(CONFIG)
        
        report = f"配置已更新: 代理={proxy_url if proxy_url else '未设置'}, "
        report += f"分片大小={download_chunk_size}, "
        report += f"最大线程数={max_download_threads}, "
        report += f"超时={download_timeout}秒, "
        report += f"多线程下载={'启用' if enable_multipart_download else '禁用'}"
        
        if save_to_config:
            report += " | 配置已保存到文件"
        
        return (report,)


# ==================== 节点映射（完整版） ====================
NODE_CLASS_MAPPINGS = {
    # 原有视频生成节点
    "ApimartText2VideoSubmitZV": ApimartText2VideoSubmitZV,
    "ApimartImage2VideoSubmitZV": ApimartImage2VideoSubmitZV,
    "ApimartDownloadSavedTaskVideoZV": ApimartDownloadSavedTaskVideoZV,
    "ApimartDownloadSavedTaskImageZV": ApimartDownloadSavedTaskImageZV,
    
    # 新增图像生成节点
    "SeeDream40Text2ImageSubmitZV": SeeDream40Text2ImageSubmitZV,
    "SeeDream45Text2ImageSubmitZV": SeeDream45Text2ImageSubmitZV,
    "NanoBananaProText2ImageSubmitZV": NanoBananaProText2ImageSubmitZV,
    
    # 视频Remix节点
    "ApimartRemixByTaskIdSubmitZV": ApimartRemixByTaskIdSubmitZV,
    "ApimartRemixVideoSubmitZV": ApimartRemixVideoSubmitZV,
    
    # Veo3.1节点
    "Veo31Image2VideoSubmitZV": Veo31Image2VideoSubmitZV,
    
    # 配置管理节点
    "ApimartConfigManagerZV": ApimartConfigManagerZV,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # 原有视频生成节点
    "ApimartText2VideoSubmitZV": "文生视频提交任务ZV",
    "ApimartImage2VideoSubmitZV": "图生视频提交任务ZV",
    "ApimartDownloadSavedTaskVideoZV": "下载已保存任务视频ZV",
    "ApimartDownloadSavedTaskImageZV": "下载已保存任务图像ZV",
    
    # 新增图像生成节点
    "SeeDream40Text2ImageSubmitZV": "SeeDream 4.0 文生图ZV",
    "SeeDream45Text2ImageSubmitZV": "SeeDream 4.5 文生图ZV",
    "NanoBananaProText2ImageSubmitZV": "Nano Banana Pro 文生图ZV",
    
    # 视频Remix节点
    "ApimartRemixByTaskIdSubmitZV": "通过TaskID视频RemixZV",
    "ApimartRemixVideoSubmitZV": "视频Remix提交任务ZV",
    
    # Veo3.1节点
    "Veo31Image2VideoSubmitZV": "Veo3.1图生视频提交ZV",
    
    # 配置管理节点
    "ApimartConfigManagerZV": "配置管理器ZV",
}