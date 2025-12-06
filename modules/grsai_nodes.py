import json
import time
import requests
import threading
import queue
import os
from PIL import Image
import numpy as np
import io
import base64
import torch
from typing import Dict, List, Optional, Tuple, Any
from comfy.comfy_types import IO
import folder_paths
from .utils import batch_upload_image_to_apimart_cdn, VideoAdapter

class GrsaiSoraVideoNodeZV:
    """Sora视频生成节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"multiline": True, "default": "A cute cat playing on the grass"}),
                "model": (["sora-2"], {"default": "sora-2"}),
                "aspect_ratio": (["9:16", "16:9", "1:1", "4:3", "3:2"], {"default": "9:16"}),
                "duration": ("INT", {"default": 10, "min": 10, "max": 15, "step": 5}),
                "size": (["small", "large"], {"default": "small"}),
                "use_cn_endpoint": ("BOOLEAN", {"default": True}),
                "response_mode": (["stream", "webhook", "polling"], {"default": "stream"}),
                "webhook_url": ("STRING", {"default": "", "multiline": False}),
                "shut_progress": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "reference_image": ("IMAGE",),
                "remix_target_id": ("STRING", {"default": "", "multiline": False}),
                "characters_json": ("STRING", {"multiline": True, "default": "[]"}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("video_url", "task_id", "pid")
    FUNCTION = "generate_video"
    CATEGORY = "Sora Video"
    
    def __init__(self):
        self.base_url_cn = "https://grsai.dakka.com.cn"
        self.base_url_overseas = "https://api.grsai.com"
        
    def image_to_base64(self, image_tensor):
        """将图像张量转换为Base64"""
        if image_tensor is None:
            return None
            
        # 假设image_tensor是形状为[批次, 高度, 宽度, 通道]的张量
        if len(image_tensor.shape) == 4:
            image_tensor = image_tensor[0]  # 取第一张
        
        # 转换为PIL图像
        image_np = image_tensor.cpu().numpy()
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)
        
        image = Image.fromarray(image_np)
        
        # 转换为Base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    
    def generate_video(self, api_key, prompt, model, aspect_ratio, duration, size, 
                      use_cn_endpoint, response_mode, webhook_url, shut_progress,
                      reference_image=None, remix_target_id="", characters_json="[]"):
        
        # 构建请求参数
        base_url = self.base_url_cn if use_cn_endpoint else self.base_url_overseas
        endpoint = f"{base_url}/v1/video/sora-video"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        # 准备请求体
        payload = {
            "model": model,
            "prompt": prompt,
            "aspectRatio": aspect_ratio,
            "duration": duration,
            "size": size,
            "shutProgress": shut_progress
        }
        
        # 处理参考图像
        # if reference_image is not None:
        #     payload["url"] = self.image_to_base64(reference_image)
        # elif reference_image_url:
        #     payload["url"] = reference_image_url

        reference_image_urls = batch_upload_image_to_apimart_cdn(reference_image, api_key, max_side=2048)
        if reference_image_urls:
            payload["url"] = reference_image_urls[0]
        else:
            payload["url"] = self.image_to_base64(reference_image)
            
        # 处理续作ID
        if remix_target_id:
            payload["remixTargetId"] = remix_target_id
            
        # 处理角色JSON
        try:
            characters = json.loads(characters_json)
            if characters:
                payload["characters"] = characters
        except:
            print(f"Warning: Invalid characters JSON: {characters_json}")
            
        # 处理响应模式
        if response_mode == "webhook" and webhook_url:
            payload["webHook"] = webhook_url
        elif response_mode == "polling":
            payload["webHook"] = "-1"
            
        print(f"Sending request to Sora API with payload: {json.dumps(payload, indent=2)}")
        
        try:
            if response_mode == "stream":
                # 流式响应
                response = requests.post(endpoint, json=payload, headers=headers, stream=True)
                
                if response.status_code != 200:
                    error_msg = f"API request failed with status {response.status_code}: {response.text}"
                    print(f"Error: {error_msg}")
                    return ("", "", "")
                
                # 处理流式响应
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            print(f"Sora API progress: {data.get('progress', 0)}% - Status: {data.get('status', 'unknown')}")
                            
                            if data.get('status') == 'succeeded':
                                results = data.get('results', [])
                                if results:
                                    video_url = results[0].get('url', '')
                                    pid = results[0].get('pid', '')
                                    task_id = data.get('id', '')
                                    print(f"Video generated successfully! URL: {video_url}")
                                    return (video_url, task_id, pid)
                                    
                        except json.JSONDecodeError as e:
                            print(f"Failed to parse JSON from stream: {e}")
                            continue
                            
                return ("", "", "")
                
            else:
                # 非流式响应
                response = requests.post(endpoint, json=payload, headers=headers)
                
                if response.status_code != 200:
                    error_msg = f"API request failed with status {response.status_code}: {response.text}"
                    print(f"Error: {error_msg}")
                    return ("", "", "")
                
                result = response.json()
                
                if response_mode == "polling":
                    # 轮询模式，返回任务ID
                    if result.get('code') == 0:
                        task_id = result['data']['id']
                        print(f"Task created successfully. Task ID: {task_id}")
                        return ("", task_id, "")
                    else:
                        print(f"Error creating task: {result.get('msg', 'Unknown error')}")
                        return ("", "", "")
                else:
                    # webhook模式或其他
                    if result.get('status') == 'succeeded':
                        results = result.get('results', [])
                        if results:
                            video_url = results[0].get('url', '')
                            pid = results[0].get('pid', '')
                            task_id = result.get('id', '')
                            print(f"Video generated successfully! URL: {video_url}")
                            return (video_url, task_id, pid)
                            
                return ("", "", "")
                
        except Exception as e:
            print(f"Error in Sora video generation: {str(e)}")
            return ("", "", "")


class GrsaiSoraUploadCharacterNodeZV:
    """上传角色视频节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "character_video_url": ("STRING", {"default": "", "multiline": False}),
                "timestamps": ("STRING", {"default": "0,3", "multiline": False}),
                "use_cn_endpoint": ("BOOLEAN", {"default": True}),
                "response_mode": (["stream", "webhook", "polling"], {"default": "stream"}),
                "webhook_url": ("STRING", {"default": "", "multiline": False}),
                "shut_progress": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("character_id", "task_id")
    FUNCTION = "upload_character"
    CATEGORY = "Sora Video"
    
    def __init__(self):
        self.base_url_cn = "https://grsai.dakka.com.cn"
        self.base_url_overseas = "https://api.grsai.com"
    
    def upload_character(self, api_key, character_video_url, timestamps, 
                        use_cn_endpoint, response_mode, webhook_url, shut_progress):
        
        base_url = self.base_url_cn if use_cn_endpoint else self.base_url_overseas
        endpoint = f"{base_url}/v1/video/sora-upload-character"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "url": character_video_url,
            "timestamps": timestamps,
            "shutProgress": shut_progress
        }
        
        if response_mode == "webhook" and webhook_url:
            payload["webHook"] = webhook_url
        elif response_mode == "polling":
            payload["webHook"] = "-1"
            
        print(f"Uploading character video to Sora API...")
        
        try:
            if response_mode == "stream":
                response = requests.post(endpoint, json=payload, headers=headers, stream=True)
                
                if response.status_code != 200:
                    error_msg = f"API request failed with status {response.status_code}: {response.text}"
                    print(f"Error: {error_msg}")
                    return ("", "")
                
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            print(f"Character upload progress: {data.get('progress', 0)}%")
                            
                            if data.get('status') == 'succeeded':
                                results = data.get('results', [])
                                if results:
                                    character_id = results[0].get('character_id', '')
                                    task_id = data.get('id', '')
                                    print(f"Character uploaded successfully! Character ID: {character_id}")
                                    return (character_id, task_id)
                                    
                        except json.JSONDecodeError:
                            continue
                            
                return ("", "")
                
            else:
                response = requests.post(endpoint, json=payload, headers=headers)
                
                if response.status_code != 200:
                    error_msg = f"API request failed with status {response.status_code}: {response.text}"
                    print(f"Error: {error_msg}")
                    return ("", "")
                
                result = response.json()
                
                if response_mode == "polling":
                    if result.get('code') == 0:
                        task_id = result['data']['id']
                        print(f"Character upload task created. Task ID: {task_id}")
                        return ("", task_id)
                else:
                    if result.get('status') == 'succeeded':
                        results = result.get('results', [])
                        if results:
                            character_id = results[0].get('character_id', '')
                            task_id = result.get('id', '')
                            print(f"Character uploaded successfully! Character ID: {character_id}")
                            return (character_id, task_id)
                            
                return ("", "")
                
        except Exception as e:
            print(f"Error in character upload: {str(e)}")
            return ("", "")


class GrsaiSoraCreateCharacterNodeZV:
    """从原视频创建角色节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "pid": ("STRING", {"default": "", "multiline": False}),
                "timestamps": ("STRING", {"default": "0,3", "multiline": False}),
                "use_cn_endpoint": ("BOOLEAN", {"default": True}),
                "response_mode": (["stream", "webhook", "polling"], {"default": "stream"}),
                "webhook_url": ("STRING", {"default": "", "multiline": False}),
                "shut_progress": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("character_id", "task_id")
    FUNCTION = "create_character"
    CATEGORY = "Sora Video"
    
    def __init__(self):
        self.base_url_cn = "https://grsai.dakka.com.cn"
        self.base_url_overseas = "https://api.grsai.com"
    
    def create_character(self, api_key, pid, timestamps, use_cn_endpoint, 
                        response_mode, webhook_url, shut_progress):
        
        base_url = self.base_url_cn if use_cn_endpoint else self.base_url_overseas
        endpoint = f"{base_url}/v1/video/sora-create-character"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "pid": pid,
            "timestamps": timestamps,
            "shutProgress": shut_progress
        }
        
        if response_mode == "webhook" and webhook_url:
            payload["webHook"] = webhook_url
        elif response_mode == "polling":
            payload["webHook"] = "-1"
            
        print(f"Creating character from video PID: {pid}")
        
        try:
            if response_mode == "stream":
                response = requests.post(endpoint, json=payload, headers=headers, stream=True)
                
                if response.status_code != 200:
                    error_msg = f"API request failed with status {response.status_code}: {response.text}"
                    print(f"Error: {error_msg}")
                    return ("", "")
                
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            print(f"Character creation progress: {data.get('progress', 0)}%")
                            
                            if data.get('status') == 'succeeded':
                                results = data.get('results', [])
                                if results:
                                    character_id = results[0].get('character_id', '')
                                    task_id = data.get('id', '')
                                    print(f"Character created successfully! Character ID: {character_id}")
                                    return (character_id, task_id)
                                    
                        except json.JSONDecodeError:
                            continue
                            
                return ("", "")
                
            else:
                response = requests.post(endpoint, json=payload, headers=headers)
                
                if response.status_code != 200:
                    error_msg = f"API request failed with status {response.status_code}: {response.text}"
                    print(f"Error: {error_msg}")
                    return ("", "")
                
                result = response.json()
                
                if response_mode == "polling":
                    if result.get('code') == 0:
                        task_id = result['data']['id']
                        print(f"Character creation task created. Task ID: {task_id}")
                        return ("", task_id)
                else:
                    if result.get('status') == 'succeeded':
                        results = result.get('results', [])
                        if results:
                            character_id = results[0].get('character_id', '')
                            task_id = result.get('id', '')
                            print(f"Character created successfully! Character ID: {character_id}")
                            return (character_id, task_id)
                            
                return ("", "")
                
        except Exception as e:
            print(f"Error in character creation: {str(e)}")
            return ("", "")



class GrsaiNanoBananaNodeZV:
    """Nano Banana图像生成节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"multiline": True, "default": "A cute cat playing on the grass"}),
                "model": (["nano-banana-fast", "nano-banana", "nano-banana-pro"], {"default": "nano-banana-fast"}),
                "aspect_ratio": (["auto", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "5:4", "4:5", "21:9"], 
                               {"default": "auto"}),
                "image_size": (["1K", "2K", "4K"], {"default": "1K"}),
                "use_cn_endpoint": ("BOOLEAN", {"default": True}),
                "response_mode": (["stream", "webhook", "polling"], {"default": "stream"}),
                "webhook_url": ("STRING", {"default": "", "multiline": False}),
                "shut_progress": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "reference_images": ("IMAGE",),
                # "reference_image_urls": ("STRING", {"default": "", "multiline": False}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("image_url", "task_id", "content")
    FUNCTION = "generate_image"
    CATEGORY = "Nano Banana"
    
    def __init__(self):
        self.base_url_cn = "https://grsai.dakka.com.cn"
        self.base_url_overseas = "https://api.grsai.com"
        
    def image_to_base64(self, image_tensor):
        """将图像张量转换为Base64"""
        if image_tensor is None:
            return None
            
        # 处理多个图像的情况
        if len(image_tensor.shape) == 4:
            base64_images = []
            for i in range(image_tensor.shape[0]):
                img_tensor = image_tensor[i]
                
                # 转换为PIL图像
                image_np = img_tensor.cpu().numpy()
                if image_np.max() <= 1.0:
                    image_np = (image_np * 255).astype(np.uint8)
                else:
                    image_np = image_np.astype(np.uint8)
                
                image = Image.fromarray(image_np)
                
                # 转换为Base64
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                base64_images.append(f"data:image/png;base64,{img_str}")
            
            return base64_images
        else:
            # 单个图像的情况
            image_np = image_tensor.cpu().numpy()
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
            else:
                image_np = image_np.astype(np.uint8)
            
            image = Image.fromarray(image_np)
            
            # 转换为Base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return [f"data:image/png;base64,{img_str}"]
    
    def generate_image(self, api_key, prompt, model, aspect_ratio, image_size, 
                      use_cn_endpoint, response_mode, webhook_url, shut_progress,
                      reference_images=None):
        
        # 构建请求参数
        base_url = self.base_url_cn if use_cn_endpoint else self.base_url_overseas
        endpoint = f"{base_url}/v1/draw/nano-banana"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        # 准备请求体
        payload = {
            "model": model,
            "prompt": prompt,
            "aspectRatio": aspect_ratio,
            "imageSize": image_size,
            "shutProgress": shut_progress
        }
        
        # 处理参考图像
        urls = []
        
        # 处理上传的图像
        # if reference_images is not None:
        #     base64_images = self.image_to_base64(reference_images)
        #     if base64_images:
        #         urls.extend(base64_images)

        reference_image_urls = batch_upload_image_to_apimart_cdn(reference_images, api_key, max_side=2048)

        
        # 处理图像URL
        # if reference_image_urls:
        #     url_list = [url.strip() for url in reference_image_urls.split(",") if url.strip()]
        #     urls.extend(url_list)
        if reference_image_urls:
            urls = reference_image_urls
        else:
            if reference_images is not None:
                base64_images = self.image_to_base64(reference_images)
                if base64_images:
                    urls.extend(base64_images)
            
        if urls:
            payload["urls"] = urls
        
        # 处理响应模式
        if response_mode == "webhook" and webhook_url:
            payload["webHook"] = webhook_url
        elif response_mode == "polling":
            payload["webHook"] = "-1"
            
        print(f"Sending request to Nano Banana API with payload: {json.dumps(payload, indent=2)}")
        
        try:
            if response_mode == "stream":
                # 流式响应
                response = requests.post(endpoint, json=payload, headers=headers, stream=True)
                
                if response.status_code != 200:
                    error_msg = f"API request failed with status {response.status_code}: {response.text}"
                    print(f"Error: {error_msg}")
                    return ("", "", "")
                
                # 处理流式响应
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            print(f"Nano Banana API progress: {data.get('progress', 0)}% - Status: {data.get('status', 'unknown')}")
                            
                            if data.get('status') == 'succeeded':
                                results = data.get('results', [])
                                if results:
                                    image_url = results[0].get('url', '')
                                    content = results[0].get('content', '')
                                    task_id = data.get('id', '')
                                    print(f"Image generated successfully! URL: {image_url}")
                                    return (image_url, task_id, content)
                                    
                        except json.JSONDecodeError as e:
                            print(f"Failed to parse JSON from stream: {e}")
                            continue
                            
                return ("", "", "")
                
            else:
                # 非流式响应
                response = requests.post(endpoint, json=payload, headers=headers)
                
                if response.status_code != 200:
                    error_msg = f"API request failed with status {response.status_code}: {response.text}"
                    print(f"Error: {error_msg}")
                    return ("", "", "")
                
                result = response.json()
                
                if response_mode == "polling":
                    # 轮询模式，返回任务ID
                    if result.get('code') == 0:
                        task_id = result['data']['id']
                        print(f"Task created successfully. Task ID: {task_id}")
                        return ("", task_id, "")
                    else:
                        print(f"Error creating task: {result.get('msg', 'Unknown error')}")
                        return ("", "", "")
                else:
                    # webhook模式或其他
                    if result.get('status') == 'succeeded':
                        results = result.get('results', [])
                        if results:
                            image_url = results[0].get('url', '')
                            content = results[0].get('content', '')
                            task_id = result.get('id', '')
                            print(f"Image generated successfully! URL: {image_url}")
                            return (image_url, task_id, content)
                            
                return ("", "", "")
                
        except Exception as e:
            print(f"Error in Nano Banana image generation: {str(e)}")
            return ("", "", "")


class GrsaiNanoBananaGeminiNodeZV:
    """Nano Banana Gemini格式接口节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"multiline": True, "default": "A cute cat playing on the grass"}),
                "model": (["nano-banana-fast"], {"default": "nano-banana-fast"}),
                "use_cn_endpoint": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "reference_images": ("IMAGE",),
                "reference_image_urls": ("STRING", {"default": "", "multiline": False}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("image_url", "task_id", "content")
    FUNCTION = "generate_image_gemini"
    CATEGORY = "Nano Banana"
    
    def __init__(self):
        self.base_url_cn = "https://grsai.dakka.com.cn"
        self.base_url_overseas = "https://api.grsai.com"
        
    def image_to_base64(self, image_tensor):
        """将图像张量转换为Base64"""
        if image_tensor is None:
            return None
            
        # 处理多个图像的情况
        if len(image_tensor.shape) == 4:
            base64_images = []
            for i in range(image_tensor.shape[0]):
                img_tensor = image_tensor[i]
                
                # 转换为PIL图像
                image_np = img_tensor.cpu().numpy()
                if image_np.max() <= 1.0:
                    image_np = (image_np * 255).astype(np.uint8)
                else:
                    image_np = image_np.astype(np.uint8)
                
                image = Image.fromarray(image_np)
                
                # 转换为Base64
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                base64_images.append(img_str)  # 注意：这里不包含data:image/png;base64,前缀
            
            return base64_images
        else:
            # 单个图像的情况
            image_np = image_tensor.cpu().numpy()
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
            else:
                image_np = image_np.astype(np.uint8)
            
            image = Image.fromarray(image_np)
            
            # 转换为Base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return [img_str]  # 注意：这里不包含data:image/png;base64,前缀
    
    def generate_image_gemini(self, api_key, prompt, model, use_cn_endpoint,
                             reference_images=None, reference_image_urls=""):
        
        # 构建请求参数 - Gemini格式
        base_url = self.base_url_cn if use_cn_endpoint else self.base_url_overseas
        endpoint = f"{base_url}/v1beta/models/{model}:streamGenerateContent"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        # 准备Gemini格式的请求体
        contents = [
            {
                "role": "user",
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
        
        # 处理参考图像
        if reference_images is not None:
            base64_images = self.image_to_base64(reference_images)
            if base64_images:
                for img_base64 in base64_images:
                    contents[0]["parts"].append({
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": img_base64
                        }
                    })
        
        # 处理图像URL
        if reference_image_urls:
            url_list = [url.strip() for url in reference_image_urls.split(",") if url.strip()]
            for url in url_list:
                contents[0]["parts"].append({
                    "inline_data": {
                        "mime_type": "image/jpeg" if url.lower().endswith('.jpg') or url.lower().endswith('.jpeg') else "image/png",
                        "data": self._download_and_encode_image(url)
                    }
                })
        
        payload = {
            "contents": contents,
            "generation_config": {
                "temperature": 0.4,
                "topP": 0.95,
                "topK": 40,
                "maxOutputTokens": 8192,
                "stopSequences": []
            },
            "safety_settings": [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
        }
        
        print(f"Sending request to Nano Banana Gemini API with payload: {json.dumps(payload, indent=2)}")
        
        try:
            # 流式响应
            response = requests.post(endpoint, json=payload, headers=headers, stream=True)
            
            if response.status_code != 200:
                error_msg = f"API request failed with status {response.status_code}: {response.text}"
                print(f"Error: {error_msg}")
                return ("", "", "")
            
            # 处理流式响应
            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        # Gemini格式的响应
                        line_str = line.decode('utf-8')
                        if line_str.startswith('data: '):
                            data_str = line_str[6:]  # 去掉 'data: ' 前缀
                            if data_str.strip():
                                data = json.loads(data_str)
                                
                                # 解析响应
                                if 'candidates' in data and data['candidates']:
                                    candidate = data['candidates'][0]
                                    if 'content' in candidate and 'parts' in candidate['content']:
                                        for part in candidate['content']['parts']:
                                            if 'text' in part:
                                                full_response += part['text']
                        
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        print(f"Failed to parse JSON from stream: {e}")
                        continue
            
            print(f"Nano Banana Gemini API response: {full_response}")
            
            # 注意：Gemini格式的响应可能不包含图像URL，这里返回响应文本作为content
            return ("", "", full_response)
                
        except Exception as e:
            print(f"Error in Nano Banana Gemini image generation: {str(e)}")
            return ("", "", "")
    
    def _download_and_encode_image(self, url):
        """下载图像并编码为Base64"""
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            # 编码为Base64
            img_base64 = base64.b64encode(response.content).decode()
            return img_base64
            
        except Exception as e:
            print(f"Error downloading image from {url}: {str(e)}")
            return ""


class GrsaiResultNodeZV:
    """统一结果获取节点 - 支持Sora视频和Nano Banana图像"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "task_id": ("STRING", {"default": "", "multiline": False}),
                "use_cn_endpoint": ("BOOLEAN", {"default": True}),
                "max_retries": ("INT", {"default": 30, "min": 1, "max": 100}),
                "retry_delay": ("INT", {"default": 5, "min": 1, "max": 30}),
                "download_media": ("BOOLEAN", {"default": True}),
                "auto_detect_type": ("BOOLEAN", {"default": True}),
                "media_type": (["auto", "image", "video"], {"default": "auto"}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "INT", "STRING", "IMAGE", IO.VIDEO)
    RETURN_NAMES = ("url", "task_id", "pid", "content", "progress", "status", "image", "video")
    FUNCTION = "get_result"
    CATEGORY = "Grsai API"
    OUTPUT_NODE = True
    
    def __init__(self):
        self.base_url_cn = "https://grsai.dakka.com.cn"
        self.base_url_overseas = "https://api.grsai.com"
    
    def get_result(self, api_key, task_id, use_cn_endpoint, max_retries, retry_delay,
                  download_media, auto_detect_type, media_type):
        
        base_url = self.base_url_cn if use_cn_endpoint else self.base_url_overseas
        endpoint = f"{base_url}/v1/draw/result"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "id": task_id
        }
        _video = VideoAdapter(None)
        
        print(f"Polling result for task ID: {task_id}")
        
        for attempt in range(max_retries):
            try:
                response = requests.post(endpoint, json=payload, headers=headers)
                
                if response.status_code != 200:
                    print(f"API request failed with status {response.status_code}, attempt {attempt + 1}/{max_retries}")
                    time.sleep(retry_delay)
                    continue
                
                result = response.json()
                
                if result.get('code') == 0:
                    data = result.get('data', {})
                    status = data.get('status', '')
                    progress = data.get('progress', 0)
                    
                    print(f"Progress: {progress}%, Status: {status}")
                    
                    if status == 'succeeded':
                        results = data.get('results', [])
                        if results:
                            url = results[0].get('url', '')
                            pid = results[0].get('pid', '')
                            content = results[0].get('content', '')
                            task_id = data.get('id', '')
                            
                            print(f"Task completed successfully! URL: {url}")
                            
                            # 初始化输出
                            image_tensor = torch.zeros(1, 512, 512, 3)
                            video_path = ""
                            
                            # 检测媒体类型
                            detected_type = self._detect_media_type(results[0], media_type, auto_detect_type)
                            
                            # 如果设置了自动下载，则下载媒体文件
                            if download_media and url:
                                if detected_type == "image":
                                    image_tensor = self._download_and_convert_image(url)
                                elif detected_type == "video":
                                    video_path = self._download_video(url, task_id)
                            
                            if video_path != "":
                                _video = VideoAdapter(video_path)
                                
                            return (url, task_id, pid, content, progress, status, image_tensor, _video)
                        else:
                            return ("", task_id, "", "", progress, status, 
                                   torch.zeros(1, 512, 512, 3), _video)
                            
                    elif status == 'failed':
                        error_msg = data.get('error', data.get('failure_reason', 'Unknown error'))
                        print(f"Task failed: {error_msg}")
                        return ("", task_id, "", "", progress, status, 
                               torch.zeros(1, 512, 512, 3), "")
                        
                    else:
                        # 任务还在进行中，等待后重试
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                            continue
                        else:
                            return ("", task_id, "", "", progress, status, 
                                   torch.zeros(1, 512, 512, 3), "")
                            
                elif result.get('code') == -22:
                    print(f"Task not found, attempt {attempt + 1}/{max_retries}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    else:
                        return ("", task_id, "", "", 0, "not_found", 
                               torch.zeros(1, 512, 512, 3), "")
                        
                else:
                    error_msg = result.get('msg', 'Unknown error')
                    print(f"API error: {error_msg}, attempt {attempt + 1}/{max_retries}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    else:
                        return ("", task_id, "", "", 0, "error", 
                               torch.zeros(1, 512, 512, 3), "")
                        
            except Exception as e:
                print(f"Error polling result, attempt {attempt + 1}/{max_retries}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    return ("", task_id, "", "", 0, "error", 
                           torch.zeros(1, 512, 512, 3), "")
        
        return ("", task_id, "", "", 0, "timeout", 
               torch.zeros(1, 512, 512, 3), "")
    
    def _detect_media_type(self, result_data, media_type, auto_detect):
        """检测媒体类型"""
        if not auto_detect and media_type != "auto":
            return media_type
        
        # 自动检测逻辑
        # 如果有pid字段，通常是视频（Sora）
        if 'pid' in result_data and result_data['pid']:
            return "video"
        
        # 如果有content字段，通常是图像（Nano Banana）
        if 'content' in result_data and result_data['content']:
            return "image"
        
        # 根据URL扩展名判断
        url = result_data.get('url', '')
        if url:
            url_lower = url.lower()
            if any(ext in url_lower for ext in ['.mp4', '.mov', '.avi', '.mkv', '.webm']):
                return "video"
            elif any(ext in url_lower for ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif']):
                return "image"
        
        # 默认返回图像类型
        return "image"
    
    def _download_and_convert_image(self, url):
        """下载图像并转换为ComfyUI Image类型"""
        try:
            print(f"Downloading image from {url}")
            
            # 下载图像
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # 读取图像数据
            image_data = io.BytesIO(response.content)
            image = Image.open(image_data)
            
            # 转换为RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 转换为numpy数组
            image_np = np.array(image).astype(np.float32) / 255.0
            
            # 转换为torch张量
            image_tensor = torch.from_numpy(image_np)[None,]
            
            print(f"Image downloaded and converted successfully")
            return image_tensor
            
        except Exception as e:
            print(f"Error downloading or converting image: {str(e)}")
            return torch.zeros(1, 512, 512, 3)
    
    def _download_video(self, url, task_id):
        """下载视频到本地"""
        try:
            # 获取输出目录
            output_dir = folder_paths.get_output_directory()
            video_filename = f"sora_video_{task_id}.mp4"
            video_path = os.path.join(output_dir, video_filename)
            
            print(f"Downloading video from {url}")
            
            # 下载视频
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # 保存视频
            with open(video_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"Video downloaded successfully: {video_path}")
            return video_path
            
        except Exception as e:
            print(f"Error downloading video: {str(e)}")
            return ""


class GrsaiLoadImageFromPathNodeZV:
    """从路径加载图像节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_path": ("STRING", {"default": "", "multiline": False}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "load_image"
    CATEGORY = "Nano Banana"
    
    def load_image(self, image_path):
        if not image_path or not os.path.exists(image_path):
            print(f"Image path does not exist: {image_path}")
            return (torch.zeros(1, 512, 512, 3),)
        
        try:
            # 加载图像
            image = Image.open(image_path)
            
            # 转换为RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 转换为numpy数组
            image_np = np.array(image).astype(np.float32) / 255.0
            
            # 转换为torch张量
            image_tensor = torch.from_numpy(image_np)[None,]
            
            print(f"Image loaded successfully: {image_path}")
            return (image_tensor,)
            
        except Exception as e:
            print(f"Error loading image: {str(e)}")
            return (torch.zeros(1, 512, 512, 3),)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "GrsaiSoraVideoNodeZV": GrsaiSoraVideoNodeZV,
    "GrsaiSoraUploadCharacterNodeZV": GrsaiSoraUploadCharacterNodeZV,
    "GrsaiSoraCreateCharacterNodeZV": GrsaiSoraCreateCharacterNodeZV,
    "GrsaiResultNodeZV": GrsaiResultNodeZV,
    "GrsaiNanoBananaNodeZV": GrsaiNanoBananaNodeZV,
    "GrsaiNanoBananaGeminiNodeZV": GrsaiNanoBananaGeminiNodeZV,
    "GrsaiLoadImageFromPathNodeZV": GrsaiLoadImageFromPathNodeZV,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GrsaiSoraVideoNodeZV": "Grsai Sora Video Generator",
    "GrsaiSoraUploadCharacterNodeZV": "Grsai Sora Upload Character",
    "GrsaiSoraCreateCharacterNodeZV": "Grsai Sora Create Character from Video",
    "GrsaiResultNodeZV": "Grsai Get Result",
    "GrsaiNanoBananaNodeZV": "Grsai Nano Banana Image Generator",
    "GrsaiNanoBananaGeminiNodeZV": "Grsai Nano Banana Gemini Format",
    "GrsaiLoadImageFromPathNodeZV": "Grsai Load Image from Path",
}