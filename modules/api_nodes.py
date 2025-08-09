from .utils import generate_node_mappings

import os
import io
from volcenginesdkarkruntime import Ark
import json
import base64
from PIL import Image, ImageOps, ImageSequence
import torch
from torchvision import transforms
import numpy as np
import math

class doubaoI2INodeZV:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_url": ("STRING", {"default": "https://ark.cn-beijing.volces.com/api/v3"}),
                "model": ("STRING", {"default": "doubao-seededit-3-0-i2i-250628"}),
                "api_key": ("STRING",), 
                "image_url": ("STRING", {"default": "https://ark-project.tos-cn-beijing.volces.com/doc_image/seededit_i2i.jpeg"}),
                "prompt": ("STRING", {"default": "改成爱心形状的泡泡"}), 
                "seed": ("INT", {"default": 2048}),
                "guidance_scale": ("FLOAT", {"default": 5.5}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "doubao"
    CATEGORY = "ZVNodes/api"

    def doubao(self, api_url, model, api_key, image_url, prompt, seed, guidance_scale):
        api_key = api_key
        base_url = api_url
        client = Ark(
            base_url=base_url,
            api_key=api_key,
        )

        size = "adaptive"

        imagesResponse = client.images.generate(
            model=model,
            prompt=prompt,
            image=image_url,
            size=size,
            seed=seed,
            guidance_scale=guidance_scale,
            watermark=False,
            response_format= "b64_json"
        )
        result = imagesResponse.data[0].b64_json
        decoded_data = base64.b64decode(result)
        
        img = Image.open(io.BytesIO(decoded_data))
        img_out = []
        for frame in ImageSequence.Iterator(img):
            frame = ImageOps.exif_transpose(frame)
            if frame.mode == "I":
                frame = frame.point(lambda i: i * (1 / 256)).convert("L")
            image = frame.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image).unsqueeze(0)
            img_out.append(image)
        img_out = img_out[0]
        return (img_out,)

class doubaoT2INodeZV:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_url": ("STRING", {"default": "https://ark.cn-beijing.volces.com/api/v3"}),
                "model": ("STRING", {"default": "doubao-seedream-3-0-t2i-250415"}),
                "api_key": ("STRING",), 
                "prompt": ("STRING", {"default": "1girl"}), 
                "aspect_ratio": ("STRING", {"default": "1:1"}),
                "seed": ("INT", {"default": 2048}),
                "guidance_scale": ("FLOAT", {"default": 2.5}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "doubao"
    CATEGORY = "ZVNodes/api"

    def doubao(self, api_url, model, api_key, prompt, aspect_ratio, seed, guidance_scale):
        api_key = api_key
        base_url = api_url
        client = Ark(
            base_url=base_url,
            api_key=api_key,
        )

        if aspect_ratio == "1:1":
            size = "1024x1024"
        elif aspect_ratio == "3:4":
            size = "864x1152"
        elif aspect_ratio == "4:3":
            size = "1152x864"
        elif aspect_ratio == "9:16":
            size = "720x1280"
        elif aspect_ratio == "16:9":
            size = "1280x720"
        elif aspect_ratio == "2:3":
            size = "832x1248"
        elif aspect_ratio == "3:2":
            size = "1248x832"
        elif aspect_ratio == "21:9":
            size = "1512x648"
        else:
            _w, _h = aspect_ratio.split(":")
            w = int(_w)
            h = int(_h)
            if w*h < 900000:
                u = math.sqrt(1024*1024/(w*h))
                w = int((u*w)//8*8)
                h = int((u*h)//8*8)
            
            size = f"{w}x{h}"
            
        

        imagesResponse = client.images.generate(
            model=model,
            prompt=prompt,
            size=size,
            seed=seed,
            guidance_scale=guidance_scale,
            watermark=False,
            response_format= "b64_json"
        )
        result = imagesResponse.data[0].b64_json
        decoded_data = base64.b64decode(result)
        
        img = Image.open(io.BytesIO(decoded_data))
        img_out = []
        for frame in ImageSequence.Iterator(img):
            frame = ImageOps.exif_transpose(frame)
            if frame.mode == "I":
                frame = frame.point(lambda i: i * (1 / 256)).convert("L")
            image = frame.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image).unsqueeze(0)
            img_out.append(image)
        img_out = img_out[0]
        return (img_out,)

NODE_CONFIG = {
    "doubaoT2INodeZV": {"class": doubaoT2INodeZV, "name": "doubao T2I API Node"},
    "doubaoI2INodeZV": {"class": doubaoI2INodeZV, "name": "doubao I2I API Node"},
}

NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = generate_node_mappings(NODE_CONFIG)