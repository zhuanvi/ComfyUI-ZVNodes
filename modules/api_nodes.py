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

class doubaoT2INodeZV:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": "https://ark.cn-beijing.volces.com/api/v3"}),
                "model": ("STRING", {"default": "doubao-seedream-3-0-t2i-250415"}),
                "api_key": ("STRING",), 
                "prompt": ("STRING", {"default": "Hello"}), 
                "width": ("INT", {"default": 1024}),
                "height": ("INT", {"default": 1024}),
                "seed": ("INT", {"default": 2048}),
                "guidance_scale": ("FLOAT", {"default": 2.5}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "doubao"
    CATEGORY = "ZVNodes/api"

    def doubao(self, url, model, api_key, prompt, width, height, seed, guidance_scale):
        api_key = api_key
        base_url = url
        client = Ark(
            base_url=base_url,
            api_key=api_key,
        )

        size = f"{width}x{height}"

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
    "doubaoT2INodeZV": {"class": doubaoT2INodeZV, "name": "doubao T2V API Node"},
}

NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = generate_node_mappings(NODE_CONFIG)