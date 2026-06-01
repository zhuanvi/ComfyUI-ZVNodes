import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import folder_paths

# ----------------------------------------------------------------------
# 辅助函数：将 BatchNorm2d 替换为 GroupNorm
# ----------------------------------------------------------------------
def convert_batchnorm_to_groupnorm(module):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            num_groups = 8
            if num_channels < num_groups or num_channels % num_groups != 0:
                for i in range(min(num_channels, 8), 1, -1):
                    if num_channels % i == 0:
                        num_groups = i
                        break
                else:
                    num_groups = 1
            setattr(module, name, nn.GroupNorm(num_groups=num_groups, num_channels=num_channels))
        else:
            convert_batchnorm_to_groupnorm(child)


# 全局模型缓存
MODEL_CACHE = {}


def load_model_from_path(model_path, device):
    if model_path in MODEL_CACHE:
        return MODEL_CACHE[model_path]

    print(f"[MangaCleaner] Loading model from: {model_path}")
    model = smp.UnetPlusPlus(
        encoder_name="tu-efficientnetv2_rw_m",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
        decoder_attention_type="scse",
    )
    convert_batchnorm_to_groupnorm(model.decoder)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    MODEL_CACHE[model_path] = model
    return model


class MangaCleanerModelLoaderZV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {
                    "default": "Manga-Text-Segmentation-2025/model.pth",
                    "multiline": False
                }),
            }
        }

    RETURN_TYPES = ("MANGA_CLEANER_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "MangaCleaner"

    def load_model(self, model_path):
        # 自动将相对路径映射到 ComfyUI/models/smp/ 下
        if not os.path.isabs(model_path):
            full_path = os.path.join(folder_paths.models_dir, "smp", model_path)
        else:
            full_path = model_path

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_model_from_path(full_path, device)
        return (model,)


# ----------------------------------------------------------------------
# 节点2：推理（输出概率图）
# ----------------------------------------------------------------------
class MangaCleanerInferenceZV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("MANGA_CLEANER_MODEL",),
                "tta_hflip": ("BOOLEAN", {"default": False}),
                "tta_vflip": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "inference"
    CATEGORY = "MangaCleaner"

    def inference(self, image, model, tta_hflip, tta_vflip):
        """
        image:  (B, H, W, C) 值域 [0,1]
        输出:  (B, H, W) 概率图，值域 [0,1]
        """
        device = next(model.parameters()).device
        B, H, W, C = image.shape

        # 预处理：归一化 + to tensor (C,H,W)
        transform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

        prob_maps = []
        for i in range(B):
            # 转为 numpy 单张图 (H,W,C) uint8
            np_img = (image[i].cpu().numpy() * 255).astype(np.uint8)
            augmented = transform(image=np_img)
            tensor = augmented["image"].unsqueeze(0).to(device)  # (1, C, H, W)

            # Pad 到 32 的倍数
            _, _, h, w = tensor.shape
            pad_h = (32 - h % 32) % 32
            pad_w = (32 - w % 32) % 32
            if pad_h > 0 or pad_w > 0:
                tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="constant", value=0)

            steps = 1 + int(tta_hflip) + int(tta_vflip)
            accumulated = None

            with torch.no_grad():
                # 原始图像推理
                if device.type == "cuda":
                    with torch.amp.autocast("cuda"):
                        logits = model(tensor)
                        probs = torch.sigmoid(logits)
                else:
                    logits = model(tensor)
                    probs = torch.sigmoid(logits)
                accumulated = probs

                # 水平翻转 TTA
                if tta_hflip:
                    tensor_flip = torch.flip(tensor, [3])
                    if device.type == "cuda":
                        with torch.amp.autocast("cuda"):
                            logits_flip = model(tensor_flip)
                            probs_flip = torch.sigmoid(logits_flip)
                    else:
                        logits_flip = model(tensor_flip)
                        probs_flip = torch.sigmoid(logits_flip)
                    accumulated += torch.flip(probs_flip, [3])

                # 垂直翻转 TTA
                if tta_vflip:
                    tensor_flip = torch.flip(tensor, [2])
                    if device.type == "cuda":
                        with torch.amp.autocast("cuda"):
                            logits_flip = model(tensor_flip)
                            probs_flip = torch.sigmoid(logits_flip)
                    else:
                        logits_flip = model(tensor_flip)
                        probs_flip = torch.sigmoid(logits_flip)
                    accumulated += torch.flip(probs_flip, [2])

            # 平均并裁剪到原始尺寸
            probs = accumulated / steps  # (1, 1, H_pad, W_pad)
            probs = probs[:, :, :H, :W]  # 去除 padding
            prob_map = probs.squeeze(0).squeeze(0)  # (H, W)
            prob_maps.append(prob_map)

        # 堆叠回 batch 维度
        prob_maps = torch.stack(prob_maps, dim=0)  # (B, H, W)
        return (prob_maps,)


# ----------------------------------------------------------------------
# 节点3：后处理（生成叠加图、清洁图、二值掩膜）
# ----------------------------------------------------------------------
class MangaCleanerPostProcessZV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prob_map": ("MASK",),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 0.9, "step": 0.05}),
                "alpha": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05}),
                "padding_iter": ("INT", {"default": 2, "min": 0, "max": 20, "step": 1}),
                "fill_holes": ("BOOLEAN", {"default": False}),
                "close_gaps_kernel": ("INT", {"default": 0, "min": 0, "max": 10, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("overlay", "cleaned", "mask")
    FUNCTION = "post_process"
    CATEGORY = "MangaCleaner"

    def post_process(self, image, prob_map, threshold, alpha, padding_iter, fill_holes, close_gaps_kernel):
        """
        image:       (B, H, W, C)  值域 [0,1]
        prob_map:    (B, H, W)     值域 [0,1]
        返回: overlay, cleaned (均为 (B, H, W, C) [0,1])，以及 mask (B, H, W) [0,1]
        """
        B, H, W, C = image.shape
        overlay_list = []
        cleaned_list = []
        mask_list = []

        for i in range(B):
            # 转为 numpy uint8
            img_np = (image[i].cpu().numpy() * 255).astype(np.uint8)          # (H, W, 3)
            prob_np = prob_map[i].cpu().numpy()                               # (H, W)

            # 二值化
            binary = (prob_np > threshold).astype(np.uint8) * 255

            # 闭运算（合拢小缝隙）
            if close_gaps_kernel > 0:
                k = int(close_gaps_kernel)
                if k % 2 == 0:
                    k += 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            # 填充内部孔洞
            if fill_holes:
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(binary, contours, -1, 255, -1)

            # ---- 制作 overlay ----
            overlay = img_np.copy()
            red = np.zeros_like(img_np)
            red[:, :, 0] = 255                       # 红色通道
            mask_bool = binary == 255
            overlay[mask_bool] = (img_np[mask_bool] * (1 - alpha) + red[mask_bool] * alpha).astype(np.uint8)

            # ---- 制作 cleaned（白化） ----
            cleaned = img_np.copy()
            if padding_iter > 0:
                kernel_dilate = np.ones((3, 3), np.uint8)
                whiteout_mask = cv2.dilate(binary, kernel_dilate, iterations=int(padding_iter))
            else:
                whiteout_mask = binary
            cleaned[whiteout_mask == 255] = [255, 255, 255]

            # 转回 float [0,1]
            overlay_t = torch.from_numpy(overlay).float() / 255.0
            cleaned_t = torch.from_numpy(cleaned).float() / 255.0
            mask_t = torch.from_numpy(binary).float() / 255.0   # 0 或 1

            overlay_list.append(overlay_t)
            cleaned_list.append(cleaned_t)
            mask_list.append(mask_t)

        overlay_batch = torch.stack(overlay_list, dim=0)   # (B, H, W, 3)
        cleaned_batch = torch.stack(cleaned_list, dim=0)
        mask_batch = torch.stack(mask_list, dim=0)         # (B, H, W)

        return (overlay_batch, cleaned_batch, mask_batch)


# ----------------------------------------------------------------------
# 节点映射（ComfyUI 会自动发现这些类）
# ----------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "MangaCleanerModelLoaderZV": MangaCleanerModelLoaderZV,
    "MangaCleanerInferenceZV": MangaCleanerInferenceZV,
    "MangaCleanerPostProcessZV": MangaCleanerPostProcessZV,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MangaCleanerModelLoaderZV": "MangaCleaner Model Loader",
    "MangaCleanerInferenceZV": "MangaCleaner Inference",
    "MangaCleanerPostProcessZV": "MangaCleaner PostProcess",
}