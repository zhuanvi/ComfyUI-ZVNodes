import os
import glob
import torchaudio
import re
import torch
from comfy_api.latest import IO
from .utils import generate_node_mappings

class LoadAudioFromDirZV:
    """
    从目录中按索引加载音频文件，支持包含子文件夹。
    输入：
        - directory: 目标目录路径
        - start_index: 要加载的文件索引（从0开始）
        - include_subfolders: 是否递归搜索子文件夹
    输出：
        - audio: 音频数据 (waveform tensor, sample_rate)
        - directory: 音频文件所在目录
        - name: 不带扩展名的文件名
        - prefix: 文件名中第一个分隔符之前的部分（用于分类或标记）
    """

    AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.aiff', '.aac', '.m4a', '.wma'}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {"default": "", "multiline": False}),
                "start_index": ("INT", {"default": 0, "min": 0, "step": 1}),
                "include_subfolders": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("audio", "directory", "name", "prefix")
    FUNCTION = "load_audio"
    CATEGORY = "audio"

    def load_audio(self, directory, start_index, include_subfolders):
        if not os.path.isdir(directory):
            raise ValueError(f"Directory does not exist: {directory}")

        audio_files = []
        if include_subfolders:
            for root, _, files in os.walk(directory):
                for file in files:
                    if os.path.splitext(file)[1].lower() in self.AUDIO_EXTENSIONS:
                        audio_files.append(os.path.join(root, file))
        else:
            pattern = os.path.join(directory, "*")
            all_files = glob.glob(pattern)
            for f in all_files:
                if os.path.isfile(f) and os.path.splitext(f)[1].lower() in self.AUDIO_EXTENSIONS:
                    audio_files.append(f)

        # 自然排序
        audio_files.sort(key=lambda p: [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', p)])

        if not audio_files:
            raise ValueError(f"No audio files found in directory: {directory}")
        if start_index < 0 or start_index >= len(audio_files):
            raise IndexError(
                f"start_index {start_index} out of range. Available files: {len(audio_files)}"
            )

        target_file = audio_files[start_index]

        try:
            waveform, sample_rate = torchaudio.load(target_file)
        except Exception as e:
            raise RuntimeError(f"Failed to load audio file: {target_file}\n{str(e)}")

        file_dir = os.path.dirname(target_file)
        file_basename = os.path.basename(target_file)
        name_no_ext = os.path.splitext(file_basename)[0]

        # 前缀：按非字母数字字符分割后的第一段
        prefix_parts = re.split(r'[^a-zA-Z0-9]+', name_no_ext)
        prefix = prefix_parts[0] if prefix_parts else name_no_ext
        audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
        
        return (audio, file_dir, name_no_ext, prefix)
    

NODE_CONFIG = {
    "LoadAudioFromDirZV":{"class": LoadAudioFromDirZV, "name": "Load One Audio (Directory)"}
}

NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = generate_node_mappings(NODE_CONFIG)