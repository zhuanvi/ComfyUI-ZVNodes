from .utils import generate_node_mappings, normalize_path

from server import PromptServer
from aiohttp import web
import json
import os
import folder_paths
from pathlib import Path



class JsonReaderZV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_file": ("STRING", {"default": ""}),
                "selected_key": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "JSON")
    RETURN_NAMES = ("selected_value", "json_data")
    FUNCTION = "process_json"
    CATEGORY = "ZVNodes/json"

    def process_json(self, json_file, selected_key):
        if not json_file:
            return ("", {})
        
        try:
            file_path = os.path.join(folder_paths.get_input_directory(), "json", json_file)
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            selected_value = json_data.get(selected_key, "")
            if isinstance(selected_value, list):
                selected_value = [str(value) for value in selected_value]
            else:
                selected_value = str(selected_value)
            return (selected_value, json_data)
        except Exception as e:
            print(f"Error processing JSON: {e}")
            return ("", {})
        
class JsonListNodeZV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_string": ("STRING", {
                    "multiline": True,
                    "default": '[{"x": 184, "y": 157}, {"x": 430, "y": 620}, {"x": 887, "y": 229}]'
                }),
            }
        }
    
    RETURN_TYPES = ("JSON",)
    RETURN_NAMES = ("json_list",)
    FUNCTION = "parse_json"
    CATEGORY = "ZVNodes/json"

    def parse_json(self, json_string):
        try:
            # 解析JSON字符串为Python对象
            parsed = json.loads(json_string)
            
            # 验证解析结果是否为列表
            if not isinstance(parsed, list):
                raise ValueError("JSON must be a list")
                
                       
            return (parsed,)
        
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format")
        except Exception as e:
            raise ValueError(f"Error: {str(e)}")

class JsonListLengthZV:
    """计算JSON列表的长度"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_list": ("JSON",),
            }
        }
    
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("length",)
    FUNCTION = "get_length"
    CATEGORY = "ZVNodes/json"

    def get_length(self, json_list):
        if not isinstance(json_list, list):
            raise ValueError("输入必须是JSON列表")
        return (len(json_list),)

class JsonListIndexerZV:
    """按索引获取JSON列表中的元素"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_list": ("JSON",),
                "index": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("JSON",)
    RETURN_NAMES = ("element",)
    FUNCTION = "get_element"
    CATEGORY = "ZVNodes/json"

    def get_element(self, json_list, index):
        if not isinstance(json_list, list):
            raise ValueError("输入必须是JSON列表")
        
        if index >= len(json_list) or index < 0:
            raise ValueError(f"索引超出范围(0-{len(json_list)-1})")
        
        return (json_list[index],)

class JsonListSlicerZV:
    """对JSON列表进行切片操作"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_list": ("JSON",),
                "start_index": ("INT", {
                    "default": 0, 
                    "min": -9999, 
                    "max": 9999,
                    "step": 1
                }),
                "end_index": ("INT", {
                    "default": 0, 
                    "min": -9999, 
                    "max": 9999,
                    "step": 1
                }),
            },
            "optional": {
                "step": ("INT", {
                    "default": 1, 
                    "min": 1, 
                    "max": 100,
                    "step": 1
                }),
            }
        }
    
    RETURN_TYPES = ("JSON",)
    RETURN_NAMES = ("sliced_list",)
    FUNCTION = "slice_list"
    CATEGORY = "ZVNodes/json"
    DESCRIPTION = "切片JSON列表，支持负索引和步长"

    def slice_list(self, json_list, start_index, end_index, step=1):
        # 验证输入类型
        if not isinstance(json_list, list):
            raise ValueError("输入必须是JSON列表")
        
        # 获取列表长度
        length = len(json_list)
        
        # 处理负索引
        start = start_index if start_index >= 0 else max(0, length + start_index)
        end = end_index if end_index >= 0 else max(0, length + end_index)
        
        # 确保索引在有效范围内
        start = max(0, min(start, length))
        end = max(0, min(end, length))
        
        # 处理特殊情况
        if start > end:
            # 当起始位置大于结束位置时，返回空列表
            return ([],)
        
        # 应用切片
        sliced = json_list[start:end:step]
        
        return (sliced,)

# 节点注册
NODE_CONFIG = {
    "JsonReaderZV": {"class": JsonReaderZV, "name": "Json Reader"},
    "JsonListNodeZV": {"class": JsonListNodeZV, "name": "Json List Node"},
    "JsonListLengthZV": {"class": JsonListLengthZV, "name": "Json List Length"},
    "JsonListIndexerZV": {"class": JsonListIndexerZV, "name": "Json List Indexer"},
    "JsonListSlicerZV": {"class": JsonListSlicerZV, "name": "Json List Slicer"},
}

NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = generate_node_mappings(NODE_CONFIG)

routes = PromptServer.instance.routes
@routes.post('/api/zvnodes/json/upload_json')
async def upload_json(request):
    reader = await request.multipart()
    field = await reader.next()
    
    if field.name != 'file':
        return web.json_response({"error": "No file provided"}, status=400)
    
    filename = field.filename
    if not filename.endswith('.json'):
        return web.json_response({"error": "Only JSON files are allowed"}, status=400)
    
    # 创建上传目录
    upload_dir = os.path.join(folder_paths.get_input_directory(), "json")
    os.makedirs(upload_dir, exist_ok=True)
    
    # 保存文件
    filepath = os.path.join(upload_dir, filename)
    with open(filepath, 'wb') as f:
        while True:
            chunk = await field.read_chunk()
            if not chunk:
                break
            f.write(chunk)
    
    return web.json_response({"filename": filename, "message": "Upload successful"})

@routes.get('/api/zvnodes/json/get_json_keys')
async def get_json_keys(request):
    filename = request.query.get('filename')
    if not filename:
        return web.json_response({"error": "No filename provided"}, status=400)
    
    try:
        file_path = os.path.join(folder_paths.get_input_directory(), "json", filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        keys = list(json_data.keys()) if isinstance(json_data, dict) else []
        return web.json_response({"keys": keys})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

@routes.get('/api/zvnodes/json/list_json_files')
async def list_json_files(request):
    upload_dir = os.path.join(folder_paths.get_input_directory(), "json")
    if not os.path.exists(upload_dir):
        return web.json_response({"files": []})
    
    files = [f for f in os.listdir(upload_dir) if f.endswith('.json')]
    return web.json_response({"files": files})