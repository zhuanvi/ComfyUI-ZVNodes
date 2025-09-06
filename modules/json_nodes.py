from .utils import generate_node_mappings, normalize_path

from server import PromptServer
from aiohttp import web
import json
import os
import folder_paths
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import comfy.utils



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




class JsonToSrtConverterZV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_input": ("STRING", {"multiline": True, "default": "[]"}),
                "srt_config": ("STRING", {"multiline": True, "default": r"""{
  "index_field": "id",
  "time_fields": {
    "start": "start",
    "end": "end"
  },
  "content_field": "字幕",
  "time_format": {
    "input_format": "min:sec.ms",
    "output_format": "hh:mm:ss,ms"
  },
  "text_formatting": {
    "max_lines": 2,
    "max_chars_per_line": 40,
    "line_break_strategy": "word"
  },
  "filters": {
    "min_duration": 0.5,
    "max_duration": 10.0,
    "where": "字幕 != ''",
    "order_by": "start",
    "order": "asc"
  }
}"""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("srt_output",)
    FUNCTION = "convert_to_srt"
    CATEGORY = "ZVNodes/json"

    def convert_to_srt(self, json_input: str, srt_config: str) -> tuple:
        try:
            # 解析输入JSON
            json_data = json.loads(json_input)
            
            # 解析SRT配置
            config = json.loads(srt_config)
            
            # 应用SRT转换
            result = self.convert_to_srt_format(json_data, config)
            
            return (result,)
        except Exception as e:
            comfy.utils.log(f"SRT转换错误: {str(e)}")
            return (f"错误: {str(e)}",)

    def convert_to_srt_format(self, data: List[Dict], config: Dict) -> str:
        # 获取配置参数
        index_field = config.get("index_field", "id")
        start_field = config.get("time_fields", {}).get("start", "start")
        end_field = config.get("time_fields", {}).get("end", "end")
        content_field = config.get("content_field", "字幕")
        
        time_format = config.get("time_format", {})
        input_format = time_format.get("input_format", "min:sec.ms")
        output_format = time_format.get("output_format", "hh:mm:ss,ms")
        
        text_formatting = config.get("text_formatting", {})
        max_lines = text_formatting.get("max_lines", 2)
        max_chars = text_formatting.get("max_chars_per_line", 40)
        line_break_strategy = text_formatting.get("line_break_strategy", "word")
        
        # 应用过滤器
        filtered_data = self.apply_srt_filters(data, config.get("filters", {}))
        
        # 确保数据按开始时间排序
        filtered_data.sort(key=lambda x: self.parse_custom_time(x.get(start_field, "0:00.000")))
        
        # 生成SRT内容
        srt_lines = []
        for i, item in enumerate(filtered_data, 1):
            # 序号 - 如果没有提供ID字段或ID不是数字，使用自增序号
            if index_field in item and str(item[index_field]).isdigit():
                index = item[index_field]
            else:
                index = i
            
            # 时间码
            start_time = self.format_time(item.get(start_field, "0:00.000"), input_format, output_format)
            end_time = self.format_time(item.get(end_field, "0:00.000"), input_format, output_format)
            timecode = f"{start_time} --> {end_time}"
            
            # 字幕文本
            text = item.get(content_field, "")
            formatted_text = self.format_subtitle_text(text, max_lines, max_chars, line_break_strategy)
            
            # 添加到SRT
            srt_lines.append(str(index))
            srt_lines.append(timecode)
            srt_lines.append(formatted_text)
            srt_lines.append("")  # 空行分隔
            
        return "\n".join(srt_lines)

    def apply_srt_filters(self, data: List[Dict], filters: Dict) -> List[Dict]:
        if not filters:
            return data
            
        filtered_data = []
        
        # 持续时间过滤
        min_duration = filters.get("min_duration", 0)
        max_duration = filters.get("max_duration", float('inf'))
        
        start_field = filters.get("start_field", "start")
        end_field = filters.get("end_field", "end")
        
        for item in data:
            # 检查是否有必要的时间字段
            if start_field not in item or end_field not in item:
                continue
                
            # 计算持续时间
            start_seconds = self.parse_custom_time(item[start_field])
            end_seconds = self.parse_custom_time(item[end_field])
            duration = end_seconds - start_seconds
            
            # 应用持续时间过滤
            if min_duration <= duration <= max_duration:
                filtered_data.append(item)
        
        # 其他过滤条件
        where_condition = filters.get("where")
        if where_condition:
            filtered_data = [item for item in filtered_data if self.evaluate_srt_condition(item, where_condition)]
        
        # 排序
        order_by = filters.get("order_by")
        if order_by:
            reverse = filters.get("order", "asc").lower() == "desc"
            # 如果是时间字段，需要特殊处理
            if order_by in ["start", "end"]:
                filtered_data.sort(key=lambda x: self.parse_custom_time(x.get(order_by, "0:00.000")), reverse=reverse)
            else:
                filtered_data.sort(key=lambda x: x.get(order_by, 0), reverse=reverse)
        
        # 限制数量
        limit = filters.get("limit")
        if limit and limit > 0:
            filtered_data = filtered_data[:limit]
        
        return filtered_data

    def parse_custom_time(self, time_str: str) -> float:
        """解析自定义时间格式，如 '0:04.522' -> 秒数"""
        try:
            # 分割分钟和秒+毫秒
            parts = time_str.split(':')
            if len(parts) != 2:
                return 0.0
                
            minutes = int(parts[0])
            seconds_parts = parts[1].split('.')
            
            if len(seconds_parts) == 2:
                seconds = int(seconds_parts[0])
                milliseconds = int(seconds_parts[1])
            else:
                seconds = int(seconds_parts[0])
                milliseconds = 0
                
            return minutes * 60 + seconds + milliseconds / 1000.0
        except:
            return 0.0

    def evaluate_srt_condition(self, item: Dict, condition: str) -> bool:
        # 简化的条件评估
        try:
            # 支持简单的字段比较条件
            if "==" in condition:
                field, value = condition.split("==", 1)
                field = field.strip()
                value = value.strip().strip('"\'')
                return str(item.get(field, "")) == value
            elif "!=" in condition:
                field, value = condition.split("!=", 1)
                field = field.strip()
                value = value.strip().strip('"\'')
                return str(item.get(field, "")) != value
            elif ">" in condition:
                field, value = condition.split(">", 1)
                field = field.strip()
                value = value.strip().strip('"\'')
                # 如果是时间字段，需要特殊处理
                if field in ["start", "end"]:
                    field_value = self.parse_custom_time(item.get(field, "0:00.000"))
                    return field_value > float(value)
                else:
                    return float(item.get(field, 0)) > float(value)
            elif "<" in condition:
                field, value = condition.split("<", 1)
                field = field.strip()
                value = value.strip().strip('"\'')
                # 如果是时间字段，需要特殊处理
                if field in ["start", "end"]:
                    field_value = self.parse_custom_time(item.get(field, "0:00.000"))
                    return field_value < float(value)
                else:
                    return float(item.get(field, 0)) < float(value)
            else:
                # 默认检查字段是否存在且非空
                field = condition.strip()
                return field in item and item[field] not in [None, ""]
        except:
            return False

    def format_time(self, time_value: Any, input_format: str, output_format: str) -> str:
        try:
            # 处理不同的输入格式
            if input_format == "min:sec.ms":
                # 解析自定义格式，如 "0:04.522"
                total_seconds = self.parse_custom_time(str(time_value))
            elif input_format == "milliseconds":
                total_seconds = float(time_value) / 1000
            elif input_format == "timestamp":
                # 假设时间戳格式为 "hh:mm:ss,ms"
                parts = str(time_value).replace(',', '.').split(':')
                if len(parts) == 3:
                    hours, minutes, seconds = parts
                    total_seconds = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
                else:
                    total_seconds = float(time_value)
            else:  # seconds
                total_seconds = float(time_value)
            
            # 格式化为SRT时间码
            if output_format == "hh:mm:ss,ms":
                hours = int(total_seconds // 3600)
                minutes = int((total_seconds % 3600) // 60)
                seconds = int(total_seconds % 60)
                milliseconds = int((total_seconds - int(total_seconds)) * 1000)
                
                return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
            else:
                # 默认返回原始格式
                return str(time_value)
                
        except:
            return "00:00:00,000"

    def format_subtitle_text(self, text: str, max_lines: int, max_chars: int, strategy: str) -> str:
        if not text:
            return ""
            
        # 按策略分割文本
        if strategy == "word":
            words = text.split()
            lines = []
            current_line = ""
            
            for word in words:
                if len(current_line) + len(word) + 1 <= max_chars:
                    if current_line:
                        current_line += " " + word
                    else:
                        current_line = word
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
                    
                    # 检查是否超过最大行数
                    if len(lines) >= max_lines:
                        break
            
            if current_line and len(lines) < max_lines:
                lines.append(current_line)
                
        else:  # character strategy
            lines = []
            remaining_text = text
            
            while remaining_text and len(lines) < max_lines:
                if len(remaining_text) <= max_chars:
                    lines.append(remaining_text)
                    break
                else:
                    # 尝试在空格处分割
                    split_pos = remaining_text.rfind(' ', 0, max_chars)
                    if split_pos == -1:
                        split_pos = max_chars
                    
                    lines.append(remaining_text[:split_pos])
                    remaining_text = remaining_text[split_pos:].strip()
        
        return "\n".join(lines)

# 节点注册
NODE_CONFIG = {
    "JsonReaderZV": {"class": JsonReaderZV, "name": "Json Reader"},
    "JsonListNodeZV": {"class": JsonListNodeZV, "name": "Json List Node"},
    "JsonListLengthZV": {"class": JsonListLengthZV, "name": "Json List Length"},
    "JsonListIndexerZV": {"class": JsonListIndexerZV, "name": "Json List Indexer"},
    "JsonListSlicerZV": {"class": JsonListSlicerZV, "name": "Json List Slicer"},
    "JsonToSrtConverterZV": {"class": JsonToSrtConverterZV, "name": "Json To Srt Converter"},
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