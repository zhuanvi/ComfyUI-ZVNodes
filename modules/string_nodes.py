from .utils import generate_node_mappings
from pathlib import Path
from typing import List, Tuple, Union

class MultiLineOperationZV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_a": ("STRING", {"multiline": True, "default": ""}),
                "text_b": ("STRING", {"multiline": True, "default": ""}),
                "operation": (["concatenate", "add", "subtract", "multiply", "divide", "compare", "with_line_number"],),
                "separator": ("STRING", {"default": " "}),
            },
            "optional": {
                "line_number_format": ("STRING", {"default": "[{num}]"}),
                "start_number": ("INT", {"default": 1, "min": 0, "max": 10000}),
                "step": ("INT", {"default": 1, "min": 1, "max": 100}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "process_lines"
    CATEGORY = "ZVNodes/string"
    OUTPUT_NODE = True

    def process_lines(self, text_a, text_b, operation, separator, line_number_format="[{num}]", start_number=1, step=1):
        # 分割文本为行
        lines_a = text_a.split('\n')
        lines_b = text_b.split('\n')
        
        # 确保两个输入有相同的行数
        max_lines = max(len(lines_a), len(lines_b))
        lines_a += [''] * (max_lines - len(lines_a))
        lines_b += [''] * (max_lines - len(lines_b))
        
        results = []
        line_number = start_number
        
        for i, (line_a, line_b) in enumerate(zip(lines_a, lines_b)):
            if operation == "concatenate":
                result = f"{line_a}{separator}{line_b}"
            elif operation == "add":
                try:
                    result = str(float(line_a or 0) + float(line_b or 0))
                except ValueError:
                    result = "Error: Non-numeric input"
            elif operation == "subtract":
                try:
                    result = str(float(line_a or 0) - float(line_b or 0))
                except ValueError:
                    result = "Error: Non-numeric input"
            elif operation == "multiply":
                try:
                    result = str(float(line_a or 0) * float(line_b or 0))
                except ValueError:
                    result = "Error: Non-numeric input"
            elif operation == "divide":
                try:
                    if float(line_b or 0) == 0:
                        result = "Error: Division by zero"
                    else:
                        result = str(float(line_a or 0) / float(line_b or 0))
                except ValueError:
                    result = "Error: Non-numeric input"
            elif operation == "compare":
                result = "Equal" if line_a == line_b else "Different"
            elif operation == "with_line_number":
                # 添加行号到第一列文本
                formatted_number = line_number_format.format(
                    num=line_number,
                    index=i,
                    line_num=line_number,
                    line_index=i,
                    total=len(lines_a)
                )
                result = f"{formatted_number}{separator}{line_a}"
                line_number += step
            
            results.append(result)
        
        # 将结果连接为多行文本
        output_text = '\n'.join(results)
        
        return (output_text,)

class MultiLineConditionalZV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "condition_text": ("STRING", {"multiline": True, "default": ""}),
                "true_text": ("STRING", {"multiline": True, "default": ""}),
                "false_text": ("STRING", {"multiline": True, "default": ""}),
                "condition_type": (["equals", "contains", "starts_with", "ends_with", "not_empty", "line_number_condition"],),
                "match_value": ("STRING", {"default": ""}),
            },
            "optional": {
                "line_number_condition": (["even", "odd", "multiple_of", "greater_than", "less_than"],),
                "line_number_value": ("INT", {"default": 3, "min": 1, "max": 1000}),
                "start_number": ("INT", {"default": 1, "min": 0, "max": 10000}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "conditional_process"
    CATEGORY = "ZVNodes/string"
    OUTPUT_NODE = True

    def conditional_process(self, condition_text, true_text, false_text, condition_type, match_value, 
                           line_number_condition="even", line_number_value=3, start_number=1):
        # 分割文本为行
        cond_lines = condition_text.split('\n')
        true_lines = true_text.split('\n')
        false_lines = false_text.split('\n')
        
        # 确保所有输入有相同的行数
        max_lines = max(len(cond_lines), len(true_lines), len(false_lines))
        cond_lines += [''] * (max_lines - len(cond_lines))
        true_lines += [''] * (max_lines - len(true_lines))
        false_lines += [''] * (max_lines - len(false_lines))
        
        results = []
        
        for i, (cond, true_val, false_val) in enumerate(zip(cond_lines, true_lines, false_lines)):
            condition_met = False
            current_line_number = start_number + i
            
            if condition_type == "equals":
                condition_met = (cond == match_value)
            elif condition_type == "contains":
                condition_met = (match_value in cond)
            elif condition_type == "starts_with":
                condition_met = cond.startswith(match_value)
            elif condition_type == "ends_with":
                condition_met = cond.endswith(match_value)
            elif condition_type == "not_empty":
                condition_met = (cond.strip() != "")
            elif condition_type == "line_number_condition":
                if line_number_condition == "even":
                    condition_met = (current_line_number % 2 == 0)
                elif line_number_condition == "odd":
                    condition_met = (current_line_number % 2 == 1)
                elif line_number_condition == "multiple_of":
                    condition_met = (current_line_number % line_number_value == 0)
                elif line_number_condition == "greater_than":
                    condition_met = (current_line_number > line_number_value)
                elif line_number_condition == "less_than":
                    condition_met = (current_line_number < line_number_value)
            
            results.append(true_val if condition_met else false_val)
        
        output_text = '\n'.join(results)
        return (output_text,)

class LineNumberGeneratorZV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_input": ("STRING", {"multiline": True, "default": ""}),
                "format": ("STRING", {"default": "{num}. "}),
                "start_number": ("INT", {"default": 1, "min": 0, "max": 10000}),
                "step": ("INT", {"default": 1, "min": 1, "max": 100}),
                "position": (["prefix", "suffix", "replace"],),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_line_numbers"
    CATEGORY = "ZVNodes/string"
    OUTPUT_NODE = True

    def generate_line_numbers(self, text_input, format, start_number, step, position):
        lines = text_input.split('\n')
        results = []
        current_number = start_number
        
        for line in lines:
            formatted_number = format.format(
                num=current_number,
                index=current_number - start_number,
                total=len(lines)
            )
            
            if position == "prefix":
                result = f"{formatted_number}{line}"
            elif position == "suffix":
                result = f"{line}{formatted_number}"
            elif position == "replace":
                result = formatted_number
            
            results.append(result)
            current_number += step
        
        output_text = '\n'.join(results)
        return (output_text,)

class StringToPathZV:
    """
    将字符串转换为PATH
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string": ("STRING", ),
                "resolve_paths": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "是否解析为绝对路径"
                }),
            },
            "optional": {
                "base_directory": ("STRING", {
                    "default": "",
                    "placeholder": "基础目录（可选）",
                    "tooltip": "为相对路径提供基础目录"
                }),
            }
        }
    
    RETURN_TYPES = ("PATH", )
    RETURN_NAMES = ("path", )
    FUNCTION = "convert"
    CATEGORY = "ZVNodes/string"
    DESCRIPTION = "将字符串列表转换为PATH列表，支持路径验证和过滤"
    
    def convert(self, 
                string: str | List[str], 
                resolve_paths: bool,
                base_directory: str = "") -> Tuple[Path]:
        """
        将字符串列表转换为PATH列表
        
        Args:
            string: 输入的路径字符串
            path_validation: 路径验证级别
            resolve_paths: 是否解析路径
            base_directory: 基础目录
            
        Returns:
            tuple: (PATH, )
        """
        
        # 清理输入
        path_str = string
               
        # 处理基础目录
        base_path = self._get_base_path(base_directory)
        
        try:
            # 创建Path对象
            path_obj = self._create_path_object(path_str, base_path, resolve_paths)
                
        except Exception as e:
            print(f"处理路径时出错 '{path_str}': {e}")
    
        
        return (path_obj, )
       
    def _get_base_path(self, base_directory: str) -> Union[Path, None]:
        """获取基础路径"""
        base_dir = base_directory.strip()
        
        try:
            base_path = Path(base_dir)
            return base_path
        except Exception as e:
            print(f"处理基础目录时出错: {e}")
            return None
    
    def _parse_extensions(self, extensions_str: str) -> List[str]:
        """解析文件扩展名列表"""
        if not extensions_str.strip():
            return []
        
        extensions = []
        for ext in extensions_str.split(','):
            ext = ext.strip().lower()
            if ext:
                if not ext.startswith('.'):
                    ext = '.' + ext
                extensions.append(ext)
        return extensions
    
    def _create_path_object(self, path_str: str, base_path: Path, resolve_paths: bool) -> Path:
        """创建Path对象"""
        path_obj = Path(path_str)
        
        # 如果有基础目录且路径是相对的，则拼接
        if base_path and not path_obj.is_absolute():
            path_obj = base_path / path_obj
        
        # 解析为绝对路径
        if resolve_paths:
            try:
                if path_obj.is_absolute():
                    path_obj = path_obj.resolve()
            except Exception:
                # 如果路径不存在，resolve会失败，但我们可以继续使用绝对路径
                pass
        
        return path_obj
    


NODE_CONFIG = {
    "StringToPathZV": {"class": StringToPathZV, "name": "String To Path"},
    "MultiLineOperationZV": {"class": MultiLineOperationZV, "name": "MultiLine Operation"},
    "MultiLineConditionalZV": {"class": MultiLineConditionalZV, "name": "MultiLine Conditional"},
    "LineNumberGeneratorZV": {"class": LineNumberGeneratorZV, "name": "LineNumber Generator"},
}

NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = generate_node_mappings(NODE_CONFIG)