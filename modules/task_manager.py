# task_manager.py
import json
import os
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class TaskRecord:
    """任务记录"""
    task_id: str
    saved_at: str
    task_type: str = "unknown"
    status: str = "pending"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskRecord':
        return cls(**data)

class TaskManager:
    """任务管理器"""
    
    def __init__(self, task_file: str):
        self.task_file = task_file
        self._ensure_file()
    
    def _ensure_file(self):
        """确保任务文件存在"""
        os.makedirs(os.path.dirname(self.task_file), exist_ok=True)
        if not os.path.exists(self.task_file):
            self._save_tasks([])
    
    def _load_tasks(self) -> List[TaskRecord]:
        """加载任务列表"""
        try:
            with open(self.task_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return [TaskRecord.from_dict(task) for task in data.get("tasks", [])]
        except Exception:
            return []
    
    def _save_tasks(self, tasks: List[TaskRecord]):
        """保存任务列表"""
        data = {
            "tasks": [asdict(task) for task in tasks],
            "updated_at": datetime.now().isoformat()
        }
        with open(self.task_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def add_task(self, task_id: str, task_type: str = "unknown"):
        """添加任务"""
        tasks = self._load_tasks()
        task = TaskRecord(
            task_id=task_id,
            saved_at=datetime.now().isoformat(),
            task_type=task_type
        )
        tasks.insert(0, task)  # 新任务放在前面
        
        # 限制任务数量
        if len(tasks) > 50:  # 可配置
            tasks = tasks[:50]
        
        self._save_tasks(tasks)
        return task
    
    def remove_task(self, task_id: str) -> bool:
        """移除任务"""
        tasks = self._load_tasks()
        initial_len = len(tasks)
        tasks = [t for t in tasks if t.task_id != task_id]
        
        if len(tasks) != initial_len:
            self._save_tasks(tasks)
            return True
        return False
    
    def get_tasks(self, limit: int = None) -> List[TaskRecord]:
        """获取任务列表"""
        tasks = self._load_tasks()
        if limit:
            tasks = tasks[:limit]
        return tasks
    
    def get_first_task(self) -> Optional[TaskRecord]:
        """获取第一个任务"""
        tasks = self.get_tasks(limit=1)
        return tasks[0] if tasks else None