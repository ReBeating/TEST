import pickle
import json
import os
from typing import Any, Type, TypeVar
from pydantic import BaseModel, TypeAdapter
from core.configs import OUTPUT_DIR_PATH

# 定义泛型，用于类型提示
T = TypeVar("T")

class CheckpointManager:
    @staticmethod
    def save_pkl(data: Any, filename: str, output_dir: str = OUTPUT_DIR_PATH):
        """
        通用保存：保存为二进制 Pickle 文件
        优点：保留所有 Python 原生类型 (Set, Tuple, Custom Objects)
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        file_path = os.path.join(output_dir, filename)
        if not file_path.endswith(".pkl"):
            file_path += ".pkl"
            
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
        
        # print(f"[Checkpoint] State saved to {file_path}")
        return file_path

    @staticmethod
    def load_pkl(filename: str, output_dir: str = OUTPUT_DIR_PATH) -> Any:
        """通用加载：从 Pickle 恢复"""
        file_path = os.path.join(output_dir, filename)
        if not file_path.endswith(".pkl"):
            file_path += ".pkl"
            
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Checkpoint not found: {file_path}")
            
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            
        # print(f"[Checkpoint] State loaded from {file_path}")
        return data

    @staticmethod
    def save_json(data: Any, filename: str, output_dir: str = OUTPUT_DIR_PATH):
        """
        调试用保存：保存为 JSON
        注意：这主要用于人工检查，Set 会被转为 List，加载回来可能需要重新转换类型
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        file_path = os.path.join(output_dir, filename)
        if not file_path.endswith(".json"):
            file_path += ".json"
        
        # 使用 Pydantic 的 TypeAdapter 来处理复杂对象的序列化
        # 它能自动处理 Pydantic Model, Dict, List 等
        adapter = TypeAdapter(type(data))
        json_bytes = adapter.dump_json(data, indent=2)
        
        with open(file_path, "wb") as f:
            f.write(json_bytes)
            
        print(f"[Debug] JSON dump saved to {file_path}")