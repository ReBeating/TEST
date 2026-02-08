import pickle
import json
import os
from typing import Any, Type, TypeVar
from pydantic import BaseModel, TypeAdapter
from core.configs import OUTPUT_DIR_PATH

# Define generic type for type hinting
T = TypeVar("T")

class CheckpointManager:
    @staticmethod
    def save_pkl(data: Any, filename: str, output_dir: str = OUTPUT_DIR_PATH):
        """
        General saving: Save as a binary Pickle file
        Pros: Retains all Python native types (Set, Tuple, Custom Objects)
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
        """General loading: Restore from Pickle"""
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
        Debugging save: Save as JSON
        Note: This is mainly for manual inspection, Sets will be converted to Lists, reloading may require type reconversion
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        file_path = os.path.join(output_dir, filename)
        if not file_path.endswith(".json"):
            file_path += ".json"
        
        # Use Pydantic's TypeAdapter to handle serialization of complex objects
        # It can automatically handle Pydantic Model, Dict, List, etc.
        adapter = TypeAdapter(type(data))
        json_bytes = adapter.dump_json(data, indent=2)
        
        with open(file_path, "wb") as f:
            f.write(json_bytes)
            
        print(f"[Debug] JSON dump saved to {file_path}")