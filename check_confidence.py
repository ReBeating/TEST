#!/usr/bin/env python3
"""
检查 outputs/results 目录下哪些 *_repo_findings.json 文件包含 confidence 键
"""
import json
import os
from pathlib import Path

def check_confidence_in_files():
    """检查所有 *_repo_findings.json 文件是否包含 confidence 键"""
    
    results_dir = Path("outputs/results")
    
    # 统计数据
    files_with_confidence = []
    files_without_confidence = []
    error_files = []
    
    # 遍历所有 *_repo_findings.json 文件
    for json_file in results_dir.rglob("*_repo_findings.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 检查是否包含 confidence 键
            has_confidence = False
            
            # 检查根级别是否有 confidence 键
            if isinstance(data, dict) and 'confidence' in data:
                has_confidence = True
            # 检查是否在嵌套结构中包含 confidence
            elif isinstance(data, dict):
                # 递归检查所有值
                def check_nested_confidence(obj):
                    if isinstance(obj, dict):
                        if 'confidence' in obj:
                            return True
                        for value in obj.values():
                            if check_nested_confidence(value):
                                return True
                    elif isinstance(obj, list):
                        for item in obj:
                            if check_nested_confidence(item):
                                return True
                    return False
                
                has_confidence = check_nested_confidence(data)
            elif isinstance(data, list):
                # 如果根是列表，检查列表中的项
                for item in data:
                    if isinstance(item, dict) and 'confidence' in item:
                        has_confidence = True
                        break
                    # 递归检查
                    def check_nested_confidence(obj):
                        if isinstance(obj, dict):
                            if 'confidence' in obj:
                                return True
                            for value in obj.values():
                                if check_nested_confidence(value):
                                    return True
                        elif isinstance(obj, list):
                            for item in obj:
                                if check_nested_confidence(item):
                                    return True
                        return False
                    
                    if check_nested_confidence(item):
                        has_confidence = True
                        break
            
            # 记录结果
            relative_path = str(json_file.relative_to(results_dir))
            if has_confidence:
                files_with_confidence.append(relative_path)
            else:
                files_without_confidence.append(relative_path)
                
        except Exception as e:
            relative_path = str(json_file.relative_to(results_dir))
            error_files.append((relative_path, str(e)))
    
    # 打印统计结果
    print("=" * 80)
    print("统计报告: *_repo_findings.json 文件中的 confidence 键")
    print("=" * 80)
    print()
    
    total_files = len(files_with_confidence) + len(files_without_confidence) + len(error_files)
    print(f"总文件数: {total_files}")
    print(f"包含 confidence 键的文件数: {len(files_with_confidence)}")
    print(f"不包含 confidence 键的文件数: {len(files_without_confidence)}")
    print(f"读取出错的文件数: {len(error_files)}")
    print()
    
    # 详细列表
    if files_with_confidence:
        print("-" * 80)
        print(f"包含 confidence 键的文件 ({len(files_with_confidence)} 个):")
        print("-" * 80)
        for file_path in sorted(files_with_confidence):
            print(f"  ✓ {file_path}")
        print()
    
    if files_without_confidence:
        print("-" * 80)
        print(f"不包含 confidence 键的文件 ({len(files_without_confidence)} 个):")
        print("-" * 80)
        for file_path in sorted(files_without_confidence):
            print(f"  ✗ {file_path}")
        print()
    
    if error_files:
        print("-" * 80)
        print(f"读取出错的文件 ({len(error_files)} 个):")
        print("-" * 80)
        for file_path, error in sorted(error_files):
            print(f"  ! {file_path}")
            print(f"    错误: {error}")
        print()
    
    # 保存结果到文件
    output_file = "confidence_check_report.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("统计报告: *_repo_findings.json 文件中的 confidence 键\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"总文件数: {total_files}\n")
        f.write(f"包含 confidence 键的文件数: {len(files_with_confidence)}\n")
        f.write(f"不包含 confidence 键的文件数: {len(files_without_confidence)}\n")
        f.write(f"读取出错的文件数: {len(error_files)}\n\n")
        
        if files_with_confidence:
            f.write("-" * 80 + "\n")
            f.write(f"包含 confidence 键的文件 ({len(files_with_confidence)} 个):\n")
            f.write("-" * 80 + "\n")
            for file_path in sorted(files_with_confidence):
                f.write(f"  ✓ {file_path}\n")
            f.write("\n")
        
        if files_without_confidence:
            f.write("-" * 80 + "\n")
            f.write(f"不包含 confidence 键的文件 ({len(files_without_confidence)} 个):\n")
            f.write("-" * 80 + "\n")
            for file_path in sorted(files_without_confidence):
                f.write(f"  ✗ {file_path}\n")
            f.write("\n")
        
        if error_files:
            f.write("-" * 80 + "\n")
            f.write(f"读取出错的文件 ({len(error_files)} 个):\n")
            f.write("-" * 80 + "\n")
            for file_path, error in sorted(error_files):
                f.write(f"  ! {file_path}\n")
                f.write(f"    错误: {error}\n")
    
    print(f"详细报告已保存到: {output_file}")

if __name__ == "__main__":
    check_confidence_in_files()
