#!/usr/bin/env python3
"""
检查 outputs/results 目录下哪些 *_repo_findings.json 文件同时满足：
1. 包含 confidence 键
2. 有至少一个 item 的 is_vulnerable 为 true
"""
import json
import os
from pathlib import Path

def check_vulnerable_with_confidence():
    """检查文件中是否有 confidence 键且有 is_vulnerable 为 true 的项"""
    
    results_dir = Path("outputs/results")
    
    # 统计数据
    files_with_both = []  # 既有 confidence 又有 is_vulnerable=true
    files_with_confidence_only = []  # 有 confidence 但都是 is_vulnerable=false
    files_without_confidence = []  # 没有 confidence
    error_files = []
    
    # 遍历所有 *_repo_findings.json 文件
    for json_file in results_dir.rglob("*_repo_findings.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            relative_path = str(json_file.relative_to(results_dir))
            
            # 检查是否有 confidence 和 is_vulnerable
            has_confidence = False
            has_vulnerable_true = False
            vulnerable_items = []
            
            if isinstance(data, list):
                for idx, item in enumerate(data):
                    if isinstance(item, dict):
                        # 检查是否有 confidence 键
                        if 'confidence' in item:
                            has_confidence = True
                        
                        # 检查是否有 is_vulnerable=true
                        if item.get('is_vulnerable') == True:
                            has_vulnerable_true = True
                            vulnerable_items.append({
                                'index': idx,
                                'target_file': item.get('target_file', 'N/A'),
                                'target_func': item.get('target_func', 'N/A'),
                                'confidence': item.get('confidence', 'N/A'),
                                'verdict_category': item.get('verdict_category', 'N/A')
                            })
            elif isinstance(data, dict):
                # 如果根是字典
                if 'confidence' in data:
                    has_confidence = True
                if data.get('is_vulnerable') == True:
                    has_vulnerable_true = True
                    vulnerable_items.append({
                        'index': 0,
                        'target_file': data.get('target_file', 'N/A'),
                        'target_func': data.get('target_func', 'N/A'),
                        'confidence': data.get('confidence', 'N/A'),
                        'verdict_category': data.get('verdict_category', 'N/A')
                    })
            
            # 分类
            if has_confidence and has_vulnerable_true:
                files_with_both.append({
                    'path': relative_path,
                    'vulnerable_count': len(vulnerable_items),
                    'items': vulnerable_items
                })
            elif has_confidence:
                files_with_confidence_only.append(relative_path)
            else:
                files_without_confidence.append(relative_path)
                
        except Exception as e:
            relative_path = str(json_file.relative_to(results_dir))
            error_files.append((relative_path, str(e)))
    
    # 打印统计结果
    print("=" * 100)
    print("统计报告: *_repo_findings.json 文件中同时包含 confidence 和 is_vulnerable=true 的情况")
    print("=" * 100)
    print()
    
    total_files = len(files_with_both) + len(files_with_confidence_only) + len(files_without_confidence) + len(error_files)
    print(f"总文件数: {total_files}")
    print(f"有 confidence 且有 is_vulnerable=true 的文件数: {len(files_with_both)}")
    print(f"有 confidence 但都是 is_vulnerable=false 的文件数: {len(files_with_confidence_only)}")
    print(f"没有 confidence 的文件数: {len(files_without_confidence)}")
    print(f"读取出错的文件数: {len(error_files)}")
    print()
    
    # 详细列表 - 同时有 confidence 和 is_vulnerable=true
    if files_with_both:
        print("=" * 100)
        print(f"同时有 confidence 和 is_vulnerable=true 的文件 ({len(files_with_both)} 个):")
        print("=" * 100)
        
        # 统计总的 vulnerable items
        total_vulnerable_items = sum(f['vulnerable_count'] for f in files_with_both)
        print(f"总共包含 {total_vulnerable_items} 个 is_vulnerable=true 的项目\n")
        
        for file_info in sorted(files_with_both, key=lambda x: x['path']):
            print(f"📁 {file_info['path']}")
            print(f"   包含 {file_info['vulnerable_count']} 个 vulnerable 项目:")
            for item in file_info['items']:
                print(f"      [{item['index']}] {item['target_file']} :: {item['target_func']}")
                print(f"          confidence: {item['confidence']}, verdict: {item['verdict_category']}")
            print()
    else:
        print("=" * 100)
        print("没有找到同时包含 confidence 和 is_vulnerable=true 的文件")
        print("=" * 100)
        print()
    
    # 保存结果到文件
    output_file = "vulnerable_with_confidence_report.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("统计报告: *_repo_findings.json 文件中同时包含 confidence 和 is_vulnerable=true 的情况\n")
        f.write("=" * 100 + "\n\n")
        
        f.write(f"总文件数: {total_files}\n")
        f.write(f"有 confidence 且有 is_vulnerable=true 的文件数: {len(files_with_both)}\n")
        f.write(f"有 confidence 但都是 is_vulnerable=false 的文件数: {len(files_with_confidence_only)}\n")
        f.write(f"没有 confidence 的文件数: {len(files_without_confidence)}\n")
        f.write(f"读取出错的文件数: {len(error_files)}\n\n")
        
        if files_with_both:
            total_vulnerable_items = sum(f['vulnerable_count'] for f in files_with_both)
            f.write("=" * 100 + "\n")
            f.write(f"同时有 confidence 和 is_vulnerable=true 的文件 ({len(files_with_both)} 个):\n")
            f.write("=" * 100 + "\n")
            f.write(f"总共包含 {total_vulnerable_items} 个 is_vulnerable=true 的项目\n\n")
            
            for file_info in sorted(files_with_both, key=lambda x: x['path']):
                f.write(f"📁 {file_info['path']}\n")
                f.write(f"   包含 {file_info['vulnerable_count']} 个 vulnerable 项目:\n")
                for item in file_info['items']:
                    f.write(f"      [{item['index']}] {item['target_file']} :: {item['target_func']}\n")
                    f.write(f"          confidence: {item['confidence']}, verdict: {item['verdict_category']}\n")
                f.write("\n")
        else:
            f.write("=" * 100 + "\n")
            f.write("没有找到同时包含 confidence 和 is_vulnerable=true 的文件\n")
            f.write("=" * 100 + "\n\n")
        
        if error_files:
            f.write("=" * 100 + "\n")
            f.write(f"读取出错的文件 ({len(error_files)} 个):\n")
            f.write("=" * 100 + "\n")
            for file_path, error in sorted(error_files):
                f.write(f"  ! {file_path}\n")
                f.write(f"    错误: {error}\n")
    
    print(f"详细报告已保存到: {output_file}")
    
    # 保存文件列表（CSV格式）
    if files_with_both:
        csv_file = "vulnerable_with_confidence_list.csv"
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write("file_path,vulnerable_count,item_index,target_file,target_func,confidence,verdict_category\n")
            for file_info in sorted(files_with_both, key=lambda x: x['path']):
                for item in file_info['items']:
                    f.write(f'"{file_info["path"]}",{file_info["vulnerable_count"]},{item["index"]},'
                           f'"{item["target_file"]}","{item["target_func"]}",{item["confidence"]},'
                           f'"{item["verdict_category"]}"\n')
        print(f"CSV 列表已保存到: {csv_file}")

if __name__ == "__main__":
    check_vulnerable_with_confidence()
