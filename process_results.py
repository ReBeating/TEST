#!/usr/bin/env python3
"""
将 New-Found_0126.csv 中的 TP 和 FP 条目整理到 results/checked_list.csv
"""

import csv
import os

def process_csv():
    # 读取源文件
    source_file = 'New-Found_0126.csv'
    target_file = 'results/checked_list.csv'
    
    # 读取现有的 checked_list 数据，用于去重
    existing_entries = set()
    if os.path.exists(target_file):
        with open(target_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 使用 vul_id + repo + target_func_name 作为唯一键
                key = (row['vul_id'], row['repo'], row['target_func_name'])
                existing_entries.add(key)
    
    # 读取 New-Found_0126.csv 并筛选 TP/FP
    new_entries = []
    with open(source_file, 'r', encoding='utf-8-sig') as f:  # 使用 utf-8-sig 处理 BOM
        reader = csv.DictReader(f)
        for row in reader:
            # 只处理 judgement 为 TP 或 FP 的条目
            if row['judgement'] in ['TP', 'FP']:
                # 检查是否已存在
                key = (row['vul_id'], row['repo'], row['target_func'])
                if key not in existing_entries:
                    new_entries.append({
                        'vul_id': row['vul_id'],
                        'repo': row['repo'],
                        'fixed_commit_sha': row['fixed_commit_sha'],
                        'target_file_path': row['target_file'],
                        'target_func_name': row['target_func'],
                        'judgement': row['judgement'],
                        'reported': '',
                        'confirmed': '',
                        'requested': '',
                        'received': ''
                    })
                    existing_entries.add(key)
    
    # 追加到 checked_list.csv
    if new_entries:
        file_exists = os.path.exists(target_file)
        with open(target_file, 'a', encoding='utf-8', newline='') as f:
            fieldnames = ['vul_id', 'repo', 'fixed_commit_sha', 'target_file_path', 
                         'target_func_name', 'judgement', 'reported', 'confirmed', 
                         'requested', 'received']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            # 如果文件不存在或为空，写入表头
            if not file_exists or os.path.getsize(target_file) == 0:
                writer.writeheader()
            
            # 写入新条目
            writer.writerows(new_entries)
        
        print(f"成功添加 {len(new_entries)} 条新记录到 {target_file}")
    else:
        print("没有需要添加的新记录（所有 TP/FP 条目已存在）")
    
    # 统计信息
    print(f"\n统计信息:")
    print(f"- 现有记录总数: {len(existing_entries) - len(new_entries)}")
    print(f"- 新增记录总数: {len(new_entries)}")
    print(f"- 更新后总数: {len(existing_entries)}")

if __name__ == '__main__':
    process_csv()
