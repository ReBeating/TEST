#!/usr/bin/env python3
"""
合并漏洞数据集脚本
- 包含 inputs/1day_vul_list.csv 中的所有漏洞
- 包含 inputs/added_vul_list.csv 中除 torvalds/linux 外的所有漏洞
"""

import csv
from pathlib import Path


def merge_datasets():
    """合并并过滤漏洞数据集"""
    
    # 输入文件路径
    oneday_file = Path("inputs/1day_vul_list.csv")
    added_file = Path("inputs/added_vul_list.csv")
    
    # 输出文件路径
    output_file = Path("merged_dataset.csv")
    
    # 存储所有漏洞数据（使用集合去重）
    vulnerabilities = []
    seen = set()  # 用于去重：(repo, vul_id, fixed_commit_sha)
    
    # 读取 1day_vul_list.csv 的所有内容
    print(f"读取 {oneday_file}...")
    with open(oneday_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            repo = row['repo'].strip()
            vul_id = row['vul_id'].strip()
            fixed_commit_sha = row['fixed_commit_sha'].strip()
            
            key = (repo, vul_id, fixed_commit_sha)
            if key not in seen:
                seen.add(key)
                vulnerabilities.append({
                    'repo': repo,
                    'vul_id': vul_id,
                    'fixed_commit_sha': fixed_commit_sha
                })
    
    oneday_count = len(vulnerabilities)
    print(f"从 1day_vul_list.csv 读取了 {oneday_count} 个漏洞")
    
    # 读取 added_vul_list.csv，但过滤掉 torvalds/linux
    print(f"读取 {added_file}（过滤 torvalds/linux）...")
    added_count = 0
    filtered_count = 0
    
    with open(added_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            repo = row['repo'].strip()
            vul_id = row['vul_id'].strip()
            fixed_commit_sha = row['fixed_commit_sha'].strip()
            
            # 过滤掉 torvalds/linux
            if repo == 'torvalds/linux':
                filtered_count += 1
                continue
            
            key = (repo, vul_id, fixed_commit_sha)
            if key not in seen:
                seen.add(key)
                vulnerabilities.append({
                    'repo': repo,
                    'vul_id': vul_id,
                    'fixed_commit_sha': fixed_commit_sha
                })
                added_count += 1
    
    print(f"从 added_vul_list.csv 读取了 {added_count} 个新漏洞")
    print(f"过滤掉 {filtered_count} 个 torvalds/linux 的漏洞")
    print(f"总共 {len(vulnerabilities)} 个漏洞（去重后）")
    
    # 写入到输出文件
    print(f"写入到 {output_file}...")
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['repo', 'vul_id', 'fixed_commit_sha']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(vulnerabilities)
    
    print(f"完成！输出文件：{output_file}")
    print(f"最终数据集包含 {len(vulnerabilities)} 个漏洞")


if __name__ == "__main__":
    merge_datasets()
