#!/usr/bin/env python3
"""
合并漏洞数据集脚本（包含 category_1_has_tp.csv）
- 包含 inputs/1day_vul_list.csv 中的所有漏洞
- 包含 inputs/added_vul_list.csv 中除 torvalds/linux 外的所有漏洞
- 包含 category_1_has_tp.csv 中的所有漏洞
"""

import csv
from pathlib import Path


def merge_datasets():
    """合并并过滤漏洞数据集"""
    
    # 输入文件路径
    oneday_file = Path("inputs/1day_vul_list.csv")
    added_file = Path("inputs/added_vul_list.csv")
    category1_file = Path("category_1_has_tp.csv")
    
    # 输出文件路径
    output_file = Path("merged_dataset_with_category1.csv")
    
    # 存储所有漏洞数据（使用集合去重）
    vulnerabilities = []
    seen = set()  # 用于去重：(repo, vul_id, fixed_commit_sha)
    
    # 统计信息
    stats = {
        '1day_vul_list': 0,
        'added_vul_list': 0,
        'added_vul_list_filtered': 0,
        'category_1': 0,
        'category_1_filtered': 0
    }
    
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
                stats['1day_vul_list'] += 1
    
    print(f"从 1day_vul_list.csv 读取了 {stats['1day_vul_list']} 个漏洞")
    
    # 读取 added_vul_list.csv，但过滤掉 torvalds/linux
    print(f"读取 {added_file}（过滤 torvalds/linux）...")
    
    with open(added_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            repo = row['repo'].strip()
            vul_id = row['vul_id'].strip()
            fixed_commit_sha = row['fixed_commit_sha'].strip()
            
            # 过滤掉 torvalds/linux
            if repo == 'torvalds/linux':
                stats['added_vul_list_filtered'] += 1
                continue
            
            key = (repo, vul_id, fixed_commit_sha)
            if key not in seen:
                seen.add(key)
                vulnerabilities.append({
                    'repo': repo,
                    'vul_id': vul_id,
                    'fixed_commit_sha': fixed_commit_sha
                })
                stats['added_vul_list'] += 1
    
    print(f"从 added_vul_list.csv 读取了 {stats['added_vul_list']} 个新漏洞")
    print(f"过滤掉 {stats['added_vul_list_filtered']} 个 torvalds/linux 的漏洞")
    
    # 读取 category_1_has_tp.csv（不过滤 torvalds/linux）
    print(f"读取 {category1_file}...")
    
    with open(category1_file, 'r', encoding='utf-8') as f:
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
                stats['category_1'] += 1
    
    print(f"从 category_1_has_tp.csv 读取了 {stats['category_1']} 个新漏洞")
    print(f"总共 {len(vulnerabilities)} 个漏洞（去重后）")
    
    # 写入到输出文件
    print(f"写入到 {output_file}...")
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['repo', 'vul_id', 'fixed_commit_sha']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(vulnerabilities)
    
    print(f"\n完成！输出文件：{output_file}")
    print(f"最终数据集包含 {len(vulnerabilities)} 个漏洞")
    print(f"\n数据来源统计：")
    print(f"  - 1day_vul_list.csv: {stats['1day_vul_list']} 个漏洞")
    print(f"  - added_vul_list.csv: {stats['added_vul_list']} 个新漏洞 (过滤 {stats['added_vul_list_filtered']} 个)")
    print(f"  - category_1_has_tp.csv: {stats['category_1']} 个新漏洞")


if __name__ == "__main__":
    merge_datasets()
