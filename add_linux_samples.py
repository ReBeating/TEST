#!/usr/bin/env python3
"""
从 category_3 和 category_4 中随机选择 Linux 漏洞添加到数据集
- 从 category_3_all_false.csv 随机选择 150 个 torvalds/linux 漏洞
- 从 category_4_no_findings.csv 随机选择 50 个 torvalds/linux 漏洞
- 添加到 merged_dataset_with_category1.csv 中
"""

import csv
import random
from pathlib import Path


def read_linux_vulnerabilities(csv_file):
    """读取指定CSV文件中的 torvalds/linux 漏洞"""
    linux_vulns = []
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            repo = row['repo'].strip()
            if repo == 'torvalds/linux':
                linux_vulns.append({
                    'repo': repo,
                    'vul_id': row['vul_id'].strip(),
                    'fixed_commit_sha': row['fixed_commit_sha'].strip()
                })
    
    return linux_vulns


def main():
    # 设置随机种子以便结果可复现
    random.seed(42)
    
    # 读取 category_3_all_false.csv 中的 Linux 漏洞
    category3_file = Path("category_3_all_false.csv")
    print(f"读取 {category3_file}...")
    category3_linux = read_linux_vulnerabilities(category3_file)
    print(f"找到 {len(category3_linux)} 个 torvalds/linux 漏洞")
    
    # 读取 category_4_no_findings.csv 中的 Linux 漏洞
    category4_file = Path("category_4_no_findings.csv")
    print(f"读取 {category4_file}...")
    category4_linux = read_linux_vulnerabilities(category4_file)
    print(f"找到 {len(category4_linux)} 个 torvalds/linux 漏洞")
    
    # 随机选择
    if len(category3_linux) < 150:
        print(f"警告: category_3 中只有 {len(category3_linux)} 个 Linux 漏洞，少于请求的 150 个")
        selected_category3 = category3_linux
    else:
        selected_category3 = random.sample(category3_linux, 150)
    
    if len(category4_linux) < 50:
        print(f"警告: category_4 中只有 {len(category4_linux)} 个 Linux 漏洞，少于请求的 50 个")
        selected_category4 = category4_linux
    else:
        selected_category4 = random.sample(category4_linux, 50)
    
    print(f"\n从 category_3 随机选择了 {len(selected_category3)} 个漏洞")
    print(f"从 category_4 随机选择了 {len(selected_category4)} 个漏洞")
    
    # 读取现有数据集
    existing_file = Path("merged_dataset_with_category1.csv")
    print(f"\n读取现有数据集 {existing_file}...")
    
    vulnerabilities = []
    seen = set()
    
    with open(existing_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            repo = row['repo'].strip()
            vul_id = row['vul_id'].strip()
            fixed_commit_sha = row['fixed_commit_sha'].strip()
            
            key = (repo, vul_id, fixed_commit_sha)
            seen.add(key)
            vulnerabilities.append({
                'repo': repo,
                'vul_id': vul_id,
                'fixed_commit_sha': fixed_commit_sha
            })
    
    existing_count = len(vulnerabilities)
    print(f"现有数据集包含 {existing_count} 个漏洞")
    
    # 添加新选择的漏洞（去重）
    added_count = 0
    duplicate_count = 0
    
    for vuln in selected_category3 + selected_category4:
        key = (vuln['repo'], vuln['vul_id'], vuln['fixed_commit_sha'])
        if key not in seen:
            seen.add(key)
            vulnerabilities.append(vuln)
            added_count += 1
        else:
            duplicate_count += 1
    
    print(f"\n添加了 {added_count} 个新漏洞")
    print(f"跳过了 {duplicate_count} 个重复漏洞")
    print(f"新数据集总共包含 {len(vulnerabilities)} 个漏洞")
    
    # 写入新数据集
    output_file = Path("merged_dataset_with_linux_samples.csv")
    print(f"\n写入到 {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['repo', 'vul_id', 'fixed_commit_sha']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(vulnerabilities)
    
    print(f"\n完成！")
    print(f"输出文件: {output_file}")
    print(f"总漏洞数: {len(vulnerabilities)} ({existing_count} + {added_count})")


if __name__ == "__main__":
    main()
