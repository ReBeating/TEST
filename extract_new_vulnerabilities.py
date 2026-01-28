#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提取 added_vul_list.csv 中不在 0day_vul_list.csv 中，且 repo 不为 torvalds/linux 的漏洞
"""

import csv

def main():
    added_file = 'inputs/added_vul_list.csv'
    original_file = 'inputs/0day_vul_list.csv'
    output_file = 'inputs/new_vulnerabilities.csv'
    
    # 读取 0day_vul_list.csv 中的所有行，存入集合
    print(f"正在读取 {original_file}...")
    original_set = set()
    with open(original_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 使用 (repo, vul_id, fixed_commit_sha) 作为唯一标识
            key = (row['repo'], row['vul_id'], row['fixed_commit_sha'])
            original_set.add(key)
    
    print(f"已加载 {len(original_set)} 条原始漏洞记录")
    
    # 读取 added_vul_list.csv，找出不在 original_set 中且 repo 不为 torvalds/linux 的行
    print(f"\n正在读取 {added_file}...")
    new_vulnerabilities = []
    with open(added_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row['repo'], row['vul_id'], row['fixed_commit_sha'])
            # 检查是否不在原始集合中，且 repo 不为 torvalds/linux
            if key not in original_set and row['repo'] != 'torvalds/linux':
                new_vulnerabilities.append(row)
    
    print(f"找到 {len(new_vulnerabilities)} 条新漏洞记录（不在原始列表中且 repo 不为 torvalds/linux）")
    
    # 写入输出文件
    print(f"\n正在写入 {output_file}...")
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['repo', 'vul_id', 'fixed_commit_sha']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(new_vulnerabilities)
    
    print(f"完成！已将 {len(new_vulnerabilities)} 条记录写入 {output_file}")
    
    # 按 repo 统计
    print("\n按仓库统计新增漏洞数量：")
    repo_counts = {}
    for vul in new_vulnerabilities:
        repo = vul['repo']
        repo_counts[repo] = repo_counts.get(repo, 0) + 1
    
    for repo in sorted(repo_counts.keys()):
        print(f"  {repo}: {repo_counts[repo]}")

if __name__ == '__main__':
    main()
