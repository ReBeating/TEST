#!/usr/bin/env python3
"""
构建扩展数据集：
- inputs/new_dataset.csv 中的所有漏洞
- category_2_only_fp.csv 中 repo 不为 torvalds/linux 的漏洞
"""

import csv
from collections import OrderedDict

def load_vul_list(csv_path):
    """加载漏洞列表，返回{(vul_id, repo): fixed_commit_sha}的字典"""
    vul_dict = {}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 根据不同的列顺序处理
            if 'repo' in row and 'vul_id' in row:
                repo = row['repo'].strip()
                vul_id = row['vul_id'].strip()
                fixed_commit_sha = row['fixed_commit_sha'].strip()
            elif 'vul_id' in row:  # category CSV格式
                vul_id = row['vul_id'].strip()
                repo = row['repo'].strip()
                fixed_commit_sha = row['fixed_commit_sha'].strip()
            else:
                continue
            
            key = (vul_id, repo)
            vul_dict[key] = fixed_commit_sha
    
    return vul_dict

def load_vul_list_exclude_linux(csv_path):
    """加载漏洞列表，排除 torvalds/linux"""
    vul_dict = {}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            vul_id = row['vul_id'].strip()
            repo = row['repo'].strip()
            fixed_commit_sha = row['fixed_commit_sha'].strip()
            
            # 排除 torvalds/linux
            if repo == 'torvalds/linux':
                continue
            
            key = (vul_id, repo)
            vul_dict[key] = fixed_commit_sha
    
    return vul_dict

def main():
    print("=" * 80)
    print("构建扩展数据集")
    print("=" * 80)
    
    # 加载两个数据源
    print("\n加载数据...")
    new_dataset = load_vul_list('inputs/new_dataset.csv')
    print(f"new_dataset.csv: {len(new_dataset)} 个漏洞")
    
    fp_vuls = load_vul_list_exclude_linux('category_2_only_fp.csv')
    print(f"category_2_only_fp.csv (排除torvalds/linux): {len(fp_vuls)} 个漏洞")
    
    # 合并（使用OrderedDict保持插入顺序）
    merged_vuls = OrderedDict()
    
    # 先添加new_dataset中的漏洞
    for key, sha in new_dataset.items():
        merged_vuls[key] = sha
    
    # 再添加FP漏洞（如果已存在则跳过，以new_dataset的数据为准）
    added_fp = 0
    for key, sha in fp_vuls.items():
        if key not in merged_vuls:
            merged_vuls[key] = sha
            added_fp += 1
    
    print(f"\n合并后总数: {len(merged_vuls)} 个漏洞")
    
    # 统计重叠
    overlap = set(new_dataset.keys()) & set(fp_vuls.keys())
    print(f"重叠漏洞: {len(overlap)} 个")
    print(f"仅在new_dataset中: {len(new_dataset) - len(overlap)} 个")
    print(f"仅在FP中: {len(fp_vuls) - len(overlap)} 个")
    print(f"实际添加的FP漏洞: {added_fp} 个")
    
    # 写入扩展数据集
    output_file = 'inputs/extended_dataset.csv'
    print(f"\n写入扩展数据集到 {output_file}...")
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['repo', 'vul_id', 'fixed_commit_sha'])
        
        # 按照 repo, vul_id 排序
        for (vul_id, repo), sha in sorted(merged_vuls.items(), key=lambda x: (x[0][1], x[0][0])):
            writer.writerow([repo, vul_id, sha])
    
    print(f"已写入 {len(merged_vuls)} 个漏洞到 {output_file}")
    
    # 显示一些统计信息
    print("\n" + "=" * 80)
    print("数据集统计:")
    print("=" * 80)
    
    # 按repo统计
    repo_counts = {}
    for (vul_id, repo), sha in merged_vuls.items():
        repo_counts[repo] = repo_counts.get(repo, 0) + 1
    
    print(f"\n按仓库统计 (共 {len(repo_counts)} 个仓库):")
    for repo, count in sorted(repo_counts.items(), key=lambda x: -x[1]):
        print(f"  {repo}: {count}")
    
    # 统计数据来源
    print("\n数据来源统计:")
    print(f"  来自new_dataset: {len(new_dataset)} 个")
    print(f"  来自category_2_only_fp (非Linux): {added_fp} 个")
    print(f"  重叠 (未重复添加): {len(overlap)} 个")
    
    print("\n" + "=" * 80)
    print("完成!")
    print("=" * 80)

if __name__ == '__main__':
    main()
