#!/usr/bin/env python3
"""
构建新数据集：合并 1day_vul_list.csv 和 category_1_has_tp.csv
"""

import csv
from collections import OrderedDict

def load_vul_list(csv_path):
    """加载漏洞列表，返回{(vul_id, repo): fixed_commit_sha}的字典"""
    vul_dict = {}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            repo = row['repo'].strip()
            vul_id = row['vul_id'].strip()
            fixed_commit_sha = row['fixed_commit_sha'].strip()
            
            key = (vul_id, repo)
            vul_dict[key] = fixed_commit_sha
    
    return vul_dict

def main():
    print("=" * 80)
    print("构建新数据集")
    print("=" * 80)
    
    # 加载两个数据源
    print("\n加载数据...")
    oneday_vuls = load_vul_list('inputs/1day_vul_list.csv')
    print(f"1day_vul_list.csv: {len(oneday_vuls)} 个漏洞")
    
    tp_vuls = load_vul_list('category_1_has_tp.csv')
    print(f"category_1_has_tp.csv: {len(tp_vuls)} 个漏洞")
    
    # 合并（使用OrderedDict保持插入顺序）
    merged_vuls = OrderedDict()
    
    # 先添加1day漏洞
    for key, sha in oneday_vuls.items():
        merged_vuls[key] = sha
    
    # 再添加TP漏洞（如果已存在则覆盖，以TP的数据为准）
    for key, sha in tp_vuls.items():
        merged_vuls[key] = sha
    
    print(f"\n合并后总数: {len(merged_vuls)} 个漏洞")
    
    # 统计重叠
    overlap = set(oneday_vuls.keys()) & set(tp_vuls.keys())
    print(f"重叠漏洞: {len(overlap)} 个")
    print(f"仅在1day中: {len(oneday_vuls) - len(overlap)} 个")
    print(f"仅在TP中: {len(tp_vuls) - len(overlap)} 个")
    
    # 写入新数据集
    output_file = 'inputs/new_dataset.csv'
    print(f"\n写入新数据集到 {output_file}...")
    
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
    
    print("\n" + "=" * 80)
    print("完成!")
    print("=" * 80)

if __name__ == '__main__':
    main()
