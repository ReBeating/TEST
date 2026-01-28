#!/usr/bin/env python3
"""
整理 New-Found_0126.csv 中的 TP 和 FP 到 checked_list.csv
"""

import pandas as pd
import sys

def main():
    # 读取 New-Found_0126.csv
    print("读取 New-Found_0126.csv...")
    new_found = pd.read_csv('New-Found_已整理.csv')
    
    # 读取现有的 checked_list.csv
    print("读取 results/checked_list.csv...")
    checked_list = pd.read_csv('results/checked_list.csv')
    
    # 过滤出 TP 和 FP 的记录
    print(f"原始记录数: {len(new_found)}")
    filtered = new_found[new_found['judgement'].isin(['TP', 'FP'])].copy()
    print(f"TP/FP 记录数: {len(filtered)}")
    
    # 转换列名以匹配 checked_list 格式
    # New-Found: target_file -> checked_list: target_file_path
    # New-Found: target_func -> checked_list: target_func_name
    filtered = filtered.rename(columns={
        'target_file': 'target_file_path',
        'target_func': 'target_func_name'
    })
    
    # 选择需要的列
    columns_to_keep = ['vul_id', 'repo', 'fixed_commit_sha', 'target_file_path', 
                       'target_func_name', 'judgement']
    
    # 添加缺失的列（reported, confirmed, requested, received）
    for col in ['reported', 'confirmed', 'requested', 'received']:
        if col not in filtered.columns:
            filtered[col] = ''
    
    # 选择最终需要的列
    all_columns = columns_to_keep + ['reported', 'confirmed', 'requested', 'received']
    new_records = filtered[all_columns].copy()
    
    # 合并数据
    print(f"checked_list 原有记录数: {len(checked_list)}")
    combined = pd.concat([checked_list, new_records], ignore_index=True)
    print(f"合并后记录数: {len(combined)}")
    
    # 去重 - 基于关键字段
    # 使用 vul_id, repo, target_file_path, target_func_name 作为去重依据
    dedup_columns = ['vul_id', 'repo', 'target_file_path', 'target_func_name']
    
    # 去重时保留第一次出现的记录（保留 checked_list 中已有的记录）
    combined_dedup = combined.drop_duplicates(subset=dedup_columns, keep='first')
    print(f"去重后记录数: {len(combined_dedup)}")
    print(f"删除了 {len(combined) - len(combined_dedup)} 条重复记录")
    
    # 排序：先按 repo，再按 vul_id
    print("按 repo, vul_id 排序...")
    combined_sorted = combined_dedup.sort_values(by=['repo', 'vul_id'], 
                                                   ignore_index=True)
    
    # 保存结果
    print("保存到 results/checked_list.csv...")
    combined_sorted.to_csv('results/checked_list.csv', index=False)
    
    print("\n完成！")
    print(f"最终记录数: {len(combined_sorted)}")
    print(f"  - TP 记录: {len(combined_sorted[combined_sorted['judgement'] == 'TP'])}")
    print(f"  - FP 记录: {len(combined_sorted[combined_sorted['judgement'] == 'FP'])}")
    
    # 显示每个 repo 的统计
    print("\n各仓库记录数统计:")
    repo_stats = combined_sorted.groupby('repo').size().sort_values(ascending=False)
    for repo, count in repo_stats.items():
        print(f"  {repo}: {count}")

if __name__ == '__main__':
    main()
