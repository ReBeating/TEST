#!/usr/bin/env python3
"""合并 New-Found_0126.csv 的判断结果到 checked_list.csv"""

import csv
from collections import defaultdict

def main():
    # 读取 New-Found_0126.csv 并建立索引
    new_found_judgements = {}
    with open('New-Found_0126.csv', 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 使用关键字段构建唯一键
            key = (
                row['vul_id'],
                row['repo'],
                row['fixed_commit_sha'],
                row['target_file'],
                row['target_func']
            )
            new_found_judgements[key] = row['judgement']
    
    print(f"从 New-Found_0126.csv 读取了 {len(new_found_judgements)} 条记录")
    
    # 读取 checked_list.csv
    checked_list = []
    updated_count = 0
    with open('results/checked_list.csv', 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        for row in reader:
            # 构建匹配键
            key = (
                row['vul_id'],
                row['repo'],
                row['fixed_commit_sha'],
                row['target_file_path'],
                row['target_func_name']
            )
            
            # 如果在 new_found 中找到匹配，更新判断
            if key in new_found_judgements:
                row['judgement'] = new_found_judgements[key]
                updated_count += 1
            
            checked_list.append(row)
    
    print(f"从 checked_list.csv 读取了 {len(checked_list)} 条记录")
    print(f"更新了 {updated_count} 条记录的判断结果")
    
    # 排序：按 vul_id, repo, fixed_commit_sha, target_file_path, target_func_name 排序
    checked_list.sort(key=lambda x: (
        x['vul_id'],
        x['repo'],
        x['fixed_commit_sha'],
        x['target_file_path'],
        x['target_func_name']
    ))
    
    # 写回 checked_list.csv
    with open('results/checked_list.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(checked_list)
    
    print(f"已将更新后的数据写回 checked_list.csv")
    
    # 统计判断结果
    judgement_stats = defaultdict(int)
    for row in checked_list:
        judgement_stats[row['judgement']] += 1
    
    print("\n判断结果统计:")
    for judgement, count in sorted(judgement_stats.items()):
        print(f"  {judgement}: {count}")

if __name__ == '__main__':
    main()
