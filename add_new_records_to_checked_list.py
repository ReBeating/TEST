#!/usr/bin/env python3
"""
将New-Found_1day对应漏洞.csv中不在checked_list.csv中的记录添加进去
"""

import pandas as pd
import numpy as np

def main():
    # 读取两个CSV文件
    print("读取 New-Found_1day对应漏洞.csv...")
    new_found_df = pd.read_csv('New-Found_1day对应漏洞.csv', encoding='utf-8-sig')
    
    print("读取 results/checked_list.csv...")
    checked_list_df = pd.read_csv('results/checked_list.csv', encoding='utf-8-sig')
    
    # 记录原始数量
    original_count = len(checked_list_df)
    print(f"checked_list.csv 原始记录数: {original_count}")
    
    # 创建匹配键
    new_found_df['match_key'] = (
        new_found_df['vul_id'].astype(str) + '|||' +
        new_found_df['repo'].astype(str) + '|||' +
        new_found_df['fixed_commit_sha'].astype(str) + '|||' +
        new_found_df['target_file'].astype(str) + '|||' +
        new_found_df['target_func'].astype(str)
    )
    
    checked_list_df['match_key'] = (
        checked_list_df['vul_id'].astype(str) + '|||' +
        checked_list_df['repo'].astype(str) + '|||' +
        checked_list_df['fixed_commit_sha'].astype(str) + '|||' +
        checked_list_df['target_file_path'].astype(str) + '|||' +
        checked_list_df['target_func_name'].astype(str)
    )
    
    # 找出New-Found中不在checked_list中的记录
    existing_keys = set(checked_list_df['match_key'])
    new_records = []
    
    for idx, row in new_found_df.iterrows():
        if row['match_key'] not in existing_keys:
            # 创建新记录，映射列名
            new_record = {
                'vul_id': row['vul_id'],
                'repo': row['repo'],
                'fixed_commit_sha': row['fixed_commit_sha'],
                'target_file_path': row['target_file'],  # 列名映射
                'target_func_name': row['target_func'],  # 列名映射
                'judgement': row['judgement'],
                'reported': np.nan,  # 新记录这些字段为空
                'confirmed': np.nan,
                'requested': np.nan,
                'received': np.nan
            }
            new_records.append(new_record)
    
    print(f"\n找到 {len(new_records)} 条新记录需要添加")
    
    if new_records:
        # 创建新记录的DataFrame
        new_records_df = pd.DataFrame(new_records)
        
        # 删除临时的match_key列
        checked_list_df = checked_list_df.drop(columns=['match_key'])
        
        # 将新记录添加到checked_list
        updated_checked_list_df = pd.concat([checked_list_df, new_records_df], ignore_index=True)
        
        # 保存更新后的文件
        print(f"\n正在保存到 results/checked_list.csv...")
        updated_checked_list_df.to_csv('results/checked_list.csv', index=False, encoding='utf-8-sig')
        
        print(f"\n更新完成!")
        print(f"原始记录数: {original_count}")
        print(f"新增记录数: {len(new_records)}")
        print(f"更新后总记录数: {len(updated_checked_list_df)}")
        
        # 显示前5条新增的记录
        print("\n前5条新增记录示例:")
        for i, record in enumerate(new_records[:5], 1):
            print(f"\n{i}. {record['vul_id']}")
            print(f"   Repo: {record['repo']}")
            print(f"   File: {record['target_file_path']}")
            print(f"   Func: {record['target_func_name']}")
            print(f"   判断: {record['judgement']}")
    else:
        print("\n没有新记录需要添加，所有记录都已存在。")
        # 删除临时的match_key列并保存
        checked_list_df = checked_list_df.drop(columns=['match_key'])
        checked_list_df.to_csv('results/checked_list.csv', index=False, encoding='utf-8-sig')

if __name__ == '__main__':
    main()
