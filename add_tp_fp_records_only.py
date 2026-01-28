#!/usr/bin/env python3
"""
只将New-Found_1day对应漏洞.csv中judgement为TP或FP的记录添加到checked_list.csv
忽略judgement为空值的记录
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
    
    # 只保留judgement为TP或FP的记录
    valid_judgements = new_found_df[new_found_df['judgement'].isin(['TP', 'FP'])].copy()
    print(f"\nNew-Found中有效判断(TP/FP)的记录数: {len(valid_judgements)}")
    print(f"  TP: {len(valid_judgements[valid_judgements['judgement'] == 'TP'])}")
    print(f"  FP: {len(valid_judgements[valid_judgements['judgement'] == 'FP'])}")
    
    # 创建匹配键
    valid_judgements['match_key'] = (
        valid_judgements['vul_id'].astype(str) + '|||' +
        valid_judgements['repo'].astype(str) + '|||' +
        valid_judgements['fixed_commit_sha'].astype(str) + '|||' +
        valid_judgements['target_file'].astype(str) + '|||' +
        valid_judgements['target_func'].astype(str)
    )
    
    checked_list_df['match_key'] = (
        checked_list_df['vul_id'].astype(str) + '|||' +
        checked_list_df['repo'].astype(str) + '|||' +
        checked_list_df['fixed_commit_sha'].astype(str) + '|||' +
        checked_list_df['target_file_path'].astype(str) + '|||' +
        checked_list_df['target_func_name'].astype(str)
    )
    
    # 1. 首先更新已存在记录的判断
    update_count = 0
    for idx, row in checked_list_df.iterrows():
        match_key = row['match_key']
        matching_rows = valid_judgements[valid_judgements['match_key'] == match_key]
        if len(matching_rows) > 0:
            new_judgement = matching_rows.iloc[0]['judgement']
            old_judgement = row['judgement']
            if old_judgement != new_judgement:
                checked_list_df.at[idx, 'judgement'] = new_judgement
                update_count += 1
                print(f"更新: {row['vul_id']} - {row['target_file_path']} - {row['target_func_name']}: {old_judgement} -> {new_judgement}")
    
    print(f"\n已更新 {update_count} 条现有记录的判断")
    
    # 2. 添加新记录（只添加TP或FP的）
    existing_keys = set(checked_list_df['match_key'])
    new_records = []
    
    for idx, row in valid_judgements.iterrows():
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
    
    print(f"\n找到 {len(new_records)} 条新记录需要添加（只包含TP和FP）")
    
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
        print(f"更新记录数: {update_count}")
        print(f"新增记录数: {len(new_records)}")
        print(f"更新后总记录数: {len(updated_checked_list_df)}")
        
        # 统计新增记录的TP/FP分布
        new_tp = sum(1 for r in new_records if r['judgement'] == 'TP')
        new_fp = sum(1 for r in new_records if r['judgement'] == 'FP')
        print(f"\n新增记录中:")
        print(f"  TP: {new_tp}")
        print(f"  FP: {new_fp}")
        
        # 显示前5条新增的记录
        print("\n前5条新增记录示例:")
        for i, record in enumerate(new_records[:5], 1):
            print(f"\n{i}. {record['vul_id']}")
            print(f"   Repo: {record['repo']}")
            print(f"   File: {record['target_file_path']}")
            print(f"   Func: {record['target_func_name']}")
            print(f"   判断: {record['judgement']}")
    else:
        print("\n没有新记录需要添加。")
        # 删除临时的match_key列并保存（如果有更新）
        if update_count > 0:
            checked_list_df = checked_list_df.drop(columns=['match_key'])
            checked_list_df.to_csv('results/checked_list.csv', index=False, encoding='utf-8-sig')
        else:
            print("checked_list.csv 未发生任何变化。")

if __name__ == '__main__':
    main()
