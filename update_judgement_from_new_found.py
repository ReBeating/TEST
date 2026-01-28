#!/usr/bin/env python3
"""
将New-Found_1day对应漏洞.csv中的TP和FP更新到results/checked_list.csv中
"""

import pandas as pd
import sys

def main():
    # 读取两个CSV文件
    print("读取 New-Found_1day对应漏洞.csv...")
    new_found_df = pd.read_csv('New-Found_1day对应漏洞.csv', encoding='utf-8-sig')
    
    print("读取 results/checked_list.csv...")
    checked_list_df = pd.read_csv('results/checked_list.csv', encoding='utf-8-sig')
    
    # 打印列名以便调试
    print(f"\nNew-Found_1day对应漏洞.csv 列名: {list(new_found_df.columns)}")
    print(f"checked_list.csv 列名: {list(checked_list_df.columns)}")
    
    # 统一列名，创建匹配键
    # New-Found中: target_file, target_func
    # checked_list中: target_file_path, target_func_name
    
    # 为new_found创建匹配键
    new_found_df['match_key'] = (
        new_found_df['vul_id'].astype(str) + '|||' +
        new_found_df['repo'].astype(str) + '|||' +
        new_found_df['fixed_commit_sha'].astype(str) + '|||' +
        new_found_df['target_file'].astype(str) + '|||' +
        new_found_df['target_func'].astype(str)
    )
    
    # 为checked_list创建匹配键
    checked_list_df['match_key'] = (
        checked_list_df['vul_id'].astype(str) + '|||' +
        checked_list_df['repo'].astype(str) + '|||' +
        checked_list_df['fixed_commit_sha'].astype(str) + '|||' +
        checked_list_df['target_file_path'].astype(str) + '|||' +
        checked_list_df['target_func_name'].astype(str)
    )
    
    # 创建从match_key到judgement的映射
    new_found_judgement_map = dict(zip(new_found_df['match_key'], new_found_df['judgement']))
    
    # 统计信息
    update_count = 0
    no_change_count = 0
    not_found_count = 0
    
    # 更新checked_list中的judgement
    original_judgements = checked_list_df['judgement'].copy()
    
    for idx, row in checked_list_df.iterrows():
        match_key = row['match_key']
        if match_key in new_found_judgement_map:
            new_judgement = new_found_judgement_map[match_key]
            old_judgement = row['judgement']
            
            # 只更新有效的judgement值（TP或FP）
            if pd.notna(new_judgement) and new_judgement in ['TP', 'FP']:
                if old_judgement != new_judgement:
                    checked_list_df.at[idx, 'judgement'] = new_judgement
                    update_count += 1
                    print(f"更新: {row['vul_id']} - {row['target_file_path']} - {row['target_func_name']}: {old_judgement} -> {new_judgement}")
                else:
                    no_change_count += 1
        else:
            not_found_count += 1
    
    # 删除临时的match_key列
    checked_list_df = checked_list_df.drop(columns=['match_key'])
    
    # 保存更新后的checked_list
    print(f"\n正在保存到 results/checked_list.csv...")
    checked_list_df.to_csv('results/checked_list.csv', index=False, encoding='utf-8-sig')
    
    # 打印统计信息
    print(f"\n更新完成!")
    print(f"总记录数: {len(checked_list_df)}")
    print(f"更新的记录: {update_count}")
    print(f"未变更的记录: {no_change_count}")
    print(f"未找到匹配的记录: {not_found_count}")
    print(f"\nNew-Found_1day对应漏洞.csv中的记录数: {len(new_found_df)}")

if __name__ == '__main__':
    main()
