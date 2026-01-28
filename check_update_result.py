#!/usr/bin/env python3
"""
检查更新结果，显示匹配和未匹配的详细信息
"""

import pandas as pd

def main():
    # 读取两个CSV文件
    new_found_df = pd.read_csv('New-Found_1day对应漏洞.csv', encoding='utf-8-sig')
    checked_list_df = pd.read_csv('results/checked_list.csv', encoding='utf-8-sig')
    
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
    
    # 查找匹配的记录
    matched_records = []
    for idx, row in new_found_df.iterrows():
        match_key = row['match_key']
        matching_rows = checked_list_df[checked_list_df['match_key'] == match_key]
        if len(matching_rows) > 0:
            for _, checked_row in matching_rows.iterrows():
                matched_records.append({
                    'vul_id': row['vul_id'],
                    'repo': row['repo'],
                    'target_file': row['target_file'],
                    'target_func': row['target_func'],
                    'new_found_judgement': row['judgement'],
                    'checked_list_judgement': checked_row['judgement']
                })
    
    print(f"总共匹配到 {len(matched_records)} 条记录\n")
    
    # 显示前10条匹配记录
    if matched_records:
        print("前10条匹配记录:")
        for i, record in enumerate(matched_records[:10], 1):
            print(f"\n{i}. {record['vul_id']}")
            print(f"   Repo: {record['repo']}")
            print(f"   File: {record['target_file']}")
            print(f"   Func: {record['target_func']}")
            print(f"   New-Found判断: {record['new_found_judgement']}")
            print(f"   Checked-List判断: {record['checked_list_judgement']}")
            print(f"   是否一致: {'是' if record['new_found_judgement'] == record['checked_list_judgement'] else '否'}")
    
    # 统计判断一致性
    consistent = sum(1 for r in matched_records if r['new_found_judgement'] == r['checked_list_judgement'])
    inconsistent = len(matched_records) - consistent
    
    print(f"\n\n统计信息:")
    print(f"判断一致的记录: {consistent}")
    print(f"判断不一致的记录: {inconsistent}")
    
    # 显示判断不一致的记录
    if inconsistent > 0:
        print("\n判断不一致的记录:")
        for record in matched_records:
            if record['new_found_judgement'] != record['checked_list_judgement']:
                print(f"\n{record['vul_id']} - {record['target_file']} - {record['target_func']}")
                print(f"  New-Found: {record['new_found_judgement']}")
                print(f"  Checked-List: {record['checked_list_judgement']}")
    
    # 显示New-Found中未匹配到的记录（前10个）
    unmatched_in_new_found = []
    for idx, row in new_found_df.iterrows():
        match_key = row['match_key']
        if match_key not in checked_list_df['match_key'].values:
            unmatched_in_new_found.append({
                'vul_id': row['vul_id'],
                'repo': row['repo'],
                'target_file': row['target_file'],
                'target_func': row['target_func'],
                'judgement': row['judgement']
            })
    
    print(f"\n\nNew-Found中未在Checked-List找到的记录: {len(unmatched_in_new_found)}")
    if unmatched_in_new_found:
        print("\n前10条未匹配记录:")
        for i, record in enumerate(unmatched_in_new_found[:10], 1):
            print(f"\n{i}. {record['vul_id']}")
            print(f"   Repo: {record['repo']}")
            print(f"   File: {record['target_file']}")
            print(f"   Func: {record['target_func']}")
            print(f"   判断: {record['judgement']}")

if __name__ == '__main__':
    main()
