#!/usr/bin/env python3
"""
验证最终的更新结果
"""

import pandas as pd

def main():
    # 读取两个CSV文件
    new_found_df = pd.read_csv('New-Found_1day对应漏洞.csv', encoding='utf-8-sig')
    checked_list_df = pd.read_csv('results/checked_list.csv', encoding='utf-8-sig')
    
    print(f"New-Found_1day对应漏洞.csv 记录数: {len(new_found_df)}")
    print(f"更新后的 checked_list.csv 记录数: {len(checked_list_df)}")
    
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
    
    # 检查每条New-Found记录是否都在checked_list中
    all_matched = True
    mismatched_judgements = []
    
    for idx, row in new_found_df.iterrows():
        match_key = row['match_key']
        matching_rows = checked_list_df[checked_list_df['match_key'] == match_key]
        
        if len(matching_rows) == 0:
            print(f"\n⚠️  未找到匹配: {row['vul_id']} - {row['target_file']} - {row['target_func']}")
            all_matched = False
        else:
            # 检查判断是否一致
            checked_judgement = matching_rows.iloc[0]['judgement']
            new_found_judgement = row['judgement']
            if checked_judgement != new_found_judgement:
                mismatched_judgements.append({
                    'vul_id': row['vul_id'],
                    'file': row['target_file'],
                    'func': row['target_func'],
                    'new_found': new_found_judgement,
                    'checked_list': checked_judgement
                })
    
    print("\n" + "="*60)
    if all_matched:
        print("✅ 所有 New-Found 记录都已包含在 checked_list.csv 中")
    else:
        print("❌ 有些记录未能匹配")
    
    if not mismatched_judgements:
        print("✅ 所有匹配记录的判断都一致")
    else:
        print(f"⚠️  发现 {len(mismatched_judgements)} 条判断不一致的记录:")
        for item in mismatched_judgements:
            print(f"\n  {item['vul_id']} - {item['file']} - {item['func']}")
            print(f"    New-Found: {item['new_found']}")
            print(f"    Checked-List: {item['checked_list']}")
    
    # 统计TP和FP数量
    new_found_stats = new_found_df['judgement'].value_counts()
    
    # 获取checked_list中New-Found对应的记录
    checked_list_matched = checked_list_df[checked_list_df['match_key'].isin(new_found_df['match_key'])]
    checked_list_stats = checked_list_matched['judgement'].value_counts()
    
    print("\n" + "="*60)
    print("判断统计:")
    print(f"\nNew-Found_1day对应漏洞.csv:")
    for judgement, count in new_found_stats.items():
        print(f"  {judgement}: {count}")
    
    print(f"\nchecked_list.csv (对应的New-Found记录):")
    for judgement, count in checked_list_stats.items():
        print(f"  {judgement}: {count}")
    
    print("\n" + "="*60)
    print("✅ 更新完成！所有 New-Found 中的 TP 和 FP 判断已同步到 checked_list.csv")

if __name__ == '__main__':
    main()
