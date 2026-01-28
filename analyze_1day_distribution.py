#!/usr/bin/env python3
"""
分析 inputs/1day_vul_list.csv 中的漏洞在5个类别中的分布
"""

import csv
from collections import defaultdict

def load_vul_list(csv_path):
    """加载漏洞列表，返回{(vul_id, repo)}的集合"""
    vul_set = set()
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            repo = row['repo'].strip()
            vul_id = row['vul_id'].strip()
            vul_set.add((vul_id, repo))
    
    return vul_set

def load_category(csv_path):
    """加载类别CSV，返回{(vul_id, repo)}的集合"""
    vul_set = set()
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            vul_id = row['vul_id'].strip()
            repo = row['repo'].strip()
            vul_set.add((vul_id, repo))
    
    return vul_set

def main():
    print("=" * 80)
    print("分析 1day_vul_list.csv 中的漏洞在5个类别中的分布")
    print("=" * 80)
    
    # 加载1day漏洞列表
    oneday_vuls = load_vul_list('inputs/1day_vul_list.csv')
    print(f"\n1day_vul_list.csv 中的漏洞总数: {len(oneday_vuls)}")
    
    # 加载5个类别
    categories = {
        '类别1 - 存在TP': load_category('category_1_has_tp.csv'),
        '类别2 - 只有FP': load_category('category_2_only_fp.csv'),
        '类别3 - 所有is_vulnerable=false': load_category('category_3_all_false.csv'),
        '类别4 - findings不存在或为空': load_category('category_4_no_findings.csv'),
        '类别5 - 不知道TP还是FP但有is_vulnerable=true': load_category('category_5_unknown.csv'),
    }
    
    # 统计每个1day漏洞在哪个类别中
    distribution = defaultdict(list)
    not_found = []
    
    for vul_key in oneday_vuls:
        found = False
        for cat_name, cat_vuls in categories.items():
            if vul_key in cat_vuls:
                distribution[cat_name].append(vul_key)
                found = True
                break  # 一个漏洞应该只在一个类别中
        
        if not found:
            not_found.append(vul_key)
    
    # 输出结果
    print("\n" + "=" * 80)
    print("分布统计:")
    print("=" * 80)
    
    for cat_name in categories.keys():
        count = len(distribution[cat_name])
        percentage = count / len(oneday_vuls) * 100 if oneday_vuls else 0
        print(f"\n{cat_name}:")
        print(f"  数量: {count} ({percentage:.1f}%)")
        
        if count > 0 and count <= 20:  # 如果数量较少，列出所有漏洞
            print(f"  漏洞列表:")
            for vul_id, repo in sorted(distribution[cat_name]):
                print(f"    - {vul_id} ({repo})")
    
    # 未找到的漏洞
    if not_found:
        print(f"\n未在任何类别中找到的漏洞:")
        print(f"  数量: {len(not_found)} ({len(not_found)/len(oneday_vuls)*100:.1f}%)")
        if len(not_found) <= 20:
            print(f"  漏洞列表:")
            for vul_id, repo in sorted(not_found):
                print(f"    - {vul_id} ({repo})")
    
    print("\n" + "=" * 80)
    print("总结:")
    print("=" * 80)
    print(f"1day漏洞总数: {len(oneday_vuls)}")
    print(f"已分类: {len(oneday_vuls) - len(not_found)} ({(len(oneday_vuls) - len(not_found))/len(oneday_vuls)*100:.1f}%)")
    print(f"未分类: {len(not_found)} ({len(not_found)/len(oneday_vuls)*100 if oneday_vuls else 0:.1f}%)")
    print("=" * 80)

if __name__ == '__main__':
    main()
