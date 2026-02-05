#!/usr/bin/env python3
"""
比较validation_report_0day.csv和verdict_0day_res.csv
找出validation中的TP但没在verdict中出现的记录
"""
import csv

# 读取validation_report中所有标记为TP的条目
validation_tps = []
with open('validation_report_0day.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['judgement'] == 'TP':
            validation_tps.append({
                'vul_id': row['vul_id'],
                'repo': row['repo'],
                'patch_file': row['patch_file'],
                'patch_func': row['patch_func'],
                'target_file': row['target_file'],
                'target_func': row['target_func']
            })

print(f"validation_report中标记为TP的记录总数: {len(validation_tps)}")

# 读取verdict中的所有条目（创建一个唯一标识）
verdict_entries = set()
with open('results/0day/verdict_0day_res.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # 使用多个字段组合作为唯一标识
        entry_key = (
            row['vul_id'],
            row['repo'],
            row['patch_file'],
            row['patch_func'],
            row['target_file'],
            row['target_func']
        )
        verdict_entries.add(entry_key)

print(f"verdict结果中的记录总数: {len(verdict_entries)}")

# 找出validation中的TP但不在verdict中的记录
missing_tps = []
for tp in validation_tps:
    entry_key = (
        tp['vul_id'],
        tp['repo'],
        tp['patch_file'],
        tp['patch_func'],
        tp['target_file'],
        tp['target_func']
    )
    if entry_key not in verdict_entries:
        missing_tps.append(tp)

print(f"\nvalidation中的TP但不在verdict中的记录数: {len(missing_tps)}")

# 按漏洞ID分组统计
from collections import defaultdict
missing_by_vul = defaultdict(list)
for tp in missing_tps:
    missing_by_vul[tp['vul_id']].append(tp)

print(f"\n涉及的漏洞数量: {len(missing_by_vul)}")

# 显示详细信息
print("\n" + "="*80)
print("缺失的TP详细信息:")
print("="*80)

for vul_id in sorted(missing_by_vul.keys()):
    tps = missing_by_vul[vul_id]
    print(f"\n【{vul_id}】({tps[0]['repo']}) - 缺失{len(tps)}个TP:")
    for i, tp in enumerate(tps, 1):
        print(f"  {i}. {tp['patch_file']}::{tp['patch_func']} -> {tp['target_file']}::{tp['target_func']}")

# 保存到文件
with open('missing_tps_from_verdict.csv', 'w', newline='') as f:
    fieldnames = ['vul_id', 'repo', 'patch_file', 'patch_func', 'target_file', 'target_func']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(missing_tps)

print(f"\n详细列表已保存到: missing_tps_from_verdict.csv")

# 统计概要
print("\n" + "="*80)
print("统计概要:")
print("="*80)
print(f"validation中的TP总数: {len(validation_tps)}")
print(f"verdict中的记录总数: {len(verdict_entries)}")
print(f"缺失的TP数量: {len(missing_tps)}")
print(f"缺失率: {len(missing_tps)/len(validation_tps)*100:.1f}%")
