#!/usr/bin/env python3
"""
找出verdict_0day_res.csv中标记为TP的记录（属于merged_0day_vul_list_sampled.csv的漏洞）
但这些TP没有出现在validation_report_0day.csv中
"""
import csv
from collections import defaultdict

# 1. 读取输入文件中的漏洞列表
input_vuls = set()
with open('inputs/merged_0day_vul_list_sampled.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        input_vuls.add(row['vul_id'])

print(f"输入文件中的漏洞总数: {len(input_vuls)}")

# 2. 读取verdict中标记为TP且属于输入漏洞列表的记录
verdict_tps = []
with open('results/0day/verdict_0day_res.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['judgement'] == 'TP' and row['vul_id'] in input_vuls:
            verdict_tps.append({
                'vul_id': row['vul_id'],
                'repo': row['repo'],
                'fixed_commit_sha': row['fixed_commit_sha'],
                'patch_file': row['patch_file'],
                'patch_func': row['patch_func'],
                'target_file': row['target_file'],
                'target_func': row['target_func'],
                'CWE': row['CWE'],
                'vuln_type': row['vuln_type']
            })

print(f"verdict中属于输入漏洞且标记为TP的记录数: {len(verdict_tps)}")

# 3. 读取validation_report中的所有条目
validation_entries = set()
with open('validation_report_0day.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # 使用关键字段组合作为唯一标识
        entry_key = (
            row['vul_id'],
            row['repo'],
            row['fixed_commit_sha'],
            row['patch_file'],
            row['patch_func'],
            row['target_file'],
            row['target_func']
        )
        validation_entries.add(entry_key)

print(f"validation_report中的记录总数: {len(validation_entries)}")

# 4. 找出verdict中的TP但不在validation中的记录
missing_in_validation = []
for tp in verdict_tps:
    entry_key = (
        tp['vul_id'],
        tp['repo'],
        tp['fixed_commit_sha'],
        tp['patch_file'],
        tp['patch_func'],
        tp['target_file'],
        tp['target_func']
    )
    if entry_key not in validation_entries:
        missing_in_validation.append(tp)

print(f"\nverdict中的TP但不在validation中的记录数: {len(missing_in_validation)}")

# 按漏洞ID分组
missing_by_vul = defaultdict(list)
for tp in missing_in_validation:
    missing_by_vul[tp['vul_id']].append(tp)

print(f"涉及的漏洞数量: {len(missing_by_vul)}")

# 显示详细信息
print("\n" + "="*100)
print("verdict中的TP但不在validation_report中的详细信息:")
print("="*100)

for vul_id in sorted(missing_by_vul.keys()):
    tps = missing_by_vul[vul_id]
    print(f"\n【{vul_id}】({tps[0]['repo']}) - 缺失{len(tps)}个TP:")
    for i, tp in enumerate(tps, 1):
        print(f"  {i}. {tp['patch_file']}::{tp['patch_func']}")
        print(f"     -> {tp['target_file']}::{tp['target_func']}")
        print(f"     CWE: {tp['CWE']}, Type: {tp['vuln_type']}")

# 保存到文件
with open('verdict_tps_missing_in_validation.csv', 'w', newline='') as f:
    fieldnames = ['vul_id', 'repo', 'fixed_commit_sha', 'patch_file', 'patch_func', 
                  'target_file', 'target_func', 'CWE', 'vuln_type']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(missing_in_validation)

print(f"\n详细列表已保存到: verdict_tps_missing_in_validation.csv")

# 统计概要
print("\n" + "="*100)
print("统计概要:")
print("="*100)
print(f"verdict中的TP总数（属于输入漏洞）: {len(verdict_tps)}")
print(f"validation_report中的记录总数: {len(validation_entries)}")
print(f"缺失的TP数量: {len(missing_in_validation)}")
print(f"覆盖率: {(len(verdict_tps) - len(missing_in_validation))/len(verdict_tps)*100:.1f}%")

# 按漏洞统计缺失数量
print("\n按漏洞统计缺失的TP数量:")
vul_missing_counts = [(vul_id, len(tps)) for vul_id, tps in missing_by_vul.items()]
vul_missing_counts.sort(key=lambda x: x[1], reverse=True)

for vul_id, count in vul_missing_counts:
    print(f"  {vul_id}: {count}个TP")
