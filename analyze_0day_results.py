#!/usr/bin/env python3
"""
统计merged_0day_vul_list_sampled.csv中的漏洞在verdict_0day_res.csv中的TP和FP数量
"""
import csv
from collections import defaultdict

# 读取输入的漏洞列表
input_vuls = set()
with open('inputs/merged_0day_vul_list_sampled.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        input_vuls.add(row['vul_id'])

print(f"输入文件中的漏洞总数: {len(input_vuls)}")

# 读取verdict结果并统计
vul_judgements = defaultdict(list)
with open('results/0day/verdict_0day_res.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        vul_id = row['vul_id']
        judgement = row['judgement']
        if vul_id in input_vuls:
            vul_judgements[vul_id].append(judgement)

# 统计TP和FP
tp_count = 0
fp_count = 0
vuls_with_tp = set()
vuls_with_fp = set()

for vul_id, judgements in vul_judgements.items():
    for judgement in judgements:
        if judgement == 'TP':
            tp_count += 1
            vuls_with_tp.add(vul_id)
        elif judgement == 'FP':
            fp_count += 1
            vuls_with_fp.add(vul_id)

# 统计在verdict中出现的漏洞数
vuls_in_verdict = len(vul_judgements)
vuls_not_in_verdict = len(input_vuls) - vuls_in_verdict

print(f"\n在verdict结果中出现的漏洞数: {vuls_in_verdict}")
print(f"在verdict结果中未出现的漏洞数: {vuls_not_in_verdict}")

print(f"\n判断结果统计:")
print(f"  TP (True Positive) 总数: {tp_count}")
print(f"  FP (False Positive) 总数: {fp_count}")
print(f"  总判断数: {tp_count + fp_count}")

print(f"\n漏洞级别统计:")
print(f"  至少有一个TP的漏洞数: {len(vuls_with_tp)}")
print(f"  至少有一个FP的漏洞数: {len(vuls_with_fp)}")
print(f"  同时有TP和FP的漏洞数: {len(vuls_with_tp & vuls_with_fp)}")
print(f"  只有TP的漏洞数: {len(vuls_with_tp - vuls_with_fp)}")
print(f"  只有FP的漏洞数: {len(vuls_with_fp - vuls_with_tp)}")

# 详细统计每个漏洞的TP/FP数量
print(f"\n每个漏洞的详细统计:")
vul_stats = []
for vul_id in sorted(input_vuls):
    if vul_id in vul_judgements:
        judgements = vul_judgements[vul_id]
        tp = judgements.count('TP')
        fp = judgements.count('FP')
        total = len(judgements)
        vul_stats.append({
            'vul_id': vul_id,
            'tp': tp,
            'fp': fp,
            'total': total,
            'tp_ratio': tp / total if total > 0 else 0
        })

# 按TP数量降序排列
vul_stats.sort(key=lambda x: (x['tp'], -x['fp']), reverse=True)

print(f"\n前20个漏洞的TP/FP分布:")
print(f"{'漏洞ID':<25} {'TP':<5} {'FP':<5} {'总数':<5} {'TP比例':<10}")
print("-" * 55)
for stat in vul_stats[:20]:
    print(f"{stat['vul_id']:<25} {stat['tp']:<5} {stat['fp']:<5} {stat['total']:<5} {stat['tp_ratio']:<10.2%}")

# 保存完整统计到文件
with open('0day_tp_fp_statistics.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['vul_id', 'tp', 'fp', 'total', 'tp_ratio'])
    writer.writeheader()
    writer.writerows(vul_stats)

print(f"\n完整统计已保存到: 0day_tp_fp_statistics.csv")
