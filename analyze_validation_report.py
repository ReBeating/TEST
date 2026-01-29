#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计validation_report_0day.csv中各个漏洞类型和软件的TP/FP占比
"""

import pandas as pd
from collections import defaultdict

def analyze_validation_report(csv_file):
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    
    print("=" * 80)
    print("漏洞类型 TP/FP 统计")
    print("=" * 80)
    
    # 按漏洞类型统计
    vuln_type_stats = defaultdict(lambda: {'TP': 0, 'FP': 0})
    for _, row in df.iterrows():
        vuln_type = row['vuln_type']
        judgement = row['judgement']
        vuln_type_stats[vuln_type][judgement] += 1
    
    # 打印漏洞类型统计结果
    vuln_results = []
    for vuln_type in sorted(vuln_type_stats.keys()):
        tp = vuln_type_stats[vuln_type]['TP']
        fp = vuln_type_stats[vuln_type]['FP']
        total = tp + fp
        tp_ratio = tp / total * 100 if total > 0 else 0
        fp_ratio = fp / total * 100 if total > 0 else 0
        vuln_results.append({
            '漏洞类型': vuln_type,
            'TP': tp,
            'FP': fp,
            '总数': total,
            'TP占比(%)': f'{tp_ratio:.2f}',
            'FP占比(%)': f'{fp_ratio:.2f}'
        })
    
    vuln_df = pd.DataFrame(vuln_results)
    print(vuln_df.to_string(index=False))
    
    # 总计
    total_tp = sum(stats['TP'] for stats in vuln_type_stats.values())
    total_fp = sum(stats['FP'] for stats in vuln_type_stats.values())
    total_all = total_tp + total_fp
    print("\n" + "-" * 80)
    print(f"总计: TP={total_tp}, FP={total_fp}, 总数={total_all}")
    print(f"总体TP占比: {total_tp/total_all*100:.2f}%, 总体FP占比: {total_fp/total_all*100:.2f}%")
    
    print("\n" + "=" * 80)
    print("软件项目 TP/FP 统计")
    print("=" * 80)
    
    # 按软件统计
    repo_stats = defaultdict(lambda: {'TP': 0, 'FP': 0})
    for _, row in df.iterrows():
        repo = row['repo']
        judgement = row['judgement']
        repo_stats[repo][judgement] += 1
    
    # 打印软件统计结果
    repo_results = []
    for repo in sorted(repo_stats.keys()):
        tp = repo_stats[repo]['TP']
        fp = repo_stats[repo]['FP']
        total = tp + fp
        tp_ratio = tp / total * 100 if total > 0 else 0
        fp_ratio = fp / total * 100 if total > 0 else 0
        repo_results.append({
            '软件项目': repo,
            'TP': tp,
            'FP': fp,
            '总数': total,
            'TP占比(%)': f'{tp_ratio:.2f}',
            'FP占比(%)': f'{fp_ratio:.2f}'
        })
    
    repo_df = pd.DataFrame(repo_results)
    print(repo_df.to_string(index=False))
    
    # 总计
    print("\n" + "-" * 80)
    print(f"总计: TP={total_tp}, FP={total_fp}, 总数={total_all}")
    print(f"总体TP占比: {total_tp/total_all*100:.2f}%, 总体FP占比: {total_fp/total_all*100:.2f}%")
    print("=" * 80)

if __name__ == "__main__":
    analyze_validation_report("validation_report_0day.csv")
