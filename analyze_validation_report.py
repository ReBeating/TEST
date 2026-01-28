#!/usr/bin/env python3
"""
统计 validation_report_0day.csv 中不同漏洞类型的 TP/FP 个数和比例
"""

import pandas as pd
from collections import defaultdict

def analyze_validation_report(csv_file):
    """分析验证报告中的 TP/FP 统计"""
    
    # 读取 CSV 文件
    df = pd.read_csv(csv_file)
    
    print(f"总记录数: {len(df)}")
    print(f"\n列名: {df.columns.tolist()}")
    
    # 统计不同漏洞类型的 TP/FP
    stats = defaultdict(lambda: {'TP': 0, 'FP': 0, 'Unknown': 0, 'Total': 0})
    
    for _, row in df.iterrows():
        vuln_type = row['vuln_type']
        judgement = str(row['judgement']).strip() if pd.notna(row['judgement']) else ''
        
        stats[vuln_type]['Total'] += 1
        
        if judgement == 'TP':
            stats[vuln_type]['TP'] += 1
        elif judgement == 'FP':
            stats[vuln_type]['FP'] += 1
        else:
            stats[vuln_type]['Unknown'] += 1
    
    # 按漏洞类型排序
    sorted_types = sorted(stats.keys())
    
    print("\n" + "="*100)
    print(f"{'漏洞类型':<30} {'TP':>8} {'FP':>8} {'未知':>8} {'总数':>8} {'TP率':>10} {'FP率':>10}")
    print("="*100)
    
    total_tp = 0
    total_fp = 0
    total_unknown = 0
    total_all = 0
    
    for vuln_type in sorted_types:
        data = stats[vuln_type]
        tp = data['TP']
        fp = data['FP']
        unknown = data['Unknown']
        total = data['Total']
        
        # 计算比例（基于已判断的样本）
        judged = tp + fp
        tp_rate = (tp / judged * 100) if judged > 0 else 0
        fp_rate = (fp / judged * 100) if judged > 0 else 0
        
        print(f"{vuln_type:<30} {tp:>8} {fp:>8} {unknown:>8} {total:>8} {tp_rate:>9.2f}% {fp_rate:>9.2f}%")
        
        total_tp += tp
        total_fp += fp
        total_unknown += unknown
        total_all += total
    
    print("="*100)
    
    # 总计
    total_judged = total_tp + total_fp
    overall_tp_rate = (total_tp / total_judged * 100) if total_judged > 0 else 0
    overall_fp_rate = (total_fp / total_judged * 100) if total_judged > 0 else 0
    
    print(f"{'总计':<30} {total_tp:>8} {total_fp:>8} {total_unknown:>8} {total_all:>8} {overall_tp_rate:>9.2f}% {overall_fp_rate:>9.2f}%")
    print("="*100)
    
    print(f"\n总体统计:")
    print(f"  - 总记录数: {total_all}")
    print(f"  - 已判断数: {total_judged} ({total_judged/total_all*100:.2f}%)")
    print(f"  - True Positive (TP): {total_tp} ({overall_tp_rate:.2f}%)")
    print(f"  - False Positive (FP): {total_fp} ({overall_fp_rate:.2f}%)")
    print(f"  - 未判断: {total_unknown} ({total_unknown/total_all*100:.2f}%)")
    
    # 按 TP 数量排序显示 Top 10
    print("\n" + "="*100)
    print("Top 10 漏洞类型 (按 TP 数量排序):")
    print("="*100)
    
    sorted_by_tp = sorted(stats.items(), key=lambda x: x[1]['TP'], reverse=True)[:10]
    for vuln_type, data in sorted_by_tp:
        judged = data['TP'] + data['FP']
        tp_rate = (data['TP'] / judged * 100) if judged > 0 else 0
        print(f"  {vuln_type:<30} TP: {data['TP']:>4}, FP: {data['FP']:>4}, TP率: {tp_rate:>6.2f}%")
    
    # 按 FP 数量排序显示 Top 10
    print("\n" + "="*100)
    print("Top 10 漏洞类型 (按 FP 数量排序):")
    print("="*100)
    
    sorted_by_fp = sorted(stats.items(), key=lambda x: x[1]['FP'], reverse=True)[:10]
    for vuln_type, data in sorted_by_fp:
        judged = data['TP'] + data['FP']
        fp_rate = (data['FP'] / judged * 100) if judged > 0 else 0
        print(f"  {vuln_type:<30} FP: {data['FP']:>4}, TP: {data['TP']:>4}, FP率: {fp_rate:>6.2f}%")
    
    return stats

if __name__ == '__main__':
    csv_file = 'validation_report_0day.csv'
    stats = analyze_validation_report(csv_file)
