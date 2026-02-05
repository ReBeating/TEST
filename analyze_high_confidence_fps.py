#!/usr/bin/env python3
"""
分析高置信度假阳性案例
用于诊断为什么提高阈值无法提升precision
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_fp_distribution(result_csv, ground_truth_csv, confidence_col='confidence', label_col='is_vulnerable'):
    """
    分析假阳性在不同置信度区间的分布
    
    Args:
        result_csv: 检测结果CSV文件路径
        ground_truth_csv: ground truth标签文件路径  
        confidence_col: 置信度列名
        label_col: 真实标签列名
    """
    # 读取数据
    results = pd.read_csv(result_csv)
    
    # 如果有单独的ground truth文件，合并
    if ground_truth_csv and Path(ground_truth_csv).exists():
        gt = pd.read_csv(ground_truth_csv)
        # 假设有共同的ID列用于合并
        # results = results.merge(gt, on='id', how='left')
    
    print("="*80)
    print("高置信度假阳性分析报告")
    print("="*80)
    
    # 定义置信度区间
    thresholds = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    print("\n1. 不同置信度区间的Precision分析")
    print("-"*80)
    print(f"{'阈值范围':<15} {'总预测数':>10} {'TP':>10} {'FP':>10} {'Precision':>10}")
    print("-"*80)
    
    for i in range(len(thresholds)-1):
        low, high = thresholds[i], thresholds[i+1]
        mask = (results[confidence_col] >= low) & (results[confidence_col] < high)
        subset = results[mask]
        
        if len(subset) > 0:
            # 根据你的标注方式调整
            tp = len(subset[subset[label_col] == True])  # 或 == 1
            fp = len(subset[subset[label_col] == False])  # 或 == 0
            total = len(subset)
            precision = tp / total if total > 0 else 0
            
            print(f"[{low:.1f}, {high:.1f})    {total:>10} {tp:>10} {fp:>10} {precision:>10.3f}")
    
    # 高置信度假阳性详细分析
    print("\n2. 高置信度假阳性案例（threshold >= 0.8）")
    print("-"*80)
    
    high_conf_fps = results[(results[confidence_col] >= 0.8) & (results[label_col] == False)]
    
    if len(high_conf_fps) > 0:
        print(f"总共有 {len(high_conf_fps)} 个高置信度假阳性")
        print(f"\n置信度分布统计：")
        print(f"  - 平均值: {high_conf_fps[confidence_col].mean():.3f}")
        print(f"  - 中位数: {high_conf_fps[confidence_col].median():.3f}")
        print(f"  - 最大值: {high_conf_fps[confidence_col].max():.3f}")
        
        # 如果有其他特征列，分析共同特征
        print(f"\n高置信度FP的前10个案例：")
        print(high_conf_fps.head(10))
        
        # 保存详细结果供人工审查
        output_path = "high_confidence_false_positives.csv"
        high_conf_fps.to_csv(output_path, index=False)
        print(f"\n已保存详细FP列表到: {output_path}")
    else:
        print("没有找到高置信度假阳性（这是好现象！）")
    
    # 置信度校准分析
    print("\n3. 置信度校准分析")
    print("-"*80)
    
    bins = np.arange(0.4, 1.05, 0.1)
    results['conf_bin'] = pd.cut(results[confidence_col], bins=bins)
    
    calibration = results.groupby('conf_bin').agg({
        label_col: ['count', 'mean']
    }).round(3)
    
    calibration.columns = ['样本数', '实际准确率']
    print(calibration)
    
    print("\n注：实际准确率应该接近该区间的置信度值")
    print("   如果实际准确率明显低于置信度，说明过度自信（over-confident）")
    
    return high_conf_fps

def compare_tp_fp_features(result_csv, confidence_threshold=0.8):
    """
    比较高置信度TP和FP的特征差异
    """
    results = pd.read_csv(result_csv)
    high_conf = results[results['confidence'] >= confidence_threshold]
    
    tp_subset = high_conf[high_conf['is_vulnerable'] == True]
    fp_subset = high_conf[high_conf['is_vulnerable'] == False]
    
    print("\n4. 高置信度TP vs FP特征对比")
    print("-"*80)
    print(f"TP样本数: {len(tp_subset)}")
    print(f"FP样本数: {len(fp_subset)}")
    
    # 这里需要根据你的实际特征列进行分析
    # 例如：漏洞类型、代码相似度、匹配方法等
    
    numeric_cols = results.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['confidence', 'is_vulnerable']]
    
    if len(numeric_cols) > 0:
        print(f"\n数值特征对比：")
        for col in numeric_cols[:5]:  # 只显示前5个
            tp_mean = tp_subset[col].mean() if len(tp_subset) > 0 else 0
            fp_mean = fp_subset[col].mean() if len(fp_subset) > 0 else 0
            print(f"  {col}: TP平均={tp_mean:.3f}, FP平均={fp_mean:.3f}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python analyze_high_confidence_fps.py <result_csv> [ground_truth_csv]")
        print("\n示例:")
        print("  python analyze_high_confidence_fps.py results/0day/verdict_0day_res.csv")
        sys.exit(1)
    
    result_csv = sys.argv[1]
    ground_truth_csv = sys.argv[2] if len(sys.argv) > 2 else None
    
    analyze_fp_distribution(result_csv, ground_truth_csv)
    compare_tp_fp_features(result_csv)
