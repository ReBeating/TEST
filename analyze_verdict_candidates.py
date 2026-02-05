#!/usr/bin/env python3
"""
统计 verdict_0day_res.csv 中的行在 *_repo_candidates.json 中的出现情况
计算 precision 并生成带 confidence 列的新 CSV
"""

import csv
import json
import os
from pathlib import Path
from typing import Optional, Tuple


def find_json_file(repo: str, vul_id: str) -> Optional[Path]:
    """
    查找对应的 JSON 文件
    优先检查 outputs/results，如果不存在则检查 outputs_0204/results
    """
    json_filename = f"{vul_id}_repo_candidates.json"
    
    # 尝试 outputs/results
    json_path = Path("outputs/results") / repo / json_filename
    if json_path.exists():
        return json_path
    
    # 尝试 outputs_0204/results
    json_path = Path("outputs_0204/results") / repo / json_filename
    if json_path.exists():
        return json_path
    
    return None


def find_confidence_in_json(json_path: Path, target_file: str, target_func: str) -> Optional[float]:
    """
    在 JSON 文件中查找匹配的 target_file 和 target_func，返回对应的 confidence
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            candidates = json.load(f)
        
        # 遍历所有候选项
        for candidate in candidates:
            # 检查 target_file 和 target_func 是否匹配
            if candidate.get('target_file') == target_file and candidate.get('target_func') == target_func:
                return candidate.get('confidence')
        
        return None
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return None


def analyze_verdict_candidates():
    """
    主函数：分析 verdict_0day_res.csv 与 repo_candidates.json 的匹配情况
    """
    input_csv = Path("results/0day/verdict_0day_res.csv")
    output_csv = Path("verdict_0day_with_confidence.csv")
    
    tp_count = 0  # 在 JSON 中找到的（True Positive from detection perspective）
    fp_count = 0  # 在 JSON 中未找到的（False Positive from detection perspective）
    no_json_count = 0  # JSON 文件不存在的
    
    results = []
    
    # 读取 CSV 文件
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            vul_id = row['vul_id']
            repo = row['repo']
            target_file = row['target_file']
            target_func = row['target_func']
            judgement = row['judgement']
            
            # 查找对应的 JSON 文件
            json_path = find_json_file(repo, vul_id)
            
            if json_path is None:
                # JSON 文件不存在
                confidence = None
                found_in_json = False
                no_json_count += 1
                print(f"Warning: JSON file not found for {vul_id} ({repo})")
            else:
                # 在 JSON 中查找
                confidence = find_confidence_in_json(json_path, target_file, target_func)
                found_in_json = (confidence is not None)
                
                if found_in_json:
                    tp_count += 1
                else:
                    fp_count += 1
            
            # 添加到结果中
            result_row = row.copy()
            result_row['confidence'] = confidence if confidence is not None else ''
            result_row['found_in_json'] = 'YES' if found_in_json else 'NO'
            results.append(result_row)
            
            # 打印一些示例
            if len(results) <= 5 or not found_in_json:
                status = "FOUND" if found_in_json else "NOT FOUND"
                conf_str = f"{confidence:.4f}" if confidence is not None else "N/A"
                print(f"{vul_id} - {target_file}:{target_func} - {status} (conf: {conf_str}, judgement: {judgement})")
    
    # 计算 precision
    # TP: 在 JSON 中找到且 judgement=TP 的数量（真正例）
    # FP: 在 JSON 中找到且 judgement=FP 的数量（假正例）
    tp_correct = sum(1 for r in results if r['found_in_json'] == 'YES' and r['judgement'] == 'TP')
    fp_incorrect = sum(1 for r in results if r['found_in_json'] == 'YES' and r['judgement'] == 'FP')
    
    total_detected = tp_correct + fp_incorrect
    if total_detected > 0:
        precision = tp_correct / total_detected
    else:
        precision = 0.0
    
    # 写入输出 CSV
    if results:
        fieldnames = list(results[0].keys())
        with open(output_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    
    # 打印统计结果
    print("\n" + "="*60)
    print("统计结果 (Statistics)")
    print("="*60)
    print(f"在 JSON 中找到的总数 (Total found in JSON): {tp_count}")
    print(f"在 JSON 中未找到的总数 (Total not found in JSON): {fp_count}")
    print(f"JSON 文件不存在 (No JSON file): {no_json_count}")
    print(f"总计 (Total): {len(results)}")
    print("\n--- 根据 Ground Truth Judgement 分类 ---")
    print(f"检测到的真漏洞 (TP - 在JSON中找到且judgement=TP): {tp_correct}")
    print(f"检测到的假漏洞 (FP - 在JSON中找到且judgement=FP): {fp_incorrect}")
    print(f"\nPrecision (检测精度): {precision:.4f} ({precision*100:.2f}%)")
    print(f"  = TP / (TP + FP) = {tp_correct} / ({tp_correct} + {fp_incorrect}) = {tp_correct}/{total_detected}")
    print("="*60)
    print(f"\n输出文件已保存到: {output_csv}")
    
    # 分析详细的分布
    print("\n" + "="*60)
    print("详细分类统计")
    print("="*60)
    
    found_tp = sum(1 for r in results if r['found_in_json'] == 'YES' and r['judgement'] == 'TP')
    found_fp = sum(1 for r in results if r['found_in_json'] == 'YES' and r['judgement'] == 'FP')
    notfound_tp = sum(1 for r in results if r['found_in_json'] == 'NO' and r['judgement'] == 'TP')
    notfound_fp = sum(1 for r in results if r['found_in_json'] == 'NO' and r['judgement'] == 'FP')
    
    print(f"检测到 & Ground Truth=TP (真正例 TP): {found_tp}")
    print(f"检测到 & Ground Truth=FP (假正例 FP): {found_fp}")
    print(f"未检测到 & Ground Truth=TP (假反例 FN): {notfound_tp}")
    print(f"未检测到 & Ground Truth=FP (真反例 TN): {notfound_fp}")
    print("\n如果考虑未检测到的情况:")
    print(f"  Recall = TP / (TP + FN) = {found_tp} / ({found_tp} + {notfound_tp}) = {found_tp/(found_tp + notfound_tp) if (found_tp + notfound_tp) > 0 else 0:.4f}")
    print("="*60)


if __name__ == "__main__":
    analyze_verdict_candidates()
