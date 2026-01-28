#!/usr/bin/env python3
"""
统计confidence分布：
1. checked_list.csv中TP和FP的行在*_repo_candidates.json中的confidence分布
2. *_repo_findings.json中is_vulnerable为false的target的confidence分布
按照0.05一个区间统计
"""

import csv
import json
import os
from pathlib import Path
from collections import defaultdict
import math

def get_confidence_bin(confidence):
    """将confidence值映射到0.05区间"""
    if confidence is None:
        return None
    # 向下取整到0.05的倍数
    return math.floor(confidence / 0.05) * 0.05

def find_candidate_file(repo, vul_id):
    """查找对应的candidates文件"""
    # 构建可能的路径
    base_path = Path("outputs/results")
    candidate_file = base_path / repo / f"{vul_id}_repo_candidates.json"
    return candidate_file if candidate_file.exists() else None

def find_findings_files():
    """查找所有findings文件"""
    base_path = Path("outputs/results")
    findings_files = []
    if base_path.exists():
        for repo_dir in base_path.iterdir():
            if repo_dir.is_dir():
                for file in repo_dir.glob("*_repo_findings.json"):
                    findings_files.append(file)
    return findings_files

def analyze_tp_fp_confidence():
    """分析TP和FP的confidence分布"""
    print("=" * 80)
    print("分析 checked_list.csv 中 TP 和 FP 的 confidence 分布")
    print("=" * 80)
    
    # 读取checked_list.csv
    tp_confidences = []
    fp_confidences = []
    not_found = []
    
    with open("results/checked_list.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vul_id = row["vul_id"]
            repo = row["repo"]
            target_file = row["target_file_path"]
            target_func = row["target_func_name"]
            judgement = row["judgement"]
            
            if judgement not in ["TP", "FP"]:
                continue
            
            # 查找对应的candidates文件
            candidate_file = find_candidate_file(repo, vul_id)
            if not candidate_file:
                not_found.append((vul_id, repo, target_file, target_func))
                continue
            
            # 读取candidates文件
            try:
                with open(candidate_file, "r", encoding="utf-8") as cf:
                    candidates_data = json.load(cf)
                    
                # 确保candidates_data是列表
                if not isinstance(candidates_data, list):
                    candidates_data = []
                    
                # 查找匹配的candidate
                found = False
                for candidate in candidates_data:
                    if (candidate.get("target_file") == target_file and
                        candidate.get("target_func") == target_func):
                        confidence = candidate.get("confidence")
                        if confidence is not None:
                            if judgement == "TP":
                                tp_confidences.append(confidence)
                            else:
                                fp_confidences.append(confidence)
                            found = True
                            break
                
                if not found:
                    not_found.append((vul_id, repo, target_file, target_func))
                    
            except Exception as e:
                print(f"错误处理 {candidate_file}: {e}")
    
    # 统计分布
    print(f"\n找到的TP数量: {len(tp_confidences)}")
    print(f"找到的FP数量: {len(fp_confidences)}")
    print(f"未找到confidence的条目数量: {len(not_found)}")
    
    # 按0.05区间统计TP
    print("\n" + "=" * 80)
    print("TP confidence 分布 (按0.05区间):")
    print("=" * 80)
    tp_bins = defaultdict(int)
    for conf in tp_confidences:
        bin_val = get_confidence_bin(conf)
        if bin_val is not None:
            tp_bins[bin_val] += 1
    
    for bin_val in sorted(tp_bins.keys()):
        count = tp_bins[bin_val]
        percentage = (count / len(tp_confidences) * 100) if tp_confidences else 0
        bar = "█" * int(percentage / 2)
        print(f"[{bin_val:.2f}, {bin_val+0.05:.2f}): {count:4d} ({percentage:5.2f}%) {bar}")
    
    # 按0.05区间统计FP
    print("\n" + "=" * 80)
    print("FP confidence 分布 (按0.05区间):")
    print("=" * 80)
    fp_bins = defaultdict(int)
    for conf in fp_confidences:
        bin_val = get_confidence_bin(conf)
        if bin_val is not None:
            fp_bins[bin_val] += 1
    
    for bin_val in sorted(fp_bins.keys()):
        count = fp_bins[bin_val]
        percentage = (count / len(fp_confidences) * 100) if fp_confidences else 0
        bar = "█" * int(percentage / 2)
        print(f"[{bin_val:.2f}, {bin_val+0.05:.2f}): {count:4d} ({percentage:5.2f}%) {bar}")
    
    return tp_confidences, fp_confidences

def analyze_non_vulnerable_confidence():
    """分析findings中is_vulnerable为false的目标在candidates中的confidence分布"""
    print("\n" + "=" * 80)
    print("分析 *_repo_findings.json 中 is_vulnerable=false 的目标在 candidates 中的 confidence 分布")
    print("=" * 80)
    
    non_vuln_confidences = []
    not_found = []
    
    # 查找所有findings文件 (递归查找)
    base_path = Path("outputs/results")
    findings_files = []
    if base_path.exists():
        findings_files = list(base_path.glob("**/*_repo_findings.json"))
    
    print(f"\n找到 {len(findings_files)} 个 findings 文件")
    
    for findings_file in findings_files:
        try:
            with open(findings_file, "r", encoding="utf-8") as f:
                findings_data = json.load(f)
                
            # 确保是列表
            if not isinstance(findings_data, list):
                continue
                
            for finding in findings_data:
                # 只处理is_vulnerable=false的记录
                if finding.get("is_vulnerable") != False:
                    continue
                    
                vul_id = finding.get("vul_id")
                repo = finding.get("repo_path", "").split("/")[-1] if "/" in finding.get("repo_path", "") else ""
                target_file = finding.get("target_file")
                target_func = finding.get("target_func")
                
                # 根据vul_id找到对应的candidates文件
                candidate_file = findings_file.parent / f"{vul_id}_repo_candidates.json"
                
                if not candidate_file.exists():
                    not_found.append((vul_id, target_file, target_func))
                    continue
                
                # 在candidates文件中查找对应的记录
                try:
                    with open(candidate_file, "r", encoding="utf-8") as cf:
                        candidates_data = json.load(cf)
                        
                    if not isinstance(candidates_data, list):
                        continue
                        
                    found = False
                    for candidate in candidates_data:
                        if (candidate.get("target_file") == target_file and
                            candidate.get("target_func") == target_func):
                            confidence = candidate.get("confidence")
                            if confidence is not None:
                                non_vuln_confidences.append(confidence)
                                found = True
                                break
                    
                    if not found:
                        not_found.append((vul_id, target_file, target_func))
                        
                except Exception as e:
                    print(f"错误读取 {candidate_file}: {e}")
                        
        except Exception as e:
            print(f"错误处理 {findings_file}: {e}")
    
    print(f"\n找到的 is_vulnerable=false 的条目数量: {len(non_vuln_confidences)}")
    print(f"未找到对应confidence的条目数量: {len(not_found)}")
    
    # 按0.05区间统计
    print("\n" + "=" * 80)
    print("is_vulnerable=false confidence 分布 (按0.05区间):")
    print("=" * 80)
    nv_bins = defaultdict(int)
    for conf in non_vuln_confidences:
        bin_val = get_confidence_bin(conf)
        if bin_val is not None:
            nv_bins[bin_val] += 1
    
    for bin_val in sorted(nv_bins.keys()):
        count = nv_bins[bin_val]
        percentage = (count / len(non_vuln_confidences) * 100) if non_vuln_confidences else 0
        bar = "█" * int(percentage / 2)
        print(f"[{bin_val:.2f}, {bin_val+0.05:.2f}): {count:4d} ({percentage:5.2f}%) {bar}")
    
    return non_vuln_confidences

def main():
    print("开始分析 confidence 分布...\n")
    
    # 分析TP和FP
    tp_confidences, fp_confidences = analyze_tp_fp_confidence()
    
    # 分析非漏洞的confidence
    nv_confidences = analyze_non_vulnerable_confidence()
    
    # 统计摘要
    print("\n" + "=" * 80)
    print("统计摘要")
    print("=" * 80)
    if tp_confidences:
        print(f"TP confidence 范围: [{min(tp_confidences):.4f}, {max(tp_confidences):.4f}]")
        print(f"TP confidence 平均值: {sum(tp_confidences)/len(tp_confidences):.4f}")
    if fp_confidences:
        print(f"FP confidence 范围: [{min(fp_confidences):.4f}, {max(fp_confidences):.4f}]")
        print(f"FP confidence 平均值: {sum(fp_confidences)/len(fp_confidences):.4f}")
    if nv_confidences:
        print(f"Non-vulnerable confidence 范围: [{min(nv_confidences):.4f}, {max(nv_confidences):.4f}]")
        print(f"Non-vulnerable confidence 平均值: {sum(nv_confidences)/len(nv_confidences):.4f}")

if __name__ == "__main__":
    main()
