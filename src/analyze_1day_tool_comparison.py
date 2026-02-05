#!/usr/bin/env python3
"""
统计inputs/filtered_1day_vul_list.csv中的漏洞在各工具的检测效果
生成一个CSV，表头为：vul_id,repo,fixed_commit_sha,tag,version,vuddy,movery,v1scan,fire,vulture,verdict
"""

import csv
import json
import os
from collections import defaultdict

def load_vul_list(filepath):
    """加载漏洞列表"""
    vul_list = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            vul_list.append({
                'repo': row['repo'],
                'vul_id': row['vul_id'],
                'fixed_commit_sha': row['fixed_commit_sha']
            })
    return vul_list

def load_vul_dict(filepath):
    """加载漏洞版本字典"""
    with open(filepath, 'r') as f:
        return json.load(f)

def load_vuddy_results(filepath):
    """加载vuddy结果 - 格式: project,cve,type,detected_version,vulnerability_paths"""
    results = defaultdict(set)  # (cve, type, version) -> set
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cve = row['cve']
            tag = row['type']
            version = row['detected_version']
            results[(cve, tag)].add(version)
    return results

def load_movery_results(filepath):
    """加载movery结果 - 格式: repo,cve,type,version"""
    results = defaultdict(set)
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cve = row['cve']
            tag = row['type']
            version = row['version']
            results[(cve, tag)].add(version)
    return results

def load_v1scan_results(filepath):
    """加载v1scan结果 - 格式: repo,cve,type,version"""
    results = defaultdict(set)
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cve = row['cve']
            tag = row['type']
            version = row['version']
            results[(cve, tag)].add(version)
    return results

def load_fire_results(filepath):
    """加载fire结果 - 格式: repo,cve,type,version"""
    results = defaultdict(set)
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cve = row['cve']
            tag = row['type']
            version = row['version']
            results[(cve, tag)].add(version)
    return results

def load_vulture_results(filepath):
    """加载vulture结果 - 格式: cve,type,version_path
    version从path中提取，例如: /home/.../CVE-xxx/vul/v5.6 -> v5.6
    """
    results = defaultdict(set)
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cve = row['cve']
            tag = row['type']
            version_path = row['version_path']
            # 从路径中提取版本号
            version = os.path.basename(version_path)
            results[(cve, tag)].add(version)
    return results

def load_verdict_results(filepath):
    """加载verdict结果 - 格式: repo,cve,type,version"""
    results = defaultdict(set)
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cve = row['cve']
            tag = row['type']
            version = row['version']
            results[(cve, tag)].add(version)
    return results

def check_detection(tool_results, cve, tag, version):
    """
    检查是否在工具结果中检测到
    返回: TP, FP, FN, TN
    """
    # 检查是否在任何tag下被检测到
    detected = False
    detected_tag = None
    
    for result_tag in ['vul', 'pre', 'fix']:
        if version in tool_results.get((cve, result_tag), set()):
            detected = True
            detected_tag = result_tag
            break
    
    if tag == 'vul':
        if detected:
            return 'TP'
        else:
            return 'FN'
    else:  # tag is 'pre' or 'fix'
        if detected:
            return 'FP'
        else:
            return 'TN'

def main():
    # 文件路径
    vul_list_path = 'inputs/filtered_1day_vul_list.csv'
    vul_dict_path = 'inputs/1day_vul_dict.json'
    
    vuddy_path = 'results/1day/vuddy_1day_res.csv'
    movery_path = 'results/1day/movery_1day_res.csv'
    v1scan_path = 'results/1day/v1scan_1day_res.csv'
    fire_path = 'results/1day/fire_1day_res.csv'
    vulture_path = 'results/1day/vulture_1day_res.csv'
    verdict_path = 'results/1day/verdict_1day_res.csv'
    
    output_path = 'results/1day/tool_comparison.csv'
    
    # 加载数据
    print("Loading vulnerability list...")
    vul_list = load_vul_list(vul_list_path)
    print(f"Loaded {len(vul_list)} vulnerabilities")
    
    print("Loading vulnerability dictionary...")
    vul_dict = load_vul_dict(vul_dict_path)
    print(f"Loaded {len(vul_dict)} vulnerability entries")
    
    # 加载各工具结果
    print("Loading tool results...")
    vuddy_results = load_vuddy_results(vuddy_path)
    movery_results = load_movery_results(movery_path)
    v1scan_results = load_v1scan_results(v1scan_path)
    fire_results = load_fire_results(fire_path)
    vulture_results = load_vulture_results(vulture_path)
    verdict_results = load_verdict_results(verdict_path)
    
    print(f"Vuddy: {sum(len(v) for v in vuddy_results.values())} entries")
    print(f"Movery: {sum(len(v) for v in movery_results.values())} entries")
    print(f"V1Scan: {sum(len(v) for v in v1scan_results.values())} entries")
    print(f"Fire: {sum(len(v) for v in fire_results.values())} entries")
    print(f"Vulture: {sum(len(v) for v in vulture_results.values())} entries")
    print(f"Verdict: {sum(len(v) for v in verdict_results.values())} entries")
    
    # 生成结果
    results = []
    missing_vul_ids = []
    
    for vul in vul_list:
        vul_id = vul['vul_id']
        repo = vul['repo']
        fixed_commit_sha = vul['fixed_commit_sha']
        
        if vul_id not in vul_dict:
            missing_vul_ids.append(vul_id)
            continue
        
        vul_info = vul_dict[vul_id]
        versions_info = vul_info.get('versions', {})
        
        # 收集所有版本和对应的tag
        samples = []
        
        # pre版本
        pre_versions = versions_info.get('pre', {})
        for version in pre_versions.keys():
            samples.append(('pre', version))
        
        # vul版本
        vul_versions = versions_info.get('vul', {})
        for version in vul_versions.keys():
            samples.append(('vul', version))
        
        # fix版本
        fix_versions = versions_info.get('fix', {})
        for version in fix_versions.keys():
            samples.append(('fix', version))
        
        # 为每个样本检测各工具的结果
        for tag, version in samples:
            row = {
                'vul_id': vul_id,
                'repo': repo,
                'fixed_commit_sha': fixed_commit_sha,
                'tag': tag,
                'version': version,
                'vuddy': check_detection(vuddy_results, vul_id, tag, version),
                'movery': check_detection(movery_results, vul_id, tag, version),
                'v1scan': check_detection(v1scan_results, vul_id, tag, version),
                'fire': check_detection(fire_results, vul_id, tag, version),
                'vulture': check_detection(vulture_results, vul_id, tag, version),
                'verdict': check_detection(verdict_results, vul_id, tag, version),
            }
            results.append(row)
    
    if missing_vul_ids:
        print(f"Warning: {len(missing_vul_ids)} vulnerabilities not found in vul_dict")
        print(f"First 10 missing: {missing_vul_ids[:10]}")
    
    # 写入CSV
    print(f"\nWriting {len(results)} samples to {output_path}...")
    with open(output_path, 'w', newline='') as f:
        fieldnames = ['vul_id', 'repo', 'fixed_commit_sha', 'tag', 'version', 
                      'vuddy', 'movery', 'v1scan', 'fire', 'vulture', 'verdict']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Done! Total samples: {len(results)}")
    
    # 统计各工具的效果
    print("\n=== Tool Performance Summary ===")
    tools = ['vuddy', 'movery', 'v1scan', 'fire', 'vulture', 'verdict']
    
    for tool in tools:
        tp = sum(1 for r in results if r[tool] == 'TP')
        fp = sum(1 for r in results if r[tool] == 'FP')
        fn = sum(1 for r in results if r[tool] == 'FN')
        tn = sum(1 for r in results if r[tool] == 'TN')
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n{tool}:")
        print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
        print(f"  Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

if __name__ == '__main__':
    main()
