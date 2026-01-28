#!/usr/bin/env python3
"""
检查 merged_dataset_with_category1.csv 中的漏洞，
哪些对应的 outputs/results/*/*/*_repo_findings.json 文件有 confidence 字段
"""

import csv
import json
from pathlib import Path


def find_repo_findings_file(repo, vul_id):
    """查找漏洞对应的 repo_findings.json 文件"""
    # 构建可能的路径模式
    base_path = Path("outputs/results")
    
    # 尝试不同的路径模式
    patterns = [
        # 直接在 repo 目录下
        base_path / repo / f"{vul_id}_repo_findings.json",
        # repo 名称中的 / 替换为其他字符
        base_path / repo.replace('/', '_') / f"{vul_id}_repo_findings.json",
        # 带子目录的旧格式
        base_path / repo / vul_id / f"{vul_id}_repo_findings.json",
        base_path / repo.replace('/', '_') / vul_id / f"{vul_id}_repo_findings.json",
    ]
    
    for pattern in patterns:
        if pattern.exists():
            return pattern
    
    return None


def check_confidence_in_json(json_file):
    """检查 JSON 文件中是否有 confidence 字段"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 检查不同可能的位置
        has_confidence = False
        confidence_locations = []
        
        # 检查顶层（如果是字典）
        if isinstance(data, dict) and 'confidence' in data:
            has_confidence = True
            confidence_locations.append('root')
        
        # 检查顶层列表
        if isinstance(data, list):
            for idx, item in enumerate(data):
                if isinstance(item, dict) and 'confidence' in item:
                    has_confidence = True
                    confidence_locations.append(f'[{idx}]')
                    # 只记录前几个，避免太多
                    if len([loc for loc in confidence_locations if loc.startswith('[')]) >= 3:
                        if idx < len(data) - 1:
                            confidence_locations.append(f'... and {len(data) - idx - 1} more')
                        break
        
        # 检查结果列表（如果是字典结构）
        if isinstance(data, dict):
            if 'results' in data and isinstance(data['results'], list):
                for idx, result in enumerate(data['results']):
                    if isinstance(result, dict) and 'confidence' in result:
                        has_confidence = True
                        confidence_locations.append(f'results[{idx}]')
                        if idx >= 2:
                            break
            
            # 检查其他可能的键
            for key, value in data.items():
                if key == 'results':
                    continue  # 已经检查过
                if isinstance(value, dict) and 'confidence' in value:
                    has_confidence = True
                    confidence_locations.append(f'{key}')
                elif isinstance(value, list):
                    for idx, item in enumerate(value):
                        if isinstance(item, dict) and 'confidence' in item:
                            has_confidence = True
                            confidence_locations.append(f'{key}[{idx}]')
                            if idx >= 2:
                                break
        
        return has_confidence, confidence_locations
    
    except Exception as e:
        return False, [f"Error: {str(e)}"]


def main():
    input_file = Path("merged_dataset_with_category1.csv")
    
    print(f"读取 {input_file}...")
    
    vulnerabilities = []
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            vulnerabilities.append({
                'repo': row['repo'].strip(),
                'vul_id': row['vul_id'].strip(),
                'fixed_commit_sha': row['fixed_commit_sha'].strip()
            })
    
    print(f"总共 {len(vulnerabilities)} 个漏洞\n")
    
    # 统计
    found_files = 0
    not_found_files = 0
    has_confidence = 0
    no_confidence = 0
    
    confidence_vulns = []
    no_confidence_vulns = []
    not_found_vulns = []
    
    print("检查结果文件...")
    for vuln in vulnerabilities:
        repo = vuln['repo']
        vul_id = vuln['vul_id']
        
        # 查找文件
        findings_file = find_repo_findings_file(repo, vul_id)
        
        if findings_file:
            found_files += 1
            # 检查 confidence 字段
            has_conf, conf_locations = check_confidence_in_json(findings_file)
            
            if has_conf:
                has_confidence += 1
                confidence_vulns.append({
                    'repo': repo,
                    'vul_id': vul_id,
                    'file': str(findings_file),
                    'confidence_locations': conf_locations
                })
            else:
                no_confidence += 1
                no_confidence_vulns.append({
                    'repo': repo,
                    'vul_id': vul_id,
                    'file': str(findings_file)
                })
        else:
            not_found_files += 1
            not_found_vulns.append({
                'repo': repo,
                'vul_id': vul_id
            })
    
    # 打印统计结果
    print(f"\n{'='*60}")
    print("统计结果:")
    print(f"{'='*60}")
    print(f"总漏洞数: {len(vulnerabilities)}")
    print(f"找到结果文件: {found_files}")
    print(f"未找到结果文件: {not_found_files}")
    print(f"")
    print(f"在找到的文件中:")
    print(f"  包含 confidence 字段: {has_confidence}")
    print(f"  不包含 confidence 字段: {no_confidence}")
    print(f"{'='*60}\n")
    
    # 保存包含 confidence 的漏洞列表
    if confidence_vulns:
        output_file = Path("vulnerabilities_with_confidence.csv")
        print(f"保存包含 confidence 的漏洞列表到 {output_file}...")
        
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['repo', 'vul_id', 'file_path', 'confidence_locations'])
            for vuln in confidence_vulns:
                writer.writerow([
                    vuln['repo'],
                    vuln['vul_id'],
                    vuln['file'],
                    '; '.join(vuln['confidence_locations'])
                ])
        
        print(f"已保存 {len(confidence_vulns)} 个漏洞\n")
        
        # 显示前10个示例
        print("前 10 个包含 confidence 的漏洞示例:")
        for i, vuln in enumerate(confidence_vulns[:10], 1):
            print(f"{i}. {vuln['repo']} / {vuln['vul_id']}")
            print(f"   文件: {vuln['file']}")
            print(f"   位置: {', '.join(vuln['confidence_locations'])}")
    
    # 保存不包含 confidence 的漏洞列表
    if no_confidence_vulns:
        output_file = Path("vulnerabilities_without_confidence.csv")
        print(f"\n保存不包含 confidence 的漏洞列表到 {output_file}...")
        
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['repo', 'vul_id', 'file_path'])
            for vuln in no_confidence_vulns:
                writer.writerow([
                    vuln['repo'],
                    vuln['vul_id'],
                    vuln['file']
                ])
        
        print(f"已保存 {len(no_confidence_vulns)} 个漏洞")
    
    # 保存未找到结果文件的漏洞列表
    if not_found_vulns:
        output_file = Path("vulnerabilities_no_results_file.csv")
        print(f"\n保存未找到结果文件的漏洞列表到 {output_file}...")
        
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['repo', 'vul_id'])
            for vuln in not_found_vulns:
                writer.writerow([
                    vuln['repo'],
                    vuln['vul_id']
                ])
        
        print(f"已保存 {len(not_found_vulns)} 个漏洞")


if __name__ == "__main__":
    main()
