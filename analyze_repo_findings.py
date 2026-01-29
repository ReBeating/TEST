#!/usr/bin/env python3
"""
分析outputs/results目录中的repo_findings.json文件
统计哪些漏洞的文件为空，哪些漏洞的所有target的is_vulnerable都为false
"""
import os
import json
import csv
from pathlib import Path
def check_repo_findings(repo, vul_id):
    """
    检查指定repo和vul_id的repo_findings.json文件
    返回: (status, details)
    status: 'empty', 'all_false', 'has_vulnerable', 'not_found'
    """
    # 构建文件路径
    # repo格式为 "owner/name"，需要分别作为两层目录
    repo_parts = repo.split('/')
    if len(repo_parts) != 2:
        return 'invalid_repo', f"Invalid repo format: {repo}"
    
    owner, name = repo_parts
    file_path = Path(f"outputs/results/{owner}/{name}/{vul_id}_repo_findings.json")
    
    # 检查文件是否存在
    if not file_path.exists():
        return 'not_found', "File not found"
    
    try:
        # 读取JSON文件
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 检查是否为空
        if not data or len(data) == 0:
            return 'empty', "Empty JSON array"
        
        # 检查所有target的is_vulnerable状态
        has_vulnerable = False
        for item in data:
            if item.get('is_vulnerable', False):
                has_vulnerable = True
                break
        
        if not has_vulnerable:
            return 'all_false', f"All {len(data)} targets have is_vulnerable=false"
        else:
            return 'has_vulnerable', f"Found vulnerable targets"
            
    except json.JSONDecodeError as e:
        return 'json_error', f"JSON decode error: {str(e)}"
    except Exception as e:
        return 'error', f"Error reading file: {str(e)}"
def main():
    # 读取added_vul_list.csv
    input_file = "inputs/added_vul_list.csv"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found")
        return
    
    # 统计结果
    empty_files = []
    all_false_files = []
    has_vulnerable_files = []
    not_found_files = []
    error_files = []
    
    print("正在分析repo_findings.json文件...")
    print("-" * 80)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        total_count = 0
        
        for row in reader:
            repo = row['repo']
            vul_id = row['vul_id']
            fixed_commit_sha = row['fixed_commit_sha']
            
            total_count += 1
            
            status, details = check_repo_findings(repo, vul_id)
            
            entry = {
                'repo': repo,
                'vul_id': vul_id,
                'fixed_commit_sha': fixed_commit_sha,
                'details': details
            }
            
            if status == 'empty':
                empty_files.append(entry)
            elif status == 'all_false':
                all_false_files.append(entry)
            elif status == 'has_vulnerable':
                has_vulnerable_files.append(entry)
            elif status == 'not_found':
                not_found_files.append(entry)
            else:
                error_files.append(entry)
    
    # 打印统计结果
    print(f"\n总计分析: {total_count} 条记录")
    print(f"文件未找到: {len(not_found_files)} 条")
    print(f"文件为空: {len(empty_files)} 条")
    print(f"所有target的is_vulnerable都为false: {len(all_false_files)} 条")
    print(f"存在vulnerable的target: {len(has_vulnerable_files)} 条")
    print(f"读取错误: {len(error_files)} 条")
    print("-" * 80)
    
    # 保存详细结果到CSV
    output_file = "repo_findings_analysis.csv"
    
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['repo', 'vul_id', 'fixed_commit_sha', 'status', 'details']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # 写入文件未找到的记录
        for entry in not_found_files:
            writer.writerow({
                'repo': entry['repo'],
                'vul_id': entry['vul_id'],
                'fixed_commit_sha': entry['fixed_commit_sha'],
                'status': 'not_found',
                'details': entry['details']
            })
        
        # 写入文件为空的记录
        for entry in empty_files:
            writer.writerow({
                'repo': entry['repo'],
                'vul_id': entry['vul_id'],
                'fixed_commit_sha': entry['fixed_commit_sha'],
                'status': 'empty',
                'details': entry['details']
            })
        
        # 写入所有false的记录
        for entry in all_false_files:
            writer.writerow({
                'repo': entry['repo'],
                'vul_id': entry['vul_id'],
                'fixed_commit_sha': entry['fixed_commit_sha'],
                'status': 'all_false',
                'details': entry['details']
            })
        
        # 写入有vulnerable的记录
        for entry in has_vulnerable_files:
            writer.writerow({
                'repo': entry['repo'],
                'vul_id': entry['vul_id'],
                'fixed_commit_sha': entry['fixed_commit_sha'],
                'status': 'has_vulnerable',
                'details': entry['details']
            })
        
        # 写入错误的记录
        for entry in error_files:
            writer.writerow({
                'repo': entry['repo'],
                'vul_id': entry['vul_id'],
                'fixed_commit_sha': entry['fixed_commit_sha'],
                'status': 'error',
                'details': entry['details']
            })
    
    print(f"\n详细结果已保存到: {output_file}")
    
    # 打印部分示例
    if empty_files:
        print(f"\n=== 文件为空的示例 (前5条) ===")
        for entry in empty_files[:5]:
            print(f"  {entry['repo']}, {entry['vul_id']}, {entry['fixed_commit_sha']}")
    
    if all_false_files:
        print(f"\n=== 所有target的is_vulnerable都为false的示例 (前5条) ===")
        for entry in all_false_files[:5]:
            print(f"  {entry['repo']}, {entry['vul_id']}, {entry['fixed_commit_sha']}")
if __name__ == "__main__":
    main()