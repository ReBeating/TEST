#!/usr/bin/env python3
"""
生成五个分类CSV文件：
1. 存在target是TP的漏洞
2. target只有FP的漏洞
3. 所有is_vulnerable=false的漏洞
4. 没有target进入验证阶段的漏洞 (findings不存在或为空)
5. 不知道TP还是FP，但是有is_vulnerable=true的漏洞
"""

import os
import json
import csv
from pathlib import Path
from collections import defaultdict

def load_checked_list(csv_path):
    """加载checked_list.csv，返回{(vul_id, repo): [judgements]}的字典"""
    judgements = defaultdict(list)
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            vul_id = row['vul_id'].strip()
            repo = row['repo'].strip()
            judgement = row['judgement'].strip()
            fixed_commit_sha = row['fixed_commit_sha'].strip()
            
            # 使用 (vul_id, repo) 作为键
            key = (vul_id, repo)
            judgements[key].append({
                'judgement': judgement,
                'fixed_commit_sha': fixed_commit_sha
            })
    
    return judgements

def load_added_vul_list(csv_path):
    """加载added_vul_list.csv，返回{(vul_id, repo): fixed_commit_sha}的字典"""
    commit_shas = {}
    
    if not os.path.exists(csv_path):
        print(f"警告: {csv_path} 不存在")
        return commit_shas
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            repo = row['repo'].strip()
            vul_id = row['vul_id'].strip()
            fixed_commit_sha = row['fixed_commit_sha'].strip()
            
            key = (vul_id, repo)
            commit_shas[key] = fixed_commit_sha
    
    return commit_shas

def load_repo_list(csv_path):
    """加载repo_list.csv，返回允许的repo集合"""
    repos = set()
    
    if not os.path.exists(csv_path):
        print(f"警告: {csv_path} 不存在，将统计所有repo")
        return None
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            repo = row['repo'].strip()
            if repo:
                repos.add(repo)
    
    return repos

def check_findings_has_vulnerable(findings_file):
    """检查findings文件是否存在且有is_vulnerable=true的项"""
    if not os.path.exists(findings_file):
        return False
    
    try:
        with open(findings_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list) or len(data) == 0:
            return False
        
        # 检查是否有任何 item 的 is_vulnerable 为 true
        for item in data:
            if isinstance(item, dict) and item.get('is_vulnerable', False):
                return True
        
        return False
    
    except Exception as e:
        return False

def check_findings_all_false(findings_file):
    """检查findings文件是否所有is_vulnerable都为false"""
    if not os.path.exists(findings_file):
        return False
    
    try:
        with open(findings_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list) or len(data) == 0:
            return False
        
        # 检查是否所有 item 的 is_vulnerable 都为 false
        for item in data:
            if isinstance(item, dict) and item.get('is_vulnerable', False):
                return False
        
        return True  # 所有都是 false
    
    except Exception as e:
        return False

def check_findings_not_exist_or_empty(findings_file):
    """检查findings文件是否不存在或为空"""
    if not os.path.exists(findings_file):
        return True
    
    try:
        with open(findings_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list) or len(data) == 0:
            return True
        
        return False
    
    except Exception:
        return True

def find_all_vulnerabilities_by_repo(results_dir):
    """
    遍历 outputs/results 目录，找到所有(vul_id, repo, findings_file)
    
    返回: [(vul_id, repo, findings_file), ...]
    """
    results = []
    
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"错误: {results_dir} 目录不存在")
        return results
    
    # 遍历所有组织/仓库目录
    for org_dir in results_path.iterdir():
        if not org_dir.is_dir():
            continue
        
        for repo_dir in org_dir.iterdir():
            if not repo_dir.is_dir():
                continue
            
            # 获取完整的 repo 名称 (org/repo)
            repo_full_name = f"{org_dir.name}/{repo_dir.name}"
            
            # 查找该仓库中的所有 findings 文件
            for findings_file in repo_dir.glob('*_repo_findings.json'):
                filename = findings_file.name
                vul_id = filename.replace('_repo_findings.json', '')
                
                results.append((vul_id, repo_full_name, str(findings_file)))
    
    return results

def categorize_vulnerabilities(results_dir, checked_list_path, added_vul_list_path, repo_list_path):
    """
    分类所有漏洞
    返回五个列表，每个列表包含(vul_id, repo, fixed_commit_sha)元组
    """
    judgements = load_checked_list(checked_list_path)
    added_commit_shas = load_added_vul_list(added_vul_list_path)
    allowed_repos = load_repo_list(repo_list_path)
    
    # 五个分类
    tp_vulnerabilities = []  # 1. 存在target是TP的漏洞
    only_fp_vulnerabilities = []  # 2. target只有FP的漏洞
    all_false_vulnerabilities = []  # 3. 所有is_vulnerable=false的漏洞
    no_findings_vulnerabilities = []  # 4. 没有target进入验证阶段的漏洞 (findings不存在或为空)
    unknown_vulnerabilities = []  # 5. 不知道TP还是FP，但是有is_vulnerable=true的漏洞
    
    # 扫描所有(vul_id, repo, findings_file)
    all_vul_repos = find_all_vulnerabilities_by_repo(results_dir)
    
    for vul_id, repo, findings_file in all_vul_repos:
        # 如果指定了allowed_repos，则只处理在列表中的repo
        if allowed_repos is not None and repo not in allowed_repos:
            continue
        
        key = (vul_id, repo)
        
        # 获取fixed_commit_sha
        # 优先从checked_list中获取，如果没有或为空则从added_vul_list中获取
        fixed_commit_sha = ''
        if key in judgements and judgements[key]:
            fixed_commit_sha = judgements[key][0]['fixed_commit_sha']
        
        # 如果没有从checked_list获取到，尝试从added_vul_list获取
        if not fixed_commit_sha and key in added_commit_shas:
            fixed_commit_sha = added_commit_shas[key]
        
        # 检查findings文件状态
        not_exist_or_empty = check_findings_not_exist_or_empty(findings_file)
        all_false = check_findings_all_false(findings_file)
        has_vulnerable = check_findings_has_vulnerable(findings_file)
        
        # 类别4: findings文件不存在或为空
        if not_exist_or_empty:
            no_findings_vulnerabilities.append((vul_id, repo, fixed_commit_sha))
            continue
        
        # 类别3: 所有is_vulnerable都为false
        if all_false:
            all_false_vulnerabilities.append((vul_id, repo, fixed_commit_sha))
            continue
        
        # 如果有is_vulnerable=true，检查在checked_list中的判断
        if has_vulnerable:
            if key not in judgements:
                # 类别5: 没有在checked_list中，说明不知道TP还是FP
                unknown_vulnerabilities.append((vul_id, repo, fixed_commit_sha))
            else:
                # 获取所有判断
                all_judgements = [item['judgement'] for item in judgements[key]]
                
                has_tp = 'TP' in all_judgements
                has_fp = 'FP' in all_judgements
                
                if has_tp:
                    # 类别1: 存在TP
                    tp_vulnerabilities.append((vul_id, repo, fixed_commit_sha))
                elif has_fp and not has_tp:
                    # 类别2: 只有FP (没有TP)
                    only_fp_vulnerabilities.append((vul_id, repo, fixed_commit_sha))
                else:
                    # 其他判断（如SAFE、Unknown等），也归入类别5
                    unknown_vulnerabilities.append((vul_id, repo, fixed_commit_sha))
    
    return tp_vulnerabilities, only_fp_vulnerabilities, all_false_vulnerabilities, no_findings_vulnerabilities, unknown_vulnerabilities

def write_csv(filename, data):
    """写入CSV文件"""
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['vul_id', 'repo', 'fixed_commit_sha'])
        # 按照 repo, vul_id 排序
        for row in sorted(data, key=lambda x: (x[1], x[0])):
            writer.writerow(row)
    print(f"已写入 {filename}: {len(data)} 条记录")

def main():
    print("=" * 80)
    print("开始生成分类CSV文件...")
    print("=" * 80)
    
    results_dir = 'outputs/results'
    checked_list_path = 'results/checked_list.csv'
    added_vul_list_path = 'inputs/added_vul_list.csv'
    repo_list_path = 'inputs/repo_list.csv'
    
    if not os.path.exists(checked_list_path):
        print(f"错误: {checked_list_path} 不存在")
        return
    
    if not os.path.exists(results_dir):
        print(f"错误: {results_dir} 目录不存在")
        return
    
    # 分类漏洞
    print("\n正在分析漏洞...")
    tp_vulns, only_fp_vulns, all_false_vulns, no_findings_vulns, unknown_vulns = categorize_vulnerabilities(
        results_dir, checked_list_path, added_vul_list_path, repo_list_path
    )
    
    # 写入CSV文件
    print("\n正在写入CSV文件...")
    write_csv('category_1_has_tp.csv', tp_vulns)
    write_csv('category_2_only_fp.csv', only_fp_vulns)
    write_csv('category_3_all_false.csv', all_false_vulns)
    write_csv('category_4_no_findings.csv', no_findings_vulns)
    write_csv('category_5_unknown.csv', unknown_vulns)
    
    print("\n" + "=" * 80)
    print("统计总结:")
    print("=" * 80)
    print(f"类别1 - 存在target是TP的漏洞: {len(tp_vulns)}")
    print(f"类别2 - target只有FP的漏洞: {len(only_fp_vulns)}")
    print(f"类别3 - 所有is_vulnerable=false的漏洞: {len(all_false_vulns)}")
    print(f"类别4 - findings不存在或为空的漏洞: {len(no_findings_vulns)}")
    print(f"类别5 - 不知道TP还是FP，但有is_vulnerable=true的漏洞: {len(unknown_vulns)}")
    total = len(tp_vulns) + len(only_fp_vulns) + len(all_false_vulns) + len(no_findings_vulns) + len(unknown_vulns)
    print(f"总计: {total}")
    print("=" * 80)

if __name__ == '__main__':
    main()
