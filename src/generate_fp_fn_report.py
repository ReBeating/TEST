#!/usr/bin/env python3
"""
生成FP/FN详细报告
包含：漏洞补丁、版本代码、analysis_report、错误类型标注
"""

import os
import json
import sqlite3
import csv
from typing import Dict, List, Tuple, Set
from collections import defaultdict


def load_vul_dict(path: str) -> Dict:
    """加载漏洞字典"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_ground_truth(db_path: str) -> Dict[str, Dict[str, str]]:
    """加载ground truth: {vul_id: {version: tag}}"""
    if not os.path.exists(db_path):
        return {}
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT vul_id, version, tag FROM benchmark_symbols")
    rows = cursor.fetchall()
    conn.close()
    
    gt = {}
    for vul_id, ver, tag in rows:
        if vul_id not in gt:
            gt[vul_id] = {}
        gt[vul_id][ver] = tag
    return gt


def load_findings_map(results_dir: str) -> Dict[str, Dict[str, List[Tuple[str, bool]]]]:
    """
    加载findings结果映射
    返回: {vul_id: {version: [(target_func, is_vulnerable)]}}
    """
    findings_map = defaultdict(lambda: defaultdict(list))
    
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith("_benchmark_findings.json"):
                vul_id = file.replace("_benchmark_findings.json", "")
                path = os.path.join(root, file)
                
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        findings = json.load(f)
                    
                    for finding in findings:
                        target_func = finding.get('target_func', '')
                        is_vuln = finding.get('is_vulnerable', False)
                        
                        # 从target_func提取版本: "vul:version:func" 或 "pre:version:func" 或 "fix:version:func"
                        parts = target_func.split(':')
                        if len(parts) >= 2:
                            version = parts[1]
                            findings_map[vul_id][version].append((target_func, is_vuln, finding))
                
                except Exception as e:
                    print(f"[Warning] Failed to load {path}: {e}")
    
    return findings_map


def load_candidates_map(results_dir: str) -> Dict[str, Set[str]]:
    """
    加载候选版本映射（用于判断是否search miss）
    返回: {vul_id: set(versions)}
    """
    candidates_map = defaultdict(set)
    
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith("_benchmark_findings.json"):
                vul_id = file.replace("_benchmark_findings.json", "")
                path = os.path.join(root, file)
                
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        findings = json.load(f)
                    
                    for finding in findings:
                        target_func = finding.get('target_func', '')
                        parts = target_func.split(':')
                        if len(parts) >= 2:
                            version = parts[1]
                            candidates_map[vul_id].add(version)
                
                except Exception:
                    pass
    
    return candidates_map


def load_allowed_cves(csv_path: str) -> Set[str]:
    """加载允许的CVE列表"""
    if not csv_path or not os.path.exists(csv_path):
        return None
    
    allowed = set()
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cve_id = row.get('vul_id', '').strip()
            if cve_id:
                allowed.add(cve_id)
    return allowed


def format_code_block(code: str, language: str = "") -> str:
    """格式化代码块为Markdown"""
    if not code:
        return "*代码不可用*"
    return f"```{language}\n{code}\n```"


def get_version_code(vul_dict: Dict, vul_id: str, version: str, tag: str) -> Dict[str, Dict[str, str]]:
    """
    获取特定版本的代码
    返回: {file_path: {func_name: code}}
    """
    if vul_id not in vul_dict:
        return {}
    
    vul_data = vul_dict[vul_id]
    versions_data = vul_data.get('versions', {})
    
    # 根据tag选择对应的版本数据
    tag_data = versions_data.get(tag, {})
    
    if version in tag_data:
        return tag_data[version]
    
    return {}


def load_candidates_data(results_dir: str, vul_id: str) -> List[Dict]:
    """加载candidates数据"""
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file == f"{vul_id}_benchmark_candidates.json":
                path = os.path.join(root, file)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except Exception:
                    pass
    return []


def generate_report_for_error(
    vul_id: str,
    version: str,
    tag: str,
    error_type: str,
    vul_dict: Dict,
    findings: List[Tuple[str, bool, Dict]],
    output_dir: str,
    results_dir: str = 'outputs/results'
):
    """为单个错误生成报告"""
    
    # 获取漏洞基本信息
    vul_data = vul_dict.get(vul_id, {})
    repo = vul_data.get('repo', 'Unknown')
    
    # 创建报告文件名
    safe_version = version.replace('/', '_').replace(':', '_')
    report_filename = f"{error_type}_{vul_id}_{safe_version}.md"
    report_path = os.path.join(output_dir, report_filename)
    
    # 开始生成报告
    lines = []
    lines.append(f"# {error_type} Report: {vul_id} @ {version}\n")
    lines.append(f"**Repository**: `{repo}`  ")
    lines.append(f"**Error Type**: `{error_type}`  ")
    lines.append(f"**Version**: `{version}`  ")
    lines.append(f"**Expected Tag**: `{tag}`\n")
    
    # ===== Section 1: 漏洞补丁信息 =====
    lines.append("---\n")
    lines.append("## 1. 漏洞补丁信息\n")
    
    # Fixing commits
    fixing_commits = vul_data.get('fixing_commits', [])
    if fixing_commits:
        lines.append("**修复提交**:\n")
        for commit in fixing_commits:
            lines.append(f"- `{commit}`\n")
    else:
        lines.append("*修复提交信息不可用*\n")
    
    lines.append("\n")
    
    # Patch modifications
    modifications = vul_data.get('modifications', {})
    if modifications:
        lines.append("### 补丁修改详情\n")
        for file_path, functions in modifications.items():
            lines.append(f"\n**文件**: `{file_path}`\n")
            for func_name, func_data in functions.items():
                lines.append(f"\n**函数**: `{func_name}`\n")
                
                # Only show diff
                if 'diff_code' in func_data:
                    lines.append(format_code_block(func_data['diff_code'], 'diff'))
                    lines.append("\n")
                else:
                    lines.append("*差异信息不可用*\n\n")
    else:
        lines.append("*补丁修改信息不可用*\n")
    
    # ===== Section 2: 版本代码 =====
    lines.append("\n---\n")
    lines.append(f"## 2. 版本代码 ({version}, tag={tag})\n")
    
    version_code = get_version_code(vul_dict, vul_id, version, tag)
    if version_code:
        for file_path, functions in version_code.items():
            lines.append(f"\n**文件**: `{file_path}`\n")
            for func_name, code in functions.items():
                lines.append(f"\n**函数**: `{func_name}`\n")
                lines.append(format_code_block(code, 'c'))
                lines.append("\n")
    else:
        lines.append("*版本代码不可用*\n")
    
    # ===== Section 3: 分析报告 =====
    lines.append("\n---\n")
    lines.append("## 3. 分析报告\n")
    
    if error_type.endswith("search-FN"):
        lines.append("**类型**: Search Miss (搜索阶段未找到候选)\n\n")
        
        # 加载candidates数据，显示搜索匹配信息
        candidates_data = load_candidates_data(results_dir, vul_id)
        if candidates_data:
            lines.append("### 搜索匹配结果\n\n")
            lines.append(f"**找到的候选函数数量**: {len(candidates_data)}\n\n")
            
            # 检查是否有匹配到该版本
            version_candidates = [c for c in candidates_data if version in c.get('target_func', '')]
            
            if version_candidates:
                lines.append(f"**匹配到版本 {version} 的候选**: {len(version_candidates)}\n\n")
                for idx, cand in enumerate(version_candidates, 1):
                    lines.append(f"#### 候选 {idx}: `{cand.get('target_func', 'N/A')}`\n\n")
                    lines.append(f"- **文件**: `{cand.get('target_file', 'N/A')}`\n")
                    lines.append(f"- **判定**: {cand.get('verdict', 'N/A')}\n")
                    lines.append(f"- **置信度**: {cand.get('confidence', 'N/A')}\n")
                    scores = cand.get('scores', {})
                    lines.append(f"- **Vuln分数**: {scores.get('score_vuln', 'N/A')}\n")
                    lines.append(f"- **Fix分数**: {scores.get('score_fix', 'N/A')}\n\n")
                    
                    # 添加详细的trace匹配信息
                    evidence = cand.get('evidence', {})
                    
                    # Vuln traces
                    vuln_traces = evidence.get('aligned_vuln_traces', [])
                    if vuln_traces:
                        lines.append(f"**漏洞模式匹配详情** ({len(vuln_traces)} 条trace):\n\n")
                        lines.append("| # | 补丁代码行 | 目标代码行 | 相似度 | 类型 |\n")
                        lines.append("|---|-----------|-----------|--------|------|\n")
                        for i, trace in enumerate(vuln_traces, 1):
                            slice_line = trace.get('slice_line', 'N/A')
                            target_line = trace.get('target_line', '*未匹配*')
                            similarity = trace.get('similarity', 0)
                            tag = trace.get('tag', 'N/A')
                            
                            # 去掉行号，完整显示
                            if slice_line and slice_line != 'N/A':
                                slice_display = slice_line.split('] ', 1)[-1] if '] ' in slice_line else slice_line
                            else:
                                slice_display = 'N/A'
                            
                            if target_line and target_line != '*未匹配*':
                                target_display = target_line.split('] ', 1)[-1] if '] ' in target_line else target_line
                            else:
                                target_display = '*未匹配*'
                            
                            lines.append(f"| {i} | `{slice_display}` | `{target_display}` | {similarity:.2f} | {tag} |\n")
                        lines.append("\n")
                    
                    # Fix traces
                    fix_traces = evidence.get('aligned_fix_traces', [])
                    if fix_traces:
                        lines.append(f"**修复模式匹配详情** ({len(fix_traces)} 条trace):\n\n")
                        lines.append("| # | 补丁代码行 | 目标代码行 | 相似度 | 类型 |\n")
                        lines.append("|---|-----------|-----------|--------|------|\n")
                        for i, trace in enumerate(fix_traces, 1):
                            slice_line = trace.get('slice_line', 'N/A')
                            target_line = trace.get('target_line', '*未匹配*')
                            similarity = trace.get('similarity', 0)
                            tag = trace.get('tag', 'N/A')
                            
                            if slice_line and slice_line != 'N/A':
                                slice_display = slice_line.split('] ', 1)[-1] if '] ' in slice_line else slice_line
                            else:
                                slice_display = 'N/A'
                            
                            if target_line and target_line != '*未匹配*':
                                target_display = target_line.split('] ', 1)[-1] if '] ' in target_line else target_line
                            else:
                                target_display = '*未匹配*'
                            
                            lines.append(f"| {i} | `{slice_display}` | `{target_display}` | {similarity:.2f} | {tag} |\n")
                        lines.append("\n")
                    
                    lines.append("---\n\n")
                
                lines.append("**说明**: 虽然搜索阶段找到了候选函数，但由于某些原因（如置信度过低、判定为PATCHED等）未被纳入最终的findings，导致该版本被判定为Search Miss。\n\n")
            else:
                lines.append(f"**未找到匹配版本 {version} 的候选函数**\n\n")
                
                # 显示所有找到的版本
                all_versions = set()
                for cand in candidates_data:
                    target_func = cand.get('target_func', '')
                    parts = target_func.split(':')
                    if len(parts) >= 2:
                        all_versions.add(parts[1])
                
                if all_versions:
                    lines.append(f"**搜索到的其他版本**: {', '.join(sorted(all_versions))}\n\n")
                
                lines.append("**可能原因**:\n")
                lines.append("1. 该版本的函数签名或代码结构变化较大，导致搜索算法未能匹配\n")
                lines.append("2. 该版本的函数名称或文件路径发生了变化\n")
                lines.append("3. 搜索算法的相似度阈值设置可能过高\n\n")
        else:
            lines.append("*无法加载candidates数据，可能该CVE的搜索结果未保存*\n\n")
            lines.append("**说明**: 搜索阶段完全未找到任何候选函数，这可能是因为：\n")
            lines.append("1. 补丁中的函数在该版本中不存在\n")
            lines.append("2. 函数名称或文件结构发生了重大变化\n")
            lines.append("3. 代码相似度低于搜索阈值\n\n")
    else:
        if findings:
            lines.append(f"**候选函数数量**: {len(findings)}\n\n")
            
            for idx, (target_func, is_vuln, finding_data) in enumerate(findings, 1):
                lines.append(f"### 候选 {idx}: `{target_func}`\n")
                lines.append(f"**判定**: {'VULNERABLE' if is_vuln else 'SAFE'}  \n")
                lines.append(f"**置信度**: {finding_data.get('confidence', 'N/A')}  \n\n")
                
                analysis_report = finding_data.get('analysis_report', '')
                if analysis_report:
                    lines.append("**详细分析**:\n")
                    lines.append(f"{analysis_report}\n\n")
                else:
                    lines.append("*分析报告不可用*\n\n")
                
                lines.append("---\n\n")
        else:
            lines.append("*无分析报告（可能是search miss）*\n")
    
    # ===== Section 4: 错误类型说明 =====
    lines.append("\n---\n")
    lines.append("## 4. 错误类型说明\n")
    
    error_explanations = {
        "search-FN": "搜索阶段漏报：在搜索阶段未找到候选函数，导致漏洞版本未被检测",
        "verification-FN": "验证阶段漏报：搜索找到了候选函数，但验证阶段判定为SAFE，导致漏洞版本未被检测",
        "pre-FP": "Pre-patch误报：在引入补丁之前的版本被错误判定为VULNERABLE",
        "fix-FP": "Fix误报：在应用补丁之后的版本被错误判定为VULNERABLE"
    }
    
    explanation = error_explanations.get(error_type, "未知错误类型")
    lines.append(f"**{error_type}**: {explanation}\n")
    
    # 写入文件
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    return report_path


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate detailed FP/FN reports')
    parser.add_argument('--csv', type=str, default='inputs/filtered_1day_vul_list.csv',
                        help='Path to CSV file with allowed CVEs')
    parser.add_argument('--db', type=str, default='databases/idx_benchmark.db',
                        help='Path to benchmark database')
    parser.add_argument('--vul-dict', type=str, default='inputs/1day_vul_dict.json',
                        help='Path to vulnerability dictionary JSON')
    parser.add_argument('--results', type=str, default='outputs/results',
                        help='Path to results directory')
    parser.add_argument('--output', type=str, default='outputs/fp_fn_reports',
                        help='Output directory for reports')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    print(f"[*] Loading vulnerability dictionary from {args.vul_dict}...")
    vul_dict = load_vul_dict(args.vul_dict)
    
    print(f"[*] Loading ground truth from {args.db}...")
    ground_truth = load_ground_truth(args.db)
    
    print(f"[*] Loading findings from {args.results}...")
    findings_map = load_findings_map(args.results)
    
    print(f"[*] Loading candidates map...")
    candidates_map = load_candidates_map(args.results)
    
    # 加载允许的CVE列表
    allowed_cves = load_allowed_cves(args.csv)
    if allowed_cves:
        print(f"[*] Filter enabled: {len(allowed_cves)} CVEs allowed.")
    
    # 统计
    stats = {
        'search-FN': 0,
        'verification-FN': 0,
        'pre-FP': 0,
        'fix-FP': 0
    }
    
    generated_reports = []
    
    print(f"\n[*] Generating reports...\n")
    
    # 遍历ground truth
    for vul_id, versions in ground_truth.items():
        # 过滤CVE
        if allowed_cves and vul_id not in allowed_cves:
            continue
        
        for version, tag in versions.items():
            # 确定期望结果
            expected_vuln = (tag == 'vul')
            
            # 获取实际结果
            version_findings = findings_map.get(vul_id, {}).get(version, [])
            has_candidates = vul_id in candidates_map and version in candidates_map[vul_id]
            
            # 判定实际是否vulnerable（任意一个函数判定为vulnerable则整个版本为vulnerable）
            actual_vuln = any(is_vuln for _, is_vuln, _ in version_findings)
            
            error_type = None
            
            # 判断错误类型
            if expected_vuln and not actual_vuln:
                # FN
                if not has_candidates:
                    error_type = "search-FN"
                else:
                    error_type = "verification-FN"
            elif not expected_vuln and actual_vuln:
                # FP
                if tag == 'pre':
                    error_type = "pre-FP"
                elif tag == 'fix':
                    error_type = "fix-FP"
            
            # 生成报告
            if error_type:
                stats[error_type] += 1
                
                try:
                    report_path = generate_report_for_error(
                        vul_id=vul_id,
                        version=version,
                        tag=tag,
                        error_type=error_type,
                        vul_dict=vul_dict,
                        findings=version_findings,
                        output_dir=args.output,
                        results_dir=args.results
                    )
                    generated_reports.append(report_path)
                    print(f"[+] Generated: {os.path.basename(report_path)} ({error_type})")
                except Exception as e:
                    print(f"[!] Failed to generate report for {vul_id}:{version} - {e}")
    
    # 生成CSV文件
    csv_path = os.path.join(args.output, "fp_fn_summary.csv")
    with open(csv_path, 'w', encoding='utf-8', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['vul_id', 'version', 'repo', 'fixed_commit_sha', 'fn_fp'])
        
        # 重新遍历ground truth生成CSV
        for vul_id, versions in ground_truth.items():
            if allowed_cves and vul_id not in allowed_cves:
                continue
            
            vul_data = vul_dict.get(vul_id, {})
            repo = vul_data.get('repo', 'Unknown')
            fixing_commits = vul_data.get('fixing_commits', [])
            # 使用第一个修复提交，如果有多个则用分号分隔
            fixed_commit_sha = ';'.join(fixing_commits) if fixing_commits else 'N/A'
            
            for version, tag in versions.items():
                expected_vuln = (tag == 'vul')
                version_findings = findings_map.get(vul_id, {}).get(version, [])
                has_candidates = vul_id in candidates_map and version in candidates_map[vul_id]
                actual_vuln = any(is_vuln for _, is_vuln, _ in version_findings)
                
                error_type = None
                
                if expected_vuln and not actual_vuln:
                    if not has_candidates:
                        error_type = "search-FN"
                    else:
                        error_type = "verification-FN"
                elif not expected_vuln and actual_vuln:
                    if tag == 'pre':
                        error_type = "pre-FP"
                    elif tag == 'fix':
                        error_type = "fix-FP"
                
                if error_type:
                    csv_writer.writerow([vul_id, version, repo, fixed_commit_sha, error_type])
    
    print(f"[*] CSV文件已生成: {csv_path}\n")
    
    # 输出统计
    print("\n" + "="*60)
    print("报告生成完成！")
    print("="*60)
    print(f"总报告数: {len(generated_reports)}")
    print(f"  - Search Miss FN:      {stats['search-FN']}")
    print(f"  - Verification FN:     {stats['verification-FN']}")
    print(f"  - Pre-patch FP:        {stats['pre-FP']}")
    print(f"  - Fix FP:              {stats['fix-FP']}")
    print(f"\n报告保存至: {args.output}")
    print("="*60)
    
    # 生成索引文件
    index_path = os.path.join(args.output, "INDEX.md")
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write("# FP/FN 报告索引\n\n")
        f.write(f"生成时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## 统计摘要\n\n")
        f.write(f"- **总报告数**: {len(generated_reports)}\n")
        f.write(f"- **Search Miss FN**: {stats['search-FN']}\n")
        f.write(f"- **Verification FN**: {stats['verification-FN']}\n")
        f.write(f"- **Pre-patch FP**: {stats['pre-FP']}\n")
        f.write(f"- **Fix FP**: {stats['fix-FP']}\n\n")
        
        f.write("## 报告列表\n\n")
        
        # 按错误类型分组
        for error_type in ['search-FN', 'verification-FN', 'pre-FP', 'fix-FP']:
            f.write(f"### {error_type} ({stats[error_type]})\n\n")
            
            type_reports = [r for r in generated_reports if error_type in os.path.basename(r)]
            for report in sorted(type_reports):
                basename = os.path.basename(report)
                f.write(f"- [{basename}]({basename})\n")
            
            f.write("\n")
    
    print(f"\n[*] 索引文件已生成: {index_path}")


if __name__ == "__main__":
    main()
