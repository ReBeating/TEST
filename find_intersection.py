#!/usr/bin/env python3
"""
计算 vulnerabilities_with_confidence.csv 和 category_2_only_fp.csv 的交集
"""

import csv
from pathlib import Path


def read_vulnerabilities(csv_file):
    """读取漏洞列表，返回 (repo, vul_id) 的集合"""
    vulns = set()
    vulns_list = []
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            repo = row['repo'].strip()
            vul_id = row['vul_id'].strip()
            vulns.add((repo, vul_id))
            vulns_list.append(row)
    
    return vulns, vulns_list


def main():
    # 读取两个文件
    confidence_file = Path("vulnerabilities_with_confidence.csv")
    category2_file = Path("category_2_only_fp.csv")
    
    print(f"读取 {confidence_file}...")
    confidence_set, confidence_list = read_vulnerabilities(confidence_file)
    print(f"包含 {len(confidence_set)} 个漏洞\n")
    
    print(f"读取 {category2_file}...")
    category2_set, category2_list = read_vulnerabilities(category2_file)
    print(f"包含 {len(category2_set)} 个漏洞\n")
    
    # 计算交集
    intersection = confidence_set & category2_set
    
    print(f"{'='*60}")
    print(f"交集结果:")
    print(f"{'='*60}")
    print(f"vulnerabilities_with_confidence.csv: {len(confidence_set)} 个漏洞")
    print(f"category_2_only_fp.csv: {len(category2_set)} 个漏洞")
    print(f"交集: {len(intersection)} 个漏洞")
    print(f"{'='*60}\n")
    
    if intersection:
        # 获取交集中漏洞的完整信息
        intersection_details = []
        confidence_dict = {(row['repo'].strip(), row['vul_id'].strip()): row 
                          for row in confidence_list}
        category2_dict = {(row['repo'].strip(), row['vul_id'].strip()): row 
                         for row in category2_list}
        
        for repo, vul_id in sorted(intersection):
            conf_row = confidence_dict.get((repo, vul_id), {})
            cat2_row = category2_dict.get((repo, vul_id), {})
            
            intersection_details.append({
                'repo': repo,
                'vul_id': vul_id,
                'fixed_commit_sha': cat2_row.get('fixed_commit_sha', ''),
                'file_path': conf_row.get('file_path', ''),
                'confidence_locations': conf_row.get('confidence_locations', '')
            })
        
        # 保存交集结果
        output_file = Path("intersection_confidence_category2.csv")
        print(f"保存交集到 {output_file}...")
        
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'repo', 'vul_id', 'fixed_commit_sha', 
                'file_path', 'confidence_locations'
            ])
            writer.writeheader()
            writer.writerows(intersection_details)
        
        print(f"已保存 {len(intersection_details)} 个漏洞\n")
        
        # 按仓库统计
        repo_counts = {}
        for repo, vul_id in intersection:
            repo_counts[repo] = repo_counts.get(repo, 0) + 1
        
        print("按仓库统计:")
        for repo, count in sorted(repo_counts.items(), key=lambda x: -x[1]):
            print(f"  {repo}: {count} 个")
        
        # 显示前 10 个示例
        print(f"\n前 10 个交集漏洞示例:")
        for i, detail in enumerate(intersection_details[:10], 1):
            print(f"{i}. {detail['repo']} / {detail['vul_id']}")
            if detail['file_path']:
                print(f"   文件: {detail['file_path']}")
            if detail['confidence_locations']:
                print(f"   Confidence 位置: {detail['confidence_locations']}")
    else:
        print("没有找到交集")
    
    # 计算差集
    only_in_confidence = confidence_set - category2_set
    only_in_category2 = category2_set - confidence_set
    
    print(f"\n{'='*60}")
    print("差集统计:")
    print(f"{'='*60}")
    print(f"只在 vulnerabilities_with_confidence.csv 中: {len(only_in_confidence)} 个")
    print(f"只在 category_2_only_fp.csv 中: {len(only_in_category2)} 个")


if __name__ == "__main__":
    main()
