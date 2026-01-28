#!/usr/bin/env python3
"""
统计合并数据集中各个软件的漏洞分布
"""

import csv
from pathlib import Path
from collections import Counter


def analyze_distribution(input_file):
    """统计各个仓库的漏洞数量"""
    
    print(f"分析文件: {input_file}\n")
    
    # 统计每个仓库的漏洞数量
    repo_counter = Counter()
    total_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            repo = row['repo'].strip()
            repo_counter[repo] += 1
            total_count += 1
    
    # 按漏洞数量降序排序
    sorted_repos = sorted(repo_counter.items(), key=lambda x: x[1], reverse=True)
    
    # 打印结果
    print(f"{'仓库':<40} {'漏洞数量':>10} {'占比':>10}")
    print("=" * 65)
    
    for repo, count in sorted_repos:
        percentage = (count / total_count) * 100
        print(f"{repo:<40} {count:>10} {percentage:>9.2f}%")
    
    print("=" * 65)
    print(f"{'总计':<40} {total_count:>10} {100.0:>9.2f}%")
    print(f"\n总共 {len(sorted_repos)} 个不同的软件仓库")
    
    # 保存到CSV文件
    output_file = Path(str(input_file).replace('.csv', '_distribution.csv'))
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['repo', 'count', 'percentage'])
        for repo, count in sorted_repos:
            percentage = (count / total_count) * 100
            writer.writerow([repo, count, f"{percentage:.2f}"])
    
    print(f"\n分布统计已保存到: {output_file}")


if __name__ == "__main__":
    input_file = Path("merged_dataset_with_category1.csv")
    analyze_distribution(input_file)
