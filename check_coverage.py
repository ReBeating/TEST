#!/usr/bin/env python3
"""
统计 3643_0day_vul_list.csv 和 diff_vul_list.csv 合并后
是否能够覆盖 merged_0day_vul_list.csv 中的所有行
"""

import csv
from pathlib import Path


def read_csv_to_set(filepath):
    """读取CSV文件并返回一个包含所有行的集合（排除表头）"""
    rows = set()
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # 跳过表头
        for row in reader:
            # 将行转换为元组以便可以加入集合
            rows.add(tuple(row))
    return rows, header


def main():
    # 文件路径
    file1 = Path('inputs/3643_0day_vul_list.csv')
    file2 = Path('inputs/diff_vul_list.csv')
    merged_file = Path('inputs/merged_0day_vul_list.csv')
    
    # 检查文件是否存在
    for f in [file1, file2, merged_file]:
        if not f.exists():
            print(f"错误: 文件 {f} 不存在")
            return
    
    # 读取三个文件
    print("正在读取文件...")
    rows_3643, header1 = read_csv_to_set(file1)
    rows_diff, header2 = read_csv_to_set(file2)
    rows_merged, header3 = read_csv_to_set(merged_file)
    
    print(f"\n文件统计:")
    print(f"  3643_0day_vul_list.csv: {len(rows_3643)} 行")
    print(f"  diff_vul_list.csv: {len(rows_diff)} 行")
    print(f"  merged_0day_vul_list.csv: {len(rows_merged)} 行")
    
    # 合并前两个文件
    combined = rows_3643.union(rows_diff)
    print(f"\n合并后的总行数: {len(combined)} 行")
    
    # 检查重叠
    overlap = rows_3643.intersection(rows_diff)
    print(f"  其中 3643 和 diff 的重叠行数: {len(overlap)} 行")
    
    # 检查覆盖情况
    print(f"\n覆盖情况分析:")
    
    # merged中有但combined中没有的
    missing_in_combined = rows_merged - combined
    
    # combined中有但merged中没有的
    extra_in_combined = combined - rows_merged
    
    # 两者的交集
    intersection = rows_merged.intersection(combined)
    
    print(f"  merged 中的行在 combined 中的覆盖数: {len(intersection)} / {len(rows_merged)}")
    print(f"  覆盖率: {len(intersection) / len(rows_merged) * 100:.2f}%")
    
    if len(missing_in_combined) == 0:
        print(f"\n✅ 结论: 3643_0day_vul_list.csv 和 diff_vul_list.csv 合并后")
        print(f"   完全覆盖了 merged_0day_vul_list.csv 中的所有 {len(rows_merged)} 行!")
    else:
        print(f"\n❌ 结论: 合并后无法完全覆盖 merged_0day_vul_list.csv")
        print(f"   缺失 {len(missing_in_combined)} 行")
        
        # 显示前10个缺失的行
        print(f"\n缺失的行示例 (最多显示10行):")
        for i, row in enumerate(sorted(missing_in_combined)[:10], 1):
            print(f"  {i}. {','.join(row)}")
    
    if len(extra_in_combined) > 0:
        print(f"\n额外信息:")
        print(f"  combined 中有但 merged 中没有的行数: {len(extra_in_combined)}")
        print(f"  这些是额外的行示例 (最多显示10行):")
        for i, row in enumerate(sorted(extra_in_combined)[:10], 1):
            print(f"  {i}. {','.join(row)}")
    
    # 详细统计
    print(f"\n详细统计:")
    print(f"  3643 独有的行数: {len(rows_3643 - rows_diff - rows_merged)}")
    print(f"  diff 独有的行数: {len(rows_diff - rows_3643 - rows_merged)}")
    print(f"  merged 独有的行数: {len(rows_merged - rows_3643 - rows_diff)}")
    print(f"  三者共有的行数: {len(rows_3643.intersection(rows_diff).intersection(rows_merged))}")


if __name__ == '__main__':
    main()
