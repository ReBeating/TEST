#!/usr/bin/env python3
"""
将 cost_res.csv 中的数值乘以指定比率
"""

import csv
from pathlib import Path

def scale_csv_values(input_path: str, output_path: str, scale_factor: float):
    """
    读取 CSV 文件，将数值列乘以比率后保存
    
    Args:
        input_path: 输入 CSV 文件路径
        output_path: 输出 CSV 文件路径
        scale_factor: 缩放因子
    """
    rows = []
    
    with open(input_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows.append(header)
        
        for row in reader:
            new_row = [row[0]]  # 保留第一列（phase名称）
            for value in row[1:]:
                try:
                    num = float(value)
                    scaled = num * scale_factor
                    # 保留两位小数
                    new_row.append(f"{scaled:.2f}")
                except ValueError:
                    new_row.append(value)
            rows.append(new_row)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    
    print(f"已将 {input_path} 中的数值乘以 {scale_factor:.6f}")
    print(f"结果保存到 {output_path}")


if __name__ == "__main__":
    # 比率 2273/242
    SCALE_FACTOR = 2273 / 242
    
    # 文件路径
    input_file = Path(__file__).parent.parent / "results" / "cost_res.csv"
    output_file = input_file  # 覆盖原文件，如需保留原文件可改为其他路径
    
    print(f"缩放因子: 2273/242 = {SCALE_FACTOR:.6f}")
    
    # 读取原始数据并显示
    print("\n原始数据:")
    with open(input_file, 'r') as f:
        print(f.read())
    
    # 执行缩放
    scale_csv_values(str(input_file), str(output_file), SCALE_FACTOR)
    
    # 显示结果
    print("\n缩放后数据:")
    with open(output_file, 'r') as f:
        print(f.read())
