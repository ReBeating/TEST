#!/usr/bin/env python3
"""
从CSV文件中随机抽取满足95%置信度的样本量，生成新的CSV文件。

样本量计算公式（有限总体修正）：
n = (Z² × p × (1-p) × N) / (e² × (N-1) + Z² × p × (1-p))

其中：
- Z = 1.96 (95%置信度)
- p = 0.5 (最大方差假设)
- e = 误差范围 (默认5%)
- N = 总体大小
"""

import argparse
import math
import random
import pandas as pd
from pathlib import Path


def calculate_sample_size(
    population_size: int,
    confidence_level: float = 0.95,
    margin_of_error: float = 0.05,
    population_proportion: float = 0.5
) -> int:
    """
    计算满足指定置信度的样本量（有限总体修正）。
    
    Args:
        population_size: 总体大小 N
        confidence_level: 置信水平（默认95%）
        margin_of_error: 误差范围（默认5%）
        population_proportion: 总体比例估计（默认0.5，最保守）
    
    Returns:
        所需的样本量
    """
    # Z值对照表
    z_scores = {
        0.80: 1.282,
        0.90: 1.645,
        0.95: 1.96,
        0.99: 2.576
    }
    
    z = z_scores.get(confidence_level, 1.96)
    p = population_proportion
    e = margin_of_error
    N = population_size
    
    # 无限总体样本量
    n_infinite = (z ** 2 * p * (1 - p)) / (e ** 2)
    
    # 有限总体修正
    n_adjusted = n_infinite / (1 + (n_infinite - 1) / N)
    
    return math.ceil(n_adjusted)


def sample_csv(
    input_file: str,
    output_file: str = None,
    confidence_level: float = 0.95,
    margin_of_error: float = 0.05,
    random_seed: int = None
) -> tuple[pd.DataFrame, int, int]:
    """
    从CSV文件中随机抽取满足置信度要求的样本。
    
    Args:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径（默认为 input_sampled.csv）
        confidence_level: 置信水平（默认95%）
        margin_of_error: 误差范围（默认5%）
        random_seed: 随机种子（可选，用于可重复性）
    
    Returns:
        (抽样后的DataFrame, 总体大小, 样本量)
    """
    # 读取CSV
    df = pd.read_csv(input_file)
    population_size = len(df)
    
    # 计算所需样本量
    sample_size = calculate_sample_size(
        population_size,
        confidence_level,
        margin_of_error
    )
    
    # 确保样本量不超过总体大小
    sample_size = min(sample_size, population_size)
    
    # 设置随机种子
    if random_seed is not None:
        random.seed(random_seed)
    
    # 随机抽样
    sampled_df = df.sample(n=sample_size, random_state=random_seed)
    
    # 生成输出文件名
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}_sampled{input_path.suffix}"
    
    # 保存结果
    sampled_df.to_csv(output_file, index=False)
    
    return sampled_df, population_size, sample_size


def main():
    parser = argparse.ArgumentParser(
        description="从CSV中随机抽取满足95%置信度的样本量"
    )
    parser.add_argument(
        "input_file",
        help="输入CSV文件路径"
    )
    parser.add_argument(
        "-o", "--output",
        help="输出CSV文件路径（默认为 input_sampled.csv）",
        default=None
    )
    parser.add_argument(
        "-c", "--confidence",
        type=float,
        default=0.95,
        choices=[0.80, 0.90, 0.95, 0.99],
        help="置信水平（默认0.95，可选0.80, 0.90, 0.95, 0.99）"
    )
    parser.add_argument(
        "-e", "--error",
        type=float,
        default=0.05,
        help="误差范围（默认0.05，即5%%）"
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=None,
        help="随机种子（用于可重复性）"
    )
    
    args = parser.parse_args()
    
    print(f"输入文件: {args.input_file}")
    print(f"置信水平: {args.confidence * 100:.0f}%")
    print(f"误差范围: {args.error * 100:.1f}%")
    
    sampled_df, population_size, sample_size = sample_csv(
        args.input_file,
        args.output,
        args.confidence,
        args.error,
        args.seed
    )
    
    output_file = args.output or f"{Path(args.input_file).stem}_sampled.csv"
    
    print(f"\n统计信息:")
    print(f"  总体大小 (N): {population_size}")
    print(f"  样本量 (n):   {sample_size}")
    print(f"  抽样比例:     {sample_size / population_size * 100:.1f}%")
    print(f"\n输出文件: {output_file}")


if __name__ == "__main__":
    main()
