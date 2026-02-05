#!/usr/bin/env python3
"""统计 results/0day 下6个 *_0day_res.csv 按漏洞大类(four_type)的准确率、TP个数和总个数"""

import pandas as pd
from pathlib import Path

# 1. 读取漏洞大类映射
vul_list = pd.read_csv("inputs/merged_0day_vul_list.csv")
# 每个 vul_id 对应一个 four_type（去重）
vul_type_map = vul_list.drop_duplicates(subset=["vul_id"])[["vul_id", "four_type"]]
print(f"漏洞列表共 {len(vul_type_map)} 个唯一漏洞")
print(f"漏洞大类分布:\n{vul_type_map['four_type'].value_counts().to_string()}\n")

# 2. 定义6个结果文件
result_files = {
    "FIRE":    "results/0day/fire_0day_res.csv",
    "Movery":  "results/0day/movery_0day_res.csv",
    "V1SCAN":  "results/0day/v1scan_0day_res.csv",
    "VERDICT": "results/0day/verdict_0day_res.csv",
    "VUDDY":   "results/0day/vuddy_0day_res.csv",
    "VulTURE": "results/0day/vulture_0day_sampled.csv",
}

# 3. 对每个工具统计
all_results = []

for tool_name, filepath in result_files.items():
    df = pd.read_csv(filepath, encoding="utf-8-sig")
    
    # 标准化 judgement 列
    df["judgement"] = df["judgement"].str.strip().str.upper()
    
    # 合并 four_type
    df = df.merge(vul_type_map, on="vul_id", how="left")
    
    # 按漏洞级别统计：每个 vul_id 只要有一行是 TP 就算 TP
    vul_level = df.groupby("vul_id").agg(
        is_tp=("judgement", lambda x: (x == "TP").any()),
        four_type=("four_type", "first")
    ).reset_index()
    
    # 按 four_type 分组统计
    for ft, group in vul_level.groupby("four_type"):
        total = len(group)
        tp_count = group["is_tp"].sum()
        precision = tp_count / total if total > 0 else 0
        all_results.append({
            "Tool": tool_name,
            "four_type": ft,
            "TP": int(tp_count),
            "Total": total,
            "Precision": f"{precision:.2%}"
        })
    
    # 总计行
    total_all = len(vul_level)
    tp_all = vul_level["is_tp"].sum()
    all_results.append({
        "Tool": tool_name,
        "four_type": "ALL",
        "TP": int(tp_all),
        "Total": total_all,
        "Precision": f"{tp_all/total_all:.2%}" if total_all > 0 else "N/A"
    })

# 4. 输出结果
results_df = pd.DataFrame(all_results)

# 按工具分组打印
print("=" * 80)
print("按漏洞大类(four_type)统计各工具的准确率 (漏洞级别: 至少一个检测为TP即视为TP)")
print("=" * 80)

for tool_name in result_files.keys():
    tool_df = results_df[results_df["Tool"] == tool_name]
    print(f"\n{'─' * 60}")
    print(f"  {tool_name}")
    print(f"{'─' * 60}")
    print(tool_df[["four_type", "TP", "Total", "Precision"]].to_string(index=False))

# 5. 生成汇总透视表
print("\n\n" + "=" * 80)
print("汇总透视表 - TP/Total (Precision)")
print("=" * 80)

# 构建透视表
pivot_data = []
four_types = sorted(vul_type_map["four_type"].dropna().unique())

for tool_name in result_files.keys():
    row = {"Tool": tool_name}
    tool_df = results_df[results_df["Tool"] == tool_name]
    for ft in four_types + ["ALL"]:
        ft_row = tool_df[tool_df["four_type"] == ft]
        if len(ft_row) > 0:
            tp = ft_row.iloc[0]["TP"]
            total = ft_row.iloc[0]["Total"]
            prec = ft_row.iloc[0]["Precision"]
            row[ft] = f"{tp}/{total} ({prec})"
        else:
            row[ft] = "0/0 (N/A)"
    pivot_data.append(row)

pivot_df = pd.DataFrame(pivot_data)
print(pivot_df.to_string(index=False))

# 6. 保存到 CSV
results_df.to_csv("results/0day/stats_by_four_type.csv", index=False)
print(f"\n结果已保存至 results/0day/stats_by_four_type.csv")
