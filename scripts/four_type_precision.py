#!/usr/bin/env python3
"""Statistics on line-level precision (Precision) of 6 tools under results/0day by 4 major categories, retaining 3 decimal places.

Vulnerability type (vuln_type) and four major categories (four_type) are obtained from inputs/merged_0day_vul_list.csv,
Matched by (vul_id, fixed_commit_sha).
"""

import pandas as pd

# 1. Read vulnerability list, get vuln_type and four_type
vul_list = pd.read_csv("inputs/merged_0day_vul_list.csv")
# Use vul_id as unique key (four_type is consistent for the same vul_id)
vul_lookup = vul_list.drop_duplicates(subset=["vul_id"])[
    ["vul_id", "vuln_type", "four_type"]
]
print(f"Vulnerability list has {len(vul_lookup)} unique vul_id records\n")

# 2. Define result files for 6 tools
result_files = {
    "FIRE":    "results/0day/fire_0day_res.csv",
    "Movery":  "results/0day/movery_0day_res.csv",
    "V1SCAN":  "results/0day/v1scan_0day_res.csv",
    "VERDICT": "results/0day/verdict_0day_res.csv",
    "VUDDY":   "results/0day/vuddy_0day_res.csv",
    "VulTURE": "results/0day/vulture_0day_sampled.csv",
}

# 3. 四大类排序
four_type_order = [
    "Access-Validity Violations",
    "Numeric-Domain Violations",
    "Resource-Lifecycle Violations",
    "Control-Logic Violations",
]

# 4. 对每个工具统计
all_results = []

for tool_name, filepath in result_files.items():
    df = pd.read_csv(filepath, encoding="utf-8-sig")
    df["judgement"] = df["judgement"].str.strip().str.upper()

    # 通过 vul_id 合并 four_type
    df = df.merge(vul_lookup[["vul_id", "four_type"]],
                  on="vul_id", how="left")

    unmapped = df[df["four_type"].isna()]
    if len(unmapped) > 0:
        print(f"[{tool_name}] 警告: {len(unmapped)} 行无法匹配到 four_type")

    # 按 four_type 统计行级准确率
    for ft in four_type_order:
        group = df[df["four_type"] == ft]
        total = len(group)
        tp = (group["judgement"] == "TP").sum()
        fp = (group["judgement"] == "FP").sum()
        precision = tp / total if total > 0 else 0
        all_results.append({
            "Tool": tool_name,
            "four_type": ft,
            "TP": tp,
            "FP": fp,
            "Total": total,
            "Precision": round(precision, 3),
        })

    # 总计行
    mapped = df[df["four_type"].notna()]
    total_all = len(mapped)
    tp_all = (mapped["judgement"] == "TP").sum()
    fp_all = (mapped["judgement"] == "FP").sum()
    precision_all = tp_all / total_all if total_all > 0 else 0
    all_results.append({
        "Tool": tool_name,
        "four_type": "ALL",
        "TP": tp_all,
        "FP": fp_all,
        "Total": total_all,
        "Precision": round(precision_all, 3),
    })

# 5. 输出结果
results_df = pd.DataFrame(all_results)

print("=" * 75)
print("0-day 按4大类行级准确率统计（每行一个检测结果）")
print("=" * 75)

for tool_name in result_files.keys():
    tool_df = results_df[results_df["Tool"] == tool_name]
    print(f"\n{'─' * 65}")
    print(f"  {tool_name}")
    print(f"{'─' * 65}")
    print(tool_df[["four_type", "TP", "FP", "Total", "Precision"]].to_string(index=False))

# 6. 汇总透视表
print("\n\n" + "=" * 75)
print("汇总透视表 — TP/Total (Precision)")
print("=" * 75)

pivot_rows = []
for tool_name in result_files.keys():
    row = {"Tool": tool_name}
    tool_df = results_df[results_df["Tool"] == tool_name]
    for ft in four_type_order + ["ALL"]:
        ft_row = tool_df[tool_df["four_type"] == ft]
        if len(ft_row) > 0:
            r = ft_row.iloc[0]
            row[ft] = f"{r['TP']}/{r['Total']} ({r['Precision']:.3f})"
        else:
            row[ft] = "0/0 (N/A)"
    pivot_rows.append(row)

pivot_df = pd.DataFrame(pivot_rows)
pd.set_option("display.max_colwidth", 30)
pd.set_option("display.width", 200)
print(pivot_df.to_string(index=False))

# 7. 保存
output_path = "results/0day/precision_by_four_type.csv"
results_df.to_csv(output_path, index=False)
print(f"\n结果已保存至 {output_path}")
