#!/usr/bin/env python3
"""
给 inputs/merged_0day_vul_list.csv 加两列 vuln_type 和 four_type。
- vuln_type: 从 outputs/results/{repo}/{CVE}_features.json 中的 taxonomy.vuln_type 获取
- four_type: 从 results/vulnerability_taxonomy.json 中找到 vuln_type 对应的大类名称
"""

import csv
import json
import os

# 1. 加载 vulnerability_taxonomy.json，构建 subcategory -> category 映射
with open("results/vulnerability_taxonomy.json", "r") as f:
    taxonomy = json.load(f)

sub_to_category = {}
for cat in taxonomy["categories"]:
    for sub in cat["subcategories"]:
        sub_to_category[sub] = cat["name"]

print("Subcategory -> Category mapping:")
for sub, cat in sub_to_category.items():
    print(f"  {sub} -> {cat}")
print()

# 2. 读取 CSV
input_csv = "inputs/merged_0day_vul_list.csv"
output_csv = "inputs/merged_0day_vul_list.csv"  # 覆盖写入

with open(input_csv, "r") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"Total rows in CSV: {len(rows)}")

# 3. 为每行查找 vuln_type
found = 0
not_found = 0
not_found_list = []

for row in rows:
    repo = row["repo"]
    vul_id = row["vul_id"]
    
    # 构建 features.json 路径
    features_path = f"outputs/results/{repo}/{vul_id}_features.json"
    
    vuln_type = ""
    four_type = ""
    
    if os.path.exists(features_path):
        try:
            with open(features_path, "r") as f:
                data = json.load(f)
            
            # data 是一个 list，取第一个元素的 taxonomy
            if isinstance(data, list) and len(data) > 0:
                tax = data[0].get("taxonomy", {})
                vuln_type = tax.get("vuln_type", "")
            elif isinstance(data, dict):
                tax = data.get("taxonomy", {})
                vuln_type = tax.get("vuln_type", "")
            
            if vuln_type:
                four_type = sub_to_category.get(vuln_type, "")
                if not four_type:
                    print(f"  WARNING: vuln_type '{vuln_type}' not found in taxonomy for {vul_id}")
                found += 1
            else:
                not_found += 1
                not_found_list.append((repo, vul_id, "no vuln_type in taxonomy"))
        except Exception as e:
            not_found += 1
            not_found_list.append((repo, vul_id, str(e)))
    else:
        not_found += 1
        not_found_list.append((repo, vul_id, "features.json not found"))
    
    row["vuln_type"] = vuln_type
    row["four_type"] = four_type

print(f"\nFound vuln_type: {found}")
print(f"Not found: {not_found}")
if not_found_list:
    print("\nNot found details (first 20):")
    for repo, vul_id, reason in not_found_list[:20]:
        print(f"  {repo}/{vul_id}: {reason}")

# 4. 写回 CSV
fieldnames = list(rows[0].keys())
# 确保 vuln_type 和 four_type 在最后
for col in ["vuln_type", "four_type"]:
    if col not in fieldnames:
        fieldnames.append(col)

with open(output_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"\nDone! Written to {output_csv}")

# 统计 four_type 分布
from collections import Counter
four_type_counts = Counter(row["four_type"] for row in rows if row["four_type"])
vuln_type_counts = Counter(row["vuln_type"] for row in rows if row["vuln_type"])

print(f"\nfour_type distribution:")
for k, v in four_type_counts.most_common():
    print(f"  {k}: {v}")

print(f"\nvuln_type distribution:")
for k, v in vuln_type_counts.most_common():
    print(f"  {k}: {v}")
