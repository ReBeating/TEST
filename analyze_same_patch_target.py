import csv

# 读取 validation_report_0day.csv
same_patch_target_rows = []

with open('validation_report_0day.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # 检查patch_file和target_file是否相同，以及patch_func和target_func是否相同
        if (row['patch_file'] == row['target_file'] and 
            row['patch_func'] == row['target_func']):
            same_patch_target_rows.append(row)

print(f"找到 {len(same_patch_target_rows)} 行patch_file/patch_func与target_file/target_func相同的记录")
print("\n详细列表：")
print("-" * 120)

# 按vul_id分组统计
vul_id_counts = {}
for row in same_patch_target_rows:
    vul_id = row['vul_id']
    if vul_id not in vul_id_counts:
        vul_id_counts[vul_id] = []
    vul_id_counts[vul_id].append(row)

# 输出详细信息
for vul_id in sorted(vul_id_counts.keys()):
    rows = vul_id_counts[vul_id]
    print(f"\n{vul_id} ({len(rows)} 条):")
    for row in rows:
        print(f"  - File: {row['patch_file']}")
        print(f"    Func: {row['patch_func']}")
        print(f"    Repo: {row['repo']}")
        print(f"    CWE: {row['CWE']}, Type: {row['vuln_type']}, Judgement: {row['judgement']}")
        print()

# 生成CSV文件
with open('same_patch_target_report.csv', 'w', encoding='utf-8', newline='') as f:
    fieldnames = ['vul_id', 'repo', 'fixed_commit_sha', 'patch_file', 'patch_func', 
                  'target_file', 'target_func', 'CWE', 'vuln_type', 'judgement']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(same_patch_target_rows)

print("-" * 120)
print(f"\n总计: {len(same_patch_target_rows)} 条记录")
print(f"涉及 {len(vul_id_counts)} 个不同的漏洞ID")
print(f"\n结果已保存到 same_patch_target_report.csv")

# 统计按repo的分布
repo_counts = {}
for row in same_patch_target_rows:
    repo = row['repo']
    repo_counts[repo] = repo_counts.get(repo, 0) + 1

print("\n按仓库统计：")
for repo in sorted(repo_counts.keys()):
    print(f"  {repo}: {repo_counts[repo]} 条")
