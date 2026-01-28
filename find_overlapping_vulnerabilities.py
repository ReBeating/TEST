import csv

# 读取 inputs/1day_vul_list.csv
vul_1day = {}
with open('inputs/1day_vul_list.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        vul_id = row['vul_id']
        vul_1day[vul_id] = {
            'repo': row['repo'],
            'fixed_commit_sha': row['fixed_commit_sha']
        }

# 读取 category_2_only_fp.csv
vul_category2 = {}
with open('category_2_only_fp.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        vul_id = row['vul_id']
        vul_category2[vul_id] = {
            'repo': row['repo'],
            'fixed_commit_sha': row['fixed_commit_sha']
        }

# 找出重合的漏洞ID
overlapping_vul_ids = set(vul_1day.keys()) & set(vul_category2.keys())

print(f"找到 {len(overlapping_vul_ids)} 个重合的漏洞")

# 生成结果CSV
overlapping_vulnerabilities = []
for vul_id in overlapping_vul_ids:
    # 使用1day_vul_list中的数据
    overlapping_vulnerabilities.append({
        'vul_id': vul_id,
        'repo': vul_1day[vul_id]['repo'],
        'fixed_commit_sha': vul_1day[vul_id]['fixed_commit_sha']
    })

# 按照repo和vul_id排序
overlapping_vulnerabilities.sort(key=lambda x: (x['repo'], x['vul_id']))

# 写入CSV文件
with open('overlapping_vulnerabilities.csv', 'w', encoding='utf-8', newline='') as f:
    fieldnames = ['vul_id', 'repo', 'fixed_commit_sha']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(overlapping_vulnerabilities)

print(f"结果已保存到 overlapping_vulnerabilities.csv")
print(f"\n重合的漏洞ID列表：")
for vul_id in sorted(overlapping_vul_ids):
    print(f"  - {vul_id}")
