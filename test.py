#!/usr/bin/env python3
"""
从inputs/1day_vul_list.csv中筛掉不在merged_vulnerabilities.csv里面的漏洞
即：保留那些在merged_vulnerabilities.csv中出现的漏洞
"""
import pandas as pd
def filter_1day_vulnerabilities():
    # 读取1day_vul_list.csv
    print("读取 inputs/1day_vul_list.csv...")
    df_1day = pd.read_csv('inputs/1day_vul_list.csv')
    print(f"  - 1day漏洞总数: {len(df_1day)}")
    
    # 读取merged_vulnerabilities.csv
    print("\n读取 merged_vulnerabilities.csv...")
    df_merged = pd.read_csv('inputs/merged_vulnerabilities.csv')
    print(f"  - 合并后漏洞总数: {len(df_merged)}")
    
    # 创建merged中的漏洞标识集合（repo + vul_id）
    merged_set = set(
        zip(df_merged['repo'], df_merged['vul_id'])
    )
    print(f"  - 唯一漏洞标识数: {len(merged_set)}")
    
    # 筛选1day中存在于merged的漏洞
    print("\n筛选漏洞...")
    mask = df_1day.apply(
        lambda row: (row['repo'], row['vul_id']) in merged_set,
        axis=1
    )
    df_filtered = df_1day[mask]
    
    removed_count = len(df_1day) - len(df_filtered)
    print(f"  - 保留的漏洞数: {len(df_filtered)}")
    print(f"  - 移除的漏洞数: {removed_count}")
    
    # 保存筛选后的结果
    output_file = 'filtered_1day_vul_list.csv'
    df_filtered.to_csv(output_file, index=False)
    print(f"\n结果已保存到: {output_file}")
    
    # 显示统计信息
    if len(df_filtered) > 0:
        print("\n保留漏洞按仓库统计:")
        repo_counts = df_filtered['repo'].value_counts()
        for repo, count in repo_counts.head(10).items():
            print(f"  {repo}: {count}")
        if len(repo_counts) > 10:
            print(f"  ... 还有 {len(repo_counts) - 10} 个仓库")
    
    # 显示被移除的漏洞示例
    if removed_count > 0:
        df_removed = df_1day[~mask]
        print(f"\n被移除漏洞示例（前10条）:")
        for idx, row in df_removed.head(10).iterrows():
            print(f"  {row['repo']}/{row['vul_id']}")
        if removed_count > 10:
            print(f"  ... 还有 {removed_count - 10} 条被移除")
    
    return df_filtered
if __name__ == '__main__':
    df_result = filter_1day_vulnerabilities()
    print(f"\n完成！")