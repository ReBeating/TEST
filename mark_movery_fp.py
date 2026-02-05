#!/usr/bin/env python3
"""
标记 movery_0day_res_new.csv 中的 FP (False Positive)。

逻辑:
1. 从 inputs/merged_0day_vul_list.csv 获取每个 vul_id 对应的 repo 和 commit
2. 对每个 commit，使用 git diff 解析补丁修改了哪些 (file_path, func_name)
3. 如果 movery 结果中的 (target_file_path, target_func_name) 与补丁的某组 (file_path, func_name) 匹配，
   说明 movery 检测到的就是补丁本身修改过的函数 —— 标记为 FP
4. 其余留空
"""

import csv
import os
import json
import subprocess
import tempfile
from collections import defaultdict
from git import Repo

REPO_DIR_PATH = '/data/cyf/repos'

# movery 短名 → vul_list 全名
REPO_SHORT_TO_FULL = {
    'ImageMagick': 'ImageMagick/ImageMagick',
    'linux': 'torvalds/linux',
    'radare2': 'radareorg/radare2',
}


def run_ctags(target_path: str) -> str:
    """运行 ctags 并返回 JSON 输出"""
    cmd = [
        "ctags", "--fields=+ne", "--output-format=json", "--c-kinds=f",
        "-o", "-", target_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True,
                                encoding='utf-8', errors='replace')
        return result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def extract_function_names(code: str, suffix='.c') -> set:
    """从代码字符串中提取所有函数名"""
    if not code:
        return set()
    with tempfile.NamedTemporaryFile(mode="w+", suffix=suffix, delete=True) as tmp:
        tmp.write(code)
        tmp.flush()
        output = run_ctags(tmp.name)
    
    names = set()
    for line in output.splitlines():
        try:
            tag = json.loads(line)
            if tag.get('kind') == 'function':
                names.add(tag.get('name', ''))
        except json.JSONDecodeError:
            continue
    return names


def extract_functions_with_code(code: str, suffix='.c') -> dict:
    """从代码字符串中提取所有函数及其代码"""
    if not code:
        return {}
    lines = code.replace('\x0c', ' ').splitlines()
    
    with tempfile.NamedTemporaryFile(mode="w+", suffix=suffix, delete=True) as tmp:
        tmp.write(code)
        tmp.flush()
        output = run_ctags(tmp.name)
    
    funcs = {}
    for line in output.splitlines():
        try:
            tag = json.loads(line)
            if tag.get('kind') == 'function':
                name = tag.get('name', '')
                start = tag.get('line', 0)
                end = tag.get('end', start)
                idx_start = max(0, start - 1)
                idx_end = end
                func_code = "\n".join(lines[idx_start:idx_end])
                funcs[name] = func_code
        except json.JSONDecodeError:
            continue
    return funcs


def normalize_simple(code: str) -> str:
    """简单归一化：去空行、strip"""
    if not code:
        return ""
    return '\n'.join([line.strip() for line in code.split('\n') if line.strip()])


def get_patch_functions(repo_path: str, commit_hash: str) -> set:
    """
    获取一个 commit 修改了哪些 (file_path, func_name) 组合。
    参考 denoising.py 的逻辑：
    1. 获取 commit 修改的文件列表
    2. 对每个 C/C++ 文件，提取 old/new 版本的函数
    3. 比较函数代码，找出被修改的函数
    """
    try:
        repo = Repo(repo_path)
    except Exception as e:
        print(f"  [Error] Cannot open repo {repo_path}: {e}")
        return set()

    changed_funcs = set()
    
    try:
        prev_commit = f"{commit_hash}^"
        diff_index = repo.commit(prev_commit).diff(commit_hash)
        files = set()
        for d in diff_index:
            if d.b_path:
                files.add(d.b_path)
            if d.a_path:
                files.add(d.a_path)
    except Exception as e:
        print(f"  [Error] Git diff failed for {commit_hash}: {e}")
        return set()

    for file_path in files:
        if not file_path.endswith(('.c', '.h', '.cpp', '.cc', '.cxx', '.hpp')):
            continue
        
        suffix = '.c' if file_path.endswith(('.c', '.h')) else '.cpp'
        
        # 获取 old/new 文件内容
        try:
            old_content = (repo.commit(prev_commit).tree / file_path).data_stream.read().decode('utf-8', errors='replace')
        except:
            old_content = ""
        
        try:
            new_content = (repo.commit(commit_hash).tree / file_path).data_stream.read().decode('utf-8', errors='replace')
        except:
            new_content = ""
        
        if not old_content and not new_content:
            continue
        
        old_funcs = extract_functions_with_code(old_content, suffix)
        new_funcs = extract_functions_with_code(new_content, suffix)
        
        # 找出被修改的函数
        all_func_names = set(old_funcs.keys()) | set(new_funcs.keys())
        for func_name in all_func_names:
            old_code = old_funcs.get(func_name, "")
            new_code = new_funcs.get(func_name, "")
            
            norm_old = normalize_simple(old_code)
            norm_new = normalize_simple(new_code)
            
            if norm_old != norm_new:
                changed_funcs.add((file_path, func_name))
    
    return changed_funcs


def main():
    # 1. 读取 vul_list，建立 vul_id -> [(repo_full, commit)] 映射
    vul_commits = defaultdict(list)
    with open('inputs/merged_0day_vul_list.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            repo_full = row['repo']
            vul_id = row['vul_id']
            commit = row['fixed_commit_sha']
            vul_commits[vul_id].append((repo_full, commit))
    
    # 2. 读取 movery 结果，获取需要处理的 vul_id 集合
    movery_rows = []
    with open('results/movery_0day_res_new.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            movery_rows.append(row)
    
    needed_vul_ids = set(row['vul_id'] for row in movery_rows)
    print(f"Movery 结果共 {len(movery_rows)} 行, 涉及 {len(needed_vul_ids)} 个 vul_id")
    
    # 3. 对每个需要的 vul_id，解析补丁修改了哪些函数
    # 缓存: vul_id -> set of (file_path, func_name)
    patch_func_cache = {}
    
    for vul_id in sorted(needed_vul_ids):
        if vul_id not in vul_commits:
            print(f"  [Warn] {vul_id} not found in merged_0day_vul_list.csv")
            patch_func_cache[vul_id] = set()
            continue
        
        all_changed = set()
        for repo_full, commit in vul_commits[vul_id]:
            repo_path = os.path.join(REPO_DIR_PATH, repo_full)
            if not os.path.exists(repo_path):
                print(f"  [Warn] Repo not found: {repo_path}")
                continue
            
            print(f"  Processing {vul_id}: {repo_full} @ {commit[:12]}...")
            changed = get_patch_functions(repo_path, commit)
            all_changed.update(changed)
        
        patch_func_cache[vul_id] = all_changed
        if all_changed:
            print(f"    -> {len(all_changed)} changed functions found")
        else:
            print(f"    -> No changed functions found")
    
    # 4. 标记 FP
    fp_count = 0
    for row in movery_rows:
        vul_id = row['vul_id']
        target_file = row['target_file_path']
        target_func = row['target_func_name']
        
        patch_funcs = patch_func_cache.get(vul_id, set())
        
        # 检查是否匹配补丁修改的某个 (file_path, func_name)
        matched = False
        for (pf_path, pf_func) in patch_funcs:
            if pf_func == target_func and pf_path == target_file:
                matched = True
                break
        
        if matched:
            row['judgement'] = 'FP'
            fp_count += 1
        else:
            row['judgement'] = ''
    
    # 5. 写出结果
    output_path = 'results/movery_0day_res_new.csv'
    fieldnames = ['vul_id', 'repo', 'fixed_commit_sha', 'patch_file_path', 'patch_func_name',
                  'target_file_path', 'target_func_name', 'judgement']
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(movery_rows)
    
    print(f"\n=== 完成 ===")
    print(f"总行数: {len(movery_rows)}")
    print(f"标记为 FP: {fp_count}")
    print(f"未标记: {len(movery_rows) - fp_count}")
    print(f"结果已写入: {output_path}")


if __name__ == '__main__':
    main()
