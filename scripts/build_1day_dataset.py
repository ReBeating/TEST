#!/usr/bin/env python3
"""
整合版漏洞数据集构建脚本

将漏洞数据整合、清洗、构建数据集三个阶段合并为一个脚本。

输入:
    - inputs/linux_kernel_cves/data/kernel_cves.json
    - inputs/linux_kernel_cves/data/stream_fixes.json
    - inputs/Dataset.json

输出:
    - outputs/1day_vul_dataset.json
"""

import os
import sys
import re
import json
import difflib
from collections import OrderedDict
from git import Repo
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from core.configs import REPO_DIR_PATH
# 添加 src 目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.parser import CtagsParser

# ============================================================
# 配置
# ============================================================

KERNEL_CVES_PATH = 'inputs/linux_kernel_cves/data/kernel_cves.json'
STREAM_FIXES_PATH = 'inputs/linux_kernel_cves/data/stream_fixes.json'
DATASET_PATH = 'inputs/Dataset.json'
OUTPUT_PATH = 'outputs/1day_vul_dataset.json'
OUTPUT_DIR = 'outputs'

# Repo映射：short_name -> (full_name, path_in_repos_dir)
REPO_MAPS = {
    'openjpeg': ('uclouvain/openjpeg', 'uclouvain/openjpeg'),
    'FFmpeg': ('FFmpeg/FFmpeg', 'FFmpeg/FFmpeg'),
    'wireshark': ('wireshark/wireshark', 'wireshark/wireshark'),
    'curl': ('curl/curl', 'curl/curl'),
    'httpd': ('apache/httpd', 'apache/httpd'),
    'ImageMagick': ('ImageMagick/ImageMagick', 'ImageMagick/ImageMagick'),
    'linux': ('torvalds/linux', 'torvalds/linux'),
}

# 反向映射: full_name -> short_name
REPO_MAPS_REVERSE = {v[0]: k for k, v in REPO_MAPS.items()}

# 需要排除的repo
EXCLUDED_REPOS = ['qemu/qemu', 'openssl/openssl', 'qemu', 'openssl']

# 版本号正则模式
VERSION_PATTERNS = [
    r'^v?\d+\.\d+(\.\d+)?(\.\d+)?$',
    r'^v?\d+\.\d+\.\d+[a-z]?$',
    r'^n\d+\.\d+(\.\d+)?$',
    r'^version\.\d+(\.\d+)*$',
    r'^release[-_]?\d+\.\d+(\.\d+)?$',
    r'^[\w]+-\d+\.\d+(\.\d+)?$',
    r'^curl-\d+_\d+(_\d+)?$',
    r'^\d+\.\d+\.\d+-\d+$',
    r'^\d+\.\d+-\d+$',
]

# 排除的tag模式
EXCLUDE_PATTERNS = [
    r'-dev$', r'-alpha\d*', r'-beta\d*', r'-pre\d*', r'_pre\d*',
    r'pre\d+$', r'-rc\d*$', r'_rc\d*$', r'\.rc\d*$', r'-test',
    r'-snapshot', r'-preview', r'-nightly', r'_alpha', r'_beta',
    r'^ethereal-', r'^ffmpeg-', r'^candidate-',
]


# ============================================================
# 通用工具函数
# ============================================================
def read_json(file_path):
    """读取JSON文件"""
    file_path = os.path.abspath(file_path)
    with open(file_path, 'rt', encoding='utf-8') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, file_path):
    """写入JSON文件"""
    dir_path = os.path.dirname(file_path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path)
    file_path = os.path.abspath(file_path)
    with open(file_path, 'wt', encoding='utf-8') as handle:
        json.dump(content, handle, indent=4)


# ============================================================
# 阶段1：数据整合
# ============================================================
def get_tag_dict(repo: Repo, repo_name: str = None):
    """获取repo的所有tag及其时间，返回按时间排序的字典"""
    tags = repo.git.tag('--sort=-creatordate').splitlines()
    
    # Linux内核特殊处理
    if repo_name == 'linux' or (repo.working_tree_dir and repo.working_tree_dir.endswith('linux')):
        new_tags = []
        for tag in tags:
            if '-' in tag:
                continue
            items = tag.split('.')
            if len(items) == 2:
                new_tags.append(tag)
            elif len(items) == 3:
                if items[0] == 'v2':
                    new_tags.append(tag)
        tags = new_tags
    
    # 过滤掉rc版本
    tags = [tag for tag in tags if 'rc' not in tag.lower()]
    
    tag_time_dict = {}
    for tag in tags:
        try:
            tag_time = repo.commit(tag).committed_datetime
            tag_time_dict[tag] = tag_time
        except:
            pass
    
    sorted_tags = dict(sorted(tag_time_dict.items(), key=lambda x: x[1], reverse=False))
    return sorted_tags


def get_all_repo_tags():
    """获取所有repo的tag信息"""
    repo_tags = {}
    
    for repo_short, (repo_full, repo_rel_path) in REPO_MAPS.items():
        if repo_full in EXCLUDED_REPOS or repo_short in EXCLUDED_REPOS:
            continue
            
        repo_path = os.path.join(REPO_DIR_PATH, repo_rel_path)
        if not os.path.exists(repo_path):
            print(f"  Warning: repo {repo_short} not found at {repo_path}")
            continue
        
        try:
            repo = Repo(repo_path)
            tag_dict = get_tag_dict(repo, repo_short)
            repo_tags[repo_full] = list(tag_dict.keys())
            print(f"  {repo_full}: {len(tag_dict)} tags")
        except Exception as e:
            print(f"  Error processing {repo_short}: {e}")
    
    return repo_tags


def analyze_linux_cve(args):
    """分析单个Linux CVE"""
    cve_id, kernel_cves_info, stream_fixes_dict, tag_dict, repo_path = args
    
    try:
        if not any(year in cve_id for year in ('-2020-', '-2021-', '-2022-', '-2023-', '-2024-')):
            return None
        
        repo = Repo(repo_path)
        stream_fixes = stream_fixes_dict.get(cve_id, {})
        fixes = kernel_cves_info[cve_id].get('fixes', '')
        breaks = kernel_cves_info[cve_id].get('breaks', '')
        cwe = kernel_cves_info[cve_id].get('cwe', '')
        
        if not fixes:
            return None
        
        fixing_commits = [stream_fixes[v]['cmt_id'] for v in stream_fixes] + [fixes]
        
        try:
            fix_commit = repo.commit(fixes)
            fix_time = fix_commit.committed_datetime
        except:
            return None
        
        affected_versions = []
        pre_versions = []
        fix_versions = []
        
        contain_fix_tags = repo.git.tag('--contains', fixes).splitlines()
        
        if breaks:
            try:
                break_commit = repo.commit(breaks)
                break_time = break_commit.committed_datetime
                contain_break_tags = repo.git.tag('--contains', breaks).splitlines()
                
                for tag, tag_time in tag_dict.items():
                    if tag_time < break_time:
                        pre_versions.append(tag)
                    elif break_time <= tag_time < fix_time and tag in contain_break_tags:
                        affected_versions.append(tag)
                    elif tag_time >= fix_time and tag in contain_fix_tags:
                        fix_versions.append(tag)
            except:
                breaks = None
                for tag, tag_time in tag_dict.items():
                    if tag_time >= fix_time and tag in contain_fix_tags:
                        fix_versions.append(tag)
        else:
            for tag, tag_time in tag_dict.items():
                if tag_time >= fix_time and tag in contain_fix_tags:
                    fix_versions.append(tag)
        
        if fixing_commits:
            return (cve_id, {
                'repo': 'torvalds/linux',
                'fixed_commit_sha': fixes,
                'breaks_commit_sha': breaks if breaks else None,
                'pre_versions': pre_versions,
                'affected_versions': affected_versions,
                'fix_versions': fix_versions,
                'fixing_commits': fixing_commits,
                'cwe': [cwe] if cwe else [],
            })
    except:
        pass
    return None


def get_linux_vulnerabilities():
    """获取Linux内核漏洞数据"""
    repo_path = os.path.join(REPO_DIR_PATH, 'torvalds/linux')
    
    if not os.path.exists(repo_path):
        print(f"  Warning: Linux repo not found at {repo_path}")
        return {}
    
    repo = Repo(repo_path)
    kernel_cves_info = read_json(KERNEL_CVES_PATH)
    stream_fixes_dict = read_json(STREAM_FIXES_PATH)
    tag_dict = get_tag_dict(repo, 'linux')
    
    linux_vuls = {}
    
    with ProcessPoolExecutor(max_workers=32) as executor:
        futures = [
            executor.submit(analyze_linux_cve,
                (cve_id, kernel_cves_info, stream_fixes_dict, tag_dict, repo_path))
            for cve_id in kernel_cves_info
        ]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="  Linux CVEs"):
            result = future.result()
            if result:
                cve_id, data = result
                linux_vuls[cve_id] = data
    
    return linux_vuls


def analyze_tpl_versions(args):
    """分析第三方库漏洞的版本信息"""
    cve_id, vul_data, repo_path = args
    
    try:
        repo = Repo(repo_path)
        fixed_commit_sha = vul_data.get('fixed_commit_sha')
        
        if not fixed_commit_sha:
            return cve_id, vul_data
        
        tag_dict = get_tag_dict(repo)
        
        try:
            fix_commit = repo.commit(fixed_commit_sha)
            fix_time = fix_commit.committed_datetime
        except:
            return cve_id, vul_data
        
        contain_fix_tags = repo.git.tag('--contains', fixed_commit_sha).splitlines()
        
        pre_versions = []
        fix_versions = []
        affected_versions = vul_data.get('affected_versions', [])
        
        earliest_affected_time = None
        if affected_versions:
            for tag in affected_versions:
                if tag in tag_dict:
                    if earliest_affected_time is None or tag_dict[tag] < earliest_affected_time:
                        earliest_affected_time = tag_dict[tag]
        
        for tag, tag_time in tag_dict.items():
            if earliest_affected_time and tag_time < earliest_affected_time and tag not in affected_versions:
                pre_versions.append(tag)
            elif tag_time >= fix_time and tag in contain_fix_tags:
                fix_versions.append(tag)
        
        vul_data['pre_versions'] = pre_versions
        vul_data['fix_versions'] = fix_versions
        
        return cve_id, vul_data
    except:
        return cve_id, vul_data


def get_dataset_vulnerabilities():
    """获取Dataset.json中的漏洞数据（非Linux）"""
    dataset = read_json(DATASET_PATH)
    dataset_vuls = {}
    
    for cve_id, vul_data in dataset.items():
        repo_short = vul_data.get('repo', '')
        
        if repo_short == 'linux' or repo_short in EXCLUDED_REPOS:
            continue
        
        if repo_short in REPO_MAPS:
            repo_full = REPO_MAPS[repo_short][0]
        else:
            continue
        
        if repo_full in EXCLUDED_REPOS:
            continue
        
        affected_versions = vul_data.get('affected_version', [])
        fixing_commits = vul_data.get('fixing_commits', [[]])
        
        if fixing_commits and fixing_commits[0]:
            fixed_commit_sha = fixing_commits[0][-1] if isinstance(fixing_commits[0], list) else fixing_commits[-1]
        else:
            fixed_commit_sha = None
        
        cwe = vul_data.get('CWE', [])
        
        dataset_vuls[cve_id] = {
            'repo': repo_full,
            'fixed_commit_sha': fixed_commit_sha,
            'breaks_commit_sha': None,
            'pre_versions': [],
            'affected_versions': affected_versions,
            'fix_versions': [],
            'fixing_commits': fixing_commits[0] if fixing_commits else [],
            'cwe': cwe,
        }
    
    return dataset_vuls


def enrich_dataset_vulnerabilities(dataset_vuls):
    """为Dataset漏洞补充版本信息"""
    repo_vuls = {}
    for cve_id, vul_data in dataset_vuls.items():
        repo_full = vul_data['repo']
        repo_short = REPO_MAPS_REVERSE.get(repo_full, repo_full)
        if repo_short not in repo_vuls:
            repo_vuls[repo_short] = []
        repo_vuls[repo_short].append((cve_id, vul_data))
    
    enriched_vuls = {}
    
    for repo_short, vuls in repo_vuls.items():
        if repo_short in REPO_MAPS:
            _, repo_rel_path = REPO_MAPS[repo_short]
        else:
            repo_rel_path = repo_short
        repo_path = os.path.join(REPO_DIR_PATH, repo_rel_path)
        
        if not os.path.exists(repo_path):
            for cve_id, vul_data in vuls:
                enriched_vuls[cve_id] = vul_data
            continue
        
        args_list = [(cve_id, vul_data.copy(), repo_path) for cve_id, vul_data in vuls]
        
        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(analyze_tpl_versions, args) for args in args_list]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"  {repo_short}"):
                cve_id, data = future.result()
                enriched_vuls[cve_id] = data
    
    return enriched_vuls


# ============================================================
# 阶段2：数据清洗
# ============================================================
def is_valid_version_tag(tag):
    """检查tag是否是有效的版本号"""
    tag_lower = tag.lower()
    
    for pattern in EXCLUDE_PATTERNS:
        if re.search(pattern, tag_lower):
            return False
    
    for pattern in VERSION_PATTERNS:
        if re.match(pattern, tag, re.IGNORECASE):
            return True
    
    return False


def normalize_tag(tag):
    """标准化tag，提取核心版本号用于去重比较"""
    prefixes = ['wireshark-', 'httpd-', 'curl-', 'ffmpeg-', 'version.', 'release-', 'release_', 'v', 'n']
    normalized = tag.lower()
    for prefix in prefixes:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
            break
    normalized = normalized.replace('_', '.')
    return normalized


def deduplicate_tags(tags):
    """去重tag列表"""
    if not tags:
        return []
    
    version_groups = {}
    for tag in tags:
        normalized = normalize_tag(tag)
        if normalized not in version_groups:
            version_groups[normalized] = []
        version_groups[normalized].append(tag)
    
    result = []
    for normalized, group in version_groups.items():
        if len(group) == 1:
            result.append(group[0])
        else:
            def score_tag(t):
                score = len(t)
                if t.startswith('v') or t.startswith('n'):
                    score -= 100
                if t.startswith('version.'):
                    score -= 50
                return score
            best_tag = min(group, key=score_tag)
            result.append(best_tag)
    
    return result


def clean_tags(tags):
    """清洗tag列表"""
    if not tags:
        return []
    valid_tags = [tag for tag in tags if is_valid_version_tag(tag)]
    deduped_tags = deduplicate_tags(valid_tags)
    return deduped_tags


def clean_repo_tags(repo_tags):
    """清洗repo_tags"""
    cleaned = {}
    
    for repo, tags in repo_tags.items():
        if repo in EXCLUDED_REPOS:
            continue
        cleaned_tags = clean_tags(tags)
        cleaned[repo] = cleaned_tags
    
    return cleaned


def clean_vulnerabilities(vulnerabilities, cleaned_repo_tags):
    """清洗漏洞数据"""
    cleaned = {}
    
    for cve_id, vul_data in vulnerabilities.items():
        repo = vul_data.get('repo', '')
        
        if repo in EXCLUDED_REPOS:
            continue
        
        valid_tags_set = set(cleaned_repo_tags.get(repo, []))
        
        pre_versions = vul_data.get('pre_versions', [])
        affected_versions = vul_data.get('affected_versions', [])
        fix_versions = vul_data.get('fix_versions', [])
        
        cleaned_pre = [v for v in pre_versions if v in valid_tags_set] if valid_tags_set else pre_versions
        cleaned_affected = [v for v in affected_versions if v in valid_tags_set] if valid_tags_set else affected_versions
        cleaned_fix = [v for v in fix_versions if v in valid_tags_set] if valid_tags_set else fix_versions
        
        cleaned[cve_id] = {
            'repo': repo,
            'fixed_commit_sha': vul_data.get('fixed_commit_sha'),
            'breaks_commit_sha': vul_data.get('breaks_commit_sha'),
            'pre_versions': cleaned_pre,
            'affected_versions': cleaned_affected,
            'fix_versions': cleaned_fix,
            'fixing_commits': vul_data.get('fixing_commits', []),
            'cwe': vul_data.get('cwe', []),
        }
    
    return cleaned


# ============================================================
# 阶段3：构建数据集
# ============================================================
def parse_functions(file_content, suffix='.c'):
    """使用 CtagsParser 解析函数"""
    if not file_content:
        return {}
    try:
        result = CtagsParser.parse_code(file_content, suffix=suffix)
        return result.get('functions', {})
    except:
        return {}


def get_modified_functions(repo, commit_hash):
    """获取commit修改的函数"""
    try:
        commit = repo.commit(commit_hash)
        parent = commit.parents[0] if commit.parents else None
        
        if not parent:
            return {}
        
        modified_funcs = {}
        diffs = parent.diff(commit)
        
        for diff in diffs.iter_change_type('M'):
            file_path = diff.a_path
            
            if not file_path.endswith(('.c', '.h', '.cpp', '.cc', '.cxx', '.hpp')):
                continue
            
            suffix = '.c' if file_path.endswith(('.c', '.h')) else '.cpp'
            
            try:
                old_content = diff.a_blob.data_stream.read().decode('utf-8', errors='replace')
                new_content = diff.b_blob.data_stream.read().decode('utf-8', errors='replace')
            except:
                continue
            
            old_funcs = parse_functions(old_content, suffix)
            new_funcs = parse_functions(new_content, suffix)
            
            file_modified_funcs = {}
            for func_name in set(old_funcs.keys()) | set(new_funcs.keys()):
                old_code = old_funcs.get(func_name, {}).get('code', '')
                new_code = new_funcs.get(func_name, {}).get('code', '')
                
                if old_code != new_code and old_code and new_code:
                    diff_lines = list(difflib.unified_diff(
                        old_code.splitlines(keepends=True),
                        new_code.splitlines(keepends=True),
                        fromfile=f'a/{file_path}:{func_name}',
                        tofile=f'b/{file_path}:{func_name}',
                        n=3
                    ))
                    diff_code = ''.join(diff_lines)
                    
                    file_modified_funcs[func_name] = {
                        'pre_patch_code': old_code,
                        'post_patch_code': new_code,
                        'diff_code': diff_code
                    }
            
            if file_modified_funcs:
                modified_funcs[file_path] = file_modified_funcs
        
        return modified_funcs
    except:
        return {}


def get_function_code_at_version(repo, version, file_path, func_name):
    """获取指定版本下指定函数的代码"""
    try:
        commit = repo.commit(version)
        file_content = (commit.tree / file_path).data_stream.read().decode('utf-8', errors='replace')
        suffix = '.c' if file_path.endswith(('.c', '.h')) else '.cpp'
        funcs = parse_functions(file_content, suffix)
        return funcs.get(func_name, {}).get('code', '')
    except:
        return ''


def get_version_code(repo, version, modifications):
    """获取指定版本的所有函数代码"""
    version_funcs = {}
    all_code = ""
    
    for file_path, funcs in modifications.items():
        file_funcs = {}
        for func_name in funcs:
            code = get_function_code_at_version(repo, version, file_path, func_name)
            if code:
                file_funcs[func_name] = code
                all_code += code
        
        if file_funcs:
            version_funcs[file_path] = file_funcs
    
    return version_funcs, all_code


def process_single_vul(args):
    """处理单个漏洞"""
    cve_id, vul_data, repo_path = args
    
    FAIL_NO_FIXED_COMMIT = 'no_fixed_commit'
    FAIL_CTAGS_PARSE = 'ctags_parse_failed'
    FAIL_NO_BREAKS_INTERSECTION = 'no_breaks_intersection'
    FAIL_NOT_ENOUGH_VERSIONS = 'not_enough_versions'
    FAIL_EXCEPTION = 'exception'
    
    try:
        repo = Repo(repo_path)
        
        repo_name = vul_data['repo']
        fixed_commit_sha = vul_data.get('fixed_commit_sha')
        breaks_commit_sha = vul_data.get('breaks_commit_sha')
        pre_versions = vul_data.get('pre_versions', [])
        affected_versions = vul_data.get('affected_versions', [])
        fix_versions = vul_data.get('fix_versions', [])
        fixing_commits = vul_data.get('fixing_commits', [])
        
        if not fixed_commit_sha:
            return None, FAIL_NO_FIXED_COMMIT, cve_id
        
        # 1. 获取 fixed commit 修改的函数
        modifications = get_modified_functions(repo, fixed_commit_sha)
        
        if not modifications:
            return None, FAIL_CTAGS_PARSE, cve_id
        
        # 2. 如果有 breaks commit，检查函数交集
        if breaks_commit_sha:
            try:
                breaks_modifications = get_modified_functions(repo, breaks_commit_sha)
                
                fixed_funcs = set()
                for file_path, funcs in modifications.items():
                    for func_name in funcs:
                        fixed_funcs.add((file_path, func_name))
                
                breaks_funcs = set()
                for file_path, funcs in breaks_modifications.items():
                    for func_name in funcs:
                        breaks_funcs.add((file_path, func_name))
                
                if breaks_funcs and not (fixed_funcs & breaks_funcs):
                    return None, FAIL_NO_BREAKS_INTERSECTION, cve_id
            except:
                pass
        
        # 3. 收集所有版本的代码并去重
        all_versions_code = {'pre': {}, 'vul': {}, 'fix': {}}
        
        for version in pre_versions:
            version_funcs, all_code = get_version_code(repo, version, modifications)
            if version_funcs and all_code:
                all_versions_code['pre'][version] = (version_funcs, all_code)
        
        for version in affected_versions:
            version_funcs, all_code = get_version_code(repo, version, modifications)
            if version_funcs and all_code:
                all_versions_code['vul'][version] = (version_funcs, all_code)
        
        for version in fix_versions:
            version_funcs, all_code = get_version_code(repo, version, modifications)
            if version_funcs and all_code:
                all_versions_code['fix'][version] = (version_funcs, all_code)
        
        # 4. 去重 - 跨所有版本类型去重
        seen_codes = set()
        unique_versions = {'pre': {}, 'vul': {}, 'fix': {}}
        
        for ver_type in ['pre', 'vul', 'fix']:
            for version, (version_funcs, all_code) in all_versions_code[ver_type].items():
                if all_code not in seen_codes:
                    seen_codes.add(all_code)
                    unique_versions[ver_type][version] = version_funcs
        
        # 基于去重后的结果判断是否有pre版本
        has_pre = len(unique_versions['pre']) > 0
        
        # 5. 选择版本
        # 策略：
        # - pre: 选最后一个（最接近漏洞引入）
        # - vul: 选首个[0]和中间[len//2]（远离补丁，内部有间隔）
        # - fix: 选中间[len//2]和末尾[-1]（远离补丁，内部有间隔）
        versions_data = {'pre': {}, 'vul': {}, 'fix': {}}
        
        # vul版本选择
        vul_list = list(unique_versions['vul'].keys())
        if len(vul_list) >= 2:
            versions_data['vul'][vul_list[0]] = unique_versions['vul'][vul_list[0]]
            mid_idx = len(vul_list) // 2
            versions_data['vul'][vul_list[mid_idx]] = unique_versions['vul'][vul_list[mid_idx]]
        elif len(vul_list) == 1:
            versions_data['vul'][vul_list[0]] = unique_versions['vul'][vul_list[0]]
        
        if has_pre:
            # 有pre: 选 1/2/1
            pre_list = list(unique_versions['pre'].keys())
            if pre_list:
                versions_data['pre'][pre_list[-1]] = unique_versions['pre'][pre_list[-1]]
            
            fix_list = list(unique_versions['fix'].keys())
            if fix_list:
                versions_data['fix'][fix_list[-1]] = unique_versions['fix'][fix_list[-1]]
        else:
            # 无pre: 选 0/2/2
            fix_list = list(unique_versions['fix'].keys())
            if len(fix_list) >= 2:
                mid_idx = len(fix_list) // 2
                last_idx = len(fix_list) - 1
                if mid_idx == last_idx:
                    versions_data['fix'][fix_list[0]] = unique_versions['fix'][fix_list[0]]
                    versions_data['fix'][fix_list[-1]] = unique_versions['fix'][fix_list[-1]]
                else:
                    versions_data['fix'][fix_list[mid_idx]] = unique_versions['fix'][fix_list[mid_idx]]
                    versions_data['fix'][fix_list[-1]] = unique_versions['fix'][fix_list[-1]]
            elif len(fix_list) == 1:
                versions_data['fix'][fix_list[0]] = unique_versions['fix'][fix_list[0]]
        
        # 6. 验证版本数量
        if has_pre:
            if len(versions_data['vul']) < 2 or len(versions_data['fix']) < 1:
                return None, FAIL_NOT_ENOUGH_VERSIONS, cve_id
        else:
            if len(versions_data['vul']) < 2 or len(versions_data['fix']) < 2:
                return None, FAIL_NOT_ENOUGH_VERSIONS, cve_id
        
        result = {
            'repo': repo_name,
            'vul_id': cve_id,
            'affected_versions': affected_versions,
            'fixing_commits': fixing_commits,
            'modifications': modifications,
            'versions': versions_data
        }
        
        return (cve_id, result), None, cve_id
    
    except Exception as e:
        return None, FAIL_EXCEPTION, cve_id


def build_dataset(vulnerabilities):
    """构建最终数据集"""
    # 按repo分组
    repo_vuls = {}
    for cve_id, vul_data in vulnerabilities.items():
        repo_name = vul_data.get('repo', '')
        if repo_name not in repo_vuls:
            repo_vuls[repo_name] = []
        repo_vuls[repo_name].append((cve_id, vul_data))
    
    all_results = {}
    stats = {
        'no_fixed_commit': [],
        'ctags_parse_failed': [],
        'no_breaks_intersection': [],
        'not_enough_versions': [],
        'exception': [],
        'success': []
    }
    
    for repo_name, vuls in repo_vuls.items():
        print(f"\n  处理 {repo_name} ({len(vuls)} 个漏洞)...")
        
        repo_short = REPO_MAPS_REVERSE.get(repo_name, repo_name)
        if repo_short in REPO_MAPS:
            _, repo_rel_path = REPO_MAPS[repo_short]
        else:
            repo_rel_path = repo_name
        
        repo_path = os.path.join(REPO_DIR_PATH, repo_rel_path)
        if not os.path.exists(repo_path):
            print(f"    跳过: repo路径不存在 {repo_path}")
            continue
        
        args_list = [(cve_id, vul_data, repo_path) for cve_id, vul_data in vuls]
        
        with ProcessPoolExecutor(max_workers=32) as executor:
            futures = [executor.submit(process_single_vul, args) for args in args_list]
            
            success_count = 0
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"    {repo_name}"):
                result = future.result()
                if result:
                    success_data, fail_reason, cve_id = result
                    if success_data:
                        cve_id, data = success_data
                        all_results[cve_id] = data
                        success_count += 1
                        stats['success'].append(cve_id)
                    elif fail_reason:
                        stats[fail_reason].append(cve_id)
            
            print(f"    成功: {success_count}/{len(vuls)}")
    
    return all_results, stats


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 70)
    print("漏洞数据集构建脚本 (整合版)")
    print("=" * 70)
    
    # ========== 阶段1：数据整合 ==========
    print("\n" + "=" * 70)
    print("[阶段1/3] 整合漏洞数据")
    print("=" * 70)
    
    print("\n获取repo tags...")
    repo_tags = get_all_repo_tags()
    
    print("\n处理Linux内核漏洞...")
    linux_vuls = get_linux_vulnerabilities()
    print(f"  Linux漏洞数量: {len(linux_vuls)}")
    
    print("\n处理其他项目漏洞...")
    dataset_vuls = get_dataset_vulnerabilities()
    print(f"  其他漏洞数量: {len(dataset_vuls)}")
    
    print("\n补充版本信息...")
    enriched_dataset_vuls = enrich_dataset_vulnerabilities(dataset_vuls)
    
    all_vuls = {}
    all_vuls.update(linux_vuls)
    all_vuls.update(enriched_dataset_vuls)
    print(f"\n整合后总漏洞数: {len(all_vuls)}")
    
    # ========== 阶段2：数据清洗 ==========
    print("\n" + "=" * 70)
    print("[阶段2/3] 清洗数据")
    print("=" * 70)
    
    print("\n清洗repo tags...")
    cleaned_repo_tags = clean_repo_tags(repo_tags)
    for repo, tags in cleaned_repo_tags.items():
        orig_count = len(repo_tags.get(repo, []))
        print(f"  {repo}: {orig_count} -> {len(tags)} tags")
    
    print("\n清洗漏洞数据...")
    cleaned_vuls = clean_vulnerabilities(all_vuls, cleaned_repo_tags)
    print(f"  清洗前: {len(all_vuls)}, 清洗后: {len(cleaned_vuls)}")
    
    # ========== 阶段3：构建数据集 ==========
    print("\n" + "=" * 70)
    print("[阶段3/3] 构建数据集")
    print("=" * 70)
    
    dataset, stats = build_dataset(cleaned_vuls)
    
    # 排序
    dataset = dict(sorted(dataset.items(), key=lambda x: x[0]))
    
    # 打印统计
    print("\n" + "=" * 70)
    print("统计信息")
    print("=" * 70)
    print(f"成功: {len(stats['success'])}")
    print(f"无fixed_commit: {len(stats['no_fixed_commit'])}")
    print(f"ctags解析失败: {len(stats['ctags_parse_failed'])}")
    print(f"有breaks但无函数交集: {len(stats['no_breaks_intersection'])}")
    print(f"版本不足: {len(stats['not_enough_versions'])}")
    print(f"异常: {len(stats['exception'])}")
    
    # 保存结果
    print("\n保存数据集...")
    write_json(dataset, OUTPUT_PATH)
    
    stats_output = os.path.join(OUTPUT_DIR, 'build_stats.json')
    write_json(stats, stats_output)
    
    print("\n" + "=" * 70)
    print("完成!")
    print("=" * 70)
    print(f"输出文件: {OUTPUT_PATH}")
    print(f"统计文件: {stats_output}")
    print(f"成功处理漏洞数: {len(dataset)}")
    print("=" * 70)


if __name__ == '__main__':
    main()
