#!/usr/bin/env python3
"""
从批量下载的NVD和OSV数据中提取CVE元数据（改进版）
修复：
1. 去除source/sources重复
2. CWE ID去重
3. 使用MITRE官方CWE名称
4. 处理OSV数据（API查询）
5. 简化输出格式
"""

import json
import csv
import os
import time
import xml.etree.ElementTree as ET
import requests
from pathlib import Path
from typing import Dict, Set, List

def parse_cwe_xml(cwe_xml_path: str) -> Dict[str, str]:
    """解析MITRE CWE XML文件，建立CWE ID到名称的映射"""
    print("="*70)
    print("解析MITRE CWE数据库")
    print("="*70)
    
    cwe_mapping = {}
    
    try:
        tree = ET.parse(cwe_xml_path)
        root = tree.getroot()
        
        # XML命名空间
        ns = {'cwe': 'http://cwe.mitre.org/cwe-7'}
        
        # 解析Weakness条目
        for weakness in root.findall('.//cwe:Weakness', ns):
            cwe_id = weakness.get('ID')
            cwe_name = weakness.get('Name')
            if cwe_id and cwe_name:
                cwe_mapping[f'CWE-{cwe_id}'] = cwe_name
        
        # 解析Category条目
        for category in root.findall('.//cwe:Category', ns):
            cwe_id = category.get('ID')
            cwe_name = category.get('Name')
            if cwe_id and cwe_name:
                cwe_mapping[f'CWE-{cwe_id}'] = cwe_name
        
        print(f"加载了 {len(cwe_mapping)} 个CWE定义")
        
    except Exception as e:
        print(f"警告: 无法解析CWE XML: {e}")
        print("将使用默认CWE映射")
        # 使用默认映射作为后备
        cwe_mapping = {
            'CWE-79': 'Cross-site Scripting (XSS)',
            'CWE-119': 'Improper Restriction of Operations within the Bounds of a Memory Buffer',
            'CWE-125': 'Out-of-bounds Read',
            'CWE-787': 'Out-of-bounds Write',
            'CWE-416': 'Use After Free',
            'CWE-20': 'Improper Input Validation',
            'CWE-200': 'Exposure of Sensitive Information to an Unauthorized Actor',
            'CWE-89': 'SQL Injection',
            'CWE-22': 'Improper Limitation of a Pathname to a Restricted Directory',
            'CWE-476': 'NULL Pointer Dereference',
            'CWE-190': 'Integer Overflow or Wraparound',
        }
    
    return cwe_mapping

def load_target_vulnerabilities(vul_list_path: str) -> Set[str]:
    """加载目标漏洞列表"""
    target_vulns = set()
    with open(vul_list_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            vul_id = row['vul_id']
            if vul_id.startswith('CVE-') or vul_id.startswith('OSV-'):
                target_vulns.add(vul_id)
    return target_vulns

def normalize_cwe_data(cwe_ids: List[str], cwe_mapping: Dict[str, str]) -> tuple:
    """
    标准化CWE数据：去重并获取官方名称
    返回: (unique_cwe_ids, cwe_names)
    """
    # 去重并保持顺序
    unique_ids = []
    seen = set()
    for cwe_id in cwe_ids:
        if cwe_id and cwe_id not in seen:
            unique_ids.append(cwe_id)
            seen.add(cwe_id)
    
    # 获取官方CWE名称
    cwe_names = []
    for cwe_id in unique_ids:
        name = cwe_mapping.get(cwe_id, f'Unknown CWE ({cwe_id})')
        cwe_names.append(name)
    
    return unique_ids, cwe_names

def parse_nvd_2_0_file(nvd_file: str, target_vulns: Set[str], cwe_mapping: Dict[str, str]) -> Dict:
    """解析NVD 2.0格式的JSON文件"""
    print(f"解析 {os.path.basename(nvd_file)}...")
    metadata = {}
    
    try:
        with open(nvd_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        total_cves = len(data.get('vulnerabilities', []))
        matched = 0
        
        for vuln in data['vulnerabilities']:
            cve_data = vuln.get('cve', {})
            cve_id = cve_data.get('id')
            
            if cve_id not in target_vulns:
                continue
            
            matched += 1
            
            # 提取描述
            descriptions = cve_data.get('descriptions', [])
            description = None
            for desc in descriptions:
                if desc.get('lang') == 'en':
                    description = desc.get('value')
                    break
            
            # 提取CWE信息
            cwe_ids = []
            weaknesses = cve_data.get('weaknesses', [])
            
            # 优先使用Primary来源的CWE
            for weakness in weaknesses:
                if weakness.get('type') == 'Primary':
                    for desc in weakness.get('description', []):
                        cwe_id = desc.get('value')
                        if cwe_id and cwe_id.startswith('CWE-'):
                            cwe_ids.append(cwe_id)
            
            # 如果没有Primary，使用任何可用的CWE
            if not cwe_ids:
                for weakness in weaknesses:
                    for desc in weakness.get('description', []):
                        cwe_id = desc.get('value')
                        if cwe_id and cwe_id.startswith('CWE-'):
                            cwe_ids.append(cwe_id)
            
            # 标准化CWE数据
            unique_ids, cwe_names = normalize_cwe_data(cwe_ids, cwe_mapping)
            
            metadata[cve_id] = {
                'cwe_ids': unique_ids,
                'cwe_names': cwe_names,
                'description': description
            }
        
        print(f"  总CVE数: {total_cves}, 匹配目标: {matched}")
        
    except Exception as e:
        print(f"  错误: {e}")
    
    return metadata

def parse_all_nvd_files(nvd_dir: str, target_vulns: Set[str], cwe_mapping: Dict[str, str]) -> Dict:
    """解析所有NVD文件"""
    print("\n" + "="*70)
    print("解析NVD 2.0批量数据")
    print("="*70)
    
    all_metadata = {}
    nvd_files = sorted(Path(nvd_dir).glob('nvdcve-2.0-*.json'))
    
    for nvd_file in nvd_files:
        metadata = parse_nvd_2_0_file(str(nvd_file), target_vulns, cwe_mapping)
        all_metadata.update(metadata)
    
    print(f"\nNVD总计匹配: {len(all_metadata)} 个CVE")
    return all_metadata

def load_kernel_cves(kernel_file: str, target_vulns: Set[str], cwe_mapping: Dict[str, str]) -> Dict:
    """加载kernel_cves.json"""
    print("\n" + "="*70)
    print("加载本地 kernel_cves.json")
    print("="*70)
    
    metadata = {}
    try:
        with open(kernel_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for cve_id, cve_data in data.items():
            if cve_id not in target_vulns:
                continue
            
            cwe_ids = []
            if cve_data.get('cwe_id'):
                cwe_ids = [cve_data['cwe_id']]
            
            # 标准化CWE数据
            unique_ids, cwe_names = normalize_cwe_data(cwe_ids, cwe_mapping)
            
            metadata[cve_id] = {
                'cwe_ids': unique_ids,
                'cwe_names': cwe_names,
                'description': cve_data.get('description')
            }
        
        print(f"匹配目标: {len(metadata)} 个CVE")
        
    except Exception as e:
        print(f"错误: {e}")
    
    return metadata

def load_dataset_json(dataset_file: str, target_vulns: Set[str], cwe_mapping: Dict[str, str]) -> Dict:
    """加载Dataset.json"""
    print("\n" + "="*70)
    print("加载本地 Dataset.json")
    print("="*70)
    
    metadata = {}
    try:
        with open(dataset_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for cve_id, cve_data in data.items():
            if cve_id not in target_vulns:
                continue
            
            cwe_ids = []
            if cve_data.get('cwe_id'):
                cwe_ids = [cve_data['cwe_id']]
            
            # 标准化CWE数据
            unique_ids, cwe_names = normalize_cwe_data(cwe_ids, cwe_mapping)
            
            metadata[cve_id] = {
                'cwe_ids': unique_ids,
                'cwe_names': cwe_names,
                'description': cve_data.get('description')
            }
        
        print(f"匹配目标: {len(metadata)} 个CVE")
        
    except Exception as e:
        print(f"错误: {e}")
    
    return metadata

def query_osv_api(osv_ids: Set[str], cwe_mapping: Dict[str, str]) -> Dict:
    """通过OSV API查询OSV漏洞信息"""
    print("\n" + "="*70)
    print("查询OSV API")
    print("="*70)
    
    metadata = {}
    total = len(osv_ids)
    
    print(f"需要查询 {total} 个OSV漏洞")
    
    for i, osv_id in enumerate(sorted(osv_ids), 1):
        try:
            # OSV API endpoint
            url = f"https://api.osv.dev/v1/vulns/{osv_id}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # 提取描述
                description = data.get('summary') or data.get('details')
                
                # OSV通常没有CWE信息，但可以尝试从database_specific中获取
                cwe_ids = []
                db_specific = data.get('database_specific', {})
                if 'cwe_ids' in db_specific:
                    cwe_ids = db_specific['cwe_ids']
                
                # 标准化CWE数据
                unique_ids, cwe_names = normalize_cwe_data(cwe_ids, cwe_mapping)
                
                metadata[osv_id] = {
                    'cwe_ids': unique_ids,
                    'cwe_names': cwe_names,
                    'description': description
                }
                
                print(f"  [{i}/{total}] ✓ {osv_id}")
            else:
                print(f"  [{i}/{total}] ✗ {osv_id} (HTTP {response.status_code})")
            
            # 避免请求过快
            if i < total:
                time.sleep(0.5)
                
        except Exception as e:
            print(f"  [{i}/{total}] ✗ {osv_id} (错误: {e})")
    
    print(f"\nOSV总计获取: {len(metadata)} 个")
    return metadata

def merge_metadata(sources: list) -> Dict:
    """
    合并多个数据源的元数据
    优先级: kernel_cves.json > Dataset.json > NVD 2.0 > OSV API
    """
    print("\n" + "="*70)
    print("合并所有数据源")
    print("="*70)
    
    merged = {}
    
    # 按优先级逆序合并（后面的优先级更高）
    for source_name, source_data in reversed(sources):
        for vul_id, metadata in source_data.items():
            if vul_id not in merged:
                merged[vul_id] = metadata
            else:
                # 如果已存在，用更高优先级的数据补充或覆盖
                existing = merged[vul_id]
                
                # 如果当前来源有CWE而已有的没有，使用当前的
                if metadata.get('cwe_ids') and not existing.get('cwe_ids'):
                    existing['cwe_ids'] = metadata['cwe_ids']
                    existing['cwe_names'] = metadata['cwe_names']
                
                # 如果当前来源有描述而已有的没有，使用当前的
                if metadata.get('description') and not existing.get('description'):
                    existing['description'] = metadata['description']
    
    return merged

def analyze_coverage(merged: Dict, target_vulns: Set[str]):
    """分析数据覆盖情况"""
    print("\n" + "="*70)
    print("数据覆盖分析")
    print("="*70)
    
    total = len(target_vulns)
    covered = len(merged)
    missing = total - covered
    
    has_cwe = sum(1 for v in merged.values() if v.get('cwe_ids'))
    has_desc = sum(1 for v in merged.values() if v.get('description'))
    complete = sum(1 for v in merged.values() 
                   if v.get('cwe_ids') and v.get('description'))
    
    print(f"目标漏洞总数:         {total:4d} (100.0%)")
    print(f"已找到元数据:         {covered:4d} ({covered/total*100:5.1f}%)")
    print(f"  - 有CWE信息:        {has_cwe:4d} ({has_cwe/total*100:5.1f}%)")
    print(f"  - 有描述信息:       {has_desc:4d} ({has_desc/total*100:5.1f}%)")
    print(f"  - 完整信息:         {complete:4d} ({complete/total*100:5.1f}%)")
    print(f"缺失元数据:           {missing:4d} ({missing/total*100:5.1f}%)")
    
    # 列出缺失的漏洞
    if missing > 0:
        missing_vulns = target_vulns - set(merged.keys())
        print(f"\n缺失的漏洞 (前10个):")
        for i, vul_id in enumerate(sorted(missing_vulns)[:10], 1):
            print(f"  {i:2d}. {vul_id}")
        if missing > 10:
            print(f"  ... 还有 {missing - 10} 个")

def main():
    # 配置路径
    VUL_LIST_PATH = 'new_vulnerabilities.csv'
    KERNEL_CVES_PATH = 'inputs/linux_kernel_cves/data/kernel_cves.json'
    DATASET_PATH = 'inputs/Dataset.json'
    NVD_DIR = '/data/cyf/vulnerabilities'
    CWE_XML_PATH = 'inputs/cwe_data/cwec_v4.19.xml'
    OUTPUT_PATH = 'inputs/cve_metadata_consolidated.json'
    
    print("="*70)
    print("CVE元数据整合工具 - 最终版本")
    print("="*70)
    
    # 1. 解析MITRE CWE数据库
    cwe_mapping = parse_cwe_xml(CWE_XML_PATH)
    
    # 2. 加载目标漏洞列表
    target_vulns = load_target_vulnerabilities(VUL_LIST_PATH)
    print(f"\n加载目标漏洞列表: {len(target_vulns)} 个")
    
    # 分离CVE和OSV
    target_cves = {v for v in target_vulns if v.startswith('CVE-')}
    target_osvs = {v for v in target_vulns if v.startswith('OSV-')}
    print(f"  - CVE: {len(target_cves)}")
    print(f"  - OSV: {len(target_osvs)}")
    
    # 3. 从各个数据源加载CVE元数据
    kernel_metadata = load_kernel_cves(KERNEL_CVES_PATH, target_cves, cwe_mapping)
    dataset_metadata = load_dataset_json(DATASET_PATH, target_cves, cwe_mapping)
    nvd_metadata = parse_all_nvd_files(NVD_DIR, target_cves, cwe_mapping)
    
    # 4. 查询OSV API
    osv_metadata = query_osv_api(target_osvs, cwe_mapping)
    
    # 5. 合并所有数据源（按优先级）
    sources = [
        ('OSV-API', osv_metadata),
        ('NVD-2.0', nvd_metadata),
        ('Dataset.json', dataset_metadata),
        ('kernel_cves.json', kernel_metadata),
    ]
    
    merged_metadata = merge_metadata(sources)
    
    # 6. 分析覆盖情况
    analyze_coverage(merged_metadata, target_vulns)
    
    # 7. 保存结果
    print("\n" + "="*70)
    print("保存结果")
    print("="*70)
    
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(merged_metadata, f, indent=2, ensure_ascii=False)
    
    print(f"已保存到: {OUTPUT_PATH}")
    print(f"总计: {len(merged_metadata)} 个漏洞的元数据")
    
    print("\n✅ 完成！")

if __name__ == '__main__':
    main()
