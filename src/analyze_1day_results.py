import os
import sqlite3
from typing import Dict, Set, Tuple
from core.state import WorkflowState
from core.models import VulnerabilityFinding, PatchFeatures, SearchResultItem
from core.utils import read_json
from core.checkpoint import CheckpointManager

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_ground_truth(db_path: str) -> Dict[str, Dict[str, str]]:
    """
    Load ground truth from the benchmark database.
    Returns: { vul_id: { version: tag } }
    """
    if not os.path.exists(db_path):
        print(f"[Error] Benchmark DB not found at {db_path}")
        return {}
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Get distinct vul_id, version, tag tuples
    cursor.execute("SELECT DISTINCT vul_id, version, tag FROM benchmark_symbols")
    rows = cursor.fetchall()
    conn.close()
    
    gt = {}
    for vul_id, ver, tag in rows:
        if vul_id not in gt:
            gt[vul_id] = {}
        gt[vul_id][ver] = tag
    return gt

def load_vuln_types(results_dir: str) -> Dict[str, str]:
    """
    Load vulnerability types from features.json files.
    Returns: { vul_id: vuln_type }
    """
    vuln_types = {}
    
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith("_features.json"):
                vul_id = file.replace("_features.json", "")
                path = os.path.join(root, file)
                
                try:
                    data = read_json(path)
                    if isinstance(data, list) and len(data) > 0:
                        item = data[0]
                        # Try forensic report first
                        vuln_type = None
                        try:
                            vuln_type = item['taxonomy'].get('vuln_type')
                        except (KeyError, TypeError):
                            pass
                        
                        # Fallback to taxonomy
                        if not vuln_type:
                            try:
                                vuln_type = item['taxonomy'].get('cwe_name')
                            except (KeyError, TypeError):
                                pass
                        
                        if not vuln_type:
                            try:
                                vuln_type = item['taxonomy'].get('vuln_type')
                            except (KeyError, TypeError):
                                pass
                        
                        if vuln_type:
                            vuln_types[vul_id] = vuln_type
                except Exception:
                    pass
    
    return vuln_types

class ReportGenerator:
    def __init__(self, state: WorkflowState):
        self.state = state
        self.repo_name = state.get("repo_name", "Unknown Repo")
        self.vul_id = state.get("vul_id", "Unknown ID")
        
        # 建立索引以便快速查找
        # 1. Map Group ID -> PatchFeatures (Phase 1 & 2 Info)
        self.features_map: Dict[str, PatchFeatures] = {
            f.group_id: f for f in state.get("analyzed_features", [])
        }
        
        # 2. Map (Group ID, Target Func) -> SearchResultItem (Phase 3 Info)
        self.search_map: Dict[str, SearchResultItem] = {}
        for cand in state.get("search_candidates", []):
            key = f"{cand.group_id}::{cand.target_func}"
            self.search_map[key] = cand

    def generate_markdown(self, finding: VulnerabilityFinding) -> str:
        """
        生成单份漏洞报告，聚合所有阶段的信息。
        """
        # --- 数据回溯 ---
        group_id = finding.group_id
        feature = self.features_map.get(group_id)
        
        search_key = f"{group_id}::{finding.target_func}"
        candidate = self.search_map.get(search_key)
        
        if not feature:
            return f"Error: Context missing for group {group_id}"

        # --- 报告构建 ---
        md = []
        md.append(f"# 🛡️ Vulnerability Analysis Report")
        md.append(f"**Vuln ID**: `{self.vul_id}` | **Target**: `{finding.target_func}`\n")
        
        # === 0. Executive Summary (Verdict) ===
        status_icon = "🔴" if finding.is_vulnerable else "🟢"
        md.append(f"## 0. Executive Summary")
        md.append(f"- **Verdict**: {status_icon} **{'VULNERABLE' if finding.is_vulnerable else 'SAFE'}**")
        md.append(f"- **File**: `{finding.target_file}`")
        md.append(f"- **Repository**: `{finding.repo_path}`")
        
        # === Phase 1: Context (Metadata) ===
        md.append(f"\n## 1. Context (Phase 1)")
        md.append(f"- **Commit Message**: {feature.commit_message.splitlines()[0] if feature.commit_message else 'N/A'}")
        md.append(f"- **Patch Group ID**: `{group_id}`")
        
        # === Phase 2: Semantics & Taxonomy (The Logic) ===
        md.append(f"\n## 2. Vulnerability Semantics (Phase 2)")
        md.append(f"### Taxonomy")
        md.append(f"- **Vuln Type**: `{feature.taxonomy.vuln_type.value if feature.taxonomy.vuln_type else 'Unknown'}`")
        md.append(f"- **Type Confidence**: `{feature.taxonomy.type_confidence.value if feature.taxonomy.type_confidence else 'Unknown'}`")
        md.append(f"- **CWE**: `{feature.taxonomy.cwe_id or 'N/A'}` - {feature.taxonomy.cwe_name or 'N/A'}")

        md.append(f"\n### Semantic Logic")
        md.append(f"**Root Cause**:\n> {feature.semantics.root_cause}\n")
        md.append(f"**Attack Path (Vulnerability Logic)**:\n> {feature.semantics.attack_path}\n")
        md.append(f"**Fix Mechanism (Defense)**:\n> {feature.semantics.fix_mechanism}\n")
        
        # === Phase 3: Structural Matching (The Hints) ===
        md.append(f"\n## 3. Structural Evidence (Phase 3)")
        if candidate:
            ev = candidate.evidence
            md.append(f"- **Structural Match Confidence**: `{candidate.confidence:.2f}`")
        else:
            md.append("> Structural matching details unavailable.")

        # === Phase 4: Adversarial Verification (The Proof) ===
        md.append(f"\n## 4. Adversarial Verification (Phase 4)")
        
        # [New] Scope & Context
        md.append(f"### Analysis Scope")
        if hasattr(finding, 'involved_functions') and finding.involved_functions:
             md.append(f"- **Involved Functions**: `{', '.join(finding.involved_functions)}`")
        if hasattr(finding, 'peer_functions') and finding.peer_functions:
             md.append(f"- **Peer Context Used**: `{', '.join(finding.peer_functions)}`")
        else:
             md.append(f"- **Peer Context Used**: None (Isolated Analysis)")

        # [New] Execution Path Evidence
        if hasattr(finding, 'origin') and finding.origin:
             md.append(f"\n### Confirmed Execution Path")
             md.append(f"- **Origin (Vulnerable State Created)**: `{finding.origin}`")
             md.append(f"- **Impact (Vulnerability Triggered)**: `{finding.impact}`")
             md.append(f"- **Defense Status**: `{finding.defense_status}`")

        md.append(f"\n### Judge's Final Decision")
        md.append(finding.analysis_report)
        
        # 如果你有在 finding 中存储 step_source / step_sink (之前的建议)，这里可以展示
        # (Moved above)
        
        # 展示目标代码片段 (可选)
        if candidate and candidate.code_content:
            md.append(f"\n### Target Code Context")
            # 截取一部分代码防止过长
            code_snippet = candidate.code_content
            if len(code_snippet) > 20000: code_snippet = code_snippet[:20000] + "\n... (truncated)"
            md.append(f"```c\n{code_snippet}\n```")

        return "\n".join(md)

# ==============================================================================
# Benchmark Analyzer
# ==============================================================================

def benchmark_result_analyzer(benchmark_dir: str, output_report_dir: str = "outputs/benchmark_errors", allowed_cves: Set[str] = None, by_type: bool = False):
    """
    分析 Benchmark 结果，计算指标，并为 FP/FN 生成详细调试报告。
    
    Args:
        by_type: If True, output stats grouped by vulnerability type
    """
    ensure_dir(output_report_dir)
    
    print(f"[*] Loading Ground Truth from databases/idx_benchmark.db ...")
    gt = load_ground_truth("databases/idx_benchmark.db")
    if not gt:
        print("[!] No ground truth loaded. Exiting.")
        return
    
    # Load vulnerability types if needed
    vuln_types = {}
    if by_type:
        print(f"[*] Loading vulnerability types from {benchmark_dir}...")
        vuln_types = load_vuln_types(benchmark_dir)
        print(f"    Found types for {len(vuln_types)} vulnerabilities")

    print(f"[*] Scanning benchmark results in {benchmark_dir}...")
    
    # actual_outcomes[vul_id][version] = is_vulnerable (bool)
    actual_outcomes: Dict[str, Dict[str, bool]] = {}
    # findings_map[vul_id][version] = [ (func_name, is_vuln), ... ]
    findings_map: Dict[str, Dict[str, list]] = {}
    # candidates_map[vul_id][version] = [ func_name, ... ]
    candidates_map: Dict[str, Dict[str, list]] = {}
    # pkl_paths[vul_id] = path_to_pkl
    pkl_paths: Dict[str, str] = {}
    # 跟踪哪些 vul_id 有 findings 文件（成功完成分析）
    vul_ids_with_findings: Set[str] = set()

    # 1. 收集实际运行结果
    for root, dirs, files in os.walk(benchmark_dir):
        for file in files:
            # Scan Candidates (Search Phase)
            if file.endswith("_benchmark_candidates.json"):
                vul_id = file.split('_')[0]
                path = os.path.join(root, file)
                data = read_json(path)
                
                if vul_id not in candidates_map: candidates_map[vul_id] = {}
                
                for item in data:
                    raw_target = item.get("target_func", "")
                    
                    # [新增] 模拟 Verifier 的筛选逻辑 (Phase 3 Metric Alignment)
                    # 只有通过筛选的 candidate 才有资格被计为 Phase 3 的检出 (TP/FP)
                    verdict = item.get("verdict", "UNKNOWN")
                    confidence = item.get("confidence", 0.0)
                    
                    pass_filter = False
                    # 1. 常规通道
                    if verdict in ("VULNERABLE", "UNKNOWN") and confidence >= 0.4:
                        pass_filter = True
                        
                    if not pass_filter:
                        continue

                    try:
                        tag, ver, func = raw_target.split(':', 2)
                        if ver not in candidates_map[vul_id]:
                            candidates_map[vul_id][ver] = []
                        candidates_map[vul_id][ver].append(raw_target)
                    except:
                        pass

            if file.endswith("_benchmark_findings.json"):
                result_path = os.path.join(root, file)
                parts = root.split(os.sep)
                if len(parts) >= 2:
                    repo = parts[-1]
                    owner = parts[-2]
                    repo_name = f"{owner}/{repo}"
                
                vul_id = file.split('_')[0]
                
                # 记录该 vul_id 有 findings 文件（成功完成分析）
                vul_ids_with_findings.add(vul_id)
                
                # 构建 pkl 路径
                pkl_path = os.path.join('outputs', 'checkpoints', repo_name, vul_id, f"{vul_id}_benchmark_phase4.pkl")
                if not os.path.exists(pkl_path):
                    pkl_path = os.path.join('outputs', 'checkpoints', repo_name, vul_id, f"{vul_id}_repo_phase4.pkl")
                pkl_paths[vul_id] = pkl_path

                result_data = read_json(result_path)
                
                if vul_id not in actual_outcomes:
                    actual_outcomes[vul_id] = {}
                    findings_map[vul_id] = {}
                
                for item in result_data:
                    raw_target = item.get("target_func", "N/A") # e.g. "vul:v5.10:function_name"
                    is_vulnerable = item.get("is_vulnerable", False)
                    
                    try:
                        tag, ver, func_name = raw_target.split(':', 2)
                    except ValueError:
                        print(f"  [Warn] Invalid target format: {raw_target}")
                        continue
                    
                    # 初始化该版本结果 (默认为 False)
                    if ver not in actual_outcomes[vul_id]:
                        actual_outcomes[vul_id][ver] = False
                        findings_map[vul_id][ver] = []
                    
                    # 只要有一个函数报漏洞，该版本即视为 Vulnerable
                    actual_outcomes[vul_id][ver] = actual_outcomes[vul_id][ver] or is_vulnerable
                    findings_map[vul_id][ver].append((raw_target, is_vulnerable))

    # 2. 计算指标 & 生成报告
    tp, fp, fn, tn = 0, 0, 0, 0
    
    # Detailed counters
    fn_search_miss = 0
    fn_verification = 0
    fp_pre = 0
    fp_fix = 0
    
    # [Phase 3 Stats]
    p3_tp = 0 # Vuln version -> Has candidates
    p3_fn = 0 # Vuln version -> No candidates (Search Miss)
    p3_fp = 0 # Safe version -> Has candidates (Filtered by Phase 4 ideally)
    p3_tn = 0 # Safe version -> No candidates

    details_fn_search = []
    details_fn_verify = []
    details_fp_pre = []
    details_fp_fix = []
    
    # [By Type Stats] - Stats per vulnerability type
    type_stats = {}  # { vuln_type: { tp, fp, fn, tn, fn_search, fn_verify, fp_pre, fp_fix, vul_total, pre_total, fix_total } }
    
    def get_type_stat(vtype):
        if vtype not in type_stats:
            type_stats[vtype] = {
                'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0,
                'fn_search': 0, 'fn_verify': 0, 'fp_pre': 0, 'fp_fix': 0,
                'vul_total': 0, 'pre_total': 0, 'fix_total': 0
            }
        return type_stats[vtype]
    
    # 缓存已加载的 state
    state_cache = {} 

    def get_state(path):
        if path in state_cache: return state_cache[path]
        if os.path.exists(path):
            try:
                dir_path = os.path.dirname(path)
                file_name = os.path.basename(path)
                data = CheckpointManager.load_pkl(file_name, dir_path)
                state_cache[path] = data
                return data
            except Exception as e:
                print(f"  [Error] Failed to load pickle {path}: {e}")
                return None
        return None

    print("[*] Analyzing metrics and generating reports...")
    
    # 统计计数器
    total_vul_versions = 0
    total_pre_versions = 0
    total_fix_versions = 0
    processed_cves = 0
    failed_cves = []  # 记录失败的 CVE（没有 findings 文件）

    # 遍历 Ground Truth 进行计算
    for vul_id, versions in gt.items():
        if allowed_cves is not None and vul_id not in allowed_cves:
            continue
            
        # [修改] 只统计有 findings 文件的 vul_id
        # 没有 findings 文件的记录为失败
        if vul_id not in vul_ids_with_findings:
            # 检查是否在 allowed_cves 中（如果有筛选）
            if allowed_cves is None or vul_id in allowed_cves:
                failed_cves.append(vul_id)
            continue
            
        processed_cves += 1
        
        # Get vulnerability type for this CVE
        vtype = vuln_types.get(vul_id, 'Unknown') if by_type else None

        # 统计该 vul_id 的正负例数量
        for t in versions.values():
            if t == 'vul': 
                total_vul_versions += 1
                if by_type: get_type_stat(vtype)['vul_total'] += 1
            elif t == 'pre': 
                total_pre_versions += 1
                if by_type: get_type_stat(vtype)['pre_total'] += 1
            elif t == 'fix': 
                total_fix_versions += 1
                if by_type: get_type_stat(vtype)['fix_total'] += 1

        pkl_path = pkl_paths.get(vul_id)
        state_loaded = False
        state = None
        generator = None
        
        for ver, tag in versions.items():
            expected_vuln = (tag == 'vul')
            # 如果 actual_outcomes 中没有该版本，默认为 False (Safe/Search Miss)
            actual_vuln = actual_outcomes.get(vul_id, {}).get(ver, False)
            
            # Check Candidates (Search Phase)
            has_candidates = (vul_id in candidates_map) and (ver in candidates_map[vul_id])
            
            # Phase 3 Metrics Calculation
            if expected_vuln:
                if has_candidates:
                    p3_tp += 1
                else:
                    p3_fn += 1
            else:
                if has_candidates:
                    p3_fp += 1
                else:
                    p3_tn += 1

            error_type = None
            
            if expected_vuln:
                if actual_vuln:
                    tp += 1
                    if by_type: get_type_stat(vtype)['tp'] += 1
                else:
                    fn += 1
                    if by_type: get_type_stat(vtype)['fn'] += 1
                    error_type = "FN"
                    if not has_candidates:
                        fn_search_miss += 1
                        if by_type: get_type_stat(vtype)['fn_search'] += 1
                        details_fn_search.append(f"{vul_id}:{ver}")
                    else:
                        fn_verification += 1
                        if by_type: get_type_stat(vtype)['fn_verify'] += 1
                        details_fn_verify.append(f"{vul_id}:{ver}")
            else:
                if actual_vuln:
                    fp += 1
                    if by_type: get_type_stat(vtype)['fp'] += 1
                    error_type = "FP"
                    if tag == 'pre':
                        fp_pre += 1
                        if by_type: get_type_stat(vtype)['fp_pre'] += 1
                        details_fp_pre.append(f"{vul_id}:{ver}")
                    elif tag == 'fix':
                        fp_fix += 1
                        if by_type: get_type_stat(vtype)['fp_fix'] += 1
                        details_fp_fix.append(f"{vul_id}:{ver}")
                else:
                    tn += 1
                    if by_type: get_type_stat(vtype)['tn'] += 1
            
            # 生成报告 (仅当有 findings 记录时，即非 Search Miss)
            if error_type and vul_id in findings_map and ver in findings_map[vul_id]:
                # 延迟加载 State
                if not state_loaded and pkl_path:
                    state = get_state(pkl_path)
                    state_loaded = True
                    if state:
                        generator = ReportGenerator(state)
                
                if not generator or not state:
                    # print(f"  [Warn] Cannot generate report for {vul_id} {error_type}: Pickle not found or Search Miss.")
                    continue
                
                target_funcs = findings_map[vul_id][ver] # [(raw_target, is_vuln), ...]
                all_findings = state.get('final_findings', [])
                
                for raw_target, func_res in target_funcs:
                    func_name = raw_target # raw_target is the full string
                    
                    # FP: 预期Safe，但该函数报Vuln
                    is_cause_fp = (error_type == "FP" and func_res is True)
                    # FN: 预期Vuln，但该函数报Safe (且整个版本最终判定为Safe)
                    is_cause_fn = (error_type == "FN" and func_res is False)
                    
                    # if is_cause_fp or is_cause_fn:
                    #     matching_finding = next((f for f in all_findings if f.target_func == func_name), None)
                        
                    #     if matching_finding:
                    #         report_content = generator.generate_markdown(matching_finding)
                    #         filename = f"{error_type}_{vul_id}_{ver}_{func_name.split(':')[-1]}.md"
                    #         filename = filename.replace('/', '_').replace(':', '_')
                    #         save_path = os.path.join(output_report_dir, filename)
                            
                    #         with open(save_path, "w", encoding="utf-8") as f:
                    #             f.write(report_content)

    # 3. 输出指标
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("-" * 30)
    print(f"Total CVEs Processed: {processed_cves}")
    print(f"Total Versions: {total_vul_versions + total_pre_versions + total_fix_versions}")
    print("-" * 30)
    
    print(f"1. VULNERABLE Versions (Total: {total_vul_versions})")
    print(f"   - Detected (TP):          {tp} ({tp/total_vul_versions*100:.1f}%)")
    print(f"   - Missed (FN):            {fn} ({fn/total_vul_versions*100:.1f}%)")
    print(f"     * Search Missed:        {fn_search_miss} (Candidate not found)")
    print(f"     * Verification Missed:  {fn_verification} (Found but rejected)")

    print(f"\n2. PRE-PATCH Versions (Total: {total_pre_versions})")
    print(f"   - Correctly Safe (TN):    {total_pre_versions - fp_pre}")
    print(f"   - False Positives (FP):   {fp_pre} ({fp_pre/total_pre_versions*100:.1f}%)")

    print(f"\n3. FIXED Versions (Total: {total_fix_versions})")
    print(f"   - Correctly Safe (TN):    {total_fix_versions - fp_fix}")
    print(f"   - False Positives (FP):   {fp_fix} ({fp_fix/total_fix_versions*100:.1f}%)")

    print("-" * 30)
    print(f"Overall Precision: {precision:.4f}")
    print(f"Overall Recall:    {recall:.4f}")
    print(f"Overall F1 Score:  {f1_score:.4f}")
    print(f"Reports saved to: {output_report_dir}")
    
    # Phase 3 Stats Output
    p3_precision = p3_tp / (p3_tp + p3_fp) if (p3_tp + p3_fp) > 0 else 0
    p3_recall = p3_tp / (p3_tp + p3_fn) if (p3_tp + p3_fn) > 0 else 0
    print("\n" + "="*40)
    print("Phase 3 (Search) Performance")
    print("="*40)
    print(f"TP: {p3_tp} | FP: {p3_fp} | FN: {p3_fn} | TN: {p3_tn}")
    print(f"Precision: {p3_precision:.4f} (Ability to filter Safe versions)")
    print(f"Recall:    {p3_recall:.4f} (Ability to find Vuln versions)")

    print("\n" + "="*40)
    print("Detailed Analysis Report")
    print("="*40)
    print(f"FN (Search Miss) Examples: {details_fn_search[:10]}")
    print(f"FN (Verification) Examples: {details_fn_verify[:10]}")
    print(f"FP (Pre) Examples: {details_fp_pre[:10]}")
    print(f"FP (Fix) Examples: {details_fp_fix[:10]}")
    
    # 报告失败的漏洞
    if failed_cves:
        print("\n" + "="*40)
        print("Failed Vulnerabilities (No findings file)")
        print("="*40)
        print(f"Total Failed: {len(failed_cves)}")
        print(f"Failed CVEs: {', '.join(failed_cves[:10])}")
    else:
        print("\n" + "="*40)
        print("All vulnerabilities completed successfully!")
        print("="*40)
    
    # Output by-type statistics if enabled
    if by_type and type_stats:
        print("\n" + "="*120)
        print("Statistics by Vulnerability Type")
        print("="*120)
        
        # Sort by total vulnerable versions (descending)
        sorted_types = sorted(type_stats.items(), key=lambda x: x[1]['vul_total'], reverse=True)
        
        # Prepare table data
        table_rows = []
        for vtype, stats in sorted_types:
            vul_total = stats['vul_total']
            pre_total = stats['pre_total']
            fix_total = stats['fix_total']
            
            if vul_total == 0 and pre_total == 0 and fix_total == 0:
                continue
            
            t_tp = stats['tp']
            t_fn = stats['fn']
            t_fp = stats['fp']
            t_fn_search = stats['fn_search']
            t_fn_verify = stats['fn_verify']
            t_fp_pre = stats['fp_pre']
            t_fp_fix = stats['fp_fix']
            
            # Calculate metrics
            t_precision = t_tp / (t_tp + t_fp) if (t_tp + t_fp) > 0 else 0
            t_recall = t_tp / (t_tp + t_fn) if (t_tp + t_fn) > 0 else 0
            t_f1 = 2 * (t_precision * t_recall) / (t_precision + t_recall) if (t_precision + t_recall) > 0 else 0
            
            # Calculate percentages
            tp_pct = f"{t_tp/vul_total*100:.1f}%" if vul_total > 0 else "-"
            fn_pct = f"{t_fn/vul_total*100:.1f}%" if vul_total > 0 else "-"
            fp_pre_pct = f"{t_fp_pre/pre_total*100:.1f}%" if pre_total > 0 else "-"
            fp_fix_pct = f"{t_fp_fix/fix_total*100:.1f}%" if fix_total > 0 else "-"
            
            table_rows.append({
                'type': vtype,
                'vul': vul_total,
                'tp': t_tp,
                'tp_pct': tp_pct,
                'fn': t_fn,
                'fn_pct': fn_pct,
                'fn_s': t_fn_search,
                'fn_v': t_fn_verify,
                'pre': pre_total,
                'fp_pre': t_fp_pre,
                'fp_pre_pct': fp_pre_pct,
                'fix': fix_total,
                'fp_fix': t_fp_fix,
                'fp_fix_pct': fp_fix_pct,
                'prec': t_precision,
                'rec': t_recall,
                'f1': t_f1
            })
        
        # Print table header
        print(f"\n{'Vulnerability Type':<25} | {'Vul':>4} | {'TP':>4} {'(%)':>7} | {'FN':>4} {'(%)':>7} | {'FN_S':>4} {'FN_V':>4} | {'Pre':>4} | {'FP':>4} {'(%)':>7} | {'Fix':>4} | {'FP':>4} {'(%)':>7} | {'Prec':>6} {'Rec':>6} {'F1':>6}")
        print("-" * 145)
        
        for r in table_rows:
            print(f"{r['type']:<25} | {r['vul']:>4} | {r['tp']:>4} {r['tp_pct']:>7} | {r['fn']:>4} {r['fn_pct']:>7} | {r['fn_s']:>4} {r['fn_v']:>4} | {r['pre']:>4} | {r['fp_pre']:>4} {r['fp_pre_pct']:>7} | {r['fix']:>4} | {r['fp_fix']:>4} {r['fp_fix_pct']:>7} | {r['prec']:>6.3f} {r['rec']:>6.3f} {r['f1']:>6.3f}")
        
        print("-" * 145)
        
        # Print totals
        total_vul = sum(r['vul'] for r in table_rows)
        total_tp = sum(r['tp'] for r in table_rows)
        total_fn = sum(r['fn'] for r in table_rows)
        total_fn_s = sum(r['fn_s'] for r in table_rows)
        total_fn_v = sum(r['fn_v'] for r in table_rows)
        total_pre = sum(r['pre'] for r in table_rows)
        total_fp_pre = sum(r['fp_pre'] for r in table_rows)
        total_fix = sum(r['fix'] for r in table_rows)
        total_fp_fix = sum(r['fp_fix'] for r in table_rows)
        
        total_tp_pct = f"{total_tp/total_vul*100:.1f}%" if total_vul > 0 else "-"
        total_fn_pct = f"{total_fn/total_vul*100:.1f}%" if total_vul > 0 else "-"
        total_fp_pre_pct = f"{total_fp_pre/total_pre*100:.1f}%" if total_pre > 0 else "-"
        total_fp_fix_pct = f"{total_fp_fix/total_fix*100:.1f}%" if total_fix > 0 else "-"
        
        print(f"{'TOTAL':<25} | {total_vul:>4} | {total_tp:>4} {total_tp_pct:>7} | {total_fn:>4} {total_fn_pct:>7} | {total_fn_s:>4} {total_fn_v:>4} | {total_pre:>4} | {total_fp_pre:>4} {total_fp_pre_pct:>7} | {total_fix:>4} | {total_fp_fix:>4} {total_fp_fix_pct:>7} | {precision:>6.3f} {recall:>6.3f} {f1_score:>6.3f}")

if __name__ == "__main__":
    import argparse
    import csv
    
    parser = argparse.ArgumentParser(description='Analyze 1-day benchmark results.')
    parser.add_argument('results_dir', nargs='?', default="outputs/results", help='Directory containing benchmark results')
    parser.add_argument('--csv', help='Path to CSV file containing allowed CVEs', default='inputs/1day_vul_list.csv')
    parser.add_argument('--by-type', action='store_true', help='Output statistics grouped by vulnerability type')
    
    args = parser.parse_args()
    
    allowed_cves = None
    if args.csv:
        print(f"[*] Loading allowed CVEs from {args.csv}...")
        allowed_cves = set()
        try:
             with open(args.csv, 'r') as f:
                reader = csv.DictReader(f)
                # Cleanup headers (bom, whitespace)
                if reader.fieldnames:
                    reader.fieldnames = [h.strip().lstrip('\ufeff') for h in reader.fieldnames]
                
                if 'vul_id' in reader.fieldnames:
                    for row in reader:
                        if row['vul_id']:
                            allowed_cves.add(row['vul_id'].strip())
                else:
                    print(f"[!] Error: CSV must have 'vul_id' header. Found: {reader.fieldnames}")
                    exit(1)
                            
        except Exception as e:
            print(f"[!] Error reading CSV: {e}")
            exit(1)
        print(f"[*] Filter enabled: {len(allowed_cves)} CVEs allowed.")

    benchmark_result_analyzer(args.results_dir, output_report_dir="outputs/benchmark_errors", allowed_cves=allowed_cves, by_type=args.by_type)