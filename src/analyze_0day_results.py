import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

def load_json(path):
    """Safely load JSON file."""
    try:
        if not path.exists():
            return None
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def load_tp_dataset(tp_csv_path):
    """
    Load TP dataset and return a set of (repo, vul_id, target_file, target_func) tuples.
    Also returns a dict keyed by (repo, vul_id) -> list of (target_file, target_func).
    """
    tp_set = set()  # (repo, vul_id, target_file, target_func)
    tp_by_vul = {}  # (repo, vul_id) -> [(target_file, target_func), ...]
    
    if not tp_csv_path or not Path(tp_csv_path).exists():
        return tp_set, tp_by_vul
    
    try:
        with open(tp_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                reader.fieldnames = [h.strip().lower().lstrip('\ufeff') for h in reader.fieldnames]
            
            for row in reader:
                repo = row.get('repo', '').strip()
                vul_id = row.get('vul_id', '').strip()
                target_file = row.get('target_file_path', '').strip()
                target_func = row.get('target_func_name', '').strip()
                
                if repo and vul_id:
                    tp_set.add((repo, vul_id, target_file, target_func))
                    key = (repo, vul_id)
                    if key not in tp_by_vul:
                        tp_by_vul[key] = []
                    tp_by_vul[key].append((target_file, target_func))
    except Exception as e:
        print(f"Error loading TP dataset: {e}")
    
    return tp_set, tp_by_vul

def load_checked_list(checked_csv_path):
    """
    Load checked list and return a dict keyed by (repo, vul_id, target_file, target_func) -> row_data.
    Also returns TP, FP, reported, and confirmed sets separately.
    """
    checked_dict = {}  # (repo, vul_id, target_file, target_func) -> row_data dict
    tp_set = set()  # TP entries
    fp_set = set()  # FP entries
    reported_set = set()  # reported entries
    confirmed_set = set()  # confirmed entries
    
    if not checked_csv_path or not Path(checked_csv_path).exists():
        return checked_dict, tp_set, fp_set, reported_set, confirmed_set
    
    try:
        with open(checked_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                reader.fieldnames = [h.strip().lower().lstrip('\ufeff') for h in reader.fieldnames]
            
            for row in reader:
                repo = row.get('repo', '').strip()
                vul_id = row.get('vul_id', '').strip()
                target_file = row.get('target_file_path', '').strip()
                target_func = row.get('target_func_name', '').strip()
                judgement = row.get('judgement', '').strip()
                reported = row.get('reported', '').strip()
                confirmed = row.get('confirmed', '').strip()
                
                if repo and vul_id:
                    key = (repo, vul_id, target_file, target_func)
                    checked_dict[key] = {
                        'judgement': judgement,
                        'reported': reported,
                        'confirmed': confirmed
                    }
                    
                    if judgement == 'TP':
                        tp_set.add(key)
                    elif judgement == 'FP':
                        fp_set.add(key)
                    
                    if reported:
                        reported_set.add(key)
                    if confirmed:
                        confirmed_set.add(key)
    except Exception as e:
        print(f"Error loading checked list: {e}")
    
    return checked_dict, tp_set, fp_set, reported_set, confirmed_set

def analyze_vulnerability(vul_id, repo, results_base, input_row, max_findings=None):
    """
    Analyze artifacts for a single vulnerability across all phases.
    Returns a LIST of dictionaries (one per finding, or one for summary if no findings).
    
    Args:
        max_findings: Optional limit on number of findings per vulnerability (default: no limit)
    """
    repo_path = None
    
    # 1. Try direct path
    candidate_path = results_base / repo
    if candidate_path.exists():
        repo_path = candidate_path
    
    # 2. Optimized search for known repos (if short name used)
    elif repo == 'linux':
        repo_path = results_base / 'torvalds' / 'linux'
    elif repo == 'php-src':
        repo_path = results_base / 'php' / 'php-src'
    elif repo == 'systemd':
        repo_path = results_base / 'systemd' / 'systemd'
    else:
        # 3. Fallback search: look for any directory named 'repo' (leaf name)
        # Handle "Owner/Repo" vs just "Repo"
        repo_name = repo.split('/')[-1]
        
        # Search owner/repo
        found = list(results_base.glob(f"*/{repo_name}"))
        if found:
            repo_path = found[0]

    # Base dictionary for common fields (using new column names)
    base_row = {
        "vul_id": vul_id,
        "repo": repo,
        "fixed_commit_sha": input_row.get('fixed_commit_sha', '')
    }

    if not repo_path or not repo_path.exists():
        row = base_row.copy()
        row.update({
            "target_file": "N/A", "target_func": "N/A",
            "patch_file": "N/A", "patch_func": "N/A",
            "CWE": "Unknown", "vuln_type": "Unknown",
            "Verdict": "Repo_Not_Found"
        })
        return [row], {'total': 0, 'valid': 0}

    # Artifact Paths
    features_path = repo_path / f"{vul_id}_features.json"
    candidates_path = repo_path / f"{vul_id}_repo_candidates.json"
    findings_path = repo_path / f"{vul_id}_repo_findings.json"

    # --- Phase 1: Feature Extraction ---
    features_data = load_json(features_path)
    phase1_found = features_data is not None
    
    cwe_id = "Unknown"
    vuln_type = "Unknown"
    patch_file = "N/A"
    patch_func = "N/A"
    hunks_count = 0
    candidates_count = 0

    if phase1_found:
        # Parse standard feature format
        if isinstance(features_data, list) and len(features_data) > 0:
            item = features_data[0]
            # Get data from semantics directly (forensic_report removed in Phase 2 refactor)
            semantics = item.get('semantics', {})
            cwe_id = semantics.get('cwe_id', 'Unknown')
            vuln_type_value = semantics.get('vuln_type', 'Unknown')
            # Extract enum value if it's a dict with 'value' key
            if isinstance(vuln_type_value, dict) and 'value' in vuln_type_value:
                vuln_type = vuln_type_value['value']
            else:
                vuln_type = str(vuln_type_value) if vuln_type_value else 'Unknown'
            
            # Count hunks and get patch info
            patches = item.get('patches', [])
            hunks_count = len(patches)
            if patches:
                patch_file = patches[0].get('file_path', 'N/A')
                patch_func = patches[0].get('function_name', 'N/A')

    # --- Phase 4: Findings/Verdict ---
    findings_data = load_json(findings_path)
    
    # Common metrics for result rows
    base_row.update({
        "CWE": cwe_id,
    })

    verified_findings = []
    p4_stats = {'total': 0, 'valid': 0}
    
    if findings_data:
        verified_findings = [f for f in findings_data if f.get('is_vulnerable') is True]
        
        # Filter 1: if both origin.func_name and impact.func_name don't match target_func, remove it
        filtered_findings = []
        for f in verified_findings:
            target_func = f.get('target_func') or f.get('func_name')
            origin = f.get('origin')
            impact = f.get('impact')
            
            # Check if at least one of origin or impact matches target_func
            origin_matches = False
            impact_matches = False
            
            if origin and isinstance(origin, dict):
                origin_func = origin.get('func_name')
                if origin_func and target_func and origin_func == target_func:
                    origin_matches = True
            
            if impact and isinstance(impact, dict):
                impact_func = impact.get('func_name')
                if impact_func and target_func and impact_func == target_func:
                    impact_matches = True
            
            # Keep the finding if at least one matches, or if origin/impact are null
            if origin_matches or impact_matches or origin is None or impact is None:
                filtered_findings.append(f)
        
        verified_findings = filtered_findings
        
        # Filter 2: Compare vuln_type and cwe_id with features.json
        # Extract vuln_type and cwe_id from features
        features_vuln_type = None
        features_cwe_id = None
        if features_data and isinstance(features_data, list) and len(features_data) > 0:
            semantics = features_data[0].get('semantics', {})
            features_cwe_id = semantics.get('cwe_id')
            features_vuln_type_value = semantics.get('vuln_type')
            if isinstance(features_vuln_type_value, dict) and 'value' in features_vuln_type_value:
                features_vuln_type = features_vuln_type_value['value']
            else:
                features_vuln_type = str(features_vuln_type_value) if features_vuln_type_value else None
        
        # Filter findings based on vuln_type or cwe_id match
        type_filtered_findings = []
        for f in verified_findings:
            finding_vuln_type = f.get('vuln_type')
            finding_cwe_id = f.get('cwe_id')
            
            # Check if at least one of vuln_type or cwe_id matches
            vuln_type_matches = False
            cwe_id_matches = False
            
            if features_vuln_type and finding_vuln_type:
                if features_vuln_type == finding_vuln_type:
                    vuln_type_matches = True
            
            if features_cwe_id and finding_cwe_id:
                if features_cwe_id == finding_cwe_id:
                    cwe_id_matches = True
            
            # Keep the finding if at least one matches, or if features data is not available
            if vuln_type_matches or cwe_id_matches or not features_data:
                type_filtered_findings.append(f)
        
        verified_findings = type_filtered_findings
        
        # Apply max_findings limit if specified
        if max_findings and len(verified_findings) > max_findings:
            verified_findings = verified_findings[:max_findings]

        # Calculate Phase 4 stats
        p4_stats['total'] = len(findings_data)
        p4_stats['valid'] = len(verified_findings)

    # Case A: We have verified findings -> Return one row per finding
    if verified_findings:
        rows = []
        for f in verified_findings:
            row = base_row.copy()
            
            # Location
            t_file = f.get('target_file') or f.get('file_path') or f.get('location') or "N/A"
            t_func = f.get('target_func') or f.get('func_name') or "N/A"
            
            # Specific patch info from finding (if differs or is more specific)
            p_file = f.get('patch_file') or patch_file
            p_func = f.get('patch_func') or patch_func
            v_type = f.get('vuln_type') or vuln_type
            
            row.update({
                "patch_file": p_file,
                "patch_func": p_func,
                "target_file": t_file,
                "target_func": t_func,
                "vuln_type": v_type,
                "Verdict": "Confirmed_Finding"
            })
            rows.append(row)
        return rows, p4_stats

    # Case B: No verified findings -> Return summary row (Missed/Failed)
    verdict = "Missed"
    # Identify max confidence even if not vulnerable (for context)
    max_conf_overall = 0.0
    if findings_data:
        for f in findings_data:
             try: c = float(f.get('confidence', 0))
             except: c = 0.0
             if c > max_conf_overall: max_conf_overall = c

    if phase1_found and hunks_count > 0:
        if candidates_count > 0:
            verdict = "Candidates_Only"
        else:
            verdict = "Search_Failed"
    elif phase1_found and hunks_count == 0:
        verdict = "Denoising_Issue"
    elif not phase1_found:
        verdict = "Analysis_Failed"

    single_row = base_row.copy()
    single_row.update({
        "patch_file": patch_file,
        "patch_func": patch_func,
        "target_file": "N/A",
        "target_func": "N/A",
        "vuln_type": vuln_type,
        "Verdict": verdict
    })
    return [single_row], p4_stats

def check_tp_match(row, tp_set):
    """
    Check if a finding row matches any TP entry.
    Returns True if it's a TP match (exact match only).
    """
    repo = row.get('repo', '')
    vul_id = row.get('vul_id', '')
    target_file = row.get('target_file', '')
    target_func = row.get('target_func', '')
    
    # Exact match only
    if (repo, vul_id, target_file, target_func) in tp_set:
        return True
    
    return False

def main():
    parser = argparse.ArgumentParser(description="Generate 0-day Analysis Report")
    parser.add_argument("-i", "--input_csv", default="inputs/0day_vul_list.csv", help="Input list of target vulnerabilities")
    parser.add_argument("-o", "--output_csv", default="validation_report_0day.csv", help="Output analysis CSV")
    parser.add_argument("-r", "--results_dir", default="outputs/results", help="Directory containing JSON results")
    parser.add_argument("-t", "--tp_csv", default=None, help="TP dataset CSV for judgement labeling (optional)")
    parser.add_argument("-c", "--checked_csv", default="results/checked_list.csv", help="Checked list CSV for comparison (optional)")
    parser.add_argument("-m", "--max_findings_per_vul", type=int, default=10, help="Max findings per vulnerability (default: no limit, e.g. 5)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)
    results_base = Path(args.results_dir)
    
    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist.")
        sys.exit(1)
    
    # Load TP dataset if provided
    tp_set, tp_by_vul = load_tp_dataset(args.tp_csv)
    if tp_set:
        print(f"Loaded {len(tp_set)} TP entries from {args.tp_csv}")
    
    # Load checked list if provided
    checked_dict, checked_tp_set, checked_fp_set, checked_reported_set, checked_confirmed_set = load_checked_list(args.checked_csv)
    if checked_dict:
        print(f"Loaded {len(checked_dict)} checked entries from {args.checked_csv}")
        print(f"  - TP: {len(checked_tp_set)}")
        print(f"  - FP: {len(checked_fp_set)}")
        print(f"  - Reported: {len(checked_reported_set)}")
        print(f"  - Confirmed: {len(checked_confirmed_set)}")
    
    print(f"Reading targets from {input_path}...")
    
    final_output_rows = []
    stats = Counter()
    processed_vuls = set()
    
    # Global Phase 4 stats
    global_p4_total = 0
    global_p4_valid = 0
    
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        # Normalize headers to lowercase just in case
        reader.fieldnames = [name.lower() for name in reader.fieldnames]
        
        for row in reader:
            vul_id = row.get('vul_id') or row.get('cve_id')
            repo = row.get('repo')
            
            if not vul_id or not repo:
                continue

            # Dedup: process each (vul_id, repo) only once
            if (vul_id, repo) in processed_vuls:
                continue
            processed_vuls.add((vul_id, repo))
                
            stats['Total_Vuls_Scanned'] += 1
            
            # Analyze
            vuln_rows, p4_stats = analyze_vulnerability(vul_id, repo, results_base, row, max_findings=args.max_findings_per_vul)
            final_output_rows.extend(vuln_rows)
            
            # Aggregate stats
            global_p4_total += p4_stats['total']
            global_p4_valid += p4_stats['valid']
            
            # Stats calculation (per vulnerability)
            has_finding = any(r['Verdict'] == 'Confirmed_Finding' for r in vuln_rows)
            
            if has_finding:
                stats['Vuls_Confirmed'] += 1
                stats['Confirmed_Finding'] += 1
            else:
                if vuln_rows:
                    verdict = vuln_rows[0].get('Verdict', 'Unknown')
                    stats[verdict] += 1
                
    # Write Output
    verified_rows = [r for r in final_output_rows if r.get('Verdict') == 'Confirmed_Finding']
    
    # === Filter CVE-Function pairs appearing more than 5 times ===
    cve_func_counts = Counter((row['vul_id'], row.get('patch_func', '')) for row in verified_rows)
    
    # Get (CVE, Function) pairs that appear > 5 times
    frequent_cve_funcs = {key: count for key, count in cve_func_counts.items() if count >= 4}
    
    if frequent_cve_funcs:
        print(f"\n=== Filtering CVE-Function pairs appearing > 5 times ===")
        print(f"Total unique (CVE, Function) pairs before filtering: {len(cve_func_counts)}")
        print(f"(CVE, Function) pairs to be filtered (appearing > 5 times): {len(frequent_cve_funcs)}")
        
        # Show top frequent CVE-Function pairs
        sorted_frequent = sorted(frequent_cve_funcs.items(), key=lambda x: x[1], reverse=True)
        for (cve, func), count in sorted_frequent[:10]:  # Show top 10
            print(f"  {cve} :: {func}: {count} occurrences")
        if len(sorted_frequent) > 10:
            print(f"  ... and {len(sorted_frequent) - 10} more")
        
        # Filter out these CVE-Function pairs
        rows_before = len(verified_rows)
        verified_rows = [r for r in verified_rows if cve_func_counts[(r['vul_id'], r.get('patch_func', ''))] < 5]
        rows_after = len(verified_rows)
        
        print(f"Rows before filtering: {rows_before}")
        print(f"Rows after filtering: {rows_after}")
        print(f"Rows removed: {rows_before - rows_after}")
    
    # Add judgement column based on checked_list or TP matching
    for row in verified_rows:
        key = (
            row.get('repo', ''),
            row.get('vul_id', ''),
            row.get('target_file', ''),
            row.get('target_func', '')
        )
        
        # First check checked_list for judgement
        if checked_dict and key in checked_dict:
            row['judgement'] = checked_dict[key]['judgement']
        # Then check TP dataset for backward compatibility
        elif tp_set and check_tp_match(row, tp_set):
            row['judgement'] = 'TP'
        else:
            row['judgement'] = ''
    
    if not verified_rows:
        print("No confirmed findings generated.")
    else:
        # Updated fieldnames with new column names
        fieldnames = [
            "vul_id", "repo", "fixed_commit_sha", "patch_file", "patch_func", 
            "target_file", "target_func", "CWE", "vuln_type", "judgement"
        ]
        
        print(f"Writing report to {output_path}...")
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            # extrasaction='ignore' is crucial
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(verified_rows)
        
        print(f"\nReport generated at: {output_path.absolute()}")
    
    # === TP Coverage Analysis ===
    if tp_set:
        print("\n=== TP Coverage Analysis ===")
        
        # Build set of found (repo, vul_id, target_file, target_func) from results
        found_set = set()
        for row in verified_rows:
            found_set.add((
                row.get('repo', ''),
                row.get('vul_id', ''),
                row.get('target_file', ''),
                row.get('target_func', '')
            ))
        
        # Check each TP entry
        missed_tps = []
        found_tps = []
        
        for tp_repo, tp_vul, tp_file, tp_func in tp_set:
            # Check if this TP was found (exact match only)
            tp_key = (tp_repo, tp_vul, tp_file, tp_func)
            if tp_key in found_set:
                found_tps.append((tp_repo, tp_vul, tp_file, tp_func))
            else:
                missed_tps.append((tp_repo, tp_vul, tp_file, tp_func))
        
        print(f"TP Total: {len(tp_set)}")
        print(f"TP Found: {len(found_tps)}")
        print(f"TP Missed: {len(missed_tps)}")
        
        if missed_tps:
            print(f"\n=== Missed TP List ({len(missed_tps)} items) ===")
            # Group by (repo, vul_id) for cleaner output
            missed_by_vul = {}
            for repo, vul, file, func in missed_tps:
                key = (repo, vul)
                if key not in missed_by_vul:
                    missed_by_vul[key] = []
                missed_by_vul[key].append((file, func))
            
            for (repo, vul), entries in sorted(missed_by_vul.items()):
                print(f"\n  [{repo}] {vul}:")
                for file, func in entries:
                    print(f"    - {file} :: {func}")
        
        # Calculate recall
        if len(tp_set) > 0:
            recall = len(found_tps) / len(tp_set) * 100
            print(f"\nTP Recall: {recall:.1f}% ({len(found_tps)}/{len(tp_set)})")
    
    # === Results Statistics ===
    if checked_dict:
        print("\n=== Results Statistics ===")
        
        # Count TP/FP/Reported/Confirmed in verified_rows
        count_tp = 0
        count_fp = 0
        count_reported = 0
        count_confirmed = 0
        
        for row in verified_rows:
            key = (
                row.get('repo', ''),
                row.get('vul_id', ''),
                row.get('target_file', ''),
                row.get('target_func', '')
            )
            
            # Check if this finding is in checked_dict
            if key in checked_dict:
                judgement = checked_dict[key]['judgement']
                if judgement == 'TP':
                    count_tp += 1
                elif judgement == 'FP':
                    count_fp += 1
            
            # Check if in reported/confirmed sets
            if key in checked_reported_set:
                count_reported += 1
            if key in checked_confirmed_set:
                count_confirmed += 1
        
        print(f"TP: {count_tp}")
        print(f"FP: {count_fp}")
        print(f"Reported: {count_reported}")
        print(f"Confirmed: {count_confirmed}")
        
        # Calculate precision
        if count_tp + count_fp > 0:
            precision = count_tp / (count_tp + count_fp) * 100
            print(f"Precision: {precision:.1f}% ({count_tp}/{count_tp + count_fp})")
    
    # Print Summary
    print("\n=== Analysis Summary ===")
    print(f"Total Targets Processed: {stats['Total_Vuls_Scanned']}")
    print(f"Phase 4 (Findings) > 0: {stats['Vuls_Confirmed']}")
    print(f"Total Output Rows: {len(verified_rows)}")
    
    # Phase 4 Global Ratio
    p4_ratio = (global_p4_valid / global_p4_total * 100) if global_p4_total > 0 else 0.0
    print(f"Overall Phase 4 TP Ratio (limit={args.max_findings_per_vul}): {global_p4_valid}/{global_p4_total} ({p4_ratio:.1f}%)")

if __name__ == "__main__":
    main()
