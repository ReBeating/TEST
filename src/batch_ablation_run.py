#!/usr/bin/env python3
"""
Batch Ablation Experiment Runner

Run ablation experiments, executing analysis for all 4 ablation versions on a specified vulnerability list.

Usage:
    # Run all versions
    python src/batch_ablation_run.py -c inputs/merged_0day_vul_list.csv
    
    # Run only specific versions
    python src/batch_ablation_run.py -c inputs/merged_0day_vul_list.csv --versions ablation1 ablation2
    
    # Specify parallelism
    python src/batch_ablation_run.py -c inputs/merged_0day_vul_list.csv -b 4
    
    # Run only specific vulnerability
    python src/batch_ablation_run.py -c inputs/merged_0day_vul_list.csv -v CVE-2024-1234
"""

import os
import argparse
import pandas as pd
from dotenv import load_dotenv
import concurrent.futures
from time import time
from datetime import datetime
import json

from run_ablation import build_ablation_pipeline, setup_directories
from core.configs import OUTPUT_DIR_PATH, REPO_DIR_PATH

# ==========================================
# 1. Task Processing Functions
# ==========================================

def process_vuln_ablation(item, args, ablation_type):
    """
    Processing task for a single ablation version of a single vulnerability
    
    Args:
        item: Vulnerability record (containing vul_id, repo, fixed_commit_sha)
        args: Command line arguments
        ablation_type: Ablation type (ablation1, ablation2, ablation3, none)
    
    Returns:
        (vul_id, ablation_type, success, error_msg)
    """
    vul_id = item["vul_id"]
    repo_name = item["repo"]
    commit_sha = item["fixed_commit_sha"]

    try:
        # Initialize app within the child process
        app = build_ablation_pipeline(ablation_type)

        # Path generation (automatically selects the correct directory based on ablation_type)
        ckpt_dir, res_dir = setup_directories(args.output_dir, repo_name, vul_id, ablation_type)
        full_repo_path = os.path.join(args.repo_path, repo_name)
        
        # Construct state
        vuln_state = {
            "repo_name": repo_name,
            "vul_id": vul_id,
            "commit_hash": commit_sha,
            "output_dir": ckpt_dir,
            "result_dir": res_dir,
            "repo_path": full_repo_path,
            "mode": args.mode,
            "start_phase": args.start,
            "end_phase": args.end,
            "force_execution": args.force,
            "ablation": ablation_type,  # Key: pass ablation parameter
            "lang": "c",
            "atomic_patches": [],
            "grouped_patches": [],
            "analyzed_features": [],
            "search_candidates": [],
            "final_findings": []
        }
        
        print(f">>> [Start] {vul_id} | Ablation: {ablation_type}")
        app.invoke(vuln_state, {"recursion_limit": 100})
        print(f">>> [Done] {vul_id} | {ablation_type}")
        return vul_id, ablation_type, True, None
    except Exception as e:
        error_msg = str(e)
        print(f"!!! [Error] {vul_id} | {ablation_type}: {error_msg}")
        return vul_id, ablation_type, False, error_msg


def process_vuln_all_ablations(item, args):
    """
    Process all ablation versions for a single vulnerability (sequentially)
    
    This function will be called by the multiprocessing pool, each process handles all versions of a single vulnerability.
    """
    vul_id = item["vul_id"]
    results = []
    
    for ablation_type in args.versions:
        result = process_vuln_ablation(item, args, ablation_type)
        results.append(result)
    
    return results


# ==========================================
# 2. Result Statistics
# ==========================================

def save_summary(all_results, output_path):
    """Save run summary"""
    summary = {
        "total_vulnerabilities": len(set(r[0] for r in all_results)),
        "total_tasks": len(all_results),
        "successful": sum(1 for r in all_results if r[2]),
        "failed": sum(1 for r in all_results if not r[2]),
        "by_ablation": {},
        "timestamp": datetime.now().isoformat()
    }
    
    # Statistics by ablation type
    for ablation in ["ablation1", "ablation2", "ablation3", "none"]:
        abl_results = [r for r in all_results if r[1] == ablation]
        summary["by_ablation"][ablation] = {
            "total": len(abl_results),
            "successful": sum(1 for r in abl_results if r[2]),
            "failed": sum(1 for r in abl_results if not r[2])
        }
    
    # Record failed tasks
    failed_tasks = [(r[0], r[1], r[3]) for r in all_results if not r[2]]
    if failed_tasks:
        summary["failed_tasks"] = [
            {"vul_id": vid, "ablation": abl, "error": err}
            for vid, abl, err in failed_tasks
        ]
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary


# ==========================================
# 3. Execution Entry Point
# ==========================================

def load_failed_tasks(summary_path, csv_path):
    """
    Load failed tasks from summary file and match with CSV to get full information
    
    Returns:
        List of (item_dict, ablation_type) tuples
    """
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    failed_tasks = summary.get("failed_tasks", [])
    if not failed_tasks:
        print("[!] No failed tasks found in summary file.")
        return []
    
    # Read CSV to get full vulnerability information
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset=["vul_id"])
    records_map = {row["vul_id"]: row.to_dict() for _, row in df.iterrows()}
    
    tasks = []
    for task in failed_tasks:
        vul_id = task["vul_id"]
        ablation = task["ablation"]
        
        if vul_id in records_map:
            tasks.append((records_map[vul_id], ablation))
        else:
            print(f"[!] Warning: {vul_id} not found in CSV, skipping")
    
    return tasks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Ablation Experiment Runner")
    parser.add_argument("-c", "--csv", required=True, help="Path to vulnerability CSV file")
    parser.add_argument("-s", "--start", type=int, default=1, help="Start phase (1-4)")
    parser.add_argument("-e", "--end", type=int, default=4, help="End phase (1-4)")
    parser.add_argument("-r", "--repo_path", default=REPO_DIR_PATH, help="Base path containing cloned repos")
    parser.add_argument("-o", "--output_dir", default=OUTPUT_DIR_PATH, help="Base path for outputs")
    parser.add_argument("-m", "--mode", default="repo", choices=["repo", "benchmark"], help="Operation mode")
    parser.add_argument("-b", "--workers", type=int, default=4, help="Max concurrent workers")
    parser.add_argument("-f", "--force", action="store_true", help="Force re-processing even if outputs exist")
    parser.add_argument("-v", "--vul_id", type=str, default=None, help="Only process a specific vulnerability ID")
    parser.add_argument("--versions", nargs='+',
                        default=["ablation1", "ablation2", "ablation3", "none"],
                        choices=["ablation1", "ablation2", "ablation3", "ablation4", "none",
                                 "plain", "with_extraction", "with_validation", "full"],
                        help="Which ablation versions to run (default: all 4 main versions)")
    parser.add_argument("--strategy", choices=["sequential", "parallel"], default="sequential",
                        help="sequential: each worker processes all versions of one vuln; "
                             "parallel: distribute all tasks (vuln x version) to workers")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of vulnerabilities to process")
    parser.add_argument("--retry-failed", type=str, default=None, metavar="SUMMARY_JSON",
                        help="Retry failed tasks from a previous run's summary JSON file "
                             "(e.g., outputs/ablation_batch_summary.json)")
    
    args = parser.parse_args()
    
    load_dotenv()
    
    # Normalize version names
    version_map = {
        "plain": "ablation1",
        "with_extraction": "ablation2",
        "with_validation": "ablation3",
        "full": "none",
        "ablation4": "none"
    }
    args.versions = [version_map.get(v, v) for v in args.versions]
    args.versions = list(dict.fromkeys(args.versions))  # Dereplicate and keep order
    
    # ==========================================
    # Mode selection: Retry failed tasks vs Normal run
    # ==========================================
    retry_mode = args.retry_failed is not None
    
    if retry_mode:
        # Retry failed tasks mode
        print("=" * 70)
        print("Batch Ablation Experiment - RETRY FAILED TASKS")
        print("=" * 70)
        
        failed_tasks = load_failed_tasks(args.retry_failed, args.csv)
        if not failed_tasks:
            print("[!] No tasks to retry. Exiting.")
            exit(0)
        
        # If --versions is specified, only retry failed tasks for those versions
        # Note: default args.versions includes all 4 versions, need to check if user explicitly set it
        if args.versions != ["ablation1", "ablation2", "ablation3", "none"]:
            original_count = len(failed_tasks)
            failed_tasks = [(item, abl) for item, abl in failed_tasks if abl in args.versions]
            print(f"[*] Filtering by versions {args.versions}: {original_count} -> {len(failed_tasks)} tasks")
        
        if not failed_tasks:
            print("[!] No matching tasks to retry after filtering. Exiting.")
            exit(0)
        
        total_tasks = len(failed_tasks)
        unique_vulns = set(t[0]["vul_id"] for t in failed_tasks)
        unique_ablations = set(t[1] for t in failed_tasks)
        
        print(f"Failed Tasks to Retry: {total_tasks}")
        print(f"Unique Vulnerabilities: {len(unique_vulns)}")
        print(f"Ablation Versions: {sorted(unique_ablations)}")
        print(f"Max Workers: {args.workers}")
        print(f"Force Re-run: {args.force}")
        print("=" * 70 + "\n")
        
        # Force enable force mode to overwrite previously failed outputs
        args.force = True
        
        start_time = time()
        all_results = []
        
        # Use parallel strategy to directly distribute failed tasks
        print("[*] Retrying failed tasks with PARALLEL strategy...")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_vuln_ablation, item, args, abl): (item["vul_id"], abl)
                      for item, abl in failed_tasks}
            
            print(f"[*] Submitted {len(futures)} retry tasks to executor...\n")
            
            completed_tasks = 0
            for future in concurrent.futures.as_completed(futures):
                vul_id, ablation = futures[future]
                try:
                    result = future.result()
                    all_results.append(result)
                    completed_tasks += 1
                    
                    status = "✓" if result[2] else "✗"
                    print(f"[{completed_tasks}/{total_tasks}] {status} {vul_id} | {ablation}")
                    
                except Exception as exc:
                    print(f"!!! [Exception] {vul_id} | {ablation}: {exc}")
                    all_results.append((vul_id, ablation, False, str(exc)))
        
        end_time = time()
        elapsed = end_time - start_time
        
        # Save retry results summary
        retry_summary_path = os.path.join(args.output_dir, "ablation_retry_summary.json")
        retry_summary = save_summary(all_results, retry_summary_path)

        
        # Print results
        print("\n" + "=" * 70)
        print("Retry Failed Tasks Completed")
        print("=" * 70)
        print(f"Total Time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
        print(f"Retried Tasks: {retry_summary['total_tasks']}")
        print(f"Now Successful: {retry_summary['successful']}")
        print(f"Still Failed: {retry_summary['failed']}")
        
        if retry_summary['failed'] > 0:
            print(f"\nStill failing tasks saved to: {retry_summary_path}")
            print("\nStill failing vulnerabilities:")
            for task in retry_summary.get("failed_tasks", []):
                print(f"  - {task['vul_id']} ({task['ablation']}): {task['error'][:80]}...")
        else:
            print("\n✓ All previously failed tasks now succeeded!")
        
        print("=" * 70)
        exit(0)
    
    # ==========================================
    # Normal Run Mode
    # ==========================================
    
    # Read vulnerability list
    df = pd.read_csv(args.csv)
    df = df.drop_duplicates(subset=["repo", "vul_id"])
    all_records = df.to_dict(orient="records")
    
    # Filter specific vulnerabilities
    if args.vul_id:
        all_records = [item for item in all_records if item.get("vul_id") == args.vul_id]
    
    # Limit number of processed items
    if args.limit:
        all_records = all_records[:args.limit]
    
    total_vulns = len(all_records)
    total_tasks = total_vulns * len(args.versions)
    
    print("=" * 70)
    print("Batch Ablation Experiment")
    print("=" * 70)
    print(f"Vulnerabilities: {total_vulns}")
    print(f"Ablation Versions: {args.versions}")
    print(f"Total Tasks: {total_tasks}")
    print(f"Max Workers: {args.workers}")
    print(f"Strategy: {args.strategy}")
    print("=" * 70 + "\n")
    
    start_time = time()
    all_results = []
    
    if args.strategy == "sequential":
        # Strategy 1: Each worker processes all versions of one vulnerability (Recommended)
        # Pros: Can reuse checkpoints from Phase 1 and Phase 3
        print("[*] Using SEQUENTIAL strategy: each worker processes all versions of one vulnerability")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_vuln_all_ablations, item, args): item["vul_id"] 
                      for item in all_records}
            
            print(f"[*] Submitted {len(futures)} vulnerability tasks to executor...\n")
            
            completed_vulns = 0
            for future in concurrent.futures.as_completed(futures):
                vul_id = futures[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                    completed_vulns += 1
                    
                    # Calculate success rate for this vulnerability
                    success_count = sum(1 for r in results if r[2])
                    total_count = len(results)
                    print(f"[{completed_vulns}/{total_vulns}] Completed: {vul_id} ({success_count}/{total_count} versions successful)")
                    
                except Exception as exc:
                    print(f"!!! [Exception] {vul_id} generated an exception: {exc}")
    
    else:  # parallel
        # Strategy 2: All tasks (vulnerability x version) distributed in parallel
        # Pros: Maximize parallelism
        # Cons: Cannot reuse checkpoints across versions
        print("[*] Using PARALLEL strategy: distributing all tasks to workers")
        
        # Generate all tasks
        tasks = []
        for item in all_records:
            for ablation_type in args.versions:
                tasks.append((item, ablation_type))
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_vuln_ablation, item, args, abl): (item["vul_id"], abl)
                      for item, abl in tasks}
            
            print(f"[*] Submitted {len(futures)} individual tasks to executor...\n")
            
            completed_tasks = 0
            for future in concurrent.futures.as_completed(futures):
                vul_id, ablation = futures[future]
                try:
                    result = future.result()
                    all_results.append(result)
                    completed_tasks += 1
                    
                    status = "✓" if result[2] else "✗"
                    print(f"[{completed_tasks}/{total_tasks}] {status} {vul_id} | {ablation}")
                    
                except Exception as exc:
                    print(f"!!! [Exception] {vul_id} | {ablation}: {exc}")
    
    end_time = time()
    elapsed = end_time - start_time
    
    # Save summary
    summary_path = os.path.join(args.output_dir, "ablation_batch_summary.json")
    summary = save_summary(all_results, summary_path)
    
    # Print results
    print("\n" + "=" * 70)
    print("Batch Ablation Experiment Completed")
    print("=" * 70)
    print(f"Total Time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    print(f"Total Tasks: {summary['total_tasks']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"\nResults by Ablation Version:")
    for ablation, stats in summary["by_ablation"].items():
        print(f"  {ablation:15s}: {stats['successful']:3d}/{stats['total']:3d} successful")
    
    if summary['failed'] > 0:
        print(f"\nFailed tasks saved to: {summary_path}")
    
    print("=" * 70)
