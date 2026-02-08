import os
import argparse
import pandas as pd
from dotenv import load_dotenv
import concurrent.futures
from time import time
from run import build_pipeline as build_single_vuln_pipeline, setup_directories

from core.configs import OUTPUT_DIR_PATH, REPO_DIR_PATH

# ==========================================
# 1. Task Processing Function
# ==========================================

def process_vuln(item, args):
    """
    Single vulnerability processing task, called by the process pool.
    Note: The app object cannot be passed across processes and must be initialized inside the child process.
    """
    vul_id = item["vul_id"]
    repo_name = item["repo"]
    commit_sha = item["fixed_commit_sha"]

    try:
        # Initialize app within the child process to avoid PickleError
        # Assuming build_single_vuln_pipeline overhead is acceptable
        app = build_single_vuln_pipeline()

        # Path generation
        ckpt_dir, res_dir = setup_directories(args.output_dir, repo_name, vul_id)
        full_repo_path = os.path.join(args.repo_path, repo_name)
        
        # Construct sub-graph state
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
            "lang": "c",
            "atomic_patches": [],
            "grouped_patches": [],
            "analyzed_features": [],
            "search_candidates": [],
            "final_findings": []
        }
        
        print(f">>> [Start] Processing {vul_id}...")
        # recursion_limit here is for the steps of a single vulnerability processing
        state_snapshot = app.invoke(vuln_state, {"recursion_limit": 10000})
        print(f">>> [Done] {vul_id} Finished.")
        
        findings = state_snapshot.get("final_findings", [])
        return vul_id, True, findings
    except Exception as e:
        print(f"!!! [Error] Processing {vul_id}: {e}")
        return vul_id, False, []

# ==========================================
# 2. Execution Entry Point
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--csv", default="inputs/0day_vul_list.csv", help="CSV file with vulnerability list")
    parser.add_argument("-s", "--start", type=int, default=1)
    parser.add_argument("-e", "--end", type=int, default=4)
    parser.add_argument("-r", "--repo_path", default=REPO_DIR_PATH, help="Base path containing cloned repos")
    parser.add_argument("-o", "--output_dir", default=OUTPUT_DIR_PATH, help="Base path for outputs")
    parser.add_argument("-m", "--mode", default="repo", choices=["repo", "benchmark"], help="Operation mode")
    parser.add_argument("-b", "--workers", type=int, default=8, help="Max concurrent workers (previously batch_size)")
    parser.add_argument("-f", "--force", action="store_true", help="Force re-processing even if outputs exist")
    parser.add_argument("-v", "--vul_id", type=str, default=None, help="Only process a specific vulnerability ID")
    args = parser.parse_args()
    
    load_dotenv()
    
    df = pd.read_csv(args.csv)
    df = df.drop_duplicates(subset=["repo", "vul_id"])
    all_records = df.to_dict(orient="records")
    
    if args.vul_id:
        all_records = [item for item in all_records if item.get("vul_id") == args.vul_id]
    
    total = len(all_records)
    max_workers = args.workers
    
    print(f"=== Parallel Pipeline Started: {total} items | Max Workers: {max_workers} ===")
    start_time = time()
    # Use ProcessPoolExecutor for multi-process parallelism
    # Python's GIL limits ThreadPoolExecutor's parallelism on CPU-bound tasks
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # submit all tasks
        futures = {executor.submit(process_vuln, item, args): item["vul_id"] for item in all_records}
        
        print(f"[*] Submitted {len(futures)} tasks to executor...")
        
        # Process results
        completed_count = 0
        collected_findings = []

        for future in concurrent.futures.as_completed(futures):
            vul_id = futures[future]
            try:
                vid, success, findings = future.result()
                completed_count += 1
                status = "Success" if success else "Failed"
                print(f"[{completed_count}/{total}] Completed: {vid} ({status})")
                
                if success and findings:
                    collected_findings.extend(findings)

            except Exception as exc:
                print(f"!!! [Exception] {vul_id} generated an exception: {exc}")

    print("\n=== All Tasks Completed ===")

    # --- Generate Summary CSV ---
    if collected_findings:
        print(f"[*] Post-processing {len(collected_findings)} findings...")
        
        # Build lookup for repo/commit info
        # all_records is list of dicts
        meta_lookup = {item["vul_id"]: item for item in all_records}
        
        csv_rows = []
        for finding in collected_findings:
            # finding is likely a Pydantic model (VulnerabilityFinding)
            # Handle both object and dict just in case
            if hasattr(finding, "vul_id"):
                 fid = finding.vul_id
                 patch_file = finding.patch_file
                 patch_func = finding.patch_func
                 target_file = finding.target_file
                 target_func = finding.target_func
            else:
                 fid = finding.get("vul_id")
                 patch_file = finding.get("patch_file")
                 patch_func = finding.get("patch_func")
                 target_file = finding.get("target_file")
                 target_func = finding.get("target_func")
            
            # Retrieve metadata from input CSV records
            meta = meta_lookup.get(fid, {})
            repo = meta.get("repo", "UNKNOWN")
            sha = meta.get("fixed_commit_sha", "UNKNOWN")
            
            csv_rows.append({
                "vul_id": fid,
                "repo": repo,
                "fixed_commit_sha": sha,
                "patch_file_path": patch_file,
                "patch_func_name": patch_func,
                "target_file_path": target_file,
                "target_func_name": target_func
            })
            
        summary_df = pd.DataFrame(csv_rows)
        # Select and reorder columns
        cols = ["vul_id", "repo", "fixed_commit_sha", "patch_file_path", "patch_func_name", "target_file_path", "target_func_name"]
        summary_df = summary_df[cols]
        
        output_csv_path = os.path.join(args.output_dir, "batch_findings.csv")
        
        if os.path.exists(output_csv_path):
            try:
                existing_df = pd.read_csv(output_csv_path)
                combined_df = pd.concat([existing_df, summary_df])
                combined_df = combined_df.drop_duplicates()
                combined_df.to_csv(output_csv_path, index=False)
                print(f"[Done] Findings appended to: {output_csv_path}")
            except Exception as e:
                print(f"!!! [Error] Failed to append to CSV: {e}")
        else:
            summary_df.to_csv(output_csv_path, index=False)
            print(f"[Done] Findings saved to: {output_csv_path}")
    else:
        print("[-] No findings collected.")
    end_time = time()
    print(f"Total time: {end_time - start_time} seconds")