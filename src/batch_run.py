import os
import argparse
import pandas as pd
from dotenv import load_dotenv
import concurrent.futures
from time import time
from run import build_pipeline as build_single_vuln_pipeline, setup_directories

from core.configs import OUTPUT_DIR_PATH, REPO_DIR_PATH

# ==========================================
# 1. 任务处理函数
# ==========================================

def process_vuln(item, args):
    """
    单个漏洞处理任务，供进程池调用
    注意：app 对象无法跨进程传递，必须在子进程内部初始化。
    """
    vul_id = item["vul_id"]
    repo_name = item["repo"]
    commit_sha = item["fixed_commit_sha"]

    try:
        # 在子进程内初始化 app，避免 PickleError
        # 假设 build_single_vuln_pipeline 开销可接受
        app = build_single_vuln_pipeline()

        # 路径生成
        ckpt_dir, res_dir = setup_directories(args.output_dir, repo_name, vul_id)
        full_repo_path = os.path.join(args.repo_path, repo_name)
        
        # 构造子图状态
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
        # 这里的 recursion_limit 是针对单个漏洞处理的步数
        app.invoke(vuln_state, {"recursion_limit": 10000})
        print(f">>> [Done] {vul_id} Finished.")
        return vul_id, True
    except Exception as e:
        print(f"!!! [Error] Processing {vul_id}: {e}")
        return vul_id, False

# ==========================================
# 2. 运行入口
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--csv", required=True)
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
    df = df.drop_duplicates(subset=["vul_id"])
    all_records = df.to_dict(orient="records")
    
    if args.vul_id:
        all_records = [item for item in all_records if item.get("vul_id") == args.vul_id]
    
    total = len(all_records)
    max_workers = args.workers
    
    print(f"=== Parallel Pipeline Started: {total} items | Max Workers: {max_workers} ===")
    start_time = time()
    # 使用 ProcessPoolExecutor 进行多进程并行
    # Python 的 GIL 会限制 ThreadPoolExecutor 在 CPU 密集型任务上的并行能力
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # submit all tasks
        futures = {executor.submit(process_vuln, item, args): item["vul_id"] for item in all_records}
        
        print(f"[*] Submitted {len(futures)} tasks to executor...")
        
        # 处理结果
        completed_count = 0
        for future in concurrent.futures.as_completed(futures):
            vul_id = futures[future]
            try:
                vid, success = future.result()
                completed_count += 1
                status = "Success" if success else "Failed"
                print(f"[{completed_count}/{total}] Completed: {vid} ({status})")
            except Exception as exc:
                print(f"!!! [Exception] {vul_id} generated an exception: {exc}")

    print("\n=== All Tasks Completed ===")
    end_time = time()
    print(f"Total time: {end_time - start_time} seconds")