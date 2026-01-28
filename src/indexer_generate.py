from core.indexer import GlobalSymbolIndexer, BenchmarkSymbolIndexer
import pandas as pd
from core.configs import REPO_DIR_PATH
import os
from core.utils import write_text, write_json
from time import time

if __name__ == "__main__":
    # 1. Build Benchmark Index
    print("[*] Checking Benchmark Index...")
    indexer = BenchmarkSymbolIndexer()
    indexer.load_index()
    print("[*] Benchmark index ready.")
    
    # 2. Get Repo List
    df = pd.read_csv("inputs/repo_list.csv")
    repo_list = []
    for _, row in df.iterrows():
        repo = row["repo"]
        repo_path = os.path.join(REPO_DIR_PATH, repo)
        if repo_path not in repo_list:
            repo_list.append(repo_path)
    print(f"[*] Total repos to index: {len(repo_list)}")
    
    # 3. Sequential Build
    # Strategy: Process repos one by one.
    # Since GlobalSymbolIndexer.build_index() uses ProcessPoolExecutor internally to utilize all CPU cores,
    # running multiple repos in parallel would cause excessive context switching and memory contention.
    # Sequential processing ensures each repo gets full system resources for maximum speed.
    
    res_dict = {}
    print(f"[*] Starting sequential build for {len(repo_list)} repos...")
    
    for repo_path in repo_list:
        try:
            print(f"[*] Start indexing repo: {repo_path}")
            start = time()
            indexer = GlobalSymbolIndexer(repo_path)
            indexer.load_index()
            end = time()
            duration = end - start
            res_dict[repo_path] = duration
            print(f"[*] Finished {repo_path} in {duration:.2f}s")
        except Exception as e:
            print(f"[!] Error indexing {repo_path}: {e}")
        
    write_json(res_dict, "0day_repo_index_time.json")
    print("[*] All done.")