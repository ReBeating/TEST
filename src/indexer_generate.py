from core.indexer import GlobalSymbolIndexer, BenchmarkSymbolIndexer
import pandas as pd
from core.configs import REPO_DIR_PATH, REPO_LIST_CSV
import os
from core.utils import write_text, write_json
from time import time
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', default='0day', choices=['0day', '1day'], help='Indexing mode')
    parser.add_argument('-c', '--csv', default=REPO_LIST_CSV, help='CSV file with repository list')
    parser.add_argument('-r', '--repo_name', default=None, help='Specific repository name to index')
    args = parser.parse_args()
    if args.mode == '1day':
    # 1. Build Benchmark Index
        print("[*] Checking Benchmark Index...")
        indexer = BenchmarkSymbolIndexer()
        indexer.load_index()
        print("[*] Benchmark index ready.")
    if args.mode == '0day':
        if args.repo_name:
            repo_path = os.path.join(REPO_DIR_PATH, args.repo_name)
            if not os.path.exists(repo_path):
                raise ValueError(f"Repository path does not exist: {repo_path}")
            repo_list = [repo_path]
        else:
        df = pd.read_csv(args.csv)
        repo_list = []
        for _, row in df.iterrows():
            repo = row["repo"]
            repo_path = os.path.join(REPO_DIR_PATH, repo)
            if not os.path.exists(repo_path):
                raise ValueError(f"Repository path does not exist: {repo_path}")
            if repo_path not in repo_list:
                repo_list.append(repo_path)
    print(f"[*] Total repos to index: {len(repo_list)}")
    
    res_dict = {}
    print(f"[*] Starting build for {len(repo_list)} repos...")
    
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
        
    # write_json(res_dict, "0day_repo_index_time.json")
    print("[*] All done.")