from core.indexer import GlobalSymbolIndexer, BenchmarkSymbolIndexer
import pandas as pd
from core.configs import REPO_DIR_PATH, REPO_LIST_CSV
import os
from core.utils import write_text, write_json
from time import time
import argparse
import subprocess

def clone_and_checkout(repo_name, version=None):
    repo_path = os.path.join(REPO_DIR_PATH, repo_name)
    
    if not os.path.exists(repo_path):
        print(f"[*] Repository {repo_name} not found locally. Cloning...")
        try:
            # Ensure the directory structure (owner/repo) is respected
            parent_dir = os.path.dirname(repo_path)
            os.makedirs(parent_dir, exist_ok=True)
            
            git_url = f"https://github.com/{repo_name}.git"
            subprocess.run(["git", "clone", git_url, repo_path], check=True)
        except subprocess.CalledProcessError as e:
            print(f"[!] Error cloning {repo_name}: {e}")
            return None

    if version:
        print(f"[*] Checking out {repo_name} to {version}...")
        try:
            subprocess.run(["git", "fetch", "--all", "--tags"], cwd=repo_path, check=True, capture_output=True)
            subprocess.run(["git", "checkout", version], cwd=repo_path, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"[!] Error checking out {repo_name} to {version}: {e}")
    
    return repo_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', default='0day', choices=['0day', '1day'], help='Indexing mode')
    parser.add_argument('-c', '--csv', default=REPO_LIST_CSV, help='CSV file with repository list')
    parser.add_argument('-r', '--repo_name', default=None, help='Specific repository name to index')
    parser.add_argument('-v', '--version', default=None, help='Version of the index')
    args = parser.parse_args()
    if args.mode == '0day':
        repo_list = []
        if args.repo_name:
            repo_path = clone_and_checkout(args.repo_name, args.version)
            if repo_path:
                repo_list.append(repo_path)
        else:
            df = pd.read_csv(args.csv)
            repo_list = []
            for _, row in df.iterrows():
                # Support both 'repository' (new format) and 'repo' (old format)
                repo_name = row.get("repository", row.get("repo"))
                version = row.get("version")
                
                # Handle NaN/None in version
                if pd.isna(version):
                    version = None
                else:
                    version = str(version)

                repo_path = clone_and_checkout(repo_name, version)
                if repo_path and repo_path not in repo_list:
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

    if args.mode == '1day':
    # 1. Build Benchmark Index
        print("[*] Checking Benchmark Index...")
        indexer = BenchmarkSymbolIndexer()
        indexer.load_index()
        print("[*] Benchmark index ready.")