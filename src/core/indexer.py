import os
import json
import subprocess
import sqlite3
import hashlib
import threading
import re
from typing import List, Dict, Tuple
from concurrent.futures import as_completed, ProcessPoolExecutor
from tqdm import tqdm
import time
from core.parser import remove_comments, indent_code, CtagsParser
import psutil
from core.configs import BENCHMARK_DICT_PATH, DATABASE_DIR_PATH, BENCHMARK_VUL_PATH
from core.utils import read_json, read_text
import pandas as pd
from rapidfuzz import fuzz

# ==========================================
# Tokenizer Helper
# ==========================================
def tokenize_code(code: str) -> List[str]:
    """
    Extract high-value tokens from code for inverted indexing.
    Filters:
    - Length >= 3
    - Must contain letters
    - Exclude C/C++ keywords and common stopwords
    - Split CamelCase and snake_case
    """
    # 1. Remove comments (simple regex, assuming code is already cleaned or raw)
    # Note: remove_comments_from_code is in matcher.py, here we do a lightweight pass or assume input is clean enough
    # For indexing, raw code usually contains comments, but we can just ignore them via regex logic
    
    # 2. Split by non-word characters
    raw_tokens = re.findall(r'\w+', code)
    
    final_tokens = set()
    
    STOPWORDS = {
        'if', 'else', 'for', 'while', 'return', 'switch', 'case', 'break', 'continue', 'goto', 'default', 
        'sizeof', 'struct', 'union', 'enum', 'typedef', 'static', 'const', 'volatile', 'extern', 'void', 
        'char', 'short', 'int', 'long', 'float', 'double', 'signed', 'unsigned', 'bool', 'true', 'false',
        'null', 'auto', 'register', 'inline', 'restrict', 'asm', 'class', 'namespace', 'template', 'typename',
        'public', 'private', 'protected', 'virtual', 'friend', 'this', 'operator', 'new', 'delete', 'try', 'catch', 'throw',
        'ret', 'err', 'rc', 'val', 'len', 'size', 'data', 'buf', 'tmp', 'result', 'status', 'ctx', 'priv', 'info', 'dev', 'skb'
    }

    for token in raw_tokens:
        # Filter 1: Length and Content
        if len(token) < 3: continue
        if not re.search(r'[a-zA-Z]', token): continue # Must have at least one letter (exclude pure numbers)
        
        # Filter 2: Stopwords
        if token.lower() in STOPWORDS: continue
        
        # Add original token
        final_tokens.add(token)
        
        # 3. Sub-tokenization (CamelCase & snake_case)
        # snake_case
        parts = token.split('_')
        if len(parts) > 1:
            for p in parts:
                if len(p) >= 3 and re.search(r'[a-zA-Z]', p) and p.lower() not in STOPWORDS:
                    final_tokens.add(p)
        
        # CamelCase (simple heuristic)
        # e.g. XMLParser -> XML, Parser
        camel_parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|[A-Z]+', token)
        if len(camel_parts) > 1:
            for p in camel_parts:
                if len(p) >= 3 and re.search(r'[a-zA-Z]', p) and p.lower() not in STOPWORDS:
                    final_tokens.add(p)
                    
    return list(final_tokens)

def chunked_iterable(iterable, size):
    """将列表切割为固定大小的块"""
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

def find_source_files(repo_path: str) -> List[str]:
    """
    快速查找所有源文件。
    优先使用 git ls-files (极快)，失败则回退到 os.walk
    """
    extensions = {'.c', '.h', '.cpp', '.hpp', '.cc', '.cc', '.cxx'} # 根据需要添加
    files = []
    
    # 方法1: 尝试 git (Linux内核通常是git仓库)
    try:
        cmd = ["git", "ls-files"]
        # 限制只查找相关后缀，或者全部拿出来后再过滤
        # git ls-files 输出的是相对路径
        process = subprocess.Popen(
            cmd, cwd=repo_path, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
        )
        for line in process.stdout:
            f = line.strip()
            if os.path.splitext(f)[1] in extensions:
                files.append(f)
        if process.wait() == 0 and files:
            print(f"[*] Found {len(files)} files using git.")
            return files
    except Exception:
        pass

    # 方法2: os.walk 回退
    print("[*] 'git ls-files' failed or not a git repo. Falling back to os.walk...")
    for root, dirs, filenames in os.walk(repo_path):
        # 排除常见非代码目录
        dirs[:] = [d for d in dirs if d not in ('.git', 'Documentation', 'scripts', 'tools')]
        
        for name in filenames:
            if os.path.splitext(name)[1] in extensions:
                rel_path = os.path.relpath(os.path.join(root, name), repo_path)
                files.append(rel_path)
    
    print(f"[*] Found {len(files)} files using os.walk.")
    return files

def process_file_chunk_robust(repo_path: str, file_chunk: List[str], chunk_id: int) -> Tuple[List[Tuple], str | None]:
    """
    [Worker 进程入口]
    增强了错误捕获和 ctags 进程状态检查。
    返回: (结果列表, 错误信息字符串 or None)
    """
    results = []
    if not file_chunk:
        return (results, None)

    cmd = [
        "ctags",
        "--languages=C,C++",
        "--output-format=json",
        "--fields=+ne",
        "--c-kinds=+f-p", # 只看函数定义，排除原型
        "--extras=+q",
        "-f", "-",        # 输出到 stdout
        "-L", "-"         # 从 stdin 读取文件列表
    ]
    
    # ctags 进程的 PID，用于之后检查
    ctags_pid = None

    try:
        process = subprocess.Popen(
            cmd, cwd=repo_path,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, # 捕获 stderr
            text=True, encoding='utf-8', errors='replace',
            bufsize=1024*1024
        )
        ctags_pid = process.pid # 获取 ctags 进程 ID

        # 使用 communicate 设置超时，例如 5 分钟
        # 注意：ctags 解析 Linux 内核可能非常耗时，需要根据实际情况调整
        # 如果是某个超大文件，可能需要更长，甚至取消超时（但风险大）
        # 更好的方法是：不直接用 communicate，而是轮询 stdout 和 stderr
        # 但这里为了简化，我们先尝试用 timeout
        try:
            input_str = "\n".join(file_chunk)
            stdout_data, stderr_data = process.communicate(input=input_str, timeout=300) # 5分钟超时
            
            if process.returncode != 0:
                error_msg = f"ctags exited with code {process.returncode}. Stderr:\n{stderr_data.strip()}"
                print(f"[WARN] Chunk {chunk_id} ctags error: {error_msg}")
                return (results, error_msg) # 返回错误信息

            # --- 解析 stdout_data ---
            file_tags_map = {}
            for line in stdout_data.splitlines():
                try:
                    tag = json.loads(line)
                    f_path = tag.get("path")
                    if not f_path: continue
                    if f_path not in file_tags_map:
                        file_tags_map[f_path] = []
                    file_tags_map[f_path].append(tag)
                except ValueError:
                    continue

            for f_path, tags in file_tags_map.items():
                full_path = os.path.join(repo_path, f_path)
                if not os.path.exists(full_path): continue
                
                try:
                    with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                        lines = f.readlines()
                    file_len = len(lines)
                    
                    for tag in tags:
                        name = tag.get("name")
                        start = tag.get("line", 0)
                        end = tag.get("end", start)
                        kind = tag.get("kind", "")
                        
                        idx_start = max(0, start - 1)
                        idx_end = end
                        
                        if kind == 'function' and idx_start > 0:
                            prev_line = lines[idx_start - 1].strip()
                            if prev_line and not prev_line.endswith((';', '}')) and not prev_line.startswith(('#', '//')):
                                 idx_start -= 1

                        if idx_start < file_len:
                            if start == end:
                                line_content = lines[idx_start].strip()
                                if line_content.endswith(';'): continue

                            real_end = min(idx_end, file_len)
                            code_content = "".join(lines[idx_start:real_end])
                            
                            if code_content:
                                # [Optimization] Tokenize in worker to offload main thread
                                tokens = []
                                if kind == 'function':
                                    tokens = tokenize_code(code_content)
                                    tokens.extend(tokenize_code(name))
                                    tokens = list(set(tokens))
                                
                                results.append((name, f_path, start, end, kind, code_content, tokens))
                                
                except Exception:
                    # 文件读取或处理单个 tag 出错，忽略该 tag
                    pass
            
            return (results, None) # 成功

        except subprocess.TimeoutExpired:
            error_msg = f"ctags process timed out after 300s. Stderr:\n{stderr_data.strip()}"
            print(f"[ERROR] Chunk {chunk_id} timeout: {error_msg}")
            # 尝试终止 ctags 进程
            if ctags_pid:
                try:
                    p = psutil.Process(ctags_pid)
                    p.terminate()
                    p.wait(timeout=5) # 等待终止
                except (psutil.NoSuchProcess, psutil.AccessDenied, TimeoutError):
                    pass # 忽略终止错误
            return ([], error_msg) # 返回超时错误
        
        except Exception as e:
            # 其他 Python 异常
            error_msg = f"Unexpected error in worker process: {e}. Stderr:\n{stderr_data.strip()}"
            print(f"[ERROR] Chunk {chunk_id} worker error: {error_msg}")
            return ([], error_msg) # 返回通用错误

    except Exception as e:
        # Popen 阶段出错
        error_msg = f"Failed to start ctags process: {e}"
        print(f"[ERROR] Chunk {chunk_id} setup error: {error_msg}")
        return ([], error_msg)

class GlobalSymbolIndexer:
    def __init__(self, repo_path: str, cache_dir: str = DATABASE_DIR_PATH):
        self.repo_path = repo_path
        self.cache_dir = cache_dir
        
        path_hash = hashlib.md5(repo_path.encode()).hexdigest()[:8]
        repo_name = os.path.basename(repo_path.rstrip(os.sep))
        self.db_file = os.path.join(cache_dir, f"idx_{repo_name}_{path_hash}.db")
        
        self._conn = None

    def find_file_paths(self, filename: str, version: str = None) -> List[str]:
        """
        Find files in the repository ending with the given filename.
        Case-insensitive match recommended or exact match.
        """
        matches = []
        try:
            # Use git ls-files if possible for speed
            cmd = ["git", "ls-files"]
            proc = subprocess.run(cmd, cwd=self.repo_path, capture_output=True, text=True, check=True)
            all_files = proc.stdout.splitlines()
            
            for f in all_files:
                if f.endswith(f"/{filename}") or f == filename:
                    matches.append(f)
        except Exception:
            # Fallback to os.walk
            for root, dirs, files in os.walk(self.repo_path):
                if filename in files:
                    rel_dir = os.path.relpath(root, self.repo_path)
                    if rel_dir == ".":
                        matches.append(filename)
                    else:
                        matches.append(os.path.join(rel_dir, filename))
        
        return matches

    def _get_conn(self):
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_file, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL;") 
            self._conn.execute("PRAGMA synchronous=NORMAL;")
        return self._conn

    def load_index(self, force_rebuild: bool = False):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        should_build = force_rebuild
        if not should_build:
            if not os.path.exists(self.db_file):
                should_build = True
            else:
                try:
                    conn = self._get_conn()
                    cursor = conn.cursor()
                    cursor.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='symbols'")
                    if cursor.fetchone()[0] == 0:
                        should_build = True
                    else:
                        cursor.execute("SELECT code FROM symbols LIMIT 1")
                        # print(f"[*] Database found: {self.db_file}")
                except Exception:
                    should_build = True

        if should_build:
            self.build_index()

    def build_index(self):
        start_time = time.time()
        print(f"[*] Building global symbol index (Parallel Map-Reduce Robust) for {self.repo_path} ...")
        
        # 1. 准备数据库
        if not os.path.exists(self.cache_dir): os.makedirs(self.cache_dir)
        if os.path.exists(self.db_file): os.remove(self.db_file)
        
        conn = sqlite3.connect(self.db_file)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        cursor = conn.cursor()
        
        # Create symbols table
        cursor.execute("""
            CREATE TABLE symbols (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                path TEXT NOT NULL,
                start_line INTEGER,
                end_line INTEGER,
                kind TEXT,
                code TEXT
            )
        """)
        
        # [New] Create inverted index tables
        # [Optimized] Remove UNIQUE constraint and Index during build for speed
        cursor.execute("""
            CREATE TABLE tokens (
                id INTEGER PRIMARY KEY,
                text TEXT
            )
        """)
        # cursor.execute("CREATE INDEX idx_token_text ON tokens(text);")  <-- Deferred
        
        # [Optimized] Create Heap table first for fast insertion, indexes later
        cursor.execute("""
            CREATE TABLE symbol_tokens (
                token_id INTEGER,
                symbol_id INTEGER
            )
        """)
        # cursor.execute("CREATE INDEX idx_sym_tok_sym ON symbol_tokens(symbol_id);")
        
        conn.commit()

        # 2. 获取文件列表
        all_files = find_source_files(self.repo_path)
        if not all_files:
            print("[!] No source files found.")
            return

        # 3. 过滤过大的文件
        MAX_FILE_SIZE = 30 * 1024 * 1024  # 30MB
        valid_files = []
        skipped_count = 0
        
        print(f"[*] Filtering files larger than {MAX_FILE_SIZE/1024/1024:.2f}MB ...")
        
        for f_path in all_files:
            full_path = os.path.join(self.repo_path, f_path)
            try:
                # 检查文件大小
                size = os.path.getsize(full_path)
                if size > MAX_FILE_SIZE:
                    # 打印一下跳过了哪些大文件，通常能看到 amdgpu 等 generated headers
                    print(f"    [SKIP] Too large ({size/1024/1024:.2f}MB): {f_path}") 
                    skipped_count += 1
                    continue
                valid_files.append(f_path)
            except OSError:
                # 文件可能被删除了或由软链接导致的问题
                continue
                
        print(f"[*] Filtered out {skipped_count} huge files. Remaining: {len(valid_files)}")
        all_files = valid_files # 更新列表

        CHUNK_SIZE = 10
        chunks = list(chunked_iterable(all_files, CHUNK_SIZE))
        print(f"[*] Split into {len(chunks)} chunks (size {CHUNK_SIZE}). Launching workers...")

        total_symbols = 0
        failed_chunks = [] # 记录失败的 chunk ID
        
        with ProcessPoolExecutor(max_workers=16) as executor:
            # 提交任务时带上 chunk_id，用于错误报告
            future_to_chunk_id = {
                executor.submit(process_file_chunk_robust, self.repo_path, chunk, i): i 
                for i, chunk in enumerate(chunks)
            }
            
            batch_buffer = []
            BATCH_WRITE_SIZE = 5000
            
            # 修改 tqdm 迭代方式，处理 f.result() 的返回值
            pbar = tqdm(as_completed(future_to_chunk_id), total=len(chunks), unit="chunk", desc="Indexing")
            
            processed_count = 0
            
            # Cache for token IDs to reduce DB lookups
            # token_text -> token_id
            token_cache = {} 
            # Manual ID tracking
            # Using a list [val] to allow modification in closure if needed, but here simple var is tricky in closure without nonlocal
            # Better use a class attribute or a dictionary to hold state
            state = {"next_token_id": 1}

            def flush_buffer():
                if not batch_buffer: return
                cursor.execute("BEGIN TRANSACTION;")
                try:
                    # 1. Pre-process: Extract all tokens and prepare symbol data
                    prepared_symbols = [] 
                    all_batch_tokens = set()

                    for res in batch_buffer:
                        if len(res) == 7:
                            name, path, start, end, kind, code, tokens = res
                        else:
                            name, path, start, end, kind, code = res
                            tokens = []
                            if kind == 'function':
                                tokens = tokenize_code(code)
                                tokens.extend(tokenize_code(name))
                                tokens = list(set(tokens))
                        
                        prepared_symbols.append((name, path, start, end, kind, code, tokens))
                        all_batch_tokens.update(tokens)
                    
                    # 2. Batch handle Tokens
                    # [Optimized] Assign IDs in Python, skip DB lookups entirely
                    unknown_tokens = [t for t in all_batch_tokens if t not in token_cache]
                    
                    if unknown_tokens:
                        tokens_to_insert = []
                        start_id = state["next_token_id"]
                        
                        for i, t in enumerate(unknown_tokens):
                            tid = start_id + i
                            token_cache[t] = tid
                            tokens_to_insert.append((tid, t))
                        
                        state["next_token_id"] += len(unknown_tokens)
                        
                        if tokens_to_insert:
                             # Raw insert with explicit ID
                            cursor.executemany("INSERT INTO tokens (id, text) VALUES (?, ?)", tokens_to_insert)

                    # 3. Insert Symbols and build Relations (Batch with RETURNING id)
                    symbol_tokens_batch = []
                    
                    # SQLite default variable limit is usually 999. 
                    # 6 params per row => max ~166 rows. We use 150 to be safe.
                    SYM_BATCH_SIZE = 150
                    
                    for i in range(0, len(prepared_symbols), SYM_BATCH_SIZE):
                        chunk = prepared_symbols[i:i + SYM_BATCH_SIZE]
                        
                        try:
                            # Try optimized batch insert using RETURNING clause (SQLite >= 3.35)
                            placeholders = ",".join(["(?, ?, ?, ?, ?, ?)"] * len(chunk))
                            sql = f"INSERT INTO symbols (name, path, start_line, end_line, kind, code) VALUES {placeholders} RETURNING id"
                            
                            params = []
                            for item in chunk:
                                params.extend(item[:6])
                            
                            cursor.execute(sql, params)
                            rows = cursor.fetchall()
                            
                            # Map returned IDs to tokens
                            for idx, row in enumerate(rows):
                                symbol_id = row[0]
                                tokens = chunk[idx][6]
                                if tokens:
                                    for t in tokens:
                                        if t in token_cache:
                                            symbol_tokens_batch.append((token_cache[t], symbol_id))
                                            
                        except sqlite3.OperationalError:
                            # Fallback to row-by-row if RETURNING is not supported (older SQLite)
                            for item in chunk:
                                cursor.execute(
                                    "INSERT INTO symbols (name, path, start_line, end_line, kind, code) VALUES (?, ?, ?, ?, ?, ?)",
                                    item[:6]
                                )
                                symbol_id = cursor.lastrowid
                                tokens = item[6]
                                if tokens:
                                    for t in tokens:
                                        if t in token_cache:
                                            symbol_tokens_batch.append((token_cache[t], symbol_id))
                    
                    # 4. Batch insert Relations
                    if symbol_tokens_batch:
                        # Use simple INSERT since we dedup in Python and have no PK constraint anymore
                        cursor.executemany("INSERT INTO symbol_tokens (token_id, symbol_id) VALUES (?, ?)", symbol_tokens_batch)

                    conn.commit()
                except Exception as e:
                    print(f"[ERROR] Flush buffer failed: {e}")
                    conn.rollback()
                finally:
                    batch_buffer.clear()
            
            for future in pbar:
                chunk_id = future_to_chunk_id[future]
                try:
                    results, error_msg = future.result()
                    processed_count += 1 # 无论成功失败，都计数
                    
                    if error_msg:
                        failed_chunks.append((chunk_id, error_msg))
                        # 可以在这里选择是否继续，或者记录后继续
                        print(f"[FAIL] Chunk {chunk_id} failed: {error_msg[:200]}...") # 打印部分错误信息
                    
                    if results:
                        batch_buffer.extend(results)
                        total_symbols += len(results)
                        
                        if len(batch_buffer) >= BATCH_WRITE_SIZE:
                            flush_buffer()
                        
                except Exception as e:
                    # 捕获 future.result() 本身的异常 (理论上 process_file_chunk_robust 应该都捕获了)
                    failed_chunks.append((chunk_id, f"Exception getting result: {e}"))
                    print(f"[FAIL] Chunk {chunk_id} unexpected result error: {e}")
                
                # 更新进度条显示
                pbar.set_postfix({"processed": processed_count, "total": len(chunks), "symbols": total_symbols})
            
            # Flush remaining
            flush_buffer()

        # 4. 创建索引
        print("[*] Creating DB indices (symbols, symbol_tokens)...")
        cursor.execute("CREATE INDEX idx_name ON symbols(name);")
        cursor.execute("CREATE INDEX idx_path ON symbols(path);")
        
        # [Optimized] Create indices for symbol_tokens at the end
        print("[*] indexing tokens and symbol_tokens...")
        cursor.execute("CREATE UNIQUE INDEX idx_token_text ON tokens(text);")
        cursor.execute("CREATE UNIQUE INDEX idx_st_pk ON symbol_tokens(token_id, symbol_id);")
        cursor.execute("CREATE INDEX idx_st_sym ON symbol_tokens(symbol_id);")
        
        conn.commit()
        conn.close()
        
        elapsed = time.time() - start_time
        print(f"[*] Done. Indexed {total_symbols} symbols in {elapsed:.2f}s.")
        
        if failed_chunks:
            print(f"\n[!] WARNING: {len(failed_chunks)} chunks failed to process.")
            # 打印前几个失败的 chunk 信息
            for i, (chunk_id, err) in enumerate(failed_chunks[:5]):
                print(f"  - Chunk {chunk_id}: {err}")
            if len(failed_chunks) > 5:
                print(f"  ... and {len(failed_chunks) - 5} more.")

    def retrieve_symbol_definitions(self, symbols: List[str]) -> List[Dict]:
        """
        Retrieves all definitions of a list of symbols and returns structured data.
        """
        self.load_index()
        conn = self._get_conn()
        cursor = conn.cursor()
        
        all_results = []
        for symbol_name in symbols:
            if ' ' in symbol_name:
                symbol_name = symbol_name.split(' ')[-1].strip()
            
            cursor.execute(
                "SELECT path, start_line, end_line, kind, code FROM symbols WHERE name = ? LIMIT 5", 
                (symbol_name,)
            )
            rows = cursor.fetchall()
            
            for row in rows:
                path, start, end, kind, code = row
                all_results.append({
                    "name": symbol_name,
                    "path": path,
                    "start_line": start,
                    "end_line": end,
                    "kind": kind,
                    "code": code
                })
        return all_results

    def list_functions_in_file(self, file_path: str) -> List[Dict]:
        """
        List all indexed functions in a specific file.
        """
        self.load_index()
        conn = self._get_conn()
        cursor = conn.cursor()
        
        # Normalize file path if possible or use LIKE
        # Assuming database paths are relative to repo root
        # We try strict match first, then suffix match if failed (caution: might be slow if not indexed by path)
        
        cursor.execute(
            "SELECT name, start_line, end_line FROM symbols WHERE path = ? AND kind IN ('function', 'method') ORDER BY start_line",
            (file_path,)
        )
        rows = cursor.fetchall()
        if not rows:
             # Fallback to loose match if file_path provided is full path but DB has relative
             cursor.execute(
                "SELECT name, start_line, end_line FROM symbols WHERE ? LIKE '%' || path AND kind IN ('function', 'method') ORDER BY start_line",
                (file_path,)
            )
             rows = cursor.fetchall()

        results = []
        for row in rows:
            results.append({
                "name": row[0],
                "start_line": row[1],
                "end_line": row[2]
            })
        return results

    def get_symbol_code(self, symbol: str) -> str:
        """
        Retrieve raw code for a function symbol.
        """
        self.load_index()
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT code FROM symbols WHERE name = ? AND kind = 'function' LIMIT 1", (symbol,))
        row = cursor.fetchone()
        return row[0] if row else ""

    def find_callers(self, symbol: str) -> List[Dict]:
        """
        Find callers of a symbol.
        Returns: List[Dict] with keys: file, callback, line, content
        """
        if not symbol or len(symbol) < 3: return []
        self.load_index()
        conn = self._get_conn()
        cursor = conn.cursor()
        query = f"%{symbol}%"
        # We need start_line to calculate absolute line numbers
        cursor.execute(
            "SELECT name, path, code, start_line FROM symbols WHERE kind = 'function' AND code LIKE ? LIMIT 10", 
            (query,)
        )
        rows = cursor.fetchall()
        results = []
        for row in rows:
            caller_name, path, code, start_line = row
            if not start_line: start_line = 1
            lines = code.split('\n')
            
            for i, line in enumerate(lines):
                # Detailed check could be expensive, here we just check presence
                if symbol in line and not line.strip().startswith(("//", "/*", "*")):
                    results.append({
                        "file": path,
                        "caller": caller_name,
                        "line": start_line + i,
                        "content": line.strip()
                    })

        return results


    def search_functions_by_tokens(
        self,
        raw_inputs: List[str],
        limit: int = 200
    ) -> List[Tuple[str, str, str, int]]:
        """
        统一的 Token 倒排索引搜索方法（替代 search_functions_containing 和 search_functions_fuzzy）。
        使用与 database 完全一致的 tokenize_code() 保证搜索和索引的一致性。
        
        参数:
            raw_inputs: 原始字符串列表（会自动使用 tokenize_code 处理）
            limit: 返回结果数量上限
        
        返回:
            List of (file_path, func_name, code_content, start_line)
        """
        self.load_index()
        conn = self._get_conn()
        cursor = conn.cursor()
        
        # 1. Tokenize 输入（使用与 database 相同的 tokenize_code）
        query_tokens = []
        for item in raw_inputs:
            query_tokens.extend(tokenize_code(item))
        
        unique_query_tokens = list(set(query_tokens))
        
        if not unique_query_tokens:
            return []
        
        # 2. 自适应策略：Token 太多时，选择最长的（通常更有区分度）
        MAX_TOKENS = 30  # SQL IN 子句的实用上限
        if len(unique_query_tokens) > MAX_TOKENS:
            unique_query_tokens.sort(key=len, reverse=True)
            selected_tokens = unique_query_tokens[:MAX_TOKENS]
        else:
            selected_tokens = unique_query_tokens
        
        # 3. 最小匹配数固定为 1（至少匹配一个 token）
        min_match = 1
        
        # 4. SQL 倒排索引查询
        placeholders = ",".join(["?"] * len(selected_tokens))
        query_sql = f"""
            SELECT s.path, s.name, s.code, s.start_line, COUNT(st.token_id) as hit_count
            FROM symbols s
            JOIN symbol_tokens st ON s.id = st.symbol_id
            JOIN tokens t ON st.token_id = t.id
            WHERE (s.kind = 'function' OR s.kind = 'member')
              AND t.text IN ({placeholders})
            GROUP BY s.id
            HAVING hit_count >= ?
            ORDER BY hit_count DESC
            LIMIT ?
        """
        
        params = selected_tokens + [min_match, limit]
        
        try:
            rows = cursor.execute(query_sql, params).fetchall()
        except Exception as e:
            print(f"[Indexer] Token search failed: {e}")
            return []
        
        # 5. 直接返回（基于 SQL hit_count 排序的结果）
        return [(r[0], r[1], r[2], r[3]) for r in rows]
    
class BenchmarkSymbolIndexer:
    def __init__(self, cache_dir: str = DATABASE_DIR_PATH):
        self.cache_dir = cache_dir
        self.db_file = os.path.join(self.cache_dir, "idx_benchmark.db")
        self._conn = None

    def _get_conn(self):
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_file, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL;") 
            self._conn.execute("PRAGMA synchronous=NORMAL;")
        return self._conn

    def build_index(self):
        content = read_json(BENCHMARK_DICT_PATH)
        total_cve_df = pd.read_csv(BENCHMARK_VUL_PATH)
        
        # 1. Prepare DB
        if not os.path.exists(self.cache_dir): os.makedirs(self.cache_dir)
        if os.path.exists(self.db_file): os.remove(self.db_file)
        
        conn = sqlite3.connect(self.db_file)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        cursor = conn.cursor()
        
        # [Restored] Specific schema for benchmark
        cursor.execute("""
            CREATE TABLE benchmark_symbols (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vul_id TEXT,
                tag TEXT,
                version TEXT,
                file_path TEXT,
                func_name TEXT,
                code_content TEXT
            )
        """)
        
        # [New] Inverted Index Tables
        cursor.execute("""
            CREATE TABLE tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT UNIQUE NOT NULL
            )
        """)
        cursor.execute("CREATE INDEX idx_token_text ON tokens(text);")
        
        cursor.execute("""
            CREATE TABLE symbol_tokens (
                token_id INTEGER,
                symbol_id INTEGER,
                PRIMARY KEY (token_id, symbol_id)
            )
        """)
        cursor.execute("CREATE INDEX idx_sym_tok_sym ON symbol_tokens(symbol_id);")
        
        conn.commit()
        
        to_insert = []
        
        for idx, row in total_cve_df.iterrows():
            repo, vul_id, fixed_commit_sha = row['repo'], row['vul_id'], row['fixed_commit_sha']
            if vul_id not in content:
                continue
            codes = content[vul_id]['versions']
            for tag in ['pre', 'vul', 'fix']:
                for ver in codes[tag]:
                    for file_path in codes[tag][ver]:
                        for func_name in codes[tag][ver][file_path]:
                            code_content = codes[tag][ver][file_path][func_name]
                            # Keep raw data for insertion
                            to_insert.append((vul_id, tag, ver, file_path, func_name, code_content))
        
        # Batch Insert
        print(f"[*] Inserting {len(to_insert)} benchmark functions...")
        
        cursor.execute("BEGIN TRANSACTION;")
        
        token_cache = {}
        
        for item in tqdm(to_insert, desc="Indexing Benchmark"):
            vul_id, tag, ver, file_path, func_name, code_content = item
            cursor.execute(
                "INSERT INTO benchmark_symbols (vul_id, tag, version, file_path, func_name, code_content) VALUES (?, ?, ?, ?, ?, ?)",
                (vul_id, tag, ver, file_path, func_name, code_content)
            )
            symbol_id = cursor.lastrowid
            
            # Tokenize
            tokens = tokenize_code(code_content)
            # Add function name parts
            tokens.extend(tokenize_code(func_name))
            
            unique_tokens = set(tokens)
            
            for token_text in unique_tokens:
                if token_text not in token_cache:
                    cursor.execute("SELECT id FROM tokens WHERE text = ?", (token_text,))
                    row = cursor.fetchone()
                    if row:
                        token_cache[token_text] = row[0]
                    else:
                        cursor.execute("INSERT INTO tokens (text) VALUES (?)", (token_text,))
                        token_cache[token_text] = cursor.lastrowid
                
                token_id = token_cache[token_text]
                cursor.execute("INSERT OR IGNORE INTO symbol_tokens (token_id, symbol_id) VALUES (?, ?)", (token_id, symbol_id))

        conn.commit()
        
        print("[*] Creating indices...")
        cursor.execute("CREATE INDEX idx_bench_vul_id ON benchmark_symbols(vul_id);")
        cursor.execute("CREATE INDEX idx_bench_lookup ON benchmark_symbols(vul_id, file_path, func_name);")
        conn.commit()
        conn.close()
        print("[*] Benchmark index built.")
            
    def load_index(self):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            
        if not os.path.exists(self.db_file):
            self.build_index()
        else:
            try:
                conn = self._get_conn()
                cursor = conn.cursor()
                cursor.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='benchmark_symbols'")
                if cursor.fetchone()[0] == 0:
                    self.build_index()
                else:
                    cursor.execute("SELECT code_content FROM benchmark_symbols LIMIT 1")
                    # print(f"[*] Benchmark database found: {self.db_file}")
            except Exception:
                self.build_index()

    def search_functions_by_tokens(self, vul_id: str, raw_inputs: List[str], limit: int = 100) -> List[Tuple[str, str, str, int]]:
        """
        Benchmark 专用的统一 Token 搜索方法。
        使用与 database 一致的 tokenize_code() 保证搜索和索引的一致性。
        至少匹配一个 token 即返回（高召回率策略）。
        
        返回:
            List of (file_path, compound_name, code_content, start_line)
            注意：Benchmark 模式下，start_line 固定为 1（因为存储的是函数代码片段）
        """
        self.load_index()
        conn = self._get_conn()
        cursor = conn.cursor()
        
        # 1. Tokenize（使用与 database 相同的逻辑）
        query_tokens = []
        for item in raw_inputs:
            query_tokens.extend(tokenize_code(item))
            
        if not query_tokens:
            return []
            
        unique_query_tokens = list(set(query_tokens))
        
        # 2. 最小匹配数固定为 1（至少匹配一个 token）
        min_match = 1
        
        placeholders = ",".join(["?"] * len(unique_query_tokens))
        
        query_sql = f"""
            SELECT s.file_path, s.tag, s.version, s.func_name, s.code_content, COUNT(st.token_id) as hit_count
            FROM benchmark_symbols s
            JOIN symbol_tokens st ON s.id = st.symbol_id
            JOIN tokens t ON st.token_id = t.id
            WHERE s.vul_id = ?
              AND t.text IN ({placeholders})
            GROUP BY s.id
            HAVING hit_count >= ?
            ORDER BY hit_count DESC
            LIMIT ?
        """
        
        params = [vul_id] + unique_query_tokens + [min_match, limit]
        
        try:
            rows = cursor.execute(query_sql, params).fetchall()
            # Return format: (file_path, compound_name, code, start_line)
            # Benchmark模式下，start_line固定为1（因为存储的是函数代码片段，没有文件上下文）
            return [(r[0], f"{r[1]}:{r[2]}:{r[3]}", r[4], 1) for r in rows]
        except Exception as e:
            print(f"[BenchmarkIndexer] Search failed: {e}")
            return []
    
    def search(self, vul_id, file_path, func_name, ver = None) -> List[Tuple[str, str, str, int]]:
        """
        基于文件路径和函数名搜索基准库中的符号。
        Returns: List of (file_path, compound_name, code_content, start_line)
        """
        self.load_index()
        conn = self._get_conn()
        cursor = conn.cursor()
        if ver:
            cursor.execute(
                "SELECT vul_id, tag, version, code_content FROM benchmark_symbols WHERE vul_id = ? AND file_path = ? AND func_name = ? AND version = ? LIMIT 20",
                (vul_id, file_path, func_name, ver)
            )
        else:
            cursor.execute(
                "SELECT vul_id, tag, version, code_content FROM benchmark_symbols WHERE vul_id = ? AND file_path = ? AND func_name = ? LIMIT 20",
                (vul_id, file_path, func_name)
            )
        rows = cursor.fetchall()
        
        candidates = []
        for row in rows:
            vul_id, tag, version, code_content = row
            # Benchmark模式下，start_line固定为1
            candidates.append((file_path, f'{tag}:{version}:{func_name}', code_content, 1))
        return candidates

class GitSymbolIndexer:
    """
    基于 Git 的动态符号索引器。
    不依赖预构建的数据库，而是直接使用 git grep 在指定版本中搜索符号定义。
    适用于 Benchmark 模式下的多版本并行分析。
    """
    def __init__(self, repo_path: str):
        self.repo_path = repo_path

    def find_file_paths(self, filename: str, version: str) -> List[str]:
        """
        Find files in the repository ending with the given filename at a specific version.
        Uses git ls-tree to list files in that version.
        """
        matches = []
        try:
            # git ls-tree -r --name-only <version>
            cmd = ["git", "ls-tree", "-r", "--name-only", version]
            proc = subprocess.run(cmd, cwd=self.repo_path, capture_output=True, text=True, check=True)
            all_files = proc.stdout.splitlines()
            
            for f in all_files:
                if f.endswith(f"/{filename}") or f == filename:
                    matches.append(f)
        except Exception as e:
            # print(f"[GitSymbolIndexer] Failed to find files for {filename} at {version}: {e}")
            pass
        
        return matches

    def retrieve_symbol_definitions_at_version(self, symbols: List[str], version: str) -> List[Dict]:
        """
        In specific version (tag/commit), find symbol definitions.
        Supports functions, structs, macros.
        Returns structured data compatible with GlobalSymbolIndexer.
        """
        if not symbols or not version:
            return []

        all_definitions = []
        for symbol in symbols:
            # 1. Construct search regex
            definitions = []
            
            # A. Struct/Union
            try:
                cmd_struct = [
                    "git", "grep", "-n", "-E",
                    f"^(struct|union|enum)\\s+{symbol}\\s*{{?",
                    version, "--", "*.c", "*.h", "*.cpp", "*.hpp"
                ]
                out_struct = subprocess.check_output(cmd_struct, cwd=self.repo_path, text=True, stderr=subprocess.DEVNULL)
                if out_struct:
                    definitions.extend(self._parse_grep_output(out_struct, version, "struct"))
            except subprocess.CalledProcessError:
                pass

            # B. Macro
            try:
                cmd_macro = [
                    "git", "grep", "-n", "-E",
                    f"^#define\\s+{symbol}\\b",
                    version, "--", "*.c", "*.h", "*.cpp", "*.hpp"
                ]
                out_macro = subprocess.check_output(cmd_macro, cwd=self.repo_path, text=True, stderr=subprocess.DEVNULL)
                if out_macro:
                    definitions.extend(self._parse_grep_output(out_macro, version, "macro"))
            except subprocess.CalledProcessError:
                pass
                
            # C. Function
            try:
                cmd_func = [
                    "git", "grep", "-n", "-E",
                    f"\\b{symbol}\\s*\\(",
                    version, "--", "*.c", "*.h", "*.cpp", "*.hpp"
                ]
                out_func = subprocess.check_output(cmd_func, cwd=self.repo_path, text=True, stderr=subprocess.DEVNULL)
                if out_func:
                    candidates = self._parse_grep_output(out_func, version, "function")
                    real_funcs = []
                    for cand in candidates:
                        line_content = cand['line_content'].strip()
                        if line_content.endswith(';'): continue
                        if '=' in line_content and not line_content.startswith(symbol): continue
                        if 'if (' in line_content or 'while (' in line_content: continue
                        real_funcs.append(cand)
                    definitions.extend(real_funcs)
            except subprocess.CalledProcessError:
                pass
            
            # Post-processing and fetching code for each definition
            for item in definitions:
                # _read_git_file_context handles extraction
                code_context = self._read_git_file_context(version, item['file'], item['line'], smart_extract=True, symbol_name=symbol, symbol_type=item['type'])
                
                # [New] Filter: Check if symbol appears in the first few lines of the code
                # This helps filter out usages within function bodies
                if not self._is_valid_definition(code_context, symbol, item['type']):
                    continue
                
                all_definitions.append({
                    "name": symbol,
                    "path": item['file'],
                    "start_line": item['line'],
                    "end_line": None, # Parsing end line would require parsing logic in _read_git_file_context to be returned
                    "kind": item['type'],
                    "code": code_context
                })

        return all_definitions

    def list_functions_in_file_at_version(self, file_path: str, version: str) -> List[Dict]:
        """
        List all functions in a file at a specific version using ctags.
        """
        try:
            # Step 1: Get file content
            git_show_cmd = ["git", "show", f"{version}:{file_path}"]
            process_show = subprocess.Popen(git_show_cmd, cwd=self.repo_path, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            code_content, _ = process_show.communicate()
            
            if process_show.returncode != 0:
                print(f"[Warn] git show failed for {file_path} @ {version}")
                return []

            # Step 2: Run ctags on content
            ctags_cmd = [
                "ctags",
                "--languages=C,C++",
                "--output-format=json",
                "--fields=+ne",
                "--c-kinds=+f-p", 
                "--extras=+q",
                "-f", "-",
                "--stdin-filename=" + file_path 
            ]
            
            process_ctags = subprocess.Popen(
                ctags_cmd, 
                cwd=self.repo_path, 
                stdin=subprocess.PIPE, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout_data, _ = process_ctags.communicate(input=code_content)
            
            functions = []
            for line in stdout_data.splitlines():
                try:
                    tag = json.loads(line)
                    if tag.get('kind') == 'function':
                        # ctags JSON: 'line' is int, 'end' is int if available (from +e)
                        functions.append({
                            "name": tag.get('name'),
                            "start_line": tag.get('line'),
                            "end_line": tag.get('end') 
                        })
                except json.JSONDecodeError:
                    continue
            
            return sorted(functions, key=lambda x: x['start_line'])

        except Exception as e:
            print(f"Error listing functions in {file_path}@{version}: {e}")
            return []

    def _is_valid_definition(self, code_context: str, symbol: str, symbol_type: str) -> bool:
        """
        检查代码片段是否真的是符号的定义，而不仅仅是使用。
        
        策略：检查符号是否在代码片段的前2行（实际定义处）出现。
        """
        if not code_context:
            return False
        
        lines = code_context.splitlines()
        check_lines = []
        matched_line = None  # 保存被grep匹配的那一行
        
        for i, line in enumerate(lines):
            if '|' in line:
                code_part = line.split('|', 1)[1] if len(line.split('|')) > 1 else line
                check_lines.append(code_part)
                # 记录被grep匹配的那一行（带 >> 标记）
                if line.strip().startswith('>>'):
                    matched_line = code_part
        
        if not check_lines:
            return False
        
        # 根据符号类型进行不同的验证
        if symbol_type == "macro":
            # 宏定义：第一行必须是 #define symbol
            first_line = check_lines[0].strip()
            if first_line.startswith('#define') and symbol in first_line:
                # 确保 symbol 紧跟在 #define 后面（不是在参数或定义体中）
                define_part = first_line.replace('#define', '', 1).strip()
                # symbol 应该是第一个标识符
                if define_part.startswith(symbol):
                    return True
            return False
        
        elif symbol_type == "struct":
            # 结构体/联合体：第一行应该包含 struct/union/enum symbol
            first_line = check_lines[0].strip()
            for keyword in ['struct', 'union', 'enum']:
                if keyword in first_line and symbol in first_line:
                    # 简单检查：symbol 应该紧跟在 keyword 后面
                    pattern = f"{keyword}\\s+{symbol}\\b"
                    if re.search(pattern, first_line):
                        return True
            return False
        
        elif symbol_type == "function":
            # [新增] 特殊情况：如果匹配行实际上是宏定义，这是真正的宏定义，应该保留
            if matched_line and matched_line.strip().startswith('#define'):
                # 检查是否是 #define symbol(...)
                # 这是真正的宏定义，不是函数
                return False
            
            # [新增] 特殊情况：如果符号是全大写（通常是宏），且匹配行看起来是宏调用
            if symbol.isupper() and len(symbol) > 3:  # 至少4个字符的大写才认为是宏
                # 检查匹配行是否看起来是宏调用而非函数定义
                # 宏调用特征：symbol( 出现在语句中间，而不是在行首作为定义
                if matched_line:
                    line_stripped = matched_line.strip()
                    # 如果这一行直接以符号开始（可能有空格），检查是否有返回类型
                    # 宏调用：GF_SAFEALLOC(ptr, Type)
                    # 函数定义：int GF_SAFEALLOC(...) 或 static void GF_SAFEALLOC(...)
                    if line_stripped.startswith(symbol + '(') or line_stripped.startswith(symbol + ' ('):
                        # 这看起来是宏调用（直接以符号开头），不是函数定义
                        return False
            
            # 函数定义通常形式：返回类型 symbol(参数)
            # 或者直接 symbol(参数)
            # 避免匹配函数调用，如：func_call(symbol(x))
            
            # 1. 检查是否包含符号和括号
            check_text = '\n'.join(check_lines[:3])
            if symbol not in check_text or '(' not in check_text:
                return False
            
            # 2. 检查是否是真正的函数定义（而不是调用）
            # 函数定义特征：
            # - 第一行或前几行包含 "symbol("
            # - 不应该在赋值语句右边
            # - 不应该在控制流语句中（if, while, for已经在之前过滤了）
            # - 应该看起来像函数签名（有返回类型或在行首）
            
            for i, line in enumerate(check_lines[:3]):
                line_stripped = line.strip()
                # 找到包含 symbol( 的行
                if f"{symbol}(" in line_stripped or f"{symbol} (" in line_stripped:
                    # 排除：这一行以 = 开头或中间有 = symbol
                    before_symbol = line_stripped.split(symbol)[0]
                    if '=' in before_symbol and not before_symbol.strip().startswith('='):
                        # 这可能是赋值语句，如 x = symbol(...)
                        continue
                    
                    # 排除：符号直接在行首（没有返回类型）
                    # 真正的函数定义通常前面有返回类型
                    # 如果这是第一行且没有返回类型，很可能是宏调用或函数调用
                    if line_stripped.startswith(symbol):
                        # 检查前面是否有类型关键字
                        # 如果before_symbol为空或只有空格，说明没有返回类型
                        if not before_symbol or not before_symbol.strip():
                            # 没有返回类型，可能是宏调用或函数调用
                            return False
                    
                    # 排除：函数调用在表达式中，如 foo(symbol(...))
                    # 简单判断：如果 symbol 之前紧跟着其他函数调用的括号，可能是嵌套调用
                    if re.search(r'\w+\s*\([^)]*' + re.escape(symbol), line_stripped):
                        # symbol 在另一个函数调用的参数中
                        continue
                    
                    # 如果通过了上述检查，认为这是有效的函数定义
                    return True
            
            return False
        
        return False

    def _parse_grep_output(self, output: str, version: str, type_label: str) -> List[Dict]:
        results = []
        for line in output.splitlines():
            # format: version:path:line:content (if version is passed to git grep, it prefixes it)
            # but we passed version as arg, so output is "version:path:line:content"
            # Wait, git grep <tree> outputs "tree:path:line:content"
            parts = line.split(':', 3)
            if len(parts) < 4: continue
            
            # parts[0] is version/tree
            path = parts[1]
            try:
                line_no = int(parts[2])
            except:
                continue
            content = parts[3]
            
            results.append({
                "file": path,
                "line": line_no,
                "line_content": content,
                "type": type_label
            })
        return results

    def _read_git_file_context(self, version: str, file_path: str, line_no: int, context_lines: int = 10, smart_extract: bool = False, symbol_name: str = None, symbol_type: str = None) -> str:
        """
        使用 git show 读取文件特定行周围的上下文。
        如果 smart_extract=True 且提供了 symbol_name，尝试使用 CtagsParser 解析完整定义（函数、结构体、宏）。
        """
        try:
            # git show version:path
            content = subprocess.check_output(["git", "show", f"{version}:{file_path}"], cwd=self.repo_path, text=True, stderr=subprocess.DEVNULL)
            
            # [新增] 基于 Ctags 的精确提取
            if smart_extract and symbol_name:
                try:
                    # 解析内存中的代码
                    parsed = CtagsParser.parse_code(content)
                    
                    # 查找匹配的符号
                    target = None
                    
                    # 优先根据类型查找
                    if symbol_type == "Function" and symbol_name in parsed['functions']:
                        target = parsed['functions'][symbol_name]
                    elif symbol_type == "Struct/Union" and symbol_name in parsed['structs']:
                        target = parsed['structs'][symbol_name]
                    elif symbol_type == "Macro" and symbol_name in parsed['macros']:
                        target = parsed['macros'][symbol_name]
                    
                    # 如果类型不匹配或未指定，尝试所有类型
                    if not target:
                        if symbol_name in parsed['functions']: target = parsed['functions'][symbol_name]
                        elif symbol_name in parsed['structs']: target = parsed['structs'][symbol_name]
                        elif symbol_name in parsed['macros']: target = parsed['macros'][symbol_name]
                    
                    # 如果找到了目标，且位置接近 grep 结果
                    if target and abs(target['start_line'] - line_no) < 50:
                        lines = target['code'].splitlines()
                        numbered_snippet = []
                        for i, l in enumerate(lines):
                            curr_line = target['start_line'] + i
                            marker = ">>" if curr_line == line_no else "  "
                            numbered_snippet.append(f"{marker} {curr_line:4d} | {l}")
                        return "\n".join(numbered_snippet)
                        
                except Exception as e:
                    # Ctags 解析失败，回退到默认逻辑
                    pass

            lines = content.splitlines()
            total_lines = len(lines)
            
            start_idx = max(0, line_no - 1 - context_lines) # line_no is 1-based
            end_idx = min(total_lines, line_no - 1 + context_lines + 10)
            
            snippet = lines[start_idx:end_idx]
            # 添加行号
            numbered_snippet = []
            for i, l in enumerate(snippet):
                curr_line = start_idx + i + 1
                marker = ">>" if curr_line == line_no else "  "
                numbered_snippet.append(f"{marker} {curr_line:4d} | {l}")
                
            return "\n".join(numbered_snippet)
        except Exception as e:
            return f"Error reading file context: {e}"

    def get_symbol_code(self, symbol: str, version: str) -> str:
        """
        Retrieve the raw source code of a symbol (function) in a specific version.
        Used for slicing/PDG analysis.
        """
        if not symbol or not version: return ""
        
        try:
            cmd_func = [
                "git", "grep", "-n", "-E", 
                f"\\b{symbol}\\s*\\(", 
                version
            ]
            out_func = subprocess.check_output(cmd_func, cwd=self.repo_path, text=True, stderr=subprocess.DEVNULL)
            if out_func:
                candidates = self._parse_grep_output(out_func, version, "Function")
                for cand in candidates:
                    line_content = cand['line_content'].strip()
                    if line_content.endswith(';'): continue
                    if '=' in line_content: continue
                    if 'if (' in line_content or 'while (' in line_content: continue
                    
                    # Found a likely definition. Now extract code.
                    try:
                        content = subprocess.check_output(["git", "show", f"{version}:{cand['file']}"], cwd=self.repo_path, text=True, stderr=subprocess.DEVNULL)
                        parsed = CtagsParser.parse_code(content)
                        if symbol in parsed['functions']:
                            return parsed['functions'][symbol]['code']
                    except:
                        pass
        except:
            pass
        return ""

    def find_callers(self, symbol: str, version: str) -> List[Dict]:
        """
        Find usages of a symbol in the codebase for a specific version.
        Resolves the enclosing function (caller).
        Returns: List[Dict] with keys: file, caller, line, content
        
        改进：
        1. 过滤掉函数定义行（只返回真正的调用点）
        2. 使用更精确的模式匹配来识别函数调用
        """
        if not symbol or len(symbol) < 3: return []
        
        results = []
        try:
            # 使用更精确的模式：匹配函数调用（symbol后跟括号）
            # 这样可以减少误匹配
            cmd = ["git", "grep", "-n", "-E", f"\\b{symbol}\\s*\\(", version]
            output = subprocess.check_output(cmd, cwd=self.repo_path, text=True, stderr=subprocess.DEVNULL)
            
            # Group by file to minimize ctags calls
            file_matches = {}
            for line in output.splitlines():
                parts = line.split(':', 3)
                if len(parts) < 4: continue
                
                path = str(parts[1])
                try:
                    line_no = int(parts[2])
                except:
                    continue
                content = parts[3]
                
                # 过滤注释行
                if content.strip().startswith(("//", "/*", "*")): continue
                
                # [新增] 过滤函数定义行的启发式规则
                content_stripped = content.strip()
                
                # 规则1: 如果是静态函数定义（static ... symbol(）
                if re.match(r'^static\s+\w+\s+' + re.escape(symbol) + r'\s*\(', content_stripped):
                    continue
                    
                # 规则2: 如果看起来像函数定义（返回类型 symbol(）且在行首或接近行首
                # 函数定义通常形式：[static] <type> symbol(...) 或 symbol(...)
                # 排除形如 "BOOL rdp_write_logon_info_v1(wStream* s, logon_info* info)"
                if re.match(r'^(\w+\s+)*\w+\s+' + re.escape(symbol) + r'\s*\([^)]*\)\s*$', content_stripped):
                    # 这看起来是函数签名（参数后直接结束或只有{）
                    continue
                
                # 规则3: 如果这行以分号结束，且符号在行首附近，可能是声明
                if content_stripped.endswith(';') and content_stripped.startswith(symbol):
                    continue
                
                if path not in file_matches:
                    file_matches[path] = []
                file_matches[path].append((line_no, content))
            
            # Process each file to resolve caller function
            for path, matches in file_matches.items():
                # Get function ranges for this file
                funcs = self.list_functions_in_file_at_version(path, version)
                
                for line_no, content in matches:
                    caller_name = "global/unknown"
                    
                    # Find which function covers this line
                    # funcs is sorted by start_line
                    for f in funcs:
                        # [修复] 检查该行是否是函数本身的定义
                        # 如果匹配的行号就是函数的起始行，说明这是函数定义，跳过
                        if f['start_line'] == line_no:
                            # 这是函数定义行，不是调用
                            caller_name = None
                            break
                        
                        # 检查该行是否在函数范围内
                        if f['start_line'] < line_no:
                            # 如果有end_line，严格检查
                            if f.get('end_line'):
                                if line_no <= f['end_line']:
                                    caller_name = f['name']
                            else:
                                # 没有end_line，使用近似判断
                                caller_name = f['name']
                        elif f['start_line'] > line_no:
                            # 已经扫描过了可能的范围
                            break
                    
                    # 只添加有效的调用点（排除定义）
                    # caller_name=None 表示这是函数定义行，应该排除
                    # caller_name="global/unknown" 表示无法确定调用者，但仍然是有效的调用，应该保留
                    if caller_name is not None:
                        results.append({
                            "file": path,
                            "caller": caller_name,
                            "line": line_no,
                            "content": content.strip()
                        })
                    
        except subprocess.CalledProcessError:
            pass
            
        return results[:10]  # 限制返回数量，避免过多结果
