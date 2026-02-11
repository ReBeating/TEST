import os
import json
import subprocess
import sqlite3
import hashlib
import threading
import re
import sys
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


class DatabaseNotFoundError(RuntimeError):
    """Raised when a required pre-built database is missing.
    
    The user must run `python build_database.py` before using the pipeline.
    """
    pass

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
    """Split the list into chunks of fixed size"""
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
    Quickly find all source files.
    Prefer git ls-files (extremely fast), fall back to os.walk if it fails.
    """
    extensions = {'.c', '.h', '.cpp', '.hpp', '.cc', '.cc', '.cxx'} # Add as needed
    files = []
    
    # Method 1: Try git (Linux kernel is usually a git repo)
    try:
        cmd = ["git", "ls-files"]
        # Limit to finding relevant extensions only, or get all and filter later
        # git ls-files outputs relative paths
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

    # Method 2: Fallback to os.walk
    print("[*] 'git ls-files' failed or not a git repo. Falling back to os.walk...")
    for root, dirs, filenames in os.walk(repo_path):
        # Exclude common non-code directories
        dirs[:] = [d for d in dirs if d not in ('.git', 'Documentation', 'scripts', 'tools')]
        
        for name in filenames:
            if os.path.splitext(name)[1] in extensions:
                rel_path = os.path.relpath(os.path.join(root, name), repo_path)
                files.append(rel_path)
    
    print(f"[*] Found {len(files)} files using os.walk.")
    return files

def process_file_chunk_robust(repo_path: str, file_chunk: List[str], chunk_id: int) -> Tuple[List[Tuple], str | None]:
    """
    [Worker Process Entry]
    Enhanced error catching and ctags process status check.
    Return: (result list, error message string or None)
    """
    results = []
    if not file_chunk:
        return (results, None)

    cmd = [
        "ctags",
        "--languages=C,C++",
        "--output-format=json",
        "--fields=+ne",
        "--c-kinds=+f-p", # Only look for function definitions, exclude prototypes
        "--extras=+q",
        "-f", "-",        # Output to stdout
        "-L", "-"         # Read file list from stdin
    ]
    
    # PID of the ctags process, used for later check
    ctags_pid = None

    try:
        process = subprocess.Popen(
            cmd, cwd=repo_path,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, # Capture stderr
            text=True, encoding='utf-8', errors='replace',
            bufsize=1024*1024
        )
        ctags_pid = process.pid # Get ctags process ID

        # Use communicate to set timeout, e.g., 5 minutes
        # Note: ctags parsing of Linux kernel may be very time-consuming, adjust according to actual situation
        # If it is a huge file, it may take longer, or even cancel the timeout (but high risk)
        # A better way is: do not use communicate directly, but poll stdout and stderr
        # But for simplicity here, we try to use timeout first
        try:
            input_str = "\n".join(file_chunk)
            stdout_data, stderr_data = process.communicate(input=input_str, timeout=300) # 5 minutes timeout
            
            if process.returncode != 0:
                error_msg = f"ctags exited with code {process.returncode}. Stderr:\n{stderr_data.strip()}"
                print(f"[WARN] Chunk {chunk_id} ctags error: {error_msg}")
                return (results, error_msg) # Return error message

            # --- Parse stdout_data ---
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
                    # Error reading file or processing single tag, ignore that tag
                    pass
            
            return (results, None) # Success

        except subprocess.TimeoutExpired:
            error_msg = f"ctags process timed out after 300s. Stderr:\n{stderr_data.strip()}"
            print(f"[ERROR] Chunk {chunk_id} timeout: {error_msg}")
            # Try to terminate ctags process
            if ctags_pid:
                try:
                    p = psutil.Process(ctags_pid)
                    p.terminate()
                    p.wait(timeout=5) # Wait for termination
                except (psutil.NoSuchProcess, psutil.AccessDenied, TimeoutError):
                    pass # Ignore termination error
            return ([], error_msg) # Return timeout error
        
        except Exception as e:
            # Other Python exceptions
            error_msg = f"Unexpected error in worker process: {e}. Stderr:\n{stderr_data.strip()}"
            print(f"[ERROR] Chunk {chunk_id} worker error: {error_msg}")
            return ([], error_msg) # Return general error

    except Exception as e:
        # Error during Popen stage
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
            if not force_rebuild:
                # Database missing or corrupted — refuse to auto-build during pipeline runs
                print(
                    f"\n{'='*70}\n"
                    f"  ERROR: Symbol database not found for repository:\n"
                    f"    {self.repo_path}\n\n"
                    f"  Expected database file:\n"
                    f"    {self.db_file}\n\n"
                    f"  Please build the database first by running:\n"
                    f"    python build_database.py -m 0day\n"
                    f"{'='*70}\n",
                    file=sys.stderr,
                )
                raise DatabaseNotFoundError(
                    f"Symbol database not found: {self.db_file}. "
                    f"Run 'python build_database.py -m 0day' first."
                )
            self.build_index()

    def build_index(self):
        start_time = time.time()
        print(f"[*] Building global symbol index (Parallel Map-Reduce Robust) for {self.repo_path} ...")
        
        # 1. Prepare database
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

        # 2. Get file list
        all_files = find_source_files(self.repo_path)
        if not all_files:
            print("[!] No source files found.")
            return

        # 3. Filter out large files
        MAX_FILE_SIZE = 30 * 1024 * 1024  # 30MB
        valid_files = []
        skipped_count = 0
        
        print(f"[*] Filtering files larger than {MAX_FILE_SIZE/1024/1024:.2f}MB ...")
        
        for f_path in all_files:
            full_path = os.path.join(self.repo_path, f_path)
            try:
                # Check file size
                size = os.path.getsize(full_path)
                if size > MAX_FILE_SIZE:
                    # Print which large files were skipped, usually see generated headers like amdgpu
                    print(f"    [SKIP] Too large ({size/1024/1024:.2f}MB): {f_path}") 
                    skipped_count += 1
                    continue
                valid_files.append(f_path)
            except OSError:
                # File may have been deleted or caused by soft links
                continue
                
        print(f"[*] Filtered out {skipped_count} huge files. Remaining: {len(valid_files)}")
        all_files = valid_files # Update list

        CHUNK_SIZE = 10
        chunks = list(chunked_iterable(all_files, CHUNK_SIZE))
        print(f"[*] Split into {len(chunks)} chunks (size {CHUNK_SIZE}). Launching workers...")

        total_symbols = 0
        failed_chunks = [] # Record failed chunk ID
        
        with ProcessPoolExecutor(max_workers=16) as executor:
            # Include chunk_id when submitting task for error reporting
            future_to_chunk_id = {
                executor.submit(process_file_chunk_robust, self.repo_path, chunk, i): i 
                for i, chunk in enumerate(chunks)
            }
            
            batch_buffer = []
            BATCH_WRITE_SIZE = 5000
            
            # Modify tqdm iteration method to handle f.result() return value
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
                    processed_count += 1 # Count regardless of success or failure
                    
                    if error_msg:
                        failed_chunks.append((chunk_id, error_msg))
                        # Can choose to continue here, or record and continue
                        print(f"[FAIL] Chunk {chunk_id} failed: {error_msg[:200]}...") # Print partial error message
                    
                    if results:
                        batch_buffer.extend(results)
                        total_symbols += len(results)
                        
                        if len(batch_buffer) >= BATCH_WRITE_SIZE:
                            flush_buffer()
                        
                except Exception as e:
                    # Capture future.result() exception itself (theoretically process_file_chunk_robust should have captured it)
                    failed_chunks.append((chunk_id, f"Exception getting result: {e}"))
                    print(f"[FAIL] Chunk {chunk_id} unexpected result error: {e}")
                
                # Update progress bar display
                pbar.set_postfix({"processed": processed_count, "total": len(chunks), "symbols": total_symbols})
            
            # Flush remaining
            flush_buffer()

        # 4. Create index
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
            # Print information about the first few failed chunks
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
        Unified Token inverted index search method (replaces search_functions_containing and search_functions_fuzzy).
        Use tokenize_code() consistent with database to ensure consistency between search and index.
        
        Args:
            raw_inputs: List of raw strings (automatically processed using tokenize_code)
            limit: Maximum number of results
        
        Returns:
            List of (file_path, func_name, code_content, start_line)
        """
        self.load_index()
        conn = self._get_conn()
        cursor = conn.cursor()
        
        # 1. Tokenize input (using the same tokenize_code as database)
        query_tokens = []
        for item in raw_inputs:
            query_tokens.extend(tokenize_code(item))
        
        unique_query_tokens = list(set(query_tokens))
        
        if not unique_query_tokens:
            return []
        
        # 2. Adaptive strategy: When there are too many Tokens, choose the longest ones (usually more distinctive)
        MAX_TOKENS = 30  # Practical limit for SQL IN clause
        if len(unique_query_tokens) > MAX_TOKENS:
            unique_query_tokens.sort(key=len, reverse=True)
            selected_tokens = unique_query_tokens[:MAX_TOKENS]
        else:
            selected_tokens = unique_query_tokens
        
        # 3. Minimum match count fixed to 1 (match at least one token)
        min_match = 1
        
        # 4. SQL inverted index query
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
        
        # 5. Return directly (results sorted based on SQL hit_count)
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
                    cursor.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='benchmark_symbols'")
                    if cursor.fetchone()[0] == 0:
                        should_build = True
                    else:
                        cursor.execute("SELECT code_content FROM benchmark_symbols LIMIT 1")
                        # print(f"[*] Benchmark database found: {self.db_file}")
                except Exception:
                    should_build = True

        if should_build:
            if not force_rebuild:
                # Database missing or corrupted — refuse to auto-build during pipeline runs
                print(
                    f"\n{'='*70}\n"
                    f"  ERROR: Benchmark database not found!\n\n"
                    f"  Expected database file:\n"
                    f"    {self.db_file}\n\n"
                    f"  Please build the database first by running:\n"
                    f"    python build_database.py -m 1day\n"
                    f"{'='*70}\n",
                    file=sys.stderr,
                )
                raise DatabaseNotFoundError(
                    f"Benchmark database not found: {self.db_file}. "
                    f"Run 'python build_database.py -m 1day' first."
                )
            self.build_index()

    def search_functions_by_tokens(self, vul_id: str, raw_inputs: List[str], limit: int = 100) -> List[Tuple[str, str, str, int]]:
        """
        Unified Token search method dedicated to Benchmark.
        Use tokenize_code() consistent with database to ensure consistency between search and index.
        Return as soon as at least one token is matched (high recall strategy).
        
        Returns:
            List of (file_path, compound_name, code_content, start_line)
            Note: In Benchmark mode, start_line is fixed to 1 (because function code snippets are stored)
        """
        self.load_index()
        conn = self._get_conn()
        cursor = conn.cursor()
        
        # 1. Tokenize (using the same logic as database)
        query_tokens = []
        for item in raw_inputs:
            query_tokens.extend(tokenize_code(item))
            
        if not query_tokens:
            return []
            
        unique_query_tokens = list(set(query_tokens))
        
        # 2. Minimum match count fixed to 1 (match at least one token)
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
            # In Benchmark mode, start_line is fixed to 1 (because function code snippets are stored, no file context)
            return [(r[0], f"{r[1]}:{r[2]}:{r[3]}", r[4], 1) for r in rows]
        except Exception as e:
            print(f"[BenchmarkIndexer] Search failed: {e}")
            return []
    
    def search(self, vul_id, file_path, func_name, ver = None) -> List[Tuple[str, str, str, int]]:
        """
        Search for symbols in the benchmark library based on file path and function name.
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
            # In Benchmark mode, start_line is fixed to 1
            candidates.append((file_path, f'{tag}:{version}:{func_name}', code_content, 1))
        return candidates

class GitSymbolIndexer:
    """
    Dynamic symbol indexer based on Git.
    Does not rely on pre-built database, but uses git grep to search for symbol definitions in specified version directly.
    Suitable for multi-version parallel analysis in Benchmark mode.
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
        Check if the code snippet is really a definition of the symbol, not just a usage.
        
        Strategy: Check if the symbol appears in the first 2 lines of the code snippet (actual definition).
        """
        if not code_context:
            return False
        
        lines = code_context.splitlines()
        check_lines = []
        matched_line = None  # Record the line matched by grep (with >> marker)
        
        for i, line in enumerate(lines):
            if '|' in line:
                code_part = line.split('|', 1)[1] if len(line.split('|')) > 1 else line
                check_lines.append(code_part)
                # Record the line matched by grep (with >> marker)
                if line.strip().startswith('>>'):
                    matched_line = code_part
        
        if not check_lines:
            return False
        
        # Perform different verifications based on symbol type
        if symbol_type == "macro":
            # Macro definition: first line must be #define symbol
            first_line = check_lines[0].strip()
            if first_line.startswith('#define') and symbol in first_line:
                # Ensure symbol follows #define immediately (not in parameters or definition body)
                define_part = first_line.replace('#define', '', 1).strip()
                # symbol should be the first identifier
                if define_part.startswith(symbol):
                    return True
            return False
        
        elif symbol_type == "struct":
            # Struct/Union: first line should contain struct/union/enum symbol
            first_line = check_lines[0].strip()
            for keyword in ['struct', 'union', 'enum']:
                if keyword in first_line and symbol in first_line:
                    # Simple check: symbol should follow keyword immediately
                    pattern = f"{keyword}\\s+{symbol}\\b"
                    if re.search(pattern, first_line):
                        return True
            return False
        
        elif symbol_type == "function":
            # [New] Special case: If the matched line is actually a macro definition, this is a real macro definition and should be kept
            if matched_line and matched_line.strip().startswith('#define'):
                # Check if it is #define symbol(...)
                # This is a real macro definition, not a function
                return False
            
            # [New] Special case: If the symbol is all uppercase (usually macro) and the matched line looks like a macro call
            if symbol.isupper() and len(symbol) > 3:  # Considered a macro only if at least 4 uppercase characters
                # Check if the matched line looks like a macro call rather than a function definition
                # Macro call feature: symbol( appears in the middle of the statement, not at the beginning of the line as a definition
                if matched_line:
                    line_stripped = matched_line.strip()
                    # If this line starts directly with the symbol (may have spaces), check if there is a return type
                    # Macro call: GF_SAFEALLOC(ptr, Type)
                    # Function definition: int GF_SAFEALLOC(...) or static void GF_SAFEALLOC(...)
                    if line_stripped.startswith(symbol + '(') or line_stripped.startswith(symbol + ' ('):
                        # This looks like a macro call (starts directly with symbol), not a function definition
                        return False
            
            # Function definition usually form: return type symbol(arguments)
            # Or directly symbol(arguments)
            # Avoid matching function calls, e.g.: func_call(symbol(x))
            
            # 1. Check if it contains symbol and parentheses
            check_text = '\n'.join(check_lines[:3])
            if symbol not in check_text or '(' not in check_text:
                return False
            
            # 2. Check if it is a real function definition (not a call)
            # Function definition features:
            # - The first line or the first few lines contain "symbol("
            # - Should not be on the right side of an assignment statement
            # - Should not be in control flow statements (if, while, for have been filtered before)
            # - Should look like a function signature (has return type or at the beginning of the line)
            
            for i, line in enumerate(check_lines[:3]):
                line_stripped = line.strip()
                # Find line with symbol(
                if f"{symbol}(" in line_stripped or f"{symbol} (" in line_stripped:
                    # Exclude: This line starts with = or has = symbol in the middle
                    before_symbol = line_stripped.split(symbol)[0]
                    if '=' in before_symbol and not before_symbol.strip().startswith('='):
                        # This may be an assignment statement, e.g., x = symbol(...)
                        continue
                    
                    # Exclude: Symbol directly at the beginning of the line (no return type)
                    # Real function definitions usually have a return type in front
                    # If this is the first line and there is no return type, it is likely a macro call or function call
                    if line_stripped.startswith(symbol):
                        # Check if there is a type keyword in front
                        # If before_symbol is empty or only spaces, there is no return type
                        if not before_symbol or not before_symbol.strip():
                            # No return type, may be a macro call or function call
                            return False
                    
                    # Exclude: Function call in expression, e.g., foo(symbol(...))
                    # Simple judgment: If symbol is immediately followed by parentheses of other function calls, it may be a nested call
                    if re.search(r'\w+\s*\([^)]*' + re.escape(symbol), line_stripped):
                        # symbol is in the parameters of another function call
                        continue
                    
                    # If passed the above checks, consider it a valid function definition
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
        Use git show to read context around specific line.
        If smart_extract=True and symbol_name provided, try to use CtagsParser to parse full definition (function, struct, macro).
        """
        try:
            # git show version:path
            content = subprocess.check_output(["git", "show", f"{version}:{file_path}"], cwd=self.repo_path, text=True, stderr=subprocess.DEVNULL)
            
            # [New] Precise extraction based on Ctags
            if smart_extract and symbol_name:
                try:
                    # Parse code in memory
                    parsed = CtagsParser.parse_code(content)
                    
                    # Find matching symbol
                    target = None
                    
                    # Prioritize searching by type
                    if symbol_type == "Function" and symbol_name in parsed['functions']:
                        target = parsed['functions'][symbol_name]
                    elif symbol_type == "Struct/Union" and symbol_name in parsed['structs']:
                        target = parsed['structs'][symbol_name]
                    elif symbol_type == "Macro" and symbol_name in parsed['macros']:
                        target = parsed['macros'][symbol_name]
                    
                    # If type mismatch or not specified, try all types
                    if not target:
                        if symbol_name in parsed['functions']: target = parsed['functions'][symbol_name]
                        elif symbol_name in parsed['structs']: target = parsed['structs'][symbol_name]
                        elif symbol_name in parsed['macros']: target = parsed['macros'][symbol_name]
                    
                    # If target found and position is close to grep result
                    if target and abs(target['start_line'] - line_no) < 50:
                        lines = target['code'].splitlines()
                        numbered_snippet = []
                        for i, l in enumerate(lines):
                            curr_line = target['start_line'] + i
                            marker = ">>" if curr_line == line_no else "  "
                            numbered_snippet.append(f"{marker} {curr_line:4d} | {l}")
                        return "\n".join(numbered_snippet)
                        
                except Exception as e:
                    # Ctags parsing failed, fallback to default logic
                    pass

            lines = content.splitlines()
            total_lines = len(lines)
            
            start_idx = max(0, line_no - 1 - context_lines) # line_no is 1-based
            end_idx = min(total_lines, line_no - 1 + context_lines + 10)
            
            snippet = lines[start_idx:end_idx]
            # Add line numbers
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
        
        Improvements:
        1. Filter out function definition lines (return only real call sites)
        2. Use more precise pattern matching to identify function calls
        """
        if not symbol or len(symbol) < 3: return []
        
        results = []
        try:
            # Use more precise pattern: match function call (symbol followed by parentheses)
            # This reduces false matches
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
                
                # Filter comment lines
                if content.strip().startswith(("//", "/*", "*")): continue
                
                # [New] Heuristic rules to filter function definition lines
                content_stripped = content.strip()
                
                # Rule 1: If it is a static function definition (static ... symbol()
                if re.match(r'^static\s+\w+\s+' + re.escape(symbol) + r'\s*\(', content_stripped):
                    continue
                    
                # Rule 2: If it looks like a function definition (return type symbol() and at/near start of line
                # Function definition usually form: [static] <type> symbol(...) or symbol(...)
                # Exclude forms like "BOOL rdp_write_logon_info_v1(wStream* s, logon_info* info)"
                if re.match(r'^(\w+\s+)*\w+\s+' + re.escape(symbol) + r'\s*\([^)]*\)\s*$', content_stripped):
                    # This looks like a function signature (ends directly after parameters or only {)
                    continue
                
                # Rule 3: If this line ends with a semicolon and the symbol is near the start of the line, it may be a declaration
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
                        # [Fix] Check if the line is the definition of the function itself
                        # If the matched line number is the function start line, it means this is a function definition, skip
                        if f['start_line'] == line_no:
                            # This is a function definition line, not a call
                            caller_name = None
                            break
                        
                        # Check if the line is within the function range
                        if f['start_line'] < line_no:
                            # If there is end_line, strict check
                            if f.get('end_line'):
                                if line_no <= f['end_line']:
                                    caller_name = f['name']
                            else:
                                # No end_line, use approximate judgment
                                caller_name = f['name']
                        elif f['start_line'] > line_no:
                            # Scanned possible range already
                            break
                    
                    # Only add valid call sites (exclude definitions)
                    # caller_name=None means this is a function definition line, should be excluded
                    # caller_name="global/unknown" means caller cannot be determined, but it is still a valid call and should be kept
                    if caller_name is not None:
                        results.append({
                            "file": path,
                            "caller": caller_name,
                            "line": line_no,
                            "content": content.strip()
                        })
                    
        except subprocess.CalledProcessError:
            pass
            
        return results[:10]  # Limit returned quantity to avoid too many results
