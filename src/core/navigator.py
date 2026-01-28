import os
import re
import networkx as nx
import subprocess
from typing import List, Tuple, Optional, Dict, Any, Union
from extraction.pdg import PDGBuilder
from core.indexer import GlobalSymbolIndexer, GitSymbolIndexer
from extraction.cfg import CFGNodeType

class CodeNavigator:
    """
    A unified tool for code analysis and navigation.
    Backs the Agent's exploration capabilities.
    """
    def __init__(self, repo_path: str, target_version: Optional[str] = None):
        self.repo_path = repo_path
        self.target_version = target_version
        
        # Cache structure: 
        # { 
        #   "file_path": {
        #       "code": str,
        #       "lines": List[str],
        #       # Cache PDGs by function start line or simply by target line request
        #       # key: function_unique_id (e.g. start_line), value: nx.MultiDiGraph
        #       "pdgs": Dict[int, nx.MultiDiGraph] 
        #   }
        # }
        self.file_cache: Dict[str, Dict] = {}
        self.current_file: Optional[str] = None
        
        # Initialize Indexer
        # If target_version is set (Benchmark Mode), we usually want GitSymbolIndexer.
        # But if the user provides a repo_path in Repo Mode, we use GlobalSymbolIndexer.
        # However, CodeNavigator is generic. 
        # Benchmark Mode -> target_version is NOT None (a git hash).
        # Repo Mode -> target_version IS None (search HEAD/Workspace).
        if target_version:
            self.indexer = GitSymbolIndexer(repo_path)
        else:
            self.indexer = GlobalSymbolIndexer(repo_path)

    def _ensure_file_loaded(self, file_path: str):
        """Load file content into cache."""
        resolved_path = file_path

        # Resolve basename to full path if needed
        if not os.path.isabs(file_path) and os.sep not in file_path:
            if self.target_version and isinstance(self.indexer, GitSymbolIndexer):
                found_paths = self.indexer.find_file_paths(file_path, version=self.target_version)
            else:
                found_paths = self.indexer.find_file_paths(file_path)
            
            if found_paths:
                resolved_path = found_paths[0]

        # Return if already cached
        if resolved_path in self.file_cache:
            self.current_file = resolved_path
            return

        # Load file content
        code = ""
        try:
            if self.target_version:
                cmd = ["git", "show", f"{self.target_version}:{resolved_path}"]
                proc = subprocess.run(cmd, cwd=self.repo_path, capture_output=True, text=True, check=True)
                code = proc.stdout
            else:
                full_path = os.path.join(self.repo_path, resolved_path) if not os.path.isabs(resolved_path) else resolved_path
                with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                    code = f.read()
        except Exception as e:
            print(f"[CodeNavigator] Failed to load file {resolved_path}: {e}")

        self.file_cache[resolved_path] = {
            "code": code,
            "lines": code.splitlines(),
            "pdgs": {}
        }
        self.current_file = resolved_path

    def _get_pdg(self, file_path: str, target_line: int) -> Optional[nx.MultiDiGraph]:
        """Get or build PDG for the function covering target_line."""
        self._ensure_file_loaded(file_path)
        data = self.file_cache.get(file_path)
        if not data or not data["code"]:
            return None

        # Find function covering target_line using indexer
        funcs = self.list_file_functions(file_path)
        func_start = -1
        for f in funcs:
            if f['start_line'] <= target_line and (not f.get('end_line') or target_line <= f['end_line']):
                func_start = f['start_line']
                break
        
        # Return cached PDG if available
        if func_start != -1 and func_start in data['pdgs']:
            return data['pdgs'][func_start]
        
        # Build new PDG
        try:
            lang = "cpp" if file_path.endswith(('.cpp', '.cc', '.cxx', '.hpp')) else "c"
            pdg = PDGBuilder(data["code"], lang=lang).build(target_line=target_line)
            
            if pdg and len(pdg.nodes) > 0:
                cache_key = func_start if func_start != -1 else target_line
                data['pdgs'][cache_key] = pdg
                return pdg
        except Exception as e:
            print(f"[CodeNavigator] PDG build failed for {file_path} line {target_line}: {e}")
            
        return None

    def _find_node_at_line(self, pdg: nx.MultiDiGraph, target_line: int) -> Optional[Any]:
        """
        Find the PDG node that corresponds to the given line number.
        Returns the node ID if found, None otherwise.
        """
        if not pdg:
            return None
        
        for node_id, node_data in pdg.nodes(data=True):
            node_line = node_data.get('start_line', -1)
            if node_line == target_line:
                return node_id
            
            # Check if target_line falls within this node's range
            end_line = node_data.get('end_line', node_line)
            if node_line <= target_line <= end_line:
                return node_id
        
        return None

    def read_code_window(self, file_path: str, start_line: int, end_line: int, with_line_numbers: bool = True) -> str:
        """Read a window of code from a specific file."""
        self._ensure_file_loaded(file_path)
        data = self.file_cache.get(file_path)
        if not data or not data["lines"]:
            return "File empty or not found."

        if start_line < 1: start_line = 1
        lines = data["lines"][start_line - 1 : end_line]
        
        if with_line_numbers:
            return '\n'.join(f"[{i + start_line:4d}] {line}" for i, line in enumerate(lines))
        else:
            return '\n'.join(lines)

    def list_file_functions(self, file_path: str) -> List[Dict]:
        """
        List functions in a file using the Indexer (fast, uses ctags/db).
        Wrapped by CodeNavigator as requested.
        """
        if self.target_version and isinstance(self.indexer, GitSymbolIndexer):
            return self.indexer.list_functions_in_file_at_version(file_path, self.target_version)
        elif isinstance(self.indexer, GlobalSymbolIndexer):
            return self.indexer.list_functions_in_file(file_path)
        return []

    def grep(self,
             pattern: str,
             file_path: str,
             mode: str = "word",
             scope_range: Optional[Tuple[int, int]] = None,
             limit: int = 20,
             cursor: int = 0) -> Dict[str, Any]:
        """
        Unified search tool with multiple matching modes.
        
        Args:
            pattern: Search pattern (variable name, function name, or regex).
            file_path: Target file to search in.
            mode: Matching mode:
                - "word": Exact whole-word match (default, best for identifiers).
                  Automatically adds word boundaries to avoid partial matches.
                - "regex": Full regular expression support for complex patterns.
                - "def_use": PDG-enhanced mode that distinguishes definitions from uses.
                  Requires scope_range to build PDG. Falls back to "word" if PDG unavailable.
            scope_range: Optional (start_line, end_line) to limit search scope.
            limit: Maximum results to return (for pagination).
            cursor: Starting index for paginated results.
            
        Returns:
            Dict with:
                - results: List of matches with {line, content, [type]}
                - total_count: Total matches found
                - has_more: Whether more results exist
                - method: "precise_pdg", "word_match", or "regex_match"
                
        Examples:
            # Find all uses of a variable (precise word matching)
            grep("ptr", "file.c", mode="word")
            
            # Find function call patterns
            grep(r"malloc\s*\(", "file.c", mode="regex")
            
            # Distinguish definitions from uses (PDG-enhanced)
            grep("ptr", "file.c", mode="def_use", scope_range=(10, 50))
        """
        self._ensure_file_loaded(file_path)
        
        results = []
        method = "word_match"  # Track which method was used
        
        # Mode: def_use (PDG-enhanced)
        if mode == "def_use" and scope_range:
            start_line, end_line = scope_range
            probe_line = start_line
            pdg = self._get_pdg(file_path, probe_line)
            
            if pdg:
                method = "precise_pdg"
                target_lines = set(range(start_line, end_line + 1))
                
                for n, d in pdg.nodes(data=True):
                    n_start = d.get('start_line', -1)
                    if n_start < 0 or n_start not in target_lines:
                        continue
                    
                    # Check Defs and Uses
                    defs = d.get('defs', {})
                    uses = d.get('uses', {})
                    
                    # Normalize to set of names
                    def_vars = set(defs.keys()) if isinstance(defs, dict) else set(defs)
                    use_vars = set(uses.keys()) if isinstance(uses, dict) else set(uses)
                    
                    if pattern in def_vars or pattern in use_vars:
                        results.append({
                            "line": n_start,
                            "content": d.get('code', '').strip(),
                            "type": "def" if pattern in def_vars else "use"
                        })
                
                results.sort(key=lambda x: x['line'])
        
        # Fallback or primary mode: text-based search
        if not results or mode != "def_use":
            data = self.file_cache[file_path]
            lines = data['lines']
            
            # Determine search window
            search_start = scope_range[0] if scope_range else 1
            search_end = scope_range[1] if scope_range else len(lines)
            search_start = max(1, search_start)
            search_end = min(len(lines), search_end)
            
            # Compile pattern based on mode
            try:
                if mode == "word" or mode == "def_use":
                    # Whole-word match: add word boundaries automatically
                    pat = re.compile(rf'\b{re.escape(pattern)}\b')
                    method = "word_match"
                else:  # mode == "regex"
                    pat = re.compile(pattern)
                    method = "regex_match"
            except re.error as e:
                return {
                    "results": [],
                    "total_count": 0,
                    "cursor": 0,
                    "limit": limit,
                    "has_more": False,
                    "next_cursor": None,
                    "method": "error",
                    "error": f"Invalid regex: {e}"
                }
            
            # Scan lines
            for i in range(search_start - 1, search_end):
                if i >= len(lines):
                    break
                line_content = lines[i]
                if pat.search(line_content):
                    results.append({
                        "line": i + 1,
                        "content": line_content.strip(),
                        "type": "match"
                    })
        
        # Pagination
        total_count = len(results)
        paginated_results = results[cursor : cursor + limit]
        has_more = (cursor + limit) < total_count
        
        return {
            "results": paginated_results,
            "total_count": total_count,
            "cursor": cursor,
            "limit": limit,
            "has_more": has_more,
            "next_cursor": cursor + limit if has_more else None,
            "method": method
        }

    def trace_variable(self, file_path: str, target_line: int, var_name: str, direction: str = "backward", limit_lines: int = 50) -> List[Dict]:
        """
        Trace data flow for a variable starting from a specific line.
        
        Args:
            file_path: Target file path.
            target_line: Starting line number.
            var_name: Variable name to trace.
            direction: "backward" (trace origins) or "forward" (trace propagation).
            limit_lines: Maximum number of trace results to return.
            
        Returns:
            List of dicts with {line, content, type} showing the data flow path.
            
        Examples:
            # Trace where a variable comes from
            trace_variable("file.c", 25, "ptr", "backward")
            
            # Trace where a variable propagates to
            trace_variable("file.c", 10, "ptr", "forward")
        """
        pdg = self._get_pdg(file_path, target_line)
        if not pdg:
            return [{"error": "PDG unavailable"}]

        # Find start node
        start_node = self._find_node_at_line(pdg, target_line)
        if not start_node:
            return [{"error": f"Node not found at line {target_line}"}]

        # BFS traversal
        results = []
        visited = set([start_node])
        worklist = [(start_node, 0)]

        while worklist and len(results) < limit_lines:
            curr, depth = worklist.pop(0)
            node_data = pdg.nodes[curr]
            
            # Add to results (skip start node)
            if depth > 0:
                results.append({
                    "line": node_data.get('start_line'),
                    "content": node_data.get('code', '').strip().split('\n')[0][:200],
                    "type": node_data.get('type')
                })

            # Get edges based on direction
            edges = pdg.in_edges(curr, data=True) if direction == "backward" else pdg.out_edges(curr, data=True)
            
            for u, v, edge_data in edges:
                neighbor = u if direction == "backward" else v
                rel = edge_data.get('relationship', '')
                
                # Filter: only follow DATA edges matching var_name
                if rel == 'DATA' and edge_data.get('var') != var_name:
                    continue

                if neighbor not in visited:
                    visited.add(neighbor)
                    worklist.append((neighbor, depth + 1))
        
        if len(results) >= limit_lines:
            results.append({"type": "TRUNCATED", "content": f"... trace limit reached ({limit_lines} items) ..."})
        
        results.sort(key=lambda x: x.get('line', 0) if isinstance(x.get('line'), int) else 0)
        return results

    def get_guard_conditions(self, file_path: str, target_line: int) -> List[str]:
        """
        Get conditional statements that guard (dominate) the execution of a target line.
        
        Args:
            file_path: Target file path.
            target_line: Line number to analyze.
            
        Returns:
            List of condition strings in format "Line X: <condition>", sorted by line number.
            Returns empty list if no guards found.
            
        Examples:
            # Find what conditions must be satisfied to reach line 25
            get_guard_conditions("file.c", 25)
            # → ["Line 10: if (ptr != NULL)", "Line 20: if (size > 0)"]
        """
        pdg = self._get_pdg(file_path, target_line)
        if not pdg:
            return []

        target_node = self._find_node_at_line(pdg, target_line)
        if not target_node:
            return []

        conditions = set()
        visited = {target_node}
        worklist = [target_node]

        while worklist:
            curr = worklist.pop(0)
            
            for pred in pdg.predecessors(curr):
                # Check if there's a CONTROL edge
                edges = pdg.get_edge_data(pred, curr)
                has_control = any(e.get('relationship') == 'CONTROL' for e in edges.values())
                
                if has_control:
                    pred_data = pdg.nodes[pred]
                    p_type = pred_data.get('type')
                    
                    # Check if it's a predicate node
                    is_predicate = (p_type == CFGNodeType.PREDICATE) or \
                                   (p_type in ('if_statement', 'while_statement', 'for_statement', 'condition'))
                    
                    if is_predicate:
                        p_line = pred_data.get('start_line')
                        p_code = pred_data.get('code', '').strip()
                        conditions.add(f"Line {p_line}: {p_code}")
                    
                    if pred not in visited:
                        visited.add(pred)
                        worklist.append(pred)

        return sorted(conditions, key=lambda x: int(x.split(':')[0].replace('Line ', '')))

    def get_next_control(self, file_path: str, target_line: int, max_depth: int = 10) -> Dict:
        """
        Analyze the control flow outcome after executing a target line.
        
        Args:
            file_path: Target file path.
            target_line: Starting line number.
            max_depth: Maximum search depth (default: 10).
            
        Returns:
            Dict with {type, statement} indicating the control flow outcome:
            - "RETURN": Execution returns from function
            - "GOTO": Execution jumps via goto
            - "FALLTHROUGH": Execution continues normally
            
        Examples:
            # Check if a defensive check causes early return
            get_next_control("file.c", 15)
            # → {"type": "RETURN", "statement": "return -1;"}
        """
        pdg = self._get_pdg(file_path, target_line)
        if not pdg:
            return {"type": "UNKNOWN", "statement": "PDG unavailable"}

        start_node = self._find_node_at_line(pdg, target_line)
        if not start_node:
            return {"type": "UNKNOWN", "statement": "Node not found"}

        # BFS for terminal nodes
        visited = {start_node}
        worklist = [start_node]
        depth = max_depth
        
        while worklist and depth > 0:
            curr = worklist.pop(0)
            depth -= 1
            
            node_data = pdg.nodes[curr]
            node_type = str(node_data.get('type', '')).upper()
            code = node_data.get('code', '')

            # Check for control flow terminals
            if 'RETURN' in node_type or code.startswith('return '):
                return {"type": "RETURN", "statement": code.strip()}
            if 'GOTO' in node_type or code.startswith('goto '):
                return {"type": "GOTO", "statement": code.strip()}
            
            # Continue BFS
            for succ in pdg.successors(curr):
                if succ not in visited:
                    visited.add(succ)
                    worklist.append(succ)
            
        return {"type": "FALLTHROUGH", "statement": "Execution continues"}

    def find_definition(self, symbol_name: str, context_path: Optional[str] = None) -> List[Dict]:
        """
        Find the definition of a symbol (function, struct, macro, etc.).
        
        Args:
            symbol_name: Name of the symbol to find.
            context_path: (Optional) Context file path for relevance ranking.
                          Prioritizes definitions in the same file or directory.
        
        Returns:
            List of definition dicts with {path, line, name, kind}, ranked by relevance.
            
        Examples:
            # Find where malloc is defined
            find_definition("malloc")
            
            # Find definition with context ranking
            find_definition("helper_func", context_path="main.c")
        """
        results = []
        if self.target_version and isinstance(self.indexer, GitSymbolIndexer):
            # Pass single symbol, get raw list
            results = self.indexer.retrieve_symbol_definitions_at_version([symbol_name], self.target_version)
        else:
            results = self.indexer.retrieve_symbol_definitions([symbol_name])
            
        if not context_path or not results:
            return results
        
        # Ranking Logic
        def score(res):
            res_path = res.get('path', '')
            if res_path == context_path: return 100 # Same file
            if os.path.dirname(res_path) == os.path.dirname(context_path): return 50 # Same dir
            # Check for header/source swap (file.h <-> file.c)
            base_ctx = os.path.splitext(os.path.basename(context_path))[0]
            base_res = os.path.splitext(os.path.basename(res_path))[0]
            if base_ctx == base_res: return 40
            return 0
        
        results.sort(key=score, reverse=True)
        return results

    def retrieve_symbol_definition(self, symbol_name: str, context_path: Optional[str] = None) -> List[Dict]:
        """Deprecated alias for find_definition. Use find_definition() instead."""
        return self.find_definition(symbol_name, context_path)

    def get_callers(self, symbol_name: str) -> List[Dict]:
        """
        Find call sites or usages of a symbol.
        Returns: List[Dict] with keys: file, line, content, [caller]
        """
        if self.target_version and isinstance(self.indexer, GitSymbolIndexer):
            return self.indexer.find_callers(symbol_name, self.target_version)
        elif isinstance(self.indexer, GlobalSymbolIndexer):
            return self.indexer.find_callers(symbol_name)
        return []