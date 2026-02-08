import json
import subprocess
import os
import tempfile
from typing import Dict, Any, Optional
import tree_sitter_cpp
import tree_sitter_c
from tree_sitter import Language, Parser
import shutil

class CtagsParser:
    """Code parser based on Universal Ctags"""
    
    @staticmethod
    def _run_ctags(target_path: str) -> str:
        """Run ctags command and return JSON output"""
        # --fields=+ne: n=line number, e=end line number
        # --c-kinds=fsdue: f=function, s=struct, d=macro, u=union, e=enum
        cmd = [
            "ctags", "--fields=+ne", "--output-format=json", "--c-kinds=fsdue",
            "-o", "-", target_path
        ]
        try:
            # Add errors='ignore' to prevent decoding non-utf-8 file errors
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding='utf-8', errors='replace')
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"[Ctags] Error processing {target_path}: {e}")
            return ""
        except FileNotFoundError:
            print("[Ctags] Error: 'ctags' command not found. Please install universal-ctags.")
            return ""

    @staticmethod
    def parse_file(file_path: str) -> Dict[str, Dict[str, Any]]:
        """
        Parse specific file
        Return: {'functions': {...}, 'structs': {...}, 'macros': {...}}
        """
        if not os.path.exists(file_path):
            return {'functions': {}, 'structs': {}, 'macros': {}}

        # Read source code for slicing
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                code = f.read()
        except Exception as e:
            print(f"[Ctags] Read file error: {e}")
            return {'functions': {}, 'structs': {}, 'macros': {}}

        ctags_output = CtagsParser._run_ctags(file_path)
        return CtagsParser._process_tags(ctags_output, code)

    @staticmethod
    def parse_code(code: str, suffix='.c') -> Dict[str, Dict[str, Any]]:
        """Parse code strings in memory"""
        with tempfile.NamedTemporaryFile(mode="w+", suffix=suffix, delete=True) as tmp_file:
            tmp_file.write(code)
            tmp_file.flush()
            ctags_output = CtagsParser._run_ctags(tmp_file.name)
            return CtagsParser._process_tags(ctags_output, code)

    @staticmethod
    def _process_tags(ctags_output: str, code: str) -> Dict[str, Dict[str, Any]]:
        """Internal logic: Process ctags JSON output and slice"""
        lines = code.replace('\x0c', ' ').splitlines() 
        parse_result = {'functions': {}, 'structs': {}, 'macros': {}}
        
        for line in ctags_output.splitlines():
            try:
                tag = json.loads(line)
            except json.JSONDecodeError:
                continue
                
            kind = tag.get('kind', '')
            name = tag.get("name", "")
            start = tag.get("line", 0)
            end = tag.get("end", start) # If no end, default to single line
            
            # ctags is 1-based, python list is 0-based
            # Your logic: If the previous line also starts with a name (maybe this line is the return type?), move forward
            # Keep your heuristic here, but add a boundary check
            idx_start = max(0, start - 1)
            
            # Try to correct upwards (handle return type line break)
            # e.g.: 
            # void
            # my_func() { ... }
            if kind == 'function' and idx_start > 0:
                prev_line = lines[idx_start - 1].strip()
                # Simple heuristic: If the previous line does not end with a semicolon or right brace, it may be part of the function header
                if prev_line and not prev_line.endswith((';', '}')) and not prev_line.startswith(('#', '//')):
                     idx_start -= 1

            idx_end = end # slice is left-closed right-open, so end does not need -1
            
            tag_code = "\n".join(lines[idx_start:idx_end])
            
            item = {
                'code': tag_code, 
                'start_line': idx_start + 1, # Convert back to 1-based for display
                'end_line': idx_end
            }

            if kind == 'function':
                parse_result['functions'][name] = item
            elif kind in ('struct', 'union', 'enum'):
                parse_result['structs'][name] = item
            elif kind == 'macro':
                parse_result['macros'][name] = item
                
        return parse_result

def remove_comments(code: str, lang: str = "cpp") -> str:
    """
    Directly remove comments from code (supports C/C++).
    Automatically handle tree-sitter new and old version API differences.
    """
    # 1. Initialize language and parser
    if lang == "c":
        language = Language(tree_sitter_c.language())
    elif lang == "cpp":
        language = Language(tree_sitter_cpp.language())
    else:
        raise ValueError(f"Unsupported language: {lang}")

    parser = Parser(language)
    
    # 2. Parse code
    # tree-sitter handles byte offsets, so encode first
    code_bytes = code.encode('utf-8', errors='replace')
    tree = parser.parse(code_bytes)
    
    # 3. Find all comment nodes
    # Use Query API to find, faster and safer than recursive traversal
    query = language.query("(comment) @comment")
    captures = query.captures(tree.root_node)

    # 4. Get the byte range of comments (start_byte, end_byte)
    # [Critical] Compatible with tree-sitter new version (dict) and old version (list) return format
    ranges = []
    if isinstance(captures, dict):
        # New version: {'comment': [Node, Node, ...]}
        for nodes in captures.values():
            for node in nodes:
                ranges.append((node.start_byte, node.end_byte))
    else:
        # Old version: [(Node, 'comment'), ...]
        for node, _ in captures:
            ranges.append((node.start_byte, node.end_byte))
    
    # If no comments, return original code directly
    if not ranges:
        return code

    # 5. Execute deletion
    # Sort by start_byte in descending order to ensure that previous deletions do not affect subsequent offsets
    ranges.sort(key=lambda x: x[0], reverse=True)
    
    for start, end in ranges:
        # Replace the comment part with empty bytes
        code_bytes = code_bytes[:start] + code_bytes[end:]
        
    return code_bytes.decode('utf-8', errors='replace')


def indent_code(code: str) -> str:
    # 1. Check if indent tool exists
    if not shutil.which("indent"):
        return code

    # Argument list (recommended to adjust according to needs, preserving your original arguments here)
    indent_args = '-nbad -bap -nbc -bbo -hnl -br -brs -c33 -cd33 -ncdb -ce -ci4 -cli0 -d0 -di1 -nfc1 -i8 -ip0 -l9999 -lp -npcs -nprs -npsl -sai -saf -saw -ncs -nsc -sob -nfca -cp33 -ss -ts8 -il1'
    
    # 2. Use split() instead of split(' ') to avoid issues caused by extra spaces
    cmd = ["indent"] + indent_args.split() + ["-"]

    try:
        result = subprocess.run(
            cmd,
            input=code.encode("utf-8", errors='replace'),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # Set timeout to prevent deadlock (optional)
            timeout=10 
        )

        # 3. Critical: Check return code
        if result.returncode != 0:
            error_msg = result.stderr.decode("utf-8", errors='replace').strip()
            # If formatting fails, be sure to return the original content to prevent code loss
            return code

        # 4. Decode output
        return result.stdout.decode("utf-8", errors='replace')

    except subprocess.TimeoutExpired:
        return code
    except Exception as e:
        return code