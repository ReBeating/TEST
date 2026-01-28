import os
import re
import difflib
from git import Repo

from typing import Dict, List, Optional, Any
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from core.parser import remove_comments, indent_code, CtagsParser

from core.models import (
    AtomicPatch, GlobalHunkContext, 
    HunkObj, HunkDecision, BatchAnalysisResult
)

from core.state import WorkflowState

class RepoManager:
    def __init__(self, repo_path: str):
        self.path = repo_path
        self.repo = Repo(repo_path) if os.path.exists(repo_path) else None

    def get_changed_files(self, commit_hash: str) -> List[str]:
        if not self.repo: return []
        try:
            prev_commit = f"{commit_hash}^"
            diff_index = self.repo.commit(prev_commit).diff(commit_hash)
            files = set()
            for d in diff_index:
                if d.b_path: files.add(d.b_path)
                if d.a_path: files.add(d.a_path)
            return list(files)
        except Exception as e:
            print(f"Git Error: {e}")
            return []

    def get_file_content(self, commit: str, file_path: str) -> Optional[str]:
        if not self.repo: return None
        try:
            return (self.repo.commit(commit).tree / file_path).data_stream.read().decode('utf-8', errors='replace')
        except:
            return ""

    def get_commit_message(self, commit_hash: str) -> str:
        if not self.repo: return ""
        try:
            commit = self.repo.commit(commit_hash)
            return commit.message
        except:
            return ""

class CodeParser:
    def normalize(self, code: str) -> str:
        """Tier 0: 去注释 + 压缩空白 + indent 规范化"""
        if not code: return ""
        code = remove_comments(code)
        code = '\n'.join([line for line in code.split('\n') if line.strip() != ''])
        return indent_code(code)

    def extract_functions(self, file_content: str, suffix: str = '.c') -> Dict[str, Dict[str, Any]]:
        funcs = CtagsParser.parse_code(file_content, suffix=suffix)['functions']
        return funcs # Directly return the dict with metadata (code, start_line)

    def generate_diff(self, file_path: str, name: str, old_code: str, new_code: str, old_start: int = 1, new_start: int = 1) -> Optional[str]:
        diff_iter = difflib.unified_diff(
            old_code.splitlines(keepends=True),
            new_code.splitlines(keepends=True),
            fromfile=f"a/{file_path}:{name}",
            tofile=f"b/{file_path}:{name}"
        )
        
        # Stream processing to patch the Hunk Headers
        patched_lines = []
        header_pattern = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)")
        
        for line in diff_iter:
            match = header_pattern.match(line)
            if match:
                # Original relative lines
                orig_old_start = int(match.group(1))
                orig_old_len = match.group(2) if match.group(2) else "1"
                orig_new_start = int(match.group(3))
                orig_new_len = match.group(4) if match.group(4) else "1"
                suffix = match.group(5)
                
                # Adjust to absolute
                # The 'orig_old_start' is relative to the start of the 'old_code' string (Function Start).
                # So if old_start=100 (Function starts at line 100) and Hunk says -10, it means line 100+10-1 = 109.
                # Logic: absolute_line = function_start_line + relative_offset - 1
                
                abs_old = orig_old_start + old_start - 1
                abs_new = orig_new_start + new_start - 1
                
                # Correction: If orig_old_start is 0 (empty file/hunk start edge case), handle carefully.
                # However, unified_diff usually is 1-based.
                # Example:
                # Function starts at 100.
                # Change at line 5 of function (104 absolute).
                # Orig Hunk: @@ -5,1 +5,1 @@
                # Calc: 5 + 100 - 1 = 104. Correct.
                
                # Reconstruct header
                # Note: unified diff format is @@ -old_start,old_len +new_start,new_len @@
                # If len is 1, it's often omitted, but let's keep it consistent or follow matched logic
                new_header = f"@@ -{abs_old},{orig_old_len} +{abs_new},{orig_new_len} @@{suffix}\n"
                patched_lines.append(new_header)
            else:
                patched_lines.append(line)
                
        diff_str = "".join(patched_lines)
        return diff_str if diff_str.strip() else None

class DiffProcessor:
    @staticmethod
    def parse_hunks(diff_text: str) -> List[HunkObj]:
        hunks = []
        lines = diff_text.splitlines()
        current_header = None
        current_lines = []
        hunk_idx = 0
        for line in lines:
            if line.startswith("@@"):
                if current_header:
                    raw = current_header + "\n" + "\n".join(current_lines)
                    hunks.append(HunkObj(hunk_idx, current_header, current_lines, raw))
                    hunk_idx += 1
                current_header = line
                current_lines = []
            else:
                if current_header: current_lines.append(line)
        if current_header:
            raw = current_header + "\n" + "\n".join(current_lines)
            hunks.append(HunkObj(hunk_idx, current_header, current_lines, raw))
        return hunks

    @staticmethod
    def reconstruct_diff(hunks: List[HunkObj], decisions: List[HunkDecision]) -> Optional[str]:
        decision_map = {d.hunk_index: d for d in decisions}
        final_lines = []
        has_content = False

        for hunk in hunks:
            decision = decision_map.get(hunk.index)
            
            if not decision or decision.classification == "KEEP_ALL":
                final_lines.append(hunk.header)
                final_lines.extend(hunk.content)
                has_content = True
            elif decision.classification == "REMOVE_ALL":
                continue
            elif decision.classification == "PARTIAL_NOISE":
                noise_indices = {nl.line_index for nl in decision.noise_lines}
                kept_lines = []
                for idx, line in enumerate(hunk.content):
                    if idx in noise_indices:
                        if line.startswith(' ') or line == '': kept_lines.append(line)
                        continue 
                    kept_lines.append(line)
                
                if any(line.startswith(('+', '-')) for line in kept_lines):
                    final_lines.append(hunk.header)
                    final_lines.extend(kept_lines)
                    has_content = True
        return "\n".join(final_lines) if has_content else None

def is_pure_static_noise(hunk: HunkObj) -> bool:
    """Tier 1 Check: 静态规则过滤"""
    added = [l[1:].strip() for l in hunk.content if l.startswith('+')]
    deleted = [l[1:].strip() for l in hunk.content if l.startswith('-')]
    
    def ignorable(lines):
        for s in lines:
            if not s: continue
            if s.startswith('//'): continue
            if s.startswith('/*'):
                # Check for code after inline comment block
                end_idx = s.find('*/')
                if end_idx == -1: continue # Open block or multi-line
                if not s[end_idx+2:].strip(): continue # Pure comment line
                return False
            if s.startswith('*'):
                 # Heuristic for block comment continuations vs pointer dereferences
                 if s.startswith('*/'): continue
                 if s == '*': continue
                 if s.startswith('* ') or s.startswith('*\t'): continue
                 if set(s).issubset({'*', '/'}): continue # Catch separator lines like /*****/
                 return False
            return False
        return True

    if ignorable(added) and ignorable(deleted):
        return True
    return False

class SemanticCleaner:
    def __init__(self):
        self.llm = ChatOpenAI(
            base_url=os.getenv("API_BASE"),
            api_key=os.getenv("API_KEY"),
            model=os.getenv("MODEL_NAME"),
            temperature=0
        )
        self.parser = PydanticOutputParser(pydantic_object=BatchAnalysisResult)
        
        self.prompt = ChatPromptTemplate.from_template(
            """
            You are a Static Analysis Engine. Identify Signal vs. Noise in the provided Hunks.
            
            ### Instructions
            - **KEEP_ALL**: If it contains Logic (if/break), Resource (alloc/free), or Data Flow.
            - **REMOVE_ALL**: If 100% Noise (Comments, Logs, Formatting, Dead Code).
            - **PARTIAL_NOISE**: If mixed. Return the INDEX of lines to remove.

            **CRITICAL:** Return a JSON object for **EVERY** Hunk ID.
            
            ### Input Hunks
            {hunk_block}

            {format_instructions}
            """
        )

    def batch_hunks_dynamic(self, hunks: List[GlobalHunkContext], max_tokens=4000) -> List[List[GlobalHunkContext]]:
        batches = []
        current_batch = []
        current_chars = 0
        MAX_CHARS = max_tokens * 4
        
        for h in hunks:
            h_len = len(h.raw_text_with_indexed_lines) + 50
            if current_chars + h_len > MAX_CHARS and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_chars = 0
            current_batch.append(h)
            current_chars += h_len
        if current_batch: batches.append(current_batch)
        return batches

    def clean_batch_hunks(self, global_hunks: List[GlobalHunkContext]) -> Dict[int, HunkDecision]:
        if not global_hunks: return {}
        hunk_block = "\n\n".join([h.raw_text_with_indexed_lines for h in global_hunks])
        try:
            print(f"    >> [AI] Processing batch of {len(global_hunks)} hunks...")
            chain = self.prompt | self.llm | self.parser
            result = chain.invoke({
                "hunk_block": hunk_block,
                "format_instructions": self.parser.get_format_instructions()
            })
            return {d.hunk_index: d for d in result.decisions}
        except Exception as e:
            print(f"    [Error] AI Batch Failed: {e}")
            return {}

def preprocessing_node(state: WorkflowState) -> Dict:
    """Phase 3.1.1 节点入口"""
    print(f"--- Phase 1: Denoising Commit {state['commit_hash']} ---")
    
    repo_mgr = RepoManager(state['repo_path'])
    parser = CodeParser()
    diff_proc = DiffProcessor()
    cleaner = SemanticCleaner()
    
    patch_blueprints = [] 
    global_hunk_registry = []
    global_id_counter = 0

    lang = 'c'
    
    # 1. 采集与 Tier 0/1 过滤
    commit_message = repo_mgr.get_commit_message(state['commit_hash'])
    files = repo_mgr.get_changed_files(state['commit_hash'])
    for file_path in files:
        if not file_path.endswith(('.c', '.h', '.cpp', '.cc', '.cxx', '.hpp')): continue
        
        lang = 'c' if file_path.endswith(('.c', '.h')) else 'cpp'
        suffix = '.c' if file_path.endswith(('.c', '.h')) else '.cpp'
        
        old_content = repo_mgr.get_file_content(f"{state['commit_hash']}^", file_path)
        new_content = repo_mgr.get_file_content(state['commit_hash'], file_path)
        if old_content is None or new_content is None: continue

        old_funcs = parser.extract_functions(old_content, suffix=suffix)
        new_funcs = parser.extract_functions(new_content, suffix=suffix)

        for func_name, new_func_data in new_funcs.items():
            new_code = new_func_data['code']
            
            old_func_data = old_funcs.get(func_name, {})
            old_code = old_func_data.get('code', "")
            
            # Tier 0: Normalize 比较
            
            norm_old = parser.normalize(old_code)
            norm_new = parser.normalize(new_code)
            if norm_old == norm_new:
                continue
            
            # Diff Gen
            # [修改] 使用原始代码生成 Diff，确保行号和内容与 Agent 看到的上下文一致
            # 同时传入 absolute start lines 进行 patch
            sl_old = old_func_data.get('start_line', 1)
            sl_new = new_func_data.get('start_line', 1)
            
            raw_diff = parser.generate_diff(file_path, func_name, old_code, new_code, old_start=sl_old, new_start=sl_new)
            if not raw_diff: continue
            # print(f'raw_diff for {func_name} in {file_path}:\n{raw_diff}')
            local_hunks = diff_proc.parse_hunks(raw_diff)
            func_hunk_ids = []
            
            for h in local_hunks:
                # Tier 1: 静态过滤
                # [Fix] Offset Hunk Lines with start_line
                # HunkObj store raw_text and content
                # We don't necessarily need to change the HunkObj indices if they are relative to the function.
                # The GlobalHunkContext stores the function name and we have start_line in metadata.
                # So we can map later.
                
                if is_pure_static_noise(h):
                    func_hunk_ids.append(-1) # 标记为直接删除
                else:
                    g_hunk = GlobalHunkContext(global_id_counter, h, file_path, func_name)
                    global_hunk_registry.append(g_hunk)
                    func_hunk_ids.append(global_id_counter)
                    global_id_counter += 1
            
            patch_blueprints.append({
                "meta": (file_path, func_name, raw_diff, old_code, new_code, old_func_data.get('start_line'), new_func_data.get('start_line')),
                "hunks": local_hunks,
                "ids": func_hunk_ids
            })

    print(f"Total Hunks: {len(global_hunk_registry)} (after static filtering)")

    # 2. Tier 2: AI 批处理
    all_decisions = {}
    batches = cleaner.batch_hunks_dynamic(global_hunk_registry)
    for batch in batches:
        all_decisions.update(cleaner.clean_batch_hunks(batch))

    # 3. 重组
    final_patches = []
    for bp in patch_blueprints:
        file_path, func_name, raw_diff, old_code, new_code, sl_old, sl_new = bp['meta']
        local_decisions = []
        
        for local_hunk, g_id in zip(bp['hunks'], bp['ids']):
            if g_id == -1:
                local_decisions.append(HunkDecision(hunk_index=local_hunk.index, classification="REMOVE_ALL", reasoning="Tier 1"))
            elif g_id in all_decisions:
                # 映射回 Local Hunk Index
                d = all_decisions[g_id]
                local_decisions.append(HunkDecision(hunk_index=local_hunk.index, classification=d.classification, noise_lines=d.noise_lines, reasoning=d.reasoning))
        
        clean_diff = diff_proc.reconstruct_diff(bp['hunks'], local_decisions)
        if clean_diff:
            final_patches.append(AtomicPatch(
                file_path=file_path, function_name=func_name,
                clean_diff=clean_diff, raw_diff=raw_diff,
                change_type="MODIFIED" if old_code else "ADDED",
                old_code=old_code, new_code=new_code,
                start_line_old=sl_old,
                start_line_new=sl_new
            ))

    # Fallback: If denoising filtered everything but there were valid changes
    if not final_patches and patch_blueprints:
        print("    [Warn] Denoising removed all content. Falling back to raw diffs.")
        for bp in patch_blueprints:
            file_path, func_name, raw_diff, old_code, new_code, sl_old, sl_new = bp['meta']
            final_patches.append(AtomicPatch(
                file_path=file_path, function_name=func_name,
                clean_diff=raw_diff, # Fallback to raw
                raw_diff=raw_diff,
                change_type="MODIFIED" if old_code else "ADDED",
                old_code=old_code, new_code=new_code,
                start_line_old=sl_old,
                start_line_new=sl_new
            ))

    return {"atomic_patches": final_patches, "commit_message": commit_message, "lang": lang}

def baseline_phase1_node(state: WorkflowState) -> Dict:
    """Phase 3.1.2 Baseline (No Denoising) 节点入口"""
    print(f"--- Phase 1 Baseline: Commit {state['commit_hash']} ---")
    
    repo_mgr = RepoManager(state['repo_path'])
    parser = CodeParser()
    
    patch_list = []
    lang = 'c'
    
    files = repo_mgr.get_changed_files(state['commit_hash'])
    for file_path in files:
        if not file_path.endswith(('.c', '.h', '.cpp', '.cc', '.cxx', '.hpp')): continue
        
        lang = 'c' if file_path.endswith(('.c', '.h')) else 'cpp'
        suffix = '.c' if file_path.endswith(('.c', '.h')) else '.cpp'
        
        old_content = repo_mgr.get_file_content(f"{state['commit_hash']}^", file_path)
        new_content = repo_mgr.get_file_content(state['commit_hash'], file_path)
        if old_content is None or new_content is None: continue

        old_funcs = parser.extract_functions(old_content, suffix=suffix)
        new_funcs = parser.extract_functions(new_content, suffix=suffix)

        for func_name, new_func_data in new_funcs.items():
            new_code = new_func_data['code']
            
            old_func_data = old_funcs.get(func_name, {})
            old_code = old_func_data.get('code', "")
            
            norm_old = parser.normalize(old_code)
            norm_new = parser.normalize(new_code)
            if norm_old == norm_new:
                continue
            
            sl_old = old_func_data.get('start_line', 1)
            sl_new = new_func_data.get('start_line', 1)
            
            raw_diff = parser.generate_diff(file_path, func_name, old_code, new_code, old_start=sl_old, new_start=sl_new)
            if not raw_diff: continue
            
            patch_list.append(AtomicPatch(
                file_path=file_path, function_name=func_name,
                clean_diff=raw_diff, raw_diff=raw_diff,
                change_type="MODIFIED" if old_code else "ADDED",
                old_code=old_code, new_code=new_code,
                start_line_old=sl_old,
                start_line_new=sl_new
            ))
        grouped_patches = [[patch] for patch in patch_list]
    return {"atomic_patches": patch_list, "grouped_patches":grouped_patches,
            "commit_message": repo_mgr.get_commit_message(state['commit_hash']), "lang": lang}