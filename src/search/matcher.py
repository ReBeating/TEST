import difflib
import re
from typing import List, Tuple, Dict
from tqdm import tqdm
from core.models import PatchFeatures, MatchEvidence, MatchTrace, SearchResultItem, AlignedTrace
from core.state import WorkflowState
import concurrent.futures
from core.indexer import GlobalSymbolIndexer, BenchmarkSymbolIndexer, tokenize_code

# ==========================================
# 0. Parameter Configuration
# ==========================================

class MatcherConfig:
    """
    Core Configuration for Vulnerability Matcher
    Optimized: Reduced from 22 parameters to 9 core parameters
    """
    
    # ========== Similarity Thresholds (2) ==========
    # [Removed] ANCHOR_SIMILARITY_THRESHOLD - Anchor checks moved to similarity ranking, no hard threshold
    # [Removed] MATCH_SCORE_THRESHOLD - Use similarity directly in DP alignment, no pre-filtering
    VERDICT_THRESHOLD_STRONG = 0.9         # Strong signal line: above this considered feature complete
    VERDICT_THRESHOLD_WEAK = 0.4           # Weak signal line: below this considered feature missing
    
    # ========== DP Alignment Parameters (1) ==========
    BASE_GAP_PENALTY = 0.1                 # Base gap penalty
    
    # ========== 2D Weighted Score Weights (2) ==========
    # final_score = α * slice_score + β * anchor_score
    WEIGHT_SLICE = 0.5                     # α: Slice overall similarity weight
    WEIGHT_ANCHOR = 0.5                    # β: Anchor match weight
    
    # ========== Search Limit Parameters (2) ==========
    SEARCH_LIMIT_FAST = 2000                # Limit for fast candidate function search

# ==========================================
# 1. Basic Utility Functions
# ==========================================

def remove_comments_from_code(code: str) -> str:
    """
    Preprocessing: Remove all comments (including multi-line types) from code, preserving newlines to maintain line number correspondence.
    """
    # 1. Remove block comments /* ... */
    # Use non-greedy match, re.DOTALL allows . to match newlines
    def replace_block_comment(match):
        # Replace comment content with same number of newlines to keep line numbers consistent
        return '\n' * match.group(0).count('\n')
    
    code = re.sub(r'/\*.*?\*/', replace_block_comment, code, flags=re.DOTALL)
    
    # 2. Remove single line comments // ...
    code = re.sub(r'//.*', '', code)
    
    return code

def extract_line_number(line: str) -> int:
    """Extract line number from code line, assuming format '[ 123] ...'"""
    match = re.match(r'^\[\s*(\d+)\]', line)
    if match:
        return int(match.group(1))
    return -1

def normalize_program_structure(code_str: str, has_line_markers: bool = False) -> Tuple[List[str], List[List[int]]]:
    """
    Normalize code structure:
    1. Remove comments
    2. Merge multi-line statements based on parenthesis depth (solving function call splitting)
    Returns: (Normalized code lines list, List of original line numbers corresponding to each line)
    """
    clean_code = remove_comments_from_code(code_str)
    raw_lines = clean_code.splitlines()
    
    normalized_lines = []
    line_mapping = []
    
    current_buffer = []
    current_indices = []
    paren_depth = 0
    
    for idx, line in enumerate(raw_lines):
        # 1. Preprocess current line
        original_line = line
        
        # Remove line head marker [ 123] (if any)
        if has_line_markers:
            line = re.sub(r'^\[\s*\d+\]', '', line)
            
        stripped = line.strip()
        if not stripped:
            continue
            
        current_buffer.append(stripped)
        # If line markers provided, try to extract real line number, else use 0-based index + 1
        if has_line_markers:
            ln = extract_line_number(original_line)
            current_indices.append(ln if ln != -1 else (idx + 1))
        else:
            current_indices.append(idx + 1)
        
        # 2. Parenthesis depth scan (simple string cleaning then count)
        # Replace string content to prevent interference "a)"
        check_line = re.sub(r'".*?"', '""', line)
        check_line = re.sub(r"'.*?'", "''", check_line)
        
        for char in check_line:
            if char == '(': paren_depth += 1
            elif char == ')': paren_depth -= 1
            
        # 3. Decision to merge
        # Allow some tolerance, prevent paren_depth from being negative for long
        if paren_depth < 0: paren_depth = 0
        
        if paren_depth == 0:
            # Complete statement, Flush
            joined_line = " ".join(current_buffer)
            if has_line_markers and current_indices:
                # [Fix] Restore line number marker for the first line of the block
                first_ln = current_indices[0]
                joined_line = f"[{first_ln:4d}] {joined_line}"
                
            normalized_lines.append(joined_line)
            line_mapping.append(current_indices)
            current_buffer = []
            current_indices = []
            
    # Flush remaining
    if current_buffer:
        joined_line = " ".join(current_buffer)
        if has_line_markers and current_indices:
             first_ln = current_indices[0]
             joined_line = f"[{first_ln:4d}] {joined_line}"
             
        normalized_lines.append(joined_line)
        line_mapping.append(current_indices)
        
    return normalized_lines, line_mapping

# [New] Ignored control flow keywords, not considered as function calls
CONTROL_KEYWORDS = {'if', 'for', 'while', 'switch', 'return', 'sizeof', 'likely', 'unlikely'}

def extract_function_name(line: str) -> str:
    """
    Extract main function name from code line.
    Return None if no obvious function call found.
    """
    # Simple heuristic: find 'name(' pattern
    # Exclude control keywords
    matches = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', line)
    candidates = [m for m in matches if m not in CONTROL_KEYWORDS]
    
    if not candidates:
        return None
    
    # If multiple, usually the first one is the non-keyword function call
    # e.g. "err = check_ctx_reg(env, ...)" -> check_ctx_reg
    return candidates[0]

def normalize_code_line(line: str) -> str:
    """
    [Deprecated] Keep for compatibility with old code, internal logic simplified.
    Mainly used to quickly check if line is empty.
    """
    # Remove line number [ 123]
    line = re.sub(r'^\[\s*\d+\]', '', line)
    # Remove single line comment
    line = re.sub(r'//.*', '', line)
    # Remove block comment marker
    line = re.sub(r'/\*.*?\*/', '', line)
    # Remove whitespace
    return re.sub(r'\s+', '', line).strip()

def tokenize_line(line: str) -> List[str]:
    """
    Convert code line into fine-grained Token list.
    Supports CamelCase, snake_case, digit splitting.
    [Modified] All tokens converted to lowercase, making len and Len considered same.
    """
    # 1. Preprocess: remove line number, comments
    line = re.sub(r'^\[\s*\d+\]', '', line) # Remove line numbers like [ 123]
    line = re.sub(r'//.*', '', line)        # Remove line comments
    line = re.sub(r'/\*.*?\*/', '', line)   # Remove inline block comments
    
    # 2. Initial tokenization: identifiers/digits vs symbols
    # \w+ match alphanumeric underscore
    # [^\w\s] match non-word non-whitespace chars (symbols)
    # [Modified] Prioritize -> operator as a single token to avoid splitting it
    raw_tokens = re.findall(r'->|\w+|[^\w\s]', line)
    
    final_tokens = []
    for token in raw_tokens:
        # If it is identifier or number (contains letters, digits, underscores)
        if re.match(r'^\w+$', token):
            # Sub-tokenization logic
            # A. Split by underscore
            parts = token.split('_')
            for part in parts:
                if not part: continue
                
                # B. Split by CamelCase and digits
                # Regex logic:
                # [A-Z]?[a-z]+ : Word starting with optional uppercase (e.g. "Word", "word")
                # [A-Z]+(?=[A-Z][a-z]|\d|\W|$) : Consecutive uppercase letters (e.g. "XML" in "XMLParser")
                # [A-Z]+ : Remaining uppercase letters
                # \d+ : Digits
                sub_parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|[A-Z]+|\d+', part)
                
                if sub_parts:
                    # [Modified] Unify to lowercase
                    final_tokens.extend([p.lower() for p in sub_parts])
                else:
                    # Fallback (should rarely happen for \w+)
                    final_tokens.append(part.lower())
        else:
            # Symbols (Operators, Punctuation) - do not convert to lowercase
            final_tokens.append(token)
    
    # [Optimized] Filter out punctuation marks with minimal semantic contribution to prevent interference with match score
    IGNORED_TOKENS = {';', '(', ')', '{', '}', '[', ']', ',', '.'}
    return [t for t in final_tokens if t not in IGNORED_TOKENS]

def get_sort_key(item: SearchResultItem):
    """
    Sorting key for search results.
    Priority: VULNERABLE > UNKNOWN > PATCHED > MISMATCH
    Secondary: Confidence Score
    """
    priority_map = {
        "VULNERABLE": 3,
        "UNKNOWN": 2,
        "PATCHED": 1,
        "MISMATCH": 0
    }
    p = priority_map.get(item.verdict, 0)
    return (p, item.confidence)
        
# ==========================================
# 2. Line Correspondence Class (For competitive scoring)
# ==========================================

class LineMappingType:
    """Line mapping type enumeration"""
    COMMON = "COMMON"       # Exactly same lines on both sides
    MODIFIED = "MODIFIED"   # Modified line pairs (pre's some line became post's some line)
    VULN_ONLY = "VULN_ONLY" # Exists only in pre-patch (Deleted)
    FIX_ONLY = "FIX_ONLY"   # Exists only in post-patch (Added)

class LineCorrespondence:
    """
    Pre/Post Line Correspondence
    
    For competitive scoring: When target line matches both sides of MODIFIED line pair,
    only count the one with higher similarity, avoiding duplicate scoring.
    """
    def __init__(self, pre_idx: int = None, post_idx: int = None,
                 mapping_type: str = None, pre_content: str = None, post_content: str = None):
        self.pre_idx = pre_idx      # pre-patch line index (0-based)
        self.post_idx = post_idx    # post-patch line index (0-based)
        self.mapping_type = mapping_type
        self.pre_content = pre_content
        self.post_content = post_content
    
    def __repr__(self):
        return f"LineCorrespondence({self.mapping_type}: pre[{self.pre_idx}] <-> post[{self.post_idx}])"


class DualChannelMatcher:
    def __init__(self, s_pre: str, s_post: str,
                 pre_origins: List[str] = None, pre_impacts: List[str] = None,
                 post_origins: List[str] = None, post_impacts: List[str] = None):
        
        # [Modified] Use normalize_program_structure for normalization
        # Slices usually carry line markers [ 123], so enable has_line_markers=True
        self.s_pre_lines, self.s_pre_map = normalize_program_structure(s_pre, has_line_markers=True)
        self.s_post_lines, self.s_post_map = normalize_program_structure(s_post, has_line_markers=True)

        # Build line correspondences and tags
        self.pre_tags, self.post_tags, self.line_correspondences = self._tag_regions_with_correspondence()
        self.has_vuln_features = 'VULN' in self.pre_tags
        self.has_fix_features = 'FIX' in self.post_tags
        
        # Build lookup table for MODIFIED line pairs (for competitive scoring)
        self.modified_pre_to_post = {}  # pre_idx -> post_idx
        self.modified_post_to_pre = {}  # post_idx -> pre_idx
        for corr in self.line_correspondences:
            if corr.mapping_type == LineMappingType.MODIFIED:
                self.modified_pre_to_post[corr.pre_idx] = corr.post_idx
                self.modified_post_to_pre[corr.post_idx] = corr.pre_idx
        
        # [New] Precise Anchors
        # Clean them similarly to how we filter lines (remove empty ones)
        # Anchors should also correspond one-to-one, simple handling here, use normalized line list during matching
        self.pre_origins = [l.strip() for l in (pre_origins or []) if tokenize_line(l)]
        self.pre_impacts = [l.strip() for l in (pre_impacts or []) if tokenize_line(l)]
        self.post_origins = [l.strip() for l in (post_origins or []) if tokenize_line(l)]
        self.post_impacts = [l.strip() for l in (post_impacts or []) if tokenize_line(l)]
        
    def _tag_regions_with_correspondence(self) -> Tuple[List[str], List[str], List[LineCorrespondence]]:
        """
        Build line tags and line correspondences
        
        Returns:
            pre_tags: tags for each line in pre-patch (COMMON/VULN)
            post_tags: tags for each line in post-patch (COMMON/FIX)
            correspondences: list of line correspondences
        """
        # Tokenize for comparison
        pre_tokens = [tuple(tokenize_line(x)) for x in self.s_pre_lines]
        post_tokens = [tuple(tokenize_line(x)) for x in self.s_post_lines]
        
        matcher = difflib.SequenceMatcher(None, pre_tokens, post_tokens)
        
        pre_tags = ['VULN'] * len(self.s_pre_lines)
        post_tags = ['FIX'] * len(self.s_post_lines)
        correspondences = []
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                # COMMON: Completely identical lines
                for k in range(i2 - i1):
                    pre_idx = i1 + k
                    post_idx = j1 + k
                    pre_tags[pre_idx] = 'COMMON'
                    post_tags[post_idx] = 'COMMON'
                    correspondences.append(LineCorrespondence(
                        pre_idx=pre_idx,
                        post_idx=post_idx,
                        mapping_type=LineMappingType.COMMON,
                        pre_content=self.s_pre_lines[pre_idx] if pre_idx < len(self.s_pre_lines) else None,
                        post_content=self.s_post_lines[post_idx] if post_idx < len(self.s_post_lines) else None
                    ))
                    
            elif tag == 'replace':
                # MODIFIED: Replaced line pairs
                # Try one-to-one mapping (when block size matches)
                pre_block_size = i2 - i1
                post_block_size = j2 - j1
                
                if pre_block_size == post_block_size:
                    # One-to-one MODIFIED line pairs
                    for k in range(pre_block_size):
                        pre_idx = i1 + k
                        post_idx = j1 + k
                        # Keep tags VULN/FIX (don't change to COMMON)
                        correspondences.append(LineCorrespondence(
                            pre_idx=pre_idx,
                            post_idx=post_idx,
                            mapping_type=LineMappingType.MODIFIED,
                            pre_content=self.s_pre_lines[pre_idx] if pre_idx < len(self.s_pre_lines) else None,
                            post_content=self.s_post_lines[post_idx] if post_idx < len(self.s_post_lines) else None
                        ))
                else:
                    # Block sizes differ, try content similarity matching
                    matched_post = set()
                    for pi in range(i1, i2):
                        best_match = None
                        best_sim = 0.5  # Min similarity threshold
                        for pj in range(j1, j2):
                            if pj in matched_post:
                                continue
                            # Calculate token similarity
                            sm = difflib.SequenceMatcher(None, pre_tokens[pi], post_tokens[pj])
                            sim = sm.ratio()
                            if sim > best_sim:
                                best_sim = sim
                                best_match = pj
                        
                        if best_match is not None:
                            matched_post.add(best_match)
                            correspondences.append(LineCorrespondence(
                                pre_idx=pi,
                                post_idx=best_match,
                                mapping_type=LineMappingType.MODIFIED,
                                pre_content=self.s_pre_lines[pi],
                                post_content=self.s_post_lines[best_match]
                            ))
                        else:
                            # No match, mark as VULN_ONLY
                            correspondences.append(LineCorrespondence(
                                pre_idx=pi,
                                post_idx=None,
                                mapping_type=LineMappingType.VULN_ONLY,
                                pre_content=self.s_pre_lines[pi],
                                post_content=None
                            ))
                    
                    # Mark unmatched post lines as FIX_ONLY
                    for pj in range(j1, j2):
                        if pj not in matched_post:
                            correspondences.append(LineCorrespondence(
                                pre_idx=None,
                                post_idx=pj,
                                mapping_type=LineMappingType.FIX_ONLY,
                                pre_content=None,
                                post_content=self.s_post_lines[pj]
                            ))
                            
            elif tag == 'delete':
                # VULN_ONLY: Exists only in pre-patch
                for k in range(i2 - i1):
                    pre_idx = i1 + k
                    correspondences.append(LineCorrespondence(
                        pre_idx=pre_idx,
                        post_idx=None,
                        mapping_type=LineMappingType.VULN_ONLY,
                        pre_content=self.s_pre_lines[pre_idx],
                        post_content=None
                    ))
                    
            elif tag == 'insert':
                # FIX_ONLY: Exists only in post-patch
                for k in range(j2 - j1):
                    post_idx = j1 + k
                    correspondences.append(LineCorrespondence(
                        pre_idx=None,
                        post_idx=post_idx,
                        mapping_type=LineMappingType.FIX_ONLY,
                        pre_content=None,
                        post_content=self.s_post_lines[post_idx]
                    ))
        
        return pre_tags, post_tags, correspondences

    def _align_channel(self, slice_lines: List[str], slice_tags: List[str], 
                       slice_map: List[List[int]], # [New] Slice line number mapping
                       target_lines: List[str], target_map: List[List[int]] # [New] Target line number mapping
                       ) -> Tuple[float, float, List[MatchTrace], List[str], List[AlignedTrace]]:
        # ... (previous code omitted for brevity in call, but need to be careful with replace)
        # Actually I cannot omit code in replace tool. I should target the specific block.
        
        # I'll update the signature and return line first? No, I need to do it in chunks or one big chunk.
        # Let's target the logic after DP matrix construction.
        pass
        n, m = len(slice_lines), len(target_lines)
        
        # [New] Safety check: Prevent excessive matrix computation
        # If matrix elements exceed 2 million (e.g., 200 slice lines x 10000 target lines), abort directly
        # if n * m > 2_000_000:
        #     print(f"  [Matcher] Warning: Matrix too large ({n}x{m} = {n*m}), skipping alignment.")
        #     return 0.0, 0.0, [], []

        # [Optimization] Preprocess Tokenized lines
        slice_tokens = [tokenize_line(line) for line in slice_lines]
        target_tokens = [tokenize_line(line) for line in target_lines]

        # [New] Extract slice line numbers (use the first line number in the normalized mapping)
        slice_line_nos = [m[0] if m else -1 for m in slice_map]

        # [Optimization] Pre-calculate similarity matrix
        sim_matrix = [[0.0] * m for _ in range(n)]
        for i in range(n):
            t1 = slice_tokens[i]
            
            if not t1:
                for j in range(m):
                    t2 = target_tokens[j]
                    sim_matrix[i][j] = 1.0 if not t2 else 0.0
                continue

            # Reuse SequenceMatcher with b=t1 (list of tokens)
            sm = difflib.SequenceMatcher(None, b=t1)
            
            for j in range(m):
                t2 = target_tokens[j]
                if not t2:
                    sim_matrix[i][j] = 0.0
                    continue
                
                sm.set_seq1(t2)
                
                # quick_ratio pre-filtering (hardcoded 0.5)
                if sm.quick_ratio() < 0.5:
                    sim_matrix[i][j] = 0.0
                    continue
                
                # [Simplify] Use ratio() directly, function call similarity is calculated separately by _compute_func_score
                sim_matrix[i][j] = sm.ratio()

        # 1. DP Matrix Construction
        # dp[i][j] stores the maximum cumulative score reaching (i, j)
        dp = [[0.0] * (m + 1) for _ in range(n + 1)]
        
        # [Phase 1: Use MatcherConfig.BASE_GAP_PENALTY]
        base_gap = MatcherConfig.BASE_GAP_PENALTY

        # Pre-calc gap costs for transitions i-1 -> i
        # [Modified] Size n+2 to handle lookahead for row i (using bond i->i+1)
        step_gap_costs = [0.0] * (n + 2)
        for i in range(2, n + 1):
            l1 = slice_line_nos[i-2]
            l2 = slice_line_nos[i-1]
            if l1 != -1 and l2 != -1 and l2 >= l1:
                dist = max(1, l2 - l1)
                step_gap_costs[i] = base_gap / float(dist)
            else:
                step_gap_costs[i] = base_gap # Default
        
        # step_gap_costs[n+1] implicitly 0.0 (Trailing gap free) or set default
        step_gap_costs[n+1] = 0.0
        step_gap_costs[1] = 0.0 # No penalty for leading gaps

        for i in range(1, n + 1):
            # [Fix] Row i horizontal moves (Target gaps) occur AFTER matching line i.
            # So they correspond to the gap between i and i+1.
            current_gap_cost = step_gap_costs[i+1]
            
            for j in range(1, m + 1):
                sim = sim_matrix[i-1][j-1]
                # [Modified] Use similarity directly as match score, no pre-filtering threshold
                match_score = sim
                
                # Dynamic Gap Cost at edges:
                # Trailing gap (i=n) usually not penalized or less penalized
                eff_gap_cost = 0.0 if i == n else current_gap_cost
                # Transition 1: Match (Diagonal)
                score_match = dp[i-1][j-1] + match_score
                
                # Transition 2: Skip Target (Horizontal) - "Gap in Target"
                score_skip_tgt = dp[i][j-1] - eff_gap_cost
                
                # Transition 3: Skip Slice (Vertical) - "Missing in Target"
                # No explicit penalty needed here, it just fails to gain score.
                score_skip_slice = dp[i-1][j]

                dp[i][j] = max(score_match, score_skip_tgt, score_skip_slice)
        # for i in range(n + 1):
        #     for j in range(m + 1):
        #         print(f"{dp[i][j]:.2f}", end=" ")
        #     print()
        # 2. Backtracking & Statistics
        matched_ctx_len, total_ctx_len = 0.0, 0.0
        matched_feat_len, total_feat_len = 0.0, 0.0
        matches, missing = [], []
        
        # We need to accumulate the *actual* penalties incurred on the optimal path
        # to correctly adjust the final score.
        accumulated_gap_penalty = 0.0
        
        alignment_trace = []
        
        i, j = n, m
        while i > 0 and j > 0:
            current_score = dp[i][j]
            sim = sim_matrix[i-1][j-1]
            # [Modified] Use similarity directly as match score
            match_gain = sim
            # [Fix] Match the forward pass gap cost logic
            gap_cost = step_gap_costs[i+1]
            
            # Check Match
            # Use strict epsilon for float equality check
            # [Modified] As long as sim > 0, consider it a match possibility
            if abs(current_score - (dp[i-1][j-1] + match_gain)) < 1e-9 and sim > 0:
                # --- MATCH ---
                line_len = len(slice_tokens[i-1])
                is_feature = (slice_tags[i-1] != 'COMMON')
                
                if is_feature:
                    total_feat_len += line_len
                    matched_feat_len += line_len * sim
                    matches.append(MatchTrace(
                        slice_line=slice_lines[i-1], target_line=target_lines[j-1], 
                        line_no=j, similarity=sim
                    ))
                else:
                    total_ctx_len += line_len
                    matched_ctx_len += line_len * sim
                
                alignment_trace.append(AlignedTrace(
                    slice_line=slice_lines[i-1], target_line=target_lines[j-1], 
                    line_no=j, similarity=sim, tag=slice_tags[i-1]
                ))
                i -= 1
                j -= 1
                
            elif abs(current_score - (dp[i][j-1] - gap_cost)) < 1e-9:
                # --- SKIP TARGET (Gap) ---
                accumulated_gap_penalty += gap_cost
                j -= 1
                
            else:
                # --- SKIP SLICE (Missing) ---
                line_len = len(slice_tokens[i-1])
                is_feature = (slice_tags[i-1] != 'COMMON')
                
                if is_feature:
                    total_feat_len += line_len
                    missing.append(slice_lines[i-1])
                else:
                    total_ctx_len += line_len
                
                alignment_trace.append(AlignedTrace(
                    slice_line=slice_lines[i-1], target_line=None, 
                    line_no=None, similarity=0.0, tag=slice_tags[i-1]
                ))
                i -= 1

        # Process Pre-matches (Skipped Slices at start)
        while i > 0:
            line_len = len(slice_tokens[i-1])
            is_feature = (slice_tags[i-1] != 'COMMON')
            
            if is_feature:
                total_feat_len += line_len
                missing.append(slice_lines[i-1])
            else:
                total_ctx_len += line_len
            
            alignment_trace.append(AlignedTrace(
                slice_line=slice_lines[i-1], target_line=None, 
                line_no=None, similarity=0.0, tag=slice_tags[i-1]
            ))
            i -= 1
        
        alignment_trace.reverse()
        matches.reverse()
        # Missing list is already somewhat ordered but reversed logic applies if we cared about order there.
        
        # 3. Final Scoring with Integrated Penalty
        # Convert accumulated penalty points (which are in 'Score Units') to 'Token Units'
        valid_slice_lines_count = sum(1 for t in slice_tokens if t)
        total_slice_tokens_count = sum(len(t) for t in slice_tokens)
        
        # [Note] valid_slice_lines_count == 0 is reasonable
        # e.g. s_post of pure deletion patch has only comment header, normalized to empty
        if valid_slice_lines_count == 0:
            # Empty slice: return 0 score, no warning needed
            penalty_deduction = 0.0
        else:
            avg_tokens = total_slice_tokens_count / valid_slice_lines_count
            penalty_deduction = accumulated_gap_penalty * avg_tokens
        
        # [Modified] Proportional Decay
        # Structural gaps weaken the confidence of ALL matches in the alignment equally.
        raw_matched_total = matched_ctx_len + matched_feat_len
        
        if raw_matched_total > 1e-9:
            # How much of the "Raw Match" survives the "Gap Penalty"?
            decay_factor = max(0.0, raw_matched_total - penalty_deduction) / raw_matched_total
        else:
            decay_factor = 0.0
            
        # print(f'  [DEBUG] RawMatch: {raw_matched_total:.1f}, Penalty: {penalty_deduction:.1f}, DecayFactor: {decay_factor:.4f}')

        # Apply decay to both components
        final_matched_ctx = matched_ctx_len * decay_factor
        final_matched_feat = matched_feat_len * decay_factor
        
        score_ctx = final_matched_ctx / total_ctx_len if total_ctx_len > 0 else 0.0
        score_feat = final_matched_feat / total_feat_len if total_feat_len > 0 else 0.0
        
        # Calculate Total Score
        final_matched_total = final_matched_ctx + final_matched_feat
        total_tokens = total_ctx_len + total_feat_len
        score_total = final_matched_total / total_tokens if total_tokens > 0 else 0.0

        # print(score_ctx, score_feat, score_total)
        return score_ctx, score_feat, score_total, matches, missing, alignment_trace

    def _apply_competitive_scoring(self, aligned_vuln: List[AlignedTrace], aligned_fix: List[AlignedTrace],
                                    target_lines: List[str]) -> Tuple[float, float, float, float]:
        """
        Apply competitive scoring for mixed-type patches
        
        Core rule: For MODIFIED line pairs, if a target line matches corresponding lines in both pre and post,
        only count similarity for the better match to avoid duplicate scoring.
        
        Args:
            aligned_vuln: pre-patch alignment results
            aligned_fix: post-patch alignment results
            target_lines: normalized target line list
            
        Returns:
            (score_vuln, score_fix, score_feat_vuln, score_feat_fix)
            - score_vuln/score_fix: Adjusted overall similarity
            - score_feat_vuln/score_feat_fix: Local feature similarity
        """
        # If no MODIFIED line pairs, compute scores directly from alignment results
        if not self.modified_pre_to_post:
            return self._compute_scores_from_alignment(aligned_vuln, aligned_fix)
        
        # Build mapping: target line index -> alignment info
        # line_no in aligned_vuln/aligned_fix is 1-based index of target line
        vuln_target_matches = {}  # target_line_idx -> (slice_idx, similarity)
        fix_target_matches = {}   # target_line_idx -> (slice_idx, similarity)
        
        for slice_idx, trace in enumerate(aligned_vuln):
            if trace.line_no is not None and trace.similarity > 0:
                target_idx = trace.line_no - 1  # Convert to 0-based
                # If same target line matched multiple times, keep highest similarity
                if target_idx not in vuln_target_matches or trace.similarity > vuln_target_matches[target_idx][1]:
                    vuln_target_matches[target_idx] = (slice_idx, trace.similarity)
        
        for slice_idx, trace in enumerate(aligned_fix):
            if trace.line_no is not None and trace.similarity > 0:
                target_idx = trace.line_no - 1
                if target_idx not in fix_target_matches or trace.similarity > fix_target_matches[target_idx][1]:
                    fix_target_matches[target_idx] = (slice_idx, trace.similarity)
        
        # Competitive Scoring: Handle duplicate scoring of MODIFIED line pairs
        # Record tokens to deduct from either side
        vuln_deduction = 0.0
        fix_deduction = 0.0
        
        for pre_idx, post_idx in self.modified_pre_to_post.items():
            # Check if this MODIFIED line pair matched to the SAME target line
            # 1. Find matched target for pre_idx
            vuln_matched_target = None
            vuln_sim = 0.0
            if pre_idx < len(aligned_vuln) and aligned_vuln[pre_idx].line_no is not None:
                vuln_matched_target = aligned_vuln[pre_idx].line_no - 1
                vuln_sim = aligned_vuln[pre_idx].similarity
            
            # 2. Find matched target for post_idx
            fix_matched_target = None
            fix_sim = 0.0
            if post_idx < len(aligned_fix) and aligned_fix[post_idx].line_no is not None:
                fix_matched_target = aligned_fix[post_idx].line_no - 1
                fix_sim = aligned_fix[post_idx].similarity
            
            # 3. If both matched same target line, perform competition
            if vuln_matched_target is not None and fix_matched_target is not None:
                if vuln_matched_target == fix_matched_target:
                    # Competition: keep only higher similarity
                    pre_tokens = len(tokenize_line(self.s_pre_lines[pre_idx])) if pre_idx < len(self.s_pre_lines) else 0
                    post_tokens = len(tokenize_line(self.s_post_lines[post_idx])) if post_idx < len(self.s_post_lines) else 0
                    
                    if vuln_sim >= fix_sim:
                        # vuln wins, deduct from fix
                        fix_deduction += post_tokens * fix_sim
                    else:
                        # fix wins, deduct from vuln
                        vuln_deduction += pre_tokens * vuln_sim
        
        # Calculate base scores
        score_vuln, score_fix, score_feat_vuln, score_feat_fix = self._compute_scores_from_alignment(aligned_vuln, aligned_fix)
        
        # Apply deductions (convert to score adjustment)
        pre_total_tokens = sum(len(tokenize_line(line)) for line in self.s_pre_lines)
        post_total_tokens = sum(len(tokenize_line(line)) for line in self.s_post_lines)
        
        if pre_total_tokens > 0 and vuln_deduction > 0:
            deduction_ratio = vuln_deduction / pre_total_tokens
            score_vuln = max(0.0, score_vuln - deduction_ratio)
        
        if post_total_tokens > 0 and fix_deduction > 0:
            deduction_ratio = fix_deduction / post_total_tokens
            score_fix = max(0.0, score_fix - deduction_ratio)
        
        return score_vuln, score_fix, score_feat_vuln, score_feat_fix

    def _compute_anchor_score(self, aligned_trace: List[AlignedTrace], raw_lines: List[str],
                               slice_map: List[List[int]],
                               sources: List[str], sinks: List[str]) -> float:
        """
        Compute anchor matching score
        
        Idea:
        - Check if source (data source) and sink (impact point) match to target
        - Return score between 0-1
        
        Note: raw_lines are normalized via normalize_program_structure, multi-line statements merged.
        sources/sinks are original multi-line anchors.
        
        Matching Strategy: Based on line number matching
        - Extract line number from anchor (e.g., "[ 227] ..." -> 227)
        - Find line in slice_lines containing that line number (normalization may merge 227-228 into one line)
        
        Args:
            aligned_trace: Alignment trace results
            raw_lines: Normalized slice line list
            slice_map: Original line number list for each line (returned by normalize_program_structure)
            sources: origin anchor line list
            sinks: impact anchor line list
        
        Returns:
            anchor_score: Anchor match score [0, 1]
                - 1.0: Both anchors matched
                - 0.5: Only one anchor matched
                - 0.0: Neither anchor matched
                - 1.0: No explicit anchors (no penalty)
        """
        if not aligned_trace:
            return 0.0
        
        n_trace = len(aligned_trace)
        
        def extract_line_no(s):
            """Extract line number from line with line number marker"""
            match = re.match(r'^\[\s*(\d+)\]', s.strip())
            return int(match.group(1)) if match else -1
        
        def get_indices_by_line_number(key_lines):
            """
            Find corresponding indices in slice_lines based on line numbers.
            
            Args:
                key_lines: anchor line list (with line number markers)
            
            Returns:
                List of matched slice_lines indices
            """
            indices = set()
            if not key_lines:
                return list(indices)
            
            # Extract all anchor line numbers
            anchor_line_nos = set()
            for anchor in key_lines:
                ln = extract_line_no(anchor)
                if ln != -1:
                    anchor_line_nos.add(ln)
            
            if not anchor_line_nos:
                return list(indices)
            
            # Find indices in slice_map containing these line numbers
            for i, orig_line_nos in enumerate(slice_map):
                # orig_line_nos is a list representing original line numbers merged into normalized line i
                # e.g.: [227, 228] means line i was merged from original lines 227, 228
                for ln in orig_line_nos:
                    if ln in anchor_line_nos:
                        indices.add(i)
                        break
            
            return list(indices)
        
        has_sources = bool(sources)
        has_sinks = bool(sinks)
        
        # If no explicit anchors, return full score (no penalty)
        if not has_sources and not has_sinks:
            return 1.0
        
        source_score = 0.0
        sink_score = 0.0
        anchor_count = 0
        
        # Source Check - Use line number matching
        if has_sources:
            anchor_count += 1
            explicit_source_indices = get_indices_by_line_number(sources)
            if explicit_source_indices:
                sims = max((aligned_trace[i].similarity for i in explicit_source_indices if i < n_trace), default=0.0)
                source_score = sims  # Use similarity directly as score
        
        # Sink Check - Use line number matching
        if has_sinks:
            anchor_count += 1
            explicit_sink_indices = get_indices_by_line_number(sinks)
            if explicit_sink_indices:
                sims = max((aligned_trace[i].similarity for i in explicit_sink_indices if i < n_trace), default=0.0)
                sink_score = sims  # Use similarity directly as score
        
        # Calculate mean score
        if anchor_count == 0:
            return 1.0
        
        return (source_score + sink_score) / anchor_count
    
    def _compute_scores_from_alignment(self, aligned_vuln: List[AlignedTrace], aligned_fix: List[AlignedTrace]
                                       ) -> Tuple[float, float, float, float]:
        """
        Compute 4 scores from alignment results
        
        Returns:
            (score_vuln, score_fix, score_feat_vuln, score_feat_fix)
        """
        # Pre-patch score calculation
        vuln_total_tokens = 0
        vuln_matched_tokens = 0.0
        vuln_feat_total = 0
        vuln_feat_matched = 0.0
        
        for i, trace in enumerate(aligned_vuln):
            tokens = len(tokenize_line(self.s_pre_lines[i])) if i < len(self.s_pre_lines) else 0
            vuln_total_tokens += tokens
            
            is_feature = (self.pre_tags[i] != 'COMMON') if i < len(self.pre_tags) else False
            if is_feature:
                vuln_feat_total += tokens
            
            if trace.similarity > 0:
                vuln_matched_tokens += tokens * trace.similarity
                if is_feature:
                    vuln_feat_matched += tokens * trace.similarity
        
        # Post-patch score calculation
        fix_total_tokens = 0
        fix_matched_tokens = 0.0
        fix_feat_total = 0
        fix_feat_matched = 0.0
        
        for i, trace in enumerate(aligned_fix):
            tokens = len(tokenize_line(self.s_post_lines[i])) if i < len(self.s_post_lines) else 0
            fix_total_tokens += tokens
            
            is_feature = (self.post_tags[i] != 'COMMON') if i < len(self.post_tags) else False
            if is_feature:
                fix_feat_total += tokens
            
            if trace.similarity > 0:
                fix_matched_tokens += tokens * trace.similarity
                if is_feature:
                    fix_feat_matched += tokens * trace.similarity
        
        # Calculate scores
        score_vuln = vuln_matched_tokens / vuln_total_tokens if vuln_total_tokens > 0 else 0.0
        score_fix = fix_matched_tokens / fix_total_tokens if fix_total_tokens > 0 else 0.0
        score_feat_vuln = vuln_feat_matched / vuln_feat_total if vuln_feat_total > 0 else 0.0
        score_feat_fix = fix_feat_matched / fix_feat_total if fix_feat_total > 0 else 0.0
        
        return score_vuln, score_fix, score_feat_vuln, score_feat_fix

    def match(self, target_function_code: str, target_start_line: int = 1) -> MatchEvidence:
        # [Preprocess] Use normalization
        target_lines, target_map = normalize_program_structure(target_function_code, has_line_markers=False)

        # Matching
        # Pass map to correctly compute line spacing and restore original line numbers
        ctx_a, score_feat_vuln, total_vuln, _, _, aligned_vuln = self._align_channel(
            self.s_pre_lines, self.pre_tags, self.s_pre_map,
            target_lines, target_map
        )
        ctx_b, score_feat_fix, total_fix, _, _, aligned_fix = self._align_channel(
            self.s_post_lines, self.post_tags, self.s_post_map,
            target_lines, target_map
        )
        
        # [New] Apply competitive scoring for mixed-type patches
        if self.has_vuln_features and self.has_fix_features and self.modified_pre_to_post:
            # Has MODIFIED line pairs, use competitive scoring
            score_vuln, score_fix, score_feat_vuln, score_feat_fix = self._apply_competitive_scoring(
                aligned_vuln, aligned_fix, target_lines
            )
        else:
            # No MODIFIED line pairs, use original scores
            score_vuln = total_vuln
            score_fix = total_fix
        
        if not self.has_vuln_features:
            score_feat_vuln = -1.0

        if not self.has_fix_features:
            score_feat_fix = -1.0
        
        # Correct line_no in Trace to real line numbers
        # [Modified] Use target_map to map normalized line index back to original file line number
        # [New] Use target_start_line to compute absolute line number in file
        def correct_line_numbers(trace_list):
             for m in trace_list:
                # m.line_no is 1-based index in target_lines
                if m.line_no is not None and 0 < m.line_no <= len(target_map):
                    # Get original line number list for this normalized line (relative to function inside)
                    original_indices = target_map[m.line_no - 1]
                    if original_indices:
                        # original_indices[0] is relative line number in function (starts from 1)
                        # Plus (target_start_line - 1) to get absolute line number in file
                        real_line_start = original_indices[0] + target_start_line - 1
                        m.line_no = real_line_start
                        # Format target_line: "[ 123] content"
                        # Use m.target_line (normalized content)
                        if m.target_line:
                            m.target_line = f"[{real_line_start:4d}] {m.target_line}"

        # correct_line_numbers(vuln_matches)
        # correct_line_numbers(fix_matches)
        correct_line_numbers(aligned_vuln)
        correct_line_numbers(aligned_fix)

        score_ctx = max(ctx_a, ctx_b)
        # Overall score: If no feature lines, use context score
        total_vuln = score_vuln if self.has_vuln_features else score_ctx
        total_fix = score_fix if self.has_fix_features else score_ctx
        verdict = "UNKNOWN"
        confidence = 0.0
        
        # ========== 2D Weighted Score Calculation ==========
        # final_score = α * slice_score + β * anchor_score
        
        # 1. Compute 2D score for Vuln channel
        anchor_score_vuln = self._compute_anchor_score(
            aligned_vuln, self.s_pre_lines, self.s_pre_map, self.pre_origins, self.pre_impacts
        )
        
        total_vuln = (
            MatcherConfig.WEIGHT_SLICE * score_vuln +
            MatcherConfig.WEIGHT_ANCHOR * anchor_score_vuln
        )
        
        # 2. Compute 2D score for Fix channel
        anchor_score_fix = self._compute_anchor_score(
            aligned_fix, self.s_post_lines, self.s_post_map, self.post_origins, self.post_impacts
        )
        
        # Weighted calculation for final_fix
        total_fix = (
            MatcherConfig.WEIGHT_SLICE * score_fix +
            MatcherConfig.WEIGHT_ANCHOR * anchor_score_fix
        )
        
        # --- Verdict Phase ---
        # Classification Verdict Logic (Consistent with Methodology.tex)
        
        # 1. Pre-screening: Garbage filter (based on overall vulnerability score)
        # If similarity to vuln pattern is too low, direct Mismatch
        if total_vuln < MatcherConfig.VERDICT_THRESHOLD_WEAK:
            verdict = "MISMATCH"
            confidence = 1.0 - total_vuln
        else:
            # --- Case 1: Pure Additive Patch ---
            # Criteria: FIX feature lines exist
            if self.has_fix_features and not self.has_vuln_features:
                if score_feat_fix >= MatcherConfig.VERDICT_THRESHOLD_STRONG:
                    verdict = "PATCHED"
                    confidence = total_fix
                else:
                    verdict = "VULNERABLE"
                    confidence = total_vuln
            
            # --- Case 2: Pure Subtractive Patch ---
            # Criteria: VULN feature lines deleted
            elif self.has_vuln_features and not self.has_fix_features:
                if score_feat_vuln < MatcherConfig.VERDICT_THRESHOLD_WEAK:
                    verdict = "PATCHED"
                    confidence = total_fix
                else:
                    verdict = "VULNERABLE"
                    confidence = total_vuln
            
            # --- Case 3: Mixed (Modified) Patch ---
            # Criteria: Compare total scores after competitive scoring
            elif self.has_vuln_features and self.has_fix_features:
                if total_fix > total_vuln:
                    verdict = "PATCHED"
                    confidence = total_fix
                else:
                    verdict = "VULNERABLE"
                    confidence = total_vuln

            # --- Case 4: No Features ---
            else:
                verdict = "UNKNOWN"
                confidence = score_ctx

        return MatchEvidence(
            verdict=verdict,
            confidence=confidence,
            score_vuln=total_vuln,           # Overall slice similarity (pre-patch vs target)
            score_fix=total_fix,             # Overall slice similarity (post-patch vs target)
            score_feat_vuln=score_feat_vuln, # Local feature similarity (VULN lines only)
            score_feat_fix=score_feat_fix,   # Local feature similarity (FIX lines only)
            aligned_vuln_traces=aligned_vuln,
            aligned_fix_traces=aligned_fix
        )

class VulnerabilitySearchEngine:
    def __init__(self, repo_path: str):
        # [Modified] No longer accept db_path, accept repo_path instead, dynamically get Indexer
        self.repo_path = repo_path
        self.indexer = GlobalSymbolIndexer(repo_path)
        self.benchmark_indexer = BenchmarkSymbolIndexer()
        
    def search_patch(self, feature: PatchFeatures, mode: str = "repo", vul_id: str = "") -> List[SearchResultItem]:
        results : List[SearchResultItem] = []
        
        if mode == "benchmark":
            print(f"[*] Using benchmark candidates for {feature.group_id[:8]}...")
            for func_name, sf in feature.slices.items():
                print(f"[*] Processing function: {func_name}")
                print(f's_pre:\n{sf.s_pre}\ns_post:\n{sf.s_post}\n')
            
            # Extract tokens from all slices (use tokenize_code consistent with database)
            # [Fix] Remove comments (including header marker comments) before extracting tokens
            search_tokens = []
            for func_name, sf in feature.slices.items():
                clean_code = remove_comments_from_code(sf.s_pre)
                tokens = tokenize_code(clean_code)
                search_tokens.extend(tokens)
            
            # Deduplicate
            search_tokens = list(set(search_tokens))
            print(f"    Extracted {len(search_tokens)} unique tokens for benchmark search")

            # Use unified Token search method
            candidates = self.benchmark_indexer.search_functions_by_tokens(vul_id, search_tokens, limit=MatcherConfig.SEARCH_LIMIT_FAST)
            
            matchers = {
                name: DualChannelMatcher(sf.s_pre, sf.s_post, 
                                         pre_origins=sf.pre_origins, pre_impacts=sf.pre_impacts,
                                         post_origins=sf.post_origins, post_impacts=sf.post_impacts) 
                for name, sf in feature.slices.items()
            }
            
            # 2. Filter and Match
            # Target function name set
            target_func_names = {p.function_name for p in feature.patches}
            for i, (path_file, compound_name, code, start_line) in enumerate(candidates):
                if i % 50 == 0:
                    print(f"    [Matcher-Bench] Processing {i+1}/{len(candidates)}: {compound_name}")

                # compound_name format: tag:version:func_name
                parts = compound_name.split(':')
                if len(parts) < 3: continue
                # if parts[0] != 'pre':
                #     continue
                func_name = parts[-1]
                
                # Only keep functions with same name
                if func_name not in target_func_names:
                    continue
                    
                matcher = matchers.get(func_name)
                if not matcher:
                    continue
                    
                # Prepare patch path (file path of the patch function)
                specific_patch = next((p for p in feature.patches if p.function_name == func_name), None)
                patch_file_path = specific_patch.file_path if specific_patch else None

                evidence = matcher.match(code, start_line)
                results.append(SearchResultItem(
                    group_id=feature.group_id,
                    repo_path=self.repo_path,
                    patch_file=patch_file_path,
                    patch_func=func_name,
                    target_file=path_file,
                    target_func=compound_name,
                    verdict=evidence.verdict,
                    confidence=evidence.confidence,
                    scores={
                        "score_vuln": evidence.score_vuln,
                        "score_fix": evidence.score_fix,
                        "score_feat_vuln": evidence.score_feat_vuln,
                        "score_feat_fix": evidence.score_feat_fix
                    },
                    evidence=evidence,
                    code_content=code
                ))
            
            results.sort(key=get_sort_key, reverse=True)
            return results
                
        else:
            # [Refactor] Iterate per function to search and match individually
            for func_name, sf in feature.slices.items():
                
                # [Optimization] Skip if s_pre is effectively empty (no context/code)
                if not any(tokenize_line(line) for line in sf.s_pre.splitlines()):
                    print(f"    [Skip] Skipping {func_name} because s_pre is empty (no context/code).")
                    continue
                
                print(f"[*] Processing function: {func_name} {feature.group_id[:8]}")
                print(f's_pre:\n{sf.s_pre}\ns_post:\n{sf.s_post}\n')
                
                # 1. Extract tokens from slice (use tokenize_code consistent with database)
                # [Fix] Remove comments (including header marker comments) before extracting tokens
                clean_code = remove_comments_from_code(sf.s_pre)
                search_tokens = tokenize_code(clean_code)
                
                print(f"    Extracted {len(search_tokens)} tokens for {func_name}")
                print(f"    Sample tokens: {search_tokens[:10]}...")

                # 2. Use unified Token search method (replace double search)
                candidates = self.indexer.search_functions_by_tokens(
                    search_tokens,
                    limit=MatcherConfig.SEARCH_LIMIT_FAST
                )
                
                print(f"    -> {len(candidates)} candidates found for {func_name}")
                
                if not candidates: continue

                # 3. Match ONLY against this function's matcher
                matcher = DualChannelMatcher(sf.s_pre, sf.s_post,
                                             pre_origins=sf.pre_origins, pre_impacts=sf.pre_impacts,
                                             post_origins=sf.post_origins, post_impacts=sf.post_impacts)
                
                for i, (file_path, target_func_name, code, start_line) in enumerate(candidates):
                    # debug only
                    # if target_func_name != 'decode_mime_type':
                    #     continue
                    # print(f'    [Matcher] Matching candidate {i+1}/{len(candidates)}: {target_func_name} in {file_path}')
                    evidence = matcher.match(code, start_line)
                    
                    # Prepare patch path
                    specific_patch = next((p for p in feature.patches if p.function_name == func_name), None)
                    patch_file_path = specific_patch.file_path if specific_patch else None

                    results.append(SearchResultItem(
                        group_id=feature.group_id,
                        repo_path=self.repo_path,
                        patch_file=patch_file_path, # [New]
                        patch_func=func_name, # Explicitly this function
                        target_file=file_path,
                        target_func=target_func_name,
                        verdict=evidence.verdict,
                        confidence=evidence.confidence,
                        scores={
                            "score_vuln": evidence.score_vuln,
                            "score_fix": evidence.score_fix,
                            "score_feat_vuln": evidence.score_feat_vuln,
                            "score_feat_fix": evidence.score_feat_fix
                        },
                        evidence=evidence,
                        code_content=code
                    ))
            
                # [Density Control] - Removed from here
                
            results.sort(key=get_sort_key, reverse=True)
            return results

# ==========================================
# 4. LangGraph Node Function
# ==========================================

def matching_node(state: WorkflowState) -> Dict:
    """
    Phase 3 Core Node: Execute vulnerability search.
    """
    features = state.get("analyzed_features", [])
    # [Critical] Must get repo path from state to initialize Indexer
    repo_path = state.get("repo_path") 
    
    mode = state.get("mode")
    vul_id = state.get("vul_id")
    
    if not features or not repo_path:
        print("  [Matching] Missing features or repo path.")
        return {"search_candidates": []}

    print(f"[*] Starting Matching Phase for {len(features)} patch groups...")
    
    all_candidates = []
    
    # Define single task
    def process_single_feature(feature):
        try:
            # [Modified] Pass repo_path and mode
            engine = VulnerabilitySearchEngine(repo_path)
            return engine.search_patch(feature, mode, vul_id)
        except Exception as e:
            print(f"  [Error] Search failed for {feature.group_id}: {e}")
            import traceback
            traceback.print_exc()
            return []

    # Sequential execution (Avoid tqdm/DB race condition issues in multi-threading)
    for feat in features:
        # [Filter] Early rejection based on Analysis Evaluation
        # If the analysis is untrustworthy or the pattern is too generic, skip search entirely.
        # if feat.semantics.forensic_report and feat.semantics.forensic_report.evaluation:
        #     eval_info = feat.semantics.forensic_report.evaluation
            
        #     if eval_info.confidence == "Low":
        #         print(f"  [Skipped] Group {feat.group_id}: Analysis Confidence is LOW.")
        #         continue
            
        #     if eval_info.homology_suitability == "Low":
        #         print(f"  [Skipped] Group {feat.group_id}: Homology Suitability is LOW.")
        #         continue

        results = process_single_feature(feat)
        if results:
            all_candidates.extend(results)

    if all_candidates:
        print(f"  [Matching] Total candidates found: {len(all_candidates)}")
        # Deduplication: (target_file, target_func) unique, but prefer based on Verdict priority and Confidence
        unique_candidates = {}
        
        # Priority mapping: VULNERABLE > UNKNOWN > PATCHED > MISMATCH
        VERDICT_PRIORITY = {
            "VULNERABLE": 3,
            "UNKNOWN": 2,
            "PATCHED": 1,
            "MISMATCH": 0
        }

        for cand in all_candidates:
            # [Optimize] Deduplicate strictly by target location to avoid redundant verification
            key = (cand.target_file, cand.target_func)
            
            if key not in unique_candidates:
                unique_candidates[key] = cand
            else:
                existing = unique_candidates[key]
                p_new = VERDICT_PRIORITY.get(cand.verdict, 0)
                p_old = VERDICT_PRIORITY.get(existing.verdict, 0)
                
                # Strategy 1: Higher priority wins (e.g. VULNERABLE overwrites PATCHED)
                if p_new > p_old:
                    unique_candidates[key] = cand
                # Strategy 2: Same priority, select higher confidence
                elif p_new == p_old:
                    if cand.confidence > existing.confidence:
                        unique_candidates[key] = cand
        
        final_list = list(unique_candidates.values())
        
        # [Density Control] Moved to after global deduplication
        # Group by patch_func
        from collections import defaultdict
        results_by_func = defaultdict(list)
        for r in final_list:
            results_by_func[r.patch_func].append(r)
            
        # [New] Assign Rank for VULNERABLE results
        # This helps the verifier to process candidates in order of confidence
        for func_name, func_results in results_by_func.items():
            vuln_results = [r for r in func_results if r.verdict == "VULNERABLE"]
            # Sort by confidence descending to determine rank
            vuln_results.sort(key=lambda x: x.confidence, reverse=True)
            
            for i, r in enumerate(vuln_results):
                r.rank = i  # 0-based rank

        # DENSITY_THRESHOLD = 4
        # RANK_PENALTY = 0.3
        
        # for func_name, func_results in results_by_func.items():
        #     vuln_results = [r for r in func_results if r.verdict == "VULNERABLE"]
        #     vuln_count = len(vuln_results)
            
        #     if vuln_count > DENSITY_THRESHOLD:
        #         print(f"    [Density Control] Function {func_name} produced {vuln_count} VULNERABLE results (Threshold: {DENSITY_THRESHOLD}). Applying rank-based decay.")
                
        #         # Sort results by confidence first, determine "rank"
        #         vuln_results.sort(key=lambda x: x.confidence, reverse=True)
                
        #         for rank, r in enumerate(vuln_results):
        #             if rank >= DENSITY_THRESHOLD:
        #                 # Apply penalty
        #                 penalty = RANK_PENALTY * (rank - DENSITY_THRESHOLD + 1)
        #                 old_conf = r.confidence
        #                 r.confidence = max(0.0, r.confidence - penalty)

        # Re-sort after adjusting confidence
        final_list.sort(key=get_sort_key, reverse=True)
        
        for cand in final_list:
            if cand.verdict == "VULNERABLE" and cand.confidence >= 0.4:
                print(f"    - {cand.patch_func} {cand.target_file}::{cand.target_func} [{cand.verdict} | Rank: {cand.rank} | {cand.confidence:.2f}]")
        
        final_list = [r for r in final_list if r.verdict == "VULNERABLE" and r.confidence >= 0.4]
        return {"search_candidates": final_list}
        
    else:
        print("  [Matching] No candidates found.")
        return {"search_candidates": []}