import difflib
import re
from typing import List, Tuple, Dict
from tqdm import tqdm
from core.models import PatchFeatures, MatchEvidence, MatchTrace, SearchResultItem, AlignedTrace
from core.state import WorkflowState
import concurrent.futures
from core.indexer import GlobalSymbolIndexer, BenchmarkSymbolIndexer, tokenize_code

# ==========================================
# 0. 参数配置类
# ==========================================

class MatcherConfig:
    """
    漏洞匹配器核心参数配置类
    优化后：从22个参数简化到9个核心参数
    """
    
    # ========== 相似度阈值 (2个) ==========
    # [删除] ANCHOR_SIMILARITY_THRESHOLD - 锚点检查改为基于相似度排名，不做硬阈值
    # [删除] MATCH_SCORE_THRESHOLD - DP 对齐中直接使用相似度，不做预过滤
    VERDICT_THRESHOLD_STRONG = 0.9         # 强信号线：高于此视为特征完整
    VERDICT_THRESHOLD_WEAK = 0.4           # 弱信号线：低于此视为特征缺失
    
    # ========== DP对齐参数 (1个) ==========
    BASE_GAP_PENALTY = 0.1                 # 基础间隙惩罚
    
    # ========== 二维加权得分权重 (2个) ==========
    # final_score = α * slice_score + β * anchor_score
    WEIGHT_SLICE = 0.5                     # α: 切片整体相似度权重
    WEIGHT_ANCHOR = 0.5                    # β: 锚点匹配权重
    
    # ========== 搜索限制参数 (2个) ==========
    SEARCH_LIMIT_FAST = 2000                # 快速搜索候选函数限制

# ==========================================
# 1. 基础工具函数
# ==========================================

def remove_comments_from_code(code: str) -> str:
    """
    预处理：移除代码中的所有注释（包括跨行注释），保留换行符以维持行号对应关系。
    """
    # 1. 移除块注释 /* ... */
    # 使用非贪婪匹配，re.DOTALL 让 . 匹配换行符
    def replace_block_comment(match):
        # 将注释内容替换为相同数量的换行符，以保持行号一致
        return '\n' * match.group(0).count('\n')
    
    code = re.sub(r'/\*.*?\*/', replace_block_comment, code, flags=re.DOTALL)
    
    # 2. 移除单行注释 // ...
    code = re.sub(r'//.*', '', code)
    
    return code

def extract_line_number(line: str) -> int:
    """从代码行中提取行号，假设格式为 '[ 123] ...'"""
    match = re.match(r'^\[\s*(\d+)\]', line)
    if match:
        return int(match.group(1))
    return -1

def normalize_program_structure(code_str: str, has_line_markers: bool = False) -> Tuple[List[str], List[List[int]]]:
    """
    对代码结构进行归一化：
    1. 移除注释
    2. 基于括号深度合并多行语句 (解决函数调用分行的问题)
    返回: (归一化后的代码行列表, 每一行对应的原始行号列表)
    """
    clean_code = remove_comments_from_code(code_str)
    raw_lines = clean_code.splitlines()
    
    normalized_lines = []
    line_mapping = []
    
    current_buffer = []
    current_indices = []
    paren_depth = 0
    
    for idx, line in enumerate(raw_lines):
        # 1. 预处理当前行
        original_line = line
        
        # 移除行首的 [ 123] 标记 (如果有)
        if has_line_markers:
            line = re.sub(r'^\[\s*\d+\]', '', line)
            
        stripped = line.strip()
        if not stripped:
            continue
            
        current_buffer.append(stripped)
        # 如果提供了行标记，这里可以尝试提取真实的行号，否则使用 0-based index + 1
        if has_line_markers:
            ln = extract_line_number(original_line)
            current_indices.append(ln if ln != -1 else (idx + 1))
        else:
            current_indices.append(idx + 1)
        
        # 2. 括号深度扫描 (简单的字符串清理后计数)
        # 替换掉字符串内容防止干扰 "a)"
        check_line = re.sub(r'".*?"', '""', line)
        check_line = re.sub(r"'.*?'", "''", check_line)
        
        for char in check_line:
            if char == '(': paren_depth += 1
            elif char == ')': paren_depth -= 1
            
        # 3. 判定是否合并
        # 允许一定的容错，防止 paren_depth 长期小于 0
        if paren_depth < 0: paren_depth = 0
        
        if paren_depth == 0:
            # 完整语句，Flush
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

# [新增] 忽略的控制流关键字，不视为函数调用
CONTROL_KEYWORDS = {'if', 'for', 'while', 'switch', 'return', 'sizeof', 'likely', 'unlikely'}

def extract_function_name(line: str) -> str:
    """
    从代码行中提取主要函数名。
    如果找不到明显的函数调用，返回 None。
    """
    # 简单启发式：查找 'name(' 模式
    # 排除掉控制关键字
    matches = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', line)
    candidates = [m for m in matches if m not in CONTROL_KEYWORDS]
    
    if not candidates:
        return None
    
    # 如果有多个，通常第一个是非关键字的函数调用
    # e.g. "err = check_ctx_reg(env, ...)" -> check_ctx_reg
    return candidates[0]

def normalize_code_line(line: str) -> str:
    """
    [Deprecated] 保留此函数以兼容旧代码，但内部逻辑已简化。
    主要用于快速判断行是否为空。
    """
    # 移除行号 [ 123]
    line = re.sub(r'^\[\s*\d+\]', '', line)
    # 移除单行注释
    line = re.sub(r'//.*', '', line)
    # 移除块注释标记
    line = re.sub(r'/\*.*?\*/', '', line)
    # 移除空白
    return re.sub(r'\s+', '', line).strip()

def tokenize_line(line: str) -> List[str]:
    """
    将代码行转换为细粒度的 Token 列表。
    支持 CamelCase, snake_case, 数字切分。
    [修改] 所有 token 统一转为小写，使 len 和 Len 被视为相同。
    """
    # 1. 预处理：移除行号、注释
    line = re.sub(r'^\[\s*\d+\]', '', line) # Remove line numbers like [ 123]
    line = re.sub(r'//.*', '', line)        # Remove line comments
    line = re.sub(r'/\*.*?\*/', '', line)   # Remove inline block comments
    
    # 2. 初步分词：标识符/数字 vs 符号
    # \w+ 匹配字母数字下划线
    # [^\w\s] 匹配非单词非空白字符 (符号)
    # [Modified] Prioritize -> operator as a single token to avoid splitting it
    raw_tokens = re.findall(r'->|\w+|[^\w\s]', line)
    
    final_tokens = []
    for token in raw_tokens:
        # 如果是标识符或数字 (包含字母、数字、下划线)
        if re.match(r'^\w+$', token):
            # Sub-tokenization logic
            # A. 按下划线切分
            parts = token.split('_')
            for part in parts:
                if not part: continue
                
                # B. 按 CamelCase 和 数字切分
                # 正则逻辑：
                # [A-Z]?[a-z]+ : 首字母可选大写的单词 (e.g. "Word", "word")
                # [A-Z]+(?=[A-Z][a-z]|\d|\W|$) : 连续大写字母 (e.g. "XML" in "XMLParser")
                # [A-Z]+ : 剩余的大写字母
                # \d+ : 数字
                sub_parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|[A-Z]+|\d+', part)
                
                if sub_parts:
                    # [修改] 统一转小写
                    final_tokens.extend([p.lower() for p in sub_parts])
                else:
                    # Fallback (should rarely happen for \w+)
                    final_tokens.append(part.lower())
        else:
            # 符号 (Operators, Punctuation) - 不需要转小写
            final_tokens.append(token)
    
    # [优化] 过滤掉对语义贡献极小的标点符号，防止它们干扰匹配分数
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
# 2. 行对应关系类（用于竞争评分）
# ==========================================

class LineMappingType:
    """行映射类型枚举"""
    COMMON = "COMMON"       # 两边完全相同的行
    MODIFIED = "MODIFIED"   # 被修改的行对（pre 的某行变成了 post 的某行）
    VULN_ONLY = "VULN_ONLY" # 仅在 pre-patch 中存在（被删除）
    FIX_ONLY = "FIX_ONLY"   # 仅在 post-patch 中存在（新增）

class LineCorrespondence:
    """
    Pre/Post 行对应关系
    
    用于竞争评分：当目标行同时匹配 MODIFIED 行对的两边时，
    只计入相似度更高的那边，避免重复计分。
    """
    def __init__(self, pre_idx: int = None, post_idx: int = None,
                 mapping_type: str = None, pre_content: str = None, post_content: str = None):
        self.pre_idx = pre_idx      # pre-patch 行索引（0-based）
        self.post_idx = post_idx    # post-patch 行索引（0-based）
        self.mapping_type = mapping_type
        self.pre_content = pre_content
        self.post_content = post_content
    
    def __repr__(self):
        return f"LineCorrespondence({self.mapping_type}: pre[{self.pre_idx}] <-> post[{self.post_idx}])"


class DualChannelMatcher:
    def __init__(self, s_pre: str, s_post: str,
                 pre_origins: List[str] = None, pre_impacts: List[str] = None,
                 post_origins: List[str] = None, post_impacts: List[str] = None):
        
        # [修改] 使用 normalize_program_structure 进行归一化
        # 切片通常带有行号标记 [ 123]，因此开启 has_line_markers=True
        self.s_pre_lines, self.s_pre_map = normalize_program_structure(s_pre, has_line_markers=True)
        self.s_post_lines, self.s_post_map = normalize_program_structure(s_post, has_line_markers=True)

        # 构建行对应关系和标签
        self.pre_tags, self.post_tags, self.line_correspondences = self._tag_regions_with_correspondence()
        self.has_vuln_features = 'VULN' in self.pre_tags
        self.has_fix_features = 'FIX' in self.post_tags
        
        # 构建 MODIFIED 行对的快速查找表（用于竞争评分）
        self.modified_pre_to_post = {}  # pre_idx -> post_idx
        self.modified_post_to_pre = {}  # post_idx -> pre_idx
        for corr in self.line_correspondences:
            if corr.mapping_type == LineMappingType.MODIFIED:
                self.modified_pre_to_post[corr.pre_idx] = corr.post_idx
                self.modified_post_to_pre[corr.post_idx] = corr.pre_idx
        
        # [New] Precise Anchors
        # Clean them similarly to how we filter lines (remove empty ones)
        # Anchors 也应该是一一对应的，这里简单处理，后续匹配时使用归一化后的行列表
        self.pre_origins = [l.strip() for l in (pre_origins or []) if tokenize_line(l)]
        self.pre_impacts = [l.strip() for l in (pre_impacts or []) if tokenize_line(l)]
        self.post_origins = [l.strip() for l in (post_origins or []) if tokenize_line(l)]
        self.post_impacts = [l.strip() for l in (post_impacts or []) if tokenize_line(l)]
        
    def _tag_regions_with_correspondence(self) -> Tuple[List[str], List[str], List[LineCorrespondence]]:
        """
        构建行标签和行对应关系
        
        Returns:
            pre_tags: pre-patch 每行的标签 (COMMON/VULN)
            post_tags: post-patch 每行的标签 (COMMON/FIX)
            correspondences: 行对应关系列表
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
                # COMMON: 完全相同的行
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
                # MODIFIED: 被替换的行对
                # 尝试一一对应（当块大小相同时）
                pre_block_size = i2 - i1
                post_block_size = j2 - j1
                
                if pre_block_size == post_block_size:
                    # 一一对应的 MODIFIED 行对
                    for k in range(pre_block_size):
                        pre_idx = i1 + k
                        post_idx = j1 + k
                        # 标签保持 VULN/FIX（不改为 COMMON）
                        correspondences.append(LineCorrespondence(
                            pre_idx=pre_idx,
                            post_idx=post_idx,
                            mapping_type=LineMappingType.MODIFIED,
                            pre_content=self.s_pre_lines[pre_idx] if pre_idx < len(self.s_pre_lines) else None,
                            post_content=self.s_post_lines[post_idx] if post_idx < len(self.s_post_lines) else None
                        ))
                else:
                    # 块大小不同，尝试内容相似度匹配
                    matched_post = set()
                    for pi in range(i1, i2):
                        best_match = None
                        best_sim = 0.5  # 最低相似度阈值
                        for pj in range(j1, j2):
                            if pj in matched_post:
                                continue
                            # 计算 token 相似度
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
                            # 没有匹配，标记为 VULN_ONLY
                            correspondences.append(LineCorrespondence(
                                pre_idx=pi,
                                post_idx=None,
                                mapping_type=LineMappingType.VULN_ONLY,
                                pre_content=self.s_pre_lines[pi],
                                post_content=None
                            ))
                    
                    # 未匹配的 post 行标记为 FIX_ONLY
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
                # VULN_ONLY: 仅在 pre-patch 中存在
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
                # FIX_ONLY: 仅在 post-patch 中存在
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
                       slice_map: List[List[int]], # [新增] Slice 行号映射
                       target_lines: List[str], target_map: List[List[int]] # [新增] Target 行号映射
                       ) -> Tuple[float, float, List[MatchTrace], List[str], List[AlignedTrace]]:
        # ... (previous code omitted for brevity in call, but need to be careful with replace)
        # Actually I cannot omit code in replace tool. I should target the specific block.
        
        # I'll update the signature and return line first? No, I need to do it in chunks or one big chunk.
        # Let's target the logic after DP matrix construction.
        pass
        n, m = len(slice_lines), len(target_lines)
        
        # [新增] 安全检查：防止过大的矩阵计算
        # 如果矩阵元素超过 200万 (e.g. 200行切片 x 10000行目标)，直接放弃
        # if n * m > 2_000_000:
        #     print(f"  [Matcher] Warning: Matrix too large ({n}x{m} = {n*m}), skipping alignment.")
        #     return 0.0, 0.0, [], []

        # [优化] 预处理 Tokenized lines
        slice_tokens = [tokenize_line(line) for line in slice_lines]
        target_tokens = [tokenize_line(line) for line in target_lines]

        # [新增] 提取切片行号 (使用归一化映射中的第一个行号)
        slice_line_nos = [m[0] if m else -1 for m in slice_map]

        # [优化] 预计算相似度矩阵
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
                
                # quick_ratio 预筛选（硬编码0.5）
                if sm.quick_ratio() < 0.5:
                    sim_matrix[i][j] = 0.0
                    continue
                
                # [简化] 直接使用 ratio()，函数调用相似度由 _compute_func_score 单独计算
                sim_matrix[i][j] = sm.ratio()

        # 1. DP 矩阵构建
        # dp[i][j] 存储到达 (i, j) 时的最大累积得分
        dp = [[0.0] * (m + 1) for _ in range(n + 1)]
        
        # [Phase 1: 使用MatcherConfig.BASE_GAP_PENALTY]
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
                # [修改] 直接使用相似度作为匹配得分，不做预过滤阈值
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
            # [修改] 直接使用相似度作为匹配得分
            match_gain = sim
            # [Fix] Match the forward pass gap cost logic
            gap_cost = step_gap_costs[i+1]
            
            # Check Match
            # Use strict epsilon for float equality check
            # [修改] 只要 sim > 0 就认为有匹配可能
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
        
        # [注意] valid_slice_lines_count == 0 是合理情况
        # 例如纯删除补丁的 s_post 只有注释头，归一化后为空
        if valid_slice_lines_count == 0:
            # 空切片：返回零分，不打印警告
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
        对混合型补丁应用竞争评分机制
        
        核心规则：对于 MODIFIED 行对，如果目标行同时匹配 pre 和 post 的对应行，
        只计入相似度更高的那边，避免重复计分。
        
        Args:
            aligned_vuln: pre-patch 对齐结果
            aligned_fix: post-patch 对齐结果
            target_lines: 目标函数的归一化行列表
            
        Returns:
            (score_vuln, score_fix, score_feat_vuln, score_feat_fix)
            - score_vuln/score_fix: 调整后的整体相似度
            - score_feat_vuln/score_feat_fix: 局部特征相似度
        """
        # 如果没有 MODIFIED 行对，直接从对齐结果计算分数
        if not self.modified_pre_to_post:
            return self._compute_scores_from_alignment(aligned_vuln, aligned_fix)
        
        # 构建目标行索引 -> 对齐信息的映射
        # aligned_vuln/aligned_fix 中的 line_no 是目标行的 1-based 索引
        vuln_target_matches = {}  # target_line_idx -> (slice_idx, similarity)
        fix_target_matches = {}   # target_line_idx -> (slice_idx, similarity)
        
        for slice_idx, trace in enumerate(aligned_vuln):
            if trace.line_no is not None and trace.similarity > 0:
                target_idx = trace.line_no - 1  # 转为 0-based
                # 如果同一目标行被多次匹配，保留相似度最高的
                if target_idx not in vuln_target_matches or trace.similarity > vuln_target_matches[target_idx][1]:
                    vuln_target_matches[target_idx] = (slice_idx, trace.similarity)
        
        for slice_idx, trace in enumerate(aligned_fix):
            if trace.line_no is not None and trace.similarity > 0:
                target_idx = trace.line_no - 1
                if target_idx not in fix_target_matches or trace.similarity > fix_target_matches[target_idx][1]:
                    fix_target_matches[target_idx] = (slice_idx, trace.similarity)
        
        # 竞争评分：处理 MODIFIED 行对的重复计分
        # 记录需要从某一边扣除的 token 数
        vuln_deduction = 0.0
        fix_deduction = 0.0
        
        for pre_idx, post_idx in self.modified_pre_to_post.items():
            # 检查这对 MODIFIED 行是否都被匹配到了同一个目标行
            # 1. 找到 pre_idx 对应的目标行
            vuln_matched_target = None
            vuln_sim = 0.0
            if pre_idx < len(aligned_vuln) and aligned_vuln[pre_idx].line_no is not None:
                vuln_matched_target = aligned_vuln[pre_idx].line_no - 1
                vuln_sim = aligned_vuln[pre_idx].similarity
            
            # 2. 找到 post_idx 对应的目标行
            fix_matched_target = None
            fix_sim = 0.0
            if post_idx < len(aligned_fix) and aligned_fix[post_idx].line_no is not None:
                fix_matched_target = aligned_fix[post_idx].line_no - 1
                fix_sim = aligned_fix[post_idx].similarity
            
            # 3. 如果都匹配到了同一个目标行，执行竞争
            if vuln_matched_target is not None and fix_matched_target is not None:
                if vuln_matched_target == fix_matched_target:
                    # 竞争：只保留相似度更高的
                    pre_tokens = len(tokenize_line(self.s_pre_lines[pre_idx])) if pre_idx < len(self.s_pre_lines) else 0
                    post_tokens = len(tokenize_line(self.s_post_lines[post_idx])) if post_idx < len(self.s_post_lines) else 0
                    
                    if vuln_sim >= fix_sim:
                        # vuln 胜出，从 fix 扣除
                        fix_deduction += post_tokens * fix_sim
                    else:
                        # fix 胜出，从 vuln 扣除
                        vuln_deduction += pre_tokens * vuln_sim
        
        # 计算基础分数
        score_vuln, score_fix, score_feat_vuln, score_feat_fix = self._compute_scores_from_alignment(aligned_vuln, aligned_fix)
        
        # 应用扣除（转换为分数调整）
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
        计算锚点匹配得分
        
        思路：
        - 检查 source（数据源）和 sink（影响点）是否匹配到目标
        - 返回 0-1 之间的分数
        
        注意：raw_lines 是经过 normalize_program_structure 处理的，多行语句会被合并成一行。
        而 sources/sinks 是原始的多行 anchor。
        
        匹配策略：基于行号匹配
        - 从 anchor 中提取行号（如 "[ 227] ..." -> 227）
        - 在 slice_lines 中找到包含该行号的行（归一化可能合并 227-228 到一行）
        
        Args:
            aligned_trace: 对齐追踪结果
            raw_lines: 归一化后的切片行列表
            slice_map: 每行对应的原始行号列表 (由 normalize_program_structure 返回)
            sources: origin anchor 行列表
            sinks: impact anchor 行列表
        
        Returns:
            anchor_score: 锚点匹配得分 [0, 1]
                - 1.0: 双锚点都匹配
                - 0.5: 只有一个锚点匹配
                - 0.0: 两个锚点都不匹配
                - 1.0: 没有显式锚点（不惩罚）
        """
        if not aligned_trace:
            return 0.0
        
        n_trace = len(aligned_trace)
        
        def extract_line_no(s):
            """从带行号标记的行中提取行号"""
            match = re.match(r'^\[\s*(\d+)\]', s.strip())
            return int(match.group(1)) if match else -1
        
        def get_indices_by_line_number(key_lines):
            """
            基于行号找到 slice_lines 中对应的索引。
            
            Args:
                key_lines: anchor 行列表（带行号标记）
            
            Returns:
                匹配到的 slice_lines 索引列表
            """
            indices = set()
            if not key_lines:
                return list(indices)
            
            # 提取所有 anchor 的行号
            anchor_line_nos = set()
            for anchor in key_lines:
                ln = extract_line_no(anchor)
                if ln != -1:
                    anchor_line_nos.add(ln)
            
            if not anchor_line_nos:
                return list(indices)
            
            # 在 slice_map 中查找包含这些行号的索引
            for i, orig_line_nos in enumerate(slice_map):
                # orig_line_nos 是一个列表，表示第 i 行归一化行对应的原始行号
                # 例如：[227, 228] 表示第 i 行是由原始的 227, 228 行合并而成
                for ln in orig_line_nos:
                    if ln in anchor_line_nos:
                        indices.add(i)
                        break
            
            return list(indices)
        
        has_sources = bool(sources)
        has_sinks = bool(sinks)
        
        # 如果没有显式锚点，返回满分（不惩罚）
        if not has_sources and not has_sinks:
            return 1.0
        
        source_score = 0.0
        sink_score = 0.0
        anchor_count = 0
        
        # Source 检查 - 使用行号匹配
        if has_sources:
            anchor_count += 1
            explicit_source_indices = get_indices_by_line_number(sources)
            if explicit_source_indices:
                sims = max((aligned_trace[i].similarity for i in explicit_source_indices if i < n_trace), default=0.0)
                source_score = sims  # 直接使用相似度作为分数
        
        # Sink 检查 - 使用行号匹配
        if has_sinks:
            anchor_count += 1
            explicit_sink_indices = get_indices_by_line_number(sinks)
            if explicit_sink_indices:
                sims = max((aligned_trace[i].similarity for i in explicit_sink_indices if i < n_trace), default=0.0)
                sink_score = sims  # 直接使用相似度作为分数
        
        # 计算平均分
        if anchor_count == 0:
            return 1.0
        
        return (source_score + sink_score) / anchor_count
    
    def _compute_scores_from_alignment(self, aligned_vuln: List[AlignedTrace], aligned_fix: List[AlignedTrace]
                                       ) -> Tuple[float, float, float, float]:
        """
        从对齐结果计算4个分数
        
        Returns:
            (score_vuln, score_fix, score_feat_vuln, score_feat_fix)
        """
        # Pre-patch 分数计算
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
        
        # Post-patch 分数计算
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
        
        # 计算分数
        score_vuln = vuln_matched_tokens / vuln_total_tokens if vuln_total_tokens > 0 else 0.0
        score_fix = fix_matched_tokens / fix_total_tokens if fix_total_tokens > 0 else 0.0
        score_feat_vuln = vuln_feat_matched / vuln_feat_total if vuln_feat_total > 0 else 0.0
        score_feat_fix = fix_feat_matched / fix_feat_total if fix_feat_total > 0 else 0.0
        
        return score_vuln, score_fix, score_feat_vuln, score_feat_fix

    def match(self, target_function_code: str, target_start_line: int = 1) -> MatchEvidence:
        # [预处理] 使用归一化
        target_lines, target_map = normalize_program_structure(target_function_code, has_line_markers=False)

        # 匹配
        # 传入 map 用于正确计算行号间距和恢复原始行号
        ctx_a, score_feat_vuln, total_vuln, _, _, aligned_vuln = self._align_channel(
            self.s_pre_lines, self.pre_tags, self.s_pre_map,
            target_lines, target_map
        )
        ctx_b, score_feat_fix, total_fix, _, _, aligned_fix = self._align_channel(
            self.s_post_lines, self.post_tags, self.s_post_map,
            target_lines, target_map
        )
        
        # [新增] 对混合型补丁应用竞争评分
        if self.has_vuln_features and self.has_fix_features and self.modified_pre_to_post:
            # 有 MODIFIED 行对，使用竞争评分
            score_vuln, score_fix, score_feat_vuln, score_feat_fix = self._apply_competitive_scoring(
                aligned_vuln, aligned_fix, target_lines
            )
        else:
            # 无 MODIFIED 行对，使用原始分数
            score_vuln = total_vuln
            score_fix = total_fix
        
        if not self.has_vuln_features:
            score_feat_vuln = -1.0

        if not self.has_fix_features:
            score_feat_fix = -1.0
        
        # 修正 Trace 中的 line_no 为真实行号
        # [修改] 使用 target_map 将归一化行索引 映射回 原始文件行号
        # [新增] 使用 target_start_line 计算文件中的绝对行号
        def correct_line_numbers(trace_list):
             for m in trace_list:
                # m.line_no 是 1-based index in target_lines
                if m.line_no is not None and 0 < m.line_no <= len(target_map):
                    # 获取该归一化行对应的原始行号列表（相对于函数内部）
                    original_indices = target_map[m.line_no - 1]
                    if original_indices:
                        # original_indices[0] 是函数内部的相对行号（从1开始）
                        # 加上 (target_start_line - 1) 得到文件中的绝对行号
                        real_line_start = original_indices[0] + target_start_line - 1
                        m.line_no = real_line_start
                        # 格式化 target_line: "[ 123] content"
                        # 使用 m.target_line (normalized content)
                        if m.target_line:
                            m.target_line = f"[{real_line_start:4d}] {m.target_line}"

        # correct_line_numbers(vuln_matches)
        # correct_line_numbers(fix_matches)
        correct_line_numbers(aligned_vuln)
        correct_line_numbers(aligned_fix)

        score_ctx = max(ctx_a, ctx_b)
        # 整体分数：如果没有特征行，使用上下文分数
        total_vuln = score_vuln if self.has_vuln_features else score_ctx
        total_fix = score_fix if self.has_fix_features else score_ctx
        verdict = "UNKNOWN"
        confidence = 0.0
        
        # ========== 二维加权得分计算 ==========
        # final_score = α * slice_score + β * anchor_score
        
        # 1. 计算 Vuln 通道的二维得分
        anchor_score_vuln = self._compute_anchor_score(
            aligned_vuln, self.s_pre_lines, self.s_pre_map, self.pre_origins, self.pre_impacts
        )
        
        total_vuln = (
            MatcherConfig.WEIGHT_SLICE * score_vuln +
            MatcherConfig.WEIGHT_ANCHOR * anchor_score_vuln
        )
        
        # 2. 计算 Fix 通道的二维得分
        anchor_score_fix = self._compute_anchor_score(
            aligned_fix, self.s_post_lines, self.s_post_map, self.post_origins, self.post_impacts
        )
        
        # 加权计算 final_fix
        total_fix = (
            MatcherConfig.WEIGHT_SLICE * score_fix +
            MatcherConfig.WEIGHT_ANCHOR * anchor_score_fix
        )
        
        # --- Verdict Phase ---
        # 分类型判定逻辑（与 Methodology.tex 保持一致）
        
        # 1. 预筛选：垃圾过滤 (基于整体漏洞分数)
        # 如果与漏洞模式相似度太低，直接 Mismatch
        if total_vuln < MatcherConfig.VERDICT_THRESHOLD_WEAK:
            verdict = "MISMATCH"
            confidence = 1.0 - total_vuln
        else:
            # --- Case 1: 纯新增补丁 (Pure Additive) ---
            # 判断标准：FIX 特征行是否存在
            if self.has_fix_features and not self.has_vuln_features:
                if score_feat_fix >= MatcherConfig.VERDICT_THRESHOLD_STRONG:
                    verdict = "PATCHED"
                    confidence = total_fix
                else:
                    verdict = "VULNERABLE"
                    confidence = total_vuln
            
            # --- Case 2: 纯删除补丁 (Pure Subtractive) ---
            # 判断标准：VULN 特征行是否已删除
            elif self.has_vuln_features and not self.has_fix_features:
                if score_feat_vuln < MatcherConfig.VERDICT_THRESHOLD_WEAK:
                    verdict = "PATCHED"
                    confidence = total_fix
                else:
                    verdict = "VULNERABLE"
                    confidence = total_vuln
            
            # --- Case 3: 混合型补丁 (Modified) ---
            # 判断标准：竞争评分后，比较整体分数
            elif self.has_vuln_features and self.has_fix_features:
                if total_fix > total_vuln:
                    verdict = "PATCHED"
                    confidence = total_fix
                else:
                    verdict = "VULNERABLE"
                    confidence = total_vuln

            # --- Case 4: 无特征 ---
            else:
                verdict = "UNKNOWN"
                confidence = score_ctx

        return MatchEvidence(
            verdict=verdict,
            confidence=confidence,
            score_vuln=total_vuln,           # 整体切片相似度（pre-patch vs target）
            score_fix=total_fix,             # 整体切片相似度（post-patch vs target）
            score_feat_vuln=score_feat_vuln, # 局部特征相似度（仅 VULN 行）
            score_feat_fix=score_feat_fix,   # 局部特征相似度（仅 FIX 行）
            aligned_vuln_traces=aligned_vuln,
            aligned_fix_traces=aligned_fix
        )

class VulnerabilitySearchEngine:
    def __init__(self, repo_path: str):
        # [修改] 不再接收 db_path，而是接收 repo_path，动态获取 Indexer
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
            
            # 从所有切片中提取 tokens（使用与 database 一致的 tokenize_code）
            # [修复] 先移除注释（包括头部标记注释），再提取 tokens
            search_tokens = []
            for func_name, sf in feature.slices.items():
                clean_code = remove_comments_from_code(sf.s_pre)
                tokens = tokenize_code(clean_code)
                search_tokens.extend(tokens)
            
            # 去重
            search_tokens = list(set(search_tokens))
            print(f"    Extracted {len(search_tokens)} unique tokens for benchmark search")

            # 使用统一的 Token 搜索方法
            candidates = self.benchmark_indexer.search_functions_by_tokens(vul_id, search_tokens, limit=MatcherConfig.SEARCH_LIMIT_FAST)
            
            matchers = {
                name: DualChannelMatcher(sf.s_pre, sf.s_post, 
                                         pre_origins=sf.pre_origins, pre_impacts=sf.pre_impacts,
                                         post_origins=sf.post_origins, post_impacts=sf.post_impacts) 
                for name, sf in feature.slices.items()
            }
            
            # 2. 过滤并匹配
            # 目标函数名集合
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
                
                # 只保留函数名相同的函数
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
                
                # 1. 从切片中提取 tokens（使用与 database 一致的 tokenize_code）
                # [修复] 先移除注释（包括头部标记注释），再提取 tokens
                clean_code = remove_comments_from_code(sf.s_pre)
                search_tokens = tokenize_code(clean_code)
                
                print(f"    Extracted {len(search_tokens)} tokens for {func_name}")
                print(f"    Sample tokens: {search_tokens[:10]}...")

                # 2. 使用统一的 Token 搜索方法（替代双重搜索）
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
# 4. LangGraph Node 函数
# ==========================================

def matching_node(state: WorkflowState) -> Dict:
    """
    Phase 3 核心节点：执行漏洞搜索。
    """
    features = state.get("analyzed_features", [])
    # [关键] 必须从 state 获取仓库路径，用于初始化 Indexer
    repo_path = state.get("repo_path") 
    
    mode = state.get("mode")
    vul_id = state.get("vul_id")
    
    if not features or not repo_path:
        print("  [Matching] Missing features or repo path.")
        return {"search_candidates": []}

    print(f"[*] Starting Matching Phase for {len(features)} patch groups...")
    
    all_candidates = []
    
    # 定义单个任务
    def process_single_feature(feature):
        try:
            # [修改] 传入 repo_path 和 mode
            engine = VulnerabilitySearchEngine(repo_path)
            return engine.search_patch(feature, mode, vul_id)
        except Exception as e:
            print(f"  [Error] Search failed for {feature.group_id}: {e}")
            import traceback
            traceback.print_exc()
            return []

    # 串行执行 (避免多线程下的 tqdm/DB 竞争问题)
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
        # 去重：(target_file, target_func) 唯一，但基于 Verdict 优先级和 Confidence 择优
        unique_candidates = {}
        
        # 优先级映射：VULNERABLE > UNKNOWN > PATCHED > MISMATCH
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
                
                # 策略 1: 优先级高的胜出 (例如 VULNERABLE 覆盖 PATCHED)
                if p_new > p_old:
                    unique_candidates[key] = cand
                # 策略 2: 优先级相同，选置信度高的
                elif p_new == p_old:
                    if cand.confidence > existing.confidence:
                        unique_candidates[key] = cand
        
        final_list = list(unique_candidates.values())
        
        # [Density Control] 移到全局去重之后进行
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
                
        #         # 先对结果按置信度排序，确定"排位"
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