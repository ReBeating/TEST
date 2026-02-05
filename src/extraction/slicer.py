"""
Program Slicing Module

Implements PDG (Program Dependence Graph) based program slicing for vulnerability code extraction.

Core Design:
- Primary View: Fixed as Pre-Patch (OLD/Vulnerable) - contains complete vulnerability trigger path
- Shadow View: Fixed as Post-Patch (NEW/Fixed) - generated via mapping for fix comparison

Main Components:
1. AnchorAnalyzer: Agent-driven anchor identifier (based on Methodology §3.3.2)
2. Slicer: Static program slicing engine (forward, backward, bidirectional slicing)
3. ShadowMapper: Shadow slice generator (sync Primary slice to Post-Patch version)
4. SliceValidator: Agent-driven slice distiller (remove noise code)
5. slicing_node: Main workflow entry function
"""

import os
import difflib
import re
import time
import networkx as nx
from typing import List, Set, Dict, Union, Tuple, Optional, Any
from core.state import PatchExtractionState
from core.models import (SliceFeature, SemanticFeature, SliceEntryPoint, FixType, AtomicPatch,
            SliceValidationResult, CodeLineReference, TaxonomyFeature, SlicingInstruction, VulnType, PatchFeatures)
from core.navigator import CodeNavigator
from core.models import GeneralVulnType
from .pdg import PDGBuilder
from .anchor_analyzer import AnchorAnalyzer, AnchorResult, AnchorItem
from .distillation_strategies import get_strategy_prompt_section
from .slice_validation import (
    validate_anchor_completeness,
    validate_slice_quality
)
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, StructuredTool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
import json

# ==============================================================================
# 重连重试工具函数
# ==============================================================================

def llm_invoke_with_retry(llm, messages, max_retries: int = 3, retry_delay: float = 5.0):
    """
    增强的 LLM 调用包装函数，带自动重试机制。
    处理网络错误 (500, 502, 503, 504) 和连接超时。
    
    Args:
        llm: ChatOpenAI 实例或带 structured_output 的 LLM
        messages: 消息列表
        max_retries: 最大重试次数
        retry_delay: 重试间隔（秒），每次重试后会指数增长
    
    Returns:
        LLM 响应，失败时返回 None（而不是抛异常）
        调用方需要检查返回值是否为 None
    """
    last_exception = None
    current_delay = retry_delay
    
    for attempt in range(max_retries):
        try:
            return llm.invoke(messages)
        except Exception as e:
            last_exception = e
            error_str = str(e).lower()
            
            # 分类错误类型
            error_type = "unknown"
            is_retryable = False
            
            # 网络/服务器错误（可重试）
            if any(x in error_str for x in ['500', '502', '503', '504', 'internal server error']):
                error_type = "server_error"
                is_retryable = True
            # 连接错误（可重试）
            elif any(x in error_str for x in ['connection', 'timeout', 'timed out', 'network', 'reset', 'refused', 'broken pipe', 'unreachable']):
                error_type = "connection_error"
                is_retryable = True
            # 速率限制（可重试，但延迟更长）
            elif any(x in error_str for x in ['rate limit', 'too many requests', '429']):
                error_type = "rate_limit"
                is_retryable = True
                current_delay = max(current_delay, 30.0)  # 速率限制至少等30秒
            # API密钥/认证错误（不可重试）
            elif any(x in error_str for x in ['api key', 'authentication', 'unauthorized', '401', '403']):
                error_type = "auth_error"
                is_retryable = False
            
            if is_retryable and attempt < max_retries - 1:
                print(f"      [LLM-Retry] {error_type} (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"      [LLM-Retry] Waiting {current_delay:.1f}s before retry...")
                time.sleep(current_delay)
                current_delay *= 2  # 指数退避
            else:
                # 非可重试错误或最后一次尝试
                print(f"      [LLM-Error] {error_type} (final attempt): {e}")
                return None  # 返回 None 而不是抛异常
    
    # 所有重试都失败了
    print(f"      [LLM-Error] All {max_retries} attempts failed. Last error: {last_exception}")
    return None


def retry_on_connection_error(func, max_retries=3, initial_delay=2.0, backoff_factor=2.0):
    """
    [Legacy] Retry wrapper for LLM calls with exponential backoff.
    保留向后兼容，新代码建议使用 llm_invoke_with_retry。
    
    Args:
        func: Callable that performs LLM invocation
        max_retries: Maximum number of retry attempts (default: 3)
        initial_delay: Initial delay in seconds before first retry (default: 2.0)
        backoff_factor: Multiplier for delay between retries (default: 2.0)
        
    Returns:
        Result of func() if successful
        
    Raises:
        Last exception if all retries fail
    """
    last_exception = None
    delay = initial_delay
    
    for attempt in range(1, max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_exception = e
            error_str = str(e).lower()
            
            # 分类错误类型（与 llm_invoke_with_retry 保持一致）
            is_retryable = any(keyword in error_str for keyword in [
                'connection', 'timeout', 'timed out', 'network',
                'refused', 'reset', 'broken pipe', 'unreachable',
                '500', '502', '503', '504', 'internal server error',
                'rate limit', 'too many requests', '429'
            ])
            
            if is_retryable and attempt < max_retries:
                print(f"      [Retry] Retryable error on attempt {attempt}/{max_retries}: {e}")
                print(f"      [Retry] Waiting {delay:.1f}s before retry...")
                time.sleep(delay)
                delay *= backoff_factor
            else:
                # Not a retryable error or last attempt, raise immediately
                raise
    
    # All retries exhausted
    raise last_exception

# ==============================================================================
# 数据模型
# ==============================================================================

class ModifiedLine(BaseModel):
    """Represents a modified code line (for diff analysis)"""
    line_number: int = Field(description="The line number in the respective version (pre-patch for deleted, post-patch for added).")
    content: str = Field(description="The code content (stripped of leading/trailing whitespace).")
    raw_content: str = Field(description="The raw content with original formatting.")
    extracted_vars: Set[str] = Field(default_factory=set, description="Variables extracted from this line.")


# ==============================================================================
# 静态程序切片引擎
# ==============================================================================

class Slicer:
    """
    PDG-based program slicer
    
    Supports three slicing strategies:
    - backward: backward slice (trace data origins)
    - forward: forward slice (trace data impacts)
    - bidirectional: bidirectional slice (complete context)
    """
    def __init__(self, pdg: nx.MultiDiGraph, code: str, start_line: int = 1):
        self.pdg = pdg
        self.code = code
        self.code_lines = code.splitlines()
        self.start_line = start_line

    def _normalize_code(self, s: str) -> str:
        return s.strip().replace(" ", "").replace("\t", "")

    def _find_line_numbers_by_content(self, target_content: str) -> List[int]:
        found_lines = []
        clean_target = self._normalize_code(target_content)
        if not clean_target:
            return []

        for i, line in enumerate(self.code_lines):
            clean_line = self._normalize_code(line)
            if clean_target in clean_line: 
                # Return strict file line number
                found_lines.append(self.start_line + i)
        return found_lines

    def get_nodes_by_location(self, line_num: int, code_content: str) -> List[str]:
        target_rel_lines = set()

        # Check if absolute line_num falls within this Slicer's file range
        # range is [start_line, start_line + len - 1]
        range_end = self.start_line + len(self.code_lines) - 1
        
        if line_num > 0 and self.start_line <= line_num <= range_end:
            # Convert passed Absolute line to Relative
            target_rel_lines.add(line_num - self.start_line + 1)
        
        # If explicitly looking for content, searching returns Absolute lines
        if code_content:
            found_abs = self._find_line_numbers_by_content(code_content)
            for fa in found_abs:
                target_rel_lines.add(fa - self.start_line + 1)

        if not target_rel_lines:
            return []

        candidates = []
        for n, d in self.pdg.nodes(data=True):
            if d.get('type') in ('EXIT', 'MERGE'): continue
            node_line = d.get('start_line', 0) # PDG nodes are Relative
            if node_line in target_rel_lines:
                candidates.append(n)
        return candidates

    def _is_relevant(self, tracked: Set[str], candidate: str) -> bool:
        """Handle struct hierarchy matching"""
        for t in tracked:
            if t == candidate: return True
            
            # Case 1: Track Parent, Edge is Child
            # Track: "skb", Edge: "skb->len" -> True
            if candidate.startswith(t + "->") or candidate.startswith(t + "."): return True
            
            # Case 2: Track Child, Edge is Parent (Refined)
            # Track: "dwc->gadget", Edge: "dwc" -> True
            # 意味着我们关注子字段，但父结构体被使用了(通常是传参)，这可能隐含对子字段的使用
            if t.startswith(candidate + "->") or t.startswith(candidate + "."): return True
            
        return False

    def backward_slice_pruned(self, start_nodes: List[str], initial_vars: Set[str]) -> Set[str]:
        slice_nodes = set(start_nodes)
        worklist = []
        for nid in start_nodes:
            worklist.append((nid, initial_vars.copy()))
        
        visited = set()

        while worklist:
            curr_node, curr_vars = worklist.pop(0)
            state = (curr_node, frozenset(curr_vars))
            if state in visited: continue
            visited.add(state)

            for pred, _, _, data in self.pdg.in_edges(curr_node, keys=True, data=True):
                rel_type = data.get('relationship')
                if rel_type == 'DATA':
                    edge_var = data.get('var')
                    if self._is_relevant(curr_vars, edge_var):
                        if pred not in slice_nodes: slice_nodes.add(pred)
                        pred_uses = set(self.pdg.nodes[pred].get('uses', {}).keys())
                        worklist.append((pred, pred_uses))
                elif rel_type == 'CONTROL':
                    if pred not in slice_nodes: slice_nodes.add(pred)
                    pred_uses = set(self.pdg.nodes[pred].get('uses', {}).keys())
                    worklist.append((pred, pred_uses))
        return slice_nodes
    
    def backward_slice_control_only(self, start_nodes: List[str]) -> Set[str]:
        """
        Trace only control dependencies, completely ignore data dependencies.
        Used to complete if/loop structures in shadow slices without introducing unrelated data computation.
        """
        slice_nodes = set(start_nodes)
        worklist = list(start_nodes)
        visited = set(start_nodes)

        while worklist:
            curr = worklist.pop(0)
            for pred, _, _, data in self.pdg.in_edges(curr, keys=True, data=True):
                if data.get('relationship') == 'CONTROL':
                    if pred not in visited:
                        visited.add(pred)
                        slice_nodes.add(pred)
                        worklist.append(pred)
        return slice_nodes

    def forward_slice_pruned(self, start_nodes: List[str], initial_vars: Set[str]) -> Set[str]:
        slice_nodes = set(start_nodes)
        # worklist 存储 (node, relevant_vars)
        # relevant_vars: 到达此节点时，引起依赖的变量集合
        worklist = []
        for nid in start_nodes:
            worklist.append((nid, initial_vars.copy()))
        
        # 修改 Visited 策略：
        # 如果我们已经以“更强”或“相同”的变量集合访问过该节点，则跳过
        # 但为了性能和防止死循环，最简单的方式是只记录 Node（变为上下文不敏感），
        # 或者限制 visited 的 key。
        # 这里我们采用一种折中方案：只在 edge 判断时过滤，入队时归一化。
        
        # 其实在 PDG Forward Slice 中，一旦被切中，该节点的所有 defs 都会成为新的污染源。
        # 因此状态里只需要存节点 ID 即可防止死循环。
        visited_nodes = set(start_nodes) 

        while worklist:
            curr_node, curr_vars = worklist.pop(0)

            # 遍历出边
            for _, succ, data in self.pdg.out_edges(curr_node, data=True):
                rel_type = data.get('relationship')
                should_add = False
                
                # 1. 数据依赖检查
                if rel_type == 'DATA':
                    edge_var = data.get('var')
                    # 只有当边上的变量是我们关心的变量时，才通过
                    if edge_var in curr_vars:
                        should_add = True
                
                # 2. 控制依赖检查
                elif rel_type == 'CONTROL':
                    # 控制依赖通常意味着 curr_node 的执行决定了 succ 是否执行
                    # 如果 curr_node 在切片里，它的所有控制子节点通常都应该在切片里
                    # 这里可以放宽条件，或者检查 curr_node 使用的变量是否在 curr_vars 里
                    curr_uses = set(self.pdg.nodes[curr_node].get('uses', {}).keys())
                    if not curr_uses.isdisjoint(curr_vars):
                        should_add = True

                if should_add:
                    # 获取 succ 定义的新变量
                    succ_defs = set(self.pdg.nodes[succ].get('defs', {}).keys())
                    
                    # --- 关键修改 ---
                    # 下一跳关心的变量，应该是 succ 产生的变量，而不是累积之前的变量。
                    # 因为在 PDG 中，curr_vars 已经被 curr_node 消费了，
                    # 产生的影响转化为了 succ_defs。
                    next_vars = succ_defs 
                    
                    # 如果 succ 尚未访问，或者是为了重新计算新的变量流（如果是上下文敏感分析），
                    # 但为了防止死循环，建议只要节点被加入切片，就只处理一次，
                    # 或者确保 visited 逻辑能覆盖。
                    if succ not in visited_nodes:
                        visited_nodes.add(succ)
                        slice_nodes.add(succ)
                        worklist.append((succ, next_vars))
        
        return slice_nodes

    def execute_strategy(self, anchors: List[str], focus_vars: Set[str], strategy: str) -> Set[str]:
        """
        Executes slicing based on the specific strategy (BACKWARD, FORWARD, etc.)
        strategy comes from SlicingStrategy enum.
        """
        if not anchors: return set()
        
        # Always expand focus vars from anchors initially
        for nid in anchors:
            if nid in self.pdg.nodes:
                uses = self.pdg.nodes[nid].get('uses', {})
                if uses: focus_vars.update(uses.keys())
        
        result_nodes = set(anchors)
        
        if strategy == "backward":
            # Root Cause Analysis: Find where data came from
            bwd = self.backward_slice_pruned(anchors, focus_vars)
            result_nodes.update(bwd)
            
        elif strategy == "forward":
            # Impact Analysis: Find where data goes
            fwd = self.forward_slice_pruned(anchors, focus_vars)
            result_nodes.update(fwd)
            
        elif strategy == "control_only":
            # Guard Analysis: Find conditions regulating execution
            ctrl = self.backward_slice_control_only(anchors)
            result_nodes.update(ctrl)
            
        else: # "bidirectional" or default
            # Context Restoration
            bwd = self.backward_slice_pruned(anchors, focus_vars)
            fwd = self.forward_slice_pruned(anchors, focus_vars)
            result_nodes.update(bwd)
            result_nodes.update(fwd)
            
        return result_nodes

    def robust_slice(self, anchors: List[str], focus_vars: Set[str]) -> Set[str]:
        # Legacy Wrapper
        return self.execute_strategy(anchors, focus_vars, "bidirectional")

    def to_code(self, node_ids: Set[str]) -> str:
        nodes_data = []
        for nid in node_ids:
            if nid not in self.pdg: continue
            data = self.pdg.nodes[nid]
            # [FIX] Include ENTRY nodes (function signatures) in output
            # Only filter out EXIT, MERGE, NO_OP
            if data.get('type') in ('EXIT', 'MERGE', 'NO_OP'): continue
            nodes_data.append(data)
        
        # 1. Collect initial lines from nodes
        lines_to_print = set()
        # file_end_limit = self.start_line + len(self.code_lines) - 1
        
        for n in nodes_data:
            start_rel = n.get('start_line', 0)
            end_rel = n.get('end_line', start_rel)
            
            # [FIX] ENTRY nodes may have start_line=0, treat them as line 1
            if n.get('type') == 'ENTRY' and start_rel == 0:
                start_rel = 1
                end_rel = 1

            if start_rel > 0:
                max_rel = len(self.code_lines)
                actual_start_rel = max(1, start_rel)
                actual_end_rel = min(end_rel, max_rel)
                
                if actual_start_rel <= actual_end_rel:
                    for rel_line in range(actual_start_rel, actual_end_rel + 1):
                        lines_to_print.add(rel_line)
        
        # 2. (Removed) Heuristic Expansion for Line Continuations
        # Since CFGNode now robustly captures `end_line` from Tree-sitter, we don't need
        # to guess line continuations based on trailing commas/operators.

        # 3. Output generation
        output = []
        sorted_lines = sorted(list(lines_to_print))
        
        for rel_line in sorted_lines:
            idx = rel_line - 1
            if 0 <= idx < len(self.code_lines):
                abs_line = rel_line - 1 + self.start_line
                output.append(f"[{abs_line:4d}] {self.code_lines[idx]}")
        
        return "\n".join(output)

# ==============================================================================
# Shadow 切片映射器（上下文同步）
# ==============================================================================

class LineMappingType:
    """Enumeration of line mapping types"""
    EXACT = "EXACT"           # Identical lines (same content and position)
    WHITESPACE = "WHITESPACE" # Lines with only indentation changes
    MOVED = "MOVED"           # Content in both deleted and added, different position
    MODIFIED = "MODIFIED"     # Similar but not identical (context-based detection)
    UNIQUE_PRE = "UNIQUE_PRE"   # Only in pre-patch (deleted, no corresponding added)
    UNIQUE_POST = "UNIQUE_POST" # Only in post-patch (added, no corresponding deleted)


class LineMapping:
    """
    Represents a mapping between pre-patch and post-patch lines.
    
    All mappings are stored as tuples:
    - EXACT/WHITESPACE/MOVED/MODIFIED: (pri_rel, shadow_rel, type, pri_content, shadow_content)
    - UNIQUE_PRE: (pri_rel, None, type, pri_content, None)
    - UNIQUE_POST: (None, shadow_rel, type, None, shadow_content)
    
    This structure is designed to be reusable in the search/matching phase,
    where we need to handle cases like: if both sides of a MODIFIED pair
    match the target, only count the higher-scoring one.
    """
    def __init__(self, pri_rel: int = None, shadow_rel: int = None,
                 mapping_type: str = None, pri_content: str = None, shadow_content: str = None):
        self.pri_rel = pri_rel
        self.shadow_rel = shadow_rel
        self.mapping_type = mapping_type
        self.pri_content = pri_content
        self.shadow_content = shadow_content
    
    def to_tuple(self):
        return (self.pri_rel, self.shadow_rel, self.mapping_type, self.pri_content, self.shadow_content)
    
    def __repr__(self):
        return f"LineMapping({self.mapping_type}: {self.pri_rel} <-> {self.shadow_rel})"


class ShadowMapper:
    """
    Shadow Slice Generator - Maps Primary (Pre-Patch) slice to Shadow (Post-Patch)
    
    Core Responsibilities:
    1. Build Primary ↔ Shadow line mapping (based on diff analysis)
    2. Map Primary nodes to Shadow nodes
    3. Inject deleted vulnerability code lines (for completeness display)
    4. Apply synchronization cleanup rules (maintain semantic consistency)
    
    Line Mapping Types (based on diff analysis):
    - EXACT: Identical lines (same content and position, not in diff)
    - WHITESPACE: Lines with only indentation changes (same stripped content)
    - MOVED: Content (stripped) appears in both deleted (-) and added (+) lines, but at different positions
    - MODIFIED: Context-based detection - if surrounding lines match, the middle line is MODIFIED
    - UNIQUE_PRE: Lines only in deleted (-), no corresponding added line
    - UNIQUE_POST: Lines only in added (+), no corresponding deleted line
    
    All mappings are stored as LineMapping objects for reuse in search/matching phase.
    """
    
    def __init__(self, code_pri: str, code_shadow: str,
                 pdg_pri: nx.MultiDiGraph, pdg_shadow: nx.MultiDiGraph,
                 start_line_pri: int = 1, start_line_shadow: int = 1,
                 diff_text: str = None):
        self.code_pri = code_pri
        self.code_shadow = code_shadow
        self.lines_pri = code_pri.splitlines()
        self.lines_shadow = code_shadow.splitlines()
        self.pdg_pri = pdg_pri
        self.pdg_shadow = pdg_shadow
        self.sl_pri = start_line_pri
        self.sl_shadow = start_line_shadow
        self.diff_text = diff_text
        
        # ============ Core Mapping Storage ============
        # All mappings stored as LineMapping objects
        self.line_mappings: List[LineMapping] = []
        
        # Quick lookup indexes (built from line_mappings)
        self.pri_to_shadow = {}      # pri_rel -> shadow_rel (for EXACT/WHITESPACE)
        self.shadow_to_pri = {}      # shadow_rel -> pri_rel (for EXACT/WHITESPACE)
        self.moved_pri_to_shadow = {}    # pri_rel -> shadow_rel (for MOVED)
        self.moved_shadow_to_pri = {}    # shadow_rel -> pri_rel (for MOVED)
        self.modified_pri_to_shadow = {}  # pri_rel -> shadow_rel (for MODIFIED)
        self.modified_shadow_to_pri = {}  # shadow_rel -> pri_rel (for MODIFIED)
        self.unique_pre_lines = set()    # pri_rel lines that are UNIQUE_PRE
        self.unique_post_lines = set()   # shadow_rel lines that are UNIQUE_POST
        
        # ============ Diff-based line storage ============
        self.deleted_lines = {}      # pri_rel -> content (lines marked with '-' in diff)
        self.added_lines = {}        # shadow_rel -> content (lines marked with '+' in diff)
        self.deleted_content_set = set()  # stripped content of deleted lines
        self.deleted_content_to_line = {} # stripped content -> pri_rel (for mapping)
        
        # ============ Content indexes ============
        self.pri_content_to_lines = {}    # stripped_content -> set of rel_lines
        self.shadow_content_to_lines = {}
        
        self._build_mappings()
    
    def _parse_diff_lines(self):
        """Extract deleted and added lines from diff text"""
        if not self.diff_text:
            return
        
        hunk_re = re.compile(r'^@@\s*-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@')
        c_old = 0  # pre-patch (primary) line counter
        c_new = 0  # post-patch (shadow) line counter
        
        for line in self.diff_text.splitlines():
            # Skip file headers
            if line.startswith('---') or line.startswith('+++'):
                continue
            
            # Parse hunk header
            m = hunk_re.match(line)
            if m:
                c_old = int(m.group(1))
                c_new = int(m.group(3))
                continue
            
            # Deleted line (-): exists in pre-patch, removed in post-patch
            if line.startswith('-') and not line.startswith('---'):
                raw_content = line[1:]
                stripped = raw_content.strip()
                if stripped and not stripped.startswith(('//', '/*', '*', '#')):
                    # Convert to relative line number
                    rel_pri = c_old - self.sl_pri + 1
                    if rel_pri > 0:
                        self.deleted_lines[rel_pri] = stripped
                        self.deleted_content_set.add(stripped)
                        # Record content -> line mapping for MOVED detection
                        if stripped not in self.deleted_content_to_line:
                            self.deleted_content_to_line[stripped] = rel_pri
                c_old += 1
            
            # Added line (+): new in post-patch
            elif line.startswith('+') and not line.startswith('+++'):
                raw_content = line[1:]
                stripped = raw_content.strip()
                if stripped and not stripped.startswith(('//', '/*', '*', '#')):
                    # Convert to relative line number
                    rel_shadow = c_new - self.sl_shadow + 1
                    if rel_shadow > 0:
                        self.added_lines[rel_shadow] = stripped
                c_new += 1
            
            # Context line (both versions)
            else:
                c_old += 1
                c_new += 1
    
    def _get_context_lines(self, lines: List[str], rel_line: int) -> Tuple[str, str]:
        """Get the stripped content of previous and next lines for context matching"""
        prev_content = ""
        next_content = ""
        
        if rel_line > 1 and rel_line - 2 < len(lines):
            prev_content = lines[rel_line - 2].strip()  # rel_line is 1-based
        if rel_line < len(lines):
            next_content = lines[rel_line].strip()  # rel_line is 1-based, so rel_line is next
        
        return prev_content, next_content
    
    def _classify_added_lines(self):
        """
        Classify added lines into MOVED, MODIFIED, or UNIQUE and build LineMapping objects.
        
        Classification rules:
        - MOVED: Content (stripped) appears in both deleted (-) and added (+) lines
        - MODIFIED: Context-based detection - surrounding lines match between deleted and added
        - UNIQUE_POST: No corresponding deleted line found
        - UNIQUE_PRE: Deleted lines with no corresponding added line
        
        All mappings are stored as LineMapping tuples for reuse in search/matching.
        """
        # Track which deleted lines have been matched
        matched_deleted_lines = set()
        
        for shadow_rel, shadow_content in self.added_lines.items():
            # ===== Check for MOVED: exact content match (after strip) =====
            if shadow_content in self.deleted_content_set:
                pri_rel = self.deleted_content_to_line.get(shadow_content)
                if pri_rel is not None:
                    pri_content = self.deleted_lines.get(pri_rel, shadow_content)
                    # Create MOVED mapping
                    mapping = LineMapping(
                        pri_rel=pri_rel,
                        shadow_rel=shadow_rel,
                        mapping_type=LineMappingType.MOVED,
                        pri_content=pri_content,
                        shadow_content=shadow_content
                    )
                    self.line_mappings.append(mapping)
                    self.moved_pri_to_shadow[pri_rel] = shadow_rel
                    self.moved_shadow_to_pri[shadow_rel] = pri_rel
                    matched_deleted_lines.add(pri_rel)
                continue
            
            # ===== Check for MODIFIED: context-based detection =====
            # If the surrounding lines (prev and next) match between deleted and added,
            # then the middle line is MODIFIED
            shadow_prev, shadow_next = self._get_context_lines(self.lines_shadow, shadow_rel)
            
            best_match_pri = None
            for pri_rel, pri_content in self.deleted_lines.items():
                if pri_rel in matched_deleted_lines:
                    continue  # Already matched
                
                pri_prev, pri_next = self._get_context_lines(self.lines_pri, pri_rel)
                
                # Context matching: both prev and next lines should match
                # Allow matching if at least one of prev/next matches (for edge cases)
                prev_match = (pri_prev and shadow_prev and pri_prev == shadow_prev)
                next_match = (pri_next and shadow_next and pri_next == shadow_next)
                
                # Require both context lines to match for MODIFIED classification
                if prev_match and next_match:
                    best_match_pri = pri_rel
                    break
            
            if best_match_pri is not None:
                pri_content = self.deleted_lines.get(best_match_pri, "")
                # Create MODIFIED mapping
                mapping = LineMapping(
                    pri_rel=best_match_pri,
                    shadow_rel=shadow_rel,
                    mapping_type=LineMappingType.MODIFIED,
                    pri_content=pri_content,
                    shadow_content=shadow_content
                )
                self.line_mappings.append(mapping)
                self.modified_pri_to_shadow[best_match_pri] = shadow_rel
                self.modified_shadow_to_pri[shadow_rel] = best_match_pri
                matched_deleted_lines.add(best_match_pri)
            else:
                # UNIQUE_POST: no matching deleted line
                mapping = LineMapping(
                    pri_rel=None,
                    shadow_rel=shadow_rel,
                    mapping_type=LineMappingType.UNIQUE_POST,
                    pri_content=None,
                    shadow_content=shadow_content
                )
                self.line_mappings.append(mapping)
                self.unique_post_lines.add(shadow_rel)
        
        # ===== Process unmatched deleted lines as UNIQUE_PRE =====
        for pri_rel, pri_content in self.deleted_lines.items():
            if pri_rel not in matched_deleted_lines:
                mapping = LineMapping(
                    pri_rel=pri_rel,
                    shadow_rel=None,
                    mapping_type=LineMappingType.UNIQUE_PRE,
                    pri_content=pri_content,
                    shadow_content=None
                )
                self.line_mappings.append(mapping)
                self.unique_pre_lines.add(pri_rel)
    
    def _build_mappings(self):
        """Unified construction of all mapping relationships"""
        from collections import defaultdict
        
        # Step 1: Parse diff to extract deleted/added lines
        self._parse_diff_lines()
        
        # Step 2: Build content indexes
        pri_content_map = defaultdict(set)
        shadow_content_map = defaultdict(set)
        
        for i, line in enumerate(self.lines_pri, 1):
            stripped = line.strip()
            if stripped and not stripped.startswith(('//', '/*', '*', '#')):
                pri_content_map[stripped].add(i)
        
        for i, line in enumerate(self.lines_shadow, 1):
            stripped = line.strip()
            if stripped and not stripped.startswith(('//', '/*', '*', '#')):
                shadow_content_map[stripped].add(i)
        
        self.pri_content_to_lines = dict(pri_content_map)
        self.shadow_content_to_lines = dict(shadow_content_map)
        
        # Step 3: Use SequenceMatcher for EXACT and WHITESPACE mappings
        matcher = difflib.SequenceMatcher(None, self.lines_pri, self.lines_shadow)
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                # EXACT: 完全相同的行 - 双向映射
                for k in range(i2 - i1):
                    p_rel, s_rel = i1 + k + 1, j1 + k + 1
                    pri_content = self.lines_pri[i1 + k].strip() if (i1 + k) < len(self.lines_pri) else ""
                    shadow_content = self.lines_shadow[j1 + k].strip() if (j1 + k) < len(self.lines_shadow) else ""
                    
                    # Create EXACT mapping
                    mapping = LineMapping(
                        pri_rel=p_rel,
                        shadow_rel=s_rel,
                        mapping_type=LineMappingType.EXACT,
                        pri_content=pri_content,
                        shadow_content=shadow_content
                    )
                    self.line_mappings.append(mapping)
                    self.pri_to_shadow[p_rel] = s_rel
                    self.shadow_to_pri[s_rel] = p_rel
                    
            elif tag == 'replace':
                # Check for WHITESPACE: same content after stripping
                pri_block = self.lines_pri[i1:i2]
                shadow_block = self.lines_shadow[j1:j2]
                
                # Try 1:1 correspondence when block sizes match
                if len(pri_block) == len(shadow_block):
                    for k in range(len(pri_block)):
                        if pri_block[k].strip() == shadow_block[k].strip():
                            # WHITESPACE: only indentation changed
                            p_rel, s_rel = i1 + k + 1, j1 + k + 1
                            pri_content = pri_block[k].strip()
                            shadow_content = shadow_block[k].strip()
                            
                            # Create WHITESPACE mapping
                            mapping = LineMapping(
                                pri_rel=p_rel,
                                shadow_rel=s_rel,
                                mapping_type=LineMappingType.WHITESPACE,
                                pri_content=pri_content,
                                shadow_content=shadow_content
                            )
                            self.line_mappings.append(mapping)
                            self.pri_to_shadow[p_rel] = s_rel
                            self.shadow_to_pri[s_rel] = p_rel
                else:
                    # Different block sizes: try content matching for WHITESPACE
                    pri_stripped = {line.strip(): (i1 + idx + 1, line.strip())
                                    for idx, line in enumerate(pri_block) if line.strip()}
                    for idx, line in enumerate(shadow_block):
                        stripped = line.strip()
                        if stripped and stripped in pri_stripped:
                            p_rel, pri_content = pri_stripped[stripped]
                            s_rel = j1 + idx + 1
                            # Only map if this shadow line is NOT in added_lines
                            # (to avoid conflicting with MOVED classification)
                            if s_rel not in self.added_lines:
                                # Create WHITESPACE mapping
                                mapping = LineMapping(
                                    pri_rel=p_rel,
                                    shadow_rel=s_rel,
                                    mapping_type=LineMappingType.WHITESPACE,
                                    pri_content=pri_content,
                                    shadow_content=stripped
                                )
                                self.line_mappings.append(mapping)
                                self.pri_to_shadow[p_rel] = s_rel
                                self.shadow_to_pri[s_rel] = p_rel
        
        # Step 4: Classify added lines into MOVED, MODIFIED, UNIQUE and build mappings
        self._classify_added_lines()
        
        # Count mappings by type for logging
        type_counts = {}
        for m in self.line_mappings:
            type_counts[m.mapping_type] = type_counts.get(m.mapping_type, 0) + 1
        
        print(f"      [Mapping] Classification: EXACT={type_counts.get(LineMappingType.EXACT, 0)}, "
              f"WHITESPACE={type_counts.get(LineMappingType.WHITESPACE, 0)}, "
              f"MOVED={type_counts.get(LineMappingType.MOVED, 0)}, "
              f"MODIFIED={type_counts.get(LineMappingType.MODIFIED, 0)}, "
              f"UNIQUE_PRE={type_counts.get(LineMappingType.UNIQUE_PRE, 0)}, "
              f"UNIQUE_POST={type_counts.get(LineMappingType.UNIQUE_POST, 0)}")
    
    def generate_shadow_slice(self, 
                              pri_nodes: Set[str], 
                              deleted_lines: List[str] = None) -> Set[str]:
        """
        一步到位生成清理后的 Shadow Slice
        
        Args:
            pri_nodes: Primary Slice 中的节点集合
            deleted_lines: 被删除的代码行内容列表 (用于注入漏洞核心)
            
        Returns:
            清理后的 Shadow 节点集合
        """
        shadow_nodes = set()
        slicer_shadow = Slicer(self.pdg_shadow, self.code_shadow, self.sl_shadow)
        
        # 收集 Primary 活跃信息
        active_pri_lines = set()      # Primary 中被保留的行号 (rel)
        active_pri_contents = set()   # Primary 中被保留的行内容 (stripped)
        
        for nid in pri_nodes:
            if nid not in self.pdg_pri.nodes:
                continue
            d = self.pdg_pri.nodes[nid]
            ln = d.get('start_line', 0)
            if ln > 0:
                active_pri_lines.add(ln)
                if 1 <= ln <= len(self.lines_pri):
                    content = self.lines_pri[ln - 1].strip()
                    if content:
                        active_pri_contents.add(content)
        
        if not active_pri_lines:
            return set()
        
        # ============ Step 1: 映射 Primary 节点 → Shadow 节点 ============
        for nid in pri_nodes:
            if nid not in self.pdg_pri.nodes:
                continue
            pri_rel = self.pdg_pri.nodes[nid].get('start_line', 0)
            
            if pri_rel in self.pri_to_shadow:
                shadow_rel = self.pri_to_shadow[pri_rel]
                found = [n for n, d in self.pdg_shadow.nodes(data=True) 
                         if d.get('start_line') == shadow_rel]
                shadow_nodes.update(found)
        
        # ============ Step 2: 处理 Replace 块（确保新增的修复代码被包含）============
        # Build replace block mapping from diff opcodes
        matcher_internal = difflib.SequenceMatcher(None, self.lines_pri, self.lines_shadow)
        replace_blocks = []  # List of ((pri_start, pri_end), (shadow_start, shadow_end))
        for tag, i1, i2, j1, j2 in matcher_internal.get_opcodes():
            if tag == 'replace':
                replace_blocks.append(((i1 + 1, i2), (j1 + 1, j2)))
        
        # For each active Primary line in a replace block, include entire corresponding Shadow block
        replace_shadow_lines = set()
        for (p_start, p_end), (s_start, s_end) in replace_blocks:
            # Check if any active Primary line falls in this replace block
            for pri_ln in active_pri_lines:
                if p_start <= pri_ln <= p_end:
                    # Include all Shadow lines in corresponding block
                    for s_ln in range(s_start, s_end + 1):
                        replace_shadow_lines.add(s_ln)
                    break
        
        # Add nodes for replace block Shadow lines
        for s_ln in replace_shadow_lines:
            found = [n for n, d in self.pdg_shadow.nodes(data=True)
                     if d.get('start_line') == s_ln]
            shadow_nodes.update(found)
        
        # ============ Step 3: 注入删除行 (漏洞核心) ============
        deleted_anchors = set()
        if deleted_lines:
            for content in deleted_lines:
                found = slicer_shadow.get_nodes_by_location(-1, content)
                deleted_anchors.update(found)
            
            if deleted_anchors:
                # 对删除行做局部切片扩展 (仅控制流，避免引入过多噪声)
                ctrl_nodes = slicer_shadow.backward_slice_control_only(list(deleted_anchors))
                shadow_nodes.update(ctrl_nodes)
                shadow_nodes.update(deleted_anchors)
        
        # ============ Step 3.5: 添加INSERT行节点（补丁新增代码）============
        # [FIX] INSERT行的节点需要主动添加，因为它们不在映射中
        print(f"      [Mapping] Adding INSERT lines ({len(self.unique_post_lines)} lines)...")
        for s_ln in self.unique_post_lines:
            # Find all nodes that COVER this line (start_line <= s_ln <= end_line)
            # This handles multi-line statements like if-blocks
            found = [n for n, d in self.pdg_shadow.nodes(data=True)
                     if (d.get('start_line', 0) <= s_ln <= d.get('end_line', d.get('start_line', 0))
                         and d.get('type') not in ('EXIT', 'MERGE', 'NO_OP'))]
            if found:
                shadow_nodes.update(found)
        
        # ============ Step 4: 计算语义范围锚点 ============
        min_pri, max_pri = min(active_pri_lines), max(active_pri_lines)
        
        # 找 Primary 范围边界在 Shadow 中的对应位置
        anchor_top_s = 0
        for r in range(min_pri - 1, 0, -1):
            if r in self.pri_to_shadow:
                anchor_top_s = self.pri_to_shadow[r]
                break
        
        anchor_bot_s = len(self.lines_shadow) + 1
        for r in range(max_pri + 1, len(self.lines_pri) + 1):
            if r in self.pri_to_shadow:
                anchor_bot_s = self.pri_to_shadow[r]
                break
        
        # ============ Step 5: 同步清理 (应用保留规则) ============
        # Line Classification (based on diff analysis):
        # - EXACT/WHITESPACE: Lines with direct mapping (pri_to_shadow/shadow_to_pri)
        # - MOVED: Content appears in both deleted (-) and added (+) lines
        # - MODIFIED: Content is similar to a deleted line (e.g., printf → safe_printf)
        # - UNIQUE: Pure new lines, no corresponding deleted line
        
        nodes_to_remove = set()
        
        for nid in shadow_nodes:
            if nid not in self.pdg_shadow.nodes:
                continue
            
            d = self.pdg_shadow.nodes[nid]
            s_ln = d.get('start_line', 0)
            
            if s_ln <= 0 or d.get('type') in ('ENTRY', 'EXIT'):
                continue
            
            # 强制保留删除行（漏洞核心代码）
            if nid in deleted_anchors:
                continue
            
            # 强制保留 replace 块的新增行
            if s_ln in replace_shadow_lines:
                continue
            
            should_keep = False
            line_type = "UNKNOWN"
            
            # Type 1: EXACT/WHITESPACE (有映射关系 - 不在 diff 中修改的行)
            if s_ln in self.shadow_to_pri:
                mapped_pri = self.shadow_to_pri[s_ln]
                # 只有当 Primary 中对应的行在活跃集合中时，才保留
                should_keep = (mapped_pri in active_pri_lines)
                line_type = "EXACT/WHITESPACE"
            
            # Type 2: MOVED (内容同时出现在删除行和增加行中，但位置不同)
            # 使用双向映射来判断：只有当 Primary 中对应的行在活跃集合中时，才保留
            elif s_ln in self.moved_shadow_to_pri:
                mapped_pri = self.moved_shadow_to_pri[s_ln]
                # MOVED 行：只有当 Primary 中对应的行在活跃集合中时，才保留
                should_keep = (mapped_pri in active_pri_lines)
                line_type = "MOVED"
            
            # Type 3: MODIFIED (内容与删除行相似但不完全相同)
            # 使用双向映射来判断：只有当 Primary 中对应的行在活跃集合中时，才保留
            elif s_ln in self.modified_shadow_to_pri:
                mapped_pri = self.modified_shadow_to_pri[s_ln]
                # MODIFIED 行：只有当 Primary 中对应的行在活跃集合中时，才保留
                should_keep = (mapped_pri in active_pri_lines)
                line_type = "MODIFIED"
            
            # Type 4: UNIQUE_POST (纯新增，删除行中找不到对应)
            # 这是补丁新增的全新代码，应该保留（没有对应的 Primary 行）
            elif s_ln in self.unique_post_lines:
                # UNIQUE_POST 行：补丁新增的代码，全部保留
                should_keep = True
                line_type = "UNIQUE_POST"
            
            # Type 5: 其他情况（不在 diff 中，也没有映射）
            # 在语义范围内则保留
            else:
                should_keep = (anchor_top_s < s_ln < anchor_bot_s)
                line_type = "OTHER"
            
            if not should_keep:
                nodes_to_remove.add(nid)
        
        shadow_nodes.difference_update(nodes_to_remove)
        
        # 确保 ENTRY 节点存在
        shadow_nodes.update([n for n, d in self.pdg_shadow.nodes(data=True) 
                            if d.get('type') == 'ENTRY'])
        
        return shadow_nodes
    
    def map_lines_to_shadow(self, pri_abs_lines: List[int]) -> List[int]:
        """
        映射 Primary 行号 → Shadow 行号
        
        Args:
            pri_abs_lines: Primary 中的绝对行号列表
            
        Returns:
            对应的 Shadow 绝对行号列表
        """
        result = []
        
        for abs_p in pri_abs_lines:
            rel_p = abs_p - self.sl_pri + 1
            
            # Case 1: 直接映射
            if rel_p in self.pri_to_shadow:
                rel_s = self.pri_to_shadow[rel_p]
                result.append(rel_s - 1 + self.sl_shadow)
                continue
            
            # Case 2: 内容相似度匹配
            if 1 <= rel_p <= len(self.lines_pri):
                p_content = self.lines_pri[rel_p - 1].strip()
                if p_content:
                    # 在 Shadow 中查找相同内容
                    if p_content in self.shadow_content_to_lines:
                        # 取最近的一个
                        candidates = list(self.shadow_content_to_lines[p_content])
                        if candidates:
                            best_s_rel = min(candidates, key=lambda x: abs(x - rel_p))
                            result.append(best_s_rel - 1 + self.sl_shadow)
        
        return result
    
    # ============ 兼容旧接口 (静态方法) ============
    @staticmethod
    def map_and_slice(s_pri_nodes: Set[str],
                      pdg_pri: nx.MultiDiGraph,
                      pdg_shadow: nx.MultiDiGraph,
                      code_pri: str,
                      code_shadow: str,
                      start_line_pri: int = 1,
                      start_line_shadow: int = 1,
                      deleted_lines: List[str] = None,
                      diff_text: str = None) -> Tuple[Set[str], Set[str]]:
        """
        [兼容旧接口] 映射并切片
        Returns: (all_shadow_nodes, mandatory_nodes)
        """
        mapper = ShadowMapper(code_pri, code_shadow, pdg_pri, pdg_shadow,
                              start_line_pri, start_line_shadow, diff_text)
        shadow_nodes = mapper.generate_shadow_slice(s_pri_nodes, deleted_lines)
        # mandatory_nodes 不再使用，返回空集合
        return (shadow_nodes, set())

# ==============================================================================
# 切片验证器与清洗器
# ==============================================================================

class SliceValidator:
    """
    Agent-driven slice distiller (Distillation)
    
    Responsibility: Semantically review static slices (forward ∪ backward) to remove noise code
    
    Distillation Strategy (based on Methodology §3.3.2):
    1. **Anchor Retention**: Origin and Impact anchors are unconditionally kept
    2. **Semantic Filtering**: Use LLM to review all data-flow and control-flow statements
       - Keep statements directly participating in vulnerability mechanism
       - Remove irrelevant noise (e.g., logging, feature flags, unrelated error handling)
    
    Uses positive selection strategy: explicitly mark lines to keep
    """
    def __init__(self):
        self.llm = ChatOpenAI(
            base_url=os.getenv("API_BASE"),
            api_key=os.getenv("API_KEY"),
            model=os.getenv("MODEL_NAME", "gpt-4o"),
            temperature=0
        )

    def _create_tools(self, navigator: CodeNavigator, file_path: str, slice_code: str):
        # Define tools bound to this specific navigator and file
        
        @tool
        def trace_variable(line: int, var_name: str, start: int = 0, end: Optional[int] = None) -> str:
            """
            Trace where a variable comes from (backward data flow analysis).
            Returns results in specified range [start:end]. If end is None, returns all results from start.
            The response includes 'total_count' to inform you of the total number of trace steps.
            """
            try:
                all_trace = navigator.trace_data_flow(file_path, line, var_name, direction="backward", limit_lines=None)
                total = len(all_trace) if isinstance(all_trace, list) else 0
                
                # Apply range
                if end is None:
                    sliced_trace = all_trace[start:]
                else:
                    sliced_trace = all_trace[start:end]
                
                return json.dumps({"total_count": total, "trace": sliced_trace, "showing_range": f"[{start}:{end if end else 'end'}]"})
            except Exception as e:
                return f"Error executing tool 'trace_variable': {e}. Please check your input arguments."

        @tool
        def get_control_dependency(line: int) -> str:
            """Find conditions (if/while) that control execution of this line."""
            try:
               return json.dumps(navigator.get_dominating_conditions(file_path, line))
            except Exception as e:
                return f"Error executing tool 'get_control_dependency': {e}. Please check your input arguments."

        @tool
        def read_original_code(start: int, end: int) -> str:
            """
            Read original source code from file (useful if slice context is unclear).
            You must specify both start and end line numbers based on your analysis needs.
            """
            try:
                return navigator.read_code_window(file_path, start, end)
            except Exception as e:
                return f"Error executing tool 'read_original_code': {e}. Please check your input arguments."

        @tool
        def check_variable_usage(var_name: str, start: int = 0, end: Optional[int] = None) -> str:
            """
            Find other usages of this variable in the function to see its role.
            Returns results in specified range [start:end]. If end is None, returns all results from start.
            The response includes 'total_count' to inform you of the total number of usages.
            """
            try:
                all_usages = navigator.find_variable_lines(file_path, var_name, limit=None)
                total = len(all_usages) if isinstance(all_usages, list) else 0
                
                # Apply range
                if end is None:
                    sliced_usages = all_usages[start:]
                else:
                    sliced_usages = all_usages[start:end]
                
                return json.dumps({"total_count": total, "usages": sliced_usages, "showing_range": f"[{start}:{end if end else 'end'}]"})
            except Exception as e:
                return f"Error executing tool 'check_variable_usage': {e}. Please check your input arguments."

        return [trace_variable, get_control_dependency, read_original_code, check_variable_usage]

    def distill_slice(self, slice_code: str, diff_text: str, commit_message: str,
                      focus_var: str, vuln_type: GeneralVulnType,
                      hypothesis: TaxonomyFeature, func_name: str,
                      origins: List[AnchorItem] = [], impacts: List[AnchorItem] = [],
                      cwe_info: str = "Unknown",
                      navigator: CodeNavigator = None,
                      file_path: str = None) -> Optional[SliceValidationResult]:
        """
        Use LLM + tools for semantic filtering to remove noise from slice
        
        Args:
            slice_code: Candidate slice code
            diff_text: Patch diff
            commit_message: Commit message
            focus_var: Focus variable
            vuln_type: Vulnerability type (GeneralVulnType enum)
            hypothesis: Vulnerability hypothesis (taxonomy)
            func_name: Function name
            origins: Origin anchors
            impacts: Impact anchors
            cwe_info: CWE information
            navigator: Code navigator
            file_path: File path
            
        Returns:
            SliceValidationResult or None
        """
        if not slice_code:
            return None
        
        # Origins/Impacts text
        origins_text = "\n".join([f"- Line {o.line}: {o.content}" for o in origins]) if origins else "None identified."
        impacts_text = "\n".join([f"- Line {i.line}: {i.content}" for i in impacts]) if impacts else "None identified."

        # Primary 固定为 Pre-Patch (Vulnerable)
        slice_version = "Pre-Patch (Vulnerable)"
        
        # Get vulnerability-type specific distillation strategy
        vuln_type_strategy = get_strategy_prompt_section(vuln_type)
        vuln_type_name = vuln_type.value if isinstance(vuln_type, GeneralVulnType) else str(vuln_type)
        
        print(f"      [Distillation] Engaging semantic filtering ({len(slice_code.splitlines())} lines).")
        print(f"      [Distillation] Using strategy for: {vuln_type_name}")
        
        # Setup Tools
        tools = self._create_tools(navigator, file_path, slice_code)
        llm_with_tools = self.llm.bind_tools(tools)

        # Count input slice lines for adaptive threshold
        input_line_count = len(slice_code.splitlines())
        
        # Adaptive line limit: base 10, +5 per 50 input lines, max 30
        max_output_lines = min(10 + (input_line_count // 50) * 5, 30)
        
        system_prompt = f"""
        You are an expert **Semantic Reviewer** for vulnerability slicing.
        
        ### Background
        The slice is generated from static program analysis (forward ∪ backward slicing from Origins and Impacts).
        It may contain noise: irrelevant data processing, logging, feature flags, etc.
        
        ### Your Task
        Review **all statements** (data-flow and control-flow) to identify which are essential to the vulnerability mechanism.
        
        **CRITICAL CONSTRAINT**: You must be **highly selective**. Keep ONLY the minimal set of lines that directly participate in the vulnerability mechanism.
        - **Maximum lines to keep**: {max_output_lines} lines (strictly enforced)
        - **Prioritize**: Origin/Impact anchors > direct data-flow > control guards > context
        - **Be ruthless**: If uncertain whether a line is essential, REMOVE it
        
        ### Context
        - **Function**: `{func_name}`
        - **Vuln Type**: {vuln_type_name} ({cwe_info})
        - **Root Cause**: {hypothesis.root_cause}
        - **Attack Path**: {hypothesis.attack_path}
        - **Key Variable**: {focus_var}
        - **Origin Anchors** (vulnerability initiation): {origins_text}
        - **Impact Anchors** (vulnerability manifestation): {impacts_text}
        - **Version**: {slice_version}
        
        ### Input Data
        - **Patch Diff**: {diff_text}
        - **Candidate Slice** ({input_line_count} lines):
        {slice_code}

        ### Classification Rules (Strict Priority Order)
        **Priority 1 - MUST KEEP**:
        - Origin or Impact anchor lines (MANDATORY)
        
        **Priority 2 - KEEP if directly relevant**:
        - Data-flow statements that directly compute or propagate the vulnerable data to Impact
        - Control guards that directly determine if vulnerability is triggered
        - Lines modified in the patch diff
        
        **Priority 3 - REMOVE (even if related)**:
        - Intermediate variable assignments that don't directly affect vulnerability
        - Logging, debugging, feature flags
        - Error handling for unrelated failures
        - Constants or non-vulnerable parameters
        - Context lines that provide background but don't participate in attack
        
        {vuln_type_strategy}
        
        ### Analysis Tools
        - Use `trace_variable` to verify if a variable reaches Impact
        - Use `get_control_dependency` to check if control statement guards vulnerability
        - Use `check_variable_usage` to see variable's role in function
        
        ### Output Format
        Return JSON with:
        - `relevant_lines`: List of line numbers to **KEEP** (positive selection, MAX {max_output_lines} lines)
        - `reasoning`: Brief explanation of retention decisions and why other lines were excluded
        
        **REMINDER**: Be extremely selective. If the output exceeds {max_output_lines} lines, you FAILED the task.
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Review this slice and identify essential statements.\nDiff:\n{diff_text}")
        ]
        
        # Agent Loop
        max_steps = 10
        curr_step = 0
        while curr_step < max_steps:
            # Wrap LLM call with retry mechanism
            try:
                response = retry_on_connection_error(
                    lambda: llm_with_tools.invoke(messages),
                    max_retries=3
                )
            except Exception as e:
                print(f"[DistillationAgent] LLM invocation failed after retries: {e}")
                return None
            
            messages.append(response)
            
            if response.tool_calls:
                for t in response.tool_calls:
                    print(f"      [DistillationAgent] Calling Tool: {t['name']}")
                    tool_result = "Error"
                    try:
                        selected_tool = next((x for x in tools if x.name == t['name']), None)
                        if selected_tool:
                            tool_result = selected_tool.invoke(t['args'])
                        else:
                            tool_result = f"Tool {t['name']} not found."
                    except Exception as e:
                        tool_result = f"Error: {e}"
                    messages.append(ToolMessage(content=str(tool_result), tool_call_id=t['id']))
                curr_step += 1
            else:
                break
        
        # Final Extraction with retry
        final_extractor = self.llm.with_structured_output(SliceValidationResult)
        try:
            return retry_on_connection_error(
                lambda: final_extractor.invoke(messages),
                max_retries=3
            )
        except Exception as e:
            print(f"[DistillationAgent] Extraction Failed after retries: {e}")
            return None

# ==============================================================================
# 工具函数
# ==============================================================================

def extract_search_hints(diff_text: str) -> Dict[str, Any]:
    """
    Extract search hints from diff (Search Hints Extraction)
    
    This is the first step of Anchor Discovery: use static analysis to extract modified lines
    and key variables as starting points for Agent exploration.
    
    Args:
        diff_text: Patch text in unified diff format
        
    Returns:
        Dict containing:
        - deleted_lines: List[ModifiedLine] - Lines deleted (in pre-patch)
          These lines may be vulnerability code itself, important anchor candidates
        - added_lines: List[ModifiedLine] - Lines added (in post-patch)
          These lines reveal fix intent, help infer missing defenses
        - key_variables: Set[str] - Variable names extracted from all modified lines
          Tell Agent which variables to trace (for trace_variable)
    """
    deleted_lines = []
    added_lines = []
    all_variables = set()
    
    if not diff_text:
        return {
            "deleted_lines": deleted_lines,
            "added_lines": added_lines,
            "key_variables": all_variables
        }
    
    # 匹配 hunk header: @@ -10,5 +10,6 @@
    hunk_re = re.compile(r'^@@\s*-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@')
    
    # 当前行号计数器
    c_old = 0  # pre-patch line counter
    c_new = 0  # post-patch line counter
    
    # Helper function: extract variable names
    def extract_variables(code: str) -> Set[str]:
        """
        Extract variable names from code (support struct member access)
        Match: var, var->field, var.field, ptr->field->subfield
        """
        vars = re.findall(
            r'\b([a-zA-Z_][a-zA-Z0-9_]*(?:(?:->|\.)[a-zA-Z_][a-zA-Z0-9_]*)*)\b',
            code
        )
        return set(vars)
    
    for line in diff_text.splitlines():
        # 跳过文件头
        if line.startswith('---') or line.startswith('+++'):
            continue
        
        # 解析 hunk header
        m = hunk_re.match(line)
        if m:
            c_old = int(m.group(1))
            c_new = int(m.group(3))
            continue
        
        # 处理删除行 (-): 在 pre-patch 中存在，在 post-patch 中被删除
        if line.startswith('-') and not line.startswith('---'):
            raw_content = line[1:]  # 保留原始格式（包括缩进）
            stripped = raw_content.strip()
            
            # 跳过空行和注释
            if stripped and not stripped.startswith(('//', '/*', '*', '#')):
                extracted_vars = extract_variables(stripped)
                
                deleted_lines.append(ModifiedLine(
                    line_number=c_old,
                    content=stripped,
                    raw_content=raw_content,
                    extracted_vars=extracted_vars
                ))
                all_variables.update(extracted_vars)
            
            c_old += 1
        
        # 处理新增行 (+): 在 post-patch 中新增
        elif line.startswith('+') and not line.startswith('+++'):
            raw_content = line[1:]
            stripped = raw_content.strip()
            
            # 跳过空行和注释
            if stripped and not stripped.startswith(('//', '/*', '*', '#')):
                extracted_vars = extract_variables(stripped)
                
                added_lines.append(ModifiedLine(
                    line_number=c_new,
                    content=stripped,
                    raw_content=raw_content,
                    extracted_vars=extracted_vars
                ))
                all_variables.update(extracted_vars)
            
            c_new += 1
        
        # 处理上下文行（两边都有）
        else:
            c_old += 1
            c_new += 1
    
    # 清理变量列表（移除常见的关键字和函数名）
    keywords = {
        'if', 'else', 'while', 'for', 'return', 'break', 'continue',
        'sizeof', 'NULL', 'true', 'false', 'void', 'int', 'char', 'long',
        'static', 'const', 'struct', 'enum', 'typedef', 'unsigned', 'signed',
        'short', 'double', 'float', 'extern', 'volatile', 'register', 'auto'
    }
    all_variables = {
        v for v in all_variables
        if v not in keywords and len(v) > 1
    }
    
    return {
        "deleted_lines": deleted_lines,
        "added_lines": added_lines,
        "key_variables": all_variables
    }


def extract_deleted_lines(diff_text: str) -> List[str]:
    """
    Extract deleted lines from diff (exist in Pre-Patch, deleted in Post-Patch)
    
    Since Primary is fixed as Pre-Patch (OLD), deleted lines are those starting with '-'
    These lines need to be injected into Shadow (Post-Patch) slice to display vulnerability code
    """
    deleted = []
    if not diff_text:
        return deleted
    
    for line in diff_text.splitlines():
        # 在 Pre-Patch 中存在但在 Post-Patch 中被删除的行（'-' 开头）
        if line.startswith('-') and not line.startswith('---'):
            content = line[1:].strip()
            # 过滤注释和空行
            if content and not content.startswith(('//','/*', '*', '#')):
                deleted.append(content)
    return deleted

# ==============================================================================
# 切片生成函数（从 Anchors 生成切片）
# ==============================================================================

def generate_slice_from_anchors(
    anchor_result: AnchorResult,
    patch: AtomicPatch,
    taxonomy: TaxonomyFeature,
    slicer_pri,
    pdg_pri: nx.MultiDiGraph,
    sl_pri: int,
    func_name: str
) -> Tuple[Set[str], str]:
    """
    Generate program slice from Anchor results (without distillation)
    
    This function is decoupled for use in retry loops
    
    Args:
        anchor_result: Anchor identification result
        patch: Atomic patch information
        taxonomy: Vulnerability taxonomy
        slicer_pri: Primary Slicer instance
        pdg_pri: Primary PDG
        sl_pri: Primary start line
        func_name: Function name
        
    Returns:
        (final_nodes, raw_code): Slice node set and raw code
    """
    import re
    
    # Convert Anchors to Instructions
    instructions = []
    for origin_item in anchor_result.origin_anchors:
        instructions.append(SlicingInstruction(
            function_name=func_name,
            target_version="OLD",
            line_number=origin_item.line,
            code_content=origin_item.content,
            strategy="forward",
            description=f"Origin Anchor ({origin_item.role.value})"
        ))
    
    for impact_item in anchor_result.impact_anchors:
        instructions.append(SlicingInstruction(
            function_name=func_name,
            target_version="OLD",
            line_number=impact_item.line,
            code_content=impact_item.content,
            strategy="backward",
            description=f"Impact Anchor ({impact_item.role.value})"
        ))
    
    # Helper function: extract start nodes and variables
    def _extract_start(instr_list):
        _nodes = []
        _vars = set()
        for _i in instr_list:
            _found = slicer_pri.get_nodes_by_location(_i.line_number, _i.code_content)
            _nodes.extend(_found)
            if _i.focus_variable: _vars.add(_i.focus_variable)
        return _nodes, _vars

    origins_instrs = [i for i in instructions if i.strategy == 'forward']
    impacts_instrs = [i for i in instructions if i.strategy == 'backward']
    
    # Execute slicing
    fwd_nodes = set()
    if origins_instrs:
        ns, vs = _extract_start(origins_instrs)
        fwd_nodes = slicer_pri.forward_slice_pruned(ns, vs)

    bwd_nodes = set()
    if impacts_instrs:
        ns, vs = _extract_start(impacts_instrs)
        bwd_nodes = slicer_pri.backward_slice_pruned(ns, vs)
    
    union_nodes = fwd_nodes.union(bwd_nodes)
    
    final_nodes = set()
    if origins_instrs and impacts_instrs:
        src_lines = [i.line_number for i in origins_instrs]
        sink_lines = [i.line_number for i in impacts_instrs]
        
        min_limit = min(min(src_lines), min(sink_lines))
        max_limit = max(max(src_lines), max(sink_lines))
        
        # Extract diff line numbers and expand range
        patch_diff = patch.clean_diff if patch.clean_diff else patch.raw_diff
        diff_lines_abs = []
        
        if patch_diff:
            hunk_re = re.compile(r'^@@\s*-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@')
            c_old = 0
            c_new = 0
            
            for line in patch_diff.splitlines():
                if line.startswith('---') or line.startswith('+++'): continue
                
                m = hunk_re.match(line)
                if m:
                    try:
                        c_old = int(m.group(1))
                        c_new = int(m.group(3))
                    except (ValueError, IndexError):
                        pass
                    continue
                
                if line.startswith('-'):
                    content = line[1:].strip()
                    if content and not content.startswith(('/', '*')):
                        diff_lines_abs.append(c_old)
                    c_old += 1
                elif line.startswith('+'):
                    c_new += 1
                else:
                    c_old += 1
                    c_new += 1
        
        # Expand range to include all diff lines
        for dln in diff_lines_abs:
            if dln < min_limit:
                min_limit = dln
            if dln > max_limit:
                max_limit = dln
        
        # Filter nodes
        for nid in union_nodes:
            if nid not in pdg_pri.nodes: continue
            n_start_rel = pdg_pri.nodes[nid].get('start_line', 0)
            n_start_abs = n_start_rel - 1 + sl_pri
            
            if min_limit <= n_start_abs <= max_limit:
                final_nodes.add(nid)
        
        # Inject diff lines
        if diff_lines_abs:
            for d_abs in diff_lines_abs:
                if min_limit <= d_abs <= max_limit:
                    d_rel = d_abs - sl_pri + 1
                    if d_rel > 0:
                        final_nodes.update(slicer_pri.get_nodes_by_location(d_rel, None))
    else:
        final_nodes = union_nodes
    
    # Helper for strict node extraction
    def _get_strict_nodes(instr_list):
        nodes = set()
        for i in instr_list:
            if i.line_number is not None and i.line_number > 0:
                found = slicer_pri.get_nodes_by_location(i.line_number, None)
                nodes.update(found)
        return nodes
    
    critical_nodes = set()
    if origins_instrs:
        critical_nodes.update(_get_strict_nodes(origins_instrs))
    if impacts_instrs:
        critical_nodes.update(_get_strict_nodes(impacts_instrs))
    
    final_nodes.update(critical_nodes)
    
    # [FIX] Force add ENTRY node if it's an Origin/Impact anchor
    # This handles cases where function definition itself is the vulnerability origin
    for nid, data in pdg_pri.nodes(data=True):
        if data.get('type') == 'ENTRY':
            entry_line_rel = data.get('start_line', 0)
            # ENTRY nodes may have start_line=0, treat as line 1 (first line of function)
            if entry_line_rel == 0:
                entry_line_rel = 1
            entry_line_abs = entry_line_rel - 1 + sl_pri
            print(f"      [Debug] ENTRY node: rel={entry_line_rel}, abs={entry_line_abs}, sl_pri={sl_pri}")
            # Check if ENTRY line matches any anchor
            is_anchor = False
            for a in anchor_result.origin_anchors + anchor_result.impact_anchors:
                print(f"      [Debug] Comparing ENTRY line {entry_line_abs} with anchor line {a.line}")
                if entry_line_abs == a.line:
                    is_anchor = True
                    break
            if is_anchor:
                final_nodes.add(nid)
                print(f"      [Slicing] Including ENTRY node at line {entry_line_abs} (matches anchor)")
    
    # Generate raw slice code
    raw_code = slicer_pri.to_code(final_nodes)
    
    return final_nodes, raw_code


# ==============================================================================
# 主流程入口函数
# ==============================================================================

def slicing_node(state: PatchExtractionState) -> Dict:
    """
    Program Slicing Main Workflow (Phase 2 core node, based on Methodology §3.3)
    
    Workflow:
    §3.3.2 Anchor-Guided Signature Extraction:
        1. Anchor Discovery (Agent-driven): Identify Origin and Impact anchors
        2. Retry Loop (max 3 attempts):
           a. Identify Anchors
           b. Validate Anchor Completeness (lightweight check)
           c. Generate Slice (apply data-flow pruning: forward ∪ backward with diff injection)
           d. Validate Slice Quality (semantic adequacy check)
        3. Distillation (Agent-driven): Semantic filtering to remove noise
        4. Shadow Slice Generation: Mapping + inject deleted lines
        5. Extract Origin/Impact line numbers for matching
    
    Args:
        state: State containing patches, taxonomy, commit info
        
    Returns:
        Dict containing slices and taxonomy
    """
    taxonomy = state['taxonomy']
    patches = state['patches']
    commit_msg = state['commit_message']
    repo_path = state.get('repo_path')
    commit_hash = state.get('commit_hash')

    print(f'[DEBUG] slicing_node received {len(patches)} patches:')
    for i, p in enumerate(patches):
        print(f'  [{i}] {p.file_path}::{p.function_name}')
    print(f'Taxonomy:\n{taxonomy}\n')

    validator = SliceValidator()
    generated_slices: Dict[str, SliceFeature] = {}
    
    for patch in patches:
        func = patch.function_name
        lang = 'c' if patch.file_path.endswith(('.c', '.h')) else 'cpp'
        
        # ===== Step 0: Initialization =====
        # Primary View: Pre-Patch (OLD/Vulnerable) - contains full vulnerability trigger path
        # Shadow View: Post-Patch (NEW/Fixed) - generated via mapping for fix comparison
        # Navigator: Points to Pre-Patch version for code analysis
        current_version_hash = f"{commit_hash}^"
        navigator = CodeNavigator(repo_path, target_version=current_version_hash)
        
        print(f"    [Slicing] Processing function: {func}")
        
        # Prepare code and PDG: Primary = Pre-Patch (OLD), Shadow = Post-Patch (NEW)
        code_pri = patch.old_code
        code_shadow = patch.new_code
        patch_diff = patch.clean_diff if patch.clean_diff else patch.raw_diff
        
        # ===== Improvement 1: Skip functions with empty old_code (ADDED functions) =====
        if not code_pri or code_pri.strip() == "":
            print(f"      [Skip] Function {func} has empty old_code (likely ADDED), skipping...")
            continue
        
        try:
            # 使用 AtomicPatch 的实际起始行号
            sl_pri = patch.start_line_old
            sl_shadow = patch.start_line_new
            
            # Ensure valid integers as fallback
            sl_pri = sl_pri if sl_pri is not None else 1
            sl_shadow = sl_shadow if sl_shadow is not None else 1
            
            pdg_pri = PDGBuilder(code_pri, lang=lang).build(target_line=sl_pri)
            pdg_shadow = PDGBuilder(code_shadow, lang=lang).build(target_line=sl_shadow) 
        except Exception as e:
            print(f"      [Error] PDG Build failed: {e}")
            continue
        
        slicer_pri = Slicer(pdg_pri, code_pri, start_line=sl_pri)
        
        # ===== §3.3.2.1: Anchor Discovery (Agent-driven) =====
        
        # Extract search hints
        search_hints = extract_search_hints(patch_diff)
        
        print(f"    [Anchor Discovery] Search Hints:")
        print(f"      - Deleted lines: {len(search_hints['deleted_lines'])}")
        print(f"      - Added lines: {len(search_hints['added_lines'])}")
        print(f"      - Key variables: {search_hints['key_variables']}")
        
        # ===== §3.3.2.2: Discovery-Validation-Refinement Loop =====
        anchor_analyzer = AnchorAnalyzer(navigator)
        max_attempts = 2
        anchor_result = None
        final_pri_nodes = None
        raw_code = None
        validation_passed = False
        
        for attempt in range(1, max_attempts + 1):
            print(f"    [Discovery & Slicing] Attempt {attempt}/{max_attempts}")
            
            # Sub-step 1: Identify Anchors
            print(f"      [Anchor Discovery] Identifying anchors...")
            anchor_result = anchor_analyzer.identify(
                code_content=code_pri,
                diff_text=patch_diff,
                search_hints=search_hints,
                taxonomy=taxonomy,
                file_path=patch.file_path,
                function_name=func,
                start_line=sl_pri,
                attempt=attempt
            )
            
            print(f"        - Origin: {[(a.line, a.role.value) for a in anchor_result.origin_anchors]}")
            print(f"        - Impact: {[(a.line, a.role.value) for a in anchor_result.impact_anchors]}")
            
            # ===== Improvement 2: Validate anchors exist in actual code =====
            print(f"      [Validation] Verifying anchor lines exist in code...")
            code_pri_lines = code_pri.splitlines()
            invalid_anchors = []
            
            for anchor_list, anchor_type in [(anchor_result.origin_anchors, "Origin"),
                                              (anchor_result.impact_anchors, "Impact")]:
                for anchor in anchor_list:
                    # Convert absolute line to relative (0-indexed)
                    rel_line = anchor.line - sl_pri
                    
                    # Check if line number is valid
                    if rel_line < 0 or rel_line >= len(code_pri_lines):
                        invalid_anchors.append(f"{anchor_type} line {anchor.line} out of range (code has {len(code_pri_lines)} lines, start={sl_pri})")
                        continue
                    
                    # Check if content matches (fuzzy match: normalized)
                    actual_content = code_pri_lines[rel_line].strip()
                    anchor_content = anchor.content.strip()
                    
                    # Normalize for comparison (remove extra whitespace)
                    actual_normalized = ' '.join(actual_content.split())
                    anchor_normalized = ' '.join(anchor_content.split())
                    
                    # Check if anchor content is substring of actual line (allows partial matches)
                    if anchor_normalized not in actual_normalized and actual_normalized not in anchor_normalized:
                        invalid_anchors.append(
                            f"{anchor_type} line {anchor.line}: content mismatch\n"
                            f"  Expected: {anchor_content}\n"
                            f"  Actual:   {actual_content}"
                        )
            
            if invalid_anchors:
                print(f"        ✗ Anchor validation failed:")
                for err in invalid_anchors:
                    print(f"          - {err}")
                
                if attempt < max_attempts:
                    print(f"      [Refinement] Stage {attempt}: Anchors don't match code, retrying...")
                    continue
                else:
                    print(f"      [Refinement] Maximum attempts reached, skipping function {func}")
                    break
            
            print(f"        ✓ All anchors verified in code")
            
            # Sub-step 2: Validate Anchor Completeness (lightweight check)
            print(f"      [Validation] Checking anchor completeness...")
            validation = validate_anchor_completeness(anchor_result=anchor_result)
            
            if not validation["is_valid"]:
                print(f"        ✗ Failed: {validation['reason']}")
                
                if attempt < max_attempts:
                    print(f"      [Refinement] Stage {attempt}: Re-discovery (trying alternative anchors)...")
                    continue
                else:
                    print(f"      [Refinement] Maximum attempts reached.")
                    break
            
            print(f"        ✓ Anchor completeness passed")
            
            # Sub-step 3: Generate Slice
            print(f"      [Slicing] Generating slice from anchors...")
            try:
                final_pri_nodes, raw_code = generate_slice_from_anchors(
                    anchor_result=anchor_result,
                    patch=patch,
                    taxonomy=taxonomy,
                    slicer_pri=slicer_pri,
                    pdg_pri=pdg_pri,
                    sl_pri=sl_pri,
                    func_name=func
                )
                print(f"        Generated slice: {len(raw_code.splitlines())} lines")
            except Exception as e:
                print(f"        ✗ Slice generation failed: {e}")
                if attempt < max_attempts:
                    continue
                else:
                    break
            
            # ===== Improvement 2b: Validate slice lines exist in actual code =====
            print(f"      [Validation] Verifying slice lines exist in code...")
            slice_validation_failed = False
            invalid_slice_lines = []
            
            for slice_line in raw_code.splitlines():
                # Extract line number from format "[1234] code content"
                import re
                m = re.match(r'^\[\s*(\d+)\]', slice_line.strip())
                if m:
                    abs_line = int(m.group(1))
                    rel_line = abs_line - sl_pri
                    
                    # Check if line is valid
                    if rel_line < 0 or rel_line >= len(code_pri_lines):
                        invalid_slice_lines.append(f"Line {abs_line} out of range")
                        slice_validation_failed = True
                        continue
                    
                    # Extract code content from slice
                    slice_content = slice_line.split(']', 1)[1].strip() if ']' in slice_line else ""
                    actual_content = code_pri_lines[rel_line].strip()
                    
                    # Normalize for comparison
                    slice_normalized = ' '.join(slice_content.split())
                    actual_normalized = ' '.join(actual_content.split())
                    
                    # Check if content matches (allow fuzzy match)
                    if slice_normalized and actual_normalized:
                        if slice_normalized != actual_normalized:
                            # Allow substring match for multi-line statements
                            if slice_normalized not in actual_normalized and actual_normalized not in slice_normalized:
                                invalid_slice_lines.append(
                                    f"Line {abs_line} content mismatch:\n"
                                    f"    Slice:  {slice_content}\n"
                                    f"    Actual: {actual_content}"
                                )
                                slice_validation_failed = True
            
            if slice_validation_failed:
                print(f"        ✗ Slice validation failed: {len(invalid_slice_lines)} mismatches")
                for err in invalid_slice_lines[:5]:  # Show first 5 errors
                    print(f"          - {err}")
                if len(invalid_slice_lines) > 5:
                    print(f"          ... and {len(invalid_slice_lines) - 5} more")
                
                if attempt < max_attempts:
                    print(f"      [Refinement] Slice contains invalid lines, retrying...")
                    continue
                else:
                    print(f"      [Refinement] Maximum attempts reached, proceeding with caution")
            else:
                print(f"        ✓ All slice lines verified in code")
            
            # Sub-step 4: Validate Slice Quality
            print(f"      [Validation] Checking slice quality...")
            slice_quality = validate_slice_quality(
                slice_code=raw_code,
                anchor_result=anchor_result,
                taxonomy=taxonomy,
                llm=validator.llm
            )
            
            if slice_quality["is_valid"]:
                print(f"        ✓ Slice quality passed")
                print(f"          - Anchor present: {slice_quality['anchor_present']}")
                print(f"          - Semantic adequate: {slice_quality['semantic_adequate']}")
                validation_passed = True
                break
            else:
                print(f"        ✗ Slice quality check failed: {slice_quality['reason']}")
                print(f"          - Anchor present: {slice_quality['anchor_present']}")
                print(f"          - Semantic adequate: {slice_quality['semantic_adequate']}")
                
                if attempt < max_attempts:
                    print(f"      [Refinement] Slice quality insufficient, retrying anchor discovery...")
                    continue
                else:
                    print(f"      [Refinement] Maximum attempts reached.")
                    break
        
        # Stage 3: Use last anchor result if validation failed
        if not validation_passed:
            print(f"    [Refinement] Stage 3: Using last anchor result (validation failed but keeping semantic anchors)")
            
            # 如果还没有生成切片，使用最后一次的 anchor_result 重新生成
            if final_pri_nodes is None or raw_code is None:
                if anchor_result is None or not anchor_result.origin_anchors or not anchor_result.impact_anchors:
                    print(f"        ✗ No valid anchor result available. Skipping function {func}")
                    continue
                
                print(f"      [Fallback] Using last anchor result:")
                print(f"        - Origin: {[(a.line, a.role.value) for a in anchor_result.origin_anchors]}")
                print(f"        - Impact: {[(a.line, a.role.value) for a in anchor_result.impact_anchors]}")
                
                print(f"      [Fallback] Generating slice from last anchors...")
                try:
                    final_pri_nodes, raw_code = generate_slice_from_anchors(
                        anchor_result=anchor_result,
                        patch=patch,
                        taxonomy=taxonomy,
                        slicer_pri=slicer_pri,
                        pdg_pri=pdg_pri,
                        sl_pri=sl_pri,
                        func_name=func
                    )
                    print(f"        Generated slice: {len(raw_code.splitlines())} lines")
                except Exception as e:
                    print(f"        ✗ Slice generation failed: {e}")
                    print(f"        Skipping function {func}")
                    continue
            else:
                print(f"      [Fallback] Slice already generated in last attempt, using it")
        
        # 检查是否成功生成切片
        if final_pri_nodes is None or raw_code is None:
            print(f"    [Error] Failed to generate slice for {func} after all attempts")
            continue
        
        print(f"    [Success] Final slice for {func}:")
        print(f"      - Origin Anchors: {[(a.line, a.role.value) for a in anchor_result.origin_anchors]}")
        print(f"      - Impact Anchors: {[(a.line, a.role.value) for a in anchor_result.impact_anchors]}")
        print(f"      - Raw slice: {len(raw_code.splitlines())} lines")
        print(f'\nRaw slice:\n{raw_code}\n')
        
        # Prepare instructions (for subsequent processing)
        origin_instrs = [SlicingInstruction(
            function_name=func,
            target_version="OLD",
            line_number=a.line,
            code_content=a.content,
            strategy="forward",
            description=f"Origin Anchor ({a.role.value})"
        ) for a in anchor_result.origin_anchors]
        
        impact_instrs = [SlicingInstruction(
            function_name=func,
            target_version="OLD",
            line_number=a.line,
            code_content=a.content,
            strategy="backward",
            description=f"Impact Anchor ({a.role.value})"
        ) for a in anchor_result.impact_anchors]
        
        # ===== §3.3.2.3: Distillation (Agent-driven semantic filtering) =====
        
        # Heuristically determine main focus variable (extract from search_hints)
        main_var = list(search_hints['key_variables'])[0] if search_hints.get('key_variables') else "N/A"

        print(f"    [Distillation] Applying semantic filtering...")
        validation_result = validator.distill_slice(
            slice_code=raw_code,
            diff_text=patch_diff,
            commit_message=commit_msg,
            func_name=func,
            focus_var=main_var,
            vuln_type=taxonomy.vuln_type,
            hypothesis=taxonomy,
            origins=anchor_result.origin_anchors,
            impacts=anchor_result.impact_anchors,
            cwe_info=f"{taxonomy.cwe_id}: {taxonomy.cwe_name}" if taxonomy.cwe_id else "Generic",
            navigator=navigator,
            file_path=patch.file_path
        )
        
        # Helper for strict node extraction (ignore content to prevent noise, trust line number)
        def _get_strict_nodes(instr_list):
            nodes = set()
            for i in instr_list:
                if i.line_number is not None and i.line_number > 0:
                    found = slicer_pri.get_nodes_by_location(i.line_number, None)
                    nodes.update(found)
            return nodes
            
        # Positive Selection Strategy with Size Constraint:
        # If validator returns results, check if output is reasonable.
        # If too large, fallback to anchor-only strategy.
        
        # Define thresholds
        input_line_count = len(raw_code.splitlines())
        max_output_lines = min(30 + (input_line_count // 50) * 5, 50)
        fallback_threshold = max_output_lines + 10  # Allow 10 lines tolerance
        
        if validation_result and validation_result.relevant_lines:
            kept_lines = set(validation_result.relevant_lines)
            kept_line_count = len(kept_lines)
            
            print(f"      [Validator] Agent returned {kept_line_count} lines out of {input_line_count} total lines.")
            
            # Check if output is too large
            if kept_line_count > fallback_threshold:
                print(f"      [Validator] ⚠️  Output too large ({kept_line_count} > {fallback_threshold}), falling back to anchor-only strategy.")
                print(f"      [Validator] Reason: Agent failed to be selective enough, using minimal anchor-based slice.")
                
                # Fallback: Keep ONLY anchor lines
                anchor_only_nodes = set()
                if origin_instrs:
                    anchor_only_nodes.update(_get_strict_nodes(origin_instrs))
                if impact_instrs:
                    anchor_only_nodes.update(_get_strict_nodes(impact_instrs))
                
                final_pri_nodes = anchor_only_nodes
                print(f"      [Validator] Fallback result: {len(final_pri_nodes)} anchor nodes kept.")
            else:
                print(f"      [Validator] ✓ Output size acceptable ({kept_line_count} ≤ {fallback_threshold}), applying distillation.")
                
                # Map Line Numbers -> Nodes
                final_kept_nodes = set()
                for ln in kept_lines:
                    found = slicer_pri.get_nodes_by_location(ln, None)
                    final_kept_nodes.update(found)
                    
                # [Protection] Critical Nodes Enforcement (Anchor lines must be present)
                if origin_instrs:
                    final_kept_nodes.update(_get_strict_nodes(origin_instrs))
                if impact_instrs:
                    final_kept_nodes.update(_get_strict_nodes(impact_instrs))
                    
                # Apply Filter: final_pri_nodes becomes the intersection of what we had & what is kept
                final_pri_nodes.intersection_update(final_kept_nodes)
        else:
            print(f"      [Validator] ⚠️  Agent returned no results or error, keeping original slice.")
        
        # [Enforcement] Force add Origin/Impact nodes to ensure they exist
        critical_nodes_force = set()
        if origin_instrs:
            critical_nodes_force.update(_get_strict_nodes(origin_instrs))
        if impact_instrs:
            critical_nodes_force.update(_get_strict_nodes(impact_instrs))
        
        # [FIX] Force add ENTRY node if it's an anchor (e.g., function definition as Origin)
        for nid, data in pdg_pri.nodes(data=True):
            if data.get('type') == 'ENTRY':
                entry_line_rel = data.get('start_line', 0)
                entry_line_abs = entry_line_rel - 1 + sl_pri if entry_line_rel > 0 else 0
                # Check if ENTRY line matches any anchor
                for a in anchor_result.origin_anchors + anchor_result.impact_anchors:
                    if entry_line_abs == a.line:
                        critical_nodes_force.add(nid)
                        print(f"      [Slicing] Force-added ENTRY node at line {entry_line_abs} (anchor)")
                        break
        
        if critical_nodes_force:
            added_count = len(critical_nodes_force - final_pri_nodes)
            if added_count > 0:
                print(f"      [Slicing] Force-added {added_count} missing critical nodes (Origin/Impact).")
            final_pri_nodes.update(critical_nodes_force)
       
       # ===== §3.3.2.4: Shadow Slice Generation (Mapping + Injection) =====
       # Generate distilled Primary code
        s_pri_text = slicer_pri.to_code(final_pri_nodes)
        
        # [FIX] Fallback: Add anchor lines that are missing from slice
        # This should rarely trigger now that ENTRY nodes are properly included
        anchor_lines_outside_pdg = []
        for a in anchor_result.origin_anchors + anchor_result.impact_anchors:
            # Check if the line is not in slice
            if not any(f"[{a.line:4d}]" in line for line in s_pri_text.splitlines()):
                anchor_lines_outside_pdg.append((a.line, a.content))
                print(f"      [Warning] Anchor line {a.line} missing from slice (PDG issue?), adding manually")
        
        if anchor_lines_outside_pdg:
            # Prepend these lines to the slice (they're typically function signatures)
            extra_lines = [f"[{ln:4d}] {content}" for ln, content in sorted(anchor_lines_outside_pdg)]
            s_pri_text = "\n".join(extra_lines + [s_pri_text]) if s_pri_text else "\n".join(extra_lines)
        
        print(f'Clean Primary Slice:\n{s_pri_text}\n')

        # Extract deleted lines (exist in OLD, deleted in NEW) for injection into Shadow
        deleted_lines_content = extract_deleted_lines(patch_diff)
        if deleted_lines_content:
            print(f"      [Slicing] Injecting {len(deleted_lines_content)} deleted lines into Shadow Slice...")

        # [FIX] Create mapper BEFORE using it (with diff_text for proper line classification)
        mapper = ShadowMapper(code_pri, code_shadow, pdg_pri, pdg_shadow, sl_pri, sl_shadow, diff_text=patch_diff)
        
        # Generate Shadow slice via mapping and injection
        shadow_nodes, mandatory_shadow_nodes = ShadowMapper.map_and_slice(
            final_pri_nodes,
            pdg_pri,
            pdg_shadow,
            code_pri,
            code_shadow,
            start_line_pri=sl_pri,
            start_line_shadow=sl_shadow,
            deleted_lines=deleted_lines_content,
            diff_text=patch_diff
        )
        # Generate Shadow slice code (already cleaned by ShadowMapper)
        slicer_shadow = Slicer(pdg_shadow, code_shadow, start_line=sl_shadow)
        s_shadow_text = slicer_shadow.to_code(shadow_nodes)
        
        # [FIX] Alignment check: Remove common lines from shadow if not present in primary
        # This ensures both slices have consistent representation of unchanged code
        import re
        
        # Extract line numbers present in primary slice
        pri_lines_present = set()
        for line in s_pri_text.splitlines():
            m = re.match(r'^\[\s*(\d+)\]', line.strip())
            if m:
                pri_lines_present.add(int(m.group(1)))
        
        # Filter shadow slice to remove common lines not in primary
        shadow_lines_filtered = []
        for line in s_shadow_text.splitlines():
            m = re.match(r'^\[\s*(\d+)\]', line.strip())
            if m:
                shadow_abs = int(m.group(1))
                shadow_rel = shadow_abs - sl_shadow + 1
                
                # Check if this is a common line (EXACT/WHITESPACE mapping)
                if shadow_rel in mapper.shadow_to_pri:
                    pri_rel = mapper.shadow_to_pri[shadow_rel]
                    pri_abs = pri_rel - 1 + sl_pri
                    
                    # Only keep if corresponding primary line is present in primary slice
                    if pri_abs in pri_lines_present:
                        shadow_lines_filtered.append(line)
                    else:
                        print(f"      [Alignment] Removing shadow line {shadow_abs} (primary line {pri_abs} not in slice)")
                else:
                    # Not a common line (MOVED/MODIFIED/UNIQUE), keep it
                    shadow_lines_filtered.append(line)
            else:
                # Non-code line, keep it
                shadow_lines_filtered.append(line)
        
        s_shadow_text = "\n".join(shadow_lines_filtered)
        
        # [FIX] Add corresponding shadow anchor lines that are outside PDG range
        # Map primary anchor lines to shadow
        shadow_anchor_lines_outside_pdg = []
        if anchor_lines_outside_pdg:
            for pri_ln, pri_content in anchor_lines_outside_pdg:
                # Map this primary line to shadow
                shadow_mapped_lines = mapper.map_lines_to_shadow([pri_ln])
                if shadow_mapped_lines:
                    for s_abs in shadow_mapped_lines:
                        # Check if this line is already in shadow slice
                        if not any(f"[{s_abs:4d}]" in line for line in s_shadow_text.splitlines()):
                            # Get shadow content
                            s_rel = s_abs - sl_shadow + 1
                            if 0 < s_rel <= len(code_shadow.splitlines()):
                                s_content = code_shadow.splitlines()[s_rel - 1]
                                shadow_anchor_lines_outside_pdg.append((s_abs, s_content))
                else:
                    # No mapping found (e.g., function signature unchanged)
                    # Use same content as primary
                    # Map line number: check if shadow has same relative position
                    s_rel_guess = pri_ln - sl_pri + 1  # Relative position in primary
                    if s_rel_guess > 0:
                        s_abs_guess = s_rel_guess - 1 + sl_shadow
                        if 0 < (s_abs_guess - sl_shadow + 1) <= len(code_shadow.splitlines()):
                            s_content = code_shadow.splitlines()[s_abs_guess - sl_shadow]
                            if not any(f"[{s_abs_guess:4d}]" in line for line in s_shadow_text.splitlines()):
                                shadow_anchor_lines_outside_pdg.append((s_abs_guess, s_content))
        
        if shadow_anchor_lines_outside_pdg:
            print(f"      [Mapping] Adding {len(shadow_anchor_lines_outside_pdg)} shadow anchor lines missing from PDG range...")
            extra_shadow_lines = [f"[{ln:4d}] {content}" for ln, content in sorted(shadow_anchor_lines_outside_pdg)]
            s_shadow_text = "\n".join(extra_shadow_lines + [s_shadow_text]) if s_shadow_text else "\n".join(extra_shadow_lines)
        
        # Build mapping for anchor mapping (needed for Origin/Impact extraction)
        # [FIX] Mapper already created above, reuse it
        common_pri_to_shadow = mapper.pri_to_shadow
        
        # Build replace_map_pri_to_shadow from difflib opcodes
        replace_map_pri_to_shadow = []
        matcher = difflib.SequenceMatcher(None, mapper.lines_pri, mapper.lines_shadow)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace':
                replace_map_pri_to_shadow.append(((i1 + 1, i2), (j1 + 1, j2)))
        
        # Remove unmodified function header (avoid function name differences affecting matching)
        def strip_header_if_unmodified(slice_text: str, diff_text: str, nodes: Set[str], pdg: nx.MultiDiGraph, anchor_lines: Set[int] = None) -> str:
            """
            Strip unmodified function header to avoid function name differences affecting matching.
            
            Args:
                slice_text: The slice code text
                diff_text: The patch diff
                nodes: PDG nodes in the slice
                pdg: The PDG
                anchor_lines: Anchor line numbers (absolute) to protect from stripping
                
            Returns:
                Slice text with header potentially stripped
            """
            lines = slice_text.splitlines()
            if not lines: return slice_text
            
            # 1. Check if the first line corresponds to an ENTRY node
            min_line = float('inf')
            min_line_abs = None  # Absolute line number
            is_entry = False
            
            for nid in nodes:
                if nid not in pdg: continue
                d = pdg.nodes[nid]
                if d.get('type') in ('EXIT', 'MERGE', 'NO_OP'): continue
                
                line = d.get('start_line', 0)
                if line > 0:
                    if line < min_line:
                        min_line = line
                        is_entry = (d.get('type') == 'ENTRY')
                        # Extract absolute line from first slice line
                        if lines:
                            m = re.match(r'^\[\s*(\d+)\]', lines[0])
                            if m:
                                min_line_abs = int(m.group(1))
                    elif line == min_line:
                        if d.get('type') == 'ENTRY':
                            is_entry = True
            
            if not is_entry:
                return slice_text

            # 2. [FIX] Check if this line is an anchor - if so, don't strip it
            if anchor_lines and min_line_abs and min_line_abs in anchor_lines:
                # print(f"      [Stripper] Protecting anchor line {min_line_abs} from stripping")
                return slice_text

            # 3. Proceed with stripping logic
            first_line = lines[0]
            if "]" in first_line:
                code_content = first_line.split("]", 1)[1].strip()
                
                # 检查 diff 中是否有此行的修改
                is_modified = False
                clean_code = code_content.replace(" ", "")
                
                for dline in diff_text.splitlines():
                    if dline.startswith(('+', '-')) and not dline.startswith(('+++', '---')):
                        clean_diff = dline[1:].strip().replace(" ", "")
                        if clean_code in clean_diff:
                            is_modified = True
                            break
                
                if not is_modified:
                    # print(f"      [Stripper] Removing unmodified header: {code_content}")
                    return "\n".join(lines[1:])
            return slice_text

        # Collect anchor lines (absolute) to protect from stripping
        anchor_lines_abs = set()
        for a in anchor_result.origin_anchors:
            anchor_lines_abs.add(a.line)
        for a in anchor_result.impact_anchors:
            anchor_lines_abs.add(a.line)
        
        s_pri_text = strip_header_if_unmodified(s_pri_text, patch_diff, final_pri_nodes, pdg_pri, anchor_lines_abs)
        
        # Map anchor lines to shadow for shadow protection
        shadow_anchor_lines_abs = set()
        for pri_abs in anchor_lines_abs:
            shadow_mapped = mapper.map_lines_to_shadow([pri_abs])
            shadow_anchor_lines_abs.update(shadow_mapped)
        
        s_shadow_text = strip_header_if_unmodified(s_shadow_text, patch_diff, shadow_nodes, pdg_shadow, shadow_anchor_lines_abs)

        # ===== §3.3.2.5: Result Encapsulation =====
        header = f"// Function: {func}"
        final_s_pre = f"{header} (Pre-Patch/Vuln)\n{s_pri_text}"
        final_s_post = f"{header} (Post-Patch/Fixed)\n{s_shadow_text}"

        print(f'    [Result] {func}: Pre={len(s_pri_text.splitlines())} lines, Post={len(s_shadow_text.splitlines())} lines\n')
        print(f'----- Pre-Patch Slice -----\n{final_s_pre}\n')
        print(f'----- Post-Patch Slice -----\n{final_s_post}\n')
        
        def get_lines_by_abs_msg(text_block, target_abs_lines):
            found_list = []
            if not target_abs_lines: return found_list
            t_set = set(target_abs_lines)
            for line in text_block.splitlines():
                if not line.strip(): continue
                # Extract [ 123]
                m = re.match(r'^\[\s*(\d+)\]', line.strip())
                if m:
                    ln = int(m.group(1))
                    if ln in t_set:
                        found_list.append(line.strip())
            return found_list
        
        # [FIX] Expand anchor lines to include full PDG nodes (multi-line statements)
        # REVISED: More conservative expansion to prevent anchor contamination
        def expand_anchor_lines(anchor_list, pdg, sl, code_lines):
            """
            Expand anchor lines to include complete PDG nodes (start_line to end_line).
            
            IMPORTANT: This function is now more conservative:
            1. Only expands if the PDG node STARTS at the anchor line (not just covers it)
            2. Avoids expanding control structures (if/for/while) which would include unrelated lines
            3. Only uses source code analysis for function calls spanning multiple lines
            """
            expanded_abs = set()
            
            for a in anchor_list:
                a_rel = a.line - sl + 1
                
                # [FIX] Only find nodes that START at this line, not nodes that merely cover it
                # This prevents including entire control structures when anchor is inside a loop
                found_nodes = [nid for nid, d in pdg.nodes(data=True)
                              if d.get('start_line', 0) == a_rel]
                
                node_expanded = False
                if found_nodes:
                    # Find the best matching node (prefer STATEMENT over PREDICATE)
                    best_node = None
                    for nid in found_nodes:
                        d = pdg.nodes[nid]
                        node_type = d.get('type', '')
                        # Skip control structure nodes (PREDICATE for if/while/for conditions)
                        # These would expand to include the entire block
                        if node_type in ('PREDICATE', 'SWITCH_HEAD', 'CASE_LABEL', 'DEFAULT_LABEL'):
                            # For predicates, only keep the single line (the condition)
                            expanded_abs.add(a.line)
                            node_expanded = True
                            continue
                        # Prefer STATEMENT nodes
                        if node_type == 'STATEMENT':
                            best_node = nid
                            break
                        if best_node is None:
                            best_node = nid
                    
                    if best_node and not node_expanded:
                        d = pdg.nodes[best_node]
                        n_start = d.get('start_line', 0)
                        n_end = d.get('end_line', n_start)
                        
                        # Only expand if it's a genuine multi-line statement
                        # and not a control structure
                        if n_start > 0:
                            if n_end > n_start:
                                # Multi-line statement: include all lines
                                for ln_rel in range(n_start, n_end + 1):
                                    abs_ln = ln_rel - 1 + sl
                                    expanded_abs.add(abs_ln)
                                node_expanded = True
                            else:
                                # Single line: just add it
                                expanded_abs.add(a.line)
                                node_expanded = True
                
                # If no PDG node found, just add the anchor line directly
                # [FIX] Removed aggressive parenthesis-based expansion which caused contamination
                if not node_expanded:
                    expanded_abs.add(a.line)
            
            return sorted(list(expanded_abs))
        
        # Get code lines for source code analysis
        pri_code_lines = code_pri.splitlines()
        
        pri_origin_abs = expand_anchor_lines(anchor_result.origin_anchors, pdg_pri, sl_pri, pri_code_lines)
        pri_impact_abs = expand_anchor_lines(anchor_result.impact_anchors, pdg_pri, sl_pri, pri_code_lines)

        # Map to Shadow Absolute (Exact + Modified Block Heuristic)
        # [FIX] Ensure these are always lists, never None
        shadow_origin_abs = [] if pri_origin_abs else []
        shadow_impact_abs = [] if pri_impact_abs else []
        
        # Prepare for content matching
        all_pri_lines = code_pri.splitlines() if code_pri else []
        all_shadow_lines = code_shadow.splitlines() if code_shadow else []

        def map_to_shadow(pri_lines_abs, dest_list):
            """Map primary absolute lines to shadow absolute lines
            
            Note: dest_list must be a list (not None) before calling this function.
            This function modifies dest_list in place by appending to it.
            """
            if not pri_lines_abs:
                return
            if not isinstance(dest_list, list):
                print(f"      [Error] dest_list is not a list in map_to_shadow: {type(dest_list)}")
                return
            for pl in pri_lines_abs:
                rel_p = pl - sl_pri + 1
                
                candidates_rel = [] # Correctly initialized per origin line iteration

                if rel_p in common_pri_to_shadow:
                    # Case 1: Direct Match
                    candidates_rel.append(common_pri_to_shadow[rel_p])
                else:
                    # Case 2: Check replace blocks with Content Similarity
                    for (p_start, p_end), (s_start, s_end) in replace_map_pri_to_shadow:
                         if p_start <= rel_p <= p_end:
                             # A. Get Primary Content (for the line we are trying to map)
                             p_idx = rel_p - 1
                             p_content = all_pri_lines[p_idx].strip() if 0 <= p_idx < len(all_pri_lines) else ""
                             
                             # B. Strategy: Exact match or context-based match (上下文匹配)
                             # Check for exact content match first
                             exact_match = None
                             for s_rel in range(s_start, s_end + 1):
                                 s_idx = s_rel - 1
                                 if 0 <= s_idx < len(all_shadow_lines):
                                     s_content = all_shadow_lines[s_idx].strip()
                                     if p_content == s_content:
                                         exact_match = s_rel
                                         break
                             
                             if exact_match:
                                 # Found exact content match
                                 candidates_rel.append(exact_match)
                             else:
                                 # Try context match: check if surrounding lines match
                                 # Get context (上下两行)
                                 p_prev = all_pri_lines[p_idx - 1].strip() if p_idx > 0 else ""
                                 p_next = all_pri_lines[p_idx + 1].strip() if p_idx + 1 < len(all_pri_lines) else ""
                                 
                                 context_match = None
                                 for s_rel in range(s_start, s_end + 1):
                                     s_idx = s_rel - 1
                                     if 0 <= s_idx < len(all_shadow_lines):
                                         s_prev = all_shadow_lines[s_idx - 1].strip() if s_idx > 0 else ""
                                         s_next = all_shadow_lines[s_idx + 1].strip() if s_idx + 1 < len(all_shadow_lines) else ""
                                         
                                         # Check if both prev and next lines match
                                         if p_prev and p_next and p_prev == s_prev and p_next == s_next:
                                             context_match = s_rel
                                             break
                                 
                                 if context_match:
                                     candidates_rel.append(context_match)
                                 else:
                                     # No exact or context match: include whole block for agent to decide
                                     for s_l in range(s_start, s_end + 1):
                                         candidates_rel.append(s_l)
                             break
                
                # Expand candidates using PDG structure (Node Expansion)
                final_rel_lines = set()
                if candidates_rel:
                    for c_rel in candidates_rel:
                        # Try to find PDG nodes covering this line
                        found_nodes = slicer_shadow.get_nodes_by_location(c_rel, None)
                        if found_nodes:
                            for nid in found_nodes:
                                if nid in pdg_shadow.nodes:
                                    d = pdg_shadow.nodes[nid]
                                    n_start = d.get('start_line', 0)
                                    n_end = d.get('end_line', n_start)
                                    if n_start > 0:
                                        for ln in range(n_start, n_end + 1):
                                            final_rel_lines.add(ln)
                        else:
                            # No node found (e.g. comment, brace), keep line as is
                            final_rel_lines.add(c_rel)

                for fr in sorted(list(final_rel_lines)):
                    dest_list.append(fr - 1 + sl_shadow)
        
        map_to_shadow(pri_origin_abs, shadow_origin_abs)
        map_to_shadow(pri_impact_abs, shadow_impact_abs)

        # Extract content from CLEANED text (s_pri_text, s_shadow_text)
        # Note: Do NOT use final_s_pre/post as they have headers
        pri_origin_txt = get_lines_by_abs_msg(s_pri_text, pri_origin_abs) if pri_origin_abs else []
        pri_impact_txt = get_lines_by_abs_msg(s_pri_text, pri_impact_abs) if pri_impact_abs else []
        shadow_origin_txt = get_lines_by_abs_msg(s_shadow_text, shadow_origin_abs) if shadow_origin_abs else []
        shadow_impact_txt = get_lines_by_abs_msg(s_shadow_text, shadow_impact_abs) if shadow_impact_abs else []
        
        # Ensure all are lists (defensive)
        pri_origin_txt = pri_origin_txt if isinstance(pri_origin_txt, list) else []
        pri_impact_txt = pri_impact_txt if isinstance(pri_impact_txt, list) else []
        shadow_origin_txt = shadow_origin_txt if isinstance(shadow_origin_txt, list) else []
        shadow_impact_txt = shadow_impact_txt if isinstance(shadow_impact_txt, list) else []
        
        # [FIX] If origin/impact lines are missing, add them directly from anchor_result
        # This handles cases where anchors are outside the PDG range or not in slice text
        if not pri_origin_txt and anchor_result.origin_anchors:
            print(f"      [SliceFeature] Adding missing Pre Origins from anchors...")
            pri_origin_txt = [f"[{a.line:4d}] {a.content}" for a in anchor_result.origin_anchors]
        
        if not pri_impact_txt and anchor_result.impact_anchors:
            print(f"      [SliceFeature] Adding missing Pre Impacts from anchors...")
            pri_impact_txt = [f"[{a.line:4d}] {a.content}" for a in anchor_result.impact_anchors]
        
        # [FIX] Add fallback for shadow anchors if mapping failed
        # If shadow mapping produced empty results, try to generate them from anchors
        # Ensure shadow_origin_txt is a list before proceeding
        if not isinstance(shadow_origin_txt, list):
            print(f"      [Warning] shadow_origin_txt is not a list ({type(shadow_origin_txt)}), converting...")
            shadow_origin_txt = []
        
        if not isinstance(shadow_impact_txt, list):
            print(f"      [Warning] shadow_impact_txt is not a list ({type(shadow_impact_txt)}), converting...")
            shadow_impact_txt = []
        
        if not shadow_origin_txt and anchor_result.origin_anchors:
            print(f"      [SliceFeature] Shadow Origins empty, attempting to generate from anchors...")
            # Try to find corresponding shadow lines for each origin anchor
            for a in anchor_result.origin_anchors:
                shadow_mapped = mapper.map_lines_to_shadow([a.line])
                if shadow_mapped:
                    found_lines = get_lines_by_abs_msg(s_shadow_text, shadow_mapped)
                    if found_lines:
                        shadow_origin_txt.extend(found_lines)
            # If still empty, keep it empty (post-patch may have removed origin entirely)
            if not shadow_origin_txt:
                print(f"      [SliceFeature] No shadow origins found (origin removed in patch)")
        
        if not shadow_impact_txt and anchor_result.impact_anchors:
            print(f"      [SliceFeature] Shadow Impacts empty, attempting to generate from anchors...")
            # Try to find corresponding shadow lines for each impact anchor
            for a in anchor_result.impact_anchors:
                shadow_mapped = mapper.map_lines_to_shadow([a.line])
                if shadow_mapped:
                    found_lines = get_lines_by_abs_msg(s_shadow_text, shadow_mapped)
                    if found_lines:
                        shadow_impact_txt.extend(found_lines)
            # If still empty, keep it empty (post-patch may have removed impact entirely)
            if not shadow_impact_txt:
                print(f"      [SliceFeature] No shadow impacts found (impact removed in patch)")

        # Primary = Pre (OLD), Shadow = Post (NEW)
        # Ensure all variables are lists (never None)
        final_pre_origin = pri_origin_txt if pri_origin_txt is not None else []
        final_pre_impact = pri_impact_txt if pri_impact_txt is not None else []
        final_post_origin = shadow_origin_txt if shadow_origin_txt is not None else []
        final_post_impact = shadow_impact_txt if shadow_impact_txt is not None else []
            
        print(f"      [SliceFeature] Origin/Impact Lines:")
        print(f"        Pre Origins: {final_pre_origin}")
        print(f"        Pre Impacts: {final_pre_impact}")
        print(f"        Post Origins: {final_post_origin}")
        print(f"        Post Impacts: {final_post_impact}")
        
        # Final defensive check before creating SliceFeature
        if final_pre_origin is None or final_pre_impact is None or final_post_origin is None or final_post_impact is None:
            print(f"      [Warning] One or more anchor lists is None, replacing with empty lists")
            final_pre_origin = final_pre_origin if final_pre_origin is not None else []
            final_pre_impact = final_pre_impact if final_pre_impact is not None else []
            final_post_origin = final_post_origin if final_post_origin is not None else []
            final_post_impact = final_post_impact if final_post_impact is not None else []
        
        generated_slices[func] = SliceFeature(
            func_name=func,
            s_pre=final_s_pre,
            s_post=final_s_post,
            pre_origins=final_pre_origin,
            pre_impacts=final_pre_impact,
            post_origins=final_post_origin,
            post_impacts=final_post_impact,
            validation_status=None,
            validation_reasoning=validation_result.reasoning if validation_result else None
        )

    # Return slices and taxonomy
    return {"slices": generated_slices, "taxonomy": taxonomy}

# ==============================================================================
# Baseline: Agent-Based 端到端提取（用于对比实验）
# ==============================================================================

class SingleFunctionAgentSlice(BaseModel):
    func_name: str
    vulnerable_logic_indices: List[int] = Field(description="Line indices from the PRE-PATCH code (0-based) that explain the vulnerability.")
    fix_logic_indices: List[int] = Field(description="Line indices from the POST-PATCH code (0-based) that explain the fix.")
    pre_origins: List[int] = Field(description="Indices of Origin lines in PRE code.", default_factory=list)
    pre_impacts: List[int] = Field(description="Indices of Impact lines in PRE code.", default_factory=list)
    post_origins: List[int] = Field(description="Indices of Origin lines in POST code.", default_factory=list)
    post_impacts: List[int] = Field(description="Indices of Impact lines in POST code.", default_factory=list)
    reasoning: str

class AgentExtractionOutput(BaseModel):
    functions: List[SingleFunctionAgentSlice]
    root_cause: str
    attack_path: str
    fix_mechanism: str
    vuln_type: str = Field(description="Specific vulnerability type from the predefined list")
    cwe_info: Optional[str] = Field(description="CWE ID and name if applicable (e.g. 'CWE-416: Use After Free').")

def baseline_extraction_node(state: PatchExtractionState) -> Dict:
    """
    Baseline Method: Agent-driven end-to-end extraction with tools
    
    For comparison experiments, skip phased analysis, let LLM complete in one go:
    - Slice extraction (identify vulnerability-related code)
    - Semantic analysis (extract root cause, attack path, fix mechanism)
    - Tool support: Agent can access code navigation tools
    
    Returns:
        Dict containing analyzed_features (encapsulated as PatchFeatures)
    """
    patches = state['patches']
    commit_msg = state['commit_message']
    vul_id = state.get('vul_id', '')
    repo_path = state.get('repo_path', '')

    print(f"!!! [BASELINE] Running Agent-based Phase 2 Extraction for group: {state['group_id']} !!!")

    # Load metadata for CWE information
    from extraction.taxonomy import load_metadata
    metadata = load_metadata(vul_id) if vul_id else None
    
    cwe_id_meta = None
    cwe_name_meta = None
    description_meta = ""
    
    if metadata:
        cwe_id_meta = metadata.get('cwe_ids', [None])[0] if metadata.get('cwe_ids') else None
        cwe_name_meta = metadata.get('cwe_names', [None])[0] if metadata.get('cwe_names') else None
        description_meta = metadata.get('description', '')
        print(f"    [Baseline] Loaded metadata for {vul_id} (CWE: {cwe_id_meta})")

    # Initialize CodeNavigator for tool support
    from core.navigator import CodeNavigator
    navigator = CodeNavigator(repo_path)
    
    # Create tools for the agent
    def create_baseline_tools(current_file: str):
        """Create code navigation tools for baseline extraction"""
        
        @tool
        def read_file(start: int, end: int, file_path: str) -> str:
            """Read specific lines from a file.
            
            Args:
                start: Start line number
                end: End line number
                file_path: Target file path (REQUIRED)
            """
            try:
                return navigator.read_code_window(file_path, start, end)
            except Exception as e:
                return f"Error: {e}"
        
        @tool
        def find_definition(symbol_name: str, file_path: str) -> str:
            """Find definition of a symbol (function, struct, macro).
            
            Args:
                symbol_name: Name of the symbol
                file_path: Context file path (REQUIRED)
            """
            try:
                result = navigator.find_definition(symbol_name, context_path=file_path)
                return json.dumps(result[:3])  # Top 3 results
            except Exception as e:
                return json.dumps({"error": str(e)})
        
        @tool
        def grep(pattern: str, file_path: str) -> str:
            """Search for a pattern in a file.
            
            Args:
                pattern: Search pattern (variable/function name)
                file_path: Target file path (REQUIRED)
            """
            try:
                result = navigator.grep(pattern, file_path, mode="word")
                return json.dumps(result)
            except Exception as e:
                return json.dumps({"error": str(e)})
        
        return [read_file, find_definition, grep]

    # Prepare LLM with tools
    llm = ChatOpenAI(
            base_url=os.getenv("API_BASE"),
            api_key=os.getenv("API_KEY"),
            model=os.getenv("MODEL_NAME", "gpt-4o"),
            temperature=0
        )

    # Gather Full Code for each function
    func_contexts = []
    for patch in patches:
        def index_code(code_str):
            if not code_str: return "N/A"
            return "\n".join([f"[{i}] {line}" for i, line in enumerate(code_str.splitlines())])

        func_contexts.append(f"""
### Function: {patch.function_name}
File: {patch.file_path}

#### Pre-Patch Code (Vulnerable):
{index_code(patch.old_code)}

#### Post-Patch Code (Fixed):
{index_code(patch.new_code)}

#### Diff:
{patch.raw_diff}
""")

    # Get list of vulnerability types for the prompt
    from core.models import GeneralVulnType
    vuln_types_list = [vt.value for vt in GeneralVulnType]
    vuln_types_str = ", ".join(vuln_types_list)
    
    # Build metadata hint
    metadata_hint = ""
    if metadata:
        metadata_hint = f"""
**CVE Metadata** (Use as hints but verify against code):
- CVE ID: {vul_id}
- CWE ID: {cwe_id_meta or 'N/A'}
- CWE Name: {cwe_name_meta or 'N/A'}
- Description: {description_meta or 'N/A'}
"""

    system_prompt = f"""You are a security expert analyzing a software patch.
Your goal is to identify the core vulnerability and the fix logic.

{metadata_hint}

### Core Concepts

**Anchors** - Critical operations that embody the vulnerability mechanism:
- **Origin Anchors**: Operations that CREATE the vulnerable state
  Examples: memory allocation, resource initialization, state assignment, lock acquisition
- **Impact Anchors**: Operations that TRIGGER/EXPLOIT the vulnerability
  Examples: memory deref/use, resource access, state read, lock release
- **Vulnerability Chain**: Origin → [data/control flow] → Impact

**Slices** - Code subsets that explain the vulnerability:
- **Pre-Patch Slice (s_pre)**: Lines showing the vulnerability trigger path
  - Should include Origin anchors, Impact anchors, and the path connecting them
  - Focus on the minimal code that demonstrates the vulnerability
- **Post-Patch Slice (s_post)**: Lines showing how the fix blocks the vulnerability
  - Should include modified lines and their context
  - Demonstrates where/how the attack chain is broken

### Available Tools

You have access to code navigation tools:
- `read_file(start, end, file_path)`: Read specific lines from a file
- `find_definition(symbol_name, file_path)`: Find where a symbol is defined
- `grep(pattern, file_path)`: Search for a pattern in a file

Use these tools to:
- Check definitions of functions/structs/macros
- Understand cross-file references
- Trace variable usage and data flow

### Your Tasks

**Task 1**: For EACH function, identify slices by finding anchors:

1. **Find Origin Anchors** (where vulnerable state is created):
   - Look at deleted lines and their context
   - Identify operations that create problematic state
   - Common origins: alloc, init, assign, acquire

2. **Find Impact Anchors** (where vulnerability is triggered):
   - Look at where the vulnerable state is used
   - Identify operations that exploit the problem
   - Common impacts: deref, access, use, free

3. **Extract Slice Indices**:
   - `vulnerable_logic_indices`: Lines in PRE-patch showing Origin→Impact path (0-based)
   - `fix_logic_indices`: Lines in POST-patch showing the fix (0-based)
   - Include anchors + minimal connecting path

**You may use tools to verify your understanding and trace data/control flow.**

**Task 2**: Provide a GLOBAL summary:
- **Vulnerability Type**: Choose the MOST SPECIFIC type from: {vuln_types_str}
- **Root Cause**: What fundamental defect enabled this? (e.g., "missing null check after allocation")
- **Attack Path**: How does Origin→Impact manifest? (e.g., "allocation failure → null return → unchecked dereference")
- **Fix Mechanism**: How does the patch break the chain? (e.g., "adds null check after allocation, returns error")
- **CWE Info**: Most appropriate CWE ID and name (use metadata if available, otherwise infer)

### Important Notes

- Focus on the SPECIFIC vulnerability this patch fixes
- Anchors must be in the analyzed function (use call sites for cross-function cases)
- Slices should be minimal but complete (show the vulnerability chain)
- Use tools when uncertain about definitions or data flow
"""

    user_prompt = f"""
Commit Message: {commit_msg}

Functions to Analyze:
{" ".join(func_contexts)}
"""

    # Run agent loop with tools
    try:
        # Get the first patch's file path for tool context
        main_file = patches[0].file_path if patches else ""
        tools = create_baseline_tools(main_file)
        
        # Bind tools to LLM
        llm_with_tools = llm.bind_tools(tools)
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        # Agent loop: allow up to 5 tool calls
        max_iterations = 10
        for iteration in range(max_iterations):
            response = llm_invoke_with_retry(llm_with_tools, messages)
            
            if not response:
                raise Exception("LLM invocation failed after retries")
            
            # Check if agent wants to use tools
            if not response.tool_calls:
                # No more tool calls, get final answer
                structured_llm = llm.with_structured_output(AgentExtractionOutput)
                output: AgentExtractionOutput = llm_invoke_with_retry(
                    structured_llm,
                    messages + [response]
                )
                break
            
            # Execute tool calls
            messages.append(response)
            for tool_call in response.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                
                # Find and execute the tool
                selected_tool = next((t for t in tools if t.name == tool_name), None)
                if selected_tool:
                    try:
                        tool_result = selected_tool.invoke(tool_args)
                        messages.append(ToolMessage(
                            content=str(tool_result),
                            tool_call_id=tool_call['id']
                        ))
                    except Exception as e:
                        messages.append(ToolMessage(
                            content=f"Tool error: {e}",
                            tool_call_id=tool_call['id']
                        ))
        else:
            # Max iterations reached, force final answer
            print(f"    [Baseline] Max tool iterations reached, forcing final answer")
            structured_llm = llm.with_structured_output(AgentExtractionOutput)
            output: AgentExtractionOutput = llm_invoke_with_retry(
                structured_llm,
                [SystemMessage(content="Provide your final analysis based on the information gathered."),
                 HumanMessage(content=user_prompt)]
            )
    except Exception as e:
        print(f"Error in Baseline Phase 2 LLM call after retries: {e}")
        # Fallback empty structure
        return {"slices": {}, "taxonomy": None, "semantics": None}

    # Convert Output to PatchFeatures structure
    generated_slices: Dict[str, SliceFeature] = {}
    
    def get_lines(code, indices):
        if not code: return ""
        lines = code.splitlines()
        result = []
        for idx in indices:
            if 0 <= idx < len(lines):
                result.append(f"[{idx}] {lines[idx]}")
        return "\n".join(result)
        
    def get_raw_lines(code, indices):
        if not code: return []
        lines = code.splitlines()
        result = []
        for idx in indices:
             if 0 <= idx < len(lines):
                result.append(f"[{idx}] {lines[idx]}")
        return result

    for f_out in output.functions:
        p = next((p for p in patches if p.function_name == f_out.func_name), None)
        if not p: continue
        
        all_pre_indices = set(f_out.vulnerable_logic_indices) | set(f_out.pre_origins) | set(f_out.pre_impacts)
        all_post_indices = set(f_out.fix_logic_indices) | set(f_out.post_origins) | set(f_out.post_impacts)
        
        # Populate formatted slices (Sorted to maintain line order)
        s_pre = get_lines(p.old_code, sorted(list(all_pre_indices)))
        s_post = get_lines(p.new_code, sorted(list(all_post_indices)))
        
        # Populate raw origin/impact lists if available
        pre_origins = get_raw_lines(p.old_code, f_out.pre_origins)
        pre_impacts = get_raw_lines(p.old_code, f_out.pre_impacts)
        post_origins = get_raw_lines(p.new_code, f_out.post_origins)
        post_impacts = get_raw_lines(p.new_code, f_out.post_impacts)

        generated_slices[f_out.func_name] = SliceFeature(
            func_name=f_out.func_name,
            s_pre=s_pre,
            s_post=s_post,
            pre_origins=pre_origins,
            pre_impacts=pre_impacts,
            post_origins=post_origins,
            post_impacts=post_impacts,
            validation_status="AgentSlicing",
            validation_reasoning=f_out.reasoning
        )

    # Simplified Semantics (Directly from Agent Output)
    # First try to use metadata CWE if available
    if metadata and cwe_id_meta:
        cwe_id = cwe_id_meta
        cwe_name = cwe_name_meta
    else:
        # Parse from agent output
        cwe_match = re.search(r'CWE-(\d+)', output.cwe_info or "")
        cwe_id = f"CWE-{cwe_match.group(1)}" if cwe_match else None
        cwe_name = output.cwe_info.split(':')[-1].strip() if output.cwe_info and ':' in output.cwe_info else None

    # Map to GeneralVulnType from agent output
    from core.models import GeneralVulnType
    vuln_type = GeneralVulnType.UNKNOWN
    
    # Try to match the agent's vuln_type string to our enum
    try:
        for vt in GeneralVulnType:
            if vt.value.lower() == output.vuln_type.lower():
                vuln_type = vt
                break
    except:
        pass
    
    if vuln_type == GeneralVulnType.UNKNOWN:
        print(f"    [Warning] Could not map vuln_type '{output.vuln_type}' to GeneralVulnType enum, using UNKNOWN")
    
    semantics = SemanticFeature(
        vuln_type=vuln_type,
        cwe_id=cwe_id,
        cwe_name=cwe_name,
        root_cause=output.root_cause,
        attack_path=output.attack_path,
        fix_mechanism=output.fix_mechanism,
        evidence_index={}
    )

    # [Fix] Need to satisfy strict PatchFeatures schema which requires TaxonomyFeature
    # Determine type_confidence based on whether we used metadata
    from core.models import TypeConfidence
    type_confidence = TypeConfidence.HIGH if (metadata and cwe_id_meta) else TypeConfidence.MEDIUM
    
    dummy_taxonomy = TaxonomyFeature(
        vuln_type=vuln_type,
        type_confidence=type_confidence,
        cwe_id=cwe_id,
        cwe_name=cwe_name,
        reasoning="Baseline Agent Inference (Skipped Taxonomy Phase)",
        root_cause=output.root_cause,
        attack_path=output.attack_path,
        fix_mechanism=output.fix_mechanism
    )

    final_feat = PatchFeatures(
        group_id=state["group_id"],
        patches=state["patches"],
        commit_message=state["commit_message"],
        taxonomy=dummy_taxonomy,
        slices=generated_slices,
        semantics=semantics
    )

    return {
        "analyzed_features": [final_feat]
    }