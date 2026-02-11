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
from .anchor_analyzer import AnchorAnalyzer, AnchorResult
from core.categories import Anchor, AnchorLocatability, DependencyType, ChainLink
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
# Connection retry utility functions
# ==============================================================================

def llm_invoke_with_retry(llm, messages, max_retries: int = 3, retry_delay: float = 5.0):
    """
    Enhanced LLM invocation wrapper with automatic retry mechanism.
    Handles network errors (500, 502, 503, 504) and connection timeouts.
    
    Args:
        llm: ChatOpenAI instance or LLM with structured_output
        messages: List of messages
        max_retries: Maximum number of retries
        retry_delay: Retry interval (seconds), increases exponentially after each retry
    
    Returns:
        LLM response, returns None on failure (instead of raising exception)
        Caller needs to check if return value is None
    """
    last_exception = None
    current_delay = retry_delay
    
    for attempt in range(max_retries):
        try:
            return llm.invoke(messages)
        except Exception as e:
            last_exception = e
            error_str = str(e).lower()
            
            # Classify error type
            error_type = "unknown"
            is_retryable = False
            
            # Network/Server errors (Retryable)
            if any(x in error_str for x in ['500', '502', '503', '504', 'internal server error']):
                error_type = "server_error"
                is_retryable = True
            # Connection errors (Retryable)
            elif any(x in error_str for x in ['connection', 'timeout', 'timed out', 'network', 'reset', 'refused', 'broken pipe', 'unreachable']):
                error_type = "connection_error"
                is_retryable = True
            # Rate limit (Retryable, but longer delay)
            elif any(x in error_str for x in ['rate limit', 'too many requests', '429']):
                error_type = "rate_limit"
                is_retryable = True
                current_delay = max(current_delay, 30.0)  # Rate limit wait at least 30s
            # API Key/Authentication errors (Non-retryable)
            elif any(x in error_str for x in ['api key', 'authentication', 'unauthorized', '401', '403']):
                error_type = "auth_error"
                is_retryable = False
            
            if is_retryable and attempt < max_retries - 1:
                print(f"      [LLM-Retry] {error_type} (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"      [LLM-Retry] Waiting {current_delay:.1f}s before retry...")
                time.sleep(current_delay)
                current_delay *= 2  # Exponential backoff
            else:
                # Non-retryable error or last attempt
                print(f"      [LLM-Error] {error_type} (final attempt): {e}")
                return None  # Return None instead of raising exception
    
    # All retries failed
    print(f"      [LLM-Error] All {max_retries} attempts failed. Last error: {last_exception}")
    return None


def retry_on_connection_error(func, max_retries=3, initial_delay=2.0, backoff_factor=2.0):
    """
    [Legacy] Retry wrapper for LLM calls with exponential backoff.
    Preserve backward compatibility, new code suggests using llm_invoke_with_retry.
    
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
            
            # Classify error types (consistent with llm_invoke_with_retry)
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
# Data Models
# ==============================================================================

class ModifiedLine(BaseModel):
    """Represents a modified code line (for diff analysis)"""
    line_number: int = Field(description="The line number in the respective version (pre-patch for deleted, post-patch for added).")
    content: str = Field(description="The code content (stripped of leading/trailing whitespace).")
    raw_content: str = Field(description="The raw content with original formatting.")
    extracted_vars: Set[str] = Field(default_factory=set, description="Variables extracted from this line.")


# ==============================================================================
# Static Program Slicing Engine
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
            node_start = d.get('start_line', 0) # PDG nodes are Relative
            node_end = d.get('end_line', node_start)
            # [FIX] Match if target line falls anywhere within the node's line range,
            # not just at start_line. This handles multi-line statements (e.g., compound
            # conditions in while/if) where an anchor may reference a continuation line.
            if node_start > 0:
                for trl in target_rel_lines:
                    if node_start <= trl <= node_end:
                        candidates.append(n)
                        break
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
            # Meaning we focus on the child field, but the parent struct was used (usually passed as an argument), which might imply usage of the child field
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
        # worklist stores (node, relevant_vars)
        # relevant_vars: set of variables causing dependency when reaching this node
        worklist = []
        for nid in start_nodes:
            worklist.append((nid, initial_vars.copy()))
        
        # Modify Visited Strategy:
        # If we have visited this node with a "stronger" or "identical" set of variables, skip it
        # But for performance and to prevent infinite loops, the simplest way is to only record the Node (becoming context-insensitive),
        # Or restrict the visited key.
        # Here we adopt a compromise: only filter during edge judgment, normalize when enqueuing.
        
        # Actually in PDG Forward Slice, once sliced, all defs of the node become new pollution sources.
        # So we only need to store the node ID in the state to prevent infinite loops.
        visited_nodes = set(start_nodes) 

        while worklist:
            curr_node, curr_vars = worklist.pop(0)

            # Traverse outgoing edges
            for _, succ, data in self.pdg.out_edges(curr_node, data=True):
                rel_type = data.get('relationship')
                should_add = False
                
                # 1. Data dependency check
                if rel_type == 'DATA':
                    edge_var = data.get('var')
                    # Pass only if the variable on the edge is one we care about
                    if edge_var in curr_vars:
                        should_add = True
                
                # 2. Control dependency check
                elif rel_type == 'CONTROL':
                    # Control dependency usually means execution of curr_node determines whether succ executes
                    # If curr_node is in the slice, all its control children should usually be in the slice
                    # Here we can relax the condition, or check if variables used by curr_node are in curr_vars
                    curr_uses = set(self.pdg.nodes[curr_node].get('uses', {}).keys())
                    if not curr_uses.isdisjoint(curr_vars):
                        should_add = True

                if should_add:
                    # Get new variables defined by succ
                    succ_defs = set(self.pdg.nodes[succ].get('defs', {}).keys())
                    
                    # --- Key Modification ---
                    # The variables of interest for the next hop should be the variables produced by succ, not the accumulated previous variables.
                    # Because in PDG, curr_vars have already been consumed by curr_node,
                    # and the impact generated is transformed into succ_defs.
                    next_vars = succ_defs 
                    
                    # If succ has not been visited, or to recalculate new variable flow (if context-sensitive analysis),
                    # but to prevent infinite loops, it is recommended to process only once as long as the node is added to the slice,
                    # or ensure the visited logic can cover it.
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
# TwoPassSlicer: Anchor Candidate Collection (Algorithm 1, Line 3)
# ==============================================================================

class TwoPassSlicer:
    """
    TwoPassSlice anchor candidate collector (based on VERDICT paper Algorithm 1, Line 3).
    
    Collects vulnerability-related candidate statements from patch modification lines (Δ),
    for use by anchor_analyzer.identify().
    
    Three parallel sources for related_nodes:
      - Δ_nodes: deleted lines' PDG nodes
      - cfg_diff_nodes: CFG reachability difference between OLD/NEW
      - body_nodes: PREDICATE branch body nodes
    
    Then extracts V_Δ (variable set) and searches PDG for all def/use nodes.
    Final: candidates = related_nodes ∪ S_def ∪ S_use
    
    See plans/two_pass_slice_design_v2.md for full design rationale.
    """
    
    def __init__(self,
                 pdg_pri: nx.MultiDiGraph,
                 cfg_pri: nx.DiGraph,
                 cfg_shadow: nx.DiGraph,
                 code_pri: str,
                 code_shadow: str,
                 sl_pri: int,
                 sl_shadow: int,
                 search_hints: Dict[str, Any],
                 patch_diff: str):
        """
        Args:
            pdg_pri: OLD (pre-patch) PDG
            cfg_pri: OLD (pre-patch) CFG
            cfg_shadow: NEW (post-patch) CFG
            code_pri: OLD source code
            code_shadow: NEW source code
            sl_pri: OLD code start line (absolute)
            sl_shadow: NEW code start line (absolute)
            search_hints: Output from extract_search_hints()
            patch_diff: Unified diff text
        """
        self.pdg_pri = pdg_pri
        self.cfg_pri = cfg_pri
        self.cfg_shadow = cfg_shadow
        self.code_pri = code_pri
        self.code_shadow = code_shadow
        self.sl_pri = sl_pri
        self.sl_shadow = sl_shadow
        self.search_hints = search_hints
        self.patch_diff = patch_diff
    
    def collect_candidates(self) -> str:
        """
        Main entry: collect anchor candidates and format as text.
        
        Returns:
            Formatted candidate text for anchor_analyzer prompt injection.
            Empty string if no candidates found.
        """
        print(f"    [TwoPassSlicer] Starting candidate collection...")
        
        # Phase 1: Collect related nodes (parallel)
        delta_nodes = self._find_delta_nodes()
        print(f"      Δ_nodes: {len(delta_nodes)} nodes")
        
        cfg_diff_nodes = self._cfg_diff_analysis(delta_nodes)
        print(f"      cfg_diff_nodes: {len(cfg_diff_nodes)} nodes")
        
        body_nodes = self._get_body_nodes(delta_nodes)
        print(f"      body_nodes: {len(body_nodes)} nodes")
        
        related_nodes = delta_nodes | cfg_diff_nodes | body_nodes
        print(f"      related_nodes total: {len(related_nodes)} nodes")
        
        # Extract V_Δ from related_nodes + search_hints
        v_delta = self._extract_variables_from_nodes(related_nodes)
        key_vars = self.search_hints.get('key_variables', set())
        v_delta |= key_vars
        print(f"      V_Δ: {v_delta}")
        
        # Fallback: all function defs
        if not v_delta:
            print(f"      V_Δ empty, falling back to all function defs")
            v_delta = self._all_function_def_vars()
            print(f"      V_Δ (fallback): {v_delta}")
        
        # Phase 2: Search def/use nodes in PDG
        s_def, s_use = self._search_def_use(v_delta)
        print(f"      S_def: {len(s_def)} nodes, S_use: {len(s_use)} nodes")
        
        # Phase 3: Merge
        candidates = related_nodes | s_def | s_use
        print(f"      Total candidates: {len(candidates)} nodes")
        
        if not candidates:
            return ""
        
        return self._format_candidates(candidates)
    
    # ==================== Phase 1: Collect Related Nodes ====================
    
    def _find_delta_nodes(self) -> Set[str]:
        """
        Find Δ nodes: deleted lines' corresponding nodes in OLD PDG.
        For Additive patches, returns empty set (relies on key_variables).
        """
        delta_nodes = set()
        deleted_lines = self.search_hints.get('deleted_lines', [])
        
        if not deleted_lines:
            return delta_nodes
        
        for mod_line in deleted_lines:
            # mod_line.line_number is absolute line number from diff
            abs_line = mod_line.line_number
            # Convert to relative line number for PDG lookup
            rel_line = abs_line - self.sl_pri + 1
            
            # Find matching PDG nodes
            for n, d in self.pdg_pri.nodes(data=True):
                if d.get('type') in ('EXIT', 'MERGE', 'ENTRY'):
                    continue
                node_rel_line = d.get('start_line', 0)
                if node_rel_line == rel_line:
                    delta_nodes.add(n)
        
        return delta_nodes
    
    def _cfg_diff_analysis(self, delta_nodes: Set[str]) -> Set[str]:
        """
        CFG reachability difference analysis.
        
        Find nodes that are reachable from Δ in NEW CFG but NOT in OLD CFG.
        Then match them back to OLD PDG nodes by code content.
        
        Key: Must start from Δ nodes, NOT from function entry.
        This ensures we only see "what the fix changed on the error path".
        """
        if not delta_nodes or self.cfg_pri is None or self.cfg_shadow is None:
            return set()
        
        try:
            # 1. Find OLD Δ nodes in OLD CFG (by relative line number)
            old_delta_cfg_nodes = set()
            delta_rel_lines = set()
            for n in delta_nodes:
                d = self.pdg_pri.nodes.get(n, {})
                rel_line = d.get('start_line', 0)
                if rel_line > 0:
                    delta_rel_lines.add(rel_line)
            
            for n, d in self.cfg_pri.nodes(data=True):
                if d.get('start_line', 0) in delta_rel_lines:
                    old_delta_cfg_nodes.add(n)
            
            # 2. Find NEW Δ nodes in NEW CFG (added lines)
            new_delta_cfg_nodes = set()
            added_lines = self.search_hints.get('added_lines', [])
            added_rel_lines = set()
            for mod_line in added_lines:
                abs_line = mod_line.line_number
                rel_line = abs_line - self.sl_shadow + 1
                if rel_line > 0:
                    added_rel_lines.add(rel_line)
            
            for n, d in self.cfg_shadow.nodes(data=True):
                if d.get('start_line', 0) in added_rel_lines:
                    new_delta_cfg_nodes.add(n)
            
            if not old_delta_cfg_nodes and not new_delta_cfg_nodes:
                return set()
            
            # 3. Compute reachable sets from Δ (not from entry!)
            old_reachable = set()
            for n in old_delta_cfg_nodes:
                old_reachable |= nx.descendants(self.cfg_pri, n)
            
            new_reachable = set()
            for n in new_delta_cfg_nodes:
                new_reachable |= nx.descendants(self.cfg_shadow, n)
            
            # 4. Find newly reachable code (by content matching)
            old_reachable_code = set()
            for n in old_reachable:
                code = self.cfg_pri.nodes[n].get('code', '').strip()
                if code and code not in ('EXIT', 'ENTRY', 'RETURN', 'BREAK', 'CONTINUE'):
                    old_reachable_code.add(code)
            
            newly_reachable_code = set()
            for n in new_reachable:
                code = self.cfg_shadow.nodes[n].get('code', '').strip()
                if code and code not in ('EXIT', 'ENTRY', 'RETURN', 'BREAK', 'CONTINUE'):
                    if code not in old_reachable_code:
                        newly_reachable_code.add(code)
            
            if not newly_reachable_code:
                return set()
            
            # 5. Match back to OLD PDG nodes by code content
            matched_nodes = set()
            for n, d in self.pdg_pri.nodes(data=True):
                if d.get('type') in ('EXIT', 'MERGE', 'ENTRY'):
                    continue
                node_code = d.get('code', '').strip()
                if node_code in newly_reachable_code:
                    matched_nodes.add(n)
            
            return matched_nodes
            
        except Exception as e:
            print(f"      [CFG Diff] Error: {e}")
            return set()
    
    def _get_body_nodes(self, delta_nodes: Set[str]) -> Set[str]:
        """
        For PREDICATE Δ nodes, collect branch body nodes via CFG TRUE/FALSE edges.
        
        Uses CFG structure (not brace matching) to handle:
        - if (flag) { ... }     (with braces)
        - if (flag) stmt;       (single statement, same line)
        - if (flag)\\n  stmt;   (single statement, next line)
        """
        if not delta_nodes or self.cfg_pri is None:
            return set()
        
        body_nodes = set()
        max_depth = 10  # Limit traversal depth to prevent explosion
        
        for node in delta_nodes:
            node_data = self.pdg_pri.nodes.get(node, {})
            node_type = node_data.get('type', '')
            
            # Only process PREDICATE nodes (if/while/for conditions)
            if node_type != 'PREDICATE':
                continue
            
            # Find this node in CFG (same node ID since PDG copies CFG nodes)
            if node not in self.cfg_pri:
                continue
            
            # Collect TRUE and FALSE branch body nodes
            for _, v, edge_data in self.cfg_pri.out_edges(node, data=True):
                edge_type = str(edge_data.get('type', ''))
                if edge_type in ('TRUE', 'FALSE', 'CFGEdgeType.TRUE', 'CFGEdgeType.FALSE'):
                    self._collect_branch_nodes(v, body_nodes, max_depth)
        
        return body_nodes
    
    def _collect_branch_nodes(self, start: str, result: Set[str], max_depth: int):
        """BFS along FLOW edges to collect branch body nodes."""
        visited = set()
        queue = [(start, 0)]
        
        while queue:
            node, depth = queue.pop(0)
            if node in visited or depth > max_depth:
                continue
            visited.add(node)
            
            # Skip virtual nodes
            node_data = self.cfg_pri.nodes.get(node, {})
            node_type = node_data.get('type', '')
            if node_type in ('EXIT', 'MERGE', 'ENTRY'):
                continue
            
            # Add to result (map back to PDG node - same ID)
            if node in self.pdg_pri:
                result.add(node)
            
            # Continue along FLOW edges only (stop at branch points)
            for _, v, edge_data in self.cfg_pri.out_edges(node, data=True):
                edge_type = str(edge_data.get('type', ''))
                if edge_type in ('FLOW', 'CFGEdgeType.FLOW'):
                    queue.append((v, depth + 1))
    
    # ==================== Phase 2: Variable Extraction & Search ====================
    
    @staticmethod
    def _normalize_var_name(raw_name: str) -> Set[str]:
        """
        Extract base variable names from a raw PDG var path.
        
        PDG's _get_var_path() may return complex expressions like:
        - "(unsigned char) token[next++]" → {token, next}
        - "obj->member" → {obj}
        - "*ptr" → {ptr}
        - "buf" → {buf}
        
        We extract all C identifier tokens and filter out type keywords/casts.
        """
        # Extract all identifier-like tokens
        tokens = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', raw_name)
        
        # Filter out C type keywords and casts
        type_keywords = {
            'unsigned', 'signed', 'char', 'short', 'int', 'long', 'float', 'double',
            'void', 'const', 'volatile', 'static', 'extern', 'register', 'auto',
            'struct', 'union', 'enum', 'typedef', 'sizeof', 'typeof',
            'NULL', 'true', 'false', 'bool', 'size_t', 'ssize_t',
            'uint8_t', 'uint16_t', 'uint32_t', 'uint64_t',
            'int8_t', 'int16_t', 'int32_t', 'int64_t',
            'u8', 'u16', 'u32', 'u64', 's8', 's16', 's32', 's64',
        }
        
        return {t for t in tokens if t not in type_keywords and len(t) > 1}
    
    def _extract_variables_from_nodes(self, nodes: Set[str]) -> Set[str]:
        """Extract all variable names from a set of PDG nodes (defs ∪ uses), with normalization."""
        variables = set()
        for node in nodes:
            data = self.pdg_pri.nodes.get(node, {})
            defs = data.get('defs', {})
            uses = data.get('uses', {})
            for raw_name in defs.keys():
                variables |= self._normalize_var_name(raw_name)
            for raw_name in uses.keys():
                variables |= self._normalize_var_name(raw_name)
        return variables
    
    def _all_function_def_vars(self) -> Set[str]:
        """Fallback: collect all def variables in the entire function."""
        variables = set()
        for _, data in self.pdg_pri.nodes(data=True):
            defs = data.get('defs', {})
            for raw_name in defs.keys():
                variables |= self._normalize_var_name(raw_name)
        return variables
    
    def _search_def_use(self, v_delta: Set[str]) -> Tuple[Set[str], Set[str]]:
        """
        Search PDG for all nodes that define or use V_Δ variables.
        Uses _normalize_var_name to match against raw PDG var paths.
        
        Returns:
            (S_def, S_use) - sets of node IDs
        """
        s_def = set()
        s_use = set()
        
        for n, data in self.pdg_pri.nodes(data=True):
            if data.get('type') in ('EXIT', 'MERGE', 'ENTRY'):
                continue
            
            # Normalize raw PDG var names before matching
            node_def_vars = set()
            for raw_name in data.get('defs', {}).keys():
                node_def_vars |= self._normalize_var_name(raw_name)
            
            node_use_vars = set()
            for raw_name in data.get('uses', {}).keys():
                node_use_vars |= self._normalize_var_name(raw_name)
            
            if node_def_vars & v_delta:
                s_def.add(n)
            if node_use_vars & v_delta:
                s_use.add(n)
        
        return s_def, s_use
    
    # ==================== Phase 3: Formatting ====================
    
    def _format_candidates(self, candidate_nodes: Set[str]) -> str:
        """
        Format candidate nodes as text for anchor_analyzer prompt injection.
        
        Format: L{abs_line}: {code}  [type={node_type}, def={defs}, use={uses}]
        """
        lines = []
        
        # Sort by line number for readability
        sorted_nodes = sorted(
            candidate_nodes,
            key=lambda n: self.pdg_pri.nodes.get(n, {}).get('start_line', 0)
        )
        
        for node in sorted_nodes:
            data = self.pdg_pri.nodes.get(node, {})
            rel_line = data.get('start_line', 0)
            abs_line = rel_line + self.sl_pri - 1 if rel_line > 0 else 0
            code = data.get('code', '').strip()
            node_type = data.get('type', '')
            defs = set(data.get('defs', {}).keys())
            uses = set(data.get('uses', {}).keys())
            
            # Skip if no meaningful code
            if not code or code in ('EXIT', 'ENTRY'):
                continue
            
            line_str = f"L{abs_line}: {code}"
            if defs or uses:
                line_str += f"  [def={defs}, use={uses}]"
            lines.append(line_str)
        
        # Limit output size (max 50 lines)
        if len(lines) > 50:
            print(f"      [TwoPassSlicer] Truncating candidates from {len(lines)} to 50 lines")
            lines = lines[:50]
        
        return '\n'.join(lines)


# ==============================================================================
# Shadow Slice Mapper (Context Synchronization)
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
                # EXACT: Identical lines - Bidirectional mapping
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
        One-step generation of cleaned Shadow Slice
        
        Args:
            pri_nodes: Set of nodes in Primary Slice
            deleted_lines: List of deleted code line contents (for injecting vulnerability core)
            
            
        Returns:
            Cleaned Shadow node set
        """
        shadow_nodes = set()
        slicer_shadow = Slicer(self.pdg_shadow, self.code_shadow, self.sl_shadow)
        
        # Collect Primary active information
        active_pri_lines = set()      # Line numbers retained in Primary (rel)
        active_pri_contents = set()   # Line contents retained in Primary (stripped)
        
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
        
        # ============ Step 1: Map Primary nodes -> Shadow nodes ============
        for nid in pri_nodes:
            if nid not in self.pdg_pri.nodes:
                continue
            pri_rel = self.pdg_pri.nodes[nid].get('start_line', 0)
            
            if pri_rel in self.pri_to_shadow:
                shadow_rel = self.pri_to_shadow[pri_rel]
                found = [n for n, d in self.pdg_shadow.nodes(data=True) 
                         if d.get('start_line') == shadow_rel]
                shadow_nodes.update(found)
        
        # ============ Step 2: Handle Replace blocks (ensure added fix code is included) ============
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
        
        # ============ Step 3: Inject deleted lines (Vulnerability Core) ============
        deleted_anchors = set()
        if deleted_lines:
            for content in deleted_lines:
                found = slicer_shadow.get_nodes_by_location(-1, content)
                deleted_anchors.update(found)
            
            if deleted_anchors:
                # Local slice extension for deleted lines (control flow only, to avoid introducing too much noise)
                ctrl_nodes = slicer_shadow.backward_slice_control_only(list(deleted_anchors))
                shadow_nodes.update(ctrl_nodes)
                shadow_nodes.update(deleted_anchors)
        
        # ============ Step 3.5: Add INSERT line nodes (patch added code) ============
        # [FIX] Nodes of INSERT lines need to be added actively because they are not in the mapping
        print(f"      [Mapping] Adding INSERT lines ({len(self.unique_post_lines)} lines)...")
        for s_ln in self.unique_post_lines:
            # Find all nodes that COVER this line (start_line <= s_ln <= end_line)
            # This handles multi-line statements like if-blocks
            found = [n for n, d in self.pdg_shadow.nodes(data=True)
                     if (d.get('start_line', 0) <= s_ln <= d.get('end_line', d.get('start_line', 0))
                         and d.get('type') not in ('EXIT', 'MERGE', 'NO_OP'))]
            if found:
                shadow_nodes.update(found)
        
        # ============ Step 4: Calculate Semantic Range Anchors ============
        min_pri, max_pri = min(active_pri_lines), max(active_pri_lines)
        
        # Find the corresponding position of Primary range boundaries in Shadow
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
        
        # ============ Step 5: Sync Cleanup (Apply Retention Rules) ============
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
            
            # Force retain deleted lines (vulnerability core code)
            if nid in deleted_anchors:
                continue
            
            # Force retain added lines in replace block
            if s_ln in replace_shadow_lines:
                continue
            
            should_keep = False
            line_type = "UNKNOWN"
            
            # Type 1: EXACT/WHITESPACE (mapped - lines not modified in diff)
            if s_ln in self.shadow_to_pri:
                mapped_pri = self.shadow_to_pri[s_ln]
                # Keep only when the corresponding line in Primary is in the active set
                should_keep = (mapped_pri in active_pri_lines)
                line_type = "EXACT/WHITESPACE"
            
            # Type 2: MOVED (content appears in both deleted and added lines, but at different positions)
            # Use bidirectional mapping to judge: keep only when the corresponding line in Primary is in the active set
            elif s_ln in self.moved_shadow_to_pri:
                mapped_pri = self.moved_shadow_to_pri[s_ln]
                # MOVED line: keep only when the corresponding line in Primary is in the active set
                should_keep = (mapped_pri in active_pri_lines)
                line_type = "MOVED"
            
            # Type 3: MODIFIED (content is similar to deleted line but not identical)
            # Use bidirectional mapping to judge: keep only when the corresponding line in Primary is in the active set
            elif s_ln in self.modified_shadow_to_pri:
                mapped_pri = self.modified_shadow_to_pri[s_ln]
                # MODIFIED line: keep only when the corresponding line in Primary is in the active set
                should_keep = (mapped_pri in active_pri_lines)
                line_type = "MODIFIED"
            
            # Type 4: UNIQUE_POST (purely new, no counterpart found in deleted lines)
            # This is brand new code added by the patch, should be kept (no corresponding Primary line)
            elif s_ln in self.unique_post_lines:
                # UNIQUE_POST line: code added by patch, keep all
                should_keep = True
                line_type = "UNIQUE_POST"
            
            # Type 5: Other cases (not in diff, no mapping)
            # Keep if within semantic range
            else:
                should_keep = (anchor_top_s < s_ln < anchor_bot_s)
                line_type = "OTHER"
            
            if not should_keep:
                nodes_to_remove.add(nid)
        
        shadow_nodes.difference_update(nodes_to_remove)
        
        # Ensure ENTRY node exists
        shadow_nodes.update([n for n, d in self.pdg_shadow.nodes(data=True) 
                            if d.get('type') == 'ENTRY'])
        
        return shadow_nodes
    
    def map_lines_to_shadow(self, pri_abs_lines: List[int]) -> List[int]:
        """
        Map Primary line numbers -> Shadow line numbers
        
        Args:
            pri_abs_lines: List of absolute line numbers in Primary
            
        Returns:
            List of corresponding Shadow absolute line numbers
        """
        result = []
        
        for abs_p in pri_abs_lines:
            rel_p = abs_p - self.sl_pri + 1
            
            # Case 1: Direct mapping (EXACT / WHITESPACE)
            if rel_p in self.pri_to_shadow:
                rel_s = self.pri_to_shadow[rel_p]
                result.append(rel_s - 1 + self.sl_shadow)
                continue
            
            # Case 2: MOVED mapping (same content, different position)
            if rel_p in self.moved_pri_to_shadow:
                rel_s = self.moved_pri_to_shadow[rel_p]
                result.append(rel_s - 1 + self.sl_shadow)
                continue
            
            # Case 3: MODIFIED mapping (context-based pairing of changed lines)
            if rel_p in self.modified_pri_to_shadow:
                rel_s = self.modified_pri_to_shadow[rel_p]
                result.append(rel_s - 1 + self.sl_shadow)
                continue
            
            # Case 4: Content similarity matching (fallback)
            if 1 <= rel_p <= len(self.lines_pri):
                p_content = self.lines_pri[rel_p - 1].strip()
                if p_content:
                    # Search for same content in Shadow
                    if p_content in self.shadow_content_to_lines:
                        # Pick the nearest one
                        candidates = list(self.shadow_content_to_lines[p_content])
                        if candidates:
                            best_s_rel = min(candidates, key=lambda x: abs(x - rel_p))
                            result.append(best_s_rel - 1 + self.sl_shadow)
        
        return result
    
    # ============ Backward Compatibility Interface (Static Method) ============
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
        [Compatible with old interface] Map and slice
        Returns: (all_shadow_nodes, mandatory_nodes)
        """
        mapper = ShadowMapper(code_pri, code_shadow, pdg_pri, pdg_shadow,
                              start_line_pri, start_line_shadow, diff_text)
        shadow_nodes = mapper.generate_shadow_slice(s_pri_nodes, deleted_lines)
        # mandatory_nodes is no longer used, return empty set
        return (shadow_nodes, set())

# ==============================================================================
# Slice Validator and Cleaner
# ==============================================================================

class SliceValidator:
    """
    Agent-driven slice distiller (Distillation)
    
    Responsibility: Semantically review static slices (forward ∪ backward) to remove noise code
    
    Distillation Strategy (based on Methodology §3.3.2):
    1. **Anchor Retention**: All typed anchors are unconditionally kept
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
                      anchors: List[Anchor] = [],
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
            anchors: Typed anchors (all types)
            cwe_info: CWE information
            navigator: Code navigator
            file_path: File path
            
        Returns:
            SliceValidationResult or None
        """
        if not slice_code:
            return None
        
        # Anchor text (typed)
        anchors_text = "\n".join([f"- Line {a.line_number}: {a.code_snippet} [{a.type.value}]" for a in anchors]) if anchors else "None identified."

        # Primary is fixed as Pre-Patch (Vulnerable)
        slice_version = "Pre-Patch (Vulnerable)"
        
        # Get vulnerability-type specific distillation strategy
        vuln_type_strategy = get_strategy_prompt_section(vuln_type)
        vuln_type_name = vuln_type.value if isinstance(vuln_type, GeneralVulnType) else str(vuln_type)
        
        # print(f"      [Distillation] Engaging semantic filtering ({len(slice_code.splitlines())} lines).")
        # print(f"      [Distillation] Using strategy for: {vuln_type_name}")
        
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
        The slice is generated from static program analysis (forward ∪ backward slicing from typed anchors).
        It may contain noise: irrelevant data processing, logging, feature flags, etc.
        
        ### Your Task
        Review **all statements** (data-flow and control-flow) to identify which are essential to the vulnerability mechanism.
        
        **CRITICAL CONSTRAINT**: You must be **highly selective**. Keep ONLY the minimal set of lines that directly participate in the vulnerability mechanism.
        - **Maximum lines to keep**: {max_output_lines} lines (strictly enforced)
        - **Prioritize**: Typed anchors > direct data-flow > control guards > context
        - **Be ruthless**: If uncertain whether a line is essential, REMOVE it
        
        ### Context
        - **Function**: `{func_name}`
        - **Vuln Type**: {vuln_type_name} ({cwe_info})
        - **Root Cause**: {hypothesis.root_cause}
        - **Attack Chain**: {hypothesis.attack_chain}
        - **Key Variable**: {focus_var}
        - **Typed Anchors** (vulnerability chain):
{anchors_text}
        - **Version**: {slice_version}
        
        ### Input Data
        - **Patch Diff**: {diff_text}
        - **Candidate Slice** ({input_line_count} lines):
        {slice_code}

        ### Classification Rules (Strict Priority Order)
        **Priority 1 - MUST KEEP**:
        - Typed anchor lines (MANDATORY)
        
        **Priority 2 - KEEP if directly relevant**:
        - Data-flow statements that directly compute or propagate the vulnerable data along the chain
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
        - Use `trace_variable` to verify if a variable reaches downstream anchors
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
                    # print(f"      [DistillationAgent] Calling Tool: {t['name']}")
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
# Utility Functions
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
    
    # Match hunk header: @@ -10,5 +10,6 @@
    hunk_re = re.compile(r'^@@\s*-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@')
    
    # Current line number counters
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
        # Skip file header
        if line.startswith('---') or line.startswith('+++'):
            continue
        
        # Parse hunk header
        m = hunk_re.match(line)
        if m:
            c_old = int(m.group(1))
            c_new = int(m.group(3))
            continue
        
        # Handle deleted lines (-): exist in pre-patch, deleted in post-patch
        if line.startswith('-') and not line.startswith('---'):
            raw_content = line[1:]  # Keep original format (including indentation)
            stripped = raw_content.strip()
            
            # Skip empty lines and comments
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
        
        # Handle added lines (+): added in post-patch
        elif line.startswith('+') and not line.startswith('+++'):
            raw_content = line[1:]
            stripped = raw_content.strip()
            
            # Skip empty lines and comments
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
        
        # Handle context lines (exist in both sides)
        else:
            c_old += 1
            c_new += 1
    
    # Clean up variable list (remove common keywords and function names)
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
        # Lines existing in Pre-Patch but deleted in Post-Patch (start with '-')
        if line.startswith('-') and not line.startswith('---'):
            content = line[1:].strip()
            # Filter comments and empty lines
            if content and not content.startswith(('//','/*', '*', '#')):
                deleted.append(content)
    return deleted

# ==============================================================================
# ShortestPath Context Extraction (Algorithm 1 L14-17)
# ==============================================================================

def _build_filtered_subgraph(
    pdg: nx.MultiDiGraph,
    dep_type: DependencyType
) -> nx.DiGraph:
    """
    Build a δ-filtered subgraph from the PDG, keeping only edges matching the
    given dependency type.
    
    Mapping:
    - DATA / TEMPORAL → keep only relationship='DATA' edges
    - CONTROL → keep only relationship='CONTROL' edges
    
    Returns a simplified nx.DiGraph (collapses multi-edges) for shortest_path.
    """
    allowed = 'DATA' if dep_type in (DependencyType.DATA, DependencyType.TEMPORAL) else 'CONTROL'
    
    filtered = nx.DiGraph()
    # Add all nodes (preserve attributes for later line-number lookup)
    for n, d in pdg.nodes(data=True):
        filtered.add_node(n, **d)
    
    # Add only matching edges (collapse multi-edges into single edges)
    for u, v, _k, d in pdg.edges(keys=True, data=True):
        if d.get('relationship') == allowed:
            if not filtered.has_edge(u, v):
                filtered.add_edge(u, v, **d)
    
    return filtered


def _find_typed_shortest_path(
    pdg: nx.MultiDiGraph,
    src_node: str,
    dst_node: str,
    dep_type: DependencyType,
    max_intermediate: int = 15
) -> Optional[List[str]]:
    """
    Find shortest path between src_node and dst_node on a δ-filtered PDG subgraph.
    
    Implements fallback strategy:
      1. δ-filtered subgraph (only matching edge type)
      2. Full PDG (all edge types) if δ-filtered fails
      3. None if no path exists at all
    
    Args:
        pdg: The full PDG (nx.MultiDiGraph)
        src_node: Source PDG node ID
        dst_node: Destination PDG node ID
        dep_type: Dependency type for edge filtering
        max_intermediate: Max intermediate nodes before truncation
        
    Returns:
        List of node IDs on the path (including endpoints), or None
    """
    if src_node == dst_node:
        return [src_node]
    
    # Fallback 1: δ-filtered subgraph
    filtered = _build_filtered_subgraph(pdg, dep_type)
    try:
        path = nx.shortest_path(filtered, src_node, dst_node)
        # Truncate if path is too long (keep first N/2 + last N/2 + endpoints)
        if len(path) - 2 > max_intermediate:
            half = max_intermediate // 2
            path = path[:half + 1] + path[-(half + 1):]
            print(f"        [ShortestPath] Path truncated to {len(path)} nodes (max_intermediate={max_intermediate})")
        return path
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        pass
    
    # Fallback 2: Full PDG (all edge types) — build a simple DiGraph from all edges
    full_digraph = nx.DiGraph()
    for n, d in pdg.nodes(data=True):
        full_digraph.add_node(n, **d)
    for u, v, _k, d in pdg.edges(keys=True, data=True):
        if not full_digraph.has_edge(u, v):
            full_digraph.add_edge(u, v, **d)
    
    try:
        path = nx.shortest_path(full_digraph, src_node, dst_node)
        if len(path) - 2 > max_intermediate:
            half = max_intermediate // 2
            path = path[:half + 1] + path[-(half + 1):]
            print(f"        [ShortestPath] Fallback path truncated to {len(path)} nodes")
        return path
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        pass
    
    # Fallback 3: No path found at all
    return None


def shortest_path_context(
    pdg: nx.MultiDiGraph,
    anchor_nodes: List[Tuple[str, str]],
    chain: List[ChainLink],
    slicer,
    sl: int
) -> Set[str]:
    """
    Algorithm 1 L14-17: Extract inter-anchor context via ShortestPath.
    
    For each ChainLink(m_i, m_j, δ) in the constraint chain:
      1. Find all anchor instances matching m_i type and m_j type
      2. For each (src, dst) pair, find shortest path on δ-filtered PDG
      3. Collect intermediate nodes into S_ctx
    
    Args:
        pdg: The full PDG (nx.MultiDiGraph)
        anchor_nodes: List of (node_id, anchor_type_str) tuples
        chain: List of ChainLink from category.constraint.chain
        slicer: Slicer instance (for get_nodes_by_location)
        sl: Start line offset (absolute line of first code line)
        
    Returns:
        Set of PDG node IDs: M ∪ S_ctx (anchor nodes + intermediate context)
    """
    # M = all anchor node IDs
    M = {nid for nid, _ in anchor_nodes}
    S_ctx = set()
    
    # Build anchor_type → [node_id, ...] mapping
    type_to_nodes: Dict[str, List[str]] = {}
    for nid, atype in anchor_nodes:
        if atype not in type_to_nodes:
            type_to_nodes[atype] = []
        type_to_nodes[atype].append(nid)
    
    print(f"      [ShortestPath] Anchor type mapping: { {k: len(v) for k, v in type_to_nodes.items()} }")
    print(f"      [ShortestPath] Chain: {[str(link) for link in chain]}")
    
    for link in chain:
        src_type = link.source.value  # e.g., 'source', 'alloc'
        dst_type = link.target.value  # e.g., 'computation', 'dealloc'
        dep_type = link.dependency    # DependencyType enum
        
        src_nodes = type_to_nodes.get(src_type, [])
        dst_nodes = type_to_nodes.get(dst_type, [])
        
        if not src_nodes or not dst_nodes:
            if link.is_optional:
                print(f"        [ShortestPath] Optional link {link}: missing anchors, skipping")
                continue
            else:
                print(f"        [ShortestPath] ⚠ Required link {link}: missing anchors "
                      f"(src={src_type}:{len(src_nodes)}, dst={dst_type}:{len(dst_nodes)})")
                continue
        
        # Cartesian product: find path for each (src, dst) pair
        paths_found = 0
        for s_nid in src_nodes:
            for d_nid in dst_nodes:
                if s_nid == d_nid:
                    continue
                
                path = _find_typed_shortest_path(pdg, s_nid, d_nid, dep_type)
                
                if path is not None:
                    # Collect intermediate nodes (exclude endpoints which are already in M)
                    intermediate = set(path[1:-1]) if len(path) > 2 else set()
                    S_ctx.update(intermediate)
                    paths_found += 1
                    
                    # Debug: show path info
                    src_line = pdg.nodes.get(s_nid, {}).get('start_line', '?')
                    dst_line = pdg.nodes.get(d_nid, {}).get('start_line', '?')
                    print(f"        [ShortestPath] {link}: L{src_line}→L{dst_line}, "
                          f"path={len(path)} nodes, intermediate={len(intermediate)}")
                else:
                    src_line = pdg.nodes.get(s_nid, {}).get('start_line', '?')
                    dst_line = pdg.nodes.get(d_nid, {}).get('start_line', '?')
                    print(f"        [ShortestPath] {link}: L{src_line}→L{dst_line}, NO PATH FOUND")
        
        if paths_found == 0 and not link.is_optional:
            print(f"        [ShortestPath] ⚠ No paths found for required link {link}")
    
    result = M | S_ctx
    print(f"      [ShortestPath] Result: M={len(M)} nodes, S_ctx={len(S_ctx)} nodes, total={len(result)}")
    return result


# ==============================================================================
# Slice generation function (generate slice from Anchors)
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
    Generate program slice from Anchor results using ShortestPath context extraction.
    
    Implements Algorithm 1 L14-17 from the VERDICT paper:
    - For each ChainLink(m_i, m_j, δ), find shortest path on δ-filtered PDG
    - Collect intermediate nodes as context (S_ctx)
    - Final slice = M (anchor nodes) ∪ S_ctx (intermediate context) ∪ diff nodes
    
    This function is decoupled for use in retry loops.
    
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
    
    all_anchors = anchor_result.anchors
    
    # =========================================================================
    # Step 1: Map Anchors to PDG nodes
    # =========================================================================
    anchor_nodes = []  # [(node_id, anchor_type_str), ...]
    for anchor in all_anchors:
        # L5-L11: Skip CONCEPTUAL anchors (no concrete code location)
        if hasattr(anchor, 'locatability') and anchor.locatability in (AnchorLocatability.CONCEPTUAL, 'conceptual'):
            print(f"      [Slicing] Skipping CONCEPTUAL anchor: ({anchor.type.value}) — {anchor.reasoning[:60]}")
            continue
        
        nodes = slicer_pri.get_nodes_by_location(anchor.line_number, anchor.code_snippet)
        if nodes:
            loc_tag = ""
            if hasattr(anchor, 'locatability') and anchor.locatability in (AnchorLocatability.ASSUMED, 'assumed'):
                loc_tag = " [ASSUMED]"
            anchor_nodes.append((nodes[0], anchor.type.value))
            print(f"      [Slicing] Anchor mapped: L{anchor.line_number} ({anchor.type.value}){loc_tag} → node {nodes[0]}")
        else:
            print(f"      [Slicing] ⚠ Anchor NOT mapped: L{anchor.line_number} ({anchor.type.value}) '{anchor.code_snippet[:50]}'")
    
    # =========================================================================
    # Step 2: Get constraint chain from taxonomy
    # =========================================================================
    category = taxonomy.category_obj
    chain = category.constraint.chain  # List[ChainLink]
    
    # =========================================================================
    # Step 3: ShortestPath Context Extraction (Algorithm 1 L14-17)
    # =========================================================================
    if chain and anchor_nodes:
        # Non-Control-Logic: use ShortestPath to find inter-anchor context
        print(f"      [Slicing] Using ShortestPath context extraction (chain={len(chain)} links)")
        final_nodes = shortest_path_context(pdg_pri, anchor_nodes, chain, slicer_pri, sl_pri)
    elif anchor_nodes:
        # Control-Logic (chain=[]) or Unknown: just return M (anchor nodes only)
        print(f"      [Slicing] Control-Logic/empty chain: returning anchor nodes only")
        final_nodes = {nid for nid, _ in anchor_nodes}
    else:
        # No anchors mapped at all — empty result
        print(f"      [Slicing] ⚠ No anchors mapped to PDG nodes")
        final_nodes = set()
    
    # =========================================================================
    # Step 4: Inject diff line nodes (deleted lines from patch)
    # =========================================================================
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
    
    # Inject diff lines within anchor range
    if diff_lines_abs and anchor_nodes:
        anchor_abs_lines = []
        for nid, _ in anchor_nodes:
            n_rel = pdg_pri.nodes.get(nid, {}).get('start_line', 0)
            if n_rel > 0:
                anchor_abs_lines.append(n_rel - 1 + sl_pri)
        
        if anchor_abs_lines:
            min_anchor = min(anchor_abs_lines)
            max_anchor = max(anchor_abs_lines)
            
            injected_count = 0
            for d_abs in diff_lines_abs:
                if min_anchor <= d_abs <= max_anchor:
                    d_rel = d_abs - sl_pri + 1
                    if d_rel > 0:
                        diff_nodes = slicer_pri.get_nodes_by_location(d_rel, None)
                        final_nodes.update(diff_nodes)
                        if diff_nodes:
                            injected_count += 1
            
            if injected_count > 0:
                print(f"      [Slicing] Injected {injected_count} diff line nodes")
    
    # =========================================================================
    # Step 5: Force add ENTRY node if it's an anchor
    # =========================================================================
    for nid, data in pdg_pri.nodes(data=True):
        if data.get('type') == 'ENTRY':
            entry_line_rel = data.get('start_line', 0)
            if entry_line_rel == 0:
                entry_line_rel = 1
            entry_line_abs = entry_line_rel - 1 + sl_pri
            # Check if ENTRY line matches any anchor
            is_anchor = any(a.line_number == entry_line_abs for a in anchor_result.anchors)
            if is_anchor:
                final_nodes.add(nid)
                print(f"      [Slicing] Including ENTRY node at line {entry_line_abs} (matches anchor)")
    
    # Generate raw slice code
    raw_code = slicer_pri.to_code(final_nodes)
    
    return final_nodes, raw_code


# ==============================================================================
# Main Workflow Entry Function
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
            # Use actual start line number from AtomicPatch
            sl_pri = patch.start_line_old
            sl_shadow = patch.start_line_new
            
            # Ensure valid integers as fallback
            sl_pri = sl_pri if sl_pri is not None else 1
            sl_shadow = sl_shadow if sl_shadow is not None else 1
            
            pdg_builder_pri = PDGBuilder(code_pri, lang=lang)
            pdg_pri = pdg_builder_pri.build(target_line=sl_pri)
            cfg_pri = pdg_builder_pri.cfg  # Expose CFG for TwoPassSlicer
            
            pdg_builder_shadow = PDGBuilder(code_shadow, lang=lang)
            pdg_shadow = pdg_builder_shadow.build(target_line=sl_shadow)
            cfg_shadow = pdg_builder_shadow.cfg  # Expose CFG for TwoPassSlicer
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
        
        # ===== TwoPassSlice: Anchor Candidate Collection (Algorithm 1, L3) =====
        candidates_text = ""
        try:
            two_pass = TwoPassSlicer(
                pdg_pri=pdg_pri,
                cfg_pri=cfg_pri,
                cfg_shadow=cfg_shadow,
                code_pri=code_pri,
                code_shadow=code_shadow,
                sl_pri=sl_pri,
                sl_shadow=sl_shadow,
                search_hints=search_hints,
                patch_diff=patch_diff
            )
            candidates_text = two_pass.collect_candidates()
            if candidates_text:
                print(f"    [TwoPassSlicer] Generated {len(candidates_text.splitlines())} candidate lines")
            else:
                print(f"    [TwoPassSlicer] No candidates generated (will rely on Agent exploration)")
        except Exception as e:
            print(f"    [TwoPassSlicer] Error: {e} (falling back to Agent-only discovery)")
        
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
                attempt=attempt,
                candidates=candidates_text  # TwoPassSlice candidates
            )
            
            print(f"        - Typed Anchors:")
            for a in anchor_result.anchors:
                assume_str = f", assumption={a.assumption_type.value}" if a.assumption_type else ""
                print(f"          L{a.line_number} [{a.type.value}] loc={a.locatability.value}{assume_str}")
            
            # ===== Improvement 2: Validate anchors exist in actual code =====
            print(f"      [Validation] Verifying anchor lines exist in code...")
            code_pri_lines = code_pri.splitlines()
            invalid_anchors = []
            
            for anchor in anchor_result.anchors:
                # Convert absolute line to relative (0-indexed)
                rel_line = anchor.line_number - sl_pri
                
                # Check if line number is valid
                if rel_line < 0 or rel_line >= len(code_pri_lines):
                    invalid_anchors.append(f"Anchor ({anchor.type.value}) line {anchor.line_number} out of range (code has {len(code_pri_lines)} lines, start={sl_pri})")
                    continue
                
                # Check if content matches (fuzzy match: normalized)
                actual_content = code_pri_lines[rel_line].strip()
                anchor_content = anchor.code_snippet.strip()
                
                # Normalize for comparison (remove extra whitespace)
                actual_normalized = ' '.join(actual_content.split())
                anchor_normalized = ' '.join(anchor_content.split())
                
                # Check if anchor content is substring of actual line (allows partial matches)
                if anchor_normalized not in actual_normalized and actual_normalized not in anchor_normalized:
                    invalid_anchors.append(
                        f"Anchor ({anchor.type.value}) line {anchor.line_number}: content mismatch\n"
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
            
            # If slice has not been generated yet, regenerate using the last anchor_result
            if final_pri_nodes is None or raw_code is None:
                if anchor_result is None or not anchor_result.anchors:
                    print(f"        ✗ No valid anchor result available. Skipping function {func}")
                    continue
                
                print(f"      [Fallback] Using last anchor result:")
                print(f"        - Anchors:")
                for a in anchor_result.anchors:
                    assume_str = f", assumption={a.assumption_type.value}" if a.assumption_type else ""
                    print(f"          L{a.line_number} [{a.type.value}] loc={a.locatability.value}{assume_str}")
                
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
        
        # Check if slice is successfully generated
        if final_pri_nodes is None or raw_code is None:
            print(f"    [Error] Failed to generate slice for {func} after all attempts")
            continue
        
        print(f"    [Success] Final slice for {func}:")
        print(f"      - Typed Anchors:")
        for a in anchor_result.anchors:
            assume_str = f", assumption={a.assumption_type.value}" if a.assumption_type else ""
            print(f"          L{a.line_number} [{a.type.value}] loc={a.locatability.value}{assume_str}")
        print(f"      - Raw slice: {len(raw_code.splitlines())} lines")
        print(f'\nRaw slice:\n{raw_code}\n')
        
        # Build unified anchor instructions (no origin/impact split)
        all_anchors = anchor_result.anchors
        anchor_instrs = [SlicingInstruction(
            function_name=func,
            target_version="OLD",
            line_number=a.line_number,
            code_content=a.code_snippet,
            strategy="bidirectional",
            description=f"Anchor ({a.type.value})"
        ) for a in all_anchors]
        
        # ===== §3.3.2.3: Distillation (Agent-driven semantic filtering) =====
        
        # Heuristically determine main focus variable (extract from search_hints)
        main_var = list(search_hints['key_variables'])[0] if search_hints.get('key_variables') else "N/A"

        # print(f"    [Distillation] Applying semantic filtering...")
        validation_result = validator.distill_slice(
            slice_code=raw_code,
            diff_text=patch_diff,
            commit_message=commit_msg,
            func_name=func,
            focus_var=main_var,
            vuln_type=taxonomy.vuln_type,
            hypothesis=taxonomy,
            anchors=anchor_result.anchors,
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
                anchor_only_nodes = _get_strict_nodes(anchor_instrs)
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
                final_kept_nodes.update(_get_strict_nodes(anchor_instrs))
                    
                # Apply Filter: final_pri_nodes becomes the intersection of what we had & what is kept
                final_pri_nodes.intersection_update(final_kept_nodes)
        else:
            print(f"      [Validator] ⚠️  Agent returned no results or error, keeping original slice.")
        
        # [Enforcement] Force add anchor nodes to ensure they exist
        critical_nodes_force = _get_strict_nodes(anchor_instrs)
        
        # [FIX] Force add ENTRY node if it's an anchor (e.g., function definition as Origin)
        for nid, data in pdg_pri.nodes(data=True):
            if data.get('type') == 'ENTRY':
                entry_line_rel = data.get('start_line', 0)
                entry_line_abs = entry_line_rel - 1 + sl_pri if entry_line_rel > 0 else 0
                # Check if ENTRY line matches any anchor
                for a in anchor_result.anchors:
                    if entry_line_abs == a.line_number:
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
        # This should rarely trigger now that get_nodes_by_location checks full line ranges
        anchor_lines_outside_pdg = {}  # Use dict to deduplicate by line number
        for a in anchor_result.anchors:
            # Check if the line is not in slice
            if not any(f"[{a.line_number:4d}]" in line for line in s_pri_text.splitlines()):
                if a.line_number not in anchor_lines_outside_pdg:
                    # Use actual source line content instead of anchor code_snippet
                    a_rel = a.line_number - sl_pri
                    if 0 <= a_rel < len(code_pri.splitlines()):
                        source_content = code_pri.splitlines()[a_rel]
                        anchor_lines_outside_pdg[a.line_number] = source_content
                    else:
                        anchor_lines_outside_pdg[a.line_number] = a.code_snippet
                    print(f"      [Warning] Anchor line {a.line_number} missing from slice (PDG issue?), adding manually")
        
        if anchor_lines_outside_pdg:
            # Merge extra lines into the slice in sorted order (not just prepend)
            import re as _re
            existing_lines = {}
            for line in s_pri_text.splitlines():
                m = _re.match(r'^\[\s*(\d+)\]', line.strip())
                if m:
                    existing_lines[int(m.group(1))] = line
            for ln, content in anchor_lines_outside_pdg.items():
                if ln not in existing_lines:
                    existing_lines[ln] = f"[{ln:4d}] {content}"
            s_pri_text = "\n".join(existing_lines[k] for k in sorted(existing_lines.keys()))
        
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
                a_rel = a.line_number - sl + 1
                
                # [FIX] Find nodes whose line range covers the anchor line.
                # This handles multi-line statements (e.g., compound conditions)
                # where the anchor may reference a continuation line.
                found_nodes = [nid for nid, d in pdg.nodes(data=True)
                              if d.get('start_line', 0) > 0
                              and d.get('start_line', 0) <= a_rel <= d.get('end_line', d.get('start_line', 0))]
                
                node_expanded = False
                if found_nodes:
                    # Find the best matching node (prefer STATEMENT over PREDICATE)
                    best_node = None
                    for nid in found_nodes:
                        d = pdg.nodes[nid]
                        node_type = d.get('type', '')
                        # Skip switch-related control nodes (these could expand to entire blocks)
                        if node_type in ('SWITCH_HEAD', 'CASE_LABEL', 'DEFAULT_LABEL'):
                            expanded_abs.add(a.line_number)
                            node_expanded = True
                            continue
                        # [FIX] For PREDICATE nodes (if/while/for conditions), expand to
                        # full condition line range. Since CFGNode now stores the complete
                        # compound condition range, this correctly covers multi-line conditions
                        # without including the block body.
                        if node_type == 'PREDICATE':
                            n_start = d.get('start_line', 0)
                            n_end = d.get('end_line', n_start)
                            if n_start > 0 and n_end >= n_start:
                                for ln_rel in range(n_start, n_end + 1):
                                    abs_ln = ln_rel - 1 + sl
                                    expanded_abs.add(abs_ln)
                            else:
                                expanded_abs.add(a.line_number)
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
                                expanded_abs.add(a.line_number)
                                node_expanded = True
                
                # If no PDG node found, just add the anchor line directly
                # [FIX] Removed aggressive parenthesis-based expansion which caused contamination
                if not node_expanded:
                    expanded_abs.add(a.line_number)
            
            return sorted(list(expanded_abs))
        
        # Get code lines for source code analysis
        pri_code_lines = code_pri.splitlines()
        
        # Expand ALL typed anchors (unified list instead of origin/impact split)
        pri_anchor_abs = expand_anchor_lines(anchor_result.anchors, pdg_pri, sl_pri, pri_code_lines)

        # Map to Shadow Absolute (Exact + Modified Block Heuristic)
        # [FIX] Ensure these are always lists, never None
        shadow_anchor_abs = []
        
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
                             
                             # B. Strategy: Exact match or context-based match
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
                                 # Get context (prev/next two lines)
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
        
        map_to_shadow(pri_anchor_abs, shadow_anchor_abs)

        # Use Anchor instances directly (zero information loss from AnchorAnalyzer)
        # AnchorAnalyzer.identify() already populates file_path and func_name on each Anchor
        pre_typed_anchors = list(anchor_result.anchors)
        
        # Build post-patch anchors via shadow mapping, inheriting all fields from pre-patch anchor
        post_typed_anchors = []
        for a in anchor_result.anchors:
            shadow_mapped = mapper.map_lines_to_shadow([a.line_number])
            if shadow_mapped:
                # Use first mapped line
                shadow_line = shadow_mapped[0]
                found_lines = get_lines_by_abs_msg(s_shadow_text, [shadow_line])
                shadow_content = found_lines[0] if found_lines else a.code_snippet
                # model_copy preserves all fields (locatability, assumption_type, etc.)
                post_anchor = a.model_copy(update={
                    "line_number": shadow_line,
                    "code_snippet": shadow_content,
                    "reasoning": f"Mapped from pre-patch line {a.line_number}",
                })
                post_typed_anchors.append(post_anchor)
            else:
                print(f"      [SliceFeature] ⚠ Anchor L{a.line_number} [{a.type.value}]: "
                      f"no shadow mapping found, skipping post-patch anchor")
        
        print(f"      [SliceFeature] Typed Anchors:")
        for label, anchor_list in [("Pre", pre_typed_anchors), ("Post", post_typed_anchors)]:
            print(f"        {label}:")
            for a in anchor_list:
                assume_str = f", assumption={a.assumption_type.value}" if a.assumption_type else ""
                print(f"          L{a.line_number} [{a.type.value}] loc={a.locatability.value}{assume_str}")
        
        generated_slices[func] = SliceFeature(
            func_name=func,
            s_pre=final_s_pre,
            s_post=final_s_post,
            pre_anchors=pre_typed_anchors,
            post_anchors=post_typed_anchors,
            validation_status=None,
            validation_reasoning=validation_result.reasoning if validation_result else None
        )

    # Return slices and taxonomy
    return {"slices": generated_slices, "taxonomy": taxonomy}

# ==============================================================================
# Baseline: Agent-Based End-to-End Extraction (For Comparison Experiment)
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
    attack_chain: str
    patch_defense: str
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
        
        # Convert baseline origin/impact indices to Anchor instances
        # Baseline doesn't have real anchor types, use CRITICAL as generic placeholder
        from core.categories import AnchorType as _AnchorType
        _pre_anchors = []
        for idx in f_out.pre_origins + f_out.pre_impacts:
            if p.old_code and 0 <= idx < len(p.old_code.splitlines()):
                _pre_anchors.append(Anchor(
                    type=_AnchorType.CRITICAL,
                    line_number=idx,
                    code_snippet=p.old_code.splitlines()[idx],
                    func_name=f_out.func_name,
                    file_path=p.file_path,
                ))
        
        _post_anchors = []
        for idx in f_out.post_origins + f_out.post_impacts:
            if p.new_code and 0 <= idx < len(p.new_code.splitlines()):
                _post_anchors.append(Anchor(
                    type=_AnchorType.CRITICAL,
                    line_number=idx,
                    code_snippet=p.new_code.splitlines()[idx],
                    func_name=f_out.func_name,
                    file_path=p.file_path,
                ))

        generated_slices[f_out.func_name] = SliceFeature(
            func_name=f_out.func_name,
            s_pre=s_pre,
            s_post=s_post,
            pre_anchors=_pre_anchors,
            post_anchors=_post_anchors,
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
        attack_chain=output.attack_chain,
        patch_defense=output.patch_defense,
        evidence_index={}
    )

    # [Fix] Need to satisfy strict PatchFeatures schema which requires TaxonomyFeature
    # Determine type_confidence based on whether we used metadata
    from core.models import TypeConfidence
    type_confidence = TypeConfidence.HIGH if (metadata and cwe_id_meta) else TypeConfidence.MEDIUM
    
    # Map vuln_type to category_name (required field since §3.1.2 refactor)
    _category_name = vuln_type.value if isinstance(vuln_type, GeneralVulnType) else str(vuln_type)
    # Validate against KB; fall back to "Unknown" if not found
    from core.categories import VulnerabilityKnowledgeBase as _KB
    if _category_name not in _KB.get_all_category_names():
        _category_name = "Unknown"
    
    dummy_taxonomy = TaxonomyFeature(
        category_name=_category_name,
        vuln_type=vuln_type,
        type_confidence=type_confidence,
        cwe_id=cwe_id,
        cwe_name=cwe_name,
        reasoning="Baseline Agent Inference (Skipped Taxonomy Phase)",
        root_cause=output.root_cause,
        attack_chain=output.attack_chain,
        patch_defense=output.patch_defense
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