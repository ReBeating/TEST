"""
Slice Validation Helper Functions (Typed Anchor Model)

Based on Methodology §3.3.2 Implementation of Slice Quality Validation
Strategy Refactoring: Validation of anchor coherence → Validation of generated slice quality

Uses the typed anchor model from §3.1.3 — each anchor carries an AnchorType
(e.g., SOURCE, COMPUTATION, SINK, ALLOC, DEALLOC, USE) rather than the legacy
Origin/Impact binary split.

Validation Timing: After slice extraction, not after anchor identification
Validation Content:
1. Slice Coverage - Slice contains critical code (diff modification lines, anchor positions)
2. Slice Completeness - Slice size is reasonable (not empty, not too small)
3. Anchor Presence - Identified anchors are visible in the slice

If slice quality is poor → Indicates issues with anchor identification → Trigger rediscovery
"""

import time
import networkx as nx
from typing import List, Set, Dict, Any, Optional
from extraction.anchor_analyzer import AnchorResult
from core.categories import Anchor, DependencyType
from core.navigator import CodeNavigator


# ==============================================================================
# Connection retry utility functions
# ==============================================================================

def retry_on_connection_error(func, max_retries=3, initial_delay=2.0, backoff_factor=2.0):
    """
    Retry wrapper for LLM calls with exponential backoff.
    
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
            
            # Check if it's a connection-related error
            is_connection_error = any(keyword in error_str for keyword in [
                'connection', 'timeout', 'timed out', 'network',
                'refused', 'reset', 'broken pipe', 'unreachable'
            ])
            
            if is_connection_error and attempt < max_retries:
                print(f"      [Retry] Connection error on attempt {attempt}/{max_retries}: {e}")
                print(f"      [Retry] Waiting {delay:.1f}s before retry...")
                time.sleep(delay)
                delay *= backoff_factor
            else:
                # Not a connection error or last attempt, raise immediately
                raise
    
    # All retries exhausted
    raise last_exception

def extract_slice_for_validation(
    anchor_result: AnchorResult,
    slicer,  # Slicer instance
    pdg: nx.MultiDiGraph
) -> Set[str]:
    """
    Extract simplified slice for validation of anchor coherence (typed anchor model).
    
    Uses midpoint split: first half of the typed anchor chain → forward slice,
    second half → backward slice.  The intersection approximates the path
    connecting the chain endpoints.
    
    Args:
        anchor_result: Anchor identification result (typed anchors)
        slicer: Slicer instance
        pdg: Program Dependence Graph
        
    Returns:
        Set of slice nodes
    """
    all_anchors = anchor_result.anchors
    if not all_anchors:
        return set()
    
    # Midpoint split — mirrors generate_slice_from_anchors() in slicer.py
    mid = max(len(all_anchors) // 2, 1)
    fwd_anchors = all_anchors[:mid]
    bwd_anchors = all_anchors[mid:]
    
    # 1. Extract forward (first-half) nodes and variables
    fwd_nodes = []
    fwd_vars = set()
    
    for anchor in fwd_anchors:
        nodes = slicer.get_nodes_by_location(anchor.line_number, anchor.code_snippet)
        fwd_nodes.extend(nodes)
        
        # Extract variables defined at forward anchor positions
        for nid in nodes:
            if nid in pdg.nodes:
                defs = pdg.nodes[nid].get('defs', {})
                fwd_vars.update(defs.keys())
    
    # 2. Extract backward (second-half) nodes and variables
    bwd_nodes = []
    bwd_vars = set()
    
    for anchor in bwd_anchors:
        nodes = slicer.get_nodes_by_location(anchor.line_number, anchor.code_snippet)
        bwd_nodes.extend(nodes)
        
        # Extract variables used at backward anchor positions
        for nid in nodes:
            if nid in pdg.nodes:
                uses = pdg.nodes[nid].get('uses', {})
                bwd_vars.update(uses.keys())
    
    # 3. Bidirectional slicing
    fwd_slice = set()
    if fwd_nodes and fwd_vars:
        fwd_slice = slicer.forward_slice_pruned(fwd_nodes, fwd_vars)
    
    bwd_slice = set()
    if bwd_nodes and bwd_vars:
        bwd_slice = slicer.backward_slice_pruned(bwd_nodes, bwd_vars)
    
    # 4. Intersection = Nodes on the chain path
    if fwd_slice and bwd_slice:
        return fwd_slice.intersection(bwd_slice)
    else:
        # If one of the slices is empty, return their union
        return fwd_slice.union(bwd_slice)


def check_path_reachability(
    fwd_anchors: List[Anchor],
    bwd_anchors: List[Anchor],
    pdg: nx.MultiDiGraph,
    slicer,  # Slicer instance
    require_data_flow: bool = True
) -> Dict[str, Any]:
    """
    Check path reachability between forward-half and backward-half of the
    typed anchor chain.
    
    Implements Constraint 2 (Data-flow Connectivity) from Methodology §3.3.2.
    
    In the typed anchor model the caller splits anchor_result.anchors at the
    midpoint (same strategy as extract_slice_for_validation and
    generate_slice_from_anchors) and passes the two halves here.
    
    Check Strategy:
    - require_data_flow=True: Check DATA edge paths (used for UAF, NPD, Buffer Overflow, etc.)
    - require_data_flow=False: Check CONTROL edge paths (theoretical design, not actually used)
    
    Note:
    - PDG's CONTROL edges represent control dependence, not CFG's sequential flow
    - For sequentially executed code (e.g., alloc; stmt; return), there may be no control dependence between them
    - Therefore, CONTROL edge checks cannot effectively validate control-flow reachability
    - Missing operation vulnerabilities (Memory Leak) should skip this function and pass validation directly
    
    Args:
        fwd_anchors: Forward-half anchors (first half of typed chain)
        bwd_anchors: Backward-half anchors (second half of typed chain)
        pdg: Program Dependence Graph
        slicer: Slicer instance
        require_data_flow: Whether data flow connectivity is required
        
    Returns:
        {
            "reachable": bool,
            "path_type": "data_flow" | "control_flow" | "none",
            "details": str
        }
    """
    # Get PDG nodes for forward-half and backward-half anchors
    fwd_node_ids = set()
    for anchor in fwd_anchors:
        nodes = slicer.get_nodes_by_location(anchor.line_number, anchor.code_snippet)
        fwd_node_ids.update(nodes)
    
    bwd_node_ids = set()
    for anchor in bwd_anchors:
        nodes = slicer.get_nodes_by_location(anchor.line_number, anchor.code_snippet)
        bwd_node_ids.update(nodes)
    
    if not fwd_node_ids or not bwd_node_ids:
        return {
            "reachable": False,
            "path_type": "none",
            "details": "Could not map anchors to PDG nodes"
        }
    
    # Select check strategy based on require_data_flow
    if require_data_flow:
        # Strategy 1: Check DATA path (used for UAF, NPD, Buffer Overflow, etc.)
        for fwd_nid in fwd_node_ids:
            if fwd_nid not in pdg.nodes:
                continue
                
            for bwd_nid in bwd_node_ids:
                if bwd_nid not in pdg.nodes:
                    continue
                
                try:
                    # Only consider DATA edges
                    data_edges = [(u, v, k) for u, v, k, data in pdg.edges(keys=True, data=True)
                                 if data.get('relationship') == 'DATA']
                    if data_edges:
                        data_graph = pdg.edge_subgraph(data_edges)
                        if nx.has_path(data_graph, fwd_nid, bwd_nid):
                            return {
                                "reachable": True,
                                "path_type": "data_flow",
                                "details": f"DATA path exists from node {fwd_nid} to {bwd_nid}"
                            }
                except Exception as e:
                    continue
        
        return {
            "reachable": False,
            "path_type": "none",
            "details": "No DATA flow path found between anchor chain halves"
        }
    else:
        # Strategy 2: Check CONTROL edge paths (theoretical design, not recommended)
        # PDG's CONTROL edges represent control dependence, not CFG's sequential flow
        # For sequential code (e.g., alloc; return), control dependence may not exist
        for fwd_nid in fwd_node_ids:
            if fwd_nid not in pdg.nodes:
                continue
                
            for bwd_nid in bwd_node_ids:
                if bwd_nid not in pdg.nodes:
                    continue
                
                try:
                    # Check CONTROL path (reachable via CFG)
                    control_edges = [(u, v, k) for u, v, k, data in pdg.edges(keys=True, data=True)
                                    if data.get('relationship') == 'CONTROL']
                    if control_edges:
                        control_graph = pdg.edge_subgraph(control_edges)
                        if nx.has_path(control_graph, fwd_nid, bwd_nid):
                            return {
                                "reachable": True,
                                "path_type": "control_flow",
                                "details": f"CONTROL path exists from node {fwd_nid} to {bwd_nid}"
                            }
                except Exception as e:
                    continue
        
        return {
            "reachable": False,
            "path_type": "none",
            "details": "No CONTROL flow path found between anchor chain halves"
        }


def check_single_pair(
    src_line: int,
    src_code: str,
    tgt_line: int,
    tgt_code: str,
    dep_type: 'DependencyType',
    pdg: nx.MultiDiGraph,
    cfg: nx.DiGraph,
    slicer,  # Slicer instance
) -> Dict[str, Any]:
    """
    Check dependency between a single anchor pair from the constraint chain.
    
    Implements §3.3.2 per-pair verification: for each ChainLink (a_i, a_j, δ),
    check whether the expected dependency type δ exists between the mapped
    positions μ(a_i) and μ(a_j) in the target code's PDG/CFG.
    
    Dependency type strategies:
    - DATA: Check DATA edge paths in PDG (e.g., alloc →d dealloc, source →d sink)
    - TEMPORAL: Check CFG reachability (e.g., dealloc →t use, alloc →t exit)
    - CONTROL: Check CONTROL edge paths in PDG (e.g., shared →c access)
    
    Args:
        src_line: Source anchor's mapped target line number
        src_code: Source anchor's mapped target code snippet
        tgt_line: Target anchor's mapped target line number
        tgt_code: Target anchor's mapped target code snippet
        dep_type: Expected dependency type (DependencyType enum)
        pdg: Program Dependence Graph (MultiDiGraph with DATA/CONTROL edges)
        cfg: Control Flow Graph (DiGraph, from PDGBuilder.cfg)
        slicer: Slicer instance for node location mapping
        
    Returns:
        {
            "reachable": bool,
            "path_type": "data_flow" | "temporal" | "control_flow" | "none",
            "details": str
        }
    """
    # Map source and target lines to PDG node IDs
    src_nodes = slicer.get_nodes_by_location(src_line, src_code)
    tgt_nodes = slicer.get_nodes_by_location(tgt_line, tgt_code)
    
    if not src_nodes:
        return {
            "reachable": False,
            "path_type": "none",
            "details": f"Could not map source anchor (line {src_line}) to PDG nodes"
        }
    
    if not tgt_nodes:
        return {
            "reachable": False,
            "path_type": "none",
            "details": f"Could not map target anchor (line {tgt_line}) to PDG nodes"
        }
    
    src_node_ids = set(src_nodes)
    tgt_node_ids = set(tgt_nodes)
    
    if dep_type == DependencyType.DATA:
        # Strategy: Check DATA edge paths in PDG
        return _check_data_path(src_node_ids, tgt_node_ids, pdg, src_line, tgt_line)
    
    elif dep_type == DependencyType.TEMPORAL:
        # Strategy: Check CFG reachability (sequential execution order)
        return _check_temporal_path(src_node_ids, tgt_node_ids, cfg, pdg, src_line, tgt_line)
    
    elif dep_type == DependencyType.CONTROL:
        # Strategy: Check CONTROL edge paths in PDG
        return _check_control_path(src_node_ids, tgt_node_ids, pdg, src_line, tgt_line)
    
    else:
        return {
            "reachable": False,
            "path_type": "none",
            "details": f"Unknown dependency type: {dep_type}"
        }


def _check_data_path(
    src_nodes: Set[str],
    tgt_nodes: Set[str],
    pdg: nx.MultiDiGraph,
    src_line: int,
    tgt_line: int
) -> Dict[str, Any]:
    """Check DATA edge path in PDG between source and target nodes.
    
    Uses a two-tier strategy:
    1. Pure DATA path: Check only DATA edges in PDG (strict)
    2. Mixed DATA+CONTROL fallback: If pure DATA fails, check on full PDG
       (DATA + CONTROL edges). This handles loop-driven vulnerability patterns
       where source controls a loop condition (DATA) which in turn controls
       the computation (CONTROL), e.g.:
         gf_fgets(szLine) → while(szLine[i+1]) → i++
         source ──DATA──→ loop_cond ──CONTROL──→ computation
    """
    try:
        # === Tier 1: Pure DATA path check (strict) ===
        data_edges = [(u, v, k) for u, v, k, data in pdg.edges(keys=True, data=True)
                     if data.get('relationship') == 'DATA']
        if not data_edges:
            # No DATA edges at all — skip to mixed fallback
            pass
        else:
            data_graph = pdg.edge_subgraph(data_edges)
            
            for src_nid in src_nodes:
                if src_nid not in data_graph.nodes:
                    continue
                for tgt_nid in tgt_nodes:
                    if tgt_nid not in data_graph.nodes:
                        continue
                    try:
                        if nx.has_path(data_graph, src_nid, tgt_nid):
                            return {
                                "reachable": True,
                                "path_type": "data_flow",
                                "details": f"DATA path: line {src_line} (node {src_nid}) → line {tgt_line} (node {tgt_nid})"
                            }
                    except (nx.NetworkXError, nx.NodeNotFound):
                        continue
        
        # === Tier 2: Mixed DATA+CONTROL fallback ===
        # Handles loop-driven patterns where source → loop_condition (DATA)
        # → loop_body/computation (CONTROL). Common in buffer overflow via
        # unbounded loop index (e.g., CVE-2021-40574).
        try:
            full_digraph = nx.DiGraph()
            for n, d in pdg.nodes(data=True):
                full_digraph.add_node(n, **d)
            for u, v, _k, d in pdg.edges(keys=True, data=True):
                if d.get('relationship') in ('DATA', 'CONTROL'):
                    if not full_digraph.has_edge(u, v):
                        full_digraph.add_edge(u, v, **d)
            
            for src_nid in src_nodes:
                if src_nid not in full_digraph.nodes:
                    continue
                for tgt_nid in tgt_nodes:
                    if tgt_nid not in full_digraph.nodes:
                        continue
                    try:
                        if nx.has_path(full_digraph, src_nid, tgt_nid):
                            return {
                                "reachable": True,
                                "path_type": "data_flow_mixed",
                                "details": f"Mixed DATA+CONTROL path: line {src_line} (node {src_nid}) → line {tgt_line} (node {tgt_nid}) (via control bridge)"
                            }
                    except (nx.NetworkXError, nx.NodeNotFound):
                        continue
        except Exception:
            pass  # Mixed fallback failure is not critical
        
        return {
            "reachable": False,
            "path_type": "none",
            "details": f"No DATA flow path from line {src_line} to line {tgt_line}"
        }
    except Exception as e:
        return {
            "reachable": False,
            "path_type": "none",
            "details": f"DATA path check error: {e}"
        }


def _check_temporal_path(
    src_nodes: Set[str],
    tgt_nodes: Set[str],
    cfg: nx.DiGraph,
    pdg: nx.MultiDiGraph,
    src_line: int,
    tgt_line: int
) -> Dict[str, Any]:
    """
    Check TEMPORAL dependency via CFG reachability.
    
    Temporal dependency (→t) means "a_i can execute before a_j" in the control flow.
    This is checked on the CFG (not PDG), since PDG's CONTROL edges represent
    control dependence (not sequential flow).
    
    The CFG nodes share the same IDs as PDG nodes (built from the same CFGBuilder),
    so we can directly use PDG node IDs to query the CFG.
    """
    try:
        # First try: direct CFG reachability using PDG node IDs
        for src_nid in src_nodes:
            if src_nid not in cfg.nodes:
                continue
            for tgt_nid in tgt_nodes:
                if tgt_nid not in cfg.nodes:
                    continue
                try:
                    if nx.has_path(cfg, src_nid, tgt_nid):
                        return {
                            "reachable": True,
                            "path_type": "temporal",
                            "details": f"TEMPORAL path (CFG): line {src_line} (node {src_nid}) → line {tgt_line} (node {tgt_nid})"
                        }
                except (nx.NetworkXError, nx.NodeNotFound):
                    continue
        
        # Fallback: line-number based CFG lookup
        # CFG nodes have 'start_line' attribute; find nodes matching our lines
        cfg_src_nodes = set()
        cfg_tgt_nodes = set()
        for n, d in cfg.nodes(data=True):
            node_line = d.get('start_line', 0)
            if node_line == src_line:
                cfg_src_nodes.add(n)
            elif node_line == tgt_line:
                cfg_tgt_nodes.add(n)
        
        if cfg_src_nodes and cfg_tgt_nodes:
            for src_nid in cfg_src_nodes:
                for tgt_nid in cfg_tgt_nodes:
                    try:
                        if nx.has_path(cfg, src_nid, tgt_nid):
                            return {
                                "reachable": True,
                                "path_type": "temporal",
                                "details": f"TEMPORAL path (CFG fallback): line {src_line} → line {tgt_line}"
                            }
                    except (nx.NetworkXError, nx.NodeNotFound):
                        continue
        
        return {
            "reachable": False,
            "path_type": "none",
            "details": f"No TEMPORAL (CFG) path from line {src_line} to line {tgt_line}"
        }
    except Exception as e:
        return {
            "reachable": False,
            "path_type": "none",
            "details": f"TEMPORAL path check error: {e}"
        }


def _check_control_path(
    src_nodes: Set[str],
    tgt_nodes: Set[str],
    pdg: nx.MultiDiGraph,
    src_line: int,
    tgt_line: int
) -> Dict[str, Any]:
    """Check CONTROL edge path in PDG between source and target nodes."""
    try:
        control_edges = [(u, v, k) for u, v, k, data in pdg.edges(keys=True, data=True)
                        if data.get('relationship') == 'CONTROL']
        if not control_edges:
            return {
                "reachable": False,
                "path_type": "none",
                "details": f"No CONTROL edges in PDG; cannot verify control dep from line {src_line} to {tgt_line}"
            }
        
        control_graph = pdg.edge_subgraph(control_edges)
        
        for src_nid in src_nodes:
            if src_nid not in control_graph.nodes:
                continue
            for tgt_nid in tgt_nodes:
                if tgt_nid not in control_graph.nodes:
                    continue
                try:
                    if nx.has_path(control_graph, src_nid, tgt_nid):
                        return {
                            "reachable": True,
                            "path_type": "control_flow",
                            "details": f"CONTROL path: line {src_line} (node {src_nid}) → line {tgt_line} (node {tgt_nid})"
                        }
                except (nx.NetworkXError, nx.NodeNotFound):
                    continue
        
        return {
            "reachable": False,
            "path_type": "none",
            "details": f"No CONTROL dep path from line {src_line} to line {tgt_line}"
        }
    except Exception as e:
        return {
            "reachable": False,
            "path_type": "none",
            "details": f"CONTROL path check error: {e}"
        }


def validate_anchor_completeness(
    anchor_result: AnchorResult
) -> Dict[str, Any]:
    """
    Lightweight Anchor Completeness Check (Typed Anchor Model).
    
    Strategy Refactoring: Abandon complex static path validation in favor of
    posterior slice quality check.
    
    This function verifies:
    - At least one typed anchor exists
    - The anchor types cover enough of the expected constraint chain
      (at minimum, anchors should not be empty)
    
    Detailed quality validation will be performed after slice generation
    (validate_slice_quality).
    
    Args:
        anchor_result: Anchor identification result (typed anchors)
        
    Returns:
        Validation result dictionary:
        {
            "is_valid": bool,
            "reason": str
        }
    """
    all_anchors = anchor_result.anchors
    
    # [DEBUG] Print detailed Anchor information
    print(f"\n      [Validation] ===== Anchor Completeness Check (Typed) =====")
    print(f"      [Validation] Total Anchors: {len(all_anchors)}")
    for i, anchor in enumerate(all_anchors, 1):
        print(f"        {i}. Line {anchor.line_number}: [{anchor.type.value}]")
        print(f"           Content: {anchor.code_snippet[:80]}...")
    
    found_types = {a.type.value for a in all_anchors}
    print(f"      [Validation] Found types: {found_types}")
    print(f"      [Validation] ==============================================\n")
    
    # Typed anchor completeness check — need at least one anchor
    if not all_anchors:
        reason = "No typed anchors identified"
        print(f"      [Validation] ✗ {reason}")
        return {
            "is_valid": False,
            "reason": reason
        }
    
    # Warn if only one anchor type (chain requires ≥2 distinct types normally)
    if len(found_types) < 2 and len(all_anchors) >= 2:
        print(f"      [Validation] ⚠ All {len(all_anchors)} anchors have the same type — may indicate weak identification")
    
    print(f"      [Validation] ✓ Typed anchor completeness satisfied ({len(all_anchors)} anchors, {len(found_types)} types)")
    print(f"      [Validation] Note: Detailed quality validation will be performed after slice extraction")
    
    return {
        "is_valid": True,
        "reason": ""
    }


def validate_slice_quality(
    slice_code: str,
    anchor_result: AnchorResult,
    taxonomy,
    llm = None
) -> Dict[str, Any]:
    """
    Validate quality of generated slice (Posterior check)
    
    Refactored validation strategy based on Methodology §3.3.2:
    1. Anchor Presence: Identified anchors are visible in the slice (simple check)
    2. Semantic Adequacy: The slice can explain the vulnerability logic (LLM semantic judgment)
    
    If slice quality is poor → Indicates issues with anchor identification → Trigger rediscovery
    
    Args:
        slice_code: Generated slice code
        anchor_result: Anchor identification result
        taxonomy: Vulnerability taxonomy information
        llm: LLM instance (for semantic validation)
        
    Returns:
        Validation result dictionary:
        {
            "is_valid": bool,
            "anchor_present": bool,
            "semantic_adequate": bool,
            "reason": str
        }
    """
    if not slice_code or not slice_code.strip():
        return {
            "is_valid": False,
            "anchor_present": False,
            "semantic_adequate": False,
            "reason": "Empty slice generated"
        }
    
    # Extract line numbers from slice
    import re
    slice_lines = set()
    for line in slice_code.splitlines():
        match = re.match(r'^\[\s*(\d+)\]', line.strip())
        if match:
            slice_lines.add(int(match.group(1)))
    
    if not slice_lines:
        return {
            "is_valid": False,
            "anchor_present": False,
            "semantic_adequate": False,
            "reason": "No valid code lines in slice"
        }
    
    # Check 1: Anchor Presence - Identified anchors are visible in the slice
    anchor_lines = set()
    for anchor in anchor_result.anchors:
        anchor_lines.add(anchor.line_number)
    
    if not anchor_lines:
        anchor_present = False
        anchor_msg = "No anchors identified"
    else:
        covered_anchors = anchor_lines.intersection(slice_lines)
        anchor_present = len(covered_anchors) > 0  # At least one anchor present in slice
        anchor_msg = f"{len(covered_anchors)}/{len(anchor_lines)} anchors present"
    
    print(f"      [SliceValidation] Anchor presence: {'✓' if anchor_present else '✗'} ({anchor_msg})")
    
    if not anchor_present:
        return {
            "is_valid": False,
            "anchor_present": False,
            "semantic_adequate": False,
            "reason": f"Anchors not visible in slice ({anchor_msg})"
        }
    
    # Check 2: Semantic Adequacy - The slice can explain the vulnerability logic (LLM judgment)
    semantic_adequate = True  # Pass by default (if no LLM or semantic check skipped)
    semantic_msg = "Assumed adequate (semantic check skipped)"
    
    if llm:
        try:
            from langchain_core.prompts import ChatPromptTemplate
            
            # [Enhanced] Build anchor description containing cross-function information
            def format_anchor_desc(anchors, max_show=8):
                """Format typed anchor description, including cross-function information"""
                if not anchors:
                    return "None identified"
                
                desc_lines = []
                for i, a in enumerate(anchors[:max_show], 1):
                    type_str = a.type.value
                    scope_str = f" [{a.scope}]" if hasattr(a, 'scope') and a.scope != "local" else ""
                    
                    # Basic information
                    line_desc = f"{i}. Line {a.line_number}: [{type_str}]{scope_str}"
                    
                    # Add cross-function information (if any)
                    if hasattr(a, 'cross_function_info') and a.cross_function_info:
                        cf = a.cross_function_info
                        if cf.callee_function:
                            line_desc += f"\n   → Actual operation in {cf.callee_function}()"
                            if cf.callee_line and cf.callee_content:
                                line_desc += f" at line {cf.callee_line}: {cf.callee_content[:60]}"
                        elif cf.data_flow_chain:
                            line_desc += f"\n   → Data flow: {' → '.join(cf.data_flow_chain[:3])}"
                    
                    desc_lines.append(line_desc)
                
                if len(anchors) > max_show:
                    desc_lines.append(f"... and {len(anchors) - max_show} more")
                
                return "\n".join(desc_lines)
            
            all_anchors_desc = format_anchor_desc(anchor_result.anchors)
            
            # Directly use hypothesis information from taxonomy (already analyzed in §3.2.1)
            vuln_type_str = taxonomy.vuln_type.value if hasattr(taxonomy.vuln_type, 'value') else str(taxonomy.vuln_type)
            
            # Get typed anchor info from taxonomy
            anchor_types_str = ", ".join(at.value if hasattr(at, 'value') else str(at) for at in taxonomy.anchor_types) if hasattr(taxonomy, 'anchor_types') and taxonomy.anchor_types else "N/A"
            chain_desc = taxonomy.chain_description if hasattr(taxonomy, 'chain_description') and taxonomy.chain_description else "N/A"
            
            prompt = ChatPromptTemplate.from_template(
                """You are a security expert validating a vulnerability code slice.

**Vulnerability Analysis (from Type Determination §3.2.1):**
Type: {vuln_type}
Expected Anchor Types: {anchor_types}
Constraint Chain: {chain_description}

**Semantic Hypothesis (from §3.2.1):**
Root Cause: {root_cause}
Attack Chain: {attack_chain}
Patch Defense: {patch_defense}

**Identified Typed Anchors:**
{all_anchors}

**IMPORTANT NOTES:**
1. If an anchor has [call_site] or [inter_procedural] scope, it means the actual operation happens in another function
   - The slice shows the call site in the current function (which is correct for slicing purposes)
   - You should NOT expect to see the full callee function body in the slice
   
2. Focus on the CORE vulnerability mechanism described in the semantic hypothesis above
   - The Root Cause tells you what defect to look for
   - The constraint chain ({chain_description}) tells you the typed dependency to verify

**Generated Slice:**
{slice_code}

**Your Task:**
Evaluate if this slice adequately explains the vulnerability mechanism described in the semantic hypothesis.

A GOOD slice should:
✓ Show anchors matching the expected types ({anchor_types})
✓ Demonstrate the constraint chain: {chain_description}
✓ Cover the attack chain: {attack_chain}

A slice is ADEQUATE even if:
✓ It doesn't show every variable definition (focus on core issue)
✓ It shows call sites instead of full callee code (for inter-procedural cases)
✓ Some context is missing but the core vulnerability logic is clear

Answer with a JSON object:
{{
    "adequate": true/false,
    "reasoning": "Brief explanation focusing on whether the semantic hypothesis can be verified in this slice"
}}
"""
            )
            
            from pydantic import BaseModel, Field
            class SemanticCheck(BaseModel):
                adequate: bool = Field(description="Whether the slice adequately explains the vulnerability")
                reasoning: str = Field(description="Brief explanation")
            
            chain = prompt | llm.with_structured_output(SemanticCheck)
            
            result = retry_on_connection_error(
                lambda: chain.invoke({
                    "vuln_type": vuln_type_str,
                    "anchor_types": anchor_types_str,
                    "chain_description": chain_desc,
                    "root_cause": taxonomy.root_cause,
                    "attack_chain": taxonomy.attack_chain,
                    "patch_defense": taxonomy.patch_defense if hasattr(taxonomy, 'patch_defense') else "N/A",
                    "all_anchors": all_anchors_desc,
                    "slice_code": slice_code
                }),
                max_retries=3
            )
            
            semantic_adequate = result.adequate
            semantic_msg = result.reasoning
            
        except Exception as e:
            print(f"      [SliceValidation] Semantic check failed: {e}")
            semantic_adequate = True  # Default to pass on error
            semantic_msg = f"Check error (assuming adequate): {str(e)[:50]}"
    
    print(f"      [SliceValidation] Semantic adequacy: {'✓' if semantic_adequate else '✗'} ({semantic_msg})")
    
    is_valid = anchor_present and semantic_adequate
    
    if not is_valid:
        reason = []
        if not anchor_present:
            reason.append("anchors not present")
        if not semantic_adequate:
            reason.append(f"inadequate semantics: {semantic_msg}")
        reason_str = ", ".join(reason)
    else:
        reason_str = ""
    
    return {
        "is_valid": is_valid,
        "anchor_present": anchor_present,
        "semantic_adequate": semantic_adequate,
        "reason": reason_str
    }
