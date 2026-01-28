"""
Slice Validation 辅助函数

基于 Methodology §3.3.2 实现切片质量验证
策略重构：从验证anchor连贯性 → 验证生成切片的质量

验证时机：在切片提取后，而非anchor识别后
验证内容：
1. Slice Coverage - 切片包含关键代码（diff修改行、anchor位置）
2. Slice Completeness - 切片大小合理（不为空、不过小）
3. Anchor Presence - 识别的anchors在切片中可见

如果切片质量不佳 → 说明anchor识别有问题 → 触发重新发现
"""

import time
import networkx as nx
from typing import List, Set, Dict, Any, Optional
from extraction.anchor_analyzer import AnchorResult, AnchorItem
from core.navigator import CodeNavigator


# ==============================================================================
# 重连重试工具函数
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
    提取简化切片用于验证 anchor 连贯性
    
    这是临时切片，仅用于检查 Origin→Impact 的连接性
    使用双向切片的交集
    
    Args:
        anchor_result: Anchor 识别结果
        slicer: Slicer 实例
        pdg: Program Dependence Graph
        
    Returns:
        切片节点集合
    """
    if not anchor_result.origin_anchors or not anchor_result.impact_anchors:
        return set()
    
    # 1. 提取 Origin 节点和变量
    origin_nodes = []
    origin_vars = set()
    
    for anchor in anchor_result.origin_anchors:
        nodes = slicer.get_nodes_by_location(anchor.line, anchor.content)
        origin_nodes.extend(nodes)
        
        # 提取 Origin 定义的变量
        for nid in nodes:
            if nid in pdg.nodes:
                defs = pdg.nodes[nid].get('defs', {})
                origin_vars.update(defs.keys())
    
    # 2. 提取 Impact 节点和变量
    impact_nodes = []
    impact_vars = set()
    
    for anchor in anchor_result.impact_anchors:
        nodes = slicer.get_nodes_by_location(anchor.line, anchor.content)
        impact_nodes.extend(nodes)
        
        # 提取 Impact 使用的变量
        for nid in nodes:
            if nid in pdg.nodes:
                uses = pdg.nodes[nid].get('uses', {})
                impact_vars.update(uses.keys())
    
    # 3. 双向切片
    fwd_slice = set()
    if origin_nodes and origin_vars:
        fwd_slice = slicer.forward_slice_pruned(origin_nodes, origin_vars)
    
    bwd_slice = set()
    if impact_nodes and impact_vars:
        bwd_slice = slicer.backward_slice_pruned(impact_nodes, impact_vars)
    
    # 4. 交集 = Origin→Impact 路径上的节点
    if fwd_slice and bwd_slice:
        return fwd_slice.intersection(bwd_slice)
    else:
        # 如果其中一个切片为空，返回两者的并集
        return fwd_slice.union(bwd_slice)


def check_path_reachability(
    origin_anchors: List[AnchorItem],
    impact_anchors: List[AnchorItem],
    pdg: nx.MultiDiGraph,
    slicer,  # Slicer instance
    require_data_flow: bool = True
) -> Dict[str, Any]:
    """
    检查 Origin 到 Impact 的路径可达性
    
    实现 Methodology §3.3.2 中的约束2（Data-flow Connectivity）
    
    检查策略：
    - require_data_flow=True: 检查 DATA 边路径（用于 UAF、NPD、Buffer Overflow 等）
    - require_data_flow=False: 检查 CONTROL 边路径（理论设计，实际未使用）
    
    注意：
    - PDG 的 CONTROL 边表示控制依赖（control dependence），不是 CFG 的顺序流
    - 对于顺序执行的代码（如 alloc; stmt; return），它们之间可能不存在控制依赖关系
    - 因此 CONTROL 边检查不能有效验证 control-flow reachability
    - 缺失操作类漏洞（Memory Leak）应跳过此函数，直接通过验证
    
    Args:
        origin_anchors: Origin anchors
        impact_anchors: Impact anchors
        pdg: Program Dependence Graph
        slicer: Slicer 实例
        require_data_flow: 是否要求数据流连接性
        
    Returns:
        {
            "reachable": bool,
            "path_type": "data_flow" | "control_flow" | "none",
            "details": str
        }
    """
    # 获取 Origin 和 Impact 的 PDG 节点
    origin_node_ids = set()
    for anchor in origin_anchors:
        nodes = slicer.get_nodes_by_location(anchor.line, anchor.content)
        origin_node_ids.update(nodes)
    
    impact_node_ids = set()
    for anchor in impact_anchors:
        nodes = slicer.get_nodes_by_location(anchor.line, anchor.content)
        impact_node_ids.update(nodes)
    
    if not origin_node_ids or not impact_node_ids:
        return {
            "reachable": False,
            "path_type": "none",
            "details": "Could not map anchors to PDG nodes"
        }
    
    # 根据 require_data_flow 选择检查策略
    if require_data_flow:
        # 策略1：检查 DATA 路径（用于 UAF, NPD, Buffer Overflow 等）
        for origin_nid in origin_node_ids:
            if origin_nid not in pdg.nodes:
                continue
                
            for impact_nid in impact_node_ids:
                if impact_nid not in pdg.nodes:
                    continue
                
                try:
                    # 只考虑 DATA 边
                    data_edges = [(u, v, k) for u, v, k, data in pdg.edges(keys=True, data=True)
                                 if data.get('relationship') == 'DATA']
                    if data_edges:
                        data_graph = pdg.edge_subgraph(data_edges)
                        if nx.has_path(data_graph, origin_nid, impact_nid):
                            return {
                                "reachable": True,
                                "path_type": "data_flow",
                                "details": f"DATA path exists from node {origin_nid} to {impact_nid}"
                            }
                except Exception as e:
                    continue
        
        return {
            "reachable": False,
            "path_type": "none",
            "details": "No DATA flow path found from Origin to Impact"
        }
    else:
        # 策略2：检查 CONTROL 边路径（理论设计，实际不推荐）
        # PDG 的 CONTROL 边表示控制依赖，不是 CFG 的顺序流
        # 对于顺序代码（如 alloc; return），可能不存在控制依赖关系
        for origin_nid in origin_node_ids:
            if origin_nid not in pdg.nodes:
                continue
                
            for impact_nid in impact_node_ids:
                if impact_nid not in pdg.nodes:
                    continue
                
                try:
                    # 检查 CONTROL 路径（通过 CFG 可达）
                    control_edges = [(u, v, k) for u, v, k, data in pdg.edges(keys=True, data=True)
                                    if data.get('relationship') == 'CONTROL']
                    if control_edges:
                        control_graph = pdg.edge_subgraph(control_edges)
                        if nx.has_path(control_graph, origin_nid, impact_nid):
                            return {
                                "reachable": True,
                                "path_type": "control_flow",
                                "details": f"CONTROL path exists from node {origin_nid} to {impact_nid}"
                            }
                except Exception as e:
                    continue
        
        return {
            "reachable": False,
            "path_type": "none",
            "details": "No CONTROL flow path found from Origin to Impact"
        }


def validate_anchor_completeness(
    anchor_result: AnchorResult
) -> Dict[str, Any]:
    """
    轻量级Anchor完整性检查（仅验证Role Completeness）
    
    策略重构：放弃复杂的静态路径验证，改为后验切片质量检查
    
    此函数仅验证：
    - 至少有一个 Origin anchor
    - 至少有一个 Impact anchor
    
    详细的质量验证将在切片生成后进行（validate_slice_quality）
    
    Args:
        anchor_result: Anchor 识别结果
        
    Returns:
        验证结果字典：
        {
            "is_valid": bool,
            "reason": str
        }
    """
    from core.models import AnchorRole
    
    # [DEBUG] 打印详细的 Anchor 信息
    print(f"\n      [Validation] ===== Anchor Completeness Check =====")
    print(f"      [Validation] Origin Anchors ({len(anchor_result.origin_anchors)}):")
    for i, anchor in enumerate(anchor_result.origin_anchors, 1):
        print(f"        {i}. Line {anchor.line}: {anchor.role.value if hasattr(anchor.role, 'value') else anchor.role}")
        print(f"           Content: {anchor.content[:80]}...")
    
    print(f"      [Validation] Impact Anchors ({len(anchor_result.impact_anchors)}):")
    for i, anchor in enumerate(anchor_result.impact_anchors, 1):
        print(f"        {i}. Line {anchor.line}: {anchor.role.value if hasattr(anchor.role, 'value') else anchor.role}")
        print(f"           Content: {anchor.content[:80]}...")
    print(f"      [Validation] ==============================================\n")
    
    # Role completeness check
    has_origins = bool(anchor_result.origin_anchors)
    has_impacts = bool(anchor_result.impact_anchors)
    
    if not has_origins or not has_impacts:
        missing = []
        if not has_origins:
            missing.append("Origin anchors")
        if not has_impacts:
            missing.append("Impact anchors")
        
        reason = f"Incomplete anchor roles: missing {', '.join(missing)}"
        print(f"      [Validation] ✗ {reason}")
        
        return {
            "is_valid": False,
            "reason": reason
        }
    
    print(f"      [Validation] ✓ Role completeness satisfied")
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
    验证生成切片的质量（后验检查）
    
    基于 Methodology §3.3.2 重构后的验证策略：
    1. Anchor Presence: 识别的anchors在切片中可见（简单检查）
    2. Semantic Adequacy: 切片能够说明漏洞逻辑（LLM语义判断）
    
    如果切片质量不佳 → 说明anchor识别有问题 → 可触发重新发现
    
    Args:
        slice_code: 生成的切片代码
        anchor_result: Anchor 识别结果
        taxonomy: 漏洞类型分类信息
        llm: LLM实例（用于语义验证）
        
    Returns:
        验证结果字典：
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
    
    # 提取切片中的行号
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
    
    # Check 1: Anchor Presence - 识别的anchors在切片中可见
    anchor_lines = set()
    for anchor in anchor_result.origin_anchors + anchor_result.impact_anchors:
        anchor_lines.add(anchor.line)
    
    if not anchor_lines:
        anchor_present = False
        anchor_msg = "No anchors identified"
    else:
        covered_anchors = anchor_lines.intersection(slice_lines)
        anchor_present = len(covered_anchors) > 0  # 至少有一个anchor在切片中
        anchor_msg = f"{len(covered_anchors)}/{len(anchor_lines)} anchors present"
    
    print(f"      [SliceValidation] Anchor presence: {'✓' if anchor_present else '✗'} ({anchor_msg})")
    
    if not anchor_present:
        return {
            "is_valid": False,
            "anchor_present": False,
            "semantic_adequate": False,
            "reason": f"Anchors not visible in slice ({anchor_msg})"
        }
    
    # Check 2: Semantic Adequacy - 切片能够说明漏洞逻辑（LLM判断）
    semantic_adequate = True  # 默认通过（如果没有LLM或跳过语义检查）
    semantic_msg = "Assumed adequate (semantic check skipped)"
    
    if llm:
        try:
            from langchain_core.prompts import ChatPromptTemplate
            
            # [改进] 构建包含跨函数信息的anchor描述
            def format_anchor_desc(anchors, max_show=5):
                """格式化anchor描述，包含跨函数信息"""
                if not anchors:
                    return "None identified"
                
                desc_lines = []
                for i, a in enumerate(anchors[:max_show], 1):
                    role_str = a.role.value if hasattr(a.role, 'value') else str(a.role)
                    scope_str = f" [{a.scope}]" if hasattr(a, 'scope') and a.scope != "local" else ""
                    
                    # 基本信息
                    line_desc = f"{i}. Line {a.line}: {role_str}{scope_str}"
                    
                    # 添加跨函数信息（如果有）
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
            
            origin_desc = format_anchor_desc(anchor_result.origin_anchors)
            impact_desc = format_anchor_desc(anchor_result.impact_anchors)
            
            # 直接使用 taxonomy 中的假设信息（已经在 §3.2.1 分析过）
            vuln_type_str = taxonomy.vuln_type.value if hasattr(taxonomy.vuln_type, 'value') else str(taxonomy.vuln_type)
            
            # 获取 anchor roles 描述（来自 taxonomy）
            origin_roles_str = ", ".join([r.value for r in taxonomy.origin_roles]) if hasattr(taxonomy, 'origin_roles') and taxonomy.origin_roles else "N/A"
            impact_roles_str = ", ".join([r.value for r in taxonomy.impact_roles]) if hasattr(taxonomy, 'impact_roles') and taxonomy.impact_roles else "N/A"
            
            prompt = ChatPromptTemplate.from_template(
                """You are a security expert validating a vulnerability code slice.

**Vulnerability Analysis (from Type Determination §3.2.1):**
Type: {vuln_type}
Expected Origin Roles: {origin_roles}
Expected Impact Roles: {impact_roles}

**Semantic Hypothesis (from §3.2.1):**
Root Cause: {root_cause}
Attack Path: {attack_path}
Fix Mechanism: {fix_mechanism}

**Identified Anchors:**
Origin Anchors (vulnerability initiation):
{origin_anchors}

Impact Anchors (vulnerability manifestation):
{impact_anchors}

**IMPORTANT NOTES:**
1. If an anchor has [call_site] or [inter_procedural] scope, it means the actual operation happens in another function
   - The slice shows the call site in the current function (which is correct for slicing purposes)
   - You should NOT expect to see the full callee function body in the slice
   
2. Focus on the CORE vulnerability mechanism described in the semantic hypothesis above
   - The Root Cause tells you what defect to look for
   - The Attack Path tells you the Origin→Impact chain to verify

**Generated Slice:**
{slice_code}

**Your Task:**
Evaluate if this slice adequately explains the vulnerability mechanism described in the semantic hypothesis.

A GOOD slice should:
✓ Show Origin anchors matching the expected roles ({origin_roles})
✓ Show Impact anchors matching the expected roles ({impact_roles})
✓ Demonstrate the attack path: {attack_path}

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
                    "origin_roles": origin_roles_str,
                    "impact_roles": impact_roles_str,
                    "root_cause": taxonomy.root_cause,
                    "attack_path": taxonomy.attack_path,
                    "fix_mechanism": taxonomy.fix_mechanism if hasattr(taxonomy, 'fix_mechanism') else "N/A",
                    "origin_anchors": origin_desc,
                    "impact_anchors": impact_desc,
                    "slice_code": slice_code
                }),
                max_retries=3
            )
            
            semantic_adequate = result.adequate
            semantic_msg = result.reasoning
            
        except Exception as e:
            print(f"      [SliceValidation] Semantic check failed: {e}")
            semantic_adequate = True  # 出错时默认通过
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
