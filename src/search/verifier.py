import os
import re
import json
import time
from typing import List, Dict, Any, Optional, Set
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# Import state and models
from core.state import VerificationState
from core.models import VulnerabilityFinding, SearchResultItem, PatchFeatures
from core.navigator import CodeNavigator
from core.indexer import BenchmarkSymbolIndexer, GlobalSymbolIndexer
from core.categories import Anchor, AnchorType, AnchorLocatability, AssumptionType, DependencyType, ChainLink

from extraction.slicer import Slicer
from extraction.pdg import PDGBuilder
from extraction.slice_validation import check_path_reachability, check_single_pair

# Import type-specific verification checklists
from search.verification_checklists import format_checklist_for_prompt


def format_anchor_for_prompt(anchor: Anchor) -> str:
    """Format a typed Anchor object for LLM prompt display.
    
    Produces a compact one-liner such as:
        [ALLOC|CONCRETE] Line 227 `ptr = kmalloc(...)` (vars: ptr)
        [DEALLOC|ASSUMED:EXISTENCE] `kfree(ptr)` — assumed to exist in callee
    """
    atype = anchor.type.value if hasattr(anchor.type, 'value') else str(anchor.type)
    loc = anchor.locatability.value if hasattr(anchor.locatability, 'value') else str(anchor.locatability)
    
    # Build tag: [TYPE|LOCATABILITY] or [TYPE|LOCATABILITY:ASSUMPTION]
    tag = f"{atype}|{loc}"
    if anchor.assumption_type:
        at = anchor.assumption_type.value if hasattr(anchor.assumption_type, 'value') else str(anchor.assumption_type)
        tag += f":{at}"
    parts = [f"[{tag}]"]
    
    if anchor.line_number:
        parts.append(f"Line {anchor.line_number}")
    if anchor.code_snippet:
        parts.append(f"`{anchor.code_snippet.strip()}`")
    if anchor.variable_names:
        parts.append(f"(vars: {', '.join(anchor.variable_names)})")
    if anchor.is_optional:
        parts.append("(optional)")
    if anchor.assumption_rationale:
        parts.append(f"— {anchor.assumption_rationale}")
    return " ".join(parts)

class StepAnalysis(BaseModel):
    """Evidence item for vulnerability analysis — typed anchor chain model.
    
    The 'role' field uses AnchorType values (e.g., 'ALLOC', 'DEALLOC', 'USE')
    or flow roles ('Trace', 'Defense') instead of legacy 'Origin'/'Impact'.
    """
    role: str = Field(description="AnchorType value (e.g. 'ALLOC','DEALLOC','USE','SOURCE','SINK') or flow role ('Trace','Defense')")
    anchor_type: Optional[str] = Field(default=None, description="Explicit AnchorType if this step corresponds to a chain anchor")
    file_path: Optional[str] = Field(default=None, description="File path where this step occurs")
    func_name: Optional[str] = Field(default=None, description="Function name containing this step")
    line_number: Optional[int] = Field(default=None, description="Line number in target file")
    code_content: Optional[str] = Field(default=None, description="Code snippet")
    observation: str = Field(description="Brief analysis of this step")


class AnchorMapping(BaseModel):
    """Typed anchor mapping result — maps a Phase 2 Anchor to a target location.
    
    Implements §3.3.1 Anchor Mapping: LLM examines aligned lines from matching phase,
    determines if they fulfill the same semantic role as the original anchors.
    Guided by root cause description for functional equivalence (not syntactic identity).
    
    Locatability-aware mapping strategy:
    - CONCRETE: Direct alignment check via Phase 3 traces; verify semantic role
    - ASSUMED:  Phase 3 hint + LLM semantic verification of assumption
    - CONCEPTUAL: LLM must discover/infer from target code context
    """
    anchor_type: str = Field(description="AnchorType value, e.g. 'alloc', 'dealloc', 'use', 'source', 'sink', etc.")
    is_optional: bool = Field(default=False, description="Whether this anchor is optional in the constraint chain")
    chain_position: Optional[str] = Field(default=None, description="Position in constraint chain, e.g. 'first', 'middle', 'last'")
    
    # --- Reference info (from Phase 2 Anchor) ---
    reference_line: str = Field(description="Reference slice anchor line or code snippet")
    reference_line_number: Optional[int] = Field(default=None, description="Reference anchor line number (from Anchor.line_number)")
    locatability: str = Field(default="concrete", description="Anchor locatability: 'concrete', 'assumed', or 'conceptual'")
    assumption_type: Optional[str] = Field(default=None, description="If locatability is 'assumed'/'conceptual': 'controllability', 'semantic', 'existence', or 'reachability'")
    assumption_rationale: Optional[str] = Field(default=None, description="Original assumption rationale from Phase 2")
    
    # --- Target mapping result ---
    target_line: Optional[int] = Field(default=None, description="Mapped target line number")
    target_code: Optional[str] = Field(default=None, description="Mapped target code")
    is_mapped: bool = Field(description="Whether mapping succeeded")
    mapping_confidence: str = Field(default="high", description="Mapping confidence: 'high' (direct match), 'medium' (semantic equivalent), 'low' (inferred)")
    semantic_role_verified: bool = Field(default=False, description="Whether LLM verified the target fulfills the same semantic role")
    
    # --- Assumption verification (for ASSUMED/CONCEPTUAL anchors) ---
    assumption_verified: Optional[bool] = Field(default=None, description="Whether the original assumption was verified in target context (only for assumed/conceptual)")
    assumption_verification_note: Optional[str] = Field(default=None, description="How the assumption was verified or why it failed")
    
    reason: Optional[str] = Field(default=None, description="Semantic equivalence explanation or failure reason")


class JudgeOutput(BaseModel):
    """Final Judge Output — typed anchor chain model (§3.3.3)."""
    c_cons_satisfied: bool = Field(description="Consistency constraint satisfied (§3.3.1)")
    c_reach_satisfied: bool = Field(description="Reachability constraint satisfied (§3.3.2 static + §3.3.3 semantic)")
    c_def_satisfied: bool = Field(description="Defense blocks attack")
    is_vulnerable: bool = Field(description="True if vulnerable, False otherwise")
    verdict_category: str = Field(description="'VULNERABLE', 'SAFE-Blocked', 'SAFE-Mismatch', 'SAFE-Unreachable', 'SAFE-TypeMismatch', or 'SAFE-OutOfScope'")
    anchor_evidence: List[StepAnalysis] = Field(default_factory=list, description="Typed anchor evidence chain (ordered per constraint chain, e.g. [ALLOC→DEALLOC→USE])")
    trace: List[StepAnalysis] = Field(default_factory=list, description="Intermediate trace steps between anchors")
    defense_mechanism: Optional[StepAnalysis] = Field(default=None, description="Defense if SAFE-Blocked")
    analysis_report: str = Field(description="Brief analysis summary (1-2 sentences)")


class BaselineJudgeOutput(BaseModel):
    """Baseline Output: Simplified verdict without complex evidence chains."""
    is_vulnerable: bool = Field(description="Verdict: True if vulnerable, False if safe.")
    reasoning: str = Field(description="Reasoning for the verdict explaining why target is safe or vulnerable.")


# ==============================================================================
# Baseline Three-Role Debate Output Models (Simplified)
# ==============================================================================

class BaselineRedOutput(BaseModel):
    """Baseline Red Agent: Argue that vulnerability EXISTS in target code."""
    vulnerability_exists: bool = Field(description="Red's claim: True if vulnerability exists")
    concedes: bool = Field(default=False, description="True if Red concedes to Blue's defense argument")
    first_anchor_line: Optional[int] = Field(default=None, description="Line number of the first anchor in the chain (e.g. ALLOC for UAF)")
    first_anchor_code: Optional[str] = Field(default=None, description="Code at the first anchor")
    last_anchor_line: Optional[int] = Field(default=None, description="Line number of the last anchor in the chain (e.g. USE for UAF)")
    last_anchor_code: Optional[str] = Field(default=None, description="Code at the last anchor")
    attack_reasoning: str = Field(description="Explanation of how the attack works or response to Blue's refutation")


class BaselineBlueOutput(BaseModel):
    """Baseline Blue Agent: Argue that vulnerability does NOT exist in target code."""
    vulnerability_exists: bool = Field(description="Blue's claim: True if vulnerability exists (usually False)")
    concedes: bool = Field(default=False, description="True if Blue concedes to Red's attack argument")
    defense_found: bool = Field(description="Whether a defense mechanism was found")
    defense_line: Optional[int] = Field(default=None, description="Line number of defense mechanism")
    defense_code: Optional[str] = Field(default=None, description="Code of defense mechanism")
    refutation_reasoning: str = Field(description="Explanation of why the vulnerability does not exist or is blocked")


# ==============================================================================
# Round 1: C_cons Validation (No Tools) - Simplified Design
# ==============================================================================

class Round1RedOutput(BaseModel):
    """Round 1 Red Agent: Validate C_cons without tools (§3.3.1 Anchor Mapping)"""
    anchor_mappings: List[AnchorMapping] = Field(default_factory=list, description="All anchor mapping results (typed)")
    attack_path_exists: bool = Field(description="Whether attack path exists in target")
    c_cons_satisfied: bool = Field(description="Whether C_cons is satisfied")
    verdict: str = Field(description="'PROCEED' or 'SAFE'")
    safe_reason: Optional[str] = Field(default=None, description="Reason if verdict is SAFE")


class Round1BlueOutput(BaseModel):
    """Round 1 Blue Agent: Refute C_cons claim"""
    refutes_mapping: bool = Field(description="Whether refuting any anchor mapping")
    refutation_reason: Optional[str] = Field(default=None, description="Refutation reason if any")
    verdict: str = Field(description="'SAFE', 'CONCEDE', or 'CONTESTED'")


class Round1JudgeOutput(BaseModel):
    """Round 1 Judge: Adjudicate C_cons (§3.3.1)"""
    c_cons_satisfied: bool = Field(description="Whether C_cons is satisfied")
    verdict: str = Field(description="'SAFE-Mismatch' or 'PROCEED'")
    validated_anchors: List[AnchorMapping] = Field(default_factory=list, description="Validated anchor mappings (typed)")


# ==============================================================================
# Round 2: C_reach + C_def Verification (With Tools) - Simplified Design
# ==============================================================================

class AttackPathStep(BaseModel):
    """A step in the attack path"""
    step_type: str = Field(description="AnchorType value (e.g. 'ALLOC','DEALLOC','USE') or flow role ('Trace','Call')")
    anchor_type: Optional[str] = Field(default=None, description="AnchorType value if this step corresponds to a chain anchor")
    target_line: Optional[int] = Field(default=None, description="Target line number")
    target_code: Optional[str] = Field(default=None, description="Target code snippet")
    matches_reference: bool = Field(description="Whether matches reference semantically")


class Round2RedOutput(BaseModel):
    """Round 2 Red Agent: Establish C_reach with tools"""
    attack_path: List[AttackPathStep] = Field(default_factory=list, description="Attack path steps")
    c_reach_satisfied: bool = Field(description="Whether C_reach is satisfied")
    verdict: str = Field(description="'VULNERABLE' or 'NOT_VULNERABLE'")
    failure_reason: Optional[str] = Field(default=None, description="Reason if not vulnerable")


class DefenseCheckResult(BaseModel):
    """Defense check result - simplified"""
    defense_type: str = Field(description="'ExactPatch', 'EquivalentDefense', 'CalleeDefense', or 'CallerDefense'")
    exists: bool = Field(description="Whether defense exists")
    location: Optional[str] = Field(default=None, description="Defense location (file:line)")
    code: Optional[str] = Field(default=None, description="Defense code snippet")


class Round2BlueOutput(BaseModel):
    """Round 2 Blue Agent: Verify C_def with tools"""
    refutes_c_reach: bool = Field(description="Whether refuting C_reach")
    path_blockers: List[str] = Field(default_factory=list, description="Path blockers found")
    defense_checks: List[DefenseCheckResult] = Field(default_factory=list, description="All defense check results")
    any_defense_found: bool = Field(description="Whether any defense blocks the attack")
    verdict: str = Field(description="'SAFE', 'VULNERABLE', or 'CONTESTED'")
    safe_reason: Optional[str] = Field(default=None, description="Reason if SAFE")


class Round2JudgeOutput(BaseModel):
    """Round 2 Judge: Final adjudication (§3.3.3 Multi-Agent Semantic Analysis)"""
    c_cons_satisfied: bool = Field(description="Whether C_cons is satisfied")
    c_reach_satisfied: bool = Field(description="Whether C_reach is satisfied")
    c_def_satisfied: bool = Field(description="Whether defense blocks attack")
    first_anchor_in_function: bool = Field(default=True, description="Whether the first anchor in the chain is in current function")
    last_anchor_in_function: bool = Field(default=True, description="Whether the last anchor in the chain is in current function")
    verdict: str = Field(description="Final verdict category")
    validated_defense: Optional[DefenseCheckResult] = Field(default=None, description="Validated defense if any")


# ==============================================================================
# Path Verification Result (§3.3.2 — Pure Static, No LLM)
# ==============================================================================

class PairCheckResult(BaseModel):
    """Result of checking a single anchor pair from the constraint chain."""
    source_type: str = Field(description="Source anchor type (e.g., 'alloc', 'source')")
    target_type: str = Field(description="Target anchor type (e.g., 'dealloc', 'sink')")
    dependency_type: str = Field(description="Expected dependency type: 'DATA', 'TEMPORAL', or 'CONTROL'")
    source_line: Optional[int] = Field(default=None, description="Source anchor's mapped target line")
    target_line: Optional[int] = Field(default=None, description="Target anchor's mapped target line")
    reachable: bool = Field(description="Whether the dependency path exists for this pair")
    path_type: str = Field(default="none", description="Actual path type found: 'data_flow', 'data_flow_mixed', 'temporal', 'control_flow', 'llm_override', or 'none'")
    details: str = Field(default="", description="Explanation of the check result")
    is_optional: bool = Field(default=False, description="Whether this chain link is optional")
    skipped: bool = Field(default=False, description="Whether this pair check was skipped (e.g., unmapped anchor)")
    skip_reason: Optional[str] = Field(default=None, description="Reason for skipping")


class PathVerificationResult(BaseModel):
    """Result of §3.3.2 static path verification via PDG analysis.
    
    This step is performed through static analysis WITHOUT LLM involvement.
    It checks whether dependency edges exist between mapped anchor positions
    in the target code's PDG (Program Dependence Graph).
    
    Per the paper §3.3.2, verification iterates over the constraint chain R
    defined by the vulnerability type, checking each (a_i, a_j, δ) pair
    individually with the appropriate dependency type (DATA/TEMPORAL/CONTROL).
    """
    reachable: bool = Field(description="Whether all required dependency paths exist between anchor chain pairs")
    path_type: str = Field(default="none", description="Summary path type: 'data_flow', 'data_flow_mixed', 'temporal', 'control_flow', 'mixed', 'llm_override', or 'none'")
    details: str = Field(default="", description="Human-readable explanation of the path check result")
    anchor_lines_checked: List[int] = Field(default_factory=list, description="Target line numbers that were checked in the PDG")
    skipped: bool = Field(default=False, description="Whether path verification was skipped (e.g., empty constraint chain)")
    skip_reason: Optional[str] = Field(default=None, description="Reason for skipping path verification")
    pair_results: List[PairCheckResult] = Field(default_factory=list, description="Per-pair check results from constraint chain iteration")
    
# ==============================================================================
# 1. Tools Factory (Updated: Using CodeNavigator)
# ==============================================================================

def create_tools(navigator: CodeNavigator, current_file: str, tool_cache: Optional[Dict[str, Any]] = None):
    """
    Create a set of tools for Verifier Agent, based on CodeNavigator.
    
    Design principles (refer to anchor_analyzer.py):
    1. Explicit file_path parameter (except get_callers) - prevent Agent from forgetting path
    2. Structured error return - return JSON for easy Agent parsing
    3. Detailed debug logs - track tool calls
    4. [New] Tool call cache - avoid reading same content repeatedly
    """
    
    # Initialize cache if not provided
    if tool_cache is None:
        tool_cache = {}

    @tool
    def find_definition(symbol_name: str, file_path: str):
        """
        Find the definition of a function, struct, macro, or variable.
        Use this to read the code of called functions to verify if they handle NULL pointers or enforce constraints.
        
        Args:
            symbol_name: Name of the symbol to find (e.g. "kfree", "usb_device")
            file_path: Context file path for relevance ranking (REQUIRED)
        
        Returns:
            JSON string with definition results including {path, line, name, kind, content}
        
        Note: Definitions in the same file or directory are ranked higher.
        """
        try:
            if not symbol_name or not symbol_name.strip():
                return json.dumps({"error": "Symbol name empty", "definitions": []})
            
            if not file_path:
                return json.dumps({"error": "file_path is required for context ranking", "definitions": []})
            
            print(f"      [Verifier] 🔧 Tool: find_definition(symbol={symbol_name}, context={file_path})")
            
            # Context-aware retrieval (same file/directory ranked higher)
            definitions = navigator.find_definition(symbol_name.strip(), context_path=file_path)
            
            if not definitions:
                print(f"      [Verifier]   → No definitions found for '{symbol_name}'")
                return json.dumps({"definitions": [], "total_count": 0})
            
            # Read content for top definition
            top_def = definitions[0]
            result_defs = []
            
            for d in definitions[:3]:  # Return top 3
                def_path = d.get('path', '')
                def_line = d.get('line', 0)
                
                # Try to read code content
                try:
                    content = navigator.read_code_window(def_path, def_line, def_line + 50, with_line_numbers=True)
                    # Truncate if too long
                    if len(content) > 2000:
                        content = content[:2000] + "\n... [Truncated] ..."
                except:
                    content = "[Content unavailable]"
                
                result_defs.append({
                    "path": def_path,
                    "line": def_line,
                    "name": d.get('name', symbol_name),
                    "kind": d.get('kind', 'unknown'),
                    "content": content
                })
            
            print(f"      [Verifier]   → Found {len(definitions)} definitions, returning top {len(result_defs)}")
            return json.dumps({"definitions": result_defs, "total_count": len(definitions)})
            
        except Exception as e:
            print(f"      [Verifier]   → ✗ Error: {e}")
            return json.dumps({"error": str(e), "definitions": []})

    @tool
    def get_callers(symbol_name: str):
        """
        Find where a function is called (Upstream Analysis).
        Use this to find calling context or where tainted data comes from.
        
        Args:
            symbol_name: The function name to search for (e.g. "vulnerable_func")
        
        Returns:
            JSON string with call sites including {file, line, content, caller}
        
        Note: Searches across all indexed files. No file_path needed.
        """
        try:
            if not symbol_name or len(symbol_name) < 3:
                return json.dumps({"error": "Symbol name too short (min 3 chars)", "callers": []})
            
            print(f"      [Verifier] 🔧 Tool: get_callers(symbol={symbol_name})")
            
            callers = navigator.get_callers(symbol_name)
            
            if not callers:
                print(f"      [Verifier]   → No callers found")
                return json.dumps({"callers": [], "total_count": 0})
            
            # Sort by relevance (same file first)
            target_dir = os.path.dirname(current_file) if current_file else ""
            def sort_key(c):
                f = c.get('file', '')
                if f == current_file: return 0
                if target_dir and os.path.dirname(f) == target_dir: return 1
                return 2
            callers.sort(key=sort_key)
            
            # Limit to top 10
            result_callers = callers[:10]
            print(f"      [Verifier]   → Found {len(callers)} callers, returning top {len(result_callers)}")
            
            return json.dumps({"callers": result_callers, "total_count": len(callers)})
            
        except Exception as e:
            print(f"      [Verifier]   → ✗ Error: {e}")
            return json.dumps({"error": str(e), "callers": []})
    
    @tool
    def trace_variable(file_path: str, line: int, var_name: str, direction: str = "backward"):
        """
        Trace data flow for a variable within a function (Intra-procedural).
        
        Args:
            file_path: Target file path (REQUIRED)
            line: Starting line number
            var_name: Variable name to trace
            direction: "backward" (where it comes from) or "forward" (where it goes)
        
        Returns:
            JSON string with trace results: [{line, content, type}, ...]
        
        Note: This does NOT trace across files. Use get_callers or find_definition for inter-procedural analysis.
        """
        try:
            if not file_path:
                return json.dumps({"error": "file_path is required", "trace": []})
            
            print(f"      [Verifier] 🔧 Tool: trace_variable(file={file_path}, line={line}, var={var_name}, dir={direction})")
            
            trace = navigator.trace_variable(file_path, line, var_name, direction=direction)
            
            if isinstance(trace, list) and len(trace) > 0 and trace[0].get('error'):
                print(f"      [Verifier]   → ✗ {trace[0]['error']}")
            else:
                print(f"      [Verifier]   → Found {len(trace)} trace items")
            
            return json.dumps({"trace": trace, "total_count": len(trace)})
            
        except Exception as e:
            print(f"      [Verifier]   → ✗ Error: {e}")
            return json.dumps({"error": str(e), "trace": []})
    
    @tool
    def read_file(file_path: str, start: int, end: int):
        """
        Read specific lines from any file in the repository.
        Use this to read code context, check implementations, or examine related files.
        
        Args:
            file_path: Target file path (REQUIRED - can be different from current file)
            start: Start line number
            end: End line number
        
        Returns:
            Code content with line numbers, or error message
        
        Note: You MUST specify file_path. To read current file, use the file_path from task context.
        """
        try:
            if not file_path:
                return "Error: file_path is required"
            
            # [Cache] Check if we've already read this exact range
            cache_key = f"read_file:{file_path}:{start}:{end}"
            repeat_count_key = f"repeat_count:{file_path}:{start}:{end}"
            
            if cache_key in tool_cache:
                # Track repeat count
                tool_cache[repeat_count_key] = tool_cache.get(repeat_count_key, 0) + 1
                repeat_count = tool_cache[repeat_count_key]
                
                print(f"      [Verifier] 🔧 Tool: read_file(file={file_path}, lines={start}-{end}) [CACHED - REPEAT #{repeat_count}]")
                
                # [FORCE STOP] After 2 repeats, return ERROR without content to force Agent to use previous context
                if repeat_count >= 2:
                    print(f"      [Verifier]   ⚠️ FORCE STOP: Exceeded repeat limit (2). Returning error only.")
                    return f"[ERROR: REPEATED CALL BLOCKED] You have already read {file_path}:{start}-{end} multiple times. The content is ALREADY in your conversation history above. DO NOT call read_file for this range again. Proceed with your analysis using the existing content."
                
                # First repeat: return content with warning
                return f"[WARNING: DUPLICATE READ] You already read this content. Use the code from previous tool calls. Content (DO NOT REQUEST AGAIN):\n{tool_cache[cache_key]}"
            
            # Check if we have a cached superset that covers this range
            for k, v in tool_cache.items():
                if k.startswith(f"read_file:{file_path}:"):
                    parts = k.split(":")
                    if len(parts) == 4:
                        try:
                            cached_start, cached_end = int(parts[2]), int(parts[3])
                        except ValueError:
                            continue
                        if cached_start <= start and cached_end >= end:
                            # Extract subset from cached content
                            cached_lines = v.splitlines()
                            offset = start - cached_start
                            subset_lines = cached_lines[offset : offset + (end - start + 1)]
                            result = '\n'.join(subset_lines)
                            print(f"      [Verifier] 🔧 Tool: read_file(file={file_path}, lines={start}-{end}) [SUBSET FROM CACHED {cached_start}-{cached_end}]")
                            return result
            
            print(f"      [Verifier] 🔧 Tool: read_file(file={file_path}, lines={start}-{end})")
            
            content = navigator.read_code_window(file_path, start, end, with_line_numbers=True)
            
            lines_count = len(content.splitlines())
            print(f"      [Verifier]   → Read {lines_count} lines")
            
            # Store in cache
            tool_cache[cache_key] = content
            
            return content
            
        except Exception as e:
            print(f"      [Verifier]   → ✗ Error: {e}")
            return f"Error reading {file_path}[{start}:{end}]: {e}"

    @tool
    def get_guard_conditions(file_path: str, line: int):
        """
        Find conditional statements that guard (dominate) execution of a target line.
        CRITICAL for Blue Agent: Use this to find if a vulnerability is reachable or blocked by checks.
        
        Args:
            file_path: Target file path (REQUIRED)
            line: Line number of the vulnerable statement
        
        Returns:
            JSON string with list of guard conditions: ["Line X: <condition>", ...]
        
        Examples:
            get_guard_conditions("file.c", 25)
            → ["Line 10: if (ptr != NULL)", "Line 20: if (size > 0)"]
        """
        try:
            if not file_path:
                return json.dumps({"error": "file_path is required", "guards": []})
            
            print(f"      [Verifier] 🔧 Tool: get_guard_conditions(file={file_path}, line={line})")
            
            conditions = navigator.get_guard_conditions(file_path, line)
            
            if not conditions:
                print(f"      [Verifier]   → No guards found (or PDG unavailable)")
                return json.dumps({"guards": [], "note": "No dominating conditions found"})
            
            print(f"      [Verifier]   → Found {len(conditions)} guard conditions")
            return json.dumps({"guards": conditions, "total_count": len(conditions)})
            
        except Exception as e:
            print(f"      [Verifier]   → ✗ Error: {e}")
            return json.dumps({"error": str(e), "guards": []})

    @tool
    def grep(pattern: str, file_path: str, mode: str = "word",
             scope_start: Optional[int] = None, scope_end: Optional[int] = None):
        """
        Search for a pattern in a specific file with multiple matching modes.
        Use this to find all usages of a variable, function calls, or specific code patterns.
        
        Args:
            pattern: Search pattern - use SHORT identifiers or simple regex patterns.
                     **CRITICAL**: Do NOT use entire code lines as patterns!
                     ✓ GOOD: "ptr", "malloc", "NULL", "size > 0"
                     ✗ BAD: "if (ptr == NULL) return -1;"  (too specific, won't match)
            file_path: Target file path (REQUIRED)
            mode: Matching mode:
                - "word": Exact whole-word match (default, best for identifiers)
                - "regex": Full regex support for complex patterns
                - "def_use": PDG-enhanced mode (distinguishes definitions from uses)
            scope_start: Optional start line to limit search scope
            scope_end: Optional end line to limit search scope
        
        Returns:
            JSON string with search results: {results: [{line, content, type}], total_count, method}
        
        **USAGE GUIDELINES**:
        - For variable tracking: Use variable name only (e.g., "ptr", "buf", "len")
        - For function calls: Use function name (e.g., "malloc", "free", "memcpy")
        - For NULL checks: Use "NULL" or the variable name, not the full condition
        - For regex patterns: Keep them short and focused (e.g., r"free\\s*\\(" for free calls)
        
        Examples:
            # Find all uses of a variable
            grep("ptr", "file.c", mode="word")
            
            # Find function call patterns
            grep(r"malloc\\s*\\(", "file.c", mode="regex")
            
            # Distinguish defs from uses in a function
            grep("ptr", "file.c", mode="def_use", scope_start=10, scope_end=50)
        """
        try:
            if not file_path:
                return json.dumps({"error": "file_path is required", "results": [], "total_count": 0})
            
            scope_str = f", scope={scope_start}-{scope_end}" if scope_start and scope_end else ""
            print(f"      [Verifier] 🔧 Tool: grep(pattern='{pattern}', file={file_path}, mode={mode}{scope_str})")
            
            scope_range = (scope_start, scope_end) if scope_start and scope_end else None
            result = navigator.grep(pattern, file_path, mode=mode, scope_range=scope_range)
            
            print(f"      [Verifier]   → Found {result.get('total_count', 0)} matches (method: {result.get('method', 'unknown')})")
            return json.dumps(result)
            
        except Exception as e:
            print(f"      [Verifier]   → ✗ Error: {e}")
            return json.dumps({"error": str(e), "results": [], "total_count": 0})

    return [find_definition, get_callers, trace_variable, read_file, get_guard_conditions, grep]


def robust_json_parse(raw_content: str, output_model: BaseModel, agent_name: str = "Agent", use_fallback: bool = False) -> Optional[BaseModel]:
    """
    Robust JSON parsing function, attempting multiple strategies to parse LLM output.
    
    Parsing strategies (tried in order):
    1. Directly parse cleaned JSON
    2. Fix common JSON errors (trailing commas, single quotes, newlines, etc.)
    3. Extract nested JSON objects
    4. [Only when use_fallback=True] Fallback: Create default object with raw content (allow flow to continue)
    
    Args:
        raw_content: Raw output content from LLM
        output_model: Pydantic model class
        agent_name: Agent name (for logging)
        use_fallback: Whether to use fallback when parsing fails (only when retries exhausted)
    
    Returns:
        Return Pydantic object on success
        On failure: if use_fallback=True return fallback object, else return None
    """
    # === Strategy 1: Basic Cleaning ===
    clean_content = raw_content
    
    # 1.1 Remove Markdown code blocks
    if "```" in clean_content:
        clean_content = re.sub(r'```(?:json)?\s*(.*?)\s*```', r'\1', clean_content, flags=re.DOTALL)
    
    # 1.2 Find JSON boundaries
    start_idx = clean_content.find('{')
    end_idx = clean_content.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        clean_content = clean_content[start_idx : end_idx+1]
    
    # 1.3 Remove "thought:" prefix
    if "thought:" in clean_content.lower():
        for prefix in ["thought:", "Thought:", "THOUGHT:"]:
            if prefix in clean_content:
                clean_content = clean_content.split(prefix)[-1]
                # Re-find JSON boundaries
                start_idx = clean_content.find('{')
                end_idx = clean_content.rfind('}')
                if start_idx != -1 and end_idx != -1:
                    clean_content = clean_content[start_idx : end_idx+1]
    
    # Attempt direct parsing
    try:
        return output_model.model_validate_json(clean_content)
    except Exception as e1:
        print(f"      [{agent_name}] Strategy 1 (basic clean) failed: {str(e1)[:100]}")
    
    # === Strategy 2: Fix Common JSON Errors ===
    fixed_content = clean_content
    
    # 2.1 Fix trailing commas (trailing comma before } or ])
    fixed_content = re.sub(r',\s*([\}\]])', r'\1', fixed_content)
    
    # 2.2 Fix single quotes -> double quotes
    # Note: Only replace single quotes involved in string boundaries, not within string content
    # This is a simplified approach, edge cases may exist
    fixed_content = re.sub(r"(?<=[{\[,:\s])'([^']*)'(?=[,}\]:\s])", r'"\1"', fixed_content)
    
    # 2.3 Fix unescaped newlines in strings
    # This is complex, simplified handling: replace newlines in string values with \n
    def escape_newlines_in_strings(match):
        return match.group(0).replace('\n', '\\n').replace('\r', '\\r')
    fixed_content = re.sub(r'"[^"]*"', escape_newlines_in_strings, fixed_content)
    
    # 2.4 Fix Python booleans (True/False -> true/false)
    fixed_content = re.sub(r'\bTrue\b', 'true', fixed_content)
    fixed_content = re.sub(r'\bFalse\b', 'false', fixed_content)
    fixed_content = re.sub(r'\bNone\b', 'null', fixed_content)
    
    try:
        return output_model.model_validate_json(fixed_content)
    except Exception as e2:
        print(f"      [{agent_name}] Strategy 2 (fix common errors) failed: {str(e2)[:100]}")
    
    # === Strategy 3: Extract Nested JSON Objects ===
    # Sometimes LLM outputs {"result": {...actual_data...}}
    try:
        parsed = json.loads(fixed_content)
        if isinstance(parsed, dict):
            # Attempt direct use
            try:
                return output_model.model_validate(parsed)
            except:
                pass
            
            # Attempt to extract from nested structure
            for key in ['result', 'output', 'response', 'data', 'json']:
                if key in parsed and isinstance(parsed[key], dict):
                    try:
                        return output_model.model_validate(parsed[key])
                    except:
                        pass
    except Exception as e3:
        print(f"      [{agent_name}] Strategy 3 (nested extraction) failed: {str(e3)[:100]}")
    
    # === Strategy 4: Fallback - Create default object with raw content ===
    # Enabled only when use_fallback=True (usually when max_parse_retries is reached)
    if use_fallback:
        print(f"      [{agent_name}] All parsing strategies failed, creating fallback with raw content")
        
        fallback = _create_fallback_output(output_model, raw_content, agent_name)
        if fallback is not None:
            print(f"      [{agent_name}] Fallback created successfully, flow can continue")
            return fallback
        
        print(f"      [{agent_name}] Could not create fallback object")
    else:
        print(f"      [{agent_name}] All parsing strategies failed (fallback disabled, will retry)")
    
    return None


def _create_fallback_output(output_model: BaseModel, raw_content: str, agent_name: str):
    """
    Create fallback object with raw content for various Output models.
    This allows agent interaction to continue even if JSON parsing fails.
    
    Args:
        output_model: Target Pydantic model class
        raw_content: Raw LLM output content
        agent_name: Agent name
    
    Returns:
        Fallback object containing raw content, or None if creation fails
    """
    # Truncate overly long content
    truncated_content = raw_content[:2000] + "..." if len(raw_content) > 2000 else raw_content
    fallback_note = f"[FALLBACK] JSON parsing failed. Raw LLM output:\n{truncated_content}"
    
    model_name = output_model.__name__
    
    try:
        # Round1RedOutput fallback (simplified)
        if model_name == "Round1RedOutput":
            return Round1RedOutput(
                anchor_mappings=[],
                attack_path_exists=False,
                c_cons_satisfied=False,  # Conservative: assume not satisfied when uncertain
                verdict="PROCEED",  # Allow flow to continue to Round 2
                safe_reason=fallback_note
            )
        
        # Round1BlueOutput fallback (simplified)
        elif model_name == "Round1BlueOutput":
            return Round1BlueOutput(
                refutes_mapping=False,
                refutation_reason=fallback_note,
                verdict="CONCEDE"  # Default concede to Red (conservative)
            )
        
        # Round1JudgeOutput fallback (simplified)
        elif model_name == "Round1JudgeOutput":
            return Round1JudgeOutput(
                c_cons_satisfied=True,  # Conservative: allow flow to continue
                verdict="PROCEED",
                validated_anchors=[]
            )
        
        # Round2RedOutput fallback (simplified)
        elif model_name == "Round2RedOutput":
            return Round2RedOutput(
                attack_path=[],
                c_reach_satisfied=False,
                verdict="NOT_VULNERABLE",  # Conservative: assume safe when uncertain
                failure_reason=fallback_note
            )
        
        # Round2BlueOutput fallback (simplified)
        elif model_name == "Round2BlueOutput":
            return Round2BlueOutput(
                refutes_c_reach=False,
                path_blockers=[],
                defense_checks=[],
                any_defense_found=False,
                verdict="CONTESTED",  # Let Round 3 decide
                safe_reason=fallback_note
            )
        
        # Round2JudgeOutput fallback (simplified)
        elif model_name == "Round2JudgeOutput":
            return Round2JudgeOutput(
                c_cons_satisfied=True,  # Conservative: allow flow to continue
                c_reach_satisfied=False,
                c_def_satisfied=False,
                first_anchor_in_function=True,
                last_anchor_in_function=True,
                verdict="PROCEED",  # Let Round 3 decide
                validated_defense=None
            )
        
        # JudgeOutput (Final) fallback (simplified)
        elif model_name == "JudgeOutput":
            return JudgeOutput(
                c_cons_satisfied=False,
                c_reach_satisfied=False,
                c_def_satisfied=False,
                is_vulnerable=False,  # Conservative: default safe
                verdict_category="SAFE-Unknown",
                anchor_evidence=[],
                trace=[],
                defense_mechanism=None,
                analysis_report=f"JSON parsing failed - {agent_name}"
            )
        
        else:
            # Unknown model type, cannot create fallback
            print(f"      [{agent_name}] Unknown model type: {model_name}, cannot create fallback")
            return None
            
    except Exception as e:
        print(f"      [{agent_name}] Failed to create fallback: {e}")
        return None


def llm_invoke_with_retry(llm, messages, max_retries: int = 3, retry_delay: float = 5.0):
    """
    Enhanced LLM invocation wrapper with automatic retry mechanism.
    Handles network errors (500, 502, 503, 504) and connection timeouts.
    
    Args:
        llm: ChatOpenAI instance
        messages: Message list
        max_retries: Max retries
        retry_delay: Retry interval (seconds), increases exponentially
    
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
            
            # Network/Server Error (Retryable)
            if any(x in error_str for x in ['500', '502', '503', '504', 'internal server error']):
                error_type = "server_error"
                is_retryable = True
            # Connection Error (Retryable)
            elif any(x in error_str for x in ['connection', 'timeout', 'timed out', 'network', 'reset', 'refused']):
                error_type = "connection_error"
                is_retryable = True
            # Rate Limit (Retryable, but longer delay)
            elif any(x in error_str for x in ['rate limit', 'too many requests', '429']):
                error_type = "rate_limit"
                is_retryable = True
                current_delay = max(current_delay, 30.0)  # Rate limit waits at least 30s
            # API Key/Auth Error (Non-retryable)
            elif any(x in error_str for x in ['api key', 'authentication', 'unauthorized', '401', '403']):
                error_type = "auth_error"
                is_retryable = False
            
            if is_retryable and attempt < max_retries - 1:
                print(f"      [LLM-Retry] {error_type} (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"      [LLM-Retry] Waiting {current_delay:.1f}s before retry...")
                time.sleep(current_delay)
                current_delay *= 2  # Exponential backoff
            else:
                # Non-retryable error or final attempt
                print(f"      [LLM-Error] {error_type} (final attempt): {e}")
                return None  # Return None instead of raising exception
    
    # All retries failed
    print(f"      [LLM-Error] All {max_retries} attempts failed. Last error: {last_exception}")
    return None

# ==============================================================================
# Round 1: C_cons Validation Execution Functions (No Tools)
# ==============================================================================

def build_phase3_mapping_summary(sf, ev) -> str:
    """
    Format Phase 3 aligned_vuln_traces into readable mapping summary.
    
    Implements §3.3.1 Anchor Mapping with locatability-aware strategy:
    - CONCRETE anchors: Direct alignment via Phase 3 traces (highest confidence)
    - ASSUMED anchors:  Phase 3 hint + assumption verification guidance
    - CONCEPTUAL anchors: No direct trace; LLM must discover from context
    
    Groups anchors by locatability tier to guide the Red Agent's mapping strategy.
    
    Args:
        sf: SliceFeature (contains pre_anchors: List[Anchor])
        ev: MatchEvidence (contains aligned_vuln_traces)
    
    Returns:
        Formatted mapping summary string
    """
    def extract_line_no(line: str) -> int:
        """Extract line number from line with line number marker"""
        match = re.match(r'^\[\s*(\d+)\]', line.strip())
        return int(match.group(1)) if match else -1
    
    def find_target_mapping_for_anchor(anchor, aligned_traces: list) -> tuple:
        """
        Find the target line that maps to a given Anchor object.
        Uses anchor.line_number first, then falls back to code_snippet content matching.
        Returns: (target_line_no, target_code, similarity) or (None, None, 0.0)
        """
        # Strategy 1: Use anchor.line_number to match slice line markers
        if anchor.line_number:
            for trace in aligned_traces:
                slice_ln = extract_line_no(trace.slice_line)
                if slice_ln == anchor.line_number and trace.target_line:
                    return (trace.line_no, trace.target_line, trace.similarity)
        
        # Strategy 2: Use code_snippet content matching
        if anchor.code_snippet:
            snippet = anchor.code_snippet.strip()
            for trace in aligned_traces:
                if trace.target_line and snippet and snippet in trace.slice_line:
                    return (trace.line_no, trace.target_line, trace.similarity)
        
        return (None, None, 0.0)
    
    lines = []
    
    # Group anchors by locatability tier
    concrete_anchors = []
    assumed_anchors = []
    conceptual_anchors = []
    
    for anchor in (sf.pre_anchors or []):
        loc = anchor.locatability
        if hasattr(loc, 'value'):
            loc_val = loc.value
        else:
            loc_val = str(loc)
        
        target_ln, target_code, sim = find_target_mapping_for_anchor(anchor, ev.aligned_vuln_traces)
        entry = (anchor, target_ln, target_code, sim)
        
        if loc_val == 'concrete':
            concrete_anchors.append(entry)
        elif loc_val == 'assumed':
            assumed_anchors.append(entry)
        else:  # conceptual
            conceptual_anchors.append(entry)
    
    # === Tier 1: CONCRETE anchors (sliceable, searchable, no assumptions) ===
    if concrete_anchors:
        lines.append("**Tier 1 — CONCRETE Anchors** (direct code locations, highest confidence):")
        lines.append("  Strategy: Verify Phase 3 alignment → confirm semantic role match")
        for anchor, target_ln, target_code, sim in concrete_anchors:
            quality = 'good' if sim > 0.6 else ('weak' if sim > 0.3 else 'missing')
            label = format_anchor_for_prompt(anchor)
            opt_tag = " (optional)" if anchor.is_optional else ""
            if target_ln:
                lines.append(f"  [{quality.upper()}] {label}{opt_tag}")
                lines.append(f"      → Target Line {target_ln}: `{target_code.strip() if target_code else ''}` (sim={sim:.2f})")
            else:
                lines.append(f"  [UNMAPPED] {label}{opt_tag}")
    
    # === Tier 2: ASSUMED anchors (has location, but requires assumption verification) ===
    if assumed_anchors:
        lines.append("")
        lines.append("**Tier 2 — ASSUMED Anchors** (location known, assumption needs verification):")
        lines.append("  Strategy: Verify Phase 3 alignment → THEN verify assumption holds in target context")
        for anchor, target_ln, target_code, sim in assumed_anchors:
            quality = 'good' if sim > 0.6 else ('weak' if sim > 0.3 else 'missing')
            label = format_anchor_for_prompt(anchor)
            opt_tag = " (optional)" if anchor.is_optional else ""
            at = ""
            if anchor.assumption_type:
                at_val = anchor.assumption_type.value if hasattr(anchor.assumption_type, 'value') else str(anchor.assumption_type)
                at = f" [Assumption: {at_val}]"
            rationale = f" — {anchor.assumption_rationale}" if anchor.assumption_rationale else ""
            if target_ln:
                lines.append(f"  [{quality.upper()}] {label}{opt_tag}{at}{rationale}")
                lines.append(f"      → Target Line {target_ln}: `{target_code.strip() if target_code else ''}` (sim={sim:.2f})")
                lines.append(f"      ⚠ VERIFY: Does assumption still hold in target context?")
            else:
                lines.append(f"  [UNMAPPED] {label}{opt_tag}{at}{rationale}")
                lines.append(f"      ⚠ VERIFY: Search target code for equivalent; verify assumption")
    
    # === Tier 3: CONCEPTUAL anchors (not directly in code, must be inferred) ===
    if conceptual_anchors:
        lines.append("")
        lines.append("**Tier 3 — CONCEPTUAL Anchors** (inferred, not directly in slice code):")
        lines.append("  Strategy: LLM must discover equivalent semantic role in target from context/root cause")
        for anchor, target_ln, target_code, sim in conceptual_anchors:
            label = format_anchor_for_prompt(anchor)
            opt_tag = " (optional)" if anchor.is_optional else ""
            at = ""
            if anchor.assumption_type:
                at_val = anchor.assumption_type.value if hasattr(anchor.assumption_type, 'value') else str(anchor.assumption_type)
                at = f" [Assumption: {at_val}]"
            rationale = f" — {anchor.assumption_rationale}" if anchor.assumption_rationale else ""
            # Conceptual anchors rarely have Phase 3 mappings, but check anyway
            if target_ln:
                lines.append(f"  [HINT] {label}{opt_tag}{at}{rationale}")
                lines.append(f"      → Possible Target Line {target_ln}: `{target_code.strip() if target_code else ''}` (sim={sim:.2f})")
            else:
                lines.append(f"  [DISCOVER] {label}{opt_tag}{at}{rationale}")
                lines.append(f"      → No Phase 3 alignment. Must be inferred from target code and root cause.")
    
    if not concrete_anchors and not assumed_anchors and not conceptual_anchors:
        lines.append("**Anchor Mappings**: [No typed anchors defined in Phase 2]")
    
    return "\n".join(lines)


def run_round1_red(
    feature: PatchFeatures,
    candidate,  # SearchResultItem
    target_code_with_lines: str,
    llm: ChatOpenAI
) -> Round1RedOutput:
    """
    Round 1 Red Agent: Anchor Mapping (§3.3.1) — Validate C_cons WITHOUT tools.
    
    Implements the paper's anchor mapping step with locatability-aware strategy:
    1. Extract Phase 2 typed anchors with locatability info (CONCRETE/ASSUMED/CONCEPTUAL)
    2. Build tiered Phase 3 mapping summary (Tier 1/2/3 by locatability)
    3. Let Red Agent produce mapping μ: M → T per anchor
    4. For ASSUMED anchors: verify assumption holds in target context
    5. For CONCEPTUAL anchors: LLM discovers from root cause + target code
    6. Verify constraint chain causality
    7. If any critical (non-optional) anchor unmapped → SAFE
    """
    print("      [Round1-Red] Anchor Mapping §3.3.1 — validating C_cons (no tools)...")
    
    sf = feature.slices.get(candidate.patch_func)
    ev = candidate.evidence
    
    # Build Phase 3 Mapping Summary (now tiered by locatability)
    phase3_summary = build_phase3_mapping_summary(sf, ev) if sf else "No slice found for this function"
    
    # Format typed anchors for prompt (now includes locatability/assumption info)
    anchor_prompt_lines = []
    if sf and sf.pre_anchors:
        for i, anchor in enumerate(sf.pre_anchors):
            pos = "first" if i == 0 else ("last" if i == len(sf.pre_anchors) - 1 else "middle")
            line = f"  {i+1}. {format_anchor_for_prompt(anchor)} [chain_position: {pos}]"
            anchor_prompt_lines.append(line)
    anchor_prompt = chr(10).join(anchor_prompt_lines) if anchor_prompt_lines else '[No typed anchors identified in Phase 2]'
    
    # Get constraint chain description if available
    constraint_chain = ""
    violation_pred = ""
    mitigation_patterns = ""
    if sf and hasattr(sf, 'taxonomy') and sf.taxonomy and hasattr(sf.taxonomy, 'category_obj'):
        try:
            cat = sf.taxonomy.category_obj
            if cat and hasattr(cat, 'constraint') and cat.constraint:
                constraint_chain = cat.constraint.chain_str()
                if cat.constraint.violation:
                    violation_pred = str(cat.constraint.violation)
                if cat.constraint.mitigations:
                    mitigation_patterns = ", ".join(str(m) for m in cat.constraint.mitigations)
        except Exception:
            pass
    
    # Count anchor stats for prompt guidance
    n_concrete = sum(1 for a in (sf.pre_anchors or []) if hasattr(a.locatability, 'value') and a.locatability.value == 'concrete') if sf else 0
    n_assumed = sum(1 for a in (sf.pre_anchors or []) if hasattr(a.locatability, 'value') and a.locatability.value == 'assumed') if sf else 0
    n_conceptual = sum(1 for a in (sf.pre_anchors or []) if hasattr(a.locatability, 'value') and a.locatability.value == 'conceptual') if sf else 0
    n_optional = sum(1 for a in (sf.pre_anchors or []) if a.is_optional) if sf else 0
    
    # Build Prompt
    user_input = f"""
### Phase 2 Typed Anchors (From Reference Vulnerability Analysis)

**Constraint Chain**: {constraint_chain or 'Not available'}
**Violation Predicate**: {violation_pred or 'Not available'}
**Known Mitigations**: {mitigation_patterns or 'None defined'}

**Anchor Points** (ordered per vulnerability constraint):
{anchor_prompt}

**Anchor Statistics**: {n_concrete} CONCRETE, {n_assumed} ASSUMED, {n_conceptual} CONCEPTUAL ({n_optional} optional)

### Phase 3 Tiered Mapping Results (Slice → Target Alignment)
{phase3_summary}

### Target Code (with line numbers)
{target_code_with_lines}

### Vulnerability Context (Root Cause Guides Semantic Equivalence)
- **Type**: {feature.semantics.vuln_type.value if feature.semantics.vuln_type else 'Unknown'}
- **CWE**: {feature.semantics.cwe_id or 'Unknown'} - {feature.semantics.cwe_name or ''}
- **Root Cause**: {feature.semantics.root_cause}
- **Attack Chain**: {feature.semantics.attack_chain}

### Reference Slice (Pre-Patch)
{sf.s_pre if sf else 'Not available'}

**TASK — Produce Anchor Mapping μ: M → T**:
1. For each anchor in the constraint chain, produce an AnchorMapping entry
2. Follow the tiered strategy: CONCRETE → ASSUMED → CONCEPTUAL
3. For ASSUMED anchors: verify the assumption holds in target context
4. For CONCEPTUAL anchors: infer from root cause + target code semantics
5. Verify constraint chain causality (execution order + data flow)
6. Optional anchors can be unmapped without failing C_cons
7. If any critical (non-optional) anchor is unmapped → verdict = SAFE
8. If chain causality fails → verdict = SAFE
9. If all critical anchors mapped with valid causality → verdict = PROCEED
"""
    
    messages = [
        SystemMessage(content=ROUND1_RED_PROMPT),
        HumanMessage(content=user_input)
    ]
    
    # Request structured output
    try:
        schema_dict = Round1RedOutput.model_json_schema()
        schema_str = json.dumps(schema_dict, indent=2)
    except:
        schema_str = "Use the known JSON schema for Round1RedOutput."
    
    messages.append(HumanMessage(content=f"\n\nProvide your analysis in structured JSON format.\n\nJSON SCHEMA:\n{schema_str}\n\nIMPORTANT: Output ONLY the raw JSON."))
    
    max_parse_retries = 3
    for attempt in range(max_parse_retries):
        try:
            ai_msg = llm_invoke_with_retry(llm, messages)
            
            if ai_msg is None:
                print(f"      [Round1-Red] LLM call failed (attempt {attempt+1})")
                if attempt == max_parse_retries - 1:
                    # Return default "proceed" to let Round 2 continue
                    return Round1RedOutput(
                        anchor_mappings=[],
                        attack_path_exists=False,
                        c_cons_satisfied=False,
                        verdict="PROCEED",
                        safe_reason="LLM call failed"
                    )
                continue
            
            raw_content = ai_msg.content
            
            # Use robust JSON parsing (enable fallback only on last attempt)
            is_last_attempt = (attempt == max_parse_retries - 1)
            result = robust_json_parse(raw_content, Round1RedOutput, "Round1-Red", use_fallback=is_last_attempt)
            
            if result is not None:
                print(f"      [Round1-Red] Result: c_cons={result.c_cons_satisfied}, verdict={result.verdict}")
                return result
            
            print(f"      [Round1-Red] Robust parsing failed (attempt {attempt+1})")
            # Add ai_msg and feedback to messages so LLM can see and fix its previous output
            if not is_last_attempt:
                messages.append(ai_msg)  # Let LLM see its previous output
                messages.append(HumanMessage(content="Parse error: Your output could not be parsed as valid JSON. Please fix the JSON format issues and output ONLY the raw JSON matching the schema, without any markdown or extra text."))
            
        except Exception as e:
            print(f"      [Round1-Red] Error (attempt {attempt+1}): {e}")
            if attempt == max_parse_retries - 1:
                return Round1RedOutput(
                    anchor_mappings=[],
                    attack_path_exists=False,
                    c_cons_satisfied=False,
                    verdict="PROCEED",
                    safe_reason=f"JSON parsing error: {str(e)}"
                )
            messages.append(HumanMessage(content=f"Parse error: {e}. Please output valid JSON only."))
    
    # All retries failed, return default value
    return Round1RedOutput(
        anchor_mappings=[],
        attack_path_exists=False,
        c_cons_satisfied=False,
        verdict="PROCEED",
        safe_reason="All parsing attempts failed"
    )


def run_round1_blue(
    red_output: Round1RedOutput,
    feature: PatchFeatures,
    candidate,  # SearchResultItem
    target_code_with_lines: str,
    llm: ChatOpenAI
) -> Round1BlueOutput:
    """
    Round 1 Blue Agent: Refute Red's C_cons claim WITHOUT tools.
    """
    print("      [Round1-Blue] Attempting to refute C_cons...")
    
    # Format Red's anchor mapping claims for Blue (typed anchor model)
    anchor_claims_lines = []
    if red_output.anchor_mappings:
        for m in red_output.anchor_mappings:
            status = "✓ MAPPED" if m.is_mapped else "✗ NOT MAPPED"
            loc_tag = f"|{m.locatability}" if m.locatability else ""
            opt_tag = " (optional)" if m.is_optional else ""
            line = f"  [{status}] [{m.anchor_type}{loc_tag}]{opt_tag} Reference: {m.reference_line}"
            if m.is_mapped:
                conf = f" (confidence: {m.mapping_confidence})" if m.mapping_confidence else ""
                line += f"\n      → Target Line {m.target_line}: `{m.target_code}`{conf}"
                if m.semantic_role_verified:
                    line += f"\n      ✓ Semantic role verified"
                if m.reason:
                    line += f"\n      Reason: {m.reason}"
            # Show assumption verification for ASSUMED/CONCEPTUAL anchors
            if m.locatability in ('assumed', 'conceptual'):
                at = f" [{m.assumption_type}]" if m.assumption_type else ""
                av = "✓" if m.assumption_verified else ("✗" if m.assumption_verified is False else "?")
                line += f"\n      Assumption{at}: {av}"
                if m.assumption_verification_note:
                    line += f" — {m.assumption_verification_note}"
            anchor_claims_lines.append(line)
    anchor_claims = chr(10).join(anchor_claims_lines) if anchor_claims_lines else "  [No anchor mappings provided]"
    
    user_input = f"""
### Red Agent's C_cons Claim

**C_cons Satisfied**: {red_output.c_cons_satisfied}
**Attack Path Exists**: {red_output.attack_path_exists}
**Verdict**: {red_output.verdict}
{f"**Safe Reason**: {red_output.safe_reason}" if red_output.safe_reason else ""}

**Anchor Mappings** (typed, ordered per constraint chain):
{anchor_claims}

### Target Code (with line numbers)
{target_code_with_lines}

### Vulnerability Context
- **Type**: {feature.semantics.vuln_type.value if feature.semantics.vuln_type else 'Unknown'}
- **Root Cause**: {feature.semantics.root_cause}
- **Attack Chain**: {feature.semantics.attack_chain}

**TASK**:
1. Examine Red's anchor mappings - are they semantically correct for their declared type?
2. **CRITICAL - Check Causality**: Does the chain satisfy causal relationships?
   - **Execution Order**: Can earlier chain anchors execute BEFORE later ones in any feasible path?
   - **Data Flow**: Does the state created at the first anchor flow through the chain to the last?
   - **Common Causality Errors**:
     * **Reversed Order**: Later chain anchor has lower line number (check if valid, e.g., callbacks/macros)
     * **Cleanup Confusion**: Multiple anchors are cleanup/deallocation code (not vulnerable use)
     * **Different Variables**: Anchors affect different variables with no connection
     * **Blocked Path**: Control flow (return/goto/exit) prevents chain traversal
   - If causality is INVALID, REFUTE with "Causality Violation"
3. If Red's mapping is wrong (different operation, different variable role), REFUTE it
4. If the target code has a fundamentally different mechanism, explain the MISMATCH
5. If Red's claim is valid AND causality is correct, CONCEDE
"""
    
    messages = [
        SystemMessage(content=ROUND1_BLUE_PROMPT),
        HumanMessage(content=user_input)
    ]
    
    # Request structured output
    try:
        schema_dict = Round1BlueOutput.model_json_schema()
        schema_str = json.dumps(schema_dict, indent=2)
    except:
        schema_str = "Use the known JSON schema for Round1BlueOutput."
    
    messages.append(HumanMessage(content=f"\n\nProvide your analysis in structured JSON format.\n\nJSON SCHEMA:\n{schema_str}\n\nIMPORTANT: Output ONLY the raw JSON."))
    
    max_parse_retries = 3
    for attempt in range(max_parse_retries):
        try:
            ai_msg = llm_invoke_with_retry(llm, messages)
            
            if ai_msg is None:
                print(f"      [Round1-Blue] LLM call failed (attempt {attempt+1})")
                if attempt == max_parse_retries - 1:
                    # Default: concede to Red (conservative)
                    return Round1BlueOutput(
                        refutes_mapping=False,
                        refutation_reason="LLM call failed, defaulting to concede",
                        verdict="CONCEDE"
                    )
                continue
            
            raw_content = ai_msg.content
            
            # Use robust JSON parsing (enable fallback only on last attempt)
            is_last_attempt = (attempt == max_parse_retries - 1)
            result = robust_json_parse(raw_content, Round1BlueOutput, "Round1-Blue", use_fallback=is_last_attempt)
            
            if result is not None:
                print(f"      [Round1-Blue] Result: refutes_mapping={result.refutes_mapping}, verdict={result.verdict}")
                return result
            
            print(f"      [Round1-Blue] Robust parsing failed (attempt {attempt+1})")
            # Add ai_msg and feedback to messages so LLM can see and fix its previous output
            if not is_last_attempt:
                messages.append(ai_msg)  # Let LLM see its previous output
                messages.append(HumanMessage(content="Parse error: Your output could not be parsed as valid JSON. Please fix the JSON format issues and output ONLY the raw JSON matching the schema, without any markdown or extra text."))
            
        except Exception as e:
            print(f"      [Round1-Blue] Error (attempt {attempt+1}): {e}")
            if attempt == max_parse_retries - 1:
                return Round1BlueOutput(
                    refutes_mapping=False,
                    refutation_reason=f"JSON parsing failed: {str(e)}",
                    verdict="CONCEDE"
                )
            messages.append(HumanMessage(content=f"Parse error: {e}. Please output valid JSON only."))
    
    # All retries failed, return default value (concede to Red)
    return Round1BlueOutput(
        refutes_mapping=False,
        refutation_reason="All parsing attempts failed, defaulting to concede",
        verdict="CONCEDE"
    )


def run_round1_judge(
    red_output: Round1RedOutput,
    blue_output: Round1BlueOutput,
    feature: PatchFeatures,
    candidate,
    target_code_with_lines: str,
    llm: ChatOpenAI
) -> Round1JudgeOutput:
    """
    Round 1 Judge: Adjudicate C_cons based on Red and Blue arguments.
    """
    print("      [Round1-Judge] Adjudicating C_cons...")
    
    # Format typed anchor mappings from Red
    anchor_info_lines = []
    for m in (red_output.anchor_mappings or []):
        status = "MAPPED" if m.is_mapped else "NOT MAPPED"
        loc_tag = f"|{m.locatability}" if m.locatability else ""
        opt_tag = " (opt)" if m.is_optional else ""
        info = f"  [{m.anchor_type}{loc_tag}] [{status}]{opt_tag} ref: {m.reference_line}"
        if m.is_mapped:
            info += f" → Target L{m.target_line}: `{m.target_code}` (conf: {m.mapping_confidence})"
        if m.locatability in ('assumed', 'conceptual') and m.assumption_verified is not None:
            info += f" [assumption: {'verified' if m.assumption_verified else 'FAILED'}]"
        anchor_info_lines.append(info)
    anchor_info = chr(10).join(anchor_info_lines) if anchor_info_lines else "  [No anchor mappings]"
    
    # Build original anchor definition summary from Phase 2
    # This tells the Judge what the reference vulnerability's anchor layout looks like
    sf = feature.slices.get(candidate.patch_func)
    ref_anchor_lines = []
    shared_line_note = ""
    if sf and sf.pre_anchors:
        line_to_types = {}  # Track which anchor types share the same line
        for anchor in sf.pre_anchors:
            atype = anchor.type.value if hasattr(anchor.type, 'value') else str(anchor.type)
            loc = anchor.locatability.value if hasattr(anchor.locatability, 'value') else str(anchor.locatability)
            opt_tag = " (optional)" if anchor.is_optional else ""
            snippet = anchor.code_snippet.strip()[:60] if anchor.code_snippet else 'N/A'
            ref_anchor_lines.append(f"  [{atype}|{loc}]{opt_tag} L{anchor.line_number}: `{snippet}`")
            # Track shared lines
            if anchor.line_number:
                line_to_types.setdefault(anchor.line_number, []).append(atype)
        # Generate shared-line note if any line has multiple anchors
        shared_lines = {ln: types for ln, types in line_to_types.items() if len(types) > 1}
        if shared_lines:
            notes = []
            for ln, types in shared_lines.items():
                notes.append(f"L{ln} hosts {'+'.join(types)}")
            shared_line_note = f"\n⚠ **Shared-line anchors in reference**: {'; '.join(notes)}. " \
                             f"Multiple anchors on the same line is VALID in the reference vulnerability — " \
                             f"the target mapping should mirror this layout."
    ref_anchor_info = chr(10).join(ref_anchor_lines) if ref_anchor_lines else "  [No reference anchors]"
    
    # Get constraint chain description
    constraint_chain_str = ""
    try:
        constraint = feature.taxonomy.constraint
        if constraint:
            constraint_chain_str = constraint.chain_str()
    except Exception:
        pass
    
    user_input = f"""
### Phase 2 Reference Anchor Definition (Original Vulnerability)
**Constraint Chain**: {constraint_chain_str or 'Not available'}

**Reference Anchors** (from patch analysis — this is the GROUND TRUTH layout):
{ref_anchor_info}
{shared_line_note}

### Red Agent's C_cons Claim
- **C_cons Satisfied**: {red_output.c_cons_satisfied}
- **Attack Path Exists**: {red_output.attack_path_exists}
- **Verdict**: {red_output.verdict}
{f"- **Safe Reason**: {red_output.safe_reason}" if red_output.safe_reason else ""}

**Anchor Mappings** (typed, ordered per constraint chain):
{anchor_info}

### Blue Agent's Refutation
- **Refutes Mapping**: {blue_output.refutes_mapping}
- **Refutation Reason**: {blue_output.refutation_reason or 'N/A'}
- **Verdict**: {blue_output.verdict}

### Target Code (for verification)
{target_code_with_lines}

### Vulnerability Context
- **Type**: {feature.semantics.vuln_type.value if feature.semantics.vuln_type else 'Unknown'}
- **Root Cause**: {feature.semantics.root_cause}
- **Attack Chain**: {feature.semantics.attack_chain}
- **Patch Defense**: {feature.semantics.patch_defense}

**TASK**:
1. Compare Red's mappings against the **Reference Anchor Definition** above
2. Evaluate if Blue's refutation is valid — note that if the reference anchors share a line, the target mapping SHOULD also share a line
3. Decide: SAFE-Mismatch (terminate) or PROCEED (to Round 2)
"""
    
    messages = [
        SystemMessage(content=ROUND1_JUDGE_PROMPT),
        HumanMessage(content=user_input)
    ]
    
    # Request structured output
    try:
        schema_dict = Round1JudgeOutput.model_json_schema()
        schema_str = json.dumps(schema_dict, indent=2)
    except:
        schema_str = "Use the known JSON schema for Round1JudgeOutput."
    
    messages.append(HumanMessage(content=f"\n\nProvide your verdict in structured JSON format.\n\nJSON SCHEMA:\n{schema_str}\n\nIMPORTANT: Output ONLY the raw JSON."))
    
    max_parse_retries = 3
    for attempt in range(max_parse_retries):
        try:
            ai_msg = llm_invoke_with_retry(llm, messages)
            
            if ai_msg is None:
                print(f"      [Round1-Judge] LLM call failed (attempt {attempt+1})")
                if attempt == max_parse_retries - 1:
                    # Default: proceed to Round 2 (conservative)
                    return Round1JudgeOutput(
                        c_cons_satisfied=True,
                        verdict="PROCEED",
                        validated_anchors=red_output.anchor_mappings
                    )
                continue
            
            raw_content = ai_msg.content
            
            # Use robust JSON parsing (enable fallback only on last attempt)
            is_last_attempt = (attempt == max_parse_retries - 1)
            result = robust_json_parse(raw_content, Round1JudgeOutput, "Round1-Judge", use_fallback=is_last_attempt)
            
            if result is not None:
                print(f"      [Round1-Judge] Result: c_cons={result.c_cons_satisfied}, verdict={result.verdict}")
                return result
            
            print(f"      [Round1-Judge] Robust parsing failed (attempt {attempt+1})")
            # Add ai_msg and feedback to messages so LLM can see and fix its previous output
            if not is_last_attempt:
                messages.append(ai_msg)  # Let LLM see its previous output
                messages.append(HumanMessage(content="Parse error: Your output could not be parsed as valid JSON. Please fix the JSON format issues and output ONLY the raw JSON matching the schema, without any markdown or extra text."))
            
        except Exception as e:
            print(f"      [Round1-Judge] Error (attempt {attempt+1}): {e}")
            if attempt == max_parse_retries - 1:
                return Round1JudgeOutput(
                    c_cons_satisfied=True,
                    verdict="PROCEED",
                    validated_anchors=red_output.anchor_mappings
                )
            messages.append(HumanMessage(content=f"Parse error: {e}. Please output valid JSON only."))
    
    # All retries failed, return default value (proceed to Round 2)
    return Round1JudgeOutput(
        c_cons_satisfied=True,
        verdict="PROCEED",
        validated_anchors=red_output.anchor_mappings
    )


def run_round1_debate(
    feature: PatchFeatures,
    candidate,  # SearchResultItem
    target_code_with_lines: str,
    llm: ChatOpenAI
) -> tuple:
    """
    Execute complete Round 1 debate (Red → Blue → Judge)
    
    Returns:
        (round1_red, round1_blue, round1_judge, should_continue)
        - round1_red: Round1RedOutput (Red's C_cons analysis)
        - round1_blue: Round1BlueOutput (Blue's refutation attempt)
        - round1_judge: Round1JudgeOutput (Judge's adjudication)
        - should_continue: bool (True = proceed to Round 2, False = terminate with SAFE-Mismatch)
    """
    print("    [Round1] Starting C_cons validation debate (no tools)...")
    
    # Step 1: Red validates C_cons
    red_output = run_round1_red(feature, candidate, target_code_with_lines, llm)
    
    # Early exit: Red says SAFE (couldn't establish C_cons)
    if red_output.verdict == "SAFE":
        print(f"    [Round1] Red verdict: SAFE - {red_output.safe_reason}")
        # Create minimal Blue output for early exit (using simplified model)
        blue_output = Round1BlueOutput(
            refutes_mapping=False,
            refutation_reason="Red already concluded SAFE, no refutation needed",
            verdict="CONCEDE"
        )
        judge_output = Round1JudgeOutput(
            c_cons_satisfied=False,
            verdict="SAFE-Mismatch",
            validated_anchors=[]
        )
        return red_output, blue_output, judge_output, False
    
    # Step 2: Blue attempts to refute
    blue_output = run_round1_blue(red_output, feature, candidate, target_code_with_lines, llm)
    
    # Step 3: Judge adjudicates
    judge_output = run_round1_judge(red_output, blue_output, feature, candidate, target_code_with_lines, llm)
    
    should_continue = (judge_output.verdict == "PROCEED")
    
    print(f"    [Round1] Final verdict: {judge_output.verdict}")
    
    return red_output, blue_output, judge_output, should_continue


# ==============================================================================
# Round 2: C_cons Reinforcement + C_reach + C_def Execution Functions (With Tools)
# ==============================================================================

def run_round2_red(
    round1_result: Round1JudgeOutput,
    feature: PatchFeatures,
    candidate,  # SearchResultItem
    target_code_with_lines: str,
    tools: List[Any],
    llm: ChatOpenAI
) -> Round2RedOutput:
    """
    Round 2 Red Agent: Reinforce C_cons + Establish C_reach (WITH tools).
    
    Core tasks:
    1. If Round 1 is controversial, use tools to verify Origin/Impact mapping
    2. Build attack path, verify semantic consistency of each step with reference path
    3. Verify data flow and control flow reachability
    """
    print("      [Round2-Red] Establishing C_reach with tools...")
    
    sf = feature.slices.get(candidate.patch_func)
    
    # Build Round 1 validated anchor mappings summary (typed model)
    validated_anchor_lines = []
    for m in (round1_result.validated_anchors or []):
        status = "VALIDATED" if m.is_mapped else "NOT VALIDATED"
        info = f"  [{m.anchor_type}] [{status}] ref: {m.reference_line}"
        if m.is_mapped:
            info += f" → Target L{m.target_line}: `{m.target_code}`"
        validated_anchor_lines.append(info)
    validated_anchors_str = chr(10).join(validated_anchor_lines) if validated_anchor_lines else "  [No validated anchors from Round 1]"
    
    user_input = f"""
### Round 1 Validated Anchor Mappings
{validated_anchors_str}
**Round 1 C_cons**: {round1_result.c_cons_satisfied}

### Reference Attack Chain
{feature.semantics.attack_chain}

### Reference Vulnerability Semantics
- **Type**: {feature.semantics.vuln_type.value if feature.semantics.vuln_type else 'Unknown'}
- **CWE**: {feature.semantics.cwe_id or 'Unknown'} - {feature.semantics.cwe_name or ''}
- **Root Cause**: {feature.semantics.root_cause}

### Reference Slice (Pre-Patch)
{sf.s_pre if sf else 'Not available'}

### Target Code (with line numbers)
File: {candidate.target_file}
{target_code_with_lines}

**TASK**:
1. If Round 1 had issues, use tools to reinforce C_cons (verify anchor mappings)
2. Construct the attack path through the constraint chain anchors
3. For each step, verify semantic consistency with reference
4. Check data flow (same variable throughout) and control flow (no blockers)
5. Report C_reach verdict
"""
    
    # Run with tools using ReAct loop
    llm_with_tools = llm.bind_tools(tools)
    messages = [
        SystemMessage(content=ROUND2_RED_PROMPT),
        HumanMessage(content=user_input)
    ]
    
    # Tool exploration loop
    max_tool_steps = 10
    for step in range(max_tool_steps):
        try:
            ai_msg = llm_invoke_with_retry(llm_with_tools, messages)
            if ai_msg is None:
                print(f"      [Round2-Red] LLM call failed (step {step+1})")
                break
        except Exception as e:
            print(f"      [Round2-Red] LLM call exception: {e}")
            break
            
        messages.append(ai_msg)
        
        if not ai_msg.tool_calls:
            break
            
        for tc in ai_msg.tool_calls:
            tool_map = {t.name: t for t in tools}
            t_func = tool_map.get(tc["name"])
            try:
                res = str(t_func.invoke(tc["args"])) if t_func else "Tool not found"
            except Exception as e:
                res = f"Error executing tool '{tc['name']}': {str(e)}"
            messages.append(ToolMessage(content=res, tool_call_id=tc["id"]))
    
    # Structured output extraction
    try:
        schema_dict = Round2RedOutput.model_json_schema()
        schema_str = json.dumps(schema_dict, indent=2)
    except:
        schema_str = "Use the known JSON schema for Round2RedOutput."
    
    messages.append(HumanMessage(content=f"\n\nProvide your analysis in structured JSON format.\n\nJSON SCHEMA:\n{schema_str}\n\nIMPORTANT: Output ONLY the raw JSON."))
    
    max_parse_retries = 3
    for attempt in range(max_parse_retries):
        try:
            ai_msg = llm_invoke_with_retry(llm, messages)
            
            if ai_msg is None:
                print(f"      [Round2-Red] Final LLM call failed (attempt {attempt+1})")
                if attempt == max_parse_retries - 1:
                    return Round2RedOutput(
                        attack_path=[],
                        c_reach_satisfied=False,
                        verdict="NOT_VULNERABLE",
                        failure_reason="Round 2 Red LLM failed"
                    )
                continue
            
            raw_content = ai_msg.content
            
            # Use robust JSON parsing (enable fallback only on last attempt)
            is_last_attempt = (attempt == max_parse_retries - 1)
            result = robust_json_parse(raw_content, Round2RedOutput, "Round2-Red", use_fallback=is_last_attempt)
            
            if result is not None:
                print(f"      [Round2-Red] Result: c_reach={result.c_reach_satisfied}, verdict={result.verdict}")
                return result
            
            print(f"      [Round2-Red] Robust parsing failed (attempt {attempt+1})")
            # Add ai_msg and feedback to messages so LLM can see and fix its previous output
            if not is_last_attempt:
                messages.append(ai_msg)  # Let LLM see its previous output
                messages.append(HumanMessage(content="Parse error: Your output could not be parsed as valid JSON. Please fix the JSON format issues and output ONLY the raw JSON matching the schema, without any markdown or extra text."))
            
        except Exception as e:
            print(f"      [Round2-Red] Error (attempt {attempt+1}): {e}")
            if attempt == max_parse_retries - 1:
                return Round2RedOutput(
                    attack_path=[],
                    c_reach_satisfied=False,
                    verdict="NOT_VULNERABLE",
                    failure_reason=f"Round 2 Red parsing failed: {str(e)}"
                )
            messages.append(HumanMessage(content=f"Parse error: {e}. Please output valid JSON only."))
    
    # All retries failed, return default value
    return Round2RedOutput(
        attack_path=[],
        c_reach_satisfied=False,
        verdict="NOT_VULNERABLE",
        failure_reason="Round 2 Red parsing failed after all retries"
    )


def run_round2_blue(
    round2_red_output: Round2RedOutput,
    round1_result: Round1JudgeOutput,
    feature: PatchFeatures,
    candidate,  # SearchResultItem
    target_code_with_lines: str,
    tools: List[Any],
    llm: ChatOpenAI
) -> Round2BlueOutput:
    """
    Round 2 Blue Agent: Refute C_cons/C_reach + Verify C_def (WITH tools).
    
    Core tasks:
    1. Attempt to refute Red's C_cons/C_reach
    2. Check Defense according to four-layer strategy
    3. Output explicit Defense check report
    """
    print("      [Round2-Blue] Refuting C_reach and verifying C_def...")
    
    sf = feature.slices.get(candidate.patch_func)
    
    # Format Red's attack path for Blue (using simplified model)
    attack_path_summary = []
    for i, step in enumerate(round2_red_output.attack_path):
        attack_path_summary.append(
            f"  Step {i+1} [{step.step_type}]: Line {step.target_line} - `{step.target_code}`"
            f"\n    Matches Reference: {step.matches_reference}"
        )
    
    user_input = f"""
### Red's Round 2 Claims

**Attack Path** (C_reach):
{chr(10).join(attack_path_summary) if attack_path_summary else '  [No attack path provided]'}

**C_reach Satisfied**: {round2_red_output.c_reach_satisfied}
**Red's Verdict**: {round2_red_output.verdict}
{f"**Failure Reason**: {round2_red_output.failure_reason}" if round2_red_output.failure_reason else ""}

### Reference Fix (What the patch does)
- **Root Cause**: {feature.semantics.root_cause}
- **Patch Defense**: {feature.semantics.patch_defense}

**Reference Post-Patch Slice**:
{sf.s_post if sf else 'Not available'}

### Target Code (with line numbers)
File: {candidate.target_file}
{target_code_with_lines}

**TASK**:
1. Try to REFUTE Red's C_reach claims
   - Show path blockers (guards, early returns, dead code)

2. Verify C_def (Defense) - CHECK ALL FOUR TYPES:
   - ExactPatch: Does target contain the same fix?
   - EquivalentDefense: Is there a different but equivalent defense?
   - CalleeDefense: Does any callee contain the defense? (USE find_definition!)
   - CallerDefense: Does the caller perform validation? (USE get_callers!)

3. For each defense type found, report:
   - If EXISTS: show location, code, and how it blocks the attack
"""
    
    # Run with tools using ReAct loop
    llm_with_tools = llm.bind_tools(tools)
    messages = [
        SystemMessage(content=ROUND2_BLUE_PROMPT),
        HumanMessage(content=user_input)
    ]
    
    # Tool exploration loop
    max_tool_steps = 12  # Blue may need more tools for defense checking
    for step in range(max_tool_steps):
        try:
            ai_msg = llm_invoke_with_retry(llm_with_tools, messages)
            if ai_msg is None:
                print(f"      [Round2-Blue] LLM call failed (step {step+1})")
                break
        except Exception as e:
            print(f"      [Round2-Blue] LLM call exception: {e}")
            break
            
        messages.append(ai_msg)
        
        if not ai_msg.tool_calls:
            break
            
        for tc in ai_msg.tool_calls:
            tool_map = {t.name: t for t in tools}
            t_func = tool_map.get(tc["name"])
            try:
                res = str(t_func.invoke(tc["args"])) if t_func else "Tool not found"
            except Exception as e:
                res = f"Error executing tool '{tc['name']}': {str(e)}"
            messages.append(ToolMessage(content=res, tool_call_id=tc["id"]))
    
    # Structured output extraction
    try:
        schema_dict = Round2BlueOutput.model_json_schema()
        schema_str = json.dumps(schema_dict, indent=2)
    except:
        schema_str = "Use the known JSON schema for Round2BlueOutput."
    
    messages.append(HumanMessage(content=f"\n\nProvide your analysis in structured JSON format.\n\nJSON SCHEMA:\n{schema_str}\n\nIMPORTANT: Output ONLY the raw JSON."))
    
    max_parse_retries = 3
    for attempt in range(max_parse_retries):
        try:
            ai_msg = llm_invoke_with_retry(llm, messages)
            
            if ai_msg is None:
                print(f"      [Round2-Blue] Final LLM call failed (attempt {attempt+1})")
                if attempt == max_parse_retries - 1:
                    return Round2BlueOutput(
                        refutes_c_reach=False,
                        path_blockers=[],
                        defense_checks=[],
                        any_defense_found=False,
                        verdict="CONTESTED",
                        safe_reason="LLM call failed"
                    )
                continue
            
            raw_content = ai_msg.content
            
            # Use robust JSON parsing (enable fallback only on last attempt)
            is_last_attempt = (attempt == max_parse_retries - 1)
            result = robust_json_parse(raw_content, Round2BlueOutput, "Round2-Blue", use_fallback=is_last_attempt)
            
            if result is not None:
                print(f"      [Round2-Blue] Result: refutes_c_reach={result.refutes_c_reach}, any_defense_found={result.any_defense_found}, verdict={result.verdict}")
                return result
            
            print(f"      [Round2-Blue] Robust parsing failed (attempt {attempt+1})")
            # Add ai_msg and feedback to messages so LLM can see and fix its previous output
            if not is_last_attempt:
                messages.append(ai_msg)  # Let LLM see its previous output
                messages.append(HumanMessage(content="Parse error: Your output could not be parsed as valid JSON. Please fix the JSON format issues and output ONLY the raw JSON matching the schema, without any markdown or extra text."))
            
        except Exception as e:
            print(f"      [Round2-Blue] Error (attempt {attempt+1}): {e}")
            if attempt == max_parse_retries - 1:
                return Round2BlueOutput(
                    refutes_c_reach=False,
                    path_blockers=[],
                    defense_checks=[],
                    any_defense_found=False,
                    verdict="CONTESTED",
                    safe_reason=f"JSON parsing failed: {str(e)}"
                )
            messages.append(HumanMessage(content=f"Parse error: {e}. Please output valid JSON only."))
    
    # All retries failed, return default value
    return Round2BlueOutput(
        refutes_c_reach=False,
        path_blockers=[],
        defense_checks=[],
        any_defense_found=False,
        verdict="CONTESTED",
        safe_reason="All parsing attempts failed"
    )


def run_round2_judge(
    round2_red_output: Round2RedOutput,
    round2_blue_output: Round2BlueOutput,
    round1_result: Round1JudgeOutput,
    feature: PatchFeatures,
    candidate,
    target_code_with_lines: str,
    llm: ChatOpenAI
) -> Round2JudgeOutput:
    """
    Round 2 Judge: Adjudicate C_cons, C_reach, and C_def.
    """
    print("      [Round2-Judge] Adjudicating all constraints...")
    
    # Format defense checks for judge (using simplified model - flat list)
    defense_summary_lines = []
    for check in round2_blue_output.defense_checks:
        exists_str = "EXISTS" if check.exists else "NOT_EXISTS"
        defense_summary_lines.append(f"  - {check.defense_type}: [{exists_str}] Location: {check.location or 'N/A'}")
        if check.code:
            defense_summary_lines.append(f"    Code: `{check.code}`")
    
    defense_summary = f"""
**Defense Checks**:
{chr(10).join(defense_summary_lines) if defense_summary_lines else '  [No defense checks performed]'}
**Any Defense Found**: {round2_blue_output.any_defense_found}
"""
    
    user_input = f"""
### Red's Round 2 Output

**Attack Path Steps**: {len(round2_red_output.attack_path)} steps
**C_reach Satisfied**: {round2_red_output.c_reach_satisfied}
**Red's Verdict**: {round2_red_output.verdict}
{f"**Failure Reason**: {round2_red_output.failure_reason}" if round2_red_output.failure_reason else ""}

### Blue's Round 2 Output

**Refutes C_reach**: {round2_blue_output.refutes_c_reach}
**Path Blockers**: {', '.join(round2_blue_output.path_blockers) if round2_blue_output.path_blockers else 'None'}

{defense_summary}

**Blue's Verdict**: {round2_blue_output.verdict}
**Safe Reason**: {round2_blue_output.safe_reason or 'N/A'}

### Round 1 Context
**C_cons from Round 1**: {round1_result.c_cons_satisfied}

### Target Code (for verification)
{target_code_with_lines}

### Vulnerability Context
- **Type**: {feature.semantics.vuln_type.value if feature.semantics.vuln_type else 'Unknown'}
- **Root Cause**: {feature.semantics.root_cause}
- **Patch Defense**: {feature.semantics.patch_defense}

**TASK**:
1. Evaluate C_cons: Is the mapping valid? Did Blue successfully refute?
2. Evaluate C_reach: Is the attack path valid? Are there blockers?
3. Evaluate C_def: Did Blue find a valid defense?
4. Render verdict based on: VULNERABLE iff C_cons ∧ C_reach ∧ ¬C_def
"""
    
    messages = [
        SystemMessage(content=ROUND2_JUDGE_PROMPT),
        HumanMessage(content=user_input)
    ]
    
    # Request structured output
    try:
        schema_dict = Round2JudgeOutput.model_json_schema()
        schema_str = json.dumps(schema_dict, indent=2)
    except:
        schema_str = "Use the known JSON schema for Round2JudgeOutput."
    
    messages.append(HumanMessage(content=f"\n\nProvide your verdict in structured JSON format.\n\nJSON SCHEMA:\n{schema_str}\n\nIMPORTANT: Output ONLY the raw JSON."))
    
    # Find strongest defense from Blue's checks
    def find_strongest_defense() -> Optional[DefenseCheckResult]:
        for check in round2_blue_output.defense_checks:
            if check.exists:
                return check
        return None
    
    max_parse_retries = 3
    for attempt in range(max_parse_retries):
        try:
            ai_msg = llm_invoke_with_retry(llm, messages)
            
            if ai_msg is None:
                print(f"      [Round2-Judge] LLM call failed (attempt {attempt+1})")
                if attempt == max_parse_retries - 1:
                    # Default: proceed if we can't decide
                    return Round2JudgeOutput(
                        c_cons_satisfied=round1_result.c_cons_satisfied,
                        c_reach_satisfied=round2_red_output.c_reach_satisfied,
                        c_def_satisfied=round2_blue_output.any_defense_found,
                        first_anchor_in_function=True,
                        last_anchor_in_function=True,
                        verdict="PROCEED",
                        validated_defense=find_strongest_defense()
                    )
                continue
            
            raw_content = ai_msg.content
            
            # Use robust JSON parsing (enable fallback only on last attempt)
            is_last_attempt = (attempt == max_parse_retries - 1)
            result = robust_json_parse(raw_content, Round2JudgeOutput, "Round2-Judge", use_fallback=is_last_attempt)
            
            if result is not None:
                print(f"      [Round2-Judge] Result: c_cons={result.c_cons_satisfied}, c_reach={result.c_reach_satisfied}, c_def={result.c_def_satisfied}, verdict={result.verdict}")
                return result
            
            print(f"      [Round2-Judge] Robust parsing failed (attempt {attempt+1})")
            # Add ai_msg and feedback to messages so LLM can see and fix its previous output
            if not is_last_attempt:
                messages.append(ai_msg)  # Let LLM see its previous output
                messages.append(HumanMessage(content="Parse error: Your output could not be parsed as valid JSON. Please fix the JSON format issues and output ONLY the raw JSON matching the schema, without any markdown or extra text."))
            
        except Exception as e:
            print(f"      [Round2-Judge] Error (attempt {attempt+1}): {e}")
            if attempt == max_parse_retries - 1:
                return Round2JudgeOutput(
                    c_cons_satisfied=round1_result.c_cons_satisfied,
                    c_reach_satisfied=round2_red_output.c_reach_satisfied,
                    c_def_satisfied=round2_blue_output.any_defense_found,
                    first_anchor_in_function=True,
                    last_anchor_in_function=True,
                    verdict="PROCEED",
                    validated_defense=find_strongest_defense()
                )
            messages.append(HumanMessage(content=f"Parse error: {e}. Please output valid JSON only."))
    
    # All retries failed, return default value
    return Round2JudgeOutput(
        c_cons_satisfied=round1_result.c_cons_satisfied,
        c_reach_satisfied=round2_red_output.c_reach_satisfied,
        c_def_satisfied=round2_blue_output.any_defense_found,
        first_anchor_in_function=True,
        last_anchor_in_function=True,
        verdict="PROCEED",
        validated_defense=find_strongest_defense()
    )


def run_round2_debate(
    round1_result: Round1JudgeOutput,
    feature: PatchFeatures,
    candidate,  # SearchResultItem
    target_code_with_lines: str,
    tools: List[Any],
    llm: ChatOpenAI
) -> tuple:
    """
    Execute complete Round 2 debate (Red → Blue → Judge)
    
    Returns:
        (round2_red, round2_blue, round2_judge, final_verdict)
        - round2_red: Round2RedOutput (Red's C_reach + attack path)
        - round2_blue: Round2BlueOutput (Blue's refutation + defense report)
        - round2_judge: Round2JudgeOutput (Judge's adjudication)
        - final_verdict: str ('SAFE-Unreachable', 'SAFE-Blocked', 'VULNERABLE', 'PROCEED')
    """
    print("    [Round2] Starting C_reach + C_def debate (with tools)...")
    
    # Step 1: Red establishes C_reach
    red_output = run_round2_red(round1_result, feature, candidate, target_code_with_lines, tools, llm)
    
    # Early exit: Red says NOT_VULNERABLE (couldn't establish C_reach)
    if red_output.verdict == "NOT_VULNERABLE":
        print(f"    [Round2] Red verdict: NOT_VULNERABLE - {red_output.failure_reason}")
        # Create minimal Blue output for early exit (using simplified model)
        blue_output = Round2BlueOutput(
            refutes_c_reach=False,
            path_blockers=[],
            defense_checks=[],
            any_defense_found=False,
            verdict="VULNERABLE",  # Blue concedes since Red already failed
            safe_reason=None
        )
        judge_output = Round2JudgeOutput(
            c_cons_satisfied=round1_result.c_cons_satisfied,
            c_reach_satisfied=False,
            c_def_satisfied=False,
            first_anchor_in_function=True,
            last_anchor_in_function=True,
            verdict="SAFE-Unreachable",
            validated_defense=None
        )
        return red_output, blue_output, judge_output, "SAFE-Unreachable"
    
    # Step 2: Blue refutes or finds defense
    blue_output = run_round2_blue(red_output, round1_result, feature, candidate, target_code_with_lines, tools, llm)
    
    # Step 3: Judge adjudicates
    judge_output = run_round2_judge(red_output, blue_output, round1_result, feature, candidate, target_code_with_lines, llm)
    
    print(f"    [Round2] Final verdict: {judge_output.verdict}")
    
    return red_output, blue_output, judge_output, judge_output.verdict




# ==============================================================================
# Round 3: Final Judge Integration (No Tools) - Summary Only
# ==============================================================================

def run_round3_final_judge(
    round1_red: Round1RedOutput,
    round1_blue: Round1BlueOutput,
    round1_judge: Round1JudgeOutput,
    round2_red: Round2RedOutput,
    round2_blue: Round2BlueOutput,
    round2_judge: Round2JudgeOutput,
    feature: PatchFeatures,
    candidate,  # SearchResultItem
    target_code_with_lines: str,
    llm: ChatOpenAI
) -> JudgeOutput:
    """
    Round 3 Final Judge: Integrate all previous round results and output final JudgeOutput.
    
    This is a simplified round with only the Judge (no Red/Blue).
    The Judge synthesizes Round 1 and Round 2 results into the final verdict format.
    
    Args:
        round1_red: Round 1 Red Agent output (typed anchor chain mappings)
        round1_blue: Round 1 Blue Agent output (refutation attempts)
        round1_judge: Round 1 Judge output (C_cons adjudication)
        round2_red: Round 2 Red Agent output (attack path steps)
        round2_blue: Round 2 Blue Agent output (defense report)
        round2_judge: Round 2 Judge output (C_reach + C_def adjudication)
        feature: Patch features with vulnerability semantics
        candidate: Target candidate being verified
        target_code_with_lines: Target code with line numbers
        llm: LLM instance for generation
    
    Returns:
        JudgeOutput with complete evidence chain
    """
    print("      [Round3-Judge] Generating final JudgeOutput...")
    
    sf = feature.slices.get(candidate.patch_func)
    
    # === Format Round 1 Evidence (Typed Anchor Model) ===
    
    # Round 1 Red: All typed anchor mappings
    r1_anchor_lines = []
    for m in (round1_red.anchor_mappings or []):
        status = "MAPPED" if m.is_mapped else "NOT MAPPED"
        info = f"  [{m.anchor_type}] [{status}] ref: {m.reference_line}"
        if m.is_mapped:
            info += f" → Target L{m.target_line}: `{m.target_code}`"
        r1_anchor_lines.append(info)
    r1_anchors_str = chr(10).join(r1_anchor_lines) if r1_anchor_lines else "  [No anchor mappings]"
    
    round1_red_summary = f"""
**Round 1 Red Agent (C_cons Evidence)**:
- C_cons Satisfied: {round1_red.c_cons_satisfied}
- Attack Path Exists: {round1_red.attack_path_exists}
- Verdict: {round1_red.verdict}
{f"- Safe Reason: {round1_red.safe_reason}" if round1_red.safe_reason else ""}

Anchor Mappings:
{r1_anchors_str}
"""
    
    round1_judge_summary = f"""
**Round 1 Judge Decision**:
- C_cons Satisfied: {round1_judge.c_cons_satisfied}
- Verdict: {round1_judge.verdict}
"""
    
    # === Format Round 2 Evidence (Simplified Model) ===
    
    # Round 2 Red: Attack path steps (using simplified AttackPathStep)
    attack_path_info = []
    for i, step in enumerate(round2_red.attack_path):
        attack_path_info.append(f"  Step {i+1} [{step.step_type}]:")
        attack_path_info.append(f"    Line {step.target_line}: `{step.target_code}`")
        attack_path_info.append(f"    Matches Reference: {step.matches_reference}")
    
    round2_red_summary = f"""
**Round 2 Red Agent (C_reach Evidence)**:
- C_reach Satisfied: {round2_red.c_reach_satisfied}
- Verdict: {round2_red.verdict}
{f"- Failure Reason: {round2_red.failure_reason}" if round2_red.failure_reason else ""}

Attack Path Steps:
{chr(10).join(attack_path_info) if attack_path_info else '  [No attack path]'}
"""
    
    # Round 2 Blue: Defense checks (flat list, not nested report)
    defense_checks_lines = []
    for check in round2_blue.defense_checks:
        exists_str = "EXISTS" if check.exists else "NOT_EXISTS"
        defense_checks_lines.append(f"  - {check.defense_type}: [{exists_str}]")
        if check.location:
            defense_checks_lines.append(f"    Location: {check.location}")
        if check.code:
            defense_checks_lines.append(f"    Code: `{check.code}`")
    
    defense_checks_info = f"""
**Round 2 Blue Agent (C_def Evidence)**:
- Refutes C_reach: {round2_blue.refutes_c_reach}
- Path Blockers: {', '.join(round2_blue.path_blockers) if round2_blue.path_blockers else 'None'}
- Verdict: {round2_blue.verdict}
- Safe Reason: {round2_blue.safe_reason or 'N/A'}

Defense Checks:
{chr(10).join(defense_checks_lines) if defense_checks_lines else '  [No defense checks]'}
- Any Defense Found: {round2_blue.any_defense_found}
"""
    
    round2_judge_summary = f"""
**Round 2 Judge Decision**:
- C_cons Satisfied: {round2_judge.c_cons_satisfied}
- C_reach Satisfied: {round2_judge.c_reach_satisfied}
- C_def Satisfied: {round2_judge.c_def_satisfied}
- First Anchor In Function: {round2_judge.first_anchor_in_function}
- Last Anchor In Function: {round2_judge.last_anchor_in_function}
- Verdict: {round2_judge.verdict}
"""
    
    # Format validated anchors (typed model)
    validated_anchor_summary_lines = []
    for m in (round1_judge.validated_anchors or []):
        if m.is_mapped:
            validated_anchor_summary_lines.append(f"  [{m.anchor_type}] Target L{m.target_line}: `{m.target_code}` (ref: {m.reference_line})")
        else:
            validated_anchor_summary_lines.append(f"  [{m.anchor_type}] NOT MAPPED (ref: {m.reference_line})")
    validated_anchors_summary = chr(10).join(validated_anchor_summary_lines) if validated_anchor_summary_lines else "  [No validated anchors]"
    
    # Format defense info (simplified model)
    defense_info = "No defense validated"
    if round2_judge.validated_defense and round2_judge.validated_defense.exists:
        d = round2_judge.validated_defense
        defense_info = f"Type: {d.defense_type}, Location: {d.location or 'N/A'}"
    
    user_input = f"""
### Task: Generate Final Verdict

You are the Final Judge. Your task is to synthesize all previous round results and generate a complete, structured verdict.

## Complete Evidence from All Rounds

### Validated Anchor Mappings (Summary)
{validated_anchors_summary}

{round1_red_summary}

{round1_judge_summary}

{round2_red_summary}

{defense_checks_info}

{round2_judge_summary}

### Defense Information (Validated)
{defense_info}

### Target Code (with line numbers)
File: {candidate.target_file}
Function: {candidate.target_func.split(':')[-1]}
{target_code_with_lines}

### Vulnerability Context
- **Type**: {feature.semantics.vuln_type.value if feature.semantics.vuln_type else 'Unknown'}
- **CWE**: {feature.semantics.cwe_id or 'Unknown'} - {feature.semantics.cwe_name or ''}
- **Root Cause**: {feature.semantics.root_cause}
- **Attack Chain**: {feature.semantics.attack_chain}
- **Patch Defense**: {feature.semantics.patch_defense}

### Reference Slice (Pre-Patch)
{sf.s_pre if sf else 'Not available'}

**YOUR TASK**:
Based on ALL the above Round 1 and Round 2 evidence (Red/Blue/Judge outputs), generate the final JudgeOutput with:

1. **Constraint evaluations**: Use the Judge decisions from Round 1 and Round 2
   - c_cons_satisfied: from Round 2 Judge
   - c_reach_satisfied: from Round 2 Judge
   - c_def_satisfied: from Round 2 Judge

2. **is_vulnerable**: True iff C_cons ∧ C_reach ∧ ¬C_def

3. **verdict_category**: 'VULNERABLE', 'SAFE-Blocked', 'SAFE-Mismatch', 'SAFE-Unreachable', 'SAFE-TypeMismatch', or 'SAFE-OutOfScope'

4. **anchor_evidence**: List of StepAnalysis for ALL typed anchors in the constraint chain
   - Each entry should have role = anchor type (e.g. "ALLOC", "DEALLOC", "USE")
   - Use the ACTUAL line numbers from the target code
   - Include the exact code content
   - Order them per the constraint chain

5. **trace**: List of StepAnalysis from Round 2 Red's attack_path
   - Include Trace/Call steps between chain anchors
   - Use actual line numbers and code from the attack path

6. **defense_mechanism**: StepAnalysis from Round 2 Blue's defense_checks (if SAFE-Blocked)
   - Use the strongest defense found
   - Include its location and code

7. **analysis_report**: Concise 1-2 sentence summary

**CRITICAL - CHAIN CAUSALITY CHECK**:
- **VERIFY chain ordering**: Each anchor in the chain MUST causally flow to the next
  * Check data flow: Does the state created at each anchor flow to the next?
  * Check control flow: Is there a feasible execution path through the chain?
  * If chain ordering is invalid → mark as SAFE-Mismatch
- **Line number order is NOT always indicative** (cross-function calls, callbacks, async operations)
- **BUT data/control flow MUST be valid**: Each anchor's effect must reach the next
- Use the ACTUAL line numbers from Round 2 Red's attack_path
- Do NOT make up line numbers - use only what's provided in the evidence

**CHAIN VALIDATION PATTERNS**:
- For UAF: ALLOC →d DEALLOC →t USE (allocation, then deallocation, then use)
- For NPD: SOURCE → USE (null-producing source, then dereference)
- For Integer Overflow: SOURCE → COMPUTATION → SINK (source, arithmetic, sensitive use)
- For OOB: OBJECT → INDEX → ACCESS (buffer, index computation, access)
- If Blue refuted causality in Round 2, carefully review the refutation evidence
"""
    
    messages = [
        SystemMessage(content=ROUND3_FINAL_JUDGE_PROMPT),
        HumanMessage(content=user_input)
    ]
    
    # Request structured output
    try:
        schema_dict = JudgeOutput.model_json_schema()
        schema_str = json.dumps(schema_dict, indent=2)
    except:
        schema_str = "Use the known JSON schema for JudgeOutput."
    
    messages.append(HumanMessage(content=f"\n\nProvide your final verdict in structured JSON format.\n\nJSON SCHEMA:\n{schema_str}\n\nIMPORTANT: Output ONLY the raw JSON."))
    
    max_parse_retries = 3
    for attempt in range(max_parse_retries):
        try:
            ai_msg = llm_invoke_with_retry(llm, messages)
            
            if ai_msg is None:
                print(f"      [Round3-Judge] LLM call failed (attempt {attempt+1})")
                if attempt == max_parse_retries - 1:
                    # Return a default JudgeOutput based on all Round 1 and Round 2 results
                    return _create_fallback_judge_output(
                        round1_red, round1_blue, round1_judge,
                        round2_red, round2_blue, round2_judge,
                        candidate, feature
                    )
                continue
            
            raw_content = ai_msg.content
            
            # Use robust parsing (enable fallback only on last attempt)
            is_last_attempt = (attempt == max_parse_retries - 1)
            result = robust_json_parse(raw_content, JudgeOutput, "Round3-Judge", use_fallback=is_last_attempt)
            
            if result is not None:
                print(f"      [Round3-Judge] Result: is_vulnerable={result.is_vulnerable}, verdict={result.verdict_category}")
                return result
            
            print(f"      [Round3-Judge] Parsing failed (attempt {attempt+1})")
            # Add ai_msg and feedback to messages so LLM can see and fix its previous output
            if attempt < max_parse_retries - 1:
                messages.append(ai_msg)  # Let LLM see its previous output
                messages.append(HumanMessage(content="Parse error: Your output could not be parsed as valid JSON. Please fix the JSON format issues and output ONLY the raw JSON matching the JudgeOutput schema, without any markdown or extra text."))
                
        except Exception as e:
            print(f"      [Round3-Judge] Error (attempt {attempt+1}): {e}")
            if attempt == max_parse_retries - 1:
                return _create_fallback_judge_output(
                    round1_red, round1_blue, round1_judge,
                    round2_red, round2_blue, round2_judge,
                    candidate, feature
                )
    
    # All attempts failed
    return _create_fallback_judge_output(
        round1_red, round1_blue, round1_judge,
        round2_red, round2_blue, round2_judge,
        candidate, feature
    )


def _create_fallback_judge_output(
    round1_red: Round1RedOutput,
    round1_blue: Round1BlueOutput,
    round1_judge: Round1JudgeOutput,
    round2_red: Round2RedOutput,
    round2_blue: Round2BlueOutput,
    round2_judge: Round2JudgeOutput,
    candidate,
    feature: PatchFeatures
) -> JudgeOutput:
    """
    Create a fallback JudgeOutput when Round 3 LLM fails.
    Uses all Round 1 and Round 2 outputs to construct complete evidence chain.
    Updated to use simplified model fields.
    """
    # Compute is_vulnerable based on constraints
    is_vulnerable = (
        round2_judge.c_cons_satisfied and
        round2_judge.c_reach_satisfied and
        not round2_judge.c_def_satisfied and
        (round2_judge.first_anchor_in_function or round2_judge.last_anchor_in_function)
    )
    
    # Determine verdict category
    if not round2_judge.c_cons_satisfied:
        verdict_category = "SAFE-Mismatch"
    elif not round2_judge.c_reach_satisfied:
        verdict_category = "SAFE-Unreachable"
    elif not round2_judge.first_anchor_in_function and not round2_judge.last_anchor_in_function:
        verdict_category = "SAFE-OutOfScope"
    elif round2_judge.c_def_satisfied:
        verdict_category = "SAFE-Blocked"
    else:
        verdict_category = "VULNERABLE" if is_vulnerable else "SAFE-Unknown"
    
    # === Build typed anchor evidence from Round 2 Red's attack_path ===
    # Collect all anchor steps (non-Trace/Call) into anchor_evidence,
    # and Trace/Call steps into trace_steps.
    anchor_evidence = []
    trace_steps = []
    
    for i, step in enumerate(round2_red.attack_path):
        step_analysis = StepAnalysis(
            role=step.anchor_type or step.step_type,
            file_path=candidate.target_file,
            func_name=candidate.target_func.split(':')[-1],
            line_number=step.target_line,
            code_content=step.target_code,
            observation=f"Step {i+1}: matches_reference={step.matches_reference}"
        )
        
        if step.step_type in ('Trace', 'Call'):
            trace_steps.append(step_analysis)
        else:
            # Anchor step (e.g. ALLOC, DEALLOC, USE, SOURCE, SINK, etc.)
            anchor_evidence.append(step_analysis)
    
    # Fallback to Round 1 Judge's validated mappings if Round 2 didn't have attack path
    if not anchor_evidence:
        for m in (round1_judge.validated_anchors or []):
            if m.is_mapped:
                anchor_evidence.append(StepAnalysis(
                    role=m.anchor_type,
                    file_path=candidate.target_file,
                    func_name=candidate.target_func.split(':')[-1],
                    line_number=m.target_line,
                    code_content=m.target_code,
                    observation=f"Mapped from reference: {m.reference_line}"
                ))
    
    # === Build defense mechanism from Round 2 Blue's defense_checks (flat list) ===
    defense_mechanism = None
    
    # Find the strongest defense that exists
    strongest_defense = None
    for check in round2_blue.defense_checks:
        if check.exists:
            strongest_defense = check
            break
    
    if strongest_defense:
        # Try to extract line number from location (format: "file:line" or just "line")
        defense_line = None
        if strongest_defense.location:
            parts = strongest_defense.location.split(':')
            if len(parts) >= 2:
                try:
                    defense_line = int(parts[-1])
                except ValueError:
                    pass
            elif parts[0].isdigit():
                defense_line = int(parts[0])
        
        defense_mechanism = StepAnalysis(
            role="Defense",
            file_path=candidate.target_file,
            func_name=candidate.target_func.split(':')[-1],
            line_number=defense_line,
            code_content=strongest_defense.code,
            observation=f"Defense type: {strongest_defense.defense_type}"
        )
    
    # Build report
    analysis_report = f"C_cons={round2_judge.c_cons_satisfied}, C_reach={round2_judge.c_reach_satisfied}, C_def={round2_judge.c_def_satisfied}. {round2_judge.verdict}"
    
    return JudgeOutput(
        c_cons_satisfied=round2_judge.c_cons_satisfied,
        c_reach_satisfied=round2_judge.c_reach_satisfied,
        c_def_satisfied=round2_judge.c_def_satisfied,
        is_vulnerable=is_vulnerable,
        verdict_category=verdict_category,
        anchor_evidence=anchor_evidence,
        trace=trace_steps,
        defense_mechanism=defense_mechanism,
        analysis_report=analysis_report
    )


ROUND3_FINAL_JUDGE_PROMPT = """You are the **Final Judge** for Round 3: Verdict Synthesis (§3.3.3).

Your task is to synthesize all evidence from Round 1 (§3.3.1 Anchor Mapping), Path Verification (§3.3.2),
and Round 2 (§3.3.3 Multi-Agent Semantic Analysis) into a final, structured verdict following the JudgeOutput schema.

## YOUR RESPONSIBILITIES

### 1. Constraint Summary
Summarize the constraint outcomes from previous rounds:
- **C_cons**: Was consistency established in Round 1 (anchor mapping)?
- **C_reach**: Was reachability established via static path verification (§3.3.2) and semantic analysis (Round 2)?
- **C_def**: Was a defense found in Round 2?

### 2. Attack Path Type Validation
Verify that the identified attack path corresponds to the original vulnerability type:
- If the reference is UAF, the target should also exhibit UAF pattern (ALLOC→DEALLOC→USE)
- If patterns don't match, set attack_path_matches_vuln_type = false

### 3. Location Validation
Check if at least one anchor in the typed chain is within the current target function:
- Use the function name from context
- If ALL anchors are in external functions → SAFE-OutOfScope

### 4. Final Verdict
Apply the decision logic:
- VULNERABLE: C_cons ∧ C_reach ∧ ¬C_def ∧ attack_path_matches_vuln_type ∧ (any anchor in function)
- SAFE-Blocked: C_cons ∧ C_reach ∧ C_def
- SAFE-Mismatch: ¬C_cons
- SAFE-Unreachable: C_cons ∧ ¬C_reach
- SAFE-TypeMismatch: ¬attack_path_matches_vuln_type
- SAFE-OutOfScope: no anchor in current function

### 5. Evidence Chain Construction
Build the validated typed anchor evidence chain:
- **anchor_evidence**: List of StepAnalysis, one per anchor in the constraint chain
  (e.g., for UAF: [ALLOC step, DEALLOC step, USE step]), each with role set to the AnchorType value
- **trace**: List of intermediate steps between anchors (if vulnerable)
- **defense_mechanism**: StepAnalysis for defense (if SAFE-Blocked)

### 6. Report Generation
- **analysis_report**: Concise 1-2 sentence summary
- **detailed_markdown_report**: Full exploitation report with sections for Verdict, Attack Path (if vulnerable), Defense (if blocked), and Evidence Summary

## OUTPUT REQUIREMENTS
Your output MUST be a valid JudgeOutput JSON with ALL required fields properly filled.
The anchor_evidence list must have one entry per anchor in the constraint chain, with role matching the AnchorType.
Use actual line numbers from the target code context provided.
"""

class TargetSlicerAdapter:
    def __init__(self, code: str, lang: str = "c"):
        self.code = code
        self.pdg = PDGBuilder(code, lang=lang).build()
        self.slicer = Slicer(self.pdg, code)

    def slice_context(self, anchors_lines: List[int], hint_vars: List[str]) -> str:
        """
        Slice based on line numbers (Phase 3) or variables (Phase 2)
        """
        anchor_nodes = set()
        
        # Strategy A: Priority to use exact line numbers provided by Phase 3
        for ln in anchors_lines:
            # get_nodes_by_location needs to support line number only
            nodes = self.slicer.get_nodes_by_location(ln, "")
            anchor_nodes.update(nodes)
            
        # Strategy B: If no line numbers, fallback to variable names
        if not anchor_nodes and hint_vars:
            for n, d in self.pdg.nodes(data=True):
                code_snippet = d.get('code', '')
                if any(v in code_snippet for v in hint_vars):
                    anchor_nodes.add(n)
        
        if not anchor_nodes:
            return self.code # Slice failed fallback

        # Execute Robust Slice
        focus_vars = set(hint_vars) if hint_vars else set()
        sliced_nodes = self.slicer.robust_slice(list(anchor_nodes), focus_vars)
        
        # Critical Safety Check: If slice is too short (e.g. only 10% left), Context might be cut off, force fallback
        sliced_code = self.slicer.to_code(sliced_nodes)
        if len(sliced_code) < len(self.code) * 0.2:
            return self.code # Safe fallback
            
        return sliced_code

# ==============================================================================
# §3.3.2 Path Verification (Pure Static, No LLM)
# ==============================================================================

def run_path_verification(
    round1_judge: 'Round1JudgeOutput',
    target_code: str,
    feature: 'PatchFeatures',
    candidate: 'SearchResultItem',
    func_start_line: int = 1
) -> PathVerificationResult:
    """
    §3.3.2 Path Verification — Pure static analysis, NO LLM involvement.
    
    Per the paper §3.3.2, this step iterates over the constraint chain R
    defined by the vulnerability type (e.g., UAF: alloc →d dealloc →t use),
    and for each ChainLink (a_i, a_j, δ), checks whether the expected
    dependency type δ exists between the mapped positions μ(a_i) and μ(a_j)
    in the target code's PDG/CFG.
    
    Dependency types:
    - DATA (→d): Check DATA edge paths in PDG
    - TEMPORAL (→t): Check CFG reachability (sequential execution order)
    - CONTROL (→c): Check CONTROL edge paths in PDG
    
    Args:
        round1_judge: Round 1 Judge output with validated anchor mappings
        target_code: The target function code
        feature: Patch features for vulnerability type info (contains constraint chain)
        candidate: Search result candidate
    
    Returns:
        PathVerificationResult with per-pair reachability info
    """
    validated_anchors = round1_judge.validated_anchors or []
    
    # Collect mapped target lines from validated anchors
    mapped_lines = []
    for m in validated_anchors:
        if m.is_mapped and m.target_line is not None:
            mapped_lines.append(m.target_line)
    
    # --- Step 1: Extract constraint chain from vulnerability type ---
    constraint_chain = []
    try:
        constraint = feature.taxonomy.constraint
        if constraint and constraint.chain:
            constraint_chain = constraint.chain  # List[ChainLink]
    except Exception as e:
        print(f"    [PathVerification] Could not extract constraint chain: {e}")
    
    # Handle empty constraint chain (e.g., Control-Logic types like LogicError)
    # These vulnerability types define reachability-only constraints with no
    # explicit anchor pairs — skip path verification and pass through.
    if not constraint_chain:
        return PathVerificationResult(
            reachable=True,
            path_type="none",
            details="Empty constraint chain — reachability-only vulnerability type, path verification skipped",
            anchor_lines_checked=mapped_lines,
            skipped=True,
            skip_reason="Empty constraint chain (reachability-only type)"
        )
    
    # --- Step 2: Build anchor_type → AnchorMapping lookup ---
    # AnchorMapping.anchor_type is a string like 'alloc', 'dealloc', 'use'
    # ChainLink.source/target are AnchorType enum with .value like 'alloc', etc.
    anchor_map = {}
    for m in validated_anchors:
        if m.is_mapped and m.target_line is not None:
            key = m.anchor_type.lower().strip()
            # If multiple mappings for same type, keep the first (most confident)
            if key not in anchor_map:
                anchor_map[key] = m
    
    if len(anchor_map) < 2:
        return PathVerificationResult(
            reachable=False,
            path_type="none",
            details=f"Insufficient mapped anchors for path verification (need >= 2, got {len(anchor_map)})",
            anchor_lines_checked=mapped_lines,
            skipped=True,
            skip_reason="Insufficient mapped anchors"
        )
    
    # --- Step 3: Build PDG and obtain CFG ---
    try:
        pdg_builder = PDGBuilder(target_code, lang="c")
        pdg = pdg_builder.build()
        cfg = pdg_builder.cfg  # CFG for temporal dependency checks
        slicer = Slicer(pdg, target_code, start_line=func_start_line)
    except Exception as e:
        print(f"    [PathVerification] PDG/CFG build failed: {e}")
        return PathVerificationResult(
            reachable=False,
            path_type="none",
            details=f"PDG/CFG construction failed: {str(e)}",
            anchor_lines_checked=mapped_lines,
            skipped=True,
            skip_reason=f"PDG build error: {str(e)}"
        )
    
    # --- Step 4: Iterate over constraint chain, check each pair ---
    pair_results = []
    
    for link in constraint_chain:
        src_type = link.source.value.lower().strip()
        tgt_type = link.target.value.lower().strip()
        dep_type = link.dependency  # DependencyType enum
        is_optional = link.is_optional
        
        src_mapping = anchor_map.get(src_type)
        tgt_mapping = anchor_map.get(tgt_type)
        
        # Handle unmapped anchors
        if not src_mapping or not tgt_mapping:
            missing = []
            if not src_mapping:
                missing.append(f"source '{src_type}'")
            if not tgt_mapping:
                missing.append(f"target '{tgt_type}'")
            
            pair_result = PairCheckResult(
                source_type=src_type,
                target_type=tgt_type,
                dependency_type=dep_type.value if hasattr(dep_type, 'value') else str(dep_type),
                reachable=False,
                is_optional=is_optional,
                skipped=True,
                skip_reason=f"Unmapped anchor(s): {', '.join(missing)}",
                details=f"Cannot check {src_type} →{dep_type.value[0].lower() if hasattr(dep_type, 'value') else '?'} {tgt_type}: anchor(s) not mapped"
            )
            pair_results.append(pair_result)
            print(f"    [PathVerification] Pair {src_type}→{tgt_type}: SKIPPED (unmapped: {', '.join(missing)})")
            continue
        
        # Perform the actual dependency check
        try:
            result = check_single_pair(
                src_line=src_mapping.target_line,
                src_code=src_mapping.target_code or '',
                tgt_line=tgt_mapping.target_line,
                tgt_code=tgt_mapping.target_code or '',
                dep_type=dep_type,
                pdg=pdg,
                cfg=cfg,
                slicer=slicer
            )
            
            pair_result = PairCheckResult(
                source_type=src_type,
                target_type=tgt_type,
                dependency_type=dep_type.value if hasattr(dep_type, 'value') else str(dep_type),
                source_line=src_mapping.target_line,
                target_line=tgt_mapping.target_line,
                reachable=result["reachable"],
                path_type=result["path_type"],
                details=result["details"],
                is_optional=is_optional
            )
            pair_results.append(pair_result)
            
            dep_symbol = dep_type.value[0].lower() if hasattr(dep_type, 'value') else '?'
            status = "REACHABLE" if result["reachable"] else "UNREACHABLE"
            print(f"    [PathVerification] Pair {src_type} →{dep_symbol} {tgt_type} "
                  f"(L{src_mapping.target_line}→L{tgt_mapping.target_line}): {status}")
            
        except Exception as e:
            pair_result = PairCheckResult(
                source_type=src_type,
                target_type=tgt_type,
                dependency_type=dep_type.value if hasattr(dep_type, 'value') else str(dep_type),
                source_line=src_mapping.target_line,
                target_line=tgt_mapping.target_line,
                reachable=False,
                is_optional=is_optional,
                skipped=True,
                skip_reason=f"Check error: {str(e)}",
                details=f"Error checking {src_type}→{tgt_type}: {str(e)}"
            )
            pair_results.append(pair_result)
            print(f"    [PathVerification] Pair {src_type}→{tgt_type}: ERROR ({e})")
    
    # --- Step 5: Aggregate results ---
    # All REQUIRED (non-optional) pairs must be reachable or skipped-due-to-unmapped
    required_pairs = [p for p in pair_results if not p.is_optional]
    
    if not required_pairs:
        # All pairs are optional — pass through
        return PathVerificationResult(
            reachable=True,
            path_type="none",
            details="All constraint chain pairs are optional — path verification passed",
            anchor_lines_checked=mapped_lines,
            skipped=True,
            skip_reason="All pairs optional",
            pair_results=pair_results
        )
    
    # A required pair fails if: not reachable AND not skipped (i.e., actually checked and failed)
    failed_required = [p for p in required_pairs if not p.reachable and not p.skipped]
    # A required pair is skipped if: skipped=True (unmapped anchor or check error)
    skipped_required = [p for p in required_pairs if p.skipped]
    passed_required = [p for p in required_pairs if p.reachable and not p.skipped]
    
    all_reachable = len(failed_required) == 0
    
    # Determine summary path type
    path_types_found = set()
    for p in pair_results:
        if p.reachable and p.path_type != "none":
            path_types_found.add(p.path_type)
    
    if len(path_types_found) == 0:
        summary_path_type = "none"
    elif len(path_types_found) == 1:
        summary_path_type = path_types_found.pop()
    else:
        summary_path_type = "mixed"
    
    # Build details string
    detail_parts = []
    for p in pair_results:
        dep_sym = p.dependency_type[0].lower() if p.dependency_type else '?'
        status = "✓" if p.reachable else ("⊘" if p.skipped else "✗")
        optional_tag = " (optional)" if p.is_optional else ""
        detail_parts.append(f"{status} {p.source_type} →{dep_sym} {p.target_type}{optional_tag}")
    
    summary = f"Chain verification: {len(passed_required)}/{len(required_pairs)} required pairs passed"
    if skipped_required:
        summary += f", {len(skipped_required)} skipped"
    if failed_required:
        summary += f", {len(failed_required)} FAILED"
    details = f"{summary}. [{', '.join(detail_parts)}]"
    
    return PathVerificationResult(
        reachable=all_reachable,
        path_type=summary_path_type,
        details=details,
        anchor_lines_checked=mapped_lines,
        pair_results=pair_results
    )


# ==============================================================================
# §3.3.2-E: LLM Override for Path Verification (Solution E)
# ==============================================================================

class PathLLMOverrideOutput(BaseModel):
    """Output schema for LLM path override confirmation."""
    anchor_relationships_hold: bool = Field(
        description="Whether the anchor dependency relationships plausibly hold in the target code, "
                    "even though static PDG analysis could not confirm them"
    )
    reasoning: str = Field(
        description="Explanation of why the relationships hold or don't hold, "
                    "referencing specific code patterns (e.g., loop-driven data flow, indirect dependencies)"
    )
    confidence: str = Field(
        default="medium",
        description="Confidence level: 'high', 'medium', or 'low'"
    )


def run_path_llm_override(
    path_result: PathVerificationResult,
    feature: 'PatchFeatures',
    candidate: 'SearchResultItem',
    target_code: str,
    llm: 'ChatOpenAI',
    func_start_line: int = 1
) -> bool:
    """
    §3.3.2-E: LLM second-confirmation when static path verification fails.
    
    This function is called ONLY when:
    1. Static path verification (§3.3.2) says unreachable
    2. The failure involves DATA dependency pairs
    
    It asks the LLM to confirm whether the anchor dependency relationships
    from the original vulnerability plausibly hold in the target code,
    considering patterns that static analysis misses (e.g., loop-driven
    data flow where variables are connected through control flow rather
    than direct data edges).
    
    Cost: 1 LLM call (lightweight, no tools).
    
    Args:
        path_result: The failed PathVerificationResult with pair_results
        feature: PatchFeatures containing original vulnerability anchors
        candidate: SearchResultItem with target function info
        target_code: Target function source code
        llm: LLM instance for the override check
        func_start_line: Starting line number of the target function
    
    Returns:
        True if LLM confirms relationships hold (override the static failure),
        False if LLM agrees with static analysis (keep SAFE-Unreachable)
    """
    print("    [§3.3.2-E] LLM override check for failed static path verification...")
    
    # --- Build context about the failed pairs ---
    failed_pairs_desc = []
    for pr in path_result.pair_results:
        if not pr.reachable and not pr.skipped:
            failed_pairs_desc.append(
                f"  - {pr.source_type} →{pr.dependency_type[0].lower()} {pr.target_type} "
                f"(L{pr.source_line}→L{pr.target_line}): FAILED — {pr.details}"
            )
    failed_pairs_str = "\n".join(failed_pairs_desc) if failed_pairs_desc else "  [No failed pairs]"
    
    # --- Build original vulnerability anchor descriptions ---
    sf = feature.slices.get(candidate.patch_func)
    ref_anchor_lines = []
    if sf and sf.pre_anchors:
        for anchor in sf.pre_anchors:
            atype = anchor.type.value if hasattr(anchor.type, 'value') else str(anchor.type)
            snippet = anchor.code_snippet.strip()[:80] if anchor.code_snippet else 'N/A'
            vars_str = ", ".join(anchor.variable_names) if anchor.variable_names else "N/A"
            reasoning = anchor.reasoning[:120] if anchor.reasoning else "N/A"
            ref_anchor_lines.append(
                f"  [{atype.upper()}] L{anchor.line_number}: `{snippet}`\n"
                f"    Variables: {vars_str}\n"
                f"    Role: {reasoning}"
            )
    ref_anchors_str = "\n".join(ref_anchor_lines) if ref_anchor_lines else "  [No reference anchors]"
    
    # --- Get constraint chain description ---
    constraint_chain_str = ""
    try:
        constraint = feature.taxonomy.constraint
        if constraint:
            constraint_chain_str = constraint.chain_str()
    except Exception:
        pass
    
    # --- Add line numbers to target code ---
    target_lines = target_code.splitlines()
    numbered_lines = []
    for i, line in enumerate(target_lines):
        line_no = func_start_line + i
        numbered_lines.append(f"[{line_no:4d}] {line}")
    target_code_with_lines = "\n".join(numbered_lines)
    
    # --- Build the LLM prompt ---
    user_input = f"""## Static Path Verification Failed — LLM Override Check

### Context
Static PDG analysis (§3.3.2) could not confirm dependency paths between certain anchor pairs
in the target code. However, static analysis has known limitations — it may miss indirect
dependencies such as:
- **Loop-driven data flow**: Variable A controls a loop condition, variable B is modified
  inside the loop body. They are connected through CONTROL flow, not direct DATA edges.
- **Pointer aliasing**: Two variables point to the same memory but static analysis doesn't
  track the alias.
- **Callback/indirect calls**: Data flows through function pointers or callbacks.

### Failed Anchor Pairs (Static Analysis)
{failed_pairs_str}

### Original Vulnerability Anchors (Reference)
**Constraint Chain**: {constraint_chain_str or 'Not available'}
**Vulnerability Type**: {feature.semantics.vuln_type.value if feature.semantics.vuln_type else 'Unknown'}
**Root Cause**: {feature.semantics.root_cause}

**Reference Anchors**:
{ref_anchors_str}

### Target Code
```c
{target_code_with_lines}
```

### Your Task
Examine the target code and determine whether the failed anchor dependency relationships
**plausibly hold** despite the static analysis failure. Consider:

1. Are the anchor variables connected through any indirect mechanism (loop, control flow,
   shared state, pointer aliasing)?
2. Does the target code exhibit the same vulnerability pattern as the reference?
3. Could the static analysis have missed the connection due to its known limitations?

**IMPORTANT**: Be conservative. Only confirm if you can identify a clear mechanism
connecting the anchors. If the variables are truly independent, confirm the static
analysis result (relationships do NOT hold).
"""

    messages = [
        SystemMessage(content="You are a vulnerability analysis expert. Your task is to determine whether "
                     "anchor dependency relationships hold in target code when static analysis fails to confirm them. "
                     "Output ONLY valid JSON matching the provided schema."),
        HumanMessage(content=user_input)
    ]
    
    # Request structured output
    try:
        schema_dict = PathLLMOverrideOutput.model_json_schema()
        schema_str = json.dumps(schema_dict, indent=2)
    except Exception:
        schema_str = "Use the known JSON schema for PathLLMOverrideOutput."
    
    messages.append(HumanMessage(content=f"\nJSON SCHEMA:\n{schema_str}\n\nOutput ONLY the raw JSON."))
    
    # Single LLM call with retry
    try:
        ai_msg = llm_invoke_with_retry(llm, messages, max_retries=2, retry_delay=3.0)
        
        if ai_msg is None:
            print("    [§3.3.2-E] LLM call failed, keeping static result (SAFE-Unreachable)")
            return False
        
        raw_content = ai_msg.content
        result = robust_json_parse(raw_content, PathLLMOverrideOutput, "PathLLMOverride", use_fallback=False)
        
        if result is None:
            print("    [§3.3.2-E] JSON parsing failed, keeping static result (SAFE-Unreachable)")
            return False
        
        override = result.anchor_relationships_hold
        print(f"    [§3.3.2-E] LLM override result: relationships_hold={override}, "
              f"confidence={result.confidence}, reason={result.reasoning[:100]}...")
        
        # Only override if confidence is not 'low'
        if override and result.confidence == "low":
            print("    [§3.3.2-E] Override rejected: confidence too low")
            return False
        
        return override
        
    except Exception as e:
        print(f"    [§3.3.2-E] LLM override error: {e}, keeping static result")
        return False


# ==============================================================================
# 2. Prompts (Updated: §3.3 Anchor-based Constraint Model)
# ==============================================================================


# ==============================================================================
# Round 1: C_cons Validation Prompts (No Tools)
# ==============================================================================

ROUND1_RED_PROMPT = """You are the **Red Agent** performing **Round 1: Anchor Mapping (§3.3.1)**.
Your task is to produce the mapping μ: M → T, determining whether each reference anchor
has a semantically equivalent counterpart in the target code.
You have **NO TOOLS** available - analyze ONLY the provided information.

## INPUTS PROVIDED
1. **Phase 2 Typed Anchors**: Category-specific anchors (e.g., ALLOC/DEALLOC/USE for UAF),
   each with a **locatability level** (CONCRETE / ASSUMED / CONCEPTUAL)
2. **Constraint Chain**: Formal dependency chain (e.g., `alloc →d dealloc →t use`)
3. **Phase 3 Mapping Results**: Tiered alignment (Tier 1/2/3) showing which anchors have
   candidate target lines and which need discovery
4. **Target Code**: The actual target function code with line numbers
5. **Root Cause Description**: Guides what counts as "functional equivalence"

## YOUR TASK: Produce Anchor Mapping μ (Mapping Function)

### Step 1: Process CONCRETE Anchors (Tier 1)
These have direct code locations in the reference slice and Phase 3 alignment hints.
For each CONCRETE anchor:
- Check the Phase 3 alignment quality (good/weak/missing)
- If mapped: verify the target line fulfills the SAME SEMANTIC ROLE
  (guided by root cause, not syntactic identity)
- If unmapped/weak: search target code for a semantically equivalent statement
- Set `semantic_role_verified = true` if the role matches
- Set `mapping_confidence` to 'high' (direct match) or 'medium' (manual search)

### Step 2: Process ASSUMED Anchors (Tier 2)
These have code locations but carry assumptions that need re-verification in target.
For each ASSUMED anchor:
- First, attempt mapping like CONCRETE (check Phase 3 alignment or manual search)
- THEN, verify whether the **assumption** still holds in the target context:
  * CONTROLLABILITY: Is the parameter/input actually controllable by attacker?
  * SEMANTIC: Does the function/operation behave the same way in target?
  * EXISTENCE: Does the assumed callee/operation actually exist in target?
  * REACHABILITY: Is the assumed code path actually reachable?
- Set `assumption_verified` to true/false based on verification
- If assumption FAILS in target → the mapping is INVALID even if code matches

### Step 3: Process CONCEPTUAL Anchors (Tier 3)
These are inferred roles with no direct code location. They CANNOT be found via Phase 3.
For each CONCEPTUAL anchor:
- Use the root cause description and vulnerability semantics to understand WHAT to look for
- Search the target code for a statement that fulfills this semantic role
- This requires deeper reasoning about code behavior, not pattern matching
- Set `mapping_confidence` to 'low' (inferred)
- If found: explain why the target code fulfills this role

### Step 4: Verify Chain Causality (CRITICAL)
After all anchors are mapped, verify the constraint chain holds:
- **Execution Order**: Can chain anchors execute in the required order?
  - Check if earlier chain elements can reach later ones in control flow
  - If a later anchor has a lower line number, verify this is valid (callbacks, macros, goto)
- **Data Flow**: Does the state flow through the chain?
  - Same variable or connected variables throughout the path?
- **Common Invalid Patterns**:
  - Both anchors are cleanup code (no vulnerable use)
  - Different variables with no flow connection
  - Control flow (return/goto/exit) blocks chain traversal

### Step 5: C_cons Decision
**CRITICAL RULES**:
- All NON-OPTIONAL anchors must be mapped for C_cons to hold
- Optional anchors (is_optional=true) can be missing without failing C_cons
- Mappings must be SEMANTICALLY equivalent (same role), not just syntactically similar
- For ASSUMED anchors: assumption must be verified in target context
- Chain causality must be valid
- If any critical (non-optional) anchor is unmapped → verdict = SAFE
- If chain causality fails → verdict = SAFE
- If all critical anchors mapped with valid causality → verdict = PROCEED

## SEMANTIC ROLE MATCHING (Guided by Root Cause)
The root cause description tells you WHAT vulnerability pattern to look for.
Match the ROLE, not the exact code:
- An `alloc` anchor = any statement that ALLOCATES the resource in question
- A `dealloc` anchor = any statement that RELEASES/FREES the resource
- A `use` anchor = any statement that ACCESSES the resource after dealloc
- A `source` anchor = where the problematic value ORIGINATES
- A `sink` anchor = where the problematic value is CONSUMED dangerously

## OUTPUT REQUIREMENTS
For EACH anchor in the constraint chain, output an AnchorMapping with:
1. `anchor_type`: The anchor's type (e.g., 'alloc', 'use', 'source')
2. `locatability`: The anchor's locatability level
3. `assumption_type` / `assumption_rationale`: From Phase 2 (if applicable)
4. `target_line` / `target_code`: The mapped target location
5. `is_mapped`: Whether mapping succeeded
6. `mapping_confidence`: 'high', 'medium', or 'low'
7. `semantic_role_verified`: Whether the semantic role was verified
8. `assumption_verified` / `assumption_verification_note`: For ASSUMED/CONCEPTUAL anchors
9. `reason`: Semantic equivalence explanation
"""

ROUND1_BLUE_PROMPT = """You are the **Blue Agent** performing **Round 1: Anchor Mapping Refutation (§3.3.1)**.
Your task is to challenge Red's anchor mapping μ: M → T.
You have **NO TOOLS** available - analyze ONLY the provided information.

## INPUTS PROVIDED
1. **Red's Anchor Mappings**: Typed anchor mappings (per constraint chain) with locatability info
2. **Target Code**: The actual target function code with line numbers
3. **Vulnerability Semantics**: Type, root cause, and attack path

## YOUR TASK: Refute Anchor Mappings if Invalid

### Mode 1: Refute Semantic Role Mismatch
For any anchor mapping Red claims is valid:
- The target statement does NOT fulfill the claimed semantic role
  (e.g., Red maps a non-allocation to an ALLOC anchor)
- The variable serves a DIFFERENT role in data flow
- The data type or context is fundamentally different

### Mode 2: Refute Assumption Failure (for ASSUMED/CONCEPTUAL anchors)
For anchors with locatability='assumed' or 'conceptual':
- The assumption does NOT hold in the target context
  * CONTROLLABILITY: The input is NOT actually attacker-controllable
  * SEMANTIC: The function behaves DIFFERENTLY in the target
  * EXISTENCE: The assumed callee/operation does NOT exist
  * REACHABILITY: The assumed path is NOT reachable
- Red failed to verify the assumption properly

### Mode 3: Refute Chain Causality
If the constraint chain causality is invalid:
- **Execution Order**: Later chain anchors cannot be reached from earlier ones
  * Check line numbers: If a later anchor has a lower line number, is this valid?
  * Valid cases: callbacks, macros, goto/error handling
  * Invalid cases: simple sequential code
- **Data Flow**: The state does NOT flow through the chain
  * Different variables with no connection
  * Both are cleanup code (no vulnerable use)
  * Control flow blocks chain traversal (return/goto/exit between anchors)

### Mode 4: Mechanism Mismatch
If the target has a fundamentally different vulnerability mechanism:
- Red's category mapping is wrong (e.g., UAF vs NPD)
- The data flow pattern is completely different
- The control flow structure prevents the claimed vulnerability

## REFUTATION REQUIREMENTS
- **Be Specific**: Quote exact code and line numbers
- **Be Semantic**: Explain WHY the role mapping is wrong, guided by root cause
- **Be Honest**: If Red's mapping is actually correct, CONCEDE rather than fabricating issues
- **Target ASSUMED anchors**: These are the most likely to have invalid assumptions in new context

## VERDICT OPTIONS
- **SAFE**: You successfully refuted C_cons (provide strong evidence)
- **CONCEDE**: Red's anchor mappings are valid, you cannot refute them
- **CONTESTED**: You raised valid concerns but they need tool verification in Round 2

## OUTPUT REQUIREMENTS
Provide a complete structured output with refutation details for each anchor and overall verdict.
"""

ROUND1_JUDGE_PROMPT = """You are the **Judge** adjudicating **Round 1: Anchor Mapping Decision (§3.3.1)**.
Your task is to evaluate Red's anchor mapping μ and Blue's refutation to decide if C_cons is satisfied.

## INPUTS PROVIDED
1. **Phase 2 Reference Anchor Definition**: The GROUND TRUTH anchor layout from the original vulnerability analysis
2. **Red's Anchor Mappings**: Typed anchor mappings per constraint chain, with locatability and assumption info
3. **Blue's Refutation**: Challenges to Red's mappings (semantic role, assumption, causality)
4. **Target Code**: For verification
5. **Vulnerability Semantics**: For context

## CRITICAL: Reference Anchor Definition is Ground Truth

The **Phase 2 Reference Anchor Definition** shows the original vulnerability's anchor layout.
This is the GROUND TRUTH — the target mapping should mirror this layout. Key implications:

- **If two anchors share the same line in the reference** (e.g., COMPUTATION and SINK both at L2139),
  then the target mapping SHOULD also have them on the same line. This is NOT a mismatch.
- **CONCEPTUAL anchors** (e.g., a SINK that manifests in a different function) may not have a direct
  code location in the target function. If the reference SINK is conceptual with assumption_type=existence,
  it means the actual sink is in a callee/caller — the target only needs to store the value that
  eventually reaches the sink. Mapping it to the same storage line as COMPUTATION is CORRECT.
- **Do NOT penalize mappings that faithfully reproduce the reference anchor layout.**

## YOUR TASK: Adjudicate C_cons (Anchor Mapping Quality)

### Evaluation Criteria
1. **Semantic Role Validity**: Does each mapped target statement fulfill the declared semantic role?
   - Guided by root cause description, not syntactic similarity
   - CONCRETE anchors: Should have high confidence mappings
   - ASSUMED anchors: Must verify assumption in target context
   - CONCEPTUAL anchors: Inferred mappings need strong reasoning
   - **Compare against reference anchor definition** — if the reference has a conceptual SINK
     at the same line as COMPUTATION, the target mapping should follow the same pattern

2. **Assumption Verification** (for ASSUMED/CONCEPTUAL anchors):
   - Did Red properly verify the assumption?
   - Did Blue successfully show the assumption fails in target?
   - Key assumption types to check: CONTROLLABILITY, SEMANTIC, EXISTENCE, REACHABILITY
   - **EXISTENCE assumptions**: If the reference says "sink exists in callee", check if the
     target has a similar callee that uses the stored value

3. **CHAIN CAUSALITY CHECK (CRITICAL)**:
   - Can chain anchors execute in the required order?
   - Does state flow through the chain? (same variable or connected variables)
   - RED FLAG Patterns:
     * Later anchor has lower line number (check if valid: callbacks, macros, goto)
     * Both anchors are cleanup/deallocation (no vulnerable use)
     * Different variables with no flow connection
   - If Blue raises "Causality Violation" and it's valid → C_cons NOT satisfied
   - **NOTE**: Two anchors on the same line is NOT a causality violation if the reference
     defines them that way (e.g., an assignment is both computation and storage-to-sink)

4. **Optional Anchors**: Missing optional anchors do NOT fail C_cons

### Decision Logic
- If any critical (non-optional) anchor is unmapped → C_cons NOT satisfied
- If Blue successfully refutes a critical anchor's semantic role → C_cons NOT satisfied
- If Blue shows an ASSUMED anchor's assumption fails → C_cons NOT satisfied
- If chain causality fails → C_cons NOT satisfied
- If Blue's refutation is weak/speculative → Give benefit of doubt to Red
- If mappings are valid but Blue raises tool-dependent concerns → PROCEED to Round 2
- **If Blue claims "anchor X not found" but the reference defines it as CONCEPTUAL with
  an existence assumption → this is NOT a valid refutation** (the anchor is expected to
  be in a different function)

## VERDICT OPTIONS
- **SAFE-Mismatch**: C_cons is NOT satisfied (terminate verification)
- **PROCEED**: C_cons appears satisfied or needs Round 2 verification

## OUTPUT REQUIREMENTS
Provide verdict with validated anchor mappings for Round 2.
**IMPORTANT**: If Blue refuted causality or assumption validity and it's correct,
you MUST set c_cons_satisfied=False and verdict='SAFE-Mismatch'.
**EQUALLY IMPORTANT**: If the reference anchor definition supports Red's mapping layout
(e.g., shared lines, conceptual anchors), do NOT let Blue's refutation override the ground truth.
"""

# ==============================================================================
# Round 2: C_cons Reinforcement + C_reach + C_def Prompts (With Tools)
# ==============================================================================

ROUND2_RED_PROMPT = """You are the **Red Agent** performing **Round 2: Semantic Analysis — VIOL Argument (§3.3.3)**.
Your task is to argue that the violation predicate (VIOL) holds AND that no mitigation exists (¬MITIGATED).
You have **TOOLS** available to verify your claims.

## TOOL USAGE BEST PRACTICES

**PREFER `find_definition` over `read_file`**:
- Use `find_definition(symbol_name, file_path)` to get the COMPLETE function/struct definition
- This returns the full symbol content in one call, reducing redundant tool invocations
- Only use `read_file` when you need to analyze a SPECIFIC code range that is NOT a complete symbol
  (e.g., checking guard conditions around a specific line, reading context before/after a function)

**Examples**:
✓ GOOD: `find_definition("kfree", "file.c")` - Gets complete kfree implementation
✗ BAD: `read_file("file.c", 100, 150)` then `read_file("file.c", 150, 200)` - Multiple calls for one function

**When to use `read_file`**:
- Reading code BEFORE a function starts (e.g., checking includes, macros)
- Reading code BETWEEN two functions
- Analyzing a specific block that spans multiple symbols

## INPUTS PROVIDED
1. **Round 1 Results**: Validated typed anchor mappings and any contested points
2. **Path Verification (§3.3.2)**: Static PDG analysis result (reachable/unreachable)
3. **Reference Attack Path**: Step-by-step trace from the vulnerability analysis
4. **Target Code**: With line numbers
5. **Vulnerability Semantics**: Type, root cause, constraint chain

## YOUR TASK

### Task 1: Reinforce C_cons (if Round 1 had disputes)
If Blue contested your anchor mappings in Round 1:
- Use tools to verify the mapped statements
- Provide additional evidence for semantic equivalence
- Show data flow connections between typed anchors in the constraint chain

### Task 2: Establish VIOL (Violation Predicate) — Core Task
VIOL requires proving the attack path through the **typed anchor chain** is semantically consistent with the reference:

**What to verify**:
1. **Chain Completeness**: Every anchor in the constraint chain has a valid target mapping
2. **Semantic Consistency**: Each anchor step performs the SAME operation as reference
3. **Data Flow Continuity**: Same variable flows through the entire chain
4. **Control Flow Feasibility**: No unconditional blockers (guards, early returns)
5. **CAUSALITY**: Earlier anchors MUST causally precede later anchors (check data/control flow, not just line numbers)

**For each attack path step, you must show**:
- Step anchor type (e.g., ALLOC, DEALLOC, USE, SOURCE, SINK) or flow role (Trace, Branch, Call)
- Target line number and code
- Corresponding reference step
- Evidence of semantic match (e.g., both do allocation, both do free)

**CRITICAL - CAUSALITY REQUIREMENT**:
- **Earlier anchor → Later anchor MUST have causal relationship**:
  * The first anchor creates/modifies the vulnerable state
  * The last anchor uses/triggers that state
  * There must be a data/control flow path connecting them through intermediate anchors
- **Line numbers can be misleading** (macros, cross-function calls, callbacks)
- **But data flow MUST be valid**: The effect of each anchor must reach the next

### Attack Path Step Types (Typed Anchors)
- Use the AnchorType from the constraint chain (e.g., ALLOC, DEALLOC, USE for UAF)
- **Trace**: Intermediate steps that transform/propagate the state between anchors
- **Branch**: Conditional that enables the vulnerable path
- **Call**: Function call that continues the chain (verify callee if needed)

## TOOLS TO USE
- `find_definition(symbol, file)` - Get callee implementation to verify internal behavior
- `trace_variable(file, line, var, direction)` - Track data flow forward/backward
- `get_guard_conditions(file, line)` - Check path guards
- `grep(pattern, file, mode)` - Find all uses of a variable

## OUTPUT REQUIREMENTS
1. **c_cons_reinforced**: Did you strengthen C_cons evidence?
2. **attack_path_steps**: List of AttackPathStep with full details (using AnchorType values)
3. **path_matches_reference**: Does the path match reference semantically?
4. **data_flow_verified**: Is data flow correct?
5. **control_flow_feasible**: Is the path reachable?
6. **c_reach_satisfied**: Final C_reach verdict
7. **verdict**: 'VULNERABLE' or 'NOT_VULNERABLE'
"""

ROUND2_BLUE_PROMPT = """You are the **Blue Agent** performing **Round 2: Semantic Analysis — Challenge VIOL + Find MITIGATED (§3.3.3)**.
Your task is to challenge Red's VIOL argument OR find mitigations/defenses that block the attack.
You have **TOOLS** available.

## TOOL USAGE BEST PRACTICES

**PREFER `find_definition` over `read_file`**:
- Use `find_definition(symbol_name, file_path)` to get the COMPLETE function/struct definition
- This returns the full symbol content in one call, reducing redundant tool invocations
- Only use `read_file` when you need to analyze a SPECIFIC code range that is NOT a complete symbol

**Examples**:
✓ GOOD: `find_definition("validate_input", "file.c")` - Check if callee has defense
✗ BAD: Multiple `read_file` calls to piece together a function

## INPUTS PROVIDED
1. **Red's Round 2 Claims**: VIOL argument, attack path steps, C_reach verdict
2. **Reference Fix**: What the patch does
3. **Path Verification (§3.3.2)**: Static PDG analysis result
4. **Target Code**: With line numbers

## YOUR TASK

### Mode 1: Challenge VIOL (Refute C_cons/C_reach)
If Red's claims are invalid:
- **Refute C_cons**: Show anchor mapping is semantically wrong (different operation, different role)
- **Refute C_reach**: Show path blockers exist (guards, early returns, dead code)
- **Refute CAUSALITY**: Check if earlier anchors → later anchors lack causal relationship
  * **CRITICAL CHECK**: Does the first anchor's effect actually reach the last anchor?
  * Use `trace_variable` to verify data flow through the typed anchor chain
  * Check execution order: Can anchors execute in chain order in any feasible path?
  * Common causality error patterns:
    - **Reversed order**: Effect happens after cause (e.g., allocation after use)
    - **Blocked path**: Control flow prevents reaching later anchor (e.g., exception/return between them)
    - **Different variables**: First anchor affects X, last anchor uses Y (no connection)
    - **Cleanup confusion**: Both anchors are cleanup code (not vulnerable use)
  * If anchor chain lacks causal relationship → REFUTE and explain why

### Mode 2: Verify C_def (EXPLICIT Defense Check)
You MUST provide a **complete defense check report** covering ALL four defense types.
For EACH defense type, report one of:
- **CHECKED**: You verified this defense type. State if defense EXISTS or NOT_EXISTS.
- **NOT_APPLICABLE**: This defense type doesn't apply (explain why).
- **PENDING**: You couldn't check this yet (explain what's needed).

## FOUR-LAYER DEFENSE STRATEGY (Check in Order)

### 1. ExactPatch
Check if target contains the SAME fix as reference patch:
- Look for identical or near-identical code patterns
- Compare with the reference post-patch code

### 2. EquivalentDefense
Check for different implementation achieving same protection:
- Different function names but same effect (e.g., `spin_lock` vs `mutex_lock`)
- Different constants but same protection (e.g., `len > 100` vs `len > MAX`)
- Wrapper functions that internally apply the defense

### 3. CalleeDefense
Check if a callee function contains the defense:
- Use `find_definition` to read callee implementations
- Look for NULL checks, bounds validation, lock protection inside callees
- **MUST SHOW THE CALLEE CODE** - no assumptions allowed

### 4. CallerDefense
Check if the caller performs validation before calling the vulnerable function:
- Use `get_callers` to find call sites
- Look for guards before the function call
- **MUST SHOW THE CALLER CODE** - no assumptions allowed

## DEFENSE VERIFICATION REQUIREMENTS
For each defense you find, you MUST prove:
1. **EXISTS**: Quote exact line number and code
2. **DOMINATES**: Defense executes BEFORE the last anchor in the chain
3. **BLOCKS**: Defense actually prevents the vulnerability (explain how)

## OUTPUT REQUIREMENTS (CRITICAL - Follow Exactly)

### Defense Report Structure
You MUST fill out the `defense_report` field with ALL four checks:

```json
{
  "defense_report": {
    "exact_patch_check": {
      "defense_type": "ExactPatch",
      "check_status": "CHECKED",  // or "NOT_APPLICABLE" or "PENDING"
      "defense_exists": true,     // or false, or null if not checked
      "defense_location": "file.c:123",
      "defense_code": "if (ptr == NULL) return;",
      "dominates_last_anchor": true,
      "blocks_attack": true,
      "evidence": "This check prevents NULL dereference at line 150"
    },
    "equivalent_defense_check": { ... },
    "callee_defense_check": { ... },
    "caller_defense_check": { ... },
    "any_defense_found": true,
    "strongest_defense": "ExactPatch",
    "defense_summary": "Found exact patch defense at line 123"
  }
}
```

### Verdict Options
- **SAFE**: Found defense OR successfully refuted C_cons/C_reach
- **VULNERABLE**: No defense found AND cannot refute Red's claims
- **CONTESTED**: Raised concerns that need Round 3 resolution
"""

ROUND2_JUDGE_PROMPT = """You are the **Judge** adjudicating **Round 2: Semantic Analysis (§3.3.3)**.
Your task is to evaluate all constraints and render a verdict.

## INPUTS PROVIDED
1. **Red's Round 2 Output**: VIOL argument, typed anchor chain attack path steps, C_reach verdict
2. **Blue's Round 2 Output**: VIOL challenges and MITIGATED (defense check) report
3. **Path Verification (§3.3.2)**: Static PDG analysis result
4. **Target Code**: For verification
5. **Target Function Name**: The specific function being analyzed

## YOUR TASK: Adjudicate All Constraints

### 1. Evaluate C_cons
- Is Red's typed anchor chain mapping semantically correct?
- Did Blue successfully refute the mapping?
- **Verdict**: C_cons satisfied (True) or not (False)

### 2. Evaluate C_reach
- Consider the §3.3.2 static path verification result
- Is Red's attack path complete and semantically consistent?
- Did Blue find path blockers?
- Are all attack path steps valid?
- **Verdict**: C_reach satisfied (True) or not (False)

### 3. Evaluate C_def (MITIGATED)
Review Blue's defense check report:
- Did Blue check ALL four defense types?
- For each CHECKED defense, verify:
  - Defense code is correctly quoted
  - Defense dominates the last anchor in the chain (executes before)
  - Defense actually blocks the attack
- **Verdict**: C_def satisfied (True = defense/mitigation exists) or not (False)

### 4. Evaluate Anchor Location (CRITICAL)
Check if the anchors are within the current target function:
- **first_anchor_in_function**: Is the first anchor in the chain within the target function?
- **last_anchor_in_function**: Is the last anchor in the chain within the target function?
- At least ONE must be TRUE for the vulnerability to be relevant
- If ALL anchors are in external helper functions → SAFE-OutOfScope

## VERDICT LOGIC

```
if not C_cons:
    return "SAFE-Mismatch"
elif not C_reach:
    return "SAFE-Unreachable"
elif not attack_path_matches_vuln_type:
    return "SAFE-TypeMismatch"
elif not first_anchor_in_function and not last_anchor_in_function:
    return "SAFE-OutOfScope"  # No anchor is in the current function
elif C_def:
    return "SAFE-Blocked"
else:
    if any contested points remain:
        return "PROCEED"  # to next debate round
    else:
        return "VULNERABLE"
```

## CONTESTED POINTS (for next debate round)
If neither side provided conclusive evidence:
- List specific points that need resolution
- These will be the focus of the next debate round

## OUTPUT REQUIREMENTS
1. **c_cons_satisfied**: Boolean
2. **c_reach_satisfied**: Boolean
3. **c_def_satisfied**: Boolean
4. **attack_path_valid**: Is Red's attack path valid?
5. **first_anchor_in_function**: Is the first anchor in the chain within the current target function? (Check func_name)
6. **last_anchor_in_function**: Is the last anchor in the chain within the current target function? (Check func_name)
7. **validated_defense**: The strongest valid defense (if any)
8. **verdict**: 'SAFE-Mismatch', 'SAFE-Unreachable', 'SAFE-TypeMismatch', 'SAFE-OutOfScope', 'SAFE-Blocked', 'VULNERABLE', or 'PROCEED'
9. **contested_points**: List of unresolved issues (for next debate round)
"""

BASELINE_PROMPT = """You are an expert vulnerability analyst performing an ablation study.
Your task is to determine if the `Target Code` contains the specific vulnerability described in `Vulnerability Logic` WITHOUT using advanced tools or debate.

**INPUTS**:
1. **Vulnerability Logic**: Description of the vulnerability, root cause, and how it manifests.
2. **Reference Patch**: The fix applied in the original software (diff and sliced code).
3. **Target Code**: The code you need to analyze.

**INSTRUCTIONS**:
- Analyze the `Target Code` carefully.
- Compare it with the `Reference Patch` and `Vulnerability Logic`.
- Determine if the vulnerability logic exists in the `Target Code`.
- If the `Target Code` is missing the fix validation or check present in the reference, it is likely VULNERABLE.
- If the `Target Code` contains the fix or equivalent logic, or if the code is structured differently such that the vulnerability is not possible, it is SAFE.
- Provide a clear, step-by-step reasoning for your verdict.

**OUTPUT**:
- Provide a STRICT validated JSON output including verdict (is_vulnerable), confidence, and reasoning.
"""

# ==============================================================================
# Baseline Three-Role Debate Prompts (Simplified)
# ==============================================================================

BASELINE_RED_PROMPT = """You are the **Red Agent** in a simplified vulnerability debate.
Your goal is to PROVE that the vulnerability EXISTS in the target code.

## YOUR TASK
1. Analyze the vulnerability description (root cause, attack path)
2. Find the **first anchor** in the typed chain in target code: where vulnerable state is created (e.g. ALLOC, SOURCE)
3. Find the **last anchor** in the typed chain in target code: where vulnerability triggers (e.g. USE, SINK)
4. Explain how the attack works step by step through the anchor chain
5. If Blue has responded, address their defense arguments

## INPUTS PROVIDED
- Vulnerability Logic: Root cause, attack path, fix mechanism
- Reference Pre-Patch Code: Shows the vulnerable pattern
- Reference Post-Patch Code: Shows what the fix looks like
- Target Code: The code you need to analyze
- [Optional] Blue's Previous Refutation: If this is a follow-up round

## OUTPUT REQUIREMENTS
Provide a JSON with:
- vulnerability_exists: true/false (your claim)
- concedes: true/false (set to true if you accept Blue's defense is valid and give up)
- first_anchor_line: line number of the first anchor in the chain (e.g. ALLOC for UAF)
- first_anchor_code: the code at that line
- last_anchor_line: line number of the last anchor in the chain (e.g. USE for UAF)
- last_anchor_code: the code at that line
- attack_reasoning: step-by-step explanation of the attack OR response to Blue's refutation

## CONCEDE RULES
- If Blue has found a valid defense that you cannot counter, set concedes=true
- If Blue has shown your anchor chain mapping is fundamentally wrong, set concedes=true
- Otherwise, continue arguing your case

Be aggressive in finding vulnerabilities - that's your job!
"""

BASELINE_BLUE_PROMPT = """You are the **Blue Agent** in a simplified vulnerability debate.
Your goal is to PROVE that the vulnerability does NOT exist or is BLOCKED in the target code.

## YOUR TASK
1. Review Red's attack claim (anchor chain mapping and reasoning)
2. Find defenses: NULL checks, bounds validation, lock protection, etc.
3. Find blockers: early returns, error handling that prevents the attack path
4. Refute Red's claim if the anchor chain mapping is incorrect

## INPUTS PROVIDED
- Red's Attack Claim: anchor chain mapping and attack reasoning
- Vulnerability Logic: Root cause, attack path, fix mechanism
- Reference Post-Patch Code: Shows what the fix looks like
- Target Code: The code you need to analyze

## OUTPUT REQUIREMENTS
Provide a JSON with:
- vulnerability_exists: true/false (your verdict)
- concedes: true/false (set to true if you accept Red's attack is valid and cannot find defense)
- defense_found: true/false
- defense_line: line number of defense (if found)
- defense_code: the defense code (if found)
- refutation_reasoning: explain why vulnerability doesn't exist or is blocked

## CONCEDE RULES
- If Red's attack path is clearly valid and you cannot find any defense, set concedes=true
- If your previous defenses were correctly refuted by Red, set concedes=true
- Otherwise, continue defending

Be thorough in finding defenses - that's your job!
"""

BASELINE_JUDGE_PROMPT = """You are the **Judge** in a simplified vulnerability debate.
Your goal is to make the FINAL VERDICT based on Red and Blue arguments from multiple rounds.

## YOUR TASK
1. Review the debate history between Red and Blue
2. Evaluate Red's attack claim: Is the typed anchor chain mapping correct?
3. Evaluate Blue's defense claim: Is the defense valid and does it block the attack?
4. Consider if either side conceded
5. Make final verdict: VULNERABLE or SAFE

## DECISION LOGIC
- If Red conceded → SAFE (Red admits no vulnerability)
- If Blue conceded → VULNERABLE (Blue admits no defense)
- If Red's anchor chain mapping is incorrect → SAFE
- If Blue found a valid defense that blocks the attack → SAFE
- If Red's attack path is valid AND no defense blocks it → VULNERABLE

## INPUTS PROVIDED
- Complete Debate History (Red and Blue arguments from all rounds)
- Vulnerability Logic
- Target Code

## OUTPUT REQUIREMENTS
Provide a JSON with:
- is_vulnerable: true/false (final verdict)
- reasoning: explain your decision based on both arguments
"""


def run_baseline_red(
    feature: PatchFeatures,
    candidate,  # SearchResultItem
    target_code: str,
    tools: List[Any],
    llm: ChatOpenAI,
    round_num: int = 1,
    debate_history: Optional[List[tuple]] = None
) -> BaselineRedOutput:
    """
    Baseline Red Agent: Argue that vulnerability EXISTS.
    Uses tools to support arguments.
    
    Args:
        round_num: Current round number (1, 2, ...)
        debate_history: List of (round_num, red_output, blue_output) from previous rounds
    """
    print(f"      [Baseline-Red] Round {round_num}: Arguing vulnerability exists...")
    
    sf = feature.slices.get(candidate.patch_func)
    
    # Build debate history summary if available
    history_section = ""
    if debate_history and len(debate_history) > 0:
        history_lines = []
        for prev_round, prev_red, prev_blue in debate_history:
            history_lines.append(f"""
### Round {prev_round} Summary
**Your Previous Attack**:
- First Anchor: Line {prev_red.first_anchor_line}: `{prev_red.first_anchor_code}`
- Last Anchor: Line {prev_red.last_anchor_line}: `{prev_red.last_anchor_code}`
- Reasoning: {prev_red.attack_reasoning}

**Blue's Refutation**:
- Defense Found: {prev_blue.defense_found}
- Defense: Line {prev_blue.defense_line}: `{prev_blue.defense_code}`
- Refutation: {prev_blue.refutation_reasoning}
- Blue Concedes: {prev_blue.concedes}
""")
        history_section = f"""
## Previous Debate History
{chr(10).join(history_lines)}

**YOU MUST ADDRESS Blue's latest refutation. If Blue's defense is valid and you cannot counter it, set concedes=true. Otherwise, explain why their defense doesn't work or provide a stronger attack argument.**
"""
    
    # Build context
    user_input = f"""
### Vulnerability Logic
- **Root Cause**: {feature.semantics.root_cause}
- **Attack Chain**: {feature.semantics.attack_chain}
- **Vulnerability Type**: {feature.semantics.vuln_type.value if feature.semantics.vuln_type else 'Unknown'}

### Reference Pre-Patch Code (Vulnerable Pattern)
{sf.s_pre if sf else 'Not available'}

### Target Code to Analyze
File: {candidate.target_file}
Function: {candidate.target_func.split(':')[-1]}

{target_code}
{history_section}
**YOUR TASK**: Find the first anchor (where vulnerable state is created, e.g. ALLOC) and last anchor (where vulnerability triggers, e.g. USE) in the target code. Use tools if needed to verify your findings.
{"Address the debate history above and respond to Blue's latest refutation." if debate_history else ""}
"""
    
    llm_with_tools = llm.bind_tools(tools)
    messages = [
        SystemMessage(content=BASELINE_RED_PROMPT),
        HumanMessage(content=user_input)
    ]
    
    # Tool exploration loop (limited)
    max_tool_steps = 5
    for step in range(max_tool_steps):
        try:
            ai_msg = llm_invoke_with_retry(llm_with_tools, messages)
            if ai_msg is None:
                break
        except Exception as e:
            print(f"      [Baseline-Red] Error: {e}")
            break
        
        messages.append(ai_msg)
        
        if not ai_msg.tool_calls:
            break
        
        for tc in ai_msg.tool_calls:
            tool_map = {t.name: t for t in tools}
            t_func = tool_map.get(tc["name"])
            try:
                res = str(t_func.invoke(tc["args"])) if t_func else "Tool not found"
            except Exception as e:
                res = f"Error: {e}"
            messages.append(ToolMessage(content=res, tool_call_id=tc["id"]))
    
    # Request structured output
    try:
        schema_str = json.dumps(BaselineRedOutput.model_json_schema(), indent=2)
    except:
        schema_str = "BaselineRedOutput schema"
    
    messages.append(HumanMessage(content=f"\n\nProvide your attack analysis in JSON format.\n\nSchema:\n{schema_str}\n\nOutput ONLY the JSON."))
    
    # Parse response
    for attempt in range(3):
        try:
            ai_msg = llm_invoke_with_retry(llm, messages)
            if ai_msg is None:
                continue
            
            result = robust_json_parse(ai_msg.content, BaselineRedOutput, "Baseline-Red", use_fallback=(attempt == 2))
            if result:
                print(f"      [Baseline-Red] Claim: vulnerability_exists={result.vulnerability_exists}")
                return result
            
            messages.append(ai_msg)
            messages.append(HumanMessage(content="Parse error. Output ONLY valid JSON."))
        except Exception as e:
            print(f"      [Baseline-Red] Parse error: {e}")
    
    # Fallback
    return BaselineRedOutput(
        vulnerability_exists=False,
        first_anchor_line=None,
        first_anchor_code=None,
        last_anchor_line=None,
        last_anchor_code=None,
        attack_reasoning="Failed to analyze"
    )


def run_baseline_blue(
    red_output: BaselineRedOutput,
    feature: PatchFeatures,
    candidate,  # SearchResultItem
    target_code: str,
    tools: List[Any],
    llm: ChatOpenAI,
    round_num: int = 1,
    debate_history: Optional[List[tuple]] = None
) -> BaselineBlueOutput:
    """
    Baseline Blue Agent: Argue that vulnerability does NOT exist.
    Uses tools to find defenses.
    
    Args:
        round_num: Current round number
        debate_history: List of (round_num, red_output, blue_output) from previous rounds (not including current red)
    """
    print(f"      [Baseline-Blue] Round {round_num}: Arguing vulnerability does not exist...")
    
    sf = feature.slices.get(candidate.patch_func)
    
    # Build debate history summary if available
    history_section = ""
    if debate_history and len(debate_history) > 0:
        history_lines = []
        for prev_round, prev_red, prev_blue in debate_history:
            history_lines.append(f"""
### Round {prev_round} Summary
**Red's Attack**:
- First Anchor: Line {prev_red.first_anchor_line}: `{prev_red.first_anchor_code}`
- Last Anchor: Line {prev_red.last_anchor_line}: `{prev_red.last_anchor_code}`
- Reasoning: {prev_red.attack_reasoning}

**Your Previous Defense**:
- Defense Found: {prev_blue.defense_found}
- Defense: Line {prev_blue.defense_line}: `{prev_blue.defense_code}`
- Refutation: {prev_blue.refutation_reasoning}
""")
        history_section = f"""
## Previous Debate History
{chr(10).join(history_lines)}
"""
    
    # Build context with Red's current claim
    concede_status = ""
    if red_output.concedes:
        concede_status = "\n**NOTE: Red has CONCEDED. You win this debate. Still provide your analysis.**"
    
    red_claim = f"""
**Red's Current Attack (Round {round_num})**:
- Vulnerability Exists: {red_output.vulnerability_exists}
- Concedes: {red_output.concedes}{concede_status}
- First Anchor: Line {red_output.first_anchor_line}: `{red_output.first_anchor_code}`
- Last Anchor: Line {red_output.last_anchor_line}: `{red_output.last_anchor_code}`
- Attack Reasoning: {red_output.attack_reasoning}
"""
    
    user_input = f"""
{history_section}
{red_claim}

### Vulnerability Logic
- **Root Cause**: {feature.semantics.root_cause}
- **Patch Defense**: {feature.semantics.patch_defense}

### Reference Post-Patch Code (Shows the Fix)
{sf.s_post if sf else 'Not available'}

### Target Code to Analyze
File: {candidate.target_file}
Function: {candidate.target_func.split(':')[-1]}

{target_code}

**YOUR TASK**: Find defenses or refute Red's claim. Use tools to check if:
1. The target has the same fix as reference
2. The target has an equivalent defense
3. Red's anchor chain mapping is incorrect

If Red's attack is clearly valid and you cannot find any defense, set concedes=true.
"""
    
    llm_with_tools = llm.bind_tools(tools)
    messages = [
        SystemMessage(content=BASELINE_BLUE_PROMPT),
        HumanMessage(content=user_input)
    ]
    
    # Tool exploration loop (limited)
    max_tool_steps = 5
    for step in range(max_tool_steps):
        try:
            ai_msg = llm_invoke_with_retry(llm_with_tools, messages)
            if ai_msg is None:
                break
        except Exception as e:
            print(f"      [Baseline-Blue] Error: {e}")
            break
        
        messages.append(ai_msg)
        
        if not ai_msg.tool_calls:
            break
        
        for tc in ai_msg.tool_calls:
            tool_map = {t.name: t for t in tools}
            t_func = tool_map.get(tc["name"])
            try:
                res = str(t_func.invoke(tc["args"])) if t_func else "Tool not found"
            except Exception as e:
                res = f"Error: {e}"
            messages.append(ToolMessage(content=res, tool_call_id=tc["id"]))
    
    # Request structured output
    try:
        schema_str = json.dumps(BaselineBlueOutput.model_json_schema(), indent=2)
    except:
        schema_str = "BaselineBlueOutput schema"
    
    messages.append(HumanMessage(content=f"\n\nProvide your defense analysis in JSON format.\n\nSchema:\n{schema_str}\n\nOutput ONLY the JSON."))
    
    # Parse response
    for attempt in range(3):
        try:
            ai_msg = llm_invoke_with_retry(llm, messages)
            if ai_msg is None:
                continue
            
            result = robust_json_parse(ai_msg.content, BaselineBlueOutput, "Baseline-Blue", use_fallback=(attempt == 2))
            if result:
                print(f"      [Baseline-Blue] Claim: vulnerability_exists={result.vulnerability_exists}, defense_found={result.defense_found}")
                return result
            
            messages.append(ai_msg)
            messages.append(HumanMessage(content="Parse error. Output ONLY valid JSON."))
        except Exception as e:
            print(f"      [Baseline-Blue] Parse error: {e}")
    
    # Fallback
    return BaselineBlueOutput(
        vulnerability_exists=True,  # Conservative: assume Red is right if Blue fails
        defense_found=False,
        defense_line=None,
        defense_code=None,
        refutation_reasoning="Failed to analyze"
    )


def run_baseline_judge(
    debate_history: List[tuple],  # List of (round_num, red_output, blue_output)
    feature: PatchFeatures,
    candidate,
    target_code: str,
    llm: ChatOpenAI
) -> BaselineJudgeOutput:
    """
    Baseline Judge: Make final verdict based on Red and Blue arguments from all rounds.
    No tools - just synthesis.
    
    Args:
        debate_history: List of tuples (round_num, red_output, blue_output) from all rounds
    """
    print("      [Baseline-Judge] Making final verdict based on all rounds...")
    
    # Build debate history summary
    history_lines = []
    final_red = None
    final_blue = None
    
    for round_num, red_output, blue_output in debate_history:
        history_lines.append(f"""
## Round {round_num}

### Red Agent's Attack (Round {round_num})
- Vulnerability Exists: {red_output.vulnerability_exists}
- Concedes: {red_output.concedes}
- First Anchor: Line {red_output.first_anchor_line}: `{red_output.first_anchor_code}`
- Last Anchor: Line {red_output.last_anchor_line}: `{red_output.last_anchor_code}`
- Attack Reasoning: {red_output.attack_reasoning}

### Blue Agent's Defense (Round {round_num})
- Vulnerability Exists: {blue_output.vulnerability_exists}
- Concedes: {blue_output.concedes}
- Defense Found: {blue_output.defense_found}
- Defense: Line {blue_output.defense_line}: `{blue_output.defense_code}`
- Refutation Reasoning: {blue_output.refutation_reasoning}
""")
        final_red = red_output
        final_blue = blue_output
    
    debate_summary = "\n".join(history_lines)
    
    user_input = f"""
# Complete Debate History

{debate_summary}

### Vulnerability Logic
- **Root Cause**: {feature.semantics.root_cause}
- **Attack Chain**: {feature.semantics.attack_chain}
- **Patch Defense**: {feature.semantics.patch_defense}

### Target Code
File: {candidate.target_file}
Function: {candidate.target_func.split(':')[-1]}

{target_code}

**YOUR TASK**: Review the complete debate history and make the final verdict.
- If Red conceded → SAFE
- If Blue conceded → VULNERABLE
- If Red's mapping is correct AND Blue found no valid defense → VULNERABLE
- If Red's mapping is wrong OR Blue found a valid defense → SAFE
"""
    
    messages = [
        SystemMessage(content=BASELINE_JUDGE_PROMPT),
        HumanMessage(content=user_input)
    ]
    
    # Request structured output
    try:
        schema_str = json.dumps(BaselineJudgeOutput.model_json_schema(), indent=2)
    except:
        schema_str = "BaselineJudgeOutput schema"
    
    messages.append(HumanMessage(content=f"\n\nProvide your final verdict in JSON format.\n\nSchema:\n{schema_str}\n\nOutput ONLY the JSON."))
    
    # Parse response
    for attempt in range(3):
        try:
            ai_msg = llm_invoke_with_retry(llm, messages)
            if ai_msg is None:
                continue
            
            result = robust_json_parse(ai_msg.content, BaselineJudgeOutput, "Baseline-Judge", use_fallback=(attempt == 2))
            if result:
                print(f"      [Baseline-Judge] Final verdict: is_vulnerable={result.is_vulnerable}")
                return result
            
            messages.append(ai_msg)
            messages.append(HumanMessage(content="Parse error. Output ONLY valid JSON."))
        except Exception as e:
            print(f"      [Baseline-Judge] Parse error: {e}")
    
    # Fallback: based on concessions or final state
    if final_red and final_red.concedes:
        return BaselineJudgeOutput(
            is_vulnerable=False,
            reasoning="Fallback: Red conceded, vulnerability not proven"
        )
    if final_blue and final_blue.concedes:
        return BaselineJudgeOutput(
            is_vulnerable=True,
            reasoning="Fallback: Blue conceded, no defense found"
        )
    if final_red and final_red.vulnerability_exists and final_blue and not final_blue.defense_found:
        return BaselineJudgeOutput(
            is_vulnerable=True,
            reasoning="Fallback: Red claimed vulnerability, Blue found no defense"
        )
    return BaselineJudgeOutput(
        is_vulnerable=False,
        reasoning="Fallback: Either Red's claim failed or Blue found defense"
    )


def run_baseline_debate(
    feature: PatchFeatures,
    candidate,  # SearchResultItem
    target_code: str,
    tools: List[Any],
    llm: ChatOpenAI,
    max_rounds: int = 2
) -> BaselineJudgeOutput:
    """
    Execute the simplified baseline three-role debate with multiple rounds.
    
    Flow:
        Round 1: Red → Blue
        Round 2: Red (responds to Blue) → Blue (responds to Red)
        ...
        Final: Judge reviews all rounds
    
    Args:
        max_rounds: Maximum number of debate rounds (default: 2)
    
    Returns:
        BaselineJudgeOutput with final verdict
    """
    print(f"    [Baseline-Debate] Starting {max_rounds}-round three-role debate...")
    
    # Track debate history for Judge (and for agents to see previous rounds)
    debate_history: List[tuple] = []  # (round_num, red_output, blue_output)
    
    for round_num in range(1, max_rounds + 1):
        print(f"    [Baseline-Debate] === Round {round_num} ===")
        
        # Red: Attack (receives full debate history from previous rounds)
        red_output = run_baseline_red(
            feature, candidate, target_code, tools, llm,
            round_num=round_num,
            debate_history=debate_history if debate_history else None
        )
        
        # Check if Red concedes
        if red_output.concedes:
            print(f"    [Baseline-Debate] Red concedes in round {round_num}!")
            # Create minimal Blue output and end debate
            blue_output = BaselineBlueOutput(
                vulnerability_exists=False,
                concedes=False,
                defense_found=True,
                defense_line=None,
                defense_code=None,
                refutation_reasoning="Red conceded the debate."
            )
            debate_history.append((round_num, red_output, blue_output))
            break
        
        # Blue: Defense (receives full debate history from previous rounds)
        blue_output = run_baseline_blue(
            red_output, feature, candidate, target_code, tools, llm,
            round_num=round_num,
            debate_history=debate_history if debate_history else None
        )
        
        # Record this round
        debate_history.append((round_num, red_output, blue_output))
        
        # Check if Blue concedes
        if blue_output.concedes:
            print(f"    [Baseline-Debate] Blue concedes in round {round_num}!")
            break
    
    # Judge: Final verdict based on all rounds
    judge_output = run_baseline_judge(
        debate_history, feature, candidate, target_code, llm
    )
    
    print(f"    [Baseline-Debate] Final: is_vulnerable={judge_output.is_vulnerable}")
    
    return judge_output


class ContextBuilder:
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.indexer = GlobalSymbolIndexer(repo_path)

    def fetch_peer_functions(self, candidate: SearchResultItem, feature: PatchFeatures) -> Dict[str, str]:
        """Expand relevant nodes: fetch Peer functions from same file in database"""
        needed_funcs = set(feature.slices.keys())
        needed_funcs.discard(candidate.patch_func)
        peers = {}
        if not needed_funcs: return peers

        conn = self.indexer._get_conn()
        cursor = conn.cursor()
        for func_name in needed_funcs:
            cursor.execute("SELECT code FROM symbols WHERE name = ? AND path = ? AND kind = 'function' LIMIT 1", 
                           (func_name, candidate.target_file))
            row = cursor.fetchone()
            if row: peers[func_name] = row[0]
        return peers

    def format_structural_hints(self, candidate: SearchResultItem, feature: PatchFeatures) -> str:
        """
        Format structural hints from Phase 3 matching results.
        
        [Enhanced] Now includes Anchor Mapping Hints from Phase 2 analysis.
        This helps Red Agent directly use the pre-computed typed anchors
        instead of re-inferring them from scratch.
        """
        ev = candidate.evidence
        hints = []
        
        # === Section 1: Anchor Mapping Hints (NEW) ===
        # Use Phase 2's pre_anchors (typed anchors) and Phase 3's alignment to provide
        # direct anchor mapping hints to Red Agent
        sf = feature.slices.get(candidate.patch_func)
        if sf:
            anchor_hints = self._format_anchor_mapping_hints(sf, ev)
            if anchor_hints:
                hints.append(anchor_hints)
        
        # === Section 2: Structural Match (existing) ===
        # [Fix] Use aligned_vuln_traces instead of removed vul_matches to find matched lines
        matched_traces = [m for m in ev.aligned_vuln_traces if m.line_no is not None]
        
        if matched_traces:
            # Take only the top 10 most representative lines
            lines = [f"Line {m.line_no}: {m.target_line.strip()}" for m in matched_traces[:10]]
            hints.append(f"2. **Structural Match**: The Matcher found these vulnerable lines:\n   " + "\n   ".join(lines))
        
        # === Section 3: Missing Defense (existing) ===
        # [Restored] Identify missing fix lines using aligned_fix_traces
        # Look for trace items where tag != 'COMMON' (Feature) and target_line is None (Missing)
        missing_fix_lines = [
            t.slice_line.strip()
            for t in ev.aligned_fix_traces
            if t.tag != 'COMMON' and t.target_line is None
        ]
        
        if missing_fix_lines:
            # Limit to top 5 missing lines to avoid clutter
            hints.append(f"3. **Missing Defense**: The Matcher indicates these FIX lines are MISSING in Target:\n   " + "\n   ".join(missing_fix_lines[:5]))
        
        return "\n".join(hints)
    
    def _format_anchor_mapping_hints(self, sf, ev) -> str:
        """
        Generate Anchor Mapping Hints by matching Phase 2's typed anchors with Phase 3's alignment.
        
        Locatability-aware hints:
        - CONCRETE: Direct alignment from Phase 3 traces
        - ASSUMED:  Alignment + assumption verification guidance
        - CONCEPTUAL: Discovery guidance from root cause
        
        Returns a formatted string with tiered anchor mapping hints for Red Agent.
        """
        import re
        
        def extract_line_no(line: str) -> int:
            """Extract line number from lines like '[ 227] code...'"""
            match = re.match(r'^\[\s*(\d+)\]', line.strip())
            return int(match.group(1)) if match else -1
        
        def find_target_mapping_for_anchor(anchor, aligned_traces: list) -> tuple:
            """
            Find the target line that maps to a given Anchor object.
            Returns: (target_line_no, target_code, similarity) or (None, None, 0.0)
            """
            # Strategy 1: Use anchor.line_number
            if anchor.line_number:
                for trace in aligned_traces:
                    slice_ln = extract_line_no(trace.slice_line)
                    if slice_ln == anchor.line_number and trace.target_line:
                        return (trace.line_no, trace.target_line, trace.similarity)
            
            # Strategy 2: Use code_snippet content matching
            if anchor.code_snippet:
                snippet = anchor.code_snippet.strip()
                for trace in aligned_traces:
                    if trace.target_line and snippet and snippet in trace.slice_line:
                        return (trace.line_no, trace.target_line, trace.similarity)
            
            return (None, None, 0.0)
        
        # Group anchors by locatability tier
        tier1_mapped = []   # CONCRETE, mapped
        tier1_unmapped = [] # CONCRETE, unmapped
        tier2_mapped = []   # ASSUMED, mapped
        tier2_unmapped = [] # ASSUMED, unmapped
        tier3 = []          # CONCEPTUAL (rarely mapped)
        
        for anchor in (sf.pre_anchors or []):
            target_ln, target_code, sim = find_target_mapping_for_anchor(anchor, ev.aligned_vuln_traces)
            label = format_anchor_for_prompt(anchor)
            loc = anchor.locatability.value if hasattr(anchor.locatability, 'value') else str(anchor.locatability)
            
            entry = {
                'label': label,
                'anchor_type': anchor.type.value if hasattr(anchor.type, 'value') else str(anchor.type),
                'target_line': target_ln,
                'target_code': target_code.strip() if target_code else '',
                'similarity': sim,
                'is_optional': anchor.is_optional,
                'assumption_type': (anchor.assumption_type.value if anchor.assumption_type and hasattr(anchor.assumption_type, 'value') else str(anchor.assumption_type)) if anchor.assumption_type else None,
                'assumption_rationale': anchor.assumption_rationale,
            }
            
            if loc == 'concrete':
                if target_ln and sim > 0.3:
                    tier1_mapped.append(entry)
                else:
                    tier1_unmapped.append(entry)
            elif loc == 'assumed':
                if target_ln and sim > 0.3:
                    tier2_mapped.append(entry)
                else:
                    tier2_unmapped.append(entry)
            else:  # conceptual
                tier3.append(entry)
        
        # Format output
        lines = []
        has_content = tier1_mapped or tier1_unmapped or tier2_mapped or tier2_unmapped or tier3
        
        if has_content:
            lines.append("1. **Tiered Anchor Mapping Hints** (from Phase 2/3 Analysis):")
            lines.append("   Anchors are grouped by locatability. Use the tiered strategy for mapping.")
            
            # Tier 1: CONCRETE
            if tier1_mapped or tier1_unmapped:
                lines.append("   ")
                lines.append("   **Tier 1 — CONCRETE** (direct code locations):")
                for m in tier1_mapped:
                    opt = " (optional)" if m['is_optional'] else ""
                    lines.append(f"   - ✓ {m['label']}{opt}")
                    lines.append(f"     → Target Line {m['target_line']}: `{m['target_code']}` (sim: {m['similarity']:.2f})")
                for m in tier1_unmapped:
                    opt = " (optional)" if m['is_optional'] else ""
                    lines.append(f"   - ✗ {m['label']}{opt} — UNMAPPED, needs manual search")
            
            # Tier 2: ASSUMED
            if tier2_mapped or tier2_unmapped:
                lines.append("   ")
                lines.append("   **Tier 2 — ASSUMED** (location known, assumption needs verification):")
                for m in tier2_mapped + tier2_unmapped:
                    opt = " (optional)" if m['is_optional'] else ""
                    at = f" [Assumption: {m['assumption_type']}]" if m['assumption_type'] else ""
                    status = "✓" if m['target_line'] else "✗"
                    lines.append(f"   - {status} {m['label']}{opt}{at}")
                    if m['target_line']:
                        lines.append(f"     → Target Line {m['target_line']}: `{m['target_code']}` (sim: {m['similarity']:.2f})")
                    lines.append(f"     ⚠ Verify assumption in target context")
            
            # Tier 3: CONCEPTUAL
            if tier3:
                lines.append("   ")
                lines.append("   **Tier 3 — CONCEPTUAL** (must be inferred from context):")
                for m in tier3:
                    opt = " (optional)" if m['is_optional'] else ""
                    at = f" [Assumption: {m['assumption_type']}]" if m['assumption_type'] else ""
                    lines.append(f"   - ? {m['label']}{opt}{at}")
                    if m['assumption_rationale']:
                        lines.append(f"     Hint: {m['assumption_rationale']}")
                    lines.append(f"     → Must discover from root cause + target code semantics")
        
        return "\n".join(lines) if lines else ""


def extract_involved_functions(judge_res=None, anchor_evidence=None, trace=None, defense_mechanism=None) -> List[str]:
    """
    Extract list of involved functions from evidence chain.
    
    Args:
        judge_res: JudgeOutput object (if provided, all info will be extracted from it)
        anchor_evidence: List of StepAnalysis for typed anchors (optional)
        trace: Trace List (optional)
        defense_mechanism: Defense StepAnalysis (optional)
    
    Returns:
        Sorted list of function names
    """
    involved_funcs = set()
    
    # If judge_res is provided, extract all info from it
    if judge_res:
        for step in (judge_res.anchor_evidence or []):
            if step.func_name:
                involved_funcs.add(step.func_name)
        for step in (judge_res.trace or []):
            if step.func_name:
                involved_funcs.add(step.func_name)
        if judge_res.defense_mechanism and judge_res.defense_mechanism.func_name:
            involved_funcs.add(judge_res.defense_mechanism.func_name)
    else:
        # Extract from individually provided parameters
        for step in (anchor_evidence or []):
            if step.func_name:
                involved_funcs.add(step.func_name)
        for step in (trace or []):
            if step.func_name:
                involved_funcs.add(step.func_name)
        if defense_mechanism and defense_mechanism.func_name:
            involved_funcs.add(defense_mechanism.func_name)
    
    return sorted(list(involved_funcs))


# ==============================================================================
# 4. Verification Node Main Logic
# ==============================================================================

def validation_node(state: VerificationState) -> Dict[str, Any]:
    """
    Verify candidates in the same patch group one by one, dynamically assemble peer context for each candidate:
    1. Prioritize using other candidates in the same group (same file/class) as peers.
    2. If peer is missing, fallback to static lookup in same file for repo mode, or use excerpts from 1day_vul_dict.json for benchmark mode.
    """
    candidates: List[SearchResultItem] = state["candidates"]
    feature: PatchFeatures = state["feature_context"]
    mode = state["mode"]
    vul_id = state["vul_id"]
    findings = []
    # Group by file, facilitating peer lookup
    file_map : Dict[str, List[SearchResultItem]] = {}
    for cand in candidates:
        file_map.setdefault(cand.target_file, []).append(cand)
    
    # [New] Record processed candidate IDs (target_func)
    processed_funcs = set()

    # [New] Filtering logic: Verdict is VULNERABLE and Confidence >= 0.4
    filtered_candidates : List[SearchResultItem] = []
    for c in candidates:
        if c.verdict in ("VULNERABLE") and c.confidence >= 0.4:
            filtered_candidates.append(c)
            
    if not filtered_candidates:
        print("    [Skip] No valid candidates after filtering (Verdict/Confidence).")
        return {"final_findings": []}

    # [Dynamic Penalty] Sort by rank (asc) to ensure efficient pruning
    filtered_candidates.sort(key=lambda x: x.rank if x.rank != -1 else 999)
    
    # [Limit] Only process top N candidates to save cost
    MAX_CANDIDATES = 10
    if len(filtered_candidates) > MAX_CANDIDATES:
        print(f"    [Limit] Truncating candidates from {len(filtered_candidates)} to {MAX_CANDIDATES}")
        filtered_candidates = filtered_candidates[:MAX_CANDIDATES]
    
    # [Adaptive Threshold]
    # Strategy: Start with base confidence (0.4). Process Top X candidates unconditionally.
    # For every False Positive (Safe verdict), raise the required confidence threshold.
    current_threshold = 0.4
    if mode == 'benchmark':
        THRESHOLD_PENALTY_STEP = 0.0
    else:
        THRESHOLD_PENALTY_STEP = 0.1

    for candidate in filtered_candidates:
        # [Robustness] Add exception isolation for each candidate
        try:
            # [Adaptive Threshold] Check
            # Only apply new threshold if rank is outside grace period
            if candidate.confidence < current_threshold:
                # print(f"    [Skip-Adaptive] Rank {candidate.rank} (Conf {candidate.confidence:.2f}) < Threshold {current_threshold:.2f}")
                continue

            # [Modified] Use file + func as unique key to prevent same-named functions in different files from being skipped by mistake
            # [DEBUG]
            # if candidate.target_func != 'sbp_make_tpg':
            #     continue
            patch_key = f'{candidate.patch_file}::{candidate.patch_func}'
            unique_key = f"{candidate.target_file}::{candidate.target_func}"
            print(f"    [Debug] patch_key: {patch_key}, unique_key: {unique_key}")
            print(f"    [Debug] processed_funcs: {processed_funcs}")
            
            if unique_key in processed_funcs:
                print(f"    [Skip] {unique_key} already processed.")
                continue
            
            repo_path: str = candidate.repo_path
            ctx_builder = ContextBuilder(repo_path)
            # Dynamically assemble peer_funcs
            peer_funcs = {}
            peer_candidates : List[SearchResultItem] = []
            # [New] Record filled patch_func slot to avoid duplicate retrieval in Step 2
            filled_slots = set()
            
            # 1. Prioritize other candidates within the same group and file
            # [Optimize] Limit peer quantity to avoid excessive context length (Top 5)
            MAX_PEERS = 200
            peers_count = 0
            
            for other in file_map.get(candidate.target_file, []):
                if other is candidate:
                    continue
                
                if peers_count >= MAX_PEERS:
                    break

                # [Benchmark Mode] Ensure peers are from the same version
                # target_func format: tag:ver:func_name
                if mode == 'benchmark':
                    c_parts = candidate.target_func.split(":")
                    o_parts = other.target_func.split(":")
                    # Compare tag and version (first 2 parts)
                    if len(c_parts) >= 2 and len(o_parts) >= 2:
                        if c_parts[:2] != o_parts[:2]:
                            continue

                # [User Restriction] Parallel candidates (same patch_func) should not be peers
                c_pfunc = getattr(candidate, "patch_func", None)
                o_pfunc = getattr(other, "patch_func", None)
                if c_pfunc and o_pfunc and c_pfunc == o_pfunc:
                    continue

                # [Modified] peer name prioritizes target_func to correctly display the actual function name used in the report
                peer_name = getattr(other, "target_func", "peer")
                if peer_name not in peer_funcs:
                    peer_funcs[peer_name] = other.code_content
                    peer_candidates.append(other)
                    peers_count += 1
                
                # Record occupied slot
                if hasattr(other, "patch_func"):
                    filled_slots.add(other.patch_func)
                
            # 2. If peer is still missing, use static lookup in repo mode
            # Calculate which slots are still needed
            needed_slots = set(feature.slices.keys())
            if hasattr(candidate, "patch_func"):
                needed_slots.discard(candidate.patch_func)
                
            if mode == 'repo' and not needed_slots.issubset(filled_slots):
                static_peers = ctx_builder.fetch_peer_functions(candidate, feature)
                for k, v in static_peers.items():
                    # k is the patch_func name
                    if k not in filled_slots:
                        peer_funcs[k] = v
                        filled_slots.add(k)

            # 3. In benchmark mode, complete 1day_vul_dict.json excerpts
            if mode == 'benchmark' and len(peer_funcs) < len(feature.slices) - 1:
                benchmark_indexer = BenchmarkSymbolIndexer()
                target_func = candidate.target_func
                parts = target_func.split(":")
                if len(parts) >= 3:
                    tag, version, func_name = parts[0], parts[1], parts[2]
                    for patch in feature.patches:
                        if patch.function_name == func_name:
                            continue
                        candidates_bm = benchmark_indexer.search(vul_id, patch.file_path, patch.function_name, version)
                        for _, _, code_content in candidates_bm:
                            if patch.function_name not in peer_funcs:
                                peer_funcs[patch.function_name] = code_content
            
            # Mark current candidate and all peer candidates as processed
            processed_funcs.add(candidate.target_func)
            for p in peer_candidates:
                processed_funcs.add(p.target_func)

            # ...existing code for slicing, prompt, agent, etc...
            raw_code = candidate.code_content
            target_context = raw_code
        
            if len(raw_code.split('\n')) > 200:
                try:
                    # [Fix] Use aligned_vuln_traces instead of removed vul_matches
                    # Filter traces that have a valid line_no (non-None, meaning it matched a specific line)
                    matched_lines = [m.line_no for m in candidate.evidence.aligned_vuln_traces if m.line_no is not None]
                    s_pre_code = feature.slices[candidate.patch_func].s_pre
                    hint_vars = list(set(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', s_pre_code)))
                    slicer = TargetSlicerAdapter(raw_code)
                    sliced = slicer.slice_context(matched_lines, hint_vars)
                    if len(sliced) < len(raw_code) * 0.9:
                        target_context = f"// [Note] Code Sliced for Focus. Original length: {len(raw_code)} chars.\n{sliced}"
                        print("    [Slicing] Applied focused slice.")
                except Exception as e:
                    print(f"    [Slicing] Failed ({e}), using full code.")
            
            # [Refactor] Do NOT dump peer code content. Just keep the names for the map.
            # We collected 'peer_funcs' which currently stores {name: code}.
            # Let's extract just the names/signatures for the prompt hint.
            
            peer_names_list = list(peer_funcs.keys())
            peer_map_hint = []
            for name in peer_names_list:
                clean_name = name.split(':')[-1]
                peer_map_hint.append(f"- {clean_name} (Use `find_definition` to read)")
            
            peer_hint_block = ""
            if peer_map_hint:
                 peer_hint_block = "\nAvailable Context Peers (NOT Loaded):\n" + "\n".join(peer_map_hint)

            structural_hints = ctx_builder.format_structural_hints(candidate, feature)
            llm = ChatOpenAI(
                base_url=os.getenv("API_BASE"),
                api_key=os.getenv("API_KEY"),
                model=os.getenv("MODEL_NAME", "gpt-4o"),
                temperature=0
            )
            
            # [Updated] Initialize CodeNavigator and Create Tools
            target_version = None
            if mode == 'benchmark':
                parts = candidate.target_func.split(':')
                if len(parts) >= 3:
                    # parts[1] is version (e.g. v5.17)
                    target_version = parts[1]
            
            # CodeNavigator automatically handles the indexer selection
            navigator = CodeNavigator(repo_path, target_version=target_version)
            
            # [Fix] Create shared tool cache for this candidate's entire debate
            # This prevents redundant read_file calls across Red/Blue/multiple rounds
            shared_tool_cache: Dict[str, Any] = {}
            tools = create_tools(navigator, candidate.target_file, shared_tool_cache)
            
            # [Fix] Get function start line to add correct line numbers
            # This ensures Agent-cited line numbers match the actual file
            func_start_line = 1  # Default to 1 if we can't determine
            try:
                funcs = navigator.list_file_functions(candidate.target_file)
                target_func_name = candidate.target_func.split(':')[-1]  # Extract actual function name
                for f in funcs:
                    if f.get('name') == target_func_name:
                        func_start_line = f.get('start_line', 1)
                        break
            except Exception as e:
                print(f"    [Warning] Could not get function start line: {e}")
            
            # [Fix] Add line numbers to target code context
            # This ensures Red/Blue/Judge agents can cite correct file line numbers
            def add_line_numbers(code: str, start_line: int) -> str:
                """Add line numbers to code, starting from start_line."""
                lines = code.splitlines()
                numbered_lines = []
                for i, line in enumerate(lines):
                    line_no = start_line + i
                    numbered_lines.append(f"[{line_no:4d}] {line}")
                return '\n'.join(numbered_lines)
            
            target_context_with_lines = add_line_numbers(target_context, func_start_line)
            
            full_code_context = f"=== Primary Target:\n - file path: {candidate.target_file}\n - func name: {candidate.target_func.split(':')[-1]}\n - start line: {func_start_line}\n===\n{target_context_with_lines}\n"
                    
            # [Context Prep] Combine all diffs and get full reference code
            all_diffs = []
            for p in feature.patches:
                header = f"--- {p.file_path} : {p.function_name} ---"
                diff_content = p.clean_diff if p.clean_diff else p.raw_diff
                all_diffs.append(f"{header}\n{diff_content}")
            combined_diffs = "\n\n".join(all_diffs)
            
            specific_patch = next((p for p in feature.patches if p.function_name == candidate.patch_func), None)
            ref_old_code = specific_patch.old_code if specific_patch else "Not Available"
            ref_new_code = specific_patch.new_code if specific_patch else "Not Available"

            # Prepare Vulnerability Info (using SemanticFeature fields directly)
            cwe_line = ""
            if feature.semantics.cwe_id:
                cwe_line = f"- Type: {feature.semantics.cwe_id} - {feature.semantics.cwe_name or 'Unknown'}"

            vuln_info = f"""
            [Root Cause]: {feature.semantics.root_cause}
            [Attack Chain]: {feature.semantics.attack_chain}
            {cwe_line}
            [Reference Template (Slice)]:
            {feature.slices[candidate.patch_func].s_pre}
            [Reference Function (Full Pre-Patch)]:
            {ref_old_code}
            [All Patch Diffs]:
            {combined_diffs}
            """
            fix_info = f"""
            [Root Cause]: {feature.semantics.root_cause}
            [Patch Defense]: {feature.semantics.patch_defense}
            [Reference Fix (Slice)]:
            {feature.slices[candidate.patch_func].s_post}
            [Reference Function (Full Post-Patch)]:
            {ref_new_code}
            [All Patch Diffs]:
            {combined_diffs}
            """
            # ============== §3.3 Three-Step Verification ==============
            # Step 1 (§3.3.1): Anchor Mapping — C_cons validation (no tools)
            # Step 2 (§3.3.2): Path Verification — pure static PDG analysis (no LLM)
            # Step 3 (§3.3.3): Multi-Agent Semantic Analysis — multi-round debate (with tools)
            
            # === Step 1 (§3.3.1): Anchor Mapping (No Tools) ===
            # Returns all three agent outputs for evidence chain construction
            round1_red, round1_blue, round1_judge, should_continue = run_round1_debate(
                feature, candidate, target_context_with_lines, llm
            )
            
            # Step 1 Early Exit: C_cons not satisfied
            if not should_continue:
                print(f"    [§3.3.1-Exit] C_cons failed: {round1_judge.verdict}")
                finding = VulnerabilityFinding(
                    vul_id=state["vul_id"],
                    cwe_id=feature.semantics.cwe_id or "Unknown",
                    cwe_name=feature.semantics.cwe_name or "Unknown",
                    group_id=feature.group_id,
                    repo_path=candidate.repo_path,
                    patch_file=candidate.patch_file,
                    patch_func=candidate.patch_func,
                    target_file=candidate.target_file,
                    target_func=candidate.target_func,
                    analysis_report=f"[§3.3.1 Exit] C_cons not satisfied. Verdict: {round1_judge.verdict}",
                    is_vulnerable=False,
                    verdict_category=round1_judge.verdict,
                    involved_functions=[],
                    peer_functions=peer_names_list,
                    anchor_evidence=[],
                    trace=[],
                    defense_mechanism=None,
                    constraint_status=f"C_cons={round1_judge.c_cons_satisfied}"
                )
                findings.append(finding)
                
                # [Adaptive Threshold] Update - Early exit also counts as SAFE verdict
                current_threshold += THRESHOLD_PENALTY_STEP
                continue  # Skip to next candidate
            
            # === Step 2 (§3.3.2): Path Verification (Pure Static, No LLM) ===
            path_result = run_path_verification(
                round1_judge, target_context, feature, candidate,
                func_start_line=func_start_line
            )
            print(f"    [§3.3.2] Path verification: reachable={path_result.reachable}, type={path_result.path_type}"
                  + (f", skipped={path_result.skip_reason}" if path_result.skipped else ""))
            
            # Path verification early exit: unreachable
            # Solution E: Before giving up, try LLM override for failed DATA pairs
            if not path_result.reachable and not path_result.skipped:
                # Check if any failed pair involves DATA dependency — candidate for LLM override
                has_failed_data_pair = any(
                    not pr.reachable and not pr.skipped and pr.dependency_type.upper() == "DATA"
                    for pr in path_result.pair_results
                )
                
                llm_override = False
                if has_failed_data_pair:
                    llm_override = run_path_llm_override(
                        path_result, feature, candidate, target_context,
                        llm, func_start_line=func_start_line
                    )
                
                if llm_override:
                    # LLM confirmed relationships hold — override static failure, continue to §3.3.3
                    print(f"    [§3.3.2-Override] LLM confirmed anchor relationships hold despite static failure")
                    # Update path_result to reflect the override
                    path_result = PathVerificationResult(
                        reachable=True,
                        path_type="llm_override",
                        details=f"[LLM Override] Static analysis failed but LLM confirmed relationships hold. Original: {path_result.details}",
                        anchor_lines_checked=path_result.anchor_lines_checked,
                        pair_results=path_result.pair_results
                    )
                else:
                    # Static failure confirmed — exit with SAFE-Unreachable
                    print(f"    [§3.3.2-Exit] Path unreachable: {path_result.details}")
                    finding = VulnerabilityFinding(
                        vul_id=state["vul_id"],
                        cwe_id=feature.semantics.cwe_id or "Unknown",
                        cwe_name=feature.semantics.cwe_name or "Unknown",
                        group_id=feature.group_id,
                        repo_path=candidate.repo_path,
                        patch_file=candidate.patch_file,
                        patch_func=candidate.patch_func,
                        target_file=candidate.target_file,
                        target_func=candidate.target_func,
                        analysis_report=f"[§3.3.2 Exit] Static path verification failed (LLM override {'not attempted (no DATA pairs)' if not has_failed_data_pair else 'also rejected'}): {path_result.details}",
                        is_vulnerable=False,
                        verdict_category="SAFE-Unreachable",
                        involved_functions=[],
                        peer_functions=peer_names_list,
                        anchor_evidence=[],
                        trace=[],
                        defense_mechanism=None,
                        constraint_status=f"C_cons=True, C_reach=False (static{'+llm' if has_failed_data_pair else ''})",
                        path_verification=path_result
                    )
                    findings.append(finding)
                    current_threshold += THRESHOLD_PENALTY_STEP
                    continue  # Skip to next candidate
            
            # === Step 3 (§3.3.3): Multi-Agent Semantic Analysis (With Tools) ===
            # Multi-round Red/Blue/Judge debate (paper: 3 rounds)
            NUM_DEBATE_ROUNDS = 2
            round2_red = None
            round2_blue = None
            round2_judge = None
            round2_verdict = None
            
            for debate_round in range(1, NUM_DEBATE_ROUNDS + 1):
                print(f"    [§3.3.3] Semantic analysis debate round {debate_round}/{NUM_DEBATE_ROUNDS}")
                round2_red, round2_blue, round2_judge, round2_verdict = run_round2_debate(
                    round1_judge, feature, candidate, target_context_with_lines, tools, llm
                )
                
                # Early termination: if verdict is conclusive, stop debating
                if round2_verdict in ("VULNERABLE", "SAFE-Blocked", "SAFE-Mismatch",
                                      "SAFE-Unreachable", "SAFE-TypeMismatch", "SAFE-OutOfScope"):
                    print(f"    [§3.3.3] Debate concluded at round {debate_round} with verdict: {round2_verdict}")
                    break
            
            # === Final Judge Integration ===
            # Synthesizes ALL step outputs into a proper JudgeOutput
            # with complete evidence chain (anchor_evidence, trace, defense_mechanism)
            judge_res = run_round3_final_judge(
                round1_red, round1_blue, round1_judge,
                round2_red, round2_blue, round2_judge,
                feature, candidate, target_context_with_lines, llm
            )
        
            # Map JudgeOutput to VulnerabilityFinding (per §3.3)
            finding = VulnerabilityFinding(
                vul_id=state["vul_id"],
                cwe_id=feature.semantics.cwe_id or "Unknown",
                cwe_name=feature.semantics.cwe_name or "Unknown",
                group_id=feature.group_id,
                repo_path=candidate.repo_path,
                patch_file=candidate.patch_file,
                patch_func=candidate.patch_func,
                target_file=candidate.target_file,
                target_func=candidate.target_func,
                analysis_report=judge_res.analysis_report,
                is_vulnerable=judge_res.is_vulnerable,
                verdict_category=judge_res.verdict_category,
                involved_functions=extract_involved_functions(judge_res),
                peer_functions=peer_names_list,
                # Typed anchor chain evidence
                anchor_evidence=judge_res.anchor_evidence,
                trace=judge_res.trace,
                defense_mechanism=judge_res.defense_mechanism,
                constraint_status=f"C_cons={judge_res.c_cons_satisfied}, C_reach={judge_res.c_reach_satisfied}, C_def={judge_res.c_def_satisfied}",
                path_verification=path_result
            )
            findings.append(finding)
        
            # Log constraint outcomes
            print(f"    [Judge] Target: {candidate.target_file}::{candidate.target_func}")
            print(f"    [Judge] Verdict: {judge_res.verdict_category}")
            print(f"    [Judge] Constraints: C_cons={judge_res.c_cons_satisfied}, C_reach={judge_res.c_reach_satisfied}, C_def={judge_res.c_def_satisfied}")

            # [Adaptive Threshold] Update
            if not judge_res.is_vulnerable:
                current_threshold += THRESHOLD_PENALTY_STEP
        
        except Exception as e:
            # [Robustness] Catch any unexpected exceptions, record errors but continue processing other candidates
            print(f"    [Error] Verification failed for {candidate.target_func}: {e}")
            import traceback
            traceback.print_exc()
            
            # Create a failure record
            finding = VulnerabilityFinding(
                vul_id=state["vul_id"],
                cwe_id=feature.semantics.cwe_id or "Unknown",
                cwe_name=feature.semantics.cwe_name or "Unknown",
                group_id=feature.group_id,
                repo_path=candidate.repo_path,
                patch_file=candidate.patch_file,
                patch_func=candidate.patch_func,
                target_file=candidate.target_file,
                target_func=candidate.target_func,
                analysis_report=f"Verification failed due to unexpected error: {str(e)}",
                is_vulnerable=False,  # Conservative judgment
                verdict_category="ERROR",
                involved_functions=[],
                peer_functions=[],
                anchor_evidence=[],
                trace=[],
                defense_mechanism=None,
                constraint_status=f"Error: {str(e)}"
            )
            findings.append(finding)
            continue  # Continue processing next candidate

    return {"final_findings": findings}

def baseline_validation_node(state: VerificationState) -> Dict[str, Any]:
    """
    Baseline (Ablation Phase 4):
    Simple Verification with Tool Support (but no multi-round debate).
    Agent can use tools to verify its understanding in a single pass.
    """
    candidates: List[SearchResultItem] = state["candidates"]
    feature: PatchFeatures = state["feature_context"]
    mode = state["mode"]
    vul_id = state.get("vul_id", "")
    findings = []
    
    # 1. Candidate Filtering (Same as main node)
    filtered_candidates : List[SearchResultItem] = []
    for c in candidates:
        if c.verdict in ("VULNERABLE") and c.confidence >= 0.4:
            filtered_candidates.append(c)
            
    if not filtered_candidates:
        print("    [Skip] No valid candidates after filtering (Verdict/Confidence).")
        return {"final_findings": []}

    # [Dynamic Penalty] Sort by rank (asc) to ensure efficient pruning
    filtered_candidates.sort(key=lambda x: x.rank if x.rank != -1 else 999)
    
    # [Limit] Only process top N candidates to save cost
    MAX_CANDIDATES = 10
    if len(filtered_candidates) > MAX_CANDIDATES:
        print(f"    [Limit] Truncating candidates from {len(filtered_candidates)} to {MAX_CANDIDATES}")
        filtered_candidates = filtered_candidates[:MAX_CANDIDATES]
    
    # [Adaptive Threshold]
    current_threshold = 0.4
    if mode == 'benchmark':
        THRESHOLD_PENALTY_STEP = 0.0
    else:
        THRESHOLD_PENALTY_STEP = 0.1
    GRACE_RANK_LIMIT = 1

    # 2. Iteration
    processed_funcs = set()
    for candidate in filtered_candidates:
        try:
            if candidate.rank != -1 and candidate.rank > GRACE_RANK_LIMIT:
                 if candidate.confidence < current_threshold:
                    print(f"    [Skip-Adaptive] Rank {candidate.rank} (Conf {candidate.confidence:.2f}) < Threshold {current_threshold:.2f}")
                    continue
            unique_key = f"{candidate.target_file}::{candidate.target_func}"
            if unique_key in processed_funcs: continue
            processed_funcs.add(unique_key)
            
            # Initialize CodeNavigator and create tools
            from core.navigator import CodeNavigator
            repo_path = candidate.repo_path
            
            target_version = None
            if mode == 'benchmark':
                parts = candidate.target_func.split(':')
                if len(parts) >= 2:
                    target_version = parts[1]
            
            navigator = CodeNavigator(repo_path, target_version=target_version)
            shared_tool_cache: Dict[str, Any] = {}
            tools = create_tools(navigator, candidate.target_file, shared_tool_cache)
            
            # 3. Context Preparation
            raw_code = candidate.code_content
            try:
                matched_lines = [m.line_no for m in candidate.evidence.aligned_vuln_traces if m.line_no is not None]
                s_pre_code = feature.slices[candidate.patch_func].s_pre
                hint_vars = list(set(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', s_pre_code)))
                slicer = TargetSlicerAdapter(raw_code)
                sliced = slicer.slice_context(matched_lines, hint_vars)
                if len(sliced) < len(raw_code) * 0.9:
                    raw_code = f"// [Note] Code Sliced for Focus.\n{sliced}"
            except:
                pass
                
            full_code_context = f"File: {candidate.target_file}\nFunction: {candidate.target_func.split(':')[-1]}\nCode:\n{raw_code}\n"
        
            # 4. Prompt Construction
            all_diffs = []
            for p in feature.patches:
                header = f"--- {p.file_path} : {p.function_name} ---"
                diff_content = p.clean_diff if p.clean_diff else p.raw_diff
                all_diffs.append(f"{header}\n{diff_content}")
            combined_diffs = "\n\n".join(all_diffs)
            
            vuln_info = f"""
[Root Cause]: {feature.semantics.root_cause}
[Attack Chain]: {feature.semantics.attack_chain}
[Patch Defense]: {feature.semantics.patch_defense}
[Reference (Pre-Patch)]:
{feature.slices[candidate.patch_func].s_pre}
[Reference (Post-Patch)]:
{feature.slices[candidate.patch_func].s_post}
[Diffs]:
{combined_diffs}
"""

            # 5. LLM and Three-Role Debate
            llm = ChatOpenAI(
                base_url=os.getenv("API_BASE"),
                api_key=os.getenv("API_KEY"),
                model=os.getenv("MODEL_NAME", "gpt-4o"),
                temperature=0
            )
            
            print(f"    [Baseline] Verifying {candidate.target_func} with three-role debate...")
            
            # Run simplified three-role debate (Red → Blue → Judge)
            # - Red: Argues vulnerability EXISTS, finds anchor chain endpoints
            # - Blue: Argues vulnerability does NOT exist, finds defenses
            # - Judge: Makes final verdict based on both arguments
            res = run_baseline_debate(
                feature=feature,
                candidate=candidate,
                target_code=full_code_context,
                tools=tools,
                llm=llm
            )
            
            # 6. Result Mapping
            finding = VulnerabilityFinding(
                vul_id=state["vul_id"],
                cwe_id=feature.semantics.cwe_id or "Unknown",
                cwe_name=feature.semantics.cwe_name or "Unknown",
                group_id=feature.group_id,
                repo_path=candidate.repo_path,
                patch_file=candidate.patch_file,
                patch_func=candidate.patch_func,
                target_file=candidate.target_file,
                target_func=candidate.target_func,
                analysis_report=res.reasoning,
                is_vulnerable=res.is_vulnerable,
                involved_functions=[],
                peer_functions=[],
                anchor_evidence=[],
                trace=[],
                defense_mechanism=None,
                constraint_status="Baseline: No tool analysis"
            )
            findings.append(finding)
            if not res.is_vulnerable:
                current_threshold += THRESHOLD_PENALTY_STEP
                # print(f"    [Adaptive] Verdict SAFE -> Raising threshold to {current_threshold:.2f}")
                
        except Exception as e:
            print(f"    [Baseline] Error for {candidate.target_func}: {e}")
            import traceback
            traceback.print_exc()
            continue

    return {"final_findings": findings}