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

from extraction.slicer import Slicer
from extraction.pdg import PDGBuilder

# Import type-specific verification checklists
from search.verification_checklists import format_checklist_for_prompt

class StepAnalysis(BaseModel):
    """Evidence item for vulnerability analysis"""
    role: str = Field(description="One of: 'Origin', 'Impact', 'Trace', 'Defense'")
    file_path: Optional[str] = Field(default=None, description="File path where this step occurs")
    func_name: Optional[str] = Field(default=None, description="Function name containing this step")
    line_number: Optional[int] = Field(default=None, description="Line number in target file")
    code_content: Optional[str] = Field(default=None, description="Code snippet")
    observation: str = Field(description="Brief analysis of this step")


class AnchorMapping(BaseModel):
    """Simplified anchor mapping result"""
    anchor_type: str = Field(description="'origin' or 'impact'")
    reference_line: str = Field(description="Reference slice anchor line")
    target_line: Optional[int] = Field(default=None, description="Mapped target line number")
    target_code: Optional[str] = Field(default=None, description="Mapped target code")
    is_mapped: bool = Field(description="Whether mapping succeeded")
    reason: Optional[str] = Field(default=None, description="Reason if mapping failed or semantic equivalence explanation")


class JudgeOutput(BaseModel):
    """Final Judge Output"""
    c_cons_satisfied: bool = Field(description="Consistency constraint satisfied")
    c_reach_satisfied: bool = Field(description="Reachability constraint satisfied")
    c_def_satisfied: bool = Field(description="Defense blocks attack")
    is_vulnerable: bool = Field(description="True if vulnerable, False otherwise")
    verdict_category: str = Field(description="'VULNERABLE', 'SAFE-Blocked', 'SAFE-Mismatch', 'SAFE-Unreachable', 'SAFE-TypeMismatch', or 'SAFE-OutOfScope'")
    origin_anchor: Optional[StepAnalysis] = Field(default=None, description="Origin anchor evidence")
    impact_anchor: Optional[StepAnalysis] = Field(default=None, description="Impact anchor evidence")
    trace: List[StepAnalysis] = Field(default_factory=list, description="Trace between origin and impact")
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
    origin_line: Optional[int] = Field(default=None, description="Line number where vulnerable state is created")
    origin_code: Optional[str] = Field(default=None, description="Code at origin point")
    impact_line: Optional[int] = Field(default=None, description="Line number where vulnerability triggers")
    impact_code: Optional[str] = Field(default=None, description="Code at impact point")
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
    """Round 1 Red Agent: Validate C_cons without tools"""
    origin_mapping: Optional[AnchorMapping] = Field(default=None, description="Origin anchor mapping result")
    impact_mapping: Optional[AnchorMapping] = Field(default=None, description="Impact anchor mapping result")
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
    """Round 1 Judge: Adjudicate C_cons"""
    c_cons_satisfied: bool = Field(description="Whether C_cons is satisfied")
    verdict: str = Field(description="'SAFE-Mismatch' or 'PROCEED'")
    validated_origin: Optional[AnchorMapping] = Field(default=None, description="Validated origin mapping")
    validated_impact: Optional[AnchorMapping] = Field(default=None, description="Validated impact mapping")


# ==============================================================================
# Round 2: C_reach + C_def Verification (With Tools) - Simplified Design
# ==============================================================================

class AttackPathStep(BaseModel):
    """A step in the attack path"""
    step_type: str = Field(description="'Origin', 'Trace', 'Impact', or 'Call'")
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
    """Round 2 Judge: Final adjudication"""
    c_cons_satisfied: bool = Field(description="Whether C_cons is satisfied")
    c_reach_satisfied: bool = Field(description="Whether C_reach is satisfied")
    c_def_satisfied: bool = Field(description="Whether defense blocks attack")
    origin_in_function: bool = Field(default=True, description="Whether origin is in current function")
    impact_in_function: bool = Field(default=True, description="Whether impact is in current function")
    verdict: str = Field(description="Final verdict category")
    validated_defense: Optional[DefenseCheckResult] = Field(default=None, description="Validated defense if any")
    
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
                origin_mapping=None,
                impact_mapping=None,
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
                validated_origin=None,
                validated_impact=None
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
                origin_in_function=True,
                impact_in_function=True,
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
                origin_anchor=None,
                impact_anchor=None,
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
    Format Phase 3 aligned_vuln_traces into readable mapping summary
    
    Args:
        sf: SliceFeature (contains pre_origins, pre_impacts)
        ev: MatchEvidence (contains aligned_vuln_traces)
    
    Returns:
        Formatted mapping summary string
    """
    def extract_line_no(line: str) -> int:
        """Extract line number from line with line number marker"""
        match = re.match(r'^\[\s*(\d+)\]', line.strip())
        return int(match.group(1)) if match else -1
    
    def find_target_mapping(anchor_line: str, aligned_traces: list) -> tuple:
        """
        Find the target line that maps to a given anchor line.
        Returns: (target_line_no, target_code, similarity) or (None, None, 0.0)
        """
        anchor_ln = extract_line_no(anchor_line)
        if anchor_ln == -1:
            # Try content-based matching if no line number
            anchor_content = re.sub(r'^\[\s*\d+\]', '', anchor_line).strip()
            for trace in aligned_traces:
                if trace.target_line and anchor_content in trace.slice_line:
                    return (trace.line_no, trace.target_line, trace.similarity)
            return (None, None, 0.0)
        
        # Line number based matching
        for trace in aligned_traces:
            slice_ln = extract_line_no(trace.slice_line)
            if slice_ln == anchor_ln and trace.target_line:
                return (trace.line_no, trace.target_line, trace.similarity)
        
        return (None, None, 0.0)
    
    lines = []
    
    # Process Origin Anchors
    origin_lines = []
    for anchor in (sf.pre_origins or []):
        target_ln, target_code, sim = find_target_mapping(anchor, ev.aligned_vuln_traces)
        quality = 'good' if sim > 0.6 else ('weak' if sim > 0.3 else 'missing')
        if target_ln:
            origin_lines.append(f"  [{quality.upper()}] `{anchor.strip()}`")
            origin_lines.append(f"      → Target Line {target_ln}: `{target_code.strip() if target_code else ''}` (sim={sim:.2f})")
        else:
            origin_lines.append(f"  [UNMAPPED] `{anchor.strip()}`")
    
    # Process Impact Anchors
    impact_lines = []
    for anchor in (sf.pre_impacts or []):
        target_ln, target_code, sim = find_target_mapping(anchor, ev.aligned_vuln_traces)
        quality = 'good' if sim > 0.6 else ('weak' if sim > 0.3 else 'missing')
        if target_ln:
            impact_lines.append(f"  [{quality.upper()}] `{anchor.strip()}`")
            impact_lines.append(f"      → Target Line {target_ln}: `{target_code.strip() if target_code else ''}` (sim={sim:.2f})")
        else:
            impact_lines.append(f"  [UNMAPPED] `{anchor.strip()}`")
    
    lines.append("**Origin Anchor Mappings**:")
    if origin_lines:
        lines.extend(origin_lines)
    else:
        lines.append("  [No origin anchors defined in Phase 2]")
    
    lines.append("")
    lines.append("**Impact Anchor Mappings**:")
    if impact_lines:
        lines.extend(impact_lines)
    else:
        lines.append("  [No impact anchors defined in Phase 2]")
    
    return "\n".join(lines)


def run_round1_red(
    feature: PatchFeatures,
    candidate,  # SearchResultItem
    target_code_with_lines: str,
    llm: ChatOpenAI
) -> Round1RedOutput:
    """
    Round 1 Red Agent: Validate C_cons (Consistency Constraint) WITHOUT tools.
    
    Core logic:
    1. Extract Phase 2 pre_origins/pre_impacts
    2. Extract Phase 3 aligned_vuln_traces mapping
    3. Let Red Agent evaluate mapping quality
    4. If mapping is missing, let Red Agent search in target code
    5. If Origin or Impact not found -> Direct SAFE
    """
    print("      [Round1-Red] Validating C_cons (no tools)...")
    
    sf = feature.slices.get(candidate.patch_func)
    ev = candidate.evidence
    
    # Build Phase 3 Mapping Summary
    phase3_summary = build_phase3_mapping_summary(sf, ev) if sf else "No slice found for this function"
    
    # Build Prompt
    user_input = f"""
### Phase 2 Anchors (From Reference Vulnerability Analysis)

**Origin Anchors** (where vulnerable state is created):
{chr(10).join(sf.pre_origins) if sf and sf.pre_origins else '[None identified in Phase 2]'}

**Impact Anchors** (where vulnerability is triggered):
{chr(10).join(sf.pre_impacts) if sf and sf.pre_impacts else '[None identified in Phase 2]'}

### Phase 3 Mapping Results (Slice → Target Alignment)
{phase3_summary}

### Target Code (with line numbers)
{target_code_with_lines}

### Vulnerability Context
- **Type**: {feature.semantics.vuln_type.value if feature.semantics.vuln_type else 'Unknown'}
- **CWE**: {feature.semantics.cwe_id or 'Unknown'} - {feature.semantics.cwe_name or ''}
- **Root Cause**: {feature.semantics.root_cause}
- **Attack Path**: {feature.semantics.attack_path}

### Reference Slice (Pre-Patch)
{sf.s_pre if sf else 'Not available'}

**TASK**:
1. Validate the Phase 3 mapping quality for each Origin and Impact anchor
2. If any anchor is unmapped or has weak mapping (sim < 0.6), search the target code for a semantically equivalent statement
3. Determine if C_cons can be satisfied (both Origin and Impact must be mapped)
4. If either is missing, verdict = SAFE; otherwise verdict = PROCEED
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
                        origin_mapping=None,
                        impact_mapping=None,
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
                    origin_mapping=None,
                    impact_mapping=None,
                    attack_path_exists=False,
                    c_cons_satisfied=False,
                    verdict="PROCEED",
                    safe_reason=f"JSON parsing error: {str(e)}"
                )
            messages.append(HumanMessage(content=f"Parse error: {e}. Please output valid JSON only."))
    
    # All retries failed, return default value
    return Round1RedOutput(
        origin_mapping=None,
        impact_mapping=None,
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
    
    # Format Red's claims for Blue (using simplified model)
    origin_claim = "  [No origin mapping]"
    if red_output.origin_mapping:
        m = red_output.origin_mapping
        status = "✓ MAPPED" if m.is_mapped else "✗ NOT MAPPED"
        origin_claim = f"  [{status}] Reference: {m.reference_line}"
        if m.is_mapped:
            origin_claim += f"\n      → Target Line {m.target_line}: `{m.target_code}`"
            if m.reason:
                origin_claim += f"\n      Reason: {m.reason}"
    
    impact_claim = "  [No impact mapping]"
    if red_output.impact_mapping:
        m = red_output.impact_mapping
        status = "✓ MAPPED" if m.is_mapped else "✗ NOT MAPPED"
        impact_claim = f"  [{status}] Reference: {m.reference_line}"
        if m.is_mapped:
            impact_claim += f"\n      → Target Line {m.target_line}: `{m.target_code}`"
            if m.reason:
                impact_claim += f"\n      Reason: {m.reason}"
    
    user_input = f"""
### Red Agent's C_cons Claim

**C_cons Satisfied**: {red_output.c_cons_satisfied}
**Attack Path Exists**: {red_output.attack_path_exists}
**Verdict**: {red_output.verdict}
{f"**Safe Reason**: {red_output.safe_reason}" if red_output.safe_reason else ""}

**Origin Mapping**:
{origin_claim}

**Impact Mapping**:
{impact_claim}

### Target Code (with line numbers)
{target_code_with_lines}

### Vulnerability Context
- **Type**: {feature.semantics.vuln_type.value if feature.semantics.vuln_type else 'Unknown'}
- **Root Cause**: {feature.semantics.root_cause}
- **Attack Path**: {feature.semantics.attack_path}

**TASK**:
1. Examine Red's Origin and Impact mappings - are they semantically correct?
2. **CRITICAL - Check Causality**: Does Origin → Impact satisfy causal relationship?
   - **Execution Order**: Can Origin execute BEFORE Impact in any feasible path?
   - **Data Flow**: Does the variable/state created at Origin flow to Impact?
   - **Common Causality Errors**:
     * **Reversed Order**: Impact line number < Origin line number (check if this is valid, e.g., callbacks/macros)
     * **Cleanup Confusion**: Both are cleanup/deallocation code (not vulnerable use)
     * **Different Variables**: Origin affects X, Impact uses Y (no connection)
     * **Blocked Path**: Control flow (return/goto/exit) prevents Origin from reaching Impact
   - If causality is INVALID (e.g., line 4011 cannot be caused by line 4014 that comes after it), REFUTE with "Causality Violation"
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
    
    # Format origin/impact info from simplified model
    origin_info = "Not mapped"
    if red_output.origin_mapping and red_output.origin_mapping.is_mapped:
        m = red_output.origin_mapping
        origin_info = f"Line {m.target_line}: `{m.target_code}` (ref: {m.reference_line})"
    
    impact_info = "Not mapped"
    if red_output.impact_mapping and red_output.impact_mapping.is_mapped:
        m = red_output.impact_mapping
        impact_info = f"Line {m.target_line}: `{m.target_code}` (ref: {m.reference_line})"
    
    user_input = f"""
### Red Agent's C_cons Claim
- **C_cons Satisfied**: {red_output.c_cons_satisfied}
- **Attack Path Exists**: {red_output.attack_path_exists}
- **Verdict**: {red_output.verdict}
- **Origin Mapping**: {origin_info}
- **Impact Mapping**: {impact_info}
{f"- **Safe Reason**: {red_output.safe_reason}" if red_output.safe_reason else ""}

### Blue Agent's Refutation
- **Refutes Mapping**: {blue_output.refutes_mapping}
- **Refutation Reason**: {blue_output.refutation_reason or 'N/A'}
- **Verdict**: {blue_output.verdict}

### Target Code (for verification)
{target_code_with_lines}

### Vulnerability Context
- **Type**: {feature.semantics.vuln_type.value if feature.semantics.vuln_type else 'Unknown'}
- **Root Cause**: {feature.semantics.root_cause}
- **Attack Path**: {feature.semantics.attack_path}
- **Fix Mechanism**: {feature.semantics.fix_mechanism}

**TASK**:
1. Evaluate if Red successfully mapped both Origin and Impact anchors
2. Evaluate if Blue's refutation is valid
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
                        validated_origin=red_output.origin_mapping,
                        validated_impact=red_output.impact_mapping
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
                    validated_origin=red_output.origin_mapping,
                    validated_impact=red_output.impact_mapping
                )
            messages.append(HumanMessage(content=f"Parse error: {e}. Please output valid JSON only."))
    
    # All retries failed, return default value (proceed to Round 2)
    return Round1JudgeOutput(
        c_cons_satisfied=True,
        verdict="PROCEED",
        validated_origin=red_output.origin_mapping,
        validated_impact=red_output.impact_mapping
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
            validated_origin=None,
            validated_impact=None
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
    
    # Build Round 1 validated mappings summary (using simplified model)
    origin_info = "Not validated"
    impact_info = "Not validated"
    if round1_result.validated_origin and round1_result.validated_origin.is_mapped:
        o = round1_result.validated_origin
        origin_info = f"Line {o.target_line}: `{o.target_code}` (from: {o.reference_line})"
    if round1_result.validated_impact and round1_result.validated_impact.is_mapped:
        i = round1_result.validated_impact
        impact_info = f"Line {i.target_line}: `{i.target_code}` (from: {i.reference_line})"
    
    user_input = f"""
### Round 1 Validated Mappings
**Origin Anchor**: {origin_info}
**Impact Anchor**: {impact_info}
**Round 1 C_cons**: {round1_result.c_cons_satisfied}

### Reference Attack Path
{feature.semantics.attack_path}

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
1. If Round 1 had issues, use tools to reinforce C_cons (verify Origin/Impact mappings)
2. Construct the attack path from Origin to Impact
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
- **Fix Mechanism**: {feature.semantics.fix_mechanism}

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
- **Fix Mechanism**: {feature.semantics.fix_mechanism}

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
                        origin_in_function=True,
                        impact_in_function=True,
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
                    origin_in_function=True,
                    impact_in_function=True,
                    verdict="PROCEED",
                    validated_defense=find_strongest_defense()
                )
            messages.append(HumanMessage(content=f"Parse error: {e}. Please output valid JSON only."))
    
    # All retries failed, return default value
    return Round2JudgeOutput(
        c_cons_satisfied=round1_result.c_cons_satisfied,
        c_reach_satisfied=round2_red_output.c_reach_satisfied,
        c_def_satisfied=round2_blue_output.any_defense_found,
        origin_in_function=True,
        impact_in_function=True,
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
            origin_in_function=True,
            impact_in_function=True,
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
        round1_red: Round 1 Red Agent output (origin/impact mappings)
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
    
    # === Format Round 1 Evidence (Simplified Model) ===
    
    # Round 1 Red: Origin/Impact mappings (single object, not list)
    origin_info_r1 = "Not mapped"
    if round1_red.origin_mapping and round1_red.origin_mapping.is_mapped:
        o = round1_red.origin_mapping
        origin_info_r1 = f"Line {o.target_line}: `{o.target_code}` (from: {o.reference_line})"
    
    impact_info_r1 = "Not mapped"
    if round1_red.impact_mapping and round1_red.impact_mapping.is_mapped:
        i = round1_red.impact_mapping
        impact_info_r1 = f"Line {i.target_line}: `{i.target_code}` (from: {i.reference_line})"
    
    round1_red_summary = f"""
**Round 1 Red Agent (C_cons Evidence)**:
- C_cons Satisfied: {round1_red.c_cons_satisfied}
- Attack Path Exists: {round1_red.attack_path_exists}
- Verdict: {round1_red.verdict}
{f"- Safe Reason: {round1_red.safe_reason}" if round1_red.safe_reason else ""}

Origin Mapping: {origin_info_r1}
Impact Mapping: {impact_info_r1}
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
- Origin In Function: {round2_judge.origin_in_function}
- Impact In Function: {round2_judge.impact_in_function}
- Verdict: {round2_judge.verdict}
"""
    
    # Format validated anchors (simplified model)
    origin_info = "Not mapped"
    impact_info = "Not mapped"
    if round1_judge.validated_origin and round1_judge.validated_origin.is_mapped:
        o = round1_judge.validated_origin
        origin_info = f"Line {o.target_line}: `{o.target_code}` (ref: {o.reference_line})"
    if round1_judge.validated_impact and round1_judge.validated_impact.is_mapped:
        i = round1_judge.validated_impact
        impact_info = f"Line {i.target_line}: `{i.target_code}` (ref: {i.reference_line})"
    
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
- **Origin Anchor**: {origin_info}
- **Impact Anchor**: {impact_info}

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
- **Attack Path**: {feature.semantics.attack_path}
- **Fix Mechanism**: {feature.semantics.fix_mechanism}

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

4. **origin_anchor**: StepAnalysis for the origin point
   - Use the ACTUAL line number from the target code
   - Include the exact code content

5. **impact_anchor**: StepAnalysis for the impact point
   - Use the ACTUAL line number from the target code
   - Include the exact code content

6. **trace**: List of StepAnalysis from Round 2 Red's attack_path
   - Include Trace/Call steps between Origin and Impact
   - Use actual line numbers and code from the attack path

7. **defense_mechanism**: StepAnalysis from Round 2 Blue's defense_checks (if SAFE-Blocked)
   - Use the strongest defense found
   - Include its location and code

8. **analysis_report**: Concise 1-2 sentence summary

**CRITICAL - CAUSALITY CHECK**:
- **VERIFY Origin → Impact Causality**: Origin MUST causally precede Impact
  * Check data flow: Does the variable/state created at Origin flow to Impact?
  * Check control flow: Is there a feasible execution path from Origin to Impact?
  * If Origin does NOT cause Impact (wrong order, different variables, disconnected flow):
    → Either SWAP them (if Impact actually creates the state and Origin triggers it)
    → Or mark as SAFE-Mismatch (if no causal relationship exists)
- **Line number order is NOT always indicative** (cross-function calls, callbacks, async operations can reverse order)
- **BUT data/control flow MUST be valid**: Origin's effect must reach Impact
- Use the ACTUAL line numbers from Round 2 Red's attack_path
- Do NOT make up line numbers - use only what's provided in the evidence
- For defense_mechanism, use the location/code from Round 2 Blue's defense_checks

**CAUSALITY VALIDATION PATTERNS**:
- For UAF: Origin should be deallocation, Impact should be subsequent use
- For NPD: Origin should create NULL state, Impact should dereference
- For Race: Origin should be conflicting access, Impact should be vulnerable access
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
        (round2_judge.origin_in_function or round2_judge.impact_in_function)
    )
    
    # Determine verdict category
    if not round2_judge.c_cons_satisfied:
        verdict_category = "SAFE-Mismatch"
    elif not round2_judge.c_reach_satisfied:
        verdict_category = "SAFE-Unreachable"
    elif not round2_judge.origin_in_function and not round2_judge.impact_in_function:
        verdict_category = "SAFE-OutOfScope"
    elif round2_judge.c_def_satisfied:
        verdict_category = "SAFE-Blocked"
    else:
        verdict_category = "VULNERABLE" if is_vulnerable else "SAFE-Unknown"
    
    # === Build origin/impact anchors from Round 2 Red's attack_path ===
    # This is the primary source for accurate line numbers (simplified model)
    origin_anchor = None
    impact_anchor = None
    trace_steps = []
    
    for i, step in enumerate(round2_red.attack_path):
        step_analysis = StepAnalysis(
            role=step.step_type,
            file_path=candidate.target_file,
            func_name=candidate.target_func.split(':')[-1],
            line_number=step.target_line,
            code_content=step.target_code,
            observation=f"Step {i+1}: matches_reference={step.matches_reference}"
        )
        
        if step.step_type == 'Origin' and origin_anchor is None:
            origin_anchor = step_analysis
        elif step.step_type == 'Impact' and impact_anchor is None:
            impact_anchor = step_analysis
        elif step.step_type in ('Trace', 'Call'):
            trace_steps.append(step_analysis)
    
    # Fallback to Round 1 Judge's validated mappings if Round 2 didn't have attack path
    if origin_anchor is None and round1_judge.validated_origin and round1_judge.validated_origin.is_mapped:
        o = round1_judge.validated_origin
        origin_anchor = StepAnalysis(
            role="Origin",
            file_path=candidate.target_file,
            func_name=candidate.target_func.split(':')[-1],
            line_number=o.target_line,
            code_content=o.target_code,
            observation=f"Mapped from reference: {o.reference_line}"
        )
    
    if impact_anchor is None and round1_judge.validated_impact and round1_judge.validated_impact.is_mapped:
        i = round1_judge.validated_impact
        impact_anchor = StepAnalysis(
            role="Impact",
            file_path=candidate.target_file,
            func_name=candidate.target_func.split(':')[-1],
            line_number=i.target_line,
            code_content=i.target_code,
            observation=f"Mapped from reference: {i.reference_line}"
        )
    
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
        origin_anchor=origin_anchor,
        impact_anchor=impact_anchor,
        trace=trace_steps,
        defense_mechanism=defense_mechanism,
        analysis_report=analysis_report
    )


ROUND3_FINAL_JUDGE_PROMPT = """You are the **Final Judge** for Round 3: Verdict Synthesis.

Your task is to synthesize all evidence from Round 1 (C_cons validation) and Round 2 (C_reach + C_def validation)
into a final, structured verdict following the JudgeOutput schema.

## YOUR RESPONSIBILITIES

### 1. Constraint Summary
Summarize the constraint outcomes from previous rounds:
- **C_cons**: Was consistency established in Round 1 and confirmed in Round 2?
- **C_reach**: Was reachability established in Round 2?
- **C_def**: Was a defense found in Round 2?

### 2. Attack Path Type Validation
Verify that the identified attack path corresponds to the original vulnerability type:
- If the reference is UAF, the target should also exhibit UAF pattern
- If patterns don't match, set attack_path_matches_vuln_type = false

### 3. Location Validation
Check if at least one anchor (Origin or Impact) is within the current target function:
- Use the function name from context
- If both anchors are in external functions → SAFE-OutOfScope

### 4. Final Verdict
Apply the decision logic:
- VULNERABLE: C_cons ∧ C_reach ∧ ¬C_def ∧ attack_path_matches_vuln_type ∧ (origin_in_function ∨ impact_in_function)
- SAFE-Blocked: C_cons ∧ C_reach ∧ C_def
- SAFE-Mismatch: ¬C_cons
- SAFE-Unreachable: C_cons ∧ ¬C_reach
- SAFE-TypeMismatch: ¬attack_path_matches_vuln_type
- SAFE-OutOfScope: ¬origin_in_function ∧ ¬impact_in_function

### 5. Evidence Chain Construction
Build the validated evidence chain:
- **origin_anchor**: StepAnalysis with exact location and code
- **impact_anchor**: StepAnalysis with exact location and code
- **trace**: List of intermediate steps (if vulnerable)
- **defense_mechanism**: StepAnalysis for defense (if SAFE-Blocked)

### 6. Report Generation
- **analysis_report**: Concise 1-2 sentence summary
- **detailed_markdown_report**: Full exploitation report with sections for Verdict, Attack Path (if vulnerable), Defense (if blocked), and Evidence Summary

## OUTPUT REQUIREMENTS
Your output MUST be a valid JudgeOutput JSON with ALL required fields properly filled.
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
# 2. Prompts (Updated: Constraint-Based per Methodology.tex Section 4.4)
# ==============================================================================


# ==============================================================================
# Round 1: C_cons Validation Prompts (No Tools)
# ==============================================================================

ROUND1_RED_PROMPT = """You are the **Red Agent** performing **Round 1: C_cons (Consistency) Validation**.
Your task is to validate whether the target code exhibits the SAME vulnerability mechanism as the reference.
You have **NO TOOLS** available - analyze ONLY the provided information.

## INPUTS PROVIDED
1. **Phase 2 Anchors**: Pre-computed Origin/Impact anchors from vulnerability analysis
2. **Phase 3 Mapping Results**: Alignment showing which slice lines map to which target lines
3. **Target Code**: The actual target function code with line numbers
4. **Vulnerability Semantics**: Type, root cause, and attack path

## YOUR TASK: Establish C_cons (Consistency Constraint)

### Step 1: Validate Phase 3 Anchor Mappings
For each Origin and Impact anchor from Phase 2:
- Check if Phase 3 successfully mapped it (has a target_line with similarity > 0.3)
- Assess mapping quality: 'good' (sim > 0.6), 'weak' (0.3-0.6), 'missing' (no match or sim < 0.3)
- Record the Phase 3 mapping details

### Step 2: Manual Search (if Phase 3 mapping is missing/weak)
If an anchor is not mapped or has weak mapping:
- Search the target code for a semantically equivalent statement
- Consider: same operation type, same variable role, same data flow position
- Report the line number and code if found
- Explain why it is semantically equivalent

### Step 3: Verify Attack Path Completeness
Using the Attack Path description:
- Trace each step in the target code
- Identify whether all key steps exist (not just Origin and Impact)
- For UAF: verify allocation → free → use pattern exists
- For NPD: verify assignment → null path → dereference exists

### Step 4: Verify Causality (CRITICAL)
**Origin MUST causally precede Impact**:
- **Execution Order**: Can Origin execute BEFORE Impact?
  - Check line numbers: If Impact line < Origin line, verify this is valid (callbacks, macros, goto/error handling)
  - If simple sequential code, later lines CANNOT cause earlier lines
- **Data Flow**: Does the state created at Origin flow to Impact?
  - Same variable throughout the path?
  - Or is Origin affecting X while Impact uses Y (disconnected)?
- **Common Invalid Patterns**:
  - Both are cleanup/deallocation (no vulnerable use)
  - Different variables with no flow connection
  - Control flow blocks Origin from reaching Impact

### Step 5: C_cons Viability Decision
**CRITICAL RULES**:
- C_cons requires BOTH Origin AND Impact anchors to be mapped (via Phase 3 or manual search)
- The mappings must be SEMANTICALLY equivalent, not just syntactically similar
- **The mappings must satisfy CAUSALITY**: Origin → Impact must have valid causal relationship
- If Origin is missing (neither Phase 3 nor manual) → verdict = SAFE
- If Impact is missing (neither Phase 3 nor manual) → verdict = SAFE
- If mechanism differs fundamentally → verdict = SAFE
- **If causality is invalid (e.g., Impact at line 4011 cannot be caused by Origin at line 4014) → verdict = SAFE**
- If both are found, semantically equivalent, AND causality is valid → verdict = PROCEED

## WHAT COUNTS AS SEMANTIC EQUIVALENCE
- **Origin anchor**: Statement that creates the vulnerable state
  - UAF: allocation (kmalloc, kzalloc, vmalloc, etc.)
  - NPD: pointer assignment that can be NULL
  - Race: lock acquisition or shared resource access
  - Integer Overflow: arithmetic operation on integer
  
- **Impact anchor**: Statement where vulnerability triggers
  - UAF: use/dereference after free (ptr->field, *ptr)
  - NPD: dereference of potentially NULL pointer
  - Race: concurrent access without protection
  - Integer Overflow: use of overflowed value in sensitive operation

## OUTPUT REQUIREMENTS
Provide a complete structured output with:
1. All Origin anchor mappings (Phase 3 result + manual search if needed)
2. All Impact anchor mappings (Phase 3 result + manual search if needed)
3. Attack path trace in target code
4. C_cons verdict with detailed reasoning
"""

ROUND1_BLUE_PROMPT = """You are the **Blue Agent** performing **Round 1: C_cons Refutation**.
Your task is to challenge Red's claim that C_cons (Consistency) is satisfied.
You have **NO TOOLS** available - analyze ONLY the provided information.

## INPUTS PROVIDED
1. **Red's C_cons Claim**: Red's Origin/Impact mappings and reasoning
2. **Target Code**: The actual target function code with line numbers
3. **Vulnerability Semantics**: Type, root cause, and attack path

## YOUR TASK: Refute C_cons if Invalid

### Mode 1: Refute Origin Mapping
If Red's Origin mapping is incorrect:
- The mapped statement performs a DIFFERENT operation
- The variable serves a DIFFERENT role in data flow
- The data type or context is fundamentally different

### Mode 2: Refute Impact Mapping
If Red's Impact mapping is incorrect:
- The mapped statement is not the actual impact point
- The variable being used is different from the one at Origin
- The operation type is different (e.g., read vs write, local vs escaped)

### Mode 3: Mechanism Mismatch
If the target code has a fundamentally different vulnerability mechanism:
- Red claims UAF but target is actually NPD or vice versa
- The data flow pattern is completely different
- The control flow structure prevents the claimed vulnerability

## REFUTATION REQUIREMENTS
- **Be Specific**: Quote exact code and line numbers
- **Be Semantic**: Explain WHY the mapping is wrong, not just that it looks different
- **Be Honest**: If Red's mapping is actually correct, CONCEDE rather than fabricating issues

## VERDICT OPTIONS
- **SAFE**: You successfully refuted C_cons (provide strong evidence)
- **CONCEDE**: Red's C_cons claim is valid, you cannot refute it
- **CONTESTED**: You raised valid concerns but they need tool verification in Round 2

## OUTPUT REQUIREMENTS
Provide a complete structured output with refutation details for each anchor and overall verdict.
"""

ROUND1_JUDGE_PROMPT = """You are the **Judge** adjudicating **Round 1: C_cons Decision**.
Your task is to evaluate Red's C_cons claim and Blue's refutation to decide if C_cons is satisfied.

## INPUTS PROVIDED
1. **Red's C_cons Claim**: Origin/Impact mappings with reasoning
2. **Blue's Refutation**: Challenges to Red's mappings
3. **Target Code**: For verification
4. **Vulnerability Semantics**: For context

## YOUR TASK: Adjudicate C_cons

### Evaluation Criteria
1. **Mapping Validity**: Are Red's Origin and Impact mappings semantically correct?
2. **CAUSALITY CHECK (CRITICAL)**: Does Origin → Impact satisfy causal relationship?
   - **Execution Order**: Can Origin execute BEFORE Impact in any feasible path?
   - **Data Flow**: Does the state created at Origin flow to Impact?
   - **RED FLAG Patterns**:
     * If Impact line < Origin line (e.g., line 4011 vs 4014): Check if this is valid
       - Valid cases: callbacks, macro expansions, error handling with goto
       - Invalid cases: simple sequential code where later line cannot affect earlier line
     * If both are cleanup/deallocation operations (no vulnerable use)
     * If different variables (no data flow connection)
   - **Decision Rule**: If Blue raises "Causality Violation" and it's valid → C_cons NOT satisfied
3. **Refutation Strength**: Did Blue provide valid counter-evidence?
4. **Evidence Quality**: Which side has more concrete, verifiable claims?

### Decision Logic
- If Red fails to map either Origin or Impact → C_cons NOT satisfied
- If Blue successfully refutes causality (invalid Origin→Impact order/flow) → C_cons NOT satisfied
- If Blue successfully refutes semantic equivalence → C_cons NOT satisfied
- If Blue's refutation is weak/speculative → Give benefit of doubt to Red
- If mappings are valid but Blue raises tool-dependent concerns → PROCEED to Round 2

## VERDICT OPTIONS
- **SAFE-Mismatch**: C_cons is NOT satisfied (terminate verification)
- **PROCEED**: C_cons appears satisfied or needs Round 2 verification

## OUTPUT REQUIREMENTS
Provide verdict with confidence score and validated anchor mappings for Round 2.
**IMPORTANT**: If Blue refuted causality and it's valid, you MUST set c_cons_satisfied=False and verdict='SAFE-Mismatch'.
"""

# ==============================================================================
# Round 2: C_cons Reinforcement + C_reach + C_def Prompts (With Tools)
# ==============================================================================

ROUND2_RED_PROMPT = """You are the **Red Agent** performing **Round 2: C_cons Reinforcement + C_reach Establishment**.
Your task is to reinforce C_cons (if contested in Round 1) and establish C_reach (attack path feasibility).
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
1. **Round 1 Results**: Validated Origin/Impact mappings and any contested points
2. **Reference Attack Path**: Step-by-step trace from the vulnerability analysis
3. **Target Code**: With line numbers
4. **Vulnerability Semantics**: Type, root cause, attack path

## YOUR TASK

### Task 1: Reinforce C_cons (if Round 1 had disputes)
If Blue contested your mappings in Round 1:
- Use tools to verify the mapped statements
- Provide additional evidence for semantic equivalence
- Show data flow connections between Origin and Impact

### Task 2: Establish C_reach (Core Task)
C_reach requires proving the attack path is **semantically consistent** with the reference:

**What to verify**:
1. **Path Completeness**: Every key step from Origin to Impact exists in target
2. **Semantic Consistency**: Each step performs the SAME operation as reference
3. **Data Flow Continuity**: Same variable flows through the entire path
4. **Control Flow Feasibility**: No unconditional blockers (guards, early returns)
5. **CAUSALITY**: Origin MUST causally precede Impact (check data/control flow, not just line numbers)

**For each attack path step, you must show**:
- Step ID and type (Origin, Trace, Impact, Branch, Call)
- Target line number and code
- Corresponding reference step
- Evidence of semantic match (e.g., both do allocation, both do free)

**CRITICAL - CAUSALITY REQUIREMENT**:
- **Origin → Impact MUST have causal relationship**:
  * Origin creates/modifies the vulnerable state
  * Impact uses/triggers that state
  * There must be a data/control flow path connecting them
- **Line numbers can be misleading** (macros, cross-function calls, callbacks)
- **But data flow MUST be valid**: The effect of Origin must reach Impact

### Attack Path Step Types
- **Origin**: Where vulnerable state is created (e.g., allocation, lock acquisition)
- **Trace**: Intermediate steps that transform/propagate the state
- **Branch**: Conditional that enables the vulnerable path
- **Call**: Function call that continues the chain (verify callee if needed)
- **Impact**: Where vulnerability triggers (e.g., use-after-free, null dereference)

## TOOLS TO USE
- `find_definition(symbol, file)` - Get callee implementation to verify internal behavior
- `trace_variable(file, line, var, direction)` - Track data flow forward/backward
- `get_guard_conditions(file, line)` - Check path guards
- `grep(pattern, file, mode)` - Find all uses of a variable

## OUTPUT REQUIREMENTS
1. **c_cons_reinforced**: Did you strengthen C_cons evidence?
2. **attack_path_steps**: List of AttackPathStep with full details
3. **path_matches_reference**: Does the path match reference semantically?
4. **data_flow_verified**: Is data flow correct?
5. **control_flow_feasible**: Is the path reachable?
6. **c_reach_satisfied**: Final C_reach verdict
7. **verdict**: 'VULNERABLE' or 'NOT_VULNERABLE'
"""

ROUND2_BLUE_PROMPT = """You are the **Blue Agent** performing **Round 2: C_cons/C_reach Refutation + C_def Verification**.
Your task is to refute Red's claims OR find defenses that block the attack.
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
1. **Red's Round 2 Claims**: C_cons evidence, attack path steps, C_reach verdict
2. **Reference Fix**: What the patch does
3. **Target Code**: With line numbers

## YOUR TASK

### Mode 1: Refute C_cons/C_reach
If Red's claims are invalid:
- **Refute C_cons**: Show mapping is semantically wrong (different operation, different role)
- **Refute C_reach**: Show path blockers exist (guards, early returns, dead code)
- **Refute CAUSALITY**: Check if Origin → Impact lacks causal relationship
  * **CRITICAL CHECK**: Does Origin's effect actually reach Impact?
  * Use `trace_variable` to verify data flow from Origin to Impact
  * Check execution order: Can Origin execute before Impact in any feasible path?
  * Common causality error patterns:
    - **Reversed order**: Effect happens after cause (e.g., allocation after use)
    - **Blocked path**: Control flow prevents reaching Impact (e.g., exception/return between them)
    - **Different variables**: Origin affects X, Impact uses Y (no connection)
    - **Cleanup confusion**: Origin is cleanup code, Impact is also cleanup (not vulnerable use)
  * If Origin and Impact have NO causal relationship → REFUTE and explain why

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
2. **DOMINATES**: Defense executes BEFORE the Impact anchor
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
      "dominates_impact": true,
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

ROUND2_JUDGE_PROMPT = """You are the **Judge** adjudicating **Round 2: C_cons, C_reach, C_def, and Location Check**.
Your task is to evaluate all constraints and render a verdict.

## INPUTS PROVIDED
1. **Red's Round 2 Output**: C_cons reinforcement, attack path steps, C_reach verdict
2. **Blue's Round 2 Output**: Refutations and defense check report
3. **Target Code**: For verification
4. **Target Function Name**: The specific function being analyzed

## YOUR TASK: Adjudicate All Constraints

### 1. Evaluate C_cons
- Is Red's Origin→Impact mapping semantically correct?
- Did Blue successfully refute the mapping?
- **Verdict**: C_cons satisfied (True) or not (False)

### 2. Evaluate C_reach
- Is Red's attack path complete and semantically consistent?
- Did Blue find path blockers?
- Are all attack path steps valid?
- **Verdict**: C_reach satisfied (True) or not (False)

### 3. Evaluate C_def
Review Blue's defense check report:
- Did Blue check ALL four defense types?
- For each CHECKED defense, verify:
  - Defense code is correctly quoted
  - Defense dominates Impact (executes before)
  - Defense actually blocks the attack
- **Verdict**: C_def satisfied (True = defense exists) or not (False)

### 4. Evaluate Anchor Location (CRITICAL - NEW)
Check if the anchors are within the current target function:
- **origin_in_function**: Is the Origin anchor's func_name the same as the target function?
- **impact_in_function**: Is the Impact anchor's func_name the same as the target function?
- At least ONE must be TRUE for the vulnerability to be relevant
- If BOTH are FALSE (anchors only exist in external helper functions) → SAFE-OutOfScope

## VERDICT LOGIC

```
if not C_cons:
    return "SAFE-Mismatch"
elif not C_reach:
    return "SAFE-Unreachable"
elif not attack_path_matches_vuln_type:
    return "SAFE-TypeMismatch"
elif not origin_in_function and not impact_in_function:
    return "SAFE-OutOfScope"  # Neither anchor is in the current function
elif C_def:
    return "SAFE-Blocked"
else:
    if any contested points remain:
        return "PROCEED"  # to Round 3
    else:
        return "VULNERABLE"
```

## CONTESTED POINTS (for Round 3)
If neither side provided conclusive evidence:
- List specific points that need resolution
- These will be the focus of Round 3

## OUTPUT REQUIREMENTS
1. **c_cons_satisfied**: Boolean
2. **c_reach_satisfied**: Boolean
3. **c_def_satisfied**: Boolean
4. **attack_path_valid**: Is Red's attack path valid?
5. **origin_in_function**: Is Origin anchor in the current target function? (Check func_name)
6. **impact_in_function**: Is Impact anchor in the current target function? (Check func_name)
7. **validated_defense**: The strongest valid defense (if any)
8. **verdict**: 'SAFE-Mismatch', 'SAFE-Unreachable', 'SAFE-TypeMismatch', 'SAFE-OutOfScope', 'SAFE-Blocked', 'VULNERABLE', or 'PROCEED'
9. **contested_points**: List of unresolved issues (for Round 3)
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
2. Find the **Origin** point in target code: where vulnerable state is created
3. Find the **Impact** point in target code: where vulnerability triggers
4. Explain how the attack works step by step
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
- origin_line: line number where vulnerable state is created
- origin_code: the code at that line
- impact_line: line number where vulnerability triggers
- impact_code: the code at that line
- attack_reasoning: step-by-step explanation of the attack OR response to Blue's refutation

## CONCEDE RULES
- If Blue has found a valid defense that you cannot counter, set concedes=true
- If Blue has shown your origin/impact mapping is fundamentally wrong, set concedes=true
- Otherwise, continue arguing your case

Be aggressive in finding vulnerabilities - that's your job!
"""

BASELINE_BLUE_PROMPT = """You are the **Blue Agent** in a simplified vulnerability debate.
Your goal is to PROVE that the vulnerability does NOT exist or is BLOCKED in the target code.

## YOUR TASK
1. Review Red's attack claim (origin, impact, reasoning)
2. Find defenses: NULL checks, bounds validation, lock protection, etc.
3. Find blockers: early returns, error handling that prevents the attack path
4. Refute Red's claim if the mapping is incorrect

## INPUTS PROVIDED
- Red's Attack Claim: origin, impact, and attack reasoning
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
2. Evaluate Red's attack claim: Is the origin/impact mapping correct?
3. Evaluate Blue's defense claim: Is the defense valid and does it block the attack?
4. Consider if either side conceded
5. Make final verdict: VULNERABLE or SAFE

## DECISION LOGIC
- If Red conceded → SAFE (Red admits no vulnerability)
- If Blue conceded → VULNERABLE (Blue admits no defense)
- If Red's mapping is incorrect (wrong origin/impact) → SAFE
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
- Origin: Line {prev_red.origin_line}: `{prev_red.origin_code}`
- Impact: Line {prev_red.impact_line}: `{prev_red.impact_code}`
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
- **Attack Path**: {feature.semantics.attack_path}
- **Vulnerability Type**: {feature.semantics.vuln_type.value if feature.semantics.vuln_type else 'Unknown'}

### Reference Pre-Patch Code (Vulnerable Pattern)
{sf.s_pre if sf else 'Not available'}

### Target Code to Analyze
File: {candidate.target_file}
Function: {candidate.target_func.split(':')[-1]}

{target_code}
{history_section}
**YOUR TASK**: Find the Origin (where vulnerable state is created) and Impact (where vulnerability triggers) in the target code. Use tools if needed to verify your findings.
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
        origin_line=None,
        origin_code=None,
        impact_line=None,
        impact_code=None,
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
- Origin: Line {prev_red.origin_line}: `{prev_red.origin_code}`
- Impact: Line {prev_red.impact_line}: `{prev_red.impact_code}`
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
- Origin: Line {red_output.origin_line}: `{red_output.origin_code}`
- Impact: Line {red_output.impact_line}: `{red_output.impact_code}`
- Attack Reasoning: {red_output.attack_reasoning}
"""
    
    user_input = f"""
{history_section}
{red_claim}

### Vulnerability Logic
- **Root Cause**: {feature.semantics.root_cause}
- **Fix Mechanism**: {feature.semantics.fix_mechanism}

### Reference Post-Patch Code (Shows the Fix)
{sf.s_post if sf else 'Not available'}

### Target Code to Analyze
File: {candidate.target_file}
Function: {candidate.target_func.split(':')[-1]}

{target_code}

**YOUR TASK**: Find defenses or refute Red's claim. Use tools to check if:
1. The target has the same fix as reference
2. The target has an equivalent defense
3. Red's origin/impact mapping is incorrect

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
- Origin: Line {red_output.origin_line}: `{red_output.origin_code}`
- Impact: Line {red_output.impact_line}: `{red_output.impact_code}`
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
- **Attack Path**: {feature.semantics.attack_path}
- **Fix Mechanism**: {feature.semantics.fix_mechanism}

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
        This helps Red Agent directly use the pre-computed origin/impact anchors
        instead of re-inferring them from scratch.
        """
        ev = candidate.evidence
        hints = []
        
        # === Section 1: Anchor Mapping Hints (NEW) ===
        # Use Phase 2's pre_origins/pre_impacts and Phase 3's alignment to provide
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
        Generate Anchor Mapping Hints by matching Phase 2's anchors with Phase 3's alignment.
        
        This bridges the gap between:
        - Phase 2 output: pre_origins/pre_impacts (reference anchor lines with [line_no] markers)
        - Phase 3 output: aligned_vuln_traces (mapping between slice lines and target lines)
        
        Returns a formatted string with origin/impact anchor mappings for Red Agent.
        """
        import re
        
        def extract_line_no(line: str) -> int:
            """Extract line number from lines like '[ 227] code...'"""
            match = re.match(r'^\[\s*(\d+)\]', line.strip())
            return int(match.group(1)) if match else -1
        
        def find_target_mapping(anchor_line: str, aligned_traces: list) -> tuple:
            """
            Find the target line that maps to a given anchor line.
            
            Returns: (target_line_no, target_code, similarity) or (None, None, 0.0)
            """
            anchor_ln = extract_line_no(anchor_line)
            if anchor_ln == -1:
                # Try content-based matching if no line number
                anchor_content = re.sub(r'^\[\s*\d+\]', '', anchor_line).strip()
                for trace in aligned_traces:
                    if trace.target_line and anchor_content in trace.slice_line:
                        return (trace.line_no, trace.target_line, trace.similarity)
                return (None, None, 0.0)
            
            # Line number based matching
            for trace in aligned_traces:
                slice_ln = extract_line_no(trace.slice_line)
                if slice_ln == anchor_ln and trace.target_line:
                    return (trace.line_no, trace.target_line, trace.similarity)
            
            return (None, None, 0.0)
        
        origin_mappings = []
        impact_mappings = []
        
        # Process Origin Anchors
        for anchor in (sf.pre_origins or []):
            target_ln, target_code, sim = find_target_mapping(anchor, ev.aligned_vuln_traces)
            if target_ln and sim > 0.3:  # Only include meaningful matches
                origin_mappings.append({
                    'reference': anchor.strip(),
                    'target_line': target_ln,
                    'target_code': target_code.strip() if target_code else '',
                    'similarity': sim
                })
        
        # Process Impact Anchors
        for anchor in (sf.pre_impacts or []):
            target_ln, target_code, sim = find_target_mapping(anchor, ev.aligned_vuln_traces)
            if target_ln and sim > 0.3:
                impact_mappings.append({
                    'reference': anchor.strip(),
                    'target_line': target_ln,
                    'target_code': target_code.strip() if target_code else '',
                    'similarity': sim
                })
        
        # Format output
        lines = []
        
        if origin_mappings or impact_mappings:
            lines.append("1. **Anchor Mapping Hints** (from Phase 2/3 Analysis):")
            lines.append("   These are the pre-computed anchor points. Use them as starting points for your mapping.")
            
            if origin_mappings:
                lines.append("   ")
                lines.append("   **Origin Anchors** (where vulnerable state is created):")
                for m in origin_mappings:
                    lines.append(f"   - Reference: `{m['reference']}`")
                    lines.append(f"     → Target Line {m['target_line']}: `{m['target_code']}` (sim: {m['similarity']:.2f})")
            
            if impact_mappings:
                lines.append("   ")
                lines.append("   **Impact Anchors** (where vulnerability is triggered):")
                for m in impact_mappings:
                    lines.append(f"   - Reference: `{m['reference']}`")
                    lines.append(f"     → Target Line {m['target_line']}: `{m['target_code']}` (sim: {m['similarity']:.2f})")
            
            # Add unmapped anchors as hints
            unmapped_origins = [a for a in (sf.pre_origins or [])
                               if not any(m['reference'] == a.strip() for m in origin_mappings)]
            unmapped_impacts = [a for a in (sf.pre_impacts or [])
                               if not any(m['reference'] == a.strip() for m in impact_mappings)]
            
            if unmapped_origins or unmapped_impacts:
                lines.append("   ")
                lines.append("   **Unmapped Anchors** (need manual verification):")
                for a in unmapped_origins[:3]:
                    lines.append(f"   - Origin (unmapped): `{a.strip()}`")
                for a in unmapped_impacts[:3]:
                    lines.append(f"   - Impact (unmapped): `{a.strip()}`")
        
        return "\n".join(lines) if lines else ""


def extract_involved_functions(judge_res=None, origin_anchor=None, impact_anchor=None, trace=None, defense_mechanism=None) -> List[str]:
    """
    Extract list of involved functions from evidence chain.
    
    Args:
        judge_res: JudgeOutput object (if provided, all info will be extracted from it)
        origin_anchor: Origin StepAnalysis (optional, if judge_res is not provided)
        impact_anchor: Impact StepAnalysis (optional)
        trace: Trace List (optional)
        defense_mechanism: Defense StepAnalysis (optional)
    
    Returns:
        Sorted list of function names
    """
    involved_funcs = set()
    
    # If judge_res is provided, extract all info from it
    if judge_res:
        if judge_res.origin_anchor and judge_res.origin_anchor.func_name:
            involved_funcs.add(judge_res.origin_anchor.func_name)
        if judge_res.impact_anchor and judge_res.impact_anchor.func_name:
            involved_funcs.add(judge_res.impact_anchor.func_name)
        for step in (judge_res.trace or []):
            if step.func_name:
                involved_funcs.add(step.func_name)
        if judge_res.defense_mechanism and judge_res.defense_mechanism.func_name:
            involved_funcs.add(judge_res.defense_mechanism.func_name)
    else:
        # Extract from individually provided parameters
        if origin_anchor and origin_anchor.func_name:
            involved_funcs.add(origin_anchor.func_name)
        if impact_anchor and impact_anchor.func_name:
            involved_funcs.add(impact_anchor.func_name)
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
                print(f"    [Skip-Adaptive] Rank {candidate.rank} (Conf {candidate.confidence:.2f}) < Threshold {current_threshold:.2f}")
                continue

            # [Modified] Use file + func as unique key to prevent same-named functions in different files from being skipped by mistake
            # [DEBUG]
            # if candidate.target_func != 'ivr_read_header':
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
            [Attack Path]: {feature.semantics.attack_path}
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
            [Fix Mechanism]: {feature.semantics.fix_mechanism}
            [Reference Fix (Slice)]:
            {feature.slices[candidate.patch_func].s_post}
            [Reference Function (Full Post-Patch)]:
            {ref_new_code}
            [All Patch Diffs]:
            {combined_diffs}
            """
            # ============== 2-Round Debate System ==============
            # Round 1: C_cons validation (no tools) - fast screening
            # Round 2: C_reach + C_def verification (with tools) - thorough analysis
            
            # === Round 1: C_cons Validation (No Tools) ===
            # Returns all three agent outputs for evidence chain construction
            round1_red, round1_blue, round1_judge, should_continue = run_round1_debate(
                feature, candidate, target_context_with_lines, llm
            )
            
            # Round 1 Early Exit: C_cons not satisfied
            if not should_continue:
                print(f"    [Round1-Exit] C_cons failed: {round1_judge.verdict}")
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
                    analysis_report=f"[Round1 Exit] C_cons not satisfied. Verdict: {round1_judge.verdict}",
                    is_vulnerable=False,
                    verdict_category=round1_judge.verdict,
                    involved_functions=[],
                    peer_functions=peer_names_list,
                    origin=None,
                    impact=None,
                    trace=[],
                    defense_step=None,
                    defense_status=f"C_cons={round1_judge.c_cons_satisfied}"
                )
                findings.append(finding)
                
                # [Adaptive Threshold] Update - Early exit also counts as SAFE verdict
                current_threshold += THRESHOLD_PENALTY_STEP
                print(f"    [Adaptive] Verdict SAFE (Round1 Exit) -> Raising threshold to {current_threshold:.2f}")
                
                continue  # Skip to next candidate
            
            # === Round 2: C_reach + C_def Verification (With Tools) ===
            # Returns all three agent outputs for evidence chain construction
            round2_red, round2_blue, round2_judge, round2_verdict = run_round2_debate(
                round1_judge, feature, candidate, target_context_with_lines, tools, llm
            )
            
            # === Round 3: Final Judge Integration (No Tools) ===
            # Round 3 synthesizes ALL Round 1 and Round 2 agent outputs into a proper JudgeOutput
            # with complete evidence chain (origin_anchor, impact_anchor, trace, defense_mechanism)
            # - Round 1 Red: origin/impact mappings
            # - Round 2 Red: attack_path_steps (for trace)
            # - Round 2 Blue: defense_report (for defense_mechanism)
            judge_res = run_round3_final_judge(
                round1_red, round1_blue, round1_judge,
                round2_red, round2_blue, round2_judge,
                feature, candidate, target_context_with_lines, llm
            )
        
            # Map new JudgeOutput fields to VulnerabilityFinding (per Methodology.tex Section 4.4)
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
                verdict_category=judge_res.verdict_category,  # VULNERABLE/SAFE-Blocked/SAFE-Mismatch/SAFE-Unreachable
                involved_functions=extract_involved_functions(judge_res),
                peer_functions=peer_names_list,
                # Map constraint-based evidence (per Methodology.tex Evidence Schema)
                origin=judge_res.origin_anchor,  # Origin anchor: where vulnerable state is created
                impact=judge_res.impact_anchor,    # Impact anchor: where vulnerability is triggered
                trace=judge_res.trace,
                defense_step=judge_res.defense_mechanism,
                defense_status=f"C_cons={judge_res.c_cons_satisfied}, C_reach={judge_res.c_reach_satisfied}, C_def={judge_res.c_def_satisfied}"
            )
            findings.append(finding)
        
            # Log constraint outcomes
            print(f"    [Judge] Target: {candidate.target_file}::{candidate.target_func}")
            print(f"    [Judge] Verdict: {judge_res.verdict_category}")
            print(f"    [Judge] Constraints: C_cons={judge_res.c_cons_satisfied}, C_reach={judge_res.c_reach_satisfied}, C_def={judge_res.c_def_satisfied}")

                # [Adaptive Threshold] Update
            if not judge_res.is_vulnerable:
                current_threshold += THRESHOLD_PENALTY_STEP
                print(f"    [Adaptive] Verdict SAFE -> Raising threshold to {current_threshold:.2f}")
        
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
                origin=None,
                impact=None,
                trace=[],
                defense_step=None,
                defense_status=f"Error: {str(e)}"
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
[Attack Path]: {feature.semantics.attack_path}
[Fix Mechanism]: {feature.semantics.fix_mechanism}
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
            # - Red: Argues vulnerability EXISTS, finds origin/impact
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
                origin=None,
                impact=None,
                trace=[],
                defense_step=None,
                defense_status="Baseline: No tool analysis"
            )
            findings.append(finding)
            if not res.is_vulnerable:
                current_threshold += THRESHOLD_PENALTY_STEP
                print(f"    [Adaptive] Verdict SAFE -> Raising threshold to {current_threshold:.2f}")
                
        except Exception as e:
            print(f"    [Baseline] Error for {candidate.target_func}: {e}")
            import traceback
            traceback.print_exc()
            continue

    return {"final_findings": findings}