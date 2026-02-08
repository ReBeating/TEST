"""
Anchor-Guided Analysis Module

Implemented based on Methodology §3.3.2 (Anchor-Guided Signature Extraction)
Uses LLM Agent + CodeNavigator toolchain to identify vulnerability anchors

Core Concepts:
- Origin Anchors: Operations that create the vulnerable state (e.g., Alloc, Def, Lock)
- Impact Anchors: Operations that trigger the vulnerability (e.g., Use, Deref, Double Unlock)

Discovery Process:
1. Start from modified lines (search hints)
2. Expand via data flow/control flow dependencies
3. Type-based search based on vulnerability type anchor roles
4. Verify Origin → Impact connectivity

Design Principles:
- Anchors must be within the currently analyzed function (for slice extraction)
- Cross-function information serves as supplementary metadata (for semantic report generation)
"""

import os
import json
import time
from typing import List, Set, Dict, Optional, Any
from enum import Enum
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from core.navigator import CodeNavigator
from core.models import AnchorRole, TaxonomyFeature
from extraction.taxonomy import get_anchor_spec


# ==============================================================================
# Connection Retry Utility Functions
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


# ==============================================================================
# Data Models
# ==============================================================================

class AnchorScope(str, Enum):
    """Scope type of the Anchor"""
    LOCAL = "local"                      # Purely intra-procedural operations
    CALL_SITE = "call_site"              # Call site (actual operation is in callee)
    INTER_PROCEDURAL = "inter_procedural" # Cross-function data flow

class CrossFunctionInfo(BaseModel):
    """Cross-function information (supplementary metadata for semantic reports)"""
    callee_file: Optional[str] = Field(
        default=None,
        description="Path to the file containing the callee"
    )
    callee_function: Optional[str] = Field(
        default=None,
        description="Name of the callee function"
    )
    callee_line: Optional[int] = Field(
        default=None,
        description="Line number of the actual operation in the callee"
    )
    callee_content: Optional[str] = Field(
        default=None,
        description="Code content of the actual operation in the callee"
    )
    callee_role: Optional[AnchorRole] = Field(
        default=None,
        description="Role of the actual operation in the callee"
    )
    data_flow_chain: Optional[List[str]] = Field(
        default=None,
        description="Data flow trace chain (for inter-procedural cases)"
    )

class AnchorItem(BaseModel):
    """
    Data model for a single anchor (Expanded)
    
    Design Principles:
    - Core fields (file_path, function_name, line, content) are used for slice extraction
    - These fields must point to the CURRENT function being analyzed
    - cross_function_info serves only as supplementary metadata for semantic reports
    """
    # ========== Core Location Info (for Slice Extraction) ==========
    file_path: str = Field(description="Full path of the file containing the Anchor")
    function_name: str = Field(description="Name of the function containing the Anchor")
    line: int = Field(description="Absolute line number of the Anchor (within current function)")
    content: str = Field(description="Code content of the Anchor (within current function)")
    
    # ========== Semantic Info ==========
    role: AnchorRole = Field(description="Semantic role of the anchor (from taxonomy definition)")
    reasoning: str = Field(description="Why this is an anchor (Agent reasoning)")
    
    # ========== Scope Info ==========
    scope: AnchorScope = Field(
        default=AnchorScope.LOCAL,
        description="Scope type of the Anchor"
    )
    
    # ========== Cross-Function Supplement (For Semantic Reports Only) ==========
    cross_function_info: Optional[CrossFunctionInfo] = Field(
        default=None,
        description="Cross-function info (when scope is CALL_SITE or INTER_PROCEDURAL)"
    )


class AnchorResult(BaseModel):
    """Anchor identification result"""
    origin_anchors: List[AnchorItem] = Field(
        description="Origin anchors (operations creating the vulnerable state)",
        default_factory=list
    )
    impact_anchors: List[AnchorItem] = Field(
        description="Impact anchors (operations triggering the vulnerability)",
        default_factory=list
    )
    reasoning: str = Field(
        description="Overall reasoning: how anchors were found from modified lines and how they form a vulnerability chain"
    )


# ==============================================================================
# Anchor Analyzer
# ==============================================================================

class AnchorAnalyzer:
    """
    Agent-driven Anchor Analyzer
    
    Implements Anchor Discovery process based on Methodology §3.3.2:
    1. Extract search hints from diff (modified lines + key variables)
    2. Get expected anchor roles based on vulnerability type
    3. Use Agent + tools to expand search from hints to find anchors
    4. Verify Origin → Impact connectivity
    """
    
    def __init__(self, navigator: CodeNavigator):
        self.navigator = navigator
        self.llm = ChatOpenAI(
            base_url=os.getenv("API_BASE"),
            api_key=os.getenv("API_KEY"),
            model=os.getenv("MODEL_NAME", "gpt-4o"),
            temperature=0
        )
    
    def identify(self,
                 code_content: str,
                 diff_text: str,
                 search_hints: Dict[str, Any],
                 taxonomy: TaxonomyFeature,
                 file_path: str,
                 function_name: str,
                 start_line: int = 1,
                 attempt: int = 1) -> AnchorResult:
        """
        Identify anchors (Agent-driven)
        
        Args:
            code_content: Complete function code (Pre-Patch Version)
            diff_text: Patch diff
            search_hints: Search hints extracted from extract_search_hints()
                - deleted_lines: List[ModifiedLine]
                - added_lines: List[ModifiedLine]
                - key_variables: Set[str]
            taxonomy: Vulnerability type and assumptions
            file_path: Target file path
            function_name: Name of the function being analyzed
            start_line: Starting line number of the code
            attempt: Current attempt count (for Refinement, default 1)
            
        Returns:
            AnchorResult: Identified Origin and Impact anchors
        """
        
        # 1. Get anchor specification (based on vulnerability type)
        anchor_spec = get_anchor_spec(taxonomy.vuln_type)
        origin_roles = anchor_spec['origin_roles']
        impact_roles = anchor_spec['impact_roles']
        vuln_chain = anchor_spec['vulnerability_chain']
        
        # 2. Format search hints
        deleted_lines_text = "\n".join([
            f"  Line {dl.line_number}: {dl.content}"
            for dl in search_hints.get('deleted_lines', [])
        ]) or "None"
        
        added_lines_text = "\n".join([
            f"  Line {dl.line_number}: {dl.content}"
            for dl in search_hints.get('added_lines', [])
        ]) or "None"
        
        key_vars = ", ".join(search_hints.get('key_variables', [])) or "None"
        
        # 3. Define tools
        tools = self._create_tools(file_path)
        llm_with_tools = self.llm.bind_tools(tools)
        
        # 4. Format code (with line numbers)
        lines = code_content.splitlines()
        formatted_code_lines = []
        for i, line in enumerate(lines):
            formatted_code_lines.append(f"[{start_line + i:4d}] {line}")
        formatted_code = "\n".join(formatted_code_lines)
        
        # 5. Build System Prompt (General guidance, no specific code or data)
        origin_roles_str = ", ".join([r.value for r in origin_roles]) if origin_roles else "N/A (Generic)"
        impact_roles_str = ", ".join([r.value for r in impact_roles]) if impact_roles else "N/A (Generic)"
        
        # Extract line ranges actually affected by the patch
        patch_affected_lines = self._extract_patch_affected_lines(diff_text, start_line)
        affected_lines_str = ", ".join([str(ln) for ln in sorted(patch_affected_lines)]) if patch_affected_lines else "None"
        
        # Add attempt context
        attempt_context = ""
        if attempt > 1:
            attempt_context = f"""
### ⚠️ IMPORTANT: This is Attempt {attempt}/3
The previous attempt failed validation (anchors were not coherent or incomplete).
**What went wrong**: Either the Origin→Impact connection was broken, or required anchor roles were missing.
**Your task now**:
- Try to find ALTERNATIVE anchor candidates (different lines with same roles)
- Expand search scope (check more variables, deeper call chains, broader context)
- Reconsider the vulnerability pattern - the actual mechanism might differ from the hypothesis
"""
        
        system_prompt = f"""You are an Elite **Vulnerability Anchor Discovery Agent** specializing in {taxonomy.vuln_type.value} vulnerabilities.
{attempt_context}

### Your Mission
Identify **Anchors** - critical operations that embody the vulnerability mechanism:
- **Origin Anchors**: Operations that CREATE the vulnerable state (e.g., {origin_roles_str})
- **Impact Anchors**: Operations that TRIGGER/EXPLOIT the vulnerability (e.g., {impact_roles_str})

### Expert Knowledge for {taxonomy.vuln_type.value}
**Typical Vulnerability Pattern**:
{vuln_chain}

**Expected Anchor Roles**:
- Origin Roles: {origin_roles_str}
- Impact Roles: {impact_roles_str}

### Your Toolkit
You have access to `CodeNavigator` tools to explore code (including cross-file references):
- `grep(pattern, file_path, mode="word"|"regex"|"def_use", scope_start=None, scope_end=None)`: Search for patterns in a specific file (file_path REQUIRED)
- `trace_variable(line, var_name, file_path, direction="backward"|"forward")`: Track data flow (file_path REQUIRED)
- `get_guard_conditions(line, file_path)`: Find control flow guards (file_path REQUIRED)
- `get_next_control(line, file_path)`: Analyze control flow outcomes (file_path REQUIRED)
- `find_definition(symbol_name, file_path)`: Locate symbol definitions (file_path REQUIRED as context)
- `get_callers(symbol_name)`: Find function call sites across all files (no file_path needed)
- `read_file(start, end, file_path)`: Read code from any file (file_path REQUIRED)

**Important**:
- **file_path is REQUIRED** for most tools (use the current file path from task context)
- **Don't repeat failed searches** - if grep returns empty results, the pattern doesn't exist in that file
- **Tool results are cached** - you can refer to previous results without re-calling

### Discovery Strategy
1. **Start from Search Hints**: The deleted/added lines and key variables are your initial anchors
2. **Expand via Data/Control Flow**: Use `trace_variable()` and `get_guard_conditions()` to find related code
3. **Type-Specific Search**: Look for operations matching the expected anchor roles

### 🔴 CRITICAL CONSTRAINT: Patch Coverage Scope
**This patch fixes a specific vulnerability instance. Your task: identify the EXACT vulnerability that this patch addresses.**

**Key Principle**: The patch may modify function A to protect against a vulnerability triggered in function B.
- ✅ Include: Operations protected by this patch (even if not directly modified)
- ✅ Include: Call sites or data flow paths leading to the vulnerability
- ❌ Exclude: Similar code patterns that this patch does NOT protect

**Example 1 (Direct Protection)**:
- Patch adds `if (!ptr) return;` before `use(ptr);`
- ✅ Impact Anchor: `use(ptr);` (protected by the check)
- ❌ NOT Impact: `use(other_ptr);` elsewhere (different variable, not protected)

**Example 2 (Indirect Protection via Lifecycle)**:
- Patch moves `wq = alloc_workqueue()` from `open()` to `probe()`
- Root cause: `resume()` uses `wq` before `open()` is called
- ✅ Origin Anchor: `alloc_workqueue()` call site in probe (the fix location)
- ✅ Impact Anchor: Use the call site or data flow in probe that leads to resume's usage
- Note: Even though resume() is not modified, it's the protected vulnerability site

**How to handle cross-function Impact**:
1. If current function calls the vulnerable function → use call site as anchor
2. If indirect relationship (callbacks, shared data) → use the connection point + cross_function_info

### Inter-Procedural Anchor Rules
**CRITICAL**: Anchors MUST be in the current function being analyzed. When the vulnerability spans multiple functions:

**Case 1: Callee - Vulnerability operations in called function** (Current function is Caller, vulnerability is in Callee)
- Use the **call site** (in current function) as the anchor
- Mark scope as "call_site"
- Record the actual operation in cross_function_info
- Example (UAF via helper):
  ```
  // Current function calls helper, helper frees ptr inside
  {{
    "line": 51,
    "content": "helper(ptr);",
    "role": "FREE",  # Semantic role based on what helper does
    "scope": "call_site",
    "reasoning": "Calls helper() which frees ptr inside",
    "cross_function_info": {{
      "callee_function": "helper",
      "callee_line": 80,
      "callee_content": "free(ptr);",
      "callee_role": "FREE"
    }}
  }}
  ```

**Case 2: Caller - Bad input from calling function** (Current function is Callee, Origin is in Caller)
- Use the **parameter reception/use point** in current function as Impact anchor
- Mark scope as "inter_procedural"
- Record the caller's bad input in cross_function_info
- Example (NULL pointer from caller):
  ```
  // Caller passes NULL, current function uses without checking
  {{
    "line": 203,
    "content": "struct data *info = dev->platform_data;",  # First use of bad param
    "role": "USE",  # This is the Impact
    "scope": "inter_procedural",
    "reasoning": "Uses platform_data which caller may pass as NULL without validation",
    "cross_function_info": {{
      "callee_function": "probe_device",  # Caller that passes bad value
      "callee_line": 150,
      "callee_content": "driver_register(&dev_driver);  # Calls with NULL platform_data",
      "data_flow_chain": [
        "Line 150 (probe_device): Registers driver with dev.platform_data=NULL",
        "Line 203 (current): Directly uses dev->platform_data without NULL check"
      ]
    }}
  }}
  ```

**Case 3: Shared State - Resource lifecycle spans peer functions** (Case of CVE-2021-46994)
- Identify the **connection point** in current function (where resource is stored/accessed)
- Mark scope as "inter_procedural"
- Record the cross-function relationship in cross_function_info
- Example (Null Pointer Dereference via shared state):
  ```
  // In probe(): stores wq to shared struct, resume() may use before initialization
  {{
    "line": 1363,
    "content": "priv->wq = alloc_workqueue(...);",
    "role": "ASSIGN",  # Creates the resource (Origin)
    "scope": "inter_procedural",
    "reasoning": "Allocates workqueue stored in shared priv struct, used by resume() which may run before open()",
    "cross_function_info": {{
      "data_flow_chain": [
        "Line 1363 (probe): priv->wq = alloc_workqueue()",
        "priv struct shared across driver lifecycle",
        "resume() accesses priv->wq without checking if open() was called"
      ],
      "callee_function": "mcp251x_can_resume",  # Peer function that uses the resource
      "callee_line": 1486,
      "callee_content": "queue_work(priv->wq, &priv->restart_work);"
    }}
  }}
  ```

**Case 4: Callbacks/Async - Indirect trigger paths**
- Identify the **registration/setup point** in current function
- Mark scope as "inter_procedural"
- Explain the trigger path in cross_function_info
- Example (Race condition via callback):
  ```
  {{
    "line": 120,
    "content": "register_callback(dev, handler);",
    "role": "USE",  # Sets up vulnerable callback
    "scope": "inter_procedural",
    "reasoning": "Registers handler which may be called before initialization completes",
    "cross_function_info": {{
      "data_flow_chain": [
        "Line 120: register_callback()",
        "handler() can be triggered asynchronously",
        "handler() uses uninitialized data"
      ],
      "callee_function": "handler"
    }}
  }}
  ```

**Key Principle for Inter-Procedural Cases**:
- Focus on what THIS function does (Origin) and how it connects to vulnerabilities elsewhere (Impact via data flow/control flow)
- Don't try to mark lines in other functions as anchors - use cross_function_info instead
- When you find yourself repeatedly reading other functions, that's a signal to use cross_function_info

### Output Requirements
Return a JSON object with:
- `origin_anchors`: List of anchor objects with fields:
  - `line`: int (absolute line number in current function)
  - `content`: str (code at that line)
  - `role`: AnchorRole (semantic role from taxonomy)
  - `reasoning`: str (why this is an anchor)
  - `scope`: "local"|"call_site"|"inter_procedural" (default: "local")
  - `cross_function_info`: object (optional, for scope != "local")
- `impact_anchors`: Same structure as origin_anchors
- `reasoning`: Detailed trace explaining your discovery process and the vulnerability chain

### Critical Rules
1. **ALL anchors MUST be in the current function** (lines within the provided code range)
2. **Anchors MUST be related to what the patch fixes** - Don't include similar vulnerabilities that this patch doesn't touch
3. For operations in callees, use the call site + cross_function_info
4. Use tools to verify relationships - don't guess
5. Try to find required anchor roles (Origin: {origin_roles_str}, Impact: {impact_roles_str})
"""

        # 6. Build User Content (Specific Task Data)
        user_content = f"""### Analysis Task
**Target File**: {file_path}
**Target Function**: {function_name}
**Vulnerability Type**: {taxonomy.vuln_type.value}
**CWE**: {taxonomy.cwe_id}: {taxonomy.cwe_name if taxonomy.cwe_name else 'Unknown'}
**Code Version**: Pre-Patch (Vulnerable)
**Code Range**: Lines [{start_line}, {start_line + len(lines) - 1}]

### The Hypothesis (Your Lead)
- **Root Cause**: {taxonomy.root_cause}
- **Attack Path**: {taxonomy.attack_path}
- **Fix Mechanism**: {taxonomy.fix_mechanism}

### Search Hints (Starting Points)
These are your **initial anchors** - expand from here using tools:

**Deleted Lines (Pre-Patch only, likely vulnerability core)**:
{deleted_lines_text}

**Added Lines (Post-Patch, reveals fix intent)**:
{added_lines_text}

**Key Variables to Track**:
{key_vars}

### Patch Diff
```diff
{diff_text}
```

### Function Code (Pre-Patch/Vulnerable)
```c
{formatted_code}
```

**Task**: Please identify the Origin and Impact anchors using the search hints and tools.
"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content)
        ]
        
        # 7. Agent Execution Loop (with intelligent caching and repetition detection)
        max_steps = 10
        curr_step = 0
        tool_call_history = {}  # Track tool call frequency: sig -> count
        tool_result_cache = {}   # Cache results: sig -> result
        repetition_threshold = 1  # Force block if same tool call repeats more than 3 times
        consecutive_blocks = 0   # Number of consecutive blocks
        max_consecutive_blocks = 3  # Force terminate after max consecutive blocks
        
        # [New] Track external function reads (for checking cross-function exploration)
        external_reads = {}  # line_range -> count (tracking reads outside current function)
        function_start = start_line
        function_end = start_line + len(lines) - 1
        
        while curr_step < max_steps:
            # Wrap LLM call with retry mechanism
            try:
                response = retry_on_connection_error(
                    lambda: llm_with_tools.invoke(messages),
                    max_retries=3
                )
            except Exception as e:
                print(f"      [AnchorAnalyzer] LLM invocation failed after retries: {e}")
                # Return empty result on connection failure
                return AnchorResult(
                    origin_anchors=[],
                    impact_anchors=[],
                    reasoning=f"LLM invocation failed after retries: {e}"
                )
            
            messages.append(response)
            
            # [New] Output Agent's thought process (if any)
            if hasattr(response, 'content') and response.content and not response.tool_calls:
                reasoning_preview = response.content[:500] if len(response.content) > 500 else response.content
                print(f"      [AnchorAgent] 💭 Thinking: {reasoning_preview}")
                if len(response.content) > 500:
                    print(f"      [AnchorAgent]    ... (truncated, total {len(response.content)} chars)")
            
            if response.tool_calls:
                blocked_in_this_round = False
                
                for t in response.tool_calls:
                    # Create tool signature for tracking and caching
                    tool_sig = f"{t['name']}({json.dumps(t['args'], sort_keys=True)})"
                    tool_call_history[tool_sig] = tool_call_history.get(tool_sig, 0) + 1
                    call_count = tool_call_history[tool_sig]
                    
                    print(f"      [AnchorAgent] 🔧 Tool: {t['name']} args={t['args']}")
                    
                    # [New] Detection of external function reads (for smart hints)
                    if t['name'] == 'read_file':
                        read_start = t['args'].get('start', 0)
                        read_end = t['args'].get('end', 0)
                        # Check if read range is outside current function
                        if read_start > 0 and read_end > 0:
                            is_external = (read_end < function_start or read_start > function_end)
                            if is_external:
                                range_key = f"{read_start}-{read_end}"
                                external_reads[range_key] = external_reads.get(range_key, 0) + 1
                                
                                # If same external range is read multiple times, give hint
                                if external_reads[range_key] == 2:
                                    print(f"      [AnchorAgent] 💡 TIP: You're reading lines {read_start}-{read_end} (outside current function {function_start}-{function_end})")
                                    print(f"      [AnchorAgent]     This suggests a cross-function vulnerability. Consider using:")
                                    print(f"      [AnchorAgent]     - scope='inter_procedural' for anchors in current function that connect to this")
                                    print(f"      [AnchorAgent]     - cross_function_info to record what you found in that external function")
                    
                    # Force block excessive repetition
                    if call_count > repetition_threshold:
                        blocked_in_this_round = True
                        consecutive_blocks += 1
                        
                        tool_result = f"""❌ CALL BLOCKED - Excessive Repetition (call #{call_count}, consecutive blocks: {consecutive_blocks})

This exact tool call has been made {call_count} times. The result will NOT change.

🔴 CRITICAL: You are stuck in a loop!

You MUST do ONE of the following NOW:
1. 📝 Provide your FINAL ANSWER with the anchors you've identified so far
2. 🔄 Try a COMPLETELY DIFFERENT tool or parameters
3. ⏭️  Move forward with the analysis based on existing information

DO NOT repeat the same tool call. This is blocking #{consecutive_blocks}/{max_consecutive_blocks}.

Blocked call: {t['name']}({t['args']})"""
                        print(f"      [AnchorAgent]   → ❌ BLOCKED (call #{call_count}, consecutive #{consecutive_blocks}/{max_consecutive_blocks})")
                        
                        # Max consecutive blocks reached, force terminate
                        if consecutive_blocks >= max_consecutive_blocks:
                            print(f"      [AnchorAgent]   → 🛑 FORCE STOP: Too many consecutive blocks ({consecutive_blocks}). Forcing final answer.")
                            tool_result += "\n\n🛑 SYSTEM: Maximum consecutive blocks reached. You MUST provide your final answer NOW."
                    
                    # Check cache
                    elif tool_sig in tool_result_cache:
                        tool_result = tool_result_cache[tool_sig]
                        consecutive_blocks = 0  # Reset consecutive block counter on cache hit
                        print(f"      [AnchorAgent]   → [CACHED] Returning previous result")
                        
                        # Progressive warning system
                        if call_count == 2:
                            tool_result = f"⚠️ WARNING: This is the 2nd call with identical parameters. Using cached result.\n\n{tool_result}"
                        elif call_count == 3:
                            tool_result = f"""🔴 CRITICAL: This is the 3rd call! You are repeating yourself.
The cached result below is IDENTICAL to previous calls. NEXT call will be BLOCKED.

If you need different information, use DIFFERENT parameters or a DIFFERENT tool.

{tool_result}"""
                    else:
                        # Execute tool for the first time
                        consecutive_blocks = 0  # Reset on successful tool execution
                        try:
                            selected_tool = next((tool for tool in tools if tool.name == t['name']), None)
                            if selected_tool:
                                tool_result = selected_tool.invoke(t['args'])
                                tool_result_cache[tool_sig] = tool_result  # Cache it
                                
                                # Print preview
                                result_preview = str(tool_result)[:200] + "..." if len(str(tool_result)) > 200 else str(tool_result)
                                print(f"      [AnchorAgent]   → ✓ {result_preview}")
                            else:
                                tool_result = f"Tool {t['name']} not found"
                                print(f"      [AnchorAgent]   → ✗ Tool not found")
                        except Exception as e:
                            tool_result = f"Error: {str(e)}"
                            print(f"      [AnchorAgent]   → ✗ Error: {e}")
                    
                    messages.append(ToolMessage(content=str(tool_result), tool_call_id=t['id']))
                
                curr_step += 1
                
                # [New] Force termination condition: Too many consecutive blocks
                if consecutive_blocks >= max_consecutive_blocks:
                    print(f"      [AnchorAgent] 🛑 FORCE TERMINATION: Agent stuck in loop after {consecutive_blocks} consecutive blocks")
                    break
            else:
                # No tool calls - Agent finished
                break
        
        print(f"      [AnchorAgent] Finished after {curr_step} steps, made {len(tool_result_cache)} unique tool calls")
        
        # 8. Extract Structured Output
        final_extractor = self.llm.with_structured_output(AnchorResult)
        
        try:
            result = retry_on_connection_error(
                lambda: final_extractor.invoke(messages),
                max_retries=3
            )
            
            # 9. Populate complete location info (file_path and function_name)
            for anchor in result.origin_anchors + result.impact_anchors:
                anchor.file_path = file_path
                anchor.function_name = function_name
            
            # 10. Role Completeness Check (OR relationship: as long as one role is found)
            found_origin_roles = set(a.role for a in result.origin_anchors)
            found_impact_roles = set(a.role for a in result.impact_anchors)
            
            expected_origin = set(origin_roles) if origin_roles else set()
            expected_impact = set(impact_roles) if impact_roles else set()
            
            # Check if at least one Origin role and one Impact role are found
            has_origin = bool(found_origin_roles & expected_origin) if expected_origin else bool(found_origin_roles)
            has_impact = bool(found_impact_roles & expected_impact) if expected_impact else bool(found_impact_roles)
            
            if not has_origin or not has_impact:
                warning_msg = "      [AnchorAnalyzer] Incomplete anchor roles:"
                if not has_origin:
                    if expected_origin:
                        warning_msg += f"\n        No Origin anchor found (expected any of: {[r.value for r in expected_origin]})"
                    else:
                        warning_msg += f"\n        No Origin anchor found"
                if not has_impact:
                    if expected_impact:
                        warning_msg += f"\n        No Impact anchor found (expected any of: {[r.value for r in expected_impact]})"
                    else:
                        warning_msg += f"\n        No Impact anchor found"
                print(warning_msg)
            
            # 11. Basic Validation (No anchors found at all)
            if not result.origin_anchors and not result.impact_anchors:
                print(f"      [AnchorAnalyzer] Warning: No anchors identified at all")
            
            return result
            
        except Exception as e:
            print(f"      [AnchorAnalyzer] Extraction Error: {e}")
            return AnchorResult(
                origin_anchors=[],
                impact_anchors=[],
                reasoning=f"Extraction failed: {e}"
            )
    
    def _create_tools(self, current_file_path: str):
        """Create toolset (supports cross-file reading)"""
        
        @tool
        def grep(pattern: str, file_path: str, mode: str = "word",
                scope_start: Optional[int] = None, scope_end: Optional[int] = None) -> str:
            """
            Search for a pattern in a specific file (required).
            
            Args:
                pattern: Search pattern (variable name, function name, or regex)
                file_path: Target file path (REQUIRED - use current file or specify another)
                mode: "word" (exact word match), "regex" (regex pattern), or "def_use" (PDG-enhanced)
                scope_start: Optional start line to limit search scope
                scope_end: Optional end line to limit search scope
            
            Returns:
                JSON string with search results
            
            Note: You MUST specify file_path explicitly. For current file, use the file_path from the task context.
            """
            try:
                scope = (scope_start, scope_end) if scope_start and scope_end else None
                result = self.navigator.grep(pattern, file_path, mode=mode, scope_range=scope)
                return json.dumps(result)
            except Exception as e:
                return json.dumps({"error": str(e), "results": [], "total_count": 0})
        
        @tool
        def trace_variable(line: int, var_name: str, file_path: str, direction: str = "backward") -> str:
            """
            Trace data flow for a variable.
            
            Args:
                line: Starting line number
                var_name: Variable name to trace
                file_path: Target file path (REQUIRED)
                direction: "backward" (where it comes from) or "forward" (where it goes)
            
            Returns:
                JSON string with trace results
            """
            try:
                result = self.navigator.trace_variable(file_path, line, var_name, direction=direction)
                return json.dumps(result)
            except Exception as e:
                return json.dumps({"error": str(e), "trace": []})
        
        @tool
        def get_guard_conditions(line: int, file_path: str) -> str:
            """
            Find conditional statements that guard (dominate) execution of a target line.
            
            Args:
                line: Target line number
                file_path: Target file path (REQUIRED)
            
            Returns:
                JSON string with list of guard conditions
            """
            try:
                result = self.navigator.get_guard_conditions(file_path, line)
                return json.dumps(result)
            except Exception as e:
                return json.dumps({"error": str(e), "guards": []})
        
        @tool
        def get_next_control(line: int, file_path: str) -> str:
            """
            Analyze control flow outcome after executing a line.
            
            Args:
                line: Starting line number
                file_path: Target file path (REQUIRED)
            
            Returns:
                JSON dict with {"type": "RETURN"|"GOTO"|"FALLTHROUGH", "statement": str}
            """
            try:
                result = self.navigator.get_next_control(file_path, line)
                return json.dumps(result)
            except Exception as e:
                return json.dumps({"error": str(e), "type": "UNKNOWN"})
        
        @tool
        def find_definition(symbol_name: str, file_path: str) -> str:
            """
            Find the definition of a symbol (function, struct, macro).
            
            Args:
                symbol_name: Name of the symbol
                file_path: Context file path for search (REQUIRED)
            
            Returns:
                JSON string with definition locations including file_path for each result
            """
            try:
                result = self.navigator.find_definition(symbol_name, context_path=file_path)
                return json.dumps(result[:3])  # Limit to top 3 results
            except Exception as e:
                return json.dumps({"error": str(e), "definitions": []})
        
        @tool
        def get_callers(symbol_name: str) -> str:
            """
            Find where a function is called (searches across all indexed files).
            
            Args:
                symbol_name: Function name
            
            Returns:
                JSON string with call sites including file_path for each result
            """
            try:
                result = self.navigator.get_callers(symbol_name)
                return json.dumps(result[:5])  # Limit to top 5
            except Exception as e:
                return json.dumps({"error": str(e), "callers": []})
        
        @tool
        def read_file(start: int, end: int, file_path: str) -> str:
            """
            Read specific lines from any file in the repository.
            Use this to read cross-file references (e.g., macro definitions in headers).
            
            Args:
                start: Start line number
                end: End line number
                file_path: Target file path (REQUIRED)
            
            Returns:
                Code content with line numbers
            """
            try:
                return self.navigator.read_code_window(file_path, start, end)
            except Exception as e:
                return f"Error reading {file_path}[{start}:{end}]: {e}"
        
        return [grep, trace_variable, get_guard_conditions, get_next_control,
                find_definition, get_callers, read_file]
    
    def _extract_patch_affected_lines(self, diff_text: str, start_line: int) -> Set[int]:
        """
        Extract line numbers affected by patch (used to constrain Impact Anchor range)
        
        Includes:
        1. Deleted lines (deleted lines) - vulnerability code itself
        2. Modified context lines (modified context) - code affected by the fix
        
        Args:
            diff_text: Patch diff text
            start_line: Code start line number
            
        Returns:
            Set of absolute line numbers affected by the patch
        """
        import re
        
        affected_lines = set()
        if not diff_text:
            return affected_lines
        
        hunk_re = re.compile(r'^@@\s*-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@')
        c_old = 0  # pre-patch line counter
        c_new = 0  # post-patch line counter
        
        # Track deleted line positions and their context
        deleted_positions = []
        added_positions = []
        
        for line in diff_text.splitlines():
            if line.startswith('---') or line.startswith('+++'):
                continue
            
            m = hunk_re.match(line)
            if m:
                c_old = int(m.group(1))
                c_new = int(m.group(3))
                continue
            
            # Deleted line: exists in pre-patch, removed in post-patch
            if line.startswith('-') and not line.startswith('---'):
                content = line[1:].strip()
                if content and not content.startswith(('//', '/*', '*', '#')):
                    deleted_positions.append(c_old)
                    affected_lines.add(c_old)
                c_old += 1
            
            # Added line: new in post-patch
            elif line.startswith('+') and not line.startswith('+++'):
                content = line[1:].strip()
                if content and not content.startswith(('//', '/*', '*', '#')):
                    added_positions.append(c_new)
                c_new += 1
            
            # Context line
            else:
                c_old += 1
                c_new += 1
        
        # Expand to include context around modifications (±2 lines)
        # This handles cases where the fix affects nearby lines
        expansion_radius = 2
        for pos in deleted_positions + added_positions:
            for offset in range(-expansion_radius, expansion_radius + 1):
                affected_lines.add(pos + offset)
        
        return affected_lines
