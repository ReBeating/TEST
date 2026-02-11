"""
Anchor-Guided Analysis Module (Paper §3.1.3)

Uses LLM Agent + CodeNavigator toolchain to identify typed vulnerability anchors.
Each anchor has a specific AnchorType from the vulnerability's constraint model (categories.py).

Core Concepts:
- Typed Anchors: Each anchor carries an AnchorType (e.g., alloc, dealloc, use, source, sink)
- Constraint Chain: Anchors form a dependency chain (e.g., alloc →d dealloc →t use)
- ViolationPredicate: Formal condition that makes the chain vulnerable

Discovery Process:
1. Start from modified lines (search hints)
2. Expand via data flow/control flow dependencies
3. Type-based search for each AnchorType in the constraint chain
4. Verify anchor connectivity matches the constraint chain

Design Principles:
- Anchors must be within the currently analyzed function (for slice extraction)
- Cross-function information serves as supplementary metadata (for semantic report generation)
"""

import os
import json
import time
from typing import List, Set, Dict, Optional, Any
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from core.navigator import CodeNavigator
from core.models import TaxonomyFeature
from core.categories import AnchorType, Anchor


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

# Note: AnchorItem has been unified into core.categories.Anchor (Pydantic BaseModel).
# AnchorScope enum removed — scope is now a plain str field on Anchor.
# CrossFunctionInfo removed — cross_function_info is now a plain dict field on Anchor.


class AnchorDecision(BaseModel):
    """Decision for a single anchor during deduplication."""
    line_number: int = Field(description="Anchor line number")
    anchor_type: str = Field(description="Anchor type (e.g., 'object', 'access')")
    keep: bool = Field(description="Whether to keep this anchor")
    reason: str = Field(description="Reason for the decision")


class AnchorDeduplicationResult(BaseModel):
    """Result of anchor deduplication by LLM."""
    decisions: List[AnchorDecision] = Field(
        description="Per-anchor keep/reject decisions"
    )
    reasoning: str = Field(
        description="Overall reasoning for the deduplication decisions"
    )


class AnchorResult(BaseModel):
    """
    Anchor identification result (paper §3.1.3 Algorithm 1 output).
    
    Uses a single `anchors` list of typed Anchor instances (from core.categories)
    instead of the old origin_anchors/impact_anchors split.
    Each anchor carries its own AnchorType.
    """
    anchors: List[Anchor] = Field(
        description="All typed anchors identified (each with type from categories.py)",
        default_factory=list
    )
    reasoning: str = Field(
        description="Overall reasoning: how anchors were found and how they form the vulnerability chain"
    )
    
    def get_by_type(self, anchor_type_value: str) -> List[Anchor]:
        """Get anchors of a specific AnchorType."""
        return [a for a in self.anchors if a.type.value == anchor_type_value]
    
    # === Phase 2 Compatibility Properties ===
    # search/verifier still reads origin_anchors/impact_anchors.
    # Derive from typed anchors: first half → origin, second half → impact.
    @property
    def origin_anchors(self) -> List[Anchor]:
        """Compat: first half of typed anchors chain → origin semantics."""
        if not self.anchors:
            return []
        mid = max(len(self.anchors) // 2, 1)
        return self.anchors[:mid]
    
    @property
    def impact_anchors(self) -> List[Anchor]:
        """Compat: second half of typed anchors chain → impact semantics."""
        if not self.anchors:
            return []
        mid = max(len(self.anchors) // 2, 1)
        return self.anchors[mid:]


# ==============================================================================
# Anchor Analyzer
# ==============================================================================

class AnchorAnalyzer:
    """
    Agent-driven Typed Anchor Analyzer (Paper §3.1.3)
    
    Implements Anchor Discovery using typed anchor model from categories.py:
    1. Extract search hints from diff (modified lines + key variables)
    2. Load expected AnchorTypes and constraint chain from VulnerabilityCategory
    3. Use Agent + tools to expand search from hints to find typed anchors
    4. Verify anchor chain connectivity matches the constraint model
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
                 attempt: int = 1,
                 candidates: str = "") -> AnchorResult:
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
            candidates: TwoPassSlice candidate text (from TwoPassSlicer.collect_candidates())
                Formatted as "L{line}: {code}  [def={defs}, use={uses}]" per line.
                Empty string if no candidates available.
            
        Returns:
            AnchorResult: Typed anchor instances matching the vulnerability chain
        """
        
        # 1. Get typed anchor specification from categories.py KB
        category = taxonomy.category_obj
        typed_anchors = category.anchors  # List[Anchor] with AnchorType
        constraint = category.constraint
        chain_str = constraint.chain_str()  # e.g., "alloc →d dealloc →t use"
        
        # Build anchor descriptions for LLM
        anchor_specs_text = "\n".join([
            f"  - **{a.type.value}** ({a.locatability.value}): {a.description}\n"
            f"    Identification: {a.type.identification_guide}"
            for a in typed_anchors
        ]) if typed_anchors else "No specific anchors defined — use generic analysis."
        
        # Build chain description
        chain_parts = [f"{a.description} ({a.type.value})" for a in typed_anchors]
        vuln_chain = (
            f"{category.description}\nChain: " + " → ".join(chain_parts)
            if chain_parts
            else category.description
        )
        
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
        
        # 5. Build System Prompt (typed anchor model)
        anchor_types_str = ", ".join([a.type.value for a in typed_anchors]) if typed_anchors else "N/A (Generic)"
        
        # Extract line ranges actually affected by the patch
        patch_affected_lines = self._extract_patch_affected_lines(diff_text, start_line)
        affected_lines_str = ", ".join([str(ln) for ln in sorted(patch_affected_lines)]) if patch_affected_lines else "None"
        
        # Add attempt context
        attempt_context = ""
        if attempt > 1:
            attempt_context = f"""
### ⚠️ IMPORTANT: This is Attempt {attempt}/3
The previous attempt failed validation (anchors were not coherent or chain was incomplete).
**What went wrong**: Either the anchor chain was broken, or required anchor types were missing.
**Your task now**:
- Try to find ALTERNATIVE anchor candidates (different lines with same types)
- Expand search scope (check more variables, deeper call chains, broader context)
- Reconsider the vulnerability pattern - the actual mechanism might differ from the hypothesis
"""
        
        system_prompt = f"""You are an Elite **Vulnerability Anchor Discovery Agent** specializing in {taxonomy.vuln_type.value} vulnerabilities.
{attempt_context}

### Your Mission
Identify **Typed Anchors** — critical operations that form the vulnerability chain.
Each anchor has a specific **type** from the vulnerability's constraint model.

### Vulnerability Chain Model for {taxonomy.vuln_type.value}
**Description**: {category.description}
**Anchor Chain**: {chain_str}
**Violation**: {constraint.violation.description}

**Required Anchor Types** (find one or more concrete code statements per type — multiple instances of the same type are allowed when the vulnerability involves multiple sources, sinks, or access points):
{anchor_specs_text}

**Dependency Chain**: The anchors must be connected as: {chain_str}

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

### 🔴 CRITICAL CONSTRAINT: Patch Relevance Test
**This patch fixes ONE specific vulnerability instance. Every anchor you report MUST pass the Patch Relevance Test below.**

**The Test — ask for EACH candidate anchor**:
> "If I remove this patch (revert to vulnerable code), does this specific code location become exploitable AS PART OF the same vulnerability chain?"

An anchor passes the test if **at least one** of these holds:
1. **Direct modification**: The anchor line is itself added/deleted/modified by the patch
2. **Data-flow connected**: A key variable in the anchor has a def-use or use-def chain that reaches a patch-modified line (trace with `trace_variable`)
3. **Control-flow protected**: The anchor is guarded by a condition that the patch adds/modifies (check with `get_guard_conditions`)
4. **Same branch/path**: The anchor is on the same execution path (same if/else branch, same switch case) as a patch-modified line

**What FAILS the test**:
- Code in a **different branch** (different `if`/`else`/`switch case`) that happens to use the same variable with the same pattern
- Code that is **structurally similar** but operates on **independent data** not touched by the patch
- Code that has the **same vulnerability pattern** but would need a **separate patch** to fix

**Example (Integer Underflow in two branches)**:
```c
if (dataset == 255) {{
    // Branch A — patch adds bounds check here
    while (len--)  // ✅ PASS: same branch as patch, data-flow connected
        WriteBlobByte(ofile, token[next++]);  // ✅ PASS: same branch
}} else {{
    // Branch B — patch does NOT touch this branch
    while (len--)  // ❌ FAIL: different branch, patch doesn't protect this
        WriteBlobByte(ofile, token[next++]);  // ❌ FAIL: same pattern but separate instance
}}
```
Even though Branch B has the identical vulnerability, it is a **separate vulnerability instance** that this patch does not address.

**How to verify**: Use `get_guard_conditions(line)` to check if the candidate anchor shares control-flow guards with patch-modified lines. If they are in different branches, the anchor fails the test.

**Cross-function anchors**:
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
   "line_number": 51,
   "code_snippet": "helper(ptr);",
   "type": "dealloc",
    "scope": "call_site",
    "reasoning": "Calls helper() which frees ptr inside",
    "cross_function_info": {{
      "callee_function": "helper",
      "callee_line": 80,
      "callee_content": "free(ptr);",
      "callee_anchor_type": "dealloc"
    }}
  }}
  ```

**Case 2: Caller - Bad input from calling function** (Current function is Callee, source is in Caller)
- Use the **parameter reception/use point** in current function as anchor
- Mark scope as "inter_procedural"
- Record the caller's bad input in cross_function_info
- Example (NULL pointer from caller):
  ```
  // Caller passes NULL, current function uses without checking
  {{
   "line_number": 203,
   "code_snippet": "struct data *info = dev->platform_data;",
   "type": "use",
    "scope": "inter_procedural",
    "reasoning": "Uses platform_data which caller may pass as NULL without validation",
    "cross_function_info": {{
      "callee_function": "probe_device",
      "callee_line": 150,
      "callee_content": "driver_register(&dev_driver);",
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
   "line_number": 1363,
   "code_snippet": "priv->wq = alloc_workqueue(...);",
   "type": "alloc",
    "scope": "inter_procedural",
    "reasoning": "Allocates workqueue stored in shared priv struct, used by resume() which may run before open()",
    "cross_function_info": {{
      "data_flow_chain": [
        "Line 1363 (probe): priv->wq = alloc_workqueue()",
        "priv struct shared across driver lifecycle",
        "resume() accesses priv->wq without checking if open() was called"
      ],
      "callee_function": "mcp251x_can_resume",
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
   "line_number": 120,
   "code_snippet": "register_callback(dev, handler);",
   "type": "use",
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
- Focus on what THIS function does and how it connects to vulnerabilities elsewhere (via data flow/control flow)
- Don't try to mark lines in other functions as anchors - use cross_function_info instead
- When you find yourself repeatedly reading other functions, that's a signal to use cross_function_info

### Output Requirements
Return a JSON object with:
- `anchors`: List of typed anchor objects, each with fields:
  - `line_number`: int (absolute line number in current function)
  - `code_snippet`: str (code at that line)
  - `type`: str (**MUST be one of: {anchor_types_str}** — any other type will be REJECTED)
  - `reasoning`: str (why this is an anchor)
  - `locatability`: "concrete"|"assumed"|"conceptual" (default: "concrete")
    - "concrete": Has specific code location with clear semantics (e.g., malloc, free, scanf)
    - "assumed": Has specific code location but semantics need assumption (e.g., function parameter assumed controllable)
    - "conceptual": No specific code location, purely inferred existence (e.g., UAF use in another function)
  - `assumption_type`: str (required when locatability != "concrete") — one of: "controllability"|"semantic"|"existence"|"reachability"
  - `assumption_rationale`: str (required when locatability != "concrete") — explain the assumption
  - `scope`: "local"|"call_site"|"inter_procedural" (default: "local")
  - `cross_function_info`: object (optional, for scope != "local")
- `reasoning`: Detailed trace explaining your discovery process and the vulnerability chain

### Critical Rules
1. **ALL anchors MUST be in the current function** (lines within the provided code range)
2. **Every anchor MUST pass the Patch Relevance Test** (see above) — verify with `get_guard_conditions()` and `trace_variable()` that each anchor is on the same execution path and data-flow chain as the patch. Do NOT include code in different branches that the patch does not touch.
3. For operations in callees, use the call site + cross_function_info
4. Use tools to verify relationships - don't guess
5. Try to find all required anchor types: {anchor_types_str}
6. The anchors should form the dependency chain: {chain_str}
7. **Multiple anchors of the same type are allowed** — but ONLY if each instance independently passes the Patch Relevance Test. E.g., two `source` anchors feeding the same patch-protected computation are fine; two `sink` anchors in different branches where only one branch is patched means only the patched branch's sink qualifies.
8. **STRICT TYPE CONSTRAINT**: The `type` field MUST be one of [{anchor_types_str}]. Do NOT use anchor types from other vulnerability categories (e.g., do NOT use "access" or "object" for Numeric-Domain vulnerabilities). If a code location doesn't fit any of the allowed types, describe it in the `reasoning` of the nearest matching anchor instead.
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
- **Attack Chain**: {taxonomy.attack_chain}
- **Patch Defense**: {taxonomy.patch_defense}

### Search Hints (Starting Points)
These are your **initial anchors** - expand from here using tools:

**Deleted Lines (Pre-Patch only, likely vulnerability core)**:
{deleted_lines_text}

**Added Lines (Post-Patch, reveals fix intent)**:
{added_lines_text}

**Key Variables to Track**:
{key_vars}

### Static Analysis Candidates (from PDG two-pass slice)
{candidates if candidates else "No static analysis candidates available — rely on tool-based exploration."}

### Patch Diff
```diff
{diff_text}
```

### Function Code (Pre-Patch/Vulnerable)
```c
{formatted_code}
```

**Task**: Please identify the **typed anchors** ({anchor_types_str}) that form the vulnerability chain: {chain_str}
Use the search hints and tools to trace each anchor type.
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
                    anchors=[],
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
            
            # 9. Populate complete location info (file_path and func_name)
            for anchor in result.anchors:
                anchor.file_path = file_path
                anchor.func_name = function_name
            
            # 10. HARD FILTER: Remove anchors with types not in current category
            allowed_types = set(a.type.value for a in typed_anchors) if typed_anchors else None
            if allowed_types:
                original_count = len(result.anchors)
                filtered = [a for a in result.anchors if a.type.value in allowed_types]
                rejected = [a for a in result.anchors if a.type.value not in allowed_types]
                if rejected:
                    for r in rejected:
                        print(f"      [AnchorAnalyzer] ⚠️ REJECTED anchor type '{r.type.value}' (not in allowed types {allowed_types}): L{r.line_number} {r.code_snippet}")
                    result.anchors = filtered
            
            # 10.5 Conditional Anchor Deduplication (Patch Relevance Filter)
            if typed_anchors and self._has_duplicate_anchor_types(result.anchors, typed_anchors):
                print(f"      [AnchorAnalyzer] 🔍 Duplicate anchor types detected, running deduplication...")
                deduped = self._deduplicate_anchors(
                    anchors=result.anchors,
                    diff_text=diff_text,
                    code_content=code_content,
                    start_line=start_line,
                    typed_anchors=typed_anchors
                )
                if deduped is not None:
                    result.anchors = deduped
            
            # 11. Anchor Type Completeness Check
            found_types = set(a.type.value for a in result.anchors)
            expected_types = set(a.type.value for a in typed_anchors) if typed_anchors else set()
            
            if expected_types:
                missing_types = expected_types - found_types
                if missing_types:
                    print(f"      [AnchorAnalyzer] Incomplete anchor chain: missing types {missing_types}")
                    print(f"        Expected chain: {chain_str}")
                    print(f"        Found types: {found_types}")
            
            # 12. Basic Validation (No anchors found at all)
            if not result.anchors:
                print(f"      [AnchorAnalyzer] Warning: No anchors identified at all")
            for anchor in result.anchors:
                print(f"      [AnchorAnalyzer] Anchor: {anchor.line_number} {anchor.code_snippet} {anchor.type.value} {anchor.reasoning} {anchor.scope} {anchor.cross_function_info}")
            return result
            
        except Exception as e:
            print(f"      [AnchorAnalyzer] Extraction Error: {e}")
            return AnchorResult(
                anchors=[],
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
        Extract line numbers affected by patch (used to constrain anchor search range)
        
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
    
    # ==============================================================================
    # Anchor Deduplication (Patch Relevance Filter)
    # ==============================================================================
    
    def _has_duplicate_anchor_types(
        self,
        anchors: List[Anchor],
        category_anchors: List[Anchor]
    ) -> bool:
        """
        Check if any anchor type has more instances than expected by the category.
        
        Triggers deduplication when the LLM found multiple instances of the same
        anchor type (e.g., 2 object + 2 access when category expects 1 + 1).
        
        Args:
            anchors: Actual anchors found by LLM
            category_anchors: Expected anchor types from VulnerabilityCategory
            
        Returns:
            True if deduplication is needed
        """
        from collections import Counter
        
        actual_counts = Counter(a.type.value for a in anchors)
        expected_counts = Counter(a.type.value for a in category_anchors)
        
        for anchor_type, actual_count in actual_counts.items():
            expected = expected_counts.get(anchor_type, 1)
            if actual_count > expected:
                print(f"      [AnchorAnalyzer] Duplicate detected: type '{anchor_type}' "
                      f"has {actual_count} instances (expected {expected})")
                return True
        
        return False
    
    def _deduplicate_anchors(
        self,
        anchors: List[Anchor],
        diff_text: str,
        code_content: str,
        start_line: int,
        typed_anchors: List[Anchor]
    ) -> Optional[List[Anchor]]:
        """
        Use LLM to select the correct anchor instances when duplicates exist.
        
        Provides the patch diff and all anchors to the LLM with deduplication rules,
        asking it to decide which anchors are truly connected to this specific patch.
        
        Args:
            anchors: All anchors (including duplicates)
            diff_text: Patch diff text
            code_content: Full function code (pre-patch)
            start_line: Starting line number of the code
            typed_anchors: Expected anchor types from category
            
        Returns:
            Filtered anchor list, or None if deduplication fails (caller keeps original)
        """
        # Format anchors for LLM
        def _fmt_anchor(i: int, a: Anchor) -> str:
            reasoning_str = (a.reasoning[:100] + "...") if len(a.reasoning) > 100 else a.reasoning
            return f"  {i+1}. Line {a.line_number} [{a.type.value}]: {a.code_snippet}  (reasoning: {reasoning_str})"
        
        anchors_text = "\n".join([_fmt_anchor(i, a) for i, a in enumerate(anchors)])
        
        # Format code with line numbers (compact: only show ±5 lines around each anchor)
        anchor_lines = {a.line_number for a in anchors}
        lines = code_content.splitlines()
        context_lines = set()
        for al in anchor_lines:
            rel = al - start_line
            for offset in range(-5, 6):
                idx = rel + offset
                if 0 <= idx < len(lines):
                    context_lines.add(idx)
        
        code_context = "\n".join([
            f"[{start_line + i:4d}] {lines[i]}"
            for i in sorted(context_lines)
            if 0 <= i < len(lines)
        ])
        
        # Expected types
        from collections import Counter
        expected_counts = Counter(a.type.value for a in typed_anchors)
        expected_str = ", ".join([f"{t}: {c}" for t, c in expected_counts.items()])
        
        prompt = f"""You are a Patch Relevance Analyst. A vulnerability anchor discovery agent found
MULTIPLE instances of the same anchor type. A patch fixes ONE specific vulnerability instance.
You must select ONLY the anchors that belong to the vulnerability instance THIS PATCH fixes.

### Patch Diff
```diff
{diff_text}
```

### Code Context (around anchor locations)
```c
{code_context}
```

### All Anchors Found (some may be duplicates)
{anchors_text}

### Expected Anchor Types (from vulnerability category)
{expected_str}

### Deduplication Rules

**Rule 1: Direct Patch Connection**
An anchor is patch-relevant if:
- The anchor line IS a patch-modified line (deleted or added), OR
- The anchor line is DIRECTLY adjacent to a patch-modified line (within ±2 lines), OR
- The anchor shares a variable that appears in a patch-modified line AND they are
  on the same execution path (not in different branches/loops)

**Rule 2: Same Execution Path**
Two anchors forming a chain (e.g., object→access) must be on the SAME execution
path as the patch modification. If the patch modifies code OUTSIDE a loop, anchors
INSIDE the loop are a different vulnerability instance (and vice versa).

**Rule 3: One Chain Per Patch**
For each anchor type that has duplicates, select the instance that has the
STRONGEST connection to the patch:
- Prefer anchors on patch-modified lines
- Prefer anchors sharing variables with patch-modified lines
- Prefer anchors on the same execution path as patch modifications
- Reject anchors in different branches/loops that the patch does not touch

**IMPORTANT**: If multiple instances of the same type are ALL genuinely connected
to the same patch (e.g., two sources feeding the same patched computation),
keep them all. Only reject instances that are SEPARATE vulnerability instances
requiring their own separate patch.

### Task
For each anchor, decide: keep (true) or reject (false), with a reason.
"""
        
        try:
            dedup_llm = self.llm.with_structured_output(AnchorDeduplicationResult)
            dedup_result: AnchorDeduplicationResult = retry_on_connection_error(
                lambda: dedup_llm.invoke(prompt),
                max_retries=3
            )
            
            # Build decision map: (line_number, anchor_type) -> keep
            decision_map = {}
            for d in dedup_result.decisions:
                decision_map[(d.line_number, d.anchor_type)] = (d.keep, d.reason)
            
            # Apply decisions
            kept = []
            rejected = []
            for anchor in anchors:
                key = (anchor.line_number, anchor.type.value)
                if key in decision_map:
                    keep, reason = decision_map[key]
                    if keep:
                        kept.append(anchor)
                        print(f"      [Dedup] ✅ KEEP L{anchor.line_number} [{anchor.type.value}]: {reason}")
                    else:
                        rejected.append(anchor)
                        print(f"      [Dedup] ❌ REJECT L{anchor.line_number} [{anchor.type.value}]: {reason}")
                else:
                    # No decision for this anchor — keep by default
                    kept.append(anchor)
                    print(f"      [Dedup] ⚠️ No decision for L{anchor.line_number} [{anchor.type.value}], keeping by default")
            
            print(f"      [Dedup] Result: {len(kept)} kept, {len(rejected)} rejected")
            print(f"      [Dedup] Reasoning: {dedup_result.reasoning}")
            
            # Safety valve: ensure we still have at least one anchor per required type
            kept_types = set(a.type.value for a in kept)
            required_types = set(a.type.value for a in typed_anchors if not a.is_optional)
            missing_required = required_types - kept_types
            
            if missing_required:
                print(f"      [Dedup] ⚠️ Safety valve: missing required types {missing_required} after dedup")
                print(f"      [Dedup] Falling back to original anchors")
                return None  # Signal caller to keep original
            
            if not kept:
                print(f"      [Dedup] ⚠️ Safety valve: no anchors kept, falling back to original")
                return None
            
            return kept
            
        except Exception as e:
            print(f"      [Dedup] ⚠️ LLM deduplication failed: {e}")
            print(f"      [Dedup] Falling back to original anchors")
            return None
