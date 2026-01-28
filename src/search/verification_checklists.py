"""
Vulnerability Type-Specific Verification Checklists
为每种漏洞类型定义验证时必须确认的关键点，避免常见的误报/漏报
"""

from typing import Dict, List
from core.models import GeneralVulnType

# 类型特定验证清单
VERIFICATION_CHECKLISTS: Dict[GeneralVulnType, Dict[str, any]] = {
    
    # ========== 1. Memory Safety (7) ==========
    
    GeneralVulnType.USE_AFTER_FREE: {
        "critical_checks": [
            "Confirm pointer is NOT reassigned (e.g., ptr = NULL) between Free and Use",
            "Verify the SAME pointer variable flows from Free to Use (not different aliases)",
            "Check Free and Use are on FEASIBLE paths (not mutually exclusive branches)"
        ],
        "common_false_positives": [
            "Pointer nullified immediately after free (ptr = NULL)",
            "Use happens in error cleanup that's unreachable after successful free",
            "Different pointer with same name in different scope"
        ],
        "defense_patterns": [
            "ExactPatch: ptr = NULL immediately after free",
            "EquivalentDefense: Reference counting prevents premature free",
            "CalleeDefense: Free function nullifies pointer (kfree_and_null)",
            "CallerDefense: Caller manages lifetime (e.g., deferred free)"
        ],
        "required_tools": ["trace_variable", "grep"],
        "verification_strategy": """
        1. Map Free operation (Origin)
        2. Map Use operation (Impact)
        3. Use `trace_variable(file_path, line, var_name, "backward")` to verify SAME pointer flows from Free to Use
        4. Use `grep(var_name, file_path, mode="def_use")` to check if pointer is reassigned between Free and Use
        5. Use `get_guard_conditions(file_path, line)` to verify path feasibility
        """
    },
    
    GeneralVulnType.DOUBLE_FREE: {
        "critical_checks": [
            "Confirm pointer is NOT nullified after first free",
            "Verify both frees target the SAME pointer (not copies)",
            "Check second free is reachable after first free"
        ],
        "common_false_positives": [
            "First free is followed by ptr = NULL",
            "Two frees are in mutually exclusive error paths",
            "Different pointers to same memory (need alias analysis)"
        ],
        "defense_patterns": [
            "ExactPatch: ptr = NULL after first free",
            "EquivalentDefense: Use free_once pattern or flag",
            "CalleeDefense: Free function checks if already freed",
            "StructuralDefense: Error paths are mutually exclusive"
        ],
        "required_tools": ["trace_variable", "grep"],
        "verification_strategy": """
        1. Locate first Free (Origin)
        2. Locate second Free (Impact)
        3. Verify pointer NOT nullified between frees
        4. Confirm both frees are on same execution path
        """
    },
    
    GeneralVulnType.NULL_POINTER_DEREFERENCE: {
        "critical_checks": [
            "Confirm dereference happens BEFORE any NULL check",
            "Verify NULL source (return value, parameter) actually returns NULL in error case",
            "Check dereference is not protected by implicit checks (e.g., IS_ERR_OR_NULL macro)"
        ],
        "common_false_positives": [
            "NULL check exists but verifier missed it (check control flow carefully)",
            "Function never returns NULL (check callee implementation)",
            "Macro like IS_ERR handles NULL (expand macros)"
        ],
        "defense_patterns": [
            "ExactPatch: if (!ptr) return -ERRNO before dereference",
            "EquivalentDefense: if (IS_ERR_OR_NULL(ptr)) handle error",
            "CalleeDefense: Called function guarantees non-NULL return",
            "CallerDefense: Caller validates pointer before passing"
        ],
        "required_tools": ["find_definition", "get_guard_conditions", "trace_variable"],
        "verification_strategy": """
        1. Map NULL source (Origin): function return, parameter, assignment
        2. Use `find_definition(symbol_name, file_path)` to verify NULL is possible (e.g., malloc can return NULL)
        3. Map dereference (Impact)
        4. Use `get_guard_conditions(file_path, line)` to verify NO NULL check dominates dereference
        5. Use `trace_variable(file_path, line, var_name, "backward")` to ensure same variable flows from source to deref
        """
    },
    
    GeneralVulnType.BUFFER_OVERFLOW: {
        "critical_checks": [
            "Confirm size/bound check is MISSING or INCORRECT (off-by-one, wrong variable)",
            "Verify overflow target is actually a fixed-size buffer (not dynamically allocated with correct size)",
            "Check copy size is influenced by untrusted input or unbounded calculation"
        ],
        "common_false_positives": [
            "Size check exists but uses different variable name (semantic aliasing)",
            "Buffer is dynamically sized to match copy size",
            "Copy is bounded by implicit buffer size (e.g., sizeof)"
        ],
        "defense_patterns": [
            "ExactPatch: if (len > sizeof(buf)) return -EINVAL",
            "EquivalentDefense: Use safe functions (strncpy, snprintf with correct size)",
            "CalleeDefense: Called function performs bound check",
            "StructuralDefense: Buffer dynamically allocated to exact needed size"
        ],
        "required_tools": ["trace_variable", "find_definition"],
        "verification_strategy": """
        1. Map size/bound source (Origin): user input, calculation
        2. Map buffer write (Impact): memcpy, strcpy, array access
        3. Trace size variable from source to sink
        4. Verify NO correct bound check exists between source and sink
        5. Confirm buffer size is smaller than possible copy size
        """
    },
    
    GeneralVulnType.OUT_OF_BOUNDS_READ: {
        "critical_checks": [
            "Confirm index is NOT validated against array bounds",
            "Verify index can exceed array size (check calculation and range)",
            "Check array access is not protected by implicit bounds (e.g., for loop limit)"
        ],
        "common_false_positives": [
            "Loop condition implicitly bounds the index",
            "Array is dynamically sized to match index range",
            "Earlier validation exists but verifier missed it"
        ],
        "defense_patterns": [
            "ExactPatch: if (index >= array_size) return -EINVAL",
            "EquivalentDefense: Use min(index, array_size - 1) to clamp",
            "StructuralDefense: Loop condition prevents out-of-bounds (for i < size)"
        ],
        "required_tools": ["trace_variable", "get_guard_conditions"],
        "verification_strategy": """
        1. Map index source (Origin)
        2. Map array access (Impact)
        3. Trace index from source to access
        4. Verify NO bound check (index < size) dominates access
        """
    },
    
    GeneralVulnType.MEMORY_LEAK: {
        "critical_checks": [
            "Confirm allocated memory is NOT freed on ALL error paths",
            "Verify leak is not a FALSE ALARM: check if caller/callee frees the memory",
            "Check error path is reachable and returns without cleanup"
        ],
        "common_false_positives": [
            "Caller is responsible for freeing (ownership transfer pattern)",
            "Memory is stored in global/long-lived structure (intentional)",
            "Error path is unreachable (dead code)"
        ],
        "defense_patterns": [
            "ExactPatch: Add kfree(ptr) before return on error path",
            "EquivalentDefense: Use goto cleanup pattern with single free point",
            "CallerDefense: Caller frees on error (ownership transfer)"
        ],
        "required_tools": ["find_definition", "get_callers"],
        "verification_strategy": """
        1. Map allocation (Origin)
        2. Identify error path that returns early (Impact: leak point)
        3. Verify NO free exists on that path
        4. Use `get_callers(symbol_name)` to check if caller frees the memory
        5. Use `find_definition(symbol_name, file_path)` to check if callee frees
        """
    },
    
    GeneralVulnType.UNINITIALIZED_USE: {
        "critical_checks": [
            "Confirm variable is NOT initialized on ALL paths to use",
            "Verify use is not a benign read (e.g., address-of operation that doesn't dereference)",
            "Check initialization is not done by called function (pass-by-reference)"
        ],
        "common_false_positives": [
            "Variable initialized in called function via pointer",
            "Conditional initialization exists but verifier missed path",
            "Use is benign (e.g., &var passed to function that initializes it)"
        ],
        "defense_patterns": [
            "ExactPatch: Initialize variable at declaration (int var = 0)",
            "EquivalentDefense: Initialize on all code paths before use",
            "CalleeDefense: Function initializes variable via pointer parameter"
        ],
        "required_tools": ["trace_variable", "find_definition"],
        "verification_strategy": """
        1. Map variable declaration (Origin)
        2. Map variable use (Impact)
        3. Trace all paths from declaration to use
        4. Verify NO initialization on AT LEAST ONE path
        5. Check if function initializes variable via pointer
        """
    },
    
    # ========== 2. Concurrency (2) ==========
    
    GeneralVulnType.RACE_CONDITION: {
        "critical_checks": [
            "Confirm Check and Use are NOT protected by the SAME lock",
            "Verify concurrent thread can modify shared state between Check and Use",
            "Check TOCTOU window is exploitable (not too narrow)"
        ],
        "common_false_positives": [
            "Check and Use are protected by same lock (verifier missed lock analysis)",
            "Shared state is immutable after Check",
            "TOCTOU window is too small to exploit in practice"
        ],
        "defense_patterns": [
            "ExactPatch: Acquire lock before Check, release after Use",
            "EquivalentDefense: Use atomic operations instead of Check-Use pattern",
            "StructuralDefense: Shared state is immutable (const, read-only)"
        ],
        "required_tools": ["find_definition", "trace_variable"],
        "verification_strategy": """
        1. Map Check operation (Origin)
        2. Map Use operation (Impact)
        3. Verify NO lock acquisition dominates both Check and Use
        4. Confirm shared variable is accessible by other threads
        """
    },
    
    GeneralVulnType.DEADLOCK: {
        "critical_checks": [
            "Confirm lock order is INCONSISTENT across different code paths/functions",
            "Verify both locks can be held simultaneously",
            "Check deadlock cycle is reachable"
        ],
        "common_false_positives": [
            "Locks are never held simultaneously",
            "Lock order is actually consistent (verifier misanalyzed)",
            "One path is unreachable"
        ],
        "defense_patterns": [
            "ExactPatch: Enforce consistent lock order (always A then B)",
            "EquivalentDefense: Use lock hierarchy or lock ranking",
            "StructuralDefense: Locks are never held simultaneously"
        ],
        "required_tools": ["find_definition", "get_callers"],
        "verification_strategy": """
        1. Map first lock order: Lock A → Lock B
        2. Map second lock order: Lock B → Lock A
        3. Verify both orderings are reachable
        4. Confirm locks are the same objects
        """
    },
    
    # ========== 3. Numeric & Type (3) ==========
    
    GeneralVulnType.INTEGER_OVERFLOW: {
        "critical_checks": [
            "Confirm overflow result is used in DANGEROUS operation (memory allocation, array index, size calculation)",
            "Verify overflow is POSSIBLE (check input ranges and arithmetic)",
            "Check overflow is NOT prevented by range check or safe arithmetic"
        ],
        "common_false_positives": [
            "Overflow is possible but result is not used dangerously",
            "Input validation prevents overflow",
            "Safe arithmetic primitives are used (e.g., check_add_overflow)"
        ],
        "defense_patterns": [
            "ExactPatch: if (a > MAX - b) return -EOVERFLOW",
            "EquivalentDefense: Use safe arithmetic (check_add_overflow, __builtin_add_overflow)",
            "EquivalentDefense: Validate inputs before arithmetic (if (size > MAX_SIZE))"
        ],
        "required_tools": ["trace_variable"],
        "verification_strategy": """
        1. Map arithmetic operation (Origin)
        2. Map dangerous use of result (Impact): malloc(overflowed_size), array[overflowed_index]
        3. Trace value from arithmetic to dangerous use
        4. Verify NO overflow check exists between Origin and Impact
        5. Confirm overflow is exploitable (can wrap to small value)
        """
    },
    
    GeneralVulnType.DIVIDE_BY_ZERO: {
        "critical_checks": [
            "Confirm divisor can be ZERO (check all paths leading to division)",
            "Verify division is reachable when divisor is zero",
            "Check NO zero-check guards the division"
        ],
        "common_false_positives": [
            "Zero-check exists but verifier missed it",
            "Divisor is guaranteed non-zero by earlier logic",
            "Division is in dead code"
        ],
        "defense_patterns": [
            "ExactPatch: if (divisor == 0) return -EINVAL",
            "EquivalentDefense: divisor = max(divisor, 1) before division",
            "CalleeDefense: Function guarantees non-zero return"
        ],
        "required_tools": ["trace_variable", "get_guard_conditions"],
        "verification_strategy": """
        1. Map divisor source (Origin)
        2. Map division operation (Impact)
        3. Trace divisor from source to division
        4. Verify NO zero-check (divisor != 0) dominates division
        """
    },
    
    GeneralVulnType.TYPE_CONFUSION: {
        "critical_checks": [
            "Confirm object is accessed as WRONG type after cast/union",
            "Verify type transition is not validated",
            "Check confused access causes memory corruption or info leak"
        ],
        "common_false_positives": [
            "Type check exists (e.g., type field in union)",
            "Cast is safe (compatible types)",
            "Access is benign (no corruption/leak)"
        ],
        "defense_patterns": [
            "ExactPatch: if (obj->type != EXPECTED_TYPE) return -EINVAL",
            "EquivalentDefense: Use container_of with type validation",
            "StructuralDefense: Types are compatible (safe cast)"
        ],
        "required_tools": ["find_definition", "trace_variable"],
        "verification_strategy": """
        1. Map type transition (Origin): cast, union assignment
        2. Map confused access (Impact): dereference as wrong type
        3. Verify NO type validation between Origin and Impact
        4. Confirm confused access is exploitable
        """
    },
    
    # ========== 4. Logic & Access Control (4) ==========
    
    GeneralVulnType.AUTHENTICATION_BYPASS: {
        "critical_checks": [
            "Confirm bypass path is REACHABLE (not dead code)",
            "Verify bypassed check is CRITICAL for security",
            "Check bypass is not intentional (e.g., local admin access)"
        ],
        "common_false_positives": [
            "Bypass path is unreachable",
            "Alternative authentication exists",
            "Access is intentionally unauthenticated for specific case"
        ],
        "defense_patterns": [
            "ExactPatch: Add authentication check on bypass path",
            "EquivalentDefense: Use centralized authentication gateway",
            "StructuralDefense: Bypass path is unreachable or intentional"
        ],
        "required_tools": ["get_guard_conditions", "get_callers"],
        "verification_strategy": """
        1. Map authentication check (Origin)
        2. Map protected operation (Impact)
        3. Find control flow path from entry to Impact that bypasses Origin
        4. Verify bypass path is reachable
        """
    },
    
    GeneralVulnType.PRIVILEGE_ESCALATION: {
        "critical_checks": [
            "Confirm privileged operation is reachable without proper privilege check",
            "Verify attacker can reach the bypass path",
            "Check escalated privilege is meaningful (not redundant)"
        ],
        "common_false_positives": [
            "Caller already has required privilege",
            "Privilege check exists in caller",
            "Operation is not actually privileged"
        ],
        "defense_patterns": [
            "ExactPatch: if (!capable(CAP_SYS_ADMIN)) return -EPERM",
            "CallerDefense: Caller performs privilege check before calling",
            "StructuralDefense: Operation is not actually privileged"
        ],
        "required_tools": ["get_guard_conditions", "get_callers"],
        "verification_strategy": """
        1. Map privilege check (Origin)
        2. Map privileged operation (Impact)
        3. Find bypass path
        4. Verify attacker can invoke the bypass
        """
    },
    
    GeneralVulnType.AUTHORIZATION_BYPASS: {
        "critical_checks": [
            "Confirm resource access bypasses ownership/permission check",
            "Verify attacker can access victim's resource",
            "Check bypass is exploitable (not theoretical)"
        ],
        "common_false_positives": [
            "Resource is intentionally shared",
            "Check exists in different layer",
            "Bypass path is unreachable"
        ],
        "defense_patterns": [
            "ExactPatch: if (resource->owner != current_user) return -EACCES",
            "EquivalentDefense: Use ACL or capability-based access control",
            "CallerDefense: Authorization enforced at higher layer"
        ],
        "required_tools": ["get_guard_conditions", "get_callers"],
        "verification_strategy": """
        1. Map authorization check (Origin)
        2. Map resource access (Impact)
        3. Find bypass path
        4. Verify attacker can trigger bypass
        """
    },
    
    GeneralVulnType.LOGIC_ERROR: {
        "critical_checks": [
            "Confirm logic error causes SECURITY impact (not just functional bug)",
            "Verify incorrect condition is exploitable",
            "Check error is not benign edge case"
        ],
        "common_false_positives": [
            "Logic error exists but has no security impact",
            "Error is in dead code or unreachable path",
            "Impact is minor (DOS at most)"
        ],
        "defense_patterns": [
            "ExactPatch: Fix incorrect condition (< to <=, wrong variable)",
            "EquivalentDefense: Restructure logic to avoid error-prone patterns",
            "StructuralDefense: Error has no security impact"
        ],
        "required_tools": ["get_guard_conditions", "trace_variable"],
        "verification_strategy": """
        1. Map incorrect logic (Origin): wrong condition, off-by-one
        2. Map security impact (Impact): corruption, bypass, leak
        3. Verify logic error causes Impact
        4. Confirm exploitability
        """
    },
    
    # ========== 5. Input & Data (5) ==========
    
    GeneralVulnType.INJECTION: {
        "critical_checks": [
            "Confirm untrusted input reaches interpreter sink WITHOUT sanitization",
            "Verify input is attacker-controllable",
            "Check injection breaks out of intended context (e.g., SQL quote escape)"
        ],
        "common_false_positives": [
            "Input is sanitized/escaped before sink",
            "Input is not actually attacker-controllable",
            "Sink is not an interpreter (safe usage)"
        ],
        "defense_patterns": [
            "ExactPatch: Escape/sanitize input before use (sql_escape, htmlspecialchars)",
            "EquivalentDefense: Use parameterized queries/prepared statements",
            "EquivalentDefense: Whitelist validation (only allow known-good patterns)",
            "StructuralDefense: Input is not attacker-controllable"
        ],
        "required_tools": ["trace_variable", "find_definition"],
        "verification_strategy": """
        1. Map untrusted input (Origin)
        2. Map interpreter sink (Impact): SQL exec, command exec, eval
        3. Trace taint from Origin to Impact
        4. Verify NO sanitization on path
        """
    },
    
    GeneralVulnType.PATH_TRAVERSAL: {
        "critical_checks": [
            "Confirm path is NOT validated/sanitized (e.g., no '../' check)",
            "Verify attacker can control path components",
            "Check traversal reaches sensitive files outside intended directory"
        ],
        "common_false_positives": [
            "Path validation exists (e.g., realpath, basename)",
            "Attacker cannot control path",
            "Access control limits damage"
        ],
        "defense_patterns": [
            "ExactPatch: Use realpath/canonicalize and verify prefix",
            "EquivalentDefense: Reject paths containing '..' or absolute paths",
            "EquivalentDefense: Use basename to strip directory components",
            "StructuralDefense: chroot jail limits traversal scope"
        ],
        "required_tools": ["trace_variable", "find_definition"],
        "verification_strategy": """
        1. Map user-controlled path (Origin)
        2. Map file access (Impact): open, read, write
        3. Trace path from Origin to Impact
        4. Verify NO path sanitization (no '..' removal, no chroot)
        """
    },
    
    GeneralVulnType.IMPROPER_VALIDATION: {
        "critical_checks": [
            "Confirm validation is MISSING, INCOMPLETE, or BYPASSABLE",
            "Verify unvalidated input reaches dangerous use",
            "Check validation logic has flaw (e.g., TOCTOU, incomplete check)"
        ],
        "common_false_positives": [
            "Validation exists but verifier missed it",
            "Input is validated in caller",
            "Dangerous use is not reachable"
        ],
        "defense_patterns": [
            "ExactPatch: Add comprehensive input validation",
            "EquivalentDefense: Use whitelist instead of blacklist",
            "CallerDefense: Validation performed in caller function"
        ],
        "required_tools": ["trace_variable", "get_guard_conditions"],
        "verification_strategy": """
        1. Map input source (Origin)
        2. Map dangerous use (Impact)
        3. Trace input from Origin to Impact
        4. Verify NO proper validation exists
        """
    },
    
    GeneralVulnType.INFORMATION_EXPOSURE: {
        "critical_checks": [
            "Confirm uninitialized/sensitive data is ACTUALLY COPIED OUT (copy_to_user, send, log)",
            "Verify struct padding/unused fields are not initialized BUT whole struct is copied",
            "Check exposed data is SENSITIVE (not public info)"
        ],
        "common_false_positives": [
            "**CRITICAL**: All struct fields are initialized even if declaration doesn't show it",
            "Only initialized portion is copied (not whole struct)",
            "Exposed data is not sensitive",
            "memset/memzero clears uninitialized bytes before copy"
        ],
        "required_tools": ["trace_variable", "grep", "find_definition"],
        "verification_strategy": """
        1. Map sensitive data source (Origin): uninitialized struct, kernel pointer
        2. Use `grep(pattern, file_path, mode="def_use")` to check ALL fields of struct - verify some remain uninitialized
        3. Map output operation (Impact): copy_to_user, sendmsg, printk
        4. Verify WHOLE struct/buffer is copied (not just initialized fields)
        5. Check NO memset/memzero before copy
        6. Confirm exposed bytes contain sensitive data (not padding only)
        """
    },
    
    GeneralVulnType.CRYPTOGRAPHIC_ISSUE: {
        "critical_checks": [
            "Confirm weak crypto primitive is used (e.g., MD5, RC4, ECB mode)",
            "Verify key management is flawed (hardcoded key, weak RNG)",
            "Check crypto failure has security impact"
        ],
        "common_false_positives": [
            "Weak crypto is used for non-security purpose (checksum)",
            "Strong crypto is available and used in parallel",
            "Impact is minimal"
        ],
        "defense_patterns": [
            "ExactPatch: Replace with strong crypto (SHA256, AES-GCM, ChaCha20)",
            "EquivalentDefense: Use crypto_secure_random instead of rand()",
            "StructuralDefense: Weak crypto used for non-security purpose only"
        ],
        "required_tools": ["find_definition", "trace_variable"],
        "verification_strategy": """
        1. Map weak crypto usage (Origin)
        2. Map security-critical operation relying on crypto (Impact)
        3. Verify crypto strength is insufficient for use case
        """
    },
    
    # ========== 6. Resource & Execution (4) ==========
    
    GeneralVulnType.INFINITE_LOOP: {
        "critical_checks": [
            "Confirm loop termination condition can NEVER be satisfied",
            "Verify loop is reachable and triggers infinite iteration",
            "Check infinite loop causes DOS (not just hang background task)"
        ],
        "common_false_positives": [
            "Loop has break condition inside body",
            "External event terminates loop (signal, timer)",
            "Infinite loop is intentional (event loop)"
        ],
        "defense_patterns": [
            "ExactPatch: Fix loop condition or add iteration limit",
            "EquivalentDefense: Add timeout or max iteration check",
            "StructuralDefense: Loop has break/return inside body"
        ],
        "required_tools": ["get_guard_conditions", "find_definition"],
        "verification_strategy": """
        1. Map loop condition (Origin)
        2. Analyze loop body for termination logic
        3. Verify condition can never become false
        4. Confirm loop is reachable
        """
    },
    
    GeneralVulnType.RECURSION_ERROR: {
        "critical_checks": [
            "Confirm recursion depth is UNBOUNDED (no depth limit check)",
            "Verify attacker can control recursion depth",
            "Check recursion causes stack exhaustion (not tail-call optimized)"
        ],
        "common_false_positives": [
            "Recursion depth is limited by input constraints",
            "Depth check exists",
            "Tail-call optimization prevents stack growth"
        ],
        "defense_patterns": [
            "ExactPatch: Add depth counter and limit (if (depth > MAX) return)",
            "EquivalentDefense: Convert to iterative loop",
            "StructuralDefense: Input constraints bound recursion depth"
        ],
        "required_tools": ["find_definition", "trace_variable"],
        "verification_strategy": """
        1. Map recursive call (Origin)
        2. Check for depth limit validation
        3. Verify attacker can trigger deep recursion
        """
    },
    
    GeneralVulnType.RESOURCE_EXHAUSTION: {
        "critical_checks": [
            "Confirm resource allocation is UNBOUNDED (no limit check)",
            "Verify attacker can trigger excessive allocation",
            "Check exhaustion causes DOS or crash"
        ],
        "common_false_positives": [
            "Allocation limit exists (e.g., quota, MAX constant)",
            "Resource is limited by system constraints",
            "DOS impact is minimal"
        ],
        "defense_patterns": [
            "ExactPatch: Add resource limit check (if (count > MAX_RESOURCES))",
            "EquivalentDefense: Use resource quota/accounting system",
            "StructuralDefense: System constraints prevent exhaustion"
        ],
        "required_tools": ["trace_variable", "get_guard_conditions"],
        "verification_strategy": """
        1. Map resource allocation (Origin)
        2. Check for limit validation
        3. Verify attacker can control allocation amount
        4. Confirm exhaustion is exploitable
        """
    },
    
    GeneralVulnType.RESOURCE_LEAK: {
        "critical_checks": [
            "Confirm resource is NOT released on error paths",
            "Verify leak is not handled by caller/callee",
            "Check leak accumulates (not one-time)"
        ],
        "common_false_positives": [
            "Caller/callee releases resource",
            "Resource is long-lived (not a leak)",
            "Error path is unreachable"
        ],
        "defense_patterns": [
            "ExactPatch: Add resource release on error path",
            "EquivalentDefense: Use RAII pattern or defer/cleanup handlers",
            "CallerDefense: Caller responsible for cleanup"
        ],
        "required_tools": ["find_definition", "get_callers"],
        "verification_strategy": """
        1. Map resource acquisition (Origin)
        2. Identify error path without release
        3. Verify leak accumulates over time
        4. Check caller doesn't release
        """
    },
    
    # ========== 7. Fallback ==========
    
    GeneralVulnType.UNKNOWN: {
        "critical_checks": [],
        "common_false_positives": [],
        "defense_patterns": [],
        "required_tools": [],
        "verification_strategy": "Manual analysis required"
    },
    
    GeneralVulnType.OTHER: {
        "critical_checks": [],
        "common_false_positives": [],
        "defense_patterns": [],
        "required_tools": [],
        "verification_strategy": "Custom analysis based on vulnerability type"
    },
}


def get_verification_checklist(vuln_type: GeneralVulnType) -> Dict[str, any]:
    """获取特定漏洞类型的验证清单"""
    return VERIFICATION_CHECKLISTS.get(vuln_type, VERIFICATION_CHECKLISTS[GeneralVulnType.UNKNOWN])


def format_checklist_for_prompt(vuln_type: GeneralVulnType, agent_role: str = "red") -> str:
    """
    格式化验证清单为 Prompt 文本
    
    Args:
        vuln_type: 漏洞类型
        agent_role: "red" 或 "blue"，决定强调哪些检查项
    
    Returns:
        格式化的检查清单文本
    """
    checklist = get_verification_checklist(vuln_type)
    
    if agent_role == "red":
        # Red Agent: 强调必须验证的关键点
        output = f"### Type-Specific Verification Checklist for {vuln_type.value}\n\n"
        output += "**CRITICAL: You MUST confirm ALL of the following before claiming VULNERABLE:**\n"
        for i, check in enumerate(checklist["critical_checks"], 1):
            output += f"{i}. {check}\n"
        
        if checklist["required_tools"]:
            output += f"\n**Recommended Tools**: {', '.join([f'`{t}`' for t in checklist['required_tools']])}\n"
        
        output += f"\n**Verification Strategy**:\n{checklist['verification_strategy']}\n"
        
    else:  # blue
        # Blue Agent: 强调常见误报场景和防御模式
        output = f"### Defense Strategy Guide for {vuln_type.value}\n\n"
        
        # 1. Common False Positives (用于反驳 Red)
        output += "**STEP 1: Look for these False Positive scenarios (to REFUTE Red's claims):**\n"
        for i, fp in enumerate(checklist["common_false_positives"], 1):
            output += f"{i}. {fp}\n"
        
        # 2. Defense Patterns (用于建立 C_def)
        if "defense_patterns" in checklist and checklist["defense_patterns"]:
            output += "\n**STEP 2: Search for Defense Mechanisms (to establish C_def):**\n"
            output += "Use this 4-layer priority order:\n"
            for pattern in checklist["defense_patterns"]:
                output += f"  - {pattern}\n"
        
        if checklist["required_tools"]:
            output += f"\n**Essential Tools**: {', '.join([f'`{t}`' for t in checklist['required_tools']])}\n"
        
        output += "\n**REMEMBER**: You must PROVE defense exists with exact code location. NO hypotheticals.\n"
    
    return output
