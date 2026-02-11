"""
Distillation Strategies for Vulnerability-Type Specific Slice Cleaning

Each strategy defines:
1. focus_dataflow: What data elements are critical for this vulnerability type
2. keep_signals: Patterns/elements that should be kept
3. remove_signals: Patterns/elements that should be removed (noise)
4. guidance: Natural language instruction for the LLM distiller

Based on GeneralVulnType (27 categories) from core/models.py
"""

from typing import Dict, Any
from core.models import GeneralVulnType


class DistillationStrategy:
    """Single vulnerability type's distillation strategy"""
    
    def __init__(
        self,
        focus_dataflow: str,
        keep_signals: list,
        remove_signals: list,
        guidance: str
    ):
        self.focus_dataflow = focus_dataflow
        self.keep_signals = keep_signals
        self.remove_signals = remove_signals
        self.guidance = guidance
    
    def to_prompt_section(self) -> str:
        """Generate prompt section for this strategy"""
        return f"""
### Vulnerability-Type Specific Filtering ({self.focus_dataflow})

**Focus Data Flow**: {self.focus_dataflow}

**KEEP signals** (high relevance):
{chr(10).join(f'- {s}' for s in self.keep_signals)}

**REMOVE signals** (noise):
{chr(10).join(f'- {s}' for s in self.remove_signals)}

**Guidance**: {self.guidance}
"""


# ==============================================================================
# Strategy Definitions for All 27 Vulnerability Types
# ==============================================================================

DISTILLATION_STRATEGIES: Dict[GeneralVulnType, DistillationStrategy] = {
    
    # =========================================================================
    # 1. Memory Safety (7 types)
    # =========================================================================
    
    GeneralVulnType.USE_AFTER_FREE: DistillationStrategy(
        focus_dataflow="Object lifecycle: allocation → use → free",
        keep_signals=[
            "Memory allocation (malloc, kmalloc, alloc_*, create_*)",
            "Free operations (free, kfree, put_*, release_*)",
            "Pointer assignments that propagate the freed object",
            "Dereferences or uses of the object after potential free",
            "Reference counting operations (get/put, inc/dec ref)",
            "Control flow that determines free vs. use ordering",
        ],
        remove_signals=[
            "Unrelated field accesses on different objects",
            "Size calculations not affecting the freed object",
            "Logging, tracing, or debugging statements",
            "Operations on other memory regions not related to the UAF object",
        ],
        guidance="Focus ONLY on the object's lifecycle from allocation through free to post-free use. "
                 "Ignore operations on other objects or fields that don't affect the UAF chain."
    ),
    
    GeneralVulnType.DOUBLE_FREE: DistillationStrategy(
        focus_dataflow="Object lifecycle: allocation → first free → second free",
        keep_signals=[
            "Memory allocation of the target object",
            "All free operations on the target pointer",
            "Pointer assignments (aliasing that creates multiple paths to free)",
            "Conditional branches controlling free paths",
            "Flag or state checks that should prevent double free",
        ],
        remove_signals=[
            "Uses of the object between allocations (not free-related)",
            "Operations on different objects",
            "Field accesses that don't affect the free decision",
        ],
        guidance="Focus on all paths that can lead to freeing the same object. "
                 "Track pointer aliasing carefully - if two pointers point to the same object, both free paths matter."
    ),
    
    GeneralVulnType.OUT_OF_BOUNDS_WRITE: DistillationStrategy(
        focus_dataflow="Buffer size vs. access size relationship",
        keep_signals=[
            "Buffer allocation with size",
            "Size variable definitions and computations",
            "Memory copy/write operations (memcpy, strcpy, read, write)",
            "Array index calculations",
            "Length/size checks (or lack thereof)",
            "User input that influences size or index",
        ],
        remove_signals=[
            "Unrelated pointer operations not affecting the buffer",
            "Operations on other buffers",
            "Content processing that doesn't affect bounds",
            "Format strings or other parameters not affecting size",
        ],
        guidance="Focus on the relationship between allocated size and accessed size. "
                 "Track all computations that affect buffer bounds."
    ),
    
    GeneralVulnType.OUT_OF_BOUNDS_READ: DistillationStrategy(
        focus_dataflow="Array/buffer index vs. bounds relationship",
        keep_signals=[
            "Array/buffer declarations with size",
            "Index calculations and assignments",
            "Read operations using the index",
            "Bounds checking (or absence)",
            "Loop counters affecting index",
            "User-controlled values affecting index",
        ],
        remove_signals=[
            "Write operations to different memory",
            "Operations on different arrays",
            "Content processing after the read",
        ],
        guidance="Focus on how the index is computed and whether it's validated against bounds before the read."
    ),
    
    GeneralVulnType.MEMORY_LEAK: DistillationStrategy(
        focus_dataflow="Allocation → (missing) free path",
        keep_signals=[
            "Memory allocation",
            "Assignment to tracking variables",
            "Error handling paths (where free might be missing)",
            "Early return statements",
            "Conditional branches that skip cleanup",
            "Free operations (to verify they exist on all paths)",
        ],
        remove_signals=[
            "Uses of the allocated memory (not affecting leak)",
            "Operations after successful free",
            "Unrelated resource operations",
        ],
        guidance="Focus on control flow paths from allocation to function exit. "
                 "Identify paths where free is missing, especially error paths."
    ),
    
    GeneralVulnType.NULL_POINTER_DEREFERENCE: DistillationStrategy(
        focus_dataflow="Pointer value origin → dereference point",
        keep_signals=[
            "Pointer assignments (especially from functions that can return NULL)",
            "Null checks (if present) and their control flow",
            "Dereference operations (direct or via -> operator)",
            "Error paths where pointer might be NULL",
            "Function calls that might set pointer to NULL",
        ],
        remove_signals=[
            "Other parameters in expressions but not containing the target pointer",
            "Operations on different pointers",
            "Content of pointed-to data (unless affects NULL state)",
            "Logging or debugging using the pointer",
        ],
        guidance="Focus ONLY on the pointer's value origin (where it might become NULL) and its dereference. "
                 "Other parameters in the same expression are typically NOT relevant to NPD."
    ),
    
    GeneralVulnType.UNINITIALIZED_USE: DistillationStrategy(
        focus_dataflow="Variable declaration → (missing initialization) → use",
        keep_signals=[
            "Variable declarations",
            "Initialization assignments",
            "Conditional paths that might skip initialization",
            "Uses of the variable",
            "Error paths with early returns",
        ],
        remove_signals=[
            "Operations after proper initialization",
            "Uses of other variables",
            "Computations not involving the uninitialized variable",
        ],
        guidance="Focus on all paths from declaration to use, identifying which paths lack initialization."
    ),
    
    # =========================================================================
    # 2. Concurrency (2 types)
    # =========================================================================
    
    GeneralVulnType.RACE_CONDITION: DistillationStrategy(
        focus_dataflow="Shared state access across concurrent contexts",
        keep_signals=[
            "Lock/unlock operations (mutex, spinlock, semaphore)",
            "Shared variable accesses (reads and writes)",
            "Atomic operations",
            "Memory barriers",
            "Thread/process creation affecting shared state",
            "Check-then-act sequences on shared data",
        ],
        remove_signals=[
            "Thread-local operations",
            "Operations on non-shared data",
            "Unrelated synchronization for other resources",
        ],
        guidance="Focus on accesses to shared state and their synchronization (or lack thereof). "
                 "Check-then-act patterns are especially important."
    ),
    
    GeneralVulnType.DEADLOCK: DistillationStrategy(
        focus_dataflow="Lock acquisition order and dependencies",
        keep_signals=[
            "All lock acquisition operations",
            "All lock release operations",
            "Nested lock acquisitions",
            "Conditional lock paths",
            "Wait operations",
        ],
        remove_signals=[
            "Operations between lock/unlock (unless they affect locking)",
            "Data processing unrelated to synchronization",
        ],
        guidance="Focus on lock acquisition order across different code paths. "
                 "Identify potential circular dependencies."
    ),
    
    # =========================================================================
    # 3. Numeric & Type (3 types)
    # =========================================================================
    
    GeneralVulnType.INTEGER_OVERFLOW: DistillationStrategy(
        focus_dataflow="Integer computation → use in size/allocation/index",
        keep_signals=[
            "Arithmetic operations (add, multiply, shift)",
            "Type conversions (especially narrowing)",
            "Uses of result in allocation sizes",
            "Uses of result in array indices",
            "Overflow checks (or absence)",
            "User input affecting the computation",
        ],
        remove_signals=[
            "Unrelated arithmetic on other variables",
            "Operations after bounds checking",
            "String formatting or logging",
        ],
        guidance="Focus on the arithmetic chain from input to memory operation. "
                 "Track type conversions that might cause truncation."
    ),
    
    GeneralVulnType.DIVIDE_BY_ZERO: DistillationStrategy(
        focus_dataflow="Divisor value origin → division operation",
        keep_signals=[
            "Division/modulo operations",
            "Divisor variable assignments",
            "Zero checks on divisor (or absence)",
            "Paths where divisor could be zero",
            "User input affecting divisor",
        ],
        remove_signals=[
            "Dividend computations (usually not security-relevant)",
            "Uses of division result",
            "Unrelated arithmetic",
        ],
        guidance="Focus on how the divisor gets its value and whether zero is possible."
    ),
    
    GeneralVulnType.TYPE_CONFUSION: DistillationStrategy(
        focus_dataflow="Type assignment/cast → mismatched use",
        keep_signals=[
            "Type casts (especially unsafe casts)",
            "Union member access",
            "Polymorphic dispatch (virtual calls)",
            "Type checks (instanceof, typeof)",
            "Object/struct assignments across types",
        ],
        remove_signals=[
            "Operations within correct type context",
            "Unrelated type operations",
        ],
        guidance="Focus on where type information is established and where it's assumed. "
                 "Look for mismatches in type expectations."
    ),
    
    # =========================================================================
    # 4. Logic & Access Control (4 types)
    # =========================================================================
    
    GeneralVulnType.AUTHENTICATION_BYPASS: DistillationStrategy(
        focus_dataflow="Authentication check → protected resource access",
        keep_signals=[
            "Authentication function calls",
            "Credential validation",
            "Session/token checks",
            "Access to protected resources",
            "Bypass conditions (special cases, defaults)",
            "Error handling in auth flow",
        ],
        remove_signals=[
            "Unrelated functionality after successful auth",
            "Logging of auth events",
            "UI/display logic",
        ],
        guidance="Focus on the authentication decision path and conditions that might bypass it."
    ),
    
    GeneralVulnType.PRIVILEGE_ESCALATION: DistillationStrategy(
        focus_dataflow="Privilege level changes and checks",
        keep_signals=[
            "Privilege/capability checks",
            "setuid/setgid/capability operations",
            "Permission validation",
            "Resource access with privilege requirements",
            "Conditions affecting privilege decisions",
        ],
        remove_signals=[
            "Operations at correct privilege level",
            "Unrelated system calls",
        ],
        guidance="Focus on how privileges are checked, granted, or dropped."
    ),
    
    GeneralVulnType.AUTHORIZATION_BYPASS: DistillationStrategy(
        focus_dataflow="Authorization check → protected action",
        keep_signals=[
            "Permission/ACL checks",
            "Role validation",
            "Resource ownership verification",
            "Access control decisions",
            "Protected operations",
            "Bypass conditions",
        ],
        remove_signals=[
            "Actions after successful authorization",
            "Unrelated permission checks",
        ],
        guidance="Focus on the authorization decision and paths that might skip or bypass it."
    ),
    
    GeneralVulnType.LOGIC_ERROR: DistillationStrategy(
        focus_dataflow="Logical condition → incorrect behavior",
        keep_signals=[
            "Conditional expressions (if, switch, ternary)",
            "State machine transitions",
            "Business logic decisions",
            "Edge cases and boundary conditions",
            "Return value handling",
        ],
        remove_signals=[
            "Correct logic paths",
            "Unrelated conditionals",
            "Logging and monitoring",
        ],
        guidance="Focus on the logical flaw - incorrect condition, missing case, or wrong operator."
    ),
    
    # =========================================================================
    # 5. Input & Data (5 types)
    # =========================================================================
    
    GeneralVulnType.INJECTION: DistillationStrategy(
        focus_dataflow="User input → command/query construction → execution",
        keep_signals=[
            "User input sources",
            "String concatenation/formatting with user data",
            "Command/query construction",
            "Execution functions (exec, system, query)",
            "Sanitization/escaping (or absence)",
        ],
        remove_signals=[
            "Hardcoded command parts (safe)",
            "Post-execution processing",
            "Unrelated input handling",
        ],
        guidance="Focus on the data flow from user input to command/query execution. "
                 "Look for missing sanitization."
    ),
    
    GeneralVulnType.PATH_TRAVERSAL: DistillationStrategy(
        focus_dataflow="User input → file path → file operation",
        keep_signals=[
            "User input containing path components",
            "Path concatenation/construction",
            "Path normalization (or absence)",
            "File operations (open, read, write, delete)",
            "Path validation checks",
        ],
        remove_signals=[
            "File content processing",
            "Unrelated file operations",
            "Hardcoded safe paths",
        ],
        guidance="Focus on how user input becomes part of file paths and whether '..' sequences are prevented."
    ),
    
    GeneralVulnType.IMPROPER_VALIDATION: DistillationStrategy(
        focus_dataflow="Input → validation → use",
        keep_signals=[
            "Input reception",
            "Validation checks (or absence)",
            "Uses of potentially invalid input",
            "Error handling for invalid input",
            "Boundary checks",
        ],
        remove_signals=[
            "Processing of validated input",
            "Unrelated input fields",
        ],
        guidance="Focus on what validation is performed and what's missing. "
                 "Look for uses of input before or without proper validation."
    ),
    
    GeneralVulnType.INFORMATION_EXPOSURE: DistillationStrategy(
        focus_dataflow="Sensitive data → exposure point",
        keep_signals=[
            "Sensitive data access (passwords, keys, PII)",
            "Data transmission/logging",
            "Error messages containing sensitive info",
            "Memory not cleared after use",
            "Unintended data leakage paths",
        ],
        remove_signals=[
            "Safe data handling",
            "Properly redacted outputs",
            "Unrelated data processing",
        ],
        guidance="Focus on where sensitive data might be exposed - logs, error messages, network, or memory."
    ),
    
    GeneralVulnType.CRYPTOGRAPHIC_ISSUE: DistillationStrategy(
        focus_dataflow="Cryptographic operation configuration and usage",
        keep_signals=[
            "Crypto algorithm selection",
            "Key generation/storage",
            "IV/nonce handling",
            "Encryption/decryption calls",
            "Random number generation",
            "Certificate validation",
        ],
        remove_signals=[
            "Application logic using encrypted data",
            "Unrelated security operations",
        ],
        guidance="Focus on cryptographic configuration and whether it follows best practices."
    ),
    
    # =========================================================================
    # 6. Resource & Execution (4 types)
    # =========================================================================
    
    GeneralVulnType.INFINITE_LOOP: DistillationStrategy(
        focus_dataflow="Loop condition → termination guarantee",
        keep_signals=[
            "Loop constructs (while, for, do-while)",
            "Loop condition variables",
            "Variables modified inside loop",
            "Break/return statements",
            "External termination conditions",
        ],
        remove_signals=[
            "Loop body operations not affecting termination",
            "Code after the loop",
        ],
        guidance="Focus on loop termination conditions and whether they're guaranteed to eventually become true."
    ),
    
    GeneralVulnType.RECURSION_ERROR: DistillationStrategy(
        focus_dataflow="Recursive call → termination condition",
        keep_signals=[
            "Recursive function calls",
            "Base case conditions",
            "Parameters passed in recursive calls",
            "Termination conditions",
            "Stack-related checks",
        ],
        remove_signals=[
            "Processing within recursion (not affecting termination)",
            "Unrelated function calls",
        ],
        guidance="Focus on whether recursion has a proper base case and whether parameters converge toward it."
    ),
    
    GeneralVulnType.RESOURCE_EXHAUSTION: DistillationStrategy(
        focus_dataflow="Resource allocation → limit checking",
        keep_signals=[
            "Resource allocation (memory, handles, connections)",
            "Resource limit checks",
            "Loop-based allocation",
            "User-controlled allocation sizes/counts",
            "Resource cleanup",
        ],
        remove_signals=[
            "Normal resource usage",
            "Unrelated allocations",
        ],
        guidance="Focus on unbounded resource allocation, especially when controlled by user input."
    ),
    
    GeneralVulnType.RESOURCE_LEAK: DistillationStrategy(
        focus_dataflow="Resource acquisition → (missing) release",
        keep_signals=[
            "Resource acquisition (open, socket, lock)",
            "Resource handles/descriptors",
            "Release operations (close, unlock)",
            "Error paths",
            "Early returns",
            "Exception handling",
        ],
        remove_signals=[
            "Resource usage between acquire and release",
            "Unrelated resources",
        ],
        guidance="Focus on ensuring every acquisition has a corresponding release on all paths."
    ),
    
    # =========================================================================
    # 7. Fallback (2 types)
    # =========================================================================
    
    GeneralVulnType.UNKNOWN: DistillationStrategy(
        focus_dataflow="General security-relevant operations",
        keep_signals=[
            "Operations modified in the patch diff",
            "Error handling paths",
            "Input validation",
            "Security-sensitive function calls",
        ],
        remove_signals=[
            "Logging and debugging",
            "Comments and whitespace",
            "Clearly unrelated code paths",
        ],
        guidance="Without specific type information, focus on diff-modified code and security-sensitive operations."
    ),
    
    GeneralVulnType.OTHER: DistillationStrategy(
        focus_dataflow="Patch-specific changes",
        keep_signals=[
            "All diff-modified lines",
            "Direct data/control dependencies of modified lines",
        ],
        remove_signals=[
            "Code far from modified regions",
            "Logging and debugging",
        ],
        guidance="Focus on the specific changes in the patch and their immediate dependencies."
    ),
}


def get_distillation_strategy(vuln_type: GeneralVulnType) -> DistillationStrategy:
    """
    Get the distillation strategy for a specific vulnerability type.
    Falls back to UNKNOWN strategy if type not found.
    """
    return DISTILLATION_STRATEGIES.get(vuln_type, DISTILLATION_STRATEGIES[GeneralVulnType.UNKNOWN])


def get_strategy_prompt_section(vuln_type: GeneralVulnType) -> str:
    """
    Get the formatted prompt section for a specific vulnerability type.
    This should be injected into the SliceValidator prompts.
    """
    strategy = get_distillation_strategy(vuln_type)
    return strategy.to_prompt_section()
