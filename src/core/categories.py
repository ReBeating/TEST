from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Type
from abc import ABC, abstractmethod
from pydantic import BaseModel as PydanticBaseModel, Field as PydanticField, field_validator


class MajorCategory(Enum):
    """
    Four major vulnerability categories for two-stage classification.
    
    Stage 1: Classify into one of these 4 major categories
    Stage 2: Classify into specific subtype within the category
    """
    NUMERIC_DOMAIN = "Numeric-Domain"
    ACCESS_VALIDITY = "Access-Validity"
    RESOURCE_LIFECYCLE = "Resource-Lifecycle"
    CONTROL_LOGIC = "Control-Logic"
    UNKNOWN = "Unknown"
    
    @property
    def description(self) -> str:
        """Description of the major category for LLM classification."""
        descriptions = {
            MajorCategory.NUMERIC_DOMAIN: "Vulnerabilities involving numeric value corruption: overflow, underflow, truncation, divide-by-zero",
            MajorCategory.ACCESS_VALIDITY: "Vulnerabilities involving invalid memory access: buffer overflow, OOB read/write, null pointer dereference",
            MajorCategory.RESOURCE_LIFECYCLE: "Vulnerabilities involving resource lifecycle violations: UAF, double-free, memory leak, uninitialized use",
            MajorCategory.CONTROL_LOGIC: "Vulnerabilities involving logic/control flow errors: race condition, auth bypass, logic errors",
            MajorCategory.UNKNOWN: "Vulnerability type could not be determined",
        }
        return descriptions.get(self, "")
    
    @property
    def anchor_types(self) -> list:
        """The anchor types used by this major category."""
        types = {
            MajorCategory.NUMERIC_DOMAIN: [AnchorType.SOURCE, AnchorType.COMPUTATION, AnchorType.SINK],
            MajorCategory.ACCESS_VALIDITY: [AnchorType.OBJECT, AnchorType.INDEX, AnchorType.ACCESS],
            MajorCategory.RESOURCE_LIFECYCLE: [AnchorType.ALLOC, AnchorType.DEALLOC, AnchorType.USE],
            MajorCategory.CONTROL_LOGIC: [AnchorType.CRITICAL],  # Per Table 7: A = {critical}
            MajorCategory.UNKNOWN: [AnchorType.CRITICAL],
        }
        return types.get(self, [])


class AnchorLocatability(Enum):
    """
    Anchor Locatability Level - Core distinction: whether a concrete code location
    is available for static analysis.
    
    Design Philosophy: This is an implementation-driven classification that directly
    answers "Can this anchor be used for slicing and searching?"
    
    - CONCRETE: Has specific location with clear semantics, no assumptions needed
      * Sliceable: YES
      * Searchable: YES
      * Requires Assumption: NO
      * Examples: malloc(size), scanf("%d", &val), free(ptr)
      
    - ASSUMED: Has specific location but semantics need assumption
      * Sliceable: YES
      * Searchable: YES
      * Requires Assumption: YES (semantic or controllability assumptions)
      * Examples: function parameter `size` (assume controllable), process(buf) (assume internal access)
      
    - CONCEPTUAL: No specific location, purely inferred existence
      * Sliceable: NO
      * Searchable: NO
      * Requires Assumption: YES (existence assumption)
      * Examples: UAF use in another function
    """
    CONCRETE = "concrete"
    ASSUMED = "assumed"
    CONCEPTUAL = "conceptual"


class AssumptionType(Enum):
    """
    Type of assumption - Used for ASSUMED and CONCEPTUAL anchors.
    
    - CONTROLLABILITY: Assume controllability - parameters/fields may be user-controlled
    - SEMANTIC: Assume semantic behavior - function call internally performs some operation
    - EXISTENCE: Assume existence - use in other function
    - REACHABILITY: Assume reachability - critical operation can be accessed without authorization
    """
    CONTROLLABILITY = "controllability"
    SEMANTIC = "semantic"
    EXISTENCE = "existence"
    REACHABILITY = "reachability"


class DependencyType(Enum):
    """
    Dependency type between anchors in the vulnerability chain.
    
    Based on VERDICT paper Table 7:
    - DATA (→d): Data dependency - value flows from source to target
    - TEMPORAL (→t): Temporal dependency - target executes after source
    - CONTROL (→c): Control dependency - target's execution depends on source's condition
    """
    DATA = "data"           # →d: data flows from A to B
    TEMPORAL = "temporal"   # →t: B executes after A
    CONTROL = "control"     # →c: B's execution depends on A's condition


@dataclass
class ChainLink:
    """
    A single link in the anchor chain, representing dependency between two anchors.
    
    Example: src →d comp means "comp has data dependency on src"
    """
    source: 'AnchorType'
    target: 'AnchorType'
    dependency: DependencyType
    is_optional: bool = False
    
    def __str__(self) -> str:
        dep_symbol = {"data": "→d", "temporal": "→t", "control": "→c"}[self.dependency.value]
        opt = "?" if self.is_optional else ""
        return f"{self.source.value}{opt} {dep_symbol} {self.target.value}"


@dataclass
class ViolationPredicate:
    """
    VIOL(P) - Predicate asserting a semantically erroneous state on path P.
    
    Each vulnerability type has specific violation conditions:
    - Integer Overflow: result exceeds type maximum
    - OOB Write: index exceeds object boundary
    - UAF: use occurs after deallocation
    - etc.
    """
    name: str
    description: str
    formal_condition: str  # Formal/semi-formal condition description
    
    def __str__(self) -> str:
        return f"VIOL({self.name}): {self.description}"


@dataclass
class MitigationPattern:
    """
    Pattern that can neutralize a vulnerability (used in ¬MITIGATED(P)).
    
    Common mitigation patterns:
    - Bounds check before access
    - NULL check before dereference
    - Overflow check before arithmetic
    - Sanitization after free
    """
    name: str
    description: str
    anchor_type: Optional['AnchorType'] = None  # Which anchor this mitigates
    code_patterns: List[str] = field(default_factory=list)  # Example code patterns
    
    def __str__(self) -> str:
        return f"MITIGATE({self.name}): {self.description}"


@dataclass
class VulnerabilityConstraint:
    """
    Complete constraint specification for a vulnerability type.
    
    C = ∃ P: CHAIN(P) ∧ VIOL(P) ∧ ¬MITIGATED(P)
    
    Where:
    - CHAIN(P): Every anchor maps to a concrete statement on path P with required dependencies
    - VIOL(P): Category-specific predicate asserting semantically erroneous state
    - MITIGATED(P): Path P contains operation that neutralizes the violation
    """
    chain: List[ChainLink]
    violation: ViolationPredicate
    mitigations: List[MitigationPattern] = field(default_factory=list)
    
    def chain_str(self) -> str:
        """Return string representation of the chain.
        
        Examples:
        - Linear: 'src →d comp →d sink'
        - Converging: 'obj →d access ←d src' (for buffer overflow with two sources)
        """
        if not self.chain:
            return ""
        
        dep_symbols = {"data": "→d", "temporal": "→t", "control": "→c"}
        
        # Group links by target to detect converging patterns
        target_groups: Dict[str, List[ChainLink]] = {}
        for link in self.chain:
            target = link.target.value
            if target not in target_groups:
                target_groups[target] = []
            target_groups[target].append(link)
        
        # Build chain representation
        parts = []
        processed_sources = set()
        
        for link in self.chain:
            source = link.source.value
            target = link.target.value
            dep = dep_symbols[link.dependency.value]
            opt = "?" if link.is_optional else ""
            
            if source not in processed_sources:
                if parts and parts[-1] == target:
                    # Converging pattern: target ←d source
                    parts.append(f"←{dep[1:]}{opt}")
                    parts.append(source)
                else:
                    # Normal pattern: source →d target
                    if parts:
                        parts.append(f"{dep}{opt}")
                    else:
                        parts.append(source)
                        parts.append(f"{dep}{opt}")
                    parts.append(target)
                processed_sources.add(source)
        
        return " ".join(parts)
    
    def __str__(self) -> str:
        return f"C = ∃P: CHAIN({self.chain_str()}) ∧ {self.violation} ∧ ¬MITIGATED"

class AnchorType(Enum):
    """
    Standardized anchor types across all vulnerability categories.
    Each type has associated identification guides and examples for LLM analysis.
    """
    # === Numeric-Domain Violation ===
    SOURCE = "source"                        # Numeric origin
    COMPUTATION = "computation"              # Arithmetic operation / type conversion
    SINK = "sink"                           # Sensitive usage point
    
    # === Access-Validity Violation ===
    OBJECT = "object"                        # Accessed entity
    INDEX = "index"                          # Offset / index computation (optional)
    ACCESS = "access"                        # Access operation
    
    # === Resource-Lifecycle Violation ===
    ALLOC = "alloc"                         # Resource allocation
    DEALLOC = "dealloc"                     # Resource release (optional)
    USE = "use"                             # Resource usage (optional)
    EXIT = "exit"                           # Function/program exit point (for leak detection)
    
    # === Control-Logic Violation ===
    CRITICAL = "critical"                    # Sensitive operation that needs protection
    
    @property
    def identification_guide(self) -> str:
        """LLM guide for identifying this anchor type in code."""
        guides = {
            AnchorType.SOURCE: "Look for user input functions (scanf, read, recv, fgets) or function parameters that receive external data",
            AnchorType.COMPUTATION: "Look for arithmetic operations (+, -, *, /), type casts, or assignments that transform values",
            AnchorType.SINK: "Look for sensitive usages: malloc/alloc size, array index, loop bound, memcpy length",
            AnchorType.OBJECT: "Look for buffer/array declarations, malloc returns, or pointer variables being accessed",
            AnchorType.INDEX: "Look for array subscripts [i], pointer arithmetic (ptr + offset), or offset calculations",
            AnchorType.ACCESS: "Look for memory read/write: array[i], *ptr, memcpy, strcpy, ptr->field",
            AnchorType.ALLOC: "Look for memory allocation: malloc, calloc, realloc, new, or custom allocators",
            AnchorType.DEALLOC: "Look for deallocation: free, delete, or custom deallocators",
            AnchorType.USE: "Look for pointer dereference, member access, or function calls using the resource",
            AnchorType.EXIT: "Look for function return points, program exit, or paths that leave scope without deallocation",
            AnchorType.CRITICAL: "Look for security-sensitive operations: privilege changes, file access, crypto operations, shared resource access without synchronization",
        }
        return guides.get(self, "")
    
    @property
    def examples(self) -> list:
        """Code examples for this anchor type."""
        examples = {
            AnchorType.SOURCE: ["scanf(\"%d\", &size)", "size = atoi(argv[1])", "read(fd, buf, len)"],
            AnchorType.COMPUTATION: ["result = a + b", "size = (int)large_value", "offset = base * factor"],
            AnchorType.SINK: ["malloc(size)", "buffer[index]", "memcpy(dst, src, len)"],
            AnchorType.OBJECT: ["char buf[100]", "ptr = malloc(size)", "struct data *obj"],
            AnchorType.INDEX: ["buf[i]", "ptr + offset", "arr[row * cols + col]"],
            AnchorType.ACCESS: ["buf[i] = val", "*ptr = data", "memcpy(dst, src, n)"],
            AnchorType.ALLOC: ["ptr = malloc(size)", "obj = new Object()", "buf = calloc(n, size)"],
            AnchorType.DEALLOC: ["free(ptr)", "delete obj", "munmap(addr, len)"],
            AnchorType.USE: ["ptr->field", "*ptr", "func(ptr)"],
            AnchorType.EXIT: ["return;", "return val;", "exit(0)", "goto cleanup;"],
            AnchorType.CRITICAL: ["setuid(0)", "open(path, O_RDWR)", "shared_data++", "list_add(&node->list, &head)"],
        }
        return examples.get(self, [])

class Anchor(PydanticBaseModel):
    """
    Anchor: A critical semantic role in a vulnerability pattern.
    
    Anchors are key code locations in vulnerability patterns, used for:
    1. PDG slicing during feature extraction
    2. Code matching during candidate search
    3. Dependency path verification between anchors
    
    This class unifies the old AnchorItem (extraction layer) and Anchor (design layer).
    It is a Pydantic BaseModel to support LangChain's with_structured_output().
    """
    
    # === Type Definition ===
    type: AnchorType = PydanticField(description="AnchorType value (e.g., 'alloc', 'dealloc', 'use', 'source', 'computation', 'sink')")
    description: str = PydanticField(default="", description="Description of the anchor's role")
    is_optional: bool = PydanticField(default=False, description="Whether this anchor is optional in the chain")
    
    # === Locatability Level ===
    locatability: AnchorLocatability = PydanticField(default=AnchorLocatability.CONCRETE, description="Locatability level of the anchor")
    
    # === Assumption Related (for ASSUMED and CONCEPTUAL) ===
    assumption_type: Optional[AssumptionType] = PydanticField(default=None, description="Type of assumption for ASSUMED/CONCEPTUAL anchors")
    assumption_rationale: Optional[str] = PydanticField(default=None, description="Rationale for the assumption")
    
    # === Code Location Info (only valid for CONCRETE and ASSUMED) ===
    file_path: Optional[str] = PydanticField(default=None, description="File path where anchor is located")
    func_name: Optional[str] = PydanticField(default=None, description="Function name containing the anchor")
    line_number: Optional[int] = PydanticField(default=None, description="Absolute line number of the Anchor")
    code_snippet: Optional[str] = PydanticField(default=None, description="Code content of the Anchor")
    variable_names: List[str] = PydanticField(default_factory=list, description="Variable names involved in this anchor")
    
    # === Fields migrated from AnchorItem ===
    reasoning: str = PydanticField(default="", description="Why this is an anchor of this type (Agent reasoning)")
    scope: Optional[str] = PydanticField(default=None, description="Scope type: 'local', 'call_site', or 'inter_procedural'")
    cross_function_info: Optional[dict] = PydanticField(default=None, description="Cross-function info (when scope is call_site or inter_procedural)")
    
    # === Pydantic Config ===
    model_config = {"use_enum_values": False}
    
    # === Field Validators for LLM string → enum conversion ===
    @field_validator('type', mode='before')
    @classmethod
    def _coerce_type(cls, v):
        """Convert string to AnchorType enum if needed."""
        if isinstance(v, str):
            return AnchorType(v)
        return v
    
    @field_validator('locatability', mode='before')
    @classmethod
    def _coerce_locatability(cls, v):
        """Convert string to AnchorLocatability enum if needed."""
        if isinstance(v, str):
            return AnchorLocatability(v)
        return v
    
    @field_validator('assumption_type', mode='before')
    @classmethod
    def _coerce_assumption_type(cls, v):
        """Convert string to AssumptionType enum if needed."""
        if v is None:
            return v
        if isinstance(v, str):
            return AssumptionType(v)
        return v
    
    # === Derived Properties ===
    @property
    def identification_guide(self) -> str:
        """Get LLM identification guide from anchor type."""
        return self.type.identification_guide
    
    @property
    def examples(self) -> list:
        """Get code examples from anchor type."""
        return self.type.examples
    
    def is_locatable(self) -> bool:
        """
        Check if this anchor can be used for slicing and searching.
        
        Returns:
            True if the anchor has a concrete location (CONCRETE or ASSUMED)
        """
        return self.locatability in [
            AnchorLocatability.CONCRETE,
            AnchorLocatability.ASSUMED
        ]
    
    def has_code_location(self) -> bool:
        """
        Check if this anchor has a concrete code location.
        
        Returns:
            True if the anchor has a line number and is locatable
        """
        return (
            self.line_number is not None and
            self.is_locatable()
        )
    
    def requires_assumption(self) -> bool:
        """
        Check if this anchor requires assumptions.
        
        Returns:
            True if the anchor requires assumptions (ASSUMED or CONCEPTUAL)
        """
        return self.locatability in [
            AnchorLocatability.ASSUMED,
            AnchorLocatability.CONCEPTUAL
        ]

class VulnerabilityCategory(ABC):
    """
    Abstract base class for all vulnerability categories.
    
    Each category defines:
    - anchors: Key semantic locations in the vulnerability pattern
    - constraint: CHAIN, VIOL, MITIGATED specification
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """The display name of the category."""
        pass
    
    @property
    @abstractmethod
    def major_category(self) -> MajorCategory:
        """The major category this vulnerability belongs to."""
        pass
        
    @property
    @abstractmethod
    def description(self) -> str:
        """A brief description of the vulnerability category."""
        pass
        
    @property
    @abstractmethod
    def anchors(self) -> List[Anchor]:
        """
        Returns the list of standard anchors associated with this category.
        These anchors define the structural pattern of the vulnerability.
        """
        pass
    
    @property
    @abstractmethod
    def constraint(self) -> VulnerabilityConstraint:
        """
        Returns the vulnerability constraint specification.
        
        C = ∃ P: CHAIN(P) ∧ VIOL(P) ∧ ¬MITIGATED(P)
        """
        pass

    def get_anchor_by_type(self, anchor_type: AnchorType) -> Optional[Anchor]:
        """Helper to find an anchor definition by its type."""
        for anchor in self.anchors:
            if anchor.type == anchor_type:
                return anchor
        return None

# --- 1. Numeric-Domain Violation ---

class NumericDomainViolation(VulnerabilityCategory):
    """
    Base class for numeric domain violations (Integer Overflow, Underflow, etc.).
    Anchor chain: src →d comp →d sink
    """
    
    @property
    def major_category(self) -> MajorCategory:
        return MajorCategory.NUMERIC_DOMAIN
    
    @property
    def anchors(self) -> List[Anchor]:
        """Default anchor chain for numeric domain violations."""
        return [
            Anchor(type=AnchorType.SOURCE, description="The origin of the numeric value"),
            Anchor(type=AnchorType.COMPUTATION, description="The arithmetic/type operation"),
            Anchor(type=AnchorType.SINK, description="The sensitive usage of the result")
        ]
    
    @property
    def constraint(self) -> VulnerabilityConstraint:
        """Default constraint: src →d comp →d sink with generic violation."""
        return VulnerabilityConstraint(
            chain=[
                ChainLink(AnchorType.SOURCE, AnchorType.COMPUTATION, DependencyType.DATA),
                ChainLink(AnchorType.COMPUTATION, AnchorType.SINK, DependencyType.DATA),
            ],
            violation=ViolationPredicate(
                name="numeric_error",
                description="Computation result is semantically invalid",
                formal_condition="result ∉ valid_range(type)"
            ),
            mitigations=[
                MitigationPattern(
                    name="range_check",
                    description="Check if operands/result are within valid range before computation",
                    anchor_type=AnchorType.COMPUTATION,
                    code_patterns=["if (a > MAX - b)", "if (__builtin_add_overflow(a, b, &result))"]
                ),
                MitigationPattern(
                    name="safe_math",
                    description="Use safe arithmetic functions that detect overflow",
                    anchor_type=AnchorType.COMPUTATION,
                    code_patterns=["safe_add(a, b)", "checked_multiply(a, b)"]
                ),
            ]
        )

class IntegerOverflow(NumericDomainViolation):
    @property
    def name(self) -> str:
        return "Integer Overflow"
        
    @property
    def description(self) -> str:
        return "Arithmetic operation results in a value exceeding the maximum representable value."
    
    @property
    def constraint(self) -> VulnerabilityConstraint:
        """CHAIN: src →d comp →d sink, VIOL: result > MAX"""
        return VulnerabilityConstraint(
            chain=[
                ChainLink(AnchorType.SOURCE, AnchorType.COMPUTATION, DependencyType.DATA),
                ChainLink(AnchorType.COMPUTATION, AnchorType.SINK, DependencyType.DATA),
            ],
            violation=ViolationPredicate(
                name="overflow",
                description="Arithmetic result exceeds the maximum value for the type",
                formal_condition="result > TYPE_MAX"
            ),
            mitigations=[
                MitigationPattern(
                    name="overflow_check",
                    description="Check for potential overflow before arithmetic operation",
                    anchor_type=AnchorType.COMPUTATION,
                    code_patterns=[
                        "if (a > INT_MAX - b) return ERROR;",
                        "if (__builtin_add_overflow(a, b, &result))",
                        "if (a > SIZE_MAX / b)"
                    ]
                ),
                MitigationPattern(
                    name="type_promotion",
                    description="Use larger type for intermediate computation",
                    anchor_type=AnchorType.COMPUTATION,
                    code_patterns=["(int64_t)a + b", "(size_t)a * b"]
                ),
                MitigationPattern(
                    name="saturating_arithmetic",
                    description="Use saturating arithmetic that clamps to max/min instead of wrapping",
                    anchor_type=AnchorType.COMPUTATION,
                    code_patterns=[
                        "result = (a > INT_MAX - b) ? INT_MAX : a + b;",
                        "clamp(a + b, 0, INT_MAX)"
                    ]
                ),
            ]
        )

class IntegerUnderflow(NumericDomainViolation):
    @property
    def name(self) -> str:
        return "Integer Underflow"
        
    @property
    def description(self) -> str:
        return "Arithmetic operation results in a value below the minimum representable value."
    
    @property
    def constraint(self) -> VulnerabilityConstraint:
        """CHAIN: src →d comp →d sink, VIOL: result < MIN"""
        return VulnerabilityConstraint(
            chain=[
                ChainLink(AnchorType.SOURCE, AnchorType.COMPUTATION, DependencyType.DATA),
                ChainLink(AnchorType.COMPUTATION, AnchorType.SINK, DependencyType.DATA),
            ],
            violation=ViolationPredicate(
                name="underflow",
                description="Arithmetic result is below the minimum value for the type",
                formal_condition="result < TYPE_MIN"
            ),
            mitigations=[
                MitigationPattern(
                    name="underflow_check",
                    description="Check for potential underflow before subtraction",
                    anchor_type=AnchorType.COMPUTATION,
                    code_patterns=[
                        "if (a < INT_MIN + b) return ERROR;",
                        "if (__builtin_sub_overflow(a, b, &result))",
                        "if (b > a) return ERROR;  // for unsigned"
                    ]
                ),
                MitigationPattern(
                    name="unsigned_check",
                    description="Check if result would wrap for unsigned subtraction",
                    anchor_type=AnchorType.COMPUTATION,
                    code_patterns=[
                        "if (b > a) { len = 0; } else { len = a - b; }",
                        "size_t result = (b <= a) ? a - b : 0;"
                    ]
                ),
            ]
        )

class DivideByZero(NumericDomainViolation):
    @property
    def name(self) -> str:
        return "Divide By Zero"
        
    @property
    def description(self) -> str:
        return "Division operation where the divisor is zero."
    
    @property
    def constraint(self) -> VulnerabilityConstraint:
        """CHAIN: src →d comp →d sink, VIOL: divisor == 0"""
        return VulnerabilityConstraint(
            chain=[
                ChainLink(AnchorType.SOURCE, AnchorType.COMPUTATION, DependencyType.DATA),
                ChainLink(AnchorType.COMPUTATION, AnchorType.SINK, DependencyType.DATA),
            ],
            violation=ViolationPredicate(
                name="div_by_zero",
                description="Division operation with zero divisor",
                formal_condition="divisor == 0"
            ),
            mitigations=[
                MitigationPattern(
                    name="zero_check",
                    description="Check if divisor is zero before division",
                    anchor_type=AnchorType.COMPUTATION,
                    code_patterns=[
                        "if (divisor == 0) return ERROR;",
                        "if (!divisor) goto error;",
                        "divisor = divisor ? divisor : 1;"
                    ]
                ),
            ]
        )

class TypeConfusion(NumericDomainViolation):
    @property
    def name(self) -> str:
        return "Type Confusion"
        
    @property
    def description(self) -> str:
        return "Using a value with incorrect type interpretation (signed/unsigned confusion, type casting errors)."
    
    @property
    def constraint(self) -> VulnerabilityConstraint:
        """CHAIN: src →d comp →d sink, VIOL: type_of(value) != expected_type"""
        return VulnerabilityConstraint(
            chain=[
                ChainLink(AnchorType.SOURCE, AnchorType.COMPUTATION, DependencyType.DATA),
                ChainLink(AnchorType.COMPUTATION, AnchorType.SINK, DependencyType.DATA),
            ],
            violation=ViolationPredicate(
                name="type_mismatch",
                description="Value interpreted with wrong type (e.g., signed as unsigned)",
                formal_condition="type_of(value) != expected_type ∨ sign_of(value) != expected_sign"
            ),
            mitigations=[
                MitigationPattern(
                    name="explicit_cast_check",
                    description="Validate value range before type cast",
                    anchor_type=AnchorType.COMPUTATION,
                    code_patterns=[
                        "if (signed_val < 0) return ERROR;",
                        "if ((unsigned)val != val) return ERROR;",
                        "assert(val >= 0 && val <= UINT_MAX);"
                    ]
                ),
            ]
        )

# --- 2. Access-Validity Violation ---

class AccessValidityViolation(VulnerabilityCategory):
    """
    Base class for memory access violations (OOB Read/Write, NPD, etc.).
    Anchor chain: obj →d access ←d idx (obj and idx both contribute to access)
    """
    
    @property
    def major_category(self) -> MajorCategory:
        return MajorCategory.ACCESS_VALIDITY
    
    @property
    def anchors(self) -> List[Anchor]:
        """Default anchor chain for access validity violations."""
        return [
            Anchor(type=AnchorType.OBJECT, description="The buffer/pointer object"),
            Anchor(type=AnchorType.INDEX, description="The index or offset for access", is_optional=True),
            Anchor(type=AnchorType.ACCESS, description="The memory access operation")
        ]
    
    @property
    def constraint(self) -> VulnerabilityConstraint:
        """Default constraint: obj →d access ←d idx with generic access violation."""
        return VulnerabilityConstraint(
            chain=[
                ChainLink(AnchorType.OBJECT, AnchorType.ACCESS, DependencyType.DATA),
                ChainLink(AnchorType.INDEX, AnchorType.ACCESS, DependencyType.DATA, is_optional=True),
            ],
            violation=ViolationPredicate(
                name="access_error",
                description="Memory access is invalid",
                formal_condition="access_addr ∉ valid_region(object)"
            ),
            mitigations=[
                MitigationPattern(
                    name="bounds_check",
                    description="Check if index is within valid bounds before access",
                    anchor_type=AnchorType.ACCESS,
                    code_patterns=["if (idx >= size) return ERROR;", "if (idx < 0 || idx >= len)"]
                ),
            ]
        )

class OutOfBoundsWrite(AccessValidityViolation):
    @property
    def name(self) -> str:
        return "Out-of-bounds Write"
        
    @property
    def description(self) -> str:
        return "Writing data outside the allocated buffer boundary."
    
    @property
    def constraint(self) -> VulnerabilityConstraint:
        """CHAIN: obj →d access ←d idx, VIOL: idx >= size(obj) or idx < 0"""
        return VulnerabilityConstraint(
            chain=[
                ChainLink(AnchorType.OBJECT, AnchorType.ACCESS, DependencyType.DATA),
                ChainLink(AnchorType.INDEX, AnchorType.ACCESS, DependencyType.DATA, is_optional=True),
            ],
            violation=ViolationPredicate(
                name="oob_write",
                description="Write operation accesses memory outside allocated buffer",
                formal_condition="idx >= size(obj) ∨ idx < 0"
            ),
            mitigations=[
                MitigationPattern(
                    name="upper_bound_check",
                    description="Check if index is less than buffer size",
                    anchor_type=AnchorType.ACCESS,
                    code_patterns=[
                        "if (idx >= sizeof(buf)) return ERROR;",
                        "if (offset + len > size) return -EINVAL;",
                        "assert(idx < capacity);"
                    ]
                ),
                MitigationPattern(
                    name="lower_bound_check",
                    description="Check if index is non-negative (for signed indices)",
                    anchor_type=AnchorType.ACCESS,
                    code_patterns=[
                        "if (idx < 0) return ERROR;",
                        "if (offset < 0) return -EINVAL;"
                    ]
                ),
                MitigationPattern(
                    name="length_check",
                    description="Check copy/write length against buffer capacity",
                    anchor_type=AnchorType.ACCESS,
                    code_patterns=[
                        "if (offset + len > size) return -EINVAL;",
                        "n = min(n, sizeof(buf) - offset);",
                        "if (count > buf_size - pos) count = buf_size - pos;"
                    ]
                ),
            ]
        )

class OutOfBoundsRead(AccessValidityViolation):
    @property
    def name(self) -> str:
        return "Out-of-bounds Read"
        
    @property
    def description(self) -> str:
        return "Reading data outside the allocated buffer boundary."
    
    @property
    def constraint(self) -> VulnerabilityConstraint:
        """CHAIN: obj →d access ←d idx, VIOL: idx >= size(obj) or idx < 0"""
        return VulnerabilityConstraint(
            chain=[
                ChainLink(AnchorType.OBJECT, AnchorType.ACCESS, DependencyType.DATA),
                ChainLink(AnchorType.INDEX, AnchorType.ACCESS, DependencyType.DATA, is_optional=True),
            ],
            violation=ViolationPredicate(
                name="oob_read",
                description="Read operation accesses memory outside allocated buffer",
                formal_condition="idx >= size(obj) ∨ idx < 0"
            ),
            mitigations=[
                MitigationPattern(
                    name="bounds_check",
                    description="Check if index is within valid range before read",
                    anchor_type=AnchorType.ACCESS,
                    code_patterns=[
                        "if (idx >= len) return ERROR;",
                        "if (offset < 0 || offset >= size) return NULL;",
                        "BUG_ON(idx >= ARRAY_SIZE(arr));"
                    ]
                ),
            ]
        )

class BufferOverflow(AccessValidityViolation):
    """
    Buffer overflow via copy/write operations (memcpy, strcpy, read, etc.).
    Unlike OOB Write which involves explicit index, this targets length-based overflows.
    """
    @property
    def name(self) -> str:
        return "Buffer Overflow"
        
    @property
    def description(self) -> str:
        return "Writing data beyond buffer boundary via copy/write operations with unchecked length."
    
    @property
    def anchors(self) -> List[Anchor]:
        """Override: No INDEX anchor, uses length parameter in the access operation."""
        return [
            Anchor(type=AnchorType.OBJECT, description="The destination buffer"),
            Anchor(type=AnchorType.SOURCE, description="The source data or length parameter", is_optional=True),
            Anchor(type=AnchorType.ACCESS, description="The copy/write operation (memcpy, strcpy, read, etc.)")
        ]
    
    @property
    def constraint(self) -> VulnerabilityConstraint:
        """CHAIN: obj →d access ←d src, VIOL: copy_len > size(obj)"""
        return VulnerabilityConstraint(
            chain=[
                ChainLink(AnchorType.OBJECT, AnchorType.ACCESS, DependencyType.DATA),
                ChainLink(AnchorType.SOURCE, AnchorType.ACCESS, DependencyType.DATA, is_optional=True),
            ],
            violation=ViolationPredicate(
                name="buffer_overflow",
                description="Copy/write operation exceeds destination buffer capacity",
                formal_condition="copy_length > size(dest_buffer) ∨ src_length > size(dest_buffer)"
            ),
            mitigations=[
                MitigationPattern(
                    name="length_check",
                    description="Check if copy length is within buffer capacity",
                    anchor_type=AnchorType.ACCESS,
                    code_patterns=[
                        "if (len > sizeof(buf)) return ERROR;",
                        "if (n > buf_size) n = buf_size;",
                        "if (count > size - offset) return -EINVAL;"
                    ]
                ),
                MitigationPattern(
                    name="safe_copy_function",
                    description="Use safe copy functions with explicit size limit",
                    anchor_type=AnchorType.ACCESS,
                    code_patterns=[
                        "strncpy(dst, src, sizeof(dst) - 1);",
                        "strlcpy(dst, src, sizeof(dst));",
                        "snprintf(buf, sizeof(buf), \"%s\", src);",
                        "memcpy_s(dst, dst_size, src, count);"
                    ]
                ),
                MitigationPattern(
                    name="null_termination",
                    description="Ensure null termination for string operations",
                    anchor_type=AnchorType.ACCESS,
                    code_patterns=[
                        "dst[sizeof(dst) - 1] = '\\0';",
                        "buf[len] = 0;"
                    ]
                ),
            ]
        )

class NullPointerDereference(AccessValidityViolation):
    @property
    def name(self) -> str:
        return "Null Pointer Dereference"
        
    @property
    def description(self) -> str:
        return "Dereferencing a pointer that is NULL."
    
    @property
    def anchors(self) -> List[Anchor]:
        """Override: NPD has no INDEX anchor (obj → access only)."""
        return [
            Anchor(type=AnchorType.OBJECT, description="The pointer origin (e.g., malloc return, function arg)"),
            Anchor(type=AnchorType.ACCESS, description="The dereference operation")
        ]
    
    @property
    def constraint(self) -> VulnerabilityConstraint:
        """CHAIN: obj →d access, VIOL: obj == NULL"""
        return VulnerabilityConstraint(
            chain=[
                ChainLink(AnchorType.OBJECT, AnchorType.ACCESS, DependencyType.DATA),
            ],
            violation=ViolationPredicate(
                name="null_deref",
                description="Pointer dereference when pointer is NULL",
                formal_condition="obj == NULL"
            ),
            mitigations=[
                MitigationPattern(
                    name="null_check",
                    description="Check if pointer is NULL before dereference",
                    anchor_type=AnchorType.ACCESS,
                    code_patterns=[
                        "if (ptr == NULL) return ERROR;",
                        "if (!ptr) goto err;",
                        "BUG_ON(!ptr);",
                        "ptr = ptr ?: default_ptr;"
                    ]
                ),
            ]
        )

# --- 3. Resource-Lifecycle Violation ---

class ResourceLifecycleViolation(VulnerabilityCategory):
    """
    Base class for resource management errors (UAF, Double Free, Leaks).
    Anchor chain: alloc →d dealloc →t use
    """
    
    @property
    def major_category(self) -> MajorCategory:
        return MajorCategory.RESOURCE_LIFECYCLE
    
    @property
    def anchors(self) -> List[Anchor]:
        """Default anchor chain for resource lifecycle violations."""
        return [
            Anchor(type=AnchorType.ALLOC, description="The resource allocation"),
            Anchor(type=AnchorType.DEALLOC, description="The resource deallocation"),
            Anchor(type=AnchorType.USE, description="The resource usage")
        ]
    
    @property
    def constraint(self) -> VulnerabilityConstraint:
        """Default constraint for resource lifecycle violations."""
        return VulnerabilityConstraint(
            chain=[
                ChainLink(AnchorType.ALLOC, AnchorType.DEALLOC, DependencyType.DATA),
                ChainLink(AnchorType.DEALLOC, AnchorType.USE, DependencyType.TEMPORAL),
            ],
            violation=ViolationPredicate(
                name="lifecycle_error",
                description="Resource used in invalid lifecycle state",
                formal_condition="state(resource) ∉ valid_states"
            ),
            mitigations=[
                MitigationPattern(
                    name="null_after_free",
                    description="Set pointer to NULL after free to prevent reuse",
                    anchor_type=AnchorType.DEALLOC,
                    code_patterns=["free(ptr); ptr = NULL;", "kfree(ptr); ptr = NULL;"]
                ),
            ]
        )

class UseAfterFree(ResourceLifecycleViolation):
    @property
    def name(self) -> str:
        return "Use After Free"
        
    @property
    def description(self) -> str:
        return "Using memory after it has been freed."
    
    @property
    def constraint(self) -> VulnerabilityConstraint:
        """CHAIN: alloc →d dealloc →t use, VIOL: use after dealloc"""
        return VulnerabilityConstraint(
            chain=[
                ChainLink(AnchorType.ALLOC, AnchorType.DEALLOC, DependencyType.DATA),
                ChainLink(AnchorType.DEALLOC, AnchorType.USE, DependencyType.TEMPORAL),
            ],
            violation=ViolationPredicate(
                name="use_after_free",
                description="Memory is accessed after being freed",
                formal_condition="time(use) > time(dealloc) ∧ same_pointer(use, dealloc)"
            ),
            mitigations=[
                MitigationPattern(
                    name="null_after_free",
                    description="Set pointer to NULL immediately after free",
                    anchor_type=AnchorType.DEALLOC,
                    code_patterns=[
                        "free(ptr); ptr = NULL;",
                        "kfree(obj); obj = NULL;",
                        "delete p; p = nullptr;"
                    ]
                ),
                MitigationPattern(
                    name="reference_counting",
                    description="Use reference counting to track pointer lifetime",
                    anchor_type=AnchorType.USE,
                    code_patterns=[
                        "if (refcount_dec_and_test(&obj->ref))",
                        "kref_put(&obj->kref, release_fn)"
                    ]
                ),
                MitigationPattern(
                    name="null_check_before_use",
                    description="Check if pointer is NULL before use",
                    anchor_type=AnchorType.USE,
                    code_patterns=[
                        "if (ptr == NULL) return;",
                        "if (!obj) return -EINVAL;"
                    ]
                ),
            ]
        )

class DoubleFree(ResourceLifecycleViolation):
    @property
    def name(self) -> str:
        return "Double Free"
        
    @property
    def description(self) -> str:
        return "Freeing the same memory address twice."
    
    @property
    def anchors(self) -> List[Anchor]:
        """Override: Double Free has two DEALLOC anchors."""
        return [
            Anchor(type=AnchorType.ALLOC, description="The original allocation"),
            Anchor(type=AnchorType.DEALLOC, description="The first free operation"),
            Anchor(type=AnchorType.DEALLOC, description="The second free operation")
        ]
    
    @property
    def constraint(self) -> VulnerabilityConstraint:
        """CHAIN: alloc →d dealloc1 →t dealloc2, VIOL: double dealloc"""
        return VulnerabilityConstraint(
            chain=[
                ChainLink(AnchorType.ALLOC, AnchorType.DEALLOC, DependencyType.DATA),
                ChainLink(AnchorType.DEALLOC, AnchorType.DEALLOC, DependencyType.TEMPORAL),
            ],
            violation=ViolationPredicate(
                name="double_free",
                description="Same memory address is freed twice",
                formal_condition="time(dealloc2) > time(dealloc1) ∧ same_pointer(dealloc1, dealloc2)"
            ),
            mitigations=[
                MitigationPattern(
                    name="null_after_free",
                    description="Set pointer to NULL after free to make second free safe",
                    anchor_type=AnchorType.DEALLOC,
                    code_patterns=[
                        "free(ptr); ptr = NULL;",
                        "if (ptr) { free(ptr); ptr = NULL; }"
                    ]
                ),
                MitigationPattern(
                    name="freed_flag",
                    description="Track freed state with a flag",
                    anchor_type=AnchorType.DEALLOC,
                    code_patterns=[
                        "if (!obj->freed) { free(obj); obj->freed = true; }",
                        "BUG_ON(obj->state == FREED);"
                    ]
                ),
            ]
        )

class MemoryLeak(ResourceLifecycleViolation):
    @property
    def name(self) -> str:
        return "Memory Leak"
        
    @property
    def description(self) -> str:
        return "Allocated memory is not freed before reaching exit point."
    
    @property
    def anchors(self) -> List[Anchor]:
        """Override: alloc → exit (allocation reaches exit without dealloc)."""
        return [
            Anchor(type=AnchorType.ALLOC, description="The memory allocation"),
            Anchor(type=AnchorType.EXIT, description="Exit point reached without deallocation")
        ]
    
    @property
    def constraint(self) -> VulnerabilityConstraint:
        """CHAIN: alloc →t exit, VIOL: no dealloc on path"""
        return VulnerabilityConstraint(
            chain=[
                ChainLink(AnchorType.ALLOC, AnchorType.EXIT, DependencyType.TEMPORAL),
            ],
            violation=ViolationPredicate(
                name="memory_leak",
                description="Allocated memory not freed before exit",
                formal_condition="∀ path(alloc, exit): ¬∃ dealloc ∈ path"
            ),
            mitigations=[
                MitigationPattern(
                    name="goto_cleanup",
                    description="Use goto cleanup pattern for error handling",
                    anchor_type=AnchorType.EXIT,
                    code_patterns=[
                        "goto err_free;",
                        "goto out_free;",
                        "goto cleanup;"
                    ]
                ),
                MitigationPattern(
                    name="free_before_return",
                    description="Free allocated memory before return",
                    anchor_type=AnchorType.EXIT,
                    code_patterns=[
                        "free(ptr); return ret;",
                        "kfree(obj); return -ENOMEM;"
                    ]
                ),
            ]
        )

class UninitializedUse(AccessValidityViolation):
    """
    Uninitialized Use belongs to Access-Validity (per Table 7).
    """
    @property
    def name(self) -> str:
        return "Uninitialized Use"
        
    @property
    def description(self) -> str:
        return "Using a variable before it has been initialized."
    
    @property
    def anchors(self) -> List[Anchor]:
        """Override: No INDEX anchor (obj → access only)."""
        return [
            Anchor(type=AnchorType.OBJECT, description="The uninitialized variable/object"),
            Anchor(type=AnchorType.ACCESS, description="The read operation on the uninitialized variable")
        ]
    
    @property
    def constraint(self) -> VulnerabilityConstraint:
        """CHAIN: obj →d access, VIOL: obj not initialized"""
        return VulnerabilityConstraint(
            chain=[
                ChainLink(AnchorType.OBJECT, AnchorType.ACCESS, DependencyType.DATA),
            ],
            violation=ViolationPredicate(
                name="uninit_use",
                description="Variable read before initialization",
                formal_condition="¬initialized(obj) ∧ read(access, obj)"
            ),
            mitigations=[
                MitigationPattern(
                    name="zero_init",
                    description="Initialize variable to zero/NULL at declaration",
                    anchor_type=AnchorType.OBJECT,
                    code_patterns=[
                        "int val = 0;",
                        "char *ptr = NULL;",
                        "memset(&obj, 0, sizeof(obj));"
                    ]
                ),
                MitigationPattern(
                    name="explicit_init",
                    description="Explicitly initialize before use",
                    anchor_type=AnchorType.OBJECT,
                    code_patterns=[
                        "struct foo bar = { .field = 0 };",
                        "obj = kzalloc(sizeof(*obj), GFP_KERNEL);"
                    ]
                ),
            ]
        )

# --- 4. Control-Logic Violation ---

class ControlLogicViolation(VulnerabilityCategory):
    """
    Base class for logic errors (Race Conditions, Missing Authorization, etc.).
    
    Per VERDICT paper Table 7:
    - Anchor: A = {critical} - the sensitive operation requiring protection
    - Chain: P → critical (feasible path reaches critical)
    - Violation: Authorization check absent OR Synchronization primitive absent
    
    Note: Unlike other categories with multi-anchor chains (src→comp→sink),
    Control-Logic has only ONE anchor: the critical operation. The violation
    is expressed through the ABSENCE of a required guard, not through anchor chains.
    The guard is part of MITIGATED(P), not part of the anchor set A.
    """
    
    @property
    def major_category(self) -> MajorCategory:
        return MajorCategory.CONTROL_LOGIC
    
    @property
    def anchors(self) -> List[Anchor]:
        """
        Per Table 7: A = {critical}
        
        Control-Logic violations have only ONE anchor - the critical/sensitive operation.
        The missing guard is expressed in the violation predicate, not as an anchor.
        """
        return [
            Anchor(type=AnchorType.CRITICAL, description="The sensitive operation requiring protection")
        ]
    
    @property
    def constraint(self) -> VulnerabilityConstraint:
        """
        Per Table 7: P → critical with missing guard.
        
        The chain represents reachability: a feasible path P reaches critical.
        The violation is that the required guard is ABSENT on path P.
        This differs from other categories where violations involve incorrect
        relationships between multiple anchors.
        """
        return VulnerabilityConstraint(
            chain=[],  # No inter-anchor chain - only reachability to critical
            violation=ViolationPredicate(
                name="missing_guard",
                description="Critical operation reachable without required guard",
                formal_condition="∃ P: reachable(critical, P) ∧ ¬∃ guard ∈ P: guard →c critical"
            ),
            mitigations=[
                MitigationPattern(
                    name="add_guard",
                    description="Add proper predicate guard before critical operation",
                    anchor_type=AnchorType.CRITICAL,
                    code_patterns=["if (!authorized) return -EPERM;", "if (!valid) return ERROR;"]
                ),
            ]
        )

class RaceCondition(ControlLogicViolation):
    """
    Race Condition (CWE-362): Concurrent access without synchronization.
    
    Per Table 7:
    - Anchor: A = {critical} - the shared resource access
    - Chain: P → critical
    - Violation: Synchronization primitive absent
    """
    @property
    def name(self) -> str:
        return "Race Condition"
        
    @property
    def description(self) -> str:
        return "Concurrent execution results in indeterminate behavior due to missing synchronization."
    
    @property
    def constraint(self) -> VulnerabilityConstraint:
        """CHAIN: P → critical, VIOL: synchronization primitive absent"""
        return VulnerabilityConstraint(
            chain=[],  # Reachability only - no anchor chain
            violation=ViolationPredicate(
                name="race_condition",
                description="Shared resource accessed concurrently without synchronization",
                formal_condition="concurrent_access(critical) ∧ ¬synchronized(critical)"
            ),
            mitigations=[
                MitigationPattern(
                    name="lock",
                    description="Add mutex/spinlock around critical section",
                    anchor_type=AnchorType.CRITICAL,
                    code_patterns=[
                        "mutex_lock(&lock); ... mutex_unlock(&lock);",
                        "spin_lock(&lock); ... spin_unlock(&lock);",
                        "pthread_mutex_lock(&mutex);"
                    ]
                ),
                MitigationPattern(
                    name="atomic",
                    description="Use atomic operations for shared data",
                    anchor_type=AnchorType.CRITICAL,
                    code_patterns=[
                        "atomic_inc(&counter);",
                        "atomic_set(&flag, 1);",
                        "__atomic_fetch_add(&val, 1, __ATOMIC_SEQ_CST);"
                    ]
                ),
                MitigationPattern(
                    name="rcu",
                    description="Use RCU for read-heavy concurrent access",
                    anchor_type=AnchorType.CRITICAL,
                    code_patterns=[
                        "rcu_read_lock(); ... rcu_read_unlock();",
                        "synchronize_rcu();",
                        "call_rcu(&obj->rcu, callback);"
                    ]
                ),
                MitigationPattern(
                    name="memory_barrier",
                    description="Use memory barriers for ordering",
                    anchor_type=AnchorType.CRITICAL,
                    code_patterns=[
                        "smp_wmb();",
                        "smp_rmb();",
                        "__sync_synchronize();"
                    ]
                ),
            ]
        )

class MissingAuthorization(ControlLogicViolation):
    """
    Missing Authorization (CWE-862): Critical operation reachable without authorization check.
    
    Per Table 7:
    - Anchor: A = {critical}
    - Chain: P → critical
    - Violation: Authorization check absent
    
    Note: This is distinct from Authentication Bypass (CWE-287) which is about
    bypassing authentication mechanisms. Missing Authorization is about lacking
    the authorization check entirely.
    """
    @property
    def name(self) -> str:
        return "Missing Authorization"
        
    @property
    def description(self) -> str:
        return "Critical operation performed without proper authorization check (CWE-862)."
    
    @property
    def constraint(self) -> VulnerabilityConstraint:
        """CHAIN: P → critical, VIOL: authorization check absent"""
        return VulnerabilityConstraint(
            chain=[],  # Reachability only
            violation=ViolationPredicate(
                name="missing_auth",
                description="Critical operation reached without authorization check",
                formal_condition="∃ P: reachable(critical, P) ∧ ¬∃ auth_check ∈ P: auth_check →c critical"
            ),
            mitigations=[
                MitigationPattern(
                    name="authorization_check",
                    description="Add authorization check before critical operation",
                    anchor_type=AnchorType.CRITICAL,
                    code_patterns=[
                        "if (!capable(CAP_SYS_ADMIN)) return -EACCES;",
                        "if (!ns_capable(ns, CAP_NET_ADMIN)) return -EPERM;",
                        "if (!inode_owner_or_capable(inode)) return -EACCES;",
                        "if (!check_permission(ctx, PERM_WRITE)) return -EACCES;"
                    ]
                ),
            ]
        )

class AuthenticationBypass(ControlLogicViolation):
    """
    Authentication Bypass (CWE-287): Bypassing authentication mechanisms.
    
    This is about circumventing authentication, not just missing it.
    """
    @property
    def name(self) -> str:
        return "Authentication Bypass"
        
    @property
    def description(self) -> str:
        return "Bypassing authentication mechanisms due to flawed or bypassable checks (CWE-287)."
    
    @property
    def constraint(self) -> VulnerabilityConstraint:
        """CHAIN: P → critical, VIOL: authentication can be bypassed"""
        return VulnerabilityConstraint(
            chain=[],  # Reachability only
            violation=ViolationPredicate(
                name="auth_bypass",
                description="Authentication mechanism can be bypassed",
                formal_condition="∃ path P: reachable(critical, P) ∧ ¬authenticated(P)"
            ),
            mitigations=[
                MitigationPattern(
                    name="auth_check",
                    description="Fix or add proper authentication check",
                    anchor_type=AnchorType.CRITICAL,
                    code_patterns=[
                        "if (!is_authenticated(user)) return -EPERM;",
                        "if (!verify_credentials(ctx)) return -EACCES;",
                        "if (!session_valid(sess)) return -EACCES;"
                    ]
                ),
            ]
        )

class LogicError(ControlLogicViolation):
    """
    Logic Error: General flaw in program logic (incorrect condition, wrong state transition).
    
    This is a catch-all for control-logic issues that don't fit specific categories.
    """
    @property
    def name(self) -> str:
        return "Logic Error"
        
    @property
    def description(self) -> str:
        return "Flaw in the program's logic (e.g., incorrect condition, wrong state transition)."
    
    @property
    def constraint(self) -> VulnerabilityConstraint:
        """VIOL: program state violates intended invariant"""
        return VulnerabilityConstraint(
            chain=[],  # Logic errors don't have anchor chains
            violation=ViolationPredicate(
                name="logic_error",
                description="Program state or behavior violates intended semantics",
                formal_condition="state ∉ valid_states ∨ transition ∉ valid_transitions"
            ),
            mitigations=[
                MitigationPattern(
                    name="condition_fix",
                    description="Fix the incorrect condition",
                    anchor_type=AnchorType.CRITICAL,
                    code_patterns=[
                        "if (a < b)  // was: if (a > b)",
                        "if (a && b)  // was: if (a || b)"
                    ]
                ),
                MitigationPattern(
                    name="state_validation",
                    description="Add state validation/assertion",
                    anchor_type=AnchorType.CRITICAL,
                    code_patterns=[
                        "BUG_ON(state != EXPECTED);",
                        "assert(obj->state == VALID);",
                        "WARN_ON(invalid_condition);"
                    ]
                ),
            ]
        )

# --- Fallback ---

class UnknownViolation(VulnerabilityCategory):
    @property
    def name(self) -> str:
        return "Unknown"
    
    @property
    def major_category(self) -> MajorCategory:
        return MajorCategory.UNKNOWN
        
    @property
    def description(self) -> str:
        return "Vulnerability type could not be determined. Analyze the code to identify any suspicious patterns using standard anchor types and dependency relationships."
        
    @property
    def anchors(self) -> List[Anchor]:
        return [
            Anchor(type=AnchorType.SOURCE, description="Potential data source", is_optional=True),
            Anchor(type=AnchorType.COMPUTATION, description="Potential computation or transformation", is_optional=True),
            Anchor(type=AnchorType.SINK, description="Potential sensitive usage", is_optional=True),
            Anchor(type=AnchorType.OBJECT, description="Potential object or buffer", is_optional=True),
            Anchor(type=AnchorType.INDEX, description="Potential index or offset", is_optional=True),
            Anchor(type=AnchorType.ACCESS, description="Potential memory access", is_optional=True),
            Anchor(type=AnchorType.ALLOC, description="Potential resource allocation", is_optional=True),
            Anchor(type=AnchorType.DEALLOC, description="Potential resource release", is_optional=True),
            Anchor(type=AnchorType.USE, description="Potential resource usage", is_optional=True),
            Anchor(type=AnchorType.EXIT, description="Potential exit point", is_optional=True),
            Anchor(type=AnchorType.CRITICAL, description="Potential critical operation", is_optional=True),
        ]
    
    @property
    def constraint(self) -> VulnerabilityConstraint:
        """Fallback constraint for unknown vulnerability types."""
        return VulnerabilityConstraint(
            chain=[],
            violation=ViolationPredicate(
                name="unknown_violation",
                description="Determine the specific violation condition based on the identified anchors and their relationships.",
                formal_condition="exists P: valid_chain(P) and violation(P)"
            ),
            mitigations=[]
        )

# --- Knowledge Base ---

class VulnerabilityKnowledgeBase:
    """
    Registry and factory for vulnerability categories.
    Supports two-stage classification: Major Category → Specific Subtype.
    
    Per VERDICT paper §3.1.2: Vulnerability Type Determination uses CWE identifiers
    from NVD/OSV databases to map vulnerabilities to categories.
    """
    
    _categories: Dict[str, Type[VulnerabilityCategory]] = {
        # Numeric-Domain (CWE-190, CWE-191, CWE-369)
        "Integer Overflow": IntegerOverflow,
        "Integer Underflow": IntegerUnderflow,
        "Divide By Zero": DivideByZero,
        "Type Confusion": TypeConfusion,
        
        # Access-Validity (CWE-787, CWE-125, CWE-476, CWE-457)
        "Out-of-bounds Write": OutOfBoundsWrite,
        "Out-of-bounds Read": OutOfBoundsRead,
        "Buffer Overflow": BufferOverflow,
        "Null Pointer Dereference": NullPointerDereference,
        "Uninitialized Use": UninitializedUse,
        
        # Resource-Lifecycle (CWE-416, CWE-415, CWE-401)
        "Use After Free": UseAfterFree,
        "Double Free": DoubleFree,
        "Memory Leak": MemoryLeak,
        
        # Control-Logic (CWE-362, CWE-862, CWE-287)
        "Race Condition": RaceCondition,
        "Missing Authorization": MissingAuthorization,  # CWE-862 per Table 7
        "Authentication Bypass": AuthenticationBypass,  # CWE-287
        "Logic Error": LogicError,
        
        # Fallback
        "Unknown": UnknownViolation
    }
    
    # CWE to vulnerability type mapping for §3.1.2 Vulnerability Type Determination
    # Per Table 7 in the paper
    _cwe_to_category: Dict[int, str] = {
        # Numeric-Domain Violations
        190: "Integer Overflow",
        191: "Integer Underflow",
        369: "Divide By Zero",
        # Access-Validity Violations
        787: "Out-of-bounds Write",
        125: "Out-of-bounds Read",
        120: "Buffer Overflow",  # Classic buffer overflow
        121: "Buffer Overflow",  # Stack-based buffer overflow
        122: "Buffer Overflow",  # Heap-based buffer overflow
        476: "Null Pointer Dereference",
        457: "Uninitialized Use",
        # Resource-Lifecycle Violations
        416: "Use After Free",
        415: "Double Free",
        401: "Memory Leak",
        # Control-Logic Violations
        362: "Race Condition",
        862: "Missing Authorization",  # Per Table 7
        287: "Authentication Bypass",
    }
    
    # Mapping from MajorCategory to subtypes (aligned with Table 7)
    # Note: Each subtype must have a corresponding VulnerabilityCategory class
    _major_to_subtypes: Dict[MajorCategory, List[str]] = {
        # Numeric-Domain: src →d comp →d sink (result exceeds type bounds or divisor is zero)
        MajorCategory.NUMERIC_DOMAIN: ["Integer Overflow", "Integer Underflow", "Divide By Zero", "Type Confusion"],
        
        # Access-Validity: obj →d access ←d idx (index exceeds boundary, pointer null, or uninitialized)
        MajorCategory.ACCESS_VALIDITY: ["Out-of-bounds Write", "Out-of-bounds Read", "Buffer Overflow", "Null Pointer Dereference", "Uninitialized Use"],
        
        # Resource-Lifecycle: alloc →d dealloc →t use (use after release, double release, or leak)
        MajorCategory.RESOURCE_LIFECYCLE: ["Use After Free", "Double Free", "Memory Leak"],
        
        # Control-Logic: P → critical (authorization check absent or synchronization primitive absent)
        MajorCategory.CONTROL_LOGIC: ["Race Condition", "Missing Authorization", "Authentication Bypass", "Logic Error"],
        
        MajorCategory.UNKNOWN: ["Unknown"],
    }
    
    @classmethod
    def get_category(cls, name: str) -> VulnerabilityCategory:
        """
        Retrieves a vulnerability category instance by name.
        Returns UnknownViolation if not found.
        """
        category_cls = cls._categories.get(name, UnknownViolation)
        return category_cls()
    
    @classmethod
    def get_category_by_cwe(cls, cwe_id: int) -> VulnerabilityCategory:
        """
        Retrieves a vulnerability category instance by CWE ID.
        
        Per §3.1.2: We query vulnerability databases (NVD and OSV) to obtain
        the CWE identifier, then map to our predefined categories.
        
        Args:
            cwe_id: The CWE identifier (e.g., 190 for Integer Overflow)
            
        Returns:
            The corresponding VulnerabilityCategory, or UnknownViolation if not mapped.
        """
        category_name = cls._cwe_to_category.get(cwe_id, "Unknown")
        return cls.get_category(category_name)
    
    @classmethod
    def list_categories(cls) -> List[str]:
        """Returns a list of all registered category names."""
        return list(cls._categories.keys())
    
    @classmethod
    def list_major_categories(cls) -> List[MajorCategory]:
        """Returns a list of all major categories (excluding UNKNOWN)."""
        return [mc for mc in MajorCategory if mc != MajorCategory.UNKNOWN]
    
    @classmethod
    def get_subtypes_for_major(cls, major: MajorCategory) -> List[str]:
        """Returns a list of subtype names for a given major category."""
        return cls._major_to_subtypes.get(major, [])
    
    @classmethod
    def get_categories_by_major(cls, major: MajorCategory) -> List[VulnerabilityCategory]:
        """Returns a list of VulnerabilityCategory instances for a given major category."""
        subtypes = cls.get_subtypes_for_major(major)
        return [cls.get_category(name) for name in subtypes]
    
    @classmethod
    def get_cwe_ids_for_category(cls, category_name: str) -> List[int]:
        """
        Returns all CWE IDs that map to a given category name.
        
        Args:
            category_name: The vulnerability category name
            
        Returns:
            List of CWE IDs that map to this category
        """
        return [cwe_id for cwe_id, name in cls._cwe_to_category.items() if name == category_name]
    
    # =========================================================================
    # §3.1.2 Helper Methods — Added for Type Determination refactoring
    # =========================================================================
    
    @classmethod
    def get_type_descriptions_for_llm(cls) -> str:
        """
        Generate structured descriptions of all available vulnerability subtypes
        for injection into LLM prompts during type determination.
        
        Returns:
            Formatted string listing all major categories and their subtypes
            with descriptions and anchor chains.
        """
        lines = []
        for major in cls.list_major_categories():
            lines.append(f"\n### {major.value}")
            lines.append(f"Description: {major.description}")
            lines.append(f"Anchor chain model: {' → '.join(at.value for at in major.anchor_types)}")
            for name in cls.get_subtypes_for_major(major):
                cat = cls.get_category(name)
                anchor_names = [a.type.value for a in cat.anchors]
                lines.append(f"  - **{name}**: {cat.description}")
                lines.append(f"    Anchors: {' → '.join(anchor_names)}")
                if cat.constraint and cat.constraint.violation:
                    lines.append(f"    Violation: {cat.constraint.violation.description}")
        return "\n".join(lines)
    
    @classmethod
    def get_major_category_for_cwe(cls, cwe_id: int) -> Optional[MajorCategory]:
        """
        Get the major category for a CWE ID.
        Returns None if the CWE is not in the mapping.
        
        Args:
            cwe_id: Numeric CWE identifier (e.g., 190 for Integer Overflow)
        """
        name = cls._cwe_to_category.get(cwe_id)
        if name is None:
            return None
        cat = cls.get_category(name)
        return cat.major_category
    
    @classmethod
    def get_category_name_by_cwe(cls, cwe_id: int) -> Optional[str]:
        """
        Get the category name string for a CWE ID.
        Returns None if not mapped.
        
        Args:
            cwe_id: Numeric CWE identifier
            
        Returns:
            Category name string (e.g., "Integer Overflow") or None
        """
        return cls._cwe_to_category.get(cwe_id)
    
    @classmethod
    def is_numeric_cwe(cls, cwe_id: int) -> bool:
        """
        Check if a CWE ID directly maps to a Numeric-Domain category.
        
        Used for the fast-path in §3.1.2: if CWE is already Numeric and
        no contradicting evidence, skip LLM root cause analysis.
        
        Args:
            cwe_id: Numeric CWE identifier
        """
        name = cls._cwe_to_category.get(cwe_id)
        if name is None:
            return False
        cat = cls.get_category(name)
        return cat.major_category == MajorCategory.NUMERIC_DOMAIN
    
    @classmethod
    def get_all_category_names(cls) -> List[str]:
        """Returns all registered category names (for LLM output validation)."""
        return list(cls._categories.keys())
