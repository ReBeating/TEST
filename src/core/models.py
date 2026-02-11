from pydantic import BaseModel, Field
from enum import Enum
from typing import List, Optional, Any, Dict, Literal, Set
from dataclasses import dataclass
from core.categories import VulnerabilityKnowledgeBase, VulnerabilityCategory, AnchorType, VulnerabilityConstraint, Anchor

class AtomicPatch(BaseModel):
    file_path: str
    function_name: str
    clean_diff: str          
    raw_diff: str            
    change_type: str         
    old_code: Optional[str] = None
    new_code: Optional[str] = None 
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # [New] Record the start line numbers of old/new code snippets in the original file respectively
    start_line_old: Optional[int] = None
    start_line_new: Optional[int] = None

    # --- Phase 3.1.2 Features ---
    calls: Set[str] = Field(default_factory=set, exclude=True)
    tokens: Set[str] = Field(default_factory=set, exclude=True)
    # [New] Access structure members/global variables (e.g., "wq", "lock", "flags")
    accessed_members: Set[str] = Field(default_factory=set, exclude=True)

    @property
    def id(self):
        return f"{self.file_path}::{self.function_name}"
    
    class Config:
        arbitrary_types_allowed = True
        
@dataclass
class HunkObj:
    index: int
    header: str
    content: List[str]
    raw_text: str

class NoiseLineDef(BaseModel):
    line_index: int = Field(description="The index number prefixed in the input lines, e.g., 0, 1, 2")
    content: str = Field(description="The content of the line for verification")

class HunkDecision(BaseModel):
    hunk_index: int
    classification: Literal["KEEP_ALL", "REMOVE_ALL", "PARTIAL_NOISE"]
    noise_lines: List[NoiseLineDef] = Field(default=[])
    reasoning: str

class BatchAnalysisResult(BaseModel):
    decisions: List[HunkDecision]

@dataclass
class GlobalHunkContext:
    global_id: int
    original_hunk: HunkObj
    file_path: str
    function_name: str

    @property
    def raw_text_with_indexed_lines(self) -> str:
        lines_str = []
        for idx, line in enumerate(self.original_hunk.content):
            lines_str.append(f"[{idx}] {line}")
        return (
            f"--- GLOBAL ID: {self.global_id} ---\n"
            f"File: {self.file_path} | Function: {self.function_name}\n"
            f"Header: {self.original_hunk.header}\n"
            f"Content:\n" + "\n".join(lines_str)
        )

class FixType(str, Enum):
    SYNCHRONIZATION = "Synchronization"
    GUARDING = "Guarding"
    COMPUTATION = "Computation"
    LIFECYCLE = "Lifecycle"
    UNKNOWN = "Unknown"

class VulnType(str, Enum):
    RACE = "Race"
    STATE = "State"
    CALCULATION = "Calculation"
    BOUND = "Bound"
    UNKNOWN = "Unknown"

class TypeConfidence(str, Enum):
    """Three-level cascaded type determination confidence"""
    HIGH = "High"      # Tier 1: CVE/CWE Meta Inference
    MEDIUM = "Medium"  # Tier 2: LLM Code Analysis Inference
    LOW = "Low"        # Tier 3: General Fallback

# NOTE: AnchorRole enum has been REMOVED per paper §3.1.3.
# The paper uses typed anchors (AnchorType from categories.py) instead of
# the binary Origin/Impact classification that AnchorRole supported.
# All downstream consumers now use AnchorType directly.

class GeneralVulnType(str, Enum):
    """27 specific General Vulnerability Types"""
    
    # 1. Memory Safety (8)
    USE_AFTER_FREE = "Use After Free"
    DOUBLE_FREE = "Double Free"
    OUT_OF_BOUNDS_WRITE = "Out-of-bounds Write"
    OUT_OF_BOUNDS_READ = "Out-of-bounds Read"
    BUFFER_OVERFLOW = "Buffer Overflow"
    MEMORY_LEAK = "Memory Leak"
    NULL_POINTER_DEREFERENCE = "Null Pointer Dereference"
    UNINITIALIZED_USE = "Uninitialized Use"
    
    # 2. Concurrency (2)
    RACE_CONDITION = "Race Condition"
    DEADLOCK = "Deadlock"
    
    # 3. Numeric & Type (4)
    INTEGER_OVERFLOW = "Integer Overflow"
    INTEGER_UNDERFLOW = "Integer Underflow"
    DIVIDE_BY_ZERO = "Divide By Zero"
    TYPE_CONFUSION = "Type Confusion"
    
    # 4. Logic & Access Control (4)
    AUTHENTICATION_BYPASS = "Authentication Bypass"
    PRIVILEGE_ESCALATION = "Privilege Escalation"
    AUTHORIZATION_BYPASS = "Authorization Bypass"
    LOGIC_ERROR = "Logic Error"
    
    # 5. Input & Data (5)
    INJECTION = "Injection"
    PATH_TRAVERSAL = "Path Traversal"
    IMPROPER_VALIDATION = "Improper Validation"
    INFORMATION_EXPOSURE = "Information Exposure"
    CRYPTOGRAPHIC_ISSUE = "Cryptographic Issue"
    
    # 6. Resource & Execution (4)
    INFINITE_LOOP = "Infinite Loop"
    RECURSION_ERROR = "Recursion Error"
    RESOURCE_EXHAUSTION = "Resource Exhaustion"
    RESOURCE_LEAK = "Resource Leak"
    
    # 7. Fallback (2)
    UNKNOWN = "Unknown"
    OTHER = "Other"

class SliceEntryPoint(BaseModel):
    """Single-point slicing instruction"""
    code_content: str = Field(description="Code line content to locate the node (e.g., 'kfree(req)')")
    focus_variable: str = Field(description="Variable to trace at this line (e.g., 'req')")
    description: Optional[str] = Field(description="Brief reason")

class KeyLine(BaseModel):
    line_number: int = Field(description="The specific line number in the provided Full Code Context.")
    content: str = Field(description="The code content.")
    description: Optional[str] = Field(description="Brief explanation of why this line is key.")

class TaxonomyFeature(BaseModel):
    """
    §3.1.2 Output - Vulnerability Type Determination & Semantic Hypothesis
    
    Primary type identifier is `category_name` — the exact name registered in
    VulnerabilityKnowledgeBase (from categories.py). This corresponds to the
    paper's subtype classification (e.g., "Integer Overflow", "Use After Free").
    
    `vuln_type` (GeneralVulnType enum) is kept for backward compatibility with
    downstream consumers but is derived from `category_name`.
    """
    
    # === Primary Type Identifier (from categories.py KB) ===
    category_name: str = Field(
        description="Exact category name in VulnerabilityKnowledgeBase "
                    "(e.g., 'Integer Overflow', 'Use After Free'). "
                    "This is the authoritative type from categories.py."
    )
    
    # === Backward-Compatible Fields ===
    vuln_type: GeneralVulnType = Field(
        default=GeneralVulnType.UNKNOWN,
        description="Legacy GeneralVulnType enum — derived from category_name for backward compatibility"
    )
    type_confidence: TypeConfidence = Field(description="Confidence level of type determination (HIGH/MEDIUM/LOW)")
    
    # CWE Information (Optional, for standardized reporting)
    cwe_id: Optional[str] = Field(default=None, description="CWE ID (e.g., 'CWE-416') - Optional")
    cwe_name: Optional[str] = Field(default=None, description="Standard CWE Name - Optional")
    
    # === Typed Anchor Specification (from categories.py KB) ===
    # No longer stores origin_roles/impact_roles. Instead, anchor types and
    # constraints are accessed via category_obj property from the KB.
    
    # === Semantic Hypothesis (Flattened, no nested class) ===
    root_cause: str = Field(
        description="Hypothesized defect description (e.g., 'missing null check before dereference')"
    )
    attack_chain: str = Field(
        description="Conceptual Origin→Impact chain with instance-specific details"
    )
    patch_defense: str = Field(
        description="What the patch modifies and why it prevents the vulnerability"
    )
    
    # Reasoning
    reasoning: str = Field(description="Reasoning process for type determination")
    
    # § 3.1.2 Numeric priority
    numeric_priority_applied: bool = Field(
        default=False,
        description="Whether Numeric priority rule was applied to override initial CWE mapping"
    )

    @property
    def category_obj(self) -> 'VulnerabilityCategory':
        """
        Returns the structured VulnerabilityCategory object from categories.py.
        Uses category_name as the primary lookup key.
        """
        return VulnerabilityKnowledgeBase.get_category(self.category_name)
    
    @property
    def major_category(self) -> 'MajorCategory':
        """Returns the major category (Numeric-Domain, Access-Validity, etc.)."""
        return self.category_obj.major_category
    
    @property
    def anchor_types(self) -> List['AnchorType']:
        """Returns the typed anchor list from KB (e.g., [ALLOC, DEALLOC, USE] for UAF)."""
        return [a.type for a in self.category_obj.anchors]
    
    @property
    def constraint(self) -> 'VulnerabilityConstraint':
        """Returns the constraint chain from KB (e.g., ALLOC →d DEALLOC →t USE)."""
        return self.category_obj.constraint
    
    @property
    def chain_description(self) -> str:
        """Human-readable chain description for LLM prompts."""
        return self.constraint.chain_str()

class CodeLineReference(BaseModel):
    line_number: int = Field(description="The specific line number in the 'Full Code Context'.")
    code_content: str = Field(description="The content of the code line (for verification).")
    reason: str = Field(description="Why this line should be removed (e.g., 'Irrelevant logging').")

class SliceValidationResult(BaseModel):
    # [Deprecated/Optional] noisy_lines (Moving successfully to relevant_lines)
    noisy_lines: List[CodeLineReference] = Field(
        default_factory=list,
        description="DEPRECATED: List of lines in the 'Input Slice' that are considered noise."
    )
    # [Added] Positive Selection Logic
    relevant_lines: List[int] = Field(
        default_factory=list,
        description="List of integer line numbers from the 'Input Slice' that are RELEVANT and MUST be KEPT."
    )
    reasoning: str = Field(description="Summary of the cleaning strategy.")

# [DEPRECATED] TypedAnchorRef — replaced by core.categories.Anchor which carries full
# locatability/assumption_type/assumption_rationale fields needed by the verifier.
# Kept as alias for backward compatibility only.
TypedAnchorRef = Anchor

class SliceFeature(BaseModel):
    """Slicing features of a single function (paper §3.1.3 output per function)"""
    func_name: str
    s_post: str     # Post-Patch (or Primary) Slice Code
    s_pre: str      # Pre-Patch (or Shadow) Slice Code
    
    # Typed anchors identified in the slice — now uses Anchor (from core.categories)
    # which carries locatability, assumption_type, assumption_rationale and other fields
    pre_anchors: List[Anchor] = Field(default_factory=list,
        description="Typed anchor instances in pre-patch code")
    post_anchors: List[Anchor] = Field(default_factory=list,
        description="Typed anchor instances in post-patch code")
    
    # [New] Validation & Hypothesis Feedback (Defer & Aggregate)
    validation_status: Optional[str] = Field(default=None, description="The local validation role/status e.g. 'Victim', 'Allocator', 'Guard'.")
    validation_reasoning: Optional[str] = Field(default="", description="Reasoning for the local validation.")

class FunctionFingerprint(BaseModel):
    key_statements: List[str] = []
    key_variables: List[str] = []
    key_functions: List[str] = []

# Note: ForensicReport removed per Methodology §3.2.3 - not part of Definition 3
# AnalysisEvaluation kept for potential future use but not required by Methodology

class EvidenceRef(BaseModel):
    """
    A machine-checkable evidence reference to a concrete line in a slice or code context.
    This is the backbone for auditable reports and later agent verification.
    """
    evidence_id: str = Field(description="Unique id, e.g., '{func}:{version}:{line_number}'.")
    func_name: str = Field(description="Function name where this evidence appears.")
    version: Literal["pre", "post", "target"] = Field(description="Code version: pre/ post-patch, or target repository.")
    line_number: int = Field(description="Absolute line number (as shown in the slice markers).")
    code: str = Field(description="Code content at the referenced line (without the line marker).")
    file_path: Optional[str] = Field(default=None, description="Optional file path if available.")

class AnchorRef(BaseModel):
    """
    Type-specific anchor located in a function (or represented by a call-site for inter-procedural anchors).
    """
    role: Literal["origin", "impact", "other"] = Field(description="Anchor role used for slicing and later verification.")
    evidence_id: Optional[str] = Field(default=None, description="Link to EvidenceRef if resolvable.")
    func_name: Optional[str] = None
    version: Optional[Literal["pre", "post"]] = None
    line_number: Optional[int] = None
    code: Optional[str] = None
    note: Optional[str] = Field(default=None, description="Optional note, e.g., 'call-site representative'.")

class LogicConstraint(BaseModel):
    expr: str = Field(description="A concrete constraint (e.g., 'len > max_len').")
    description: Optional[str] = None
    evidence_ids: List[str] = Field(default_factory=list, description="Evidence supporting this constraint.")

class ChainStep(BaseModel):
    step: str = Field(description="One step in the vulnerability chain.")
    evidence_ids: List[str] = Field(default_factory=list, description="Evidence supporting this step.")

class ReachabilitySummary(BaseModel):
    conditions: List[str] = Field(default_factory=list, description="Reachability conditions (e.g., config/ifdef/caller checks).")
    evidence_ids: List[str] = Field(default_factory=list)

class ExploitabilitySummary(BaseModel):
    argument: Optional[str] = Field(default=None, description="Why this is exploitable / impactful.")
    evidence_ids: List[str] = Field(default_factory=list)

class AttackChain(BaseModel):
    """
    A structured chain connecting origin to impact, not limited to taint-style bugs.
    """
    origin: List[AnchorRef] = Field(default_factory=list)
    impact: List[AnchorRef] = Field(default_factory=list)
    steps: List[ChainStep] = Field(default_factory=list)
    constraints: List[LogicConstraint] = Field(default_factory=list)
    reachability: Optional[ReachabilitySummary] = None
    exploitability: Optional[ExploitabilitySummary] = None

class DefenseRef(BaseModel):
    description: str
    evidence_ids: List[str] = Field(default_factory=list)

class FixEffect(BaseModel):
    """
    How the fix blocks the chain, expressed as checkable claims.
    """
    patch_defense: Optional[str] = None
    security_guarantee: Optional[str] = None
    blocking_points: List[DefenseRef] = Field(default_factory=list)
    residual_risks: List[str] = Field(default_factory=list)

class FunctionRoleMeta(BaseModel):
    role: Optional[str] = Field(default=None, description="Local role summary (e.g., Victim/Allocator/Guard-like).")
    reasoning: Optional[str] = Field(default=None, description="Why this function is relevant / why lines are kept.")
    key_evidence_ids: List[str] = Field(default_factory=list)

class SearchProfile(BaseModel):
    """
    Lightweight retrieval/ranking signals derived from vulnerable (pre-patch) slices.
    """
    fingerprints: Dict[str, FunctionFingerprint] = Field(default_factory=dict, description="Per-function fingerprints.")
    discriminators: Dict[str, List[str]] = Field(default_factory=dict, description="Optional extra discriminators (API pairs, fields, constants).")
    negative_signals: List[str] = Field(default_factory=list, description="Signals suggesting a candidate is already fixed / mismatched.")

class SemanticFeature(BaseModel):
    """
    Phase 3.2.3 Output - Grounded Vulnerability Report (Methodology Definition 3)
    
    This is the final output of semantic extraction, containing:
    1. Vulnerability Type & Root Cause: Confirmed type with refined description
    2. Attack Chain: Step-by-step trace with concrete evidence (line numbers, functions)
    3. Patch Defense: What the patch changes and why it prevents exploitation
    
    Used for search (Phase III) and verification (Phase IV).
    """
    # Component 1: Vulnerability Type & Root Cause
    vuln_type: GeneralVulnType = Field(description="Confirmed vulnerability type (27 categories)")
    cwe_id: Optional[str] = Field(default=None, description="CWE ID if available")
    cwe_name: Optional[str] = Field(default=None, description="CWE name if available")
    root_cause: str = Field(description="Refined root cause description with concrete evidence")
    
    # Component 2: Attack Chain (with evidence references)
    attack_chain: str = Field(
        description="Step-by-step trace from Origin to Impact, citing functions, line numbers, data/control flows"
    )
    
    # Component 3: Patch Defense
    patch_defense: str = Field(
        description="What the patch modifies and why it prevents the attack chain"
    )
    
    # Evidence Index (for verification phase - maps evidence_id to concrete code references)
    evidence_index: Dict[str, EvidenceRef] = Field(default_factory=dict)

# Alias for backward compatibility
VulnerabilityReport = SemanticFeature

class PatchFeatures(BaseModel):    # Originally AnalyzedGroup, renamed to emphasize it's a feature set
    """Phase 3.2 Final Artifact: Patch Feature Set"""
    group_id: str
    patches: List[AtomicPatch]
    commit_message: str
    # Naming can also be simplified here
    taxonomy: TaxonomyFeature
    
    # [Modified] Changed to dictionary, Key is function unique identifier (e.g. "funcname")
    # This allows subsequent steps to iterate over this dictionary for point analysis
    slices: Dict[str, SliceFeature] 
    
    # Global semantics (for coarse screening)
    semantics: SemanticFeature

class SlicingStrategy(str, Enum):
    """Defines how the slicer should process an instruction"""
    BACKWARD = "backward"  # Trace sources (Root Cause Analysis)
    FORWARD = "forward"    # Trace consequences (Impact Analysis)
    BIDIRECTIONAL = "bidirectional" # Context restoration (standard)
    CONTROL_ONLY = "control_only" # Logic flow only (for guards/checks)

class SlicingInstruction(BaseModel):
    """
    Enhanced instruction for the slicing engine.
    Now dictates *intent* rather than just location.
    """
    function_name: str
    target_version: str = Field(description="'OLD' or 'NEW'")
    line_number: int    # Critical: Physical line number in the target file
    focus_variable: Optional[str] = None # The variable to trace (if specific)
    code_content: str   # For verification
    
    # New Fields for Agentic Slicing
    strategy: SlicingStrategy = SlicingStrategy.BIDIRECTIONAL
    depth: int = Field(default=5, description="Search depth limit")
    description: Optional[str] = Field(default=None, description="Reason for this anchor (e.g., 'Hypothesis Keyword')")

class MatchTrace(BaseModel):
    """Single line match evidence"""
    slice_line: str      # Original code in slice
    target_line: str     # Matching code in target function
    line_no: int = Field(exclude=True) # Line number in target function (1-based), internal use, not output to JSON
    similarity: float    # Similarity

class AlignedTrace(BaseModel):
    """Complete alignment trace including unmatched items"""
    slice_line: str
    target_line: Optional[str] = None
    line_no: Optional[int] = Field(default=None, exclude=True)
    similarity: float
    tag: Optional[str] = None # 'COMMON', 'VULN', 'FIX'

class MatchEvidence(BaseModel):
    """
    Complete match evidence chain (Corresponding to Methodology.tex §3.4 Signature Matching)
    
    Score Explanation:
    - score_vuln/score_fix: overall slice similarity (Total score after DP alignment)
    - score_feat_vuln/score_feat_fix: local feature similarity (VULN/FIX lines only)
    - confidence: Valid only when VULNERABLE (= α*score_vuln + β*anchor_score)
    """
    verdict: Literal["VULNERABLE", "PATCHED", "UNKNOWN", "MISMATCH"]
    confidence: float  # Used only when VULNERABLE (2D weighted score)
    
    # Four-score system (consistent with paper)
    score_vuln: float       # Overall slice similarity (pre-patch vs target)
    score_fix: float        # Overall slice similarity (post-patch vs target)
    score_feat_vuln: float  # Local feature similarity (VULN lines only)
    score_feat_fix: float   # Local feature similarity (FIX lines only)
    
    # Global alignment details (for debugging and visualization)
    aligned_vuln_traces: List[AlignedTrace] = Field(default_factory=list)
    aligned_fix_traces: List[AlignedTrace] = Field(default_factory=list)

class SearchResultItem(BaseModel):
    """Single search result"""
    group_id: str
    repo_path: str
    target_file: str
    target_func: str
    patch_func: str
    patch_file: Optional[str] = None # [New] File path of the patch function
    verdict: str
    confidence: float
    rank: int = -1 # [New] Rank among VULNERABLE results (0-based), -1 for others
    scores: Dict[str, float]
    evidence: MatchEvidence
    code_content: Optional[str] = None # Used for Phase 4 to pass full code
    
class VulnerabilityFinding(BaseModel):
    """
    Phase 4 Final Output: Confirmed Vulnerabilities
    
    Per §3.3 Semantic-driven Validation:
    Verdict is determined by three constraints: C_cons, C_reach, C_def
    - VULNERABLE: C_cons ∧ C_reach ∧ ¬C_def
    - SAFE-Blocked: C_cons ∧ C_reach ∧ C_def
    - SAFE-Mismatch: ¬C_cons
    - SAFE-Unreachable: C_cons ∧ ¬C_reach
    """
    vul_id: str
    cwe_id: str
    cwe_name: str
    group_id: str
    repo_path: str
    patch_file: str
    patch_func: str
    target_file: str
    target_func: str
    analysis_report: str = Field(description="Exploitation Report per §3.3")
    is_vulnerable: bool
    
    # Verdict Category (per §3.3.3 Multi-Agent decision logic)
    verdict_category: str = Field(
        default="UNKNOWN",
        description="One of: 'VULNERABLE', 'SAFE-Blocked', 'SAFE-Mismatch', 'SAFE-Unreachable', 'SAFE-TypeMismatch', 'SAFE-OutOfScope', 'UNKNOWN'"
    )
    
    # Context information
    involved_functions: List[str] = Field(default_factory=list, description="Functions involved in the vulnerability chain.")
    peer_functions: List[str] = Field(default_factory=list, description="Peer functions used as context during analysis.")
    
    # Typed anchor evidence chain (per §3.3 Anchor-based model)
    # Uses Any type to avoid circular imports with StepAnalysis from verifier.py
    anchor_evidence: Optional[List[Any]] = Field(default=None, description="Ordered typed anchor evidence chain (e.g., [ALLOC, DEALLOC, USE] for UAF)")
    trace: Optional[List[Any]] = Field(default=None, description="Intermediate trace steps between anchors in the chain")
    defense_mechanism: Optional[Any] = Field(default=None, description="Defense mechanism location (if SAFE-Blocked)")
    constraint_status: Optional[str] = Field(default=None, description="Constraint outcomes: C_cons, C_reach, C_def status")
    path_verification: Optional[Any] = Field(default=None, description="Static path verification result from §3.3.2 (PDG-based)")
    
class TargetFunctionInfo(BaseModel):
    name: str
    code: str
    start_line: int
    end_line: int

class VerificationResult(BaseModel):
    is_vulnerable: bool = Field(description="True if the target code contains the vulnerability.")
    confidence: float = Field(description="0.0 to 1.0 confidence score.")
    analysis_chain: str = Field(description="Step-by-step reasoning linking logic to code.")