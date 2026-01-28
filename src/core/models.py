from pydantic import BaseModel, Field
from enum import Enum
from typing import List, Optional, Any, Dict, Literal, Set
from dataclasses import dataclass

class AtomicPatch(BaseModel):
    file_path: str
    function_name: str
    clean_diff: str          
    raw_diff: str            
    change_type: str         
    old_code: Optional[str] = None
    new_code: Optional[str] = None 
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # [新增] 分别记录 old/new 代码片段在原文件中的起始行号
    start_line_old: Optional[int] = None
    start_line_new: Optional[int] = None

    # --- Phase 3.1.2 特征 ---
    calls: Set[str] = Field(default_factory=set, exclude=True)
    tokens: Set[str] = Field(default_factory=set, exclude=True)
    # [新增] 访问的结构体成员/全局变量 (e.g., "wq", "lock", "flags")
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
    """三级级联的类型确定置信度"""
    HIGH = "High"      # Tier 1: CVE/CWE元数据推断
    MEDIUM = "Medium"  # Tier 2: LLM代码分析推断
    LOW = "Low"        # Tier 3: 通用Fallback

class AnchorRole(str, Enum):
    """预定义的锚点角色（用于定位和切片）"""
    # Memory Operations
    ALLOC = "Alloc"
    FREE = "Free"
    USE = "Use"
    DEREF = "Deref"
    
    # Data Flow
    SOURCE = "Source"
    SINK = "Sink"
    DEF = "Def"
    ASSIGN = "Assign"
    
    # Control Flow
    CHECK = "Check"
    ACCESS = "Access"
    BRANCH = "Branch"
    
    # Resource Management
    ACQUIRE = "Acquire"
    RELEASE = "Release"
    
    # Synchronization
    LOCK = "Lock"
    UNLOCK = "Unlock"
    
    # Computation
    COMPUTE = "Compute"
    DIVIDE = "Divide"
    
    # Impact Points
    LEAK = "Leak"
    CRASH = "Crash"
    OVERFLOW = "Overflow"
    CORRUPTION = "Corruption"
    
    # Fallback
    GENERIC = "Generic"

class GeneralVulnType(str, Enum):
    """27个具体漏洞类型（General Vulnerability Types）"""
    
    # 1. Memory Safety (7)
    USE_AFTER_FREE = "Use After Free"
    DOUBLE_FREE = "Double Free"
    BUFFER_OVERFLOW = "Buffer Overflow"
    OUT_OF_BOUNDS_READ = "Out-of-bounds Read"
    MEMORY_LEAK = "Memory Leak"
    NULL_POINTER_DEREFERENCE = "Null Pointer Dereference"
    UNINITIALIZED_USE = "Uninitialized Use"
    
    # 2. Concurrency (2)
    RACE_CONDITION = "Race Condition"
    DEADLOCK = "Deadlock"
    
    # 3. Numeric & Type (3)
    INTEGER_OVERFLOW = "Integer Overflow"
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
    """单点的切片指令"""
    code_content: str = Field(description="Code line content to locate the node (e.g., 'kfree(req)')")
    focus_variable: str = Field(description="Variable to trace at this line (e.g., 'req')")
    description: Optional[str] = Field(description="Brief reason")

class KeyLine(BaseModel):
    line_number: int = Field(description="The specific line number in the provided Full Code Context.")
    content: str = Field(description="The code content.")
    description: Optional[str] = Field(description="Brief explanation of why this line is key.")

class TaxonomyFeature(BaseModel):
    """
    Phase 3.2.1 Output - Type Determination & Semantic Hypothesis
    
    This is a hypothetical vulnerability report operating at conceptual level (per paper Definition 3).
    Generated before concrete analysis and guides subsequent slicing/verification.
    """
    
    # === Core Fields ===
    vuln_type: GeneralVulnType = Field(description="Specific vulnerability type (27 categories)")
    type_confidence: TypeConfidence = Field(description="Confidence level of type determination (HIGH/MEDIUM/LOW)")
    
    # CWE Information (Optional, for standardized reporting)
    cwe_id: Optional[str] = Field(default=None, description="CWE ID (e.g., 'CWE-416') - Optional")
    cwe_name: Optional[str] = Field(default=None, description="Standard CWE Name - Optional")
    
    # Anchor Roles (Determined by vulnerability type)
    origin_roles: List[AnchorRole] = Field(default_factory=list, description="Origin anchor roles (OR relationship)")
    impact_roles: List[AnchorRole] = Field(default_factory=list, description="Impact anchor roles (OR relationship)")
    
    # === Semantic Hypothesis (Flattened, no nested class) ===
    root_cause: str = Field(
        description="Hypothesized defect description (e.g., 'missing null check before dereference')"
    )
    attack_path: str = Field(
        description="Conceptual Origin→Impact chain with instance-specific details (e.g., 'attacker controls input → triggers allocation failure → null pointer → dereference')"
    )
    fix_mechanism: str = Field(
        description="What the patch modifies and why it prevents the vulnerability"
    )
    
    # Reasoning
    reasoning: str = Field(description="Reasoning process for type determination")

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

class SliceFeature(BaseModel):
    """单函数的切片特征"""
    func_name: str
    s_post: str     # Post-Patch (or Primary) Slice Code
    s_pre: str  # Pre-Patch (or Shadow) Slice Code
    
    # [New] Origin/Impact Info for Searcher
    # Stores the exact line content (including line number "[123]") that are identified as Origin/Impact anchors
    pre_origins: List[str] = Field(default_factory=list)
    pre_impacts: List[str] = Field(default_factory=list)
    post_origins: List[str] = Field(default_factory=list)
    post_impacts: List[str] = Field(default_factory=list)
    
    # [New] Validation & Hypothesis Feedback (Defer & Aggregate)
    validation_status: Optional[str] = Field(default=None, description="The local validation role/status e.g. 'Victim', 'Allocator', 'Guard'.")
    validation_reasoning: Optional[str] = Field(default="", description="Reasoning for the local validation.")

    # 可选：如果你想在切片级别保留局部指纹，可以在这里加，但通常全局指纹够用了
    # local_fingerprints: List[str] 

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
    fix_mechanism: Optional[str] = None
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
    2. Attack Path: Step-by-step trace with concrete evidence (line numbers, functions)
    3. Fix Mechanism: What the patch changes and why it prevents exploitation
    
    Used for search (Phase III) and verification (Phase IV).
    """
    # Component 1: Vulnerability Type & Root Cause
    vuln_type: GeneralVulnType = Field(description="Confirmed vulnerability type (27 categories)")
    cwe_id: Optional[str] = Field(default=None, description="CWE ID if available")
    cwe_name: Optional[str] = Field(default=None, description="CWE name if available")
    root_cause: str = Field(description="Refined root cause description with concrete evidence")
    
    # Component 2: Attack Path (with evidence references)
    attack_path: str = Field(
        description="Step-by-step trace from Origin to Impact, citing functions, line numbers, data/control flows"
    )
    
    # Component 3: Fix Mechanism
    fix_mechanism: str = Field(
        description="What the patch modifies and why it prevents the attack path"
    )
    
    # Evidence Index (for verification phase - maps evidence_id to concrete code references)
    evidence_index: Dict[str, EvidenceRef] = Field(default_factory=dict)

# Alias for backward compatibility
VulnerabilityReport = SemanticFeature

class PatchFeatures(BaseModel):    # 原 AnalyzedGroup，改名强调它是特征集合
    """Phase 3.2 的最终产出：补丁特征集"""
    group_id: str
    patches: List[AtomicPatch]
    commit_message: str
    # 这里的命名也可以简化
    taxonomy: TaxonomyFeature
    
    # [修改] 变为字典，Key 为函数唯一标识 (如 "funcname")
    # 这样后续步骤可以遍历这个字典进行单点分析
    slices: Dict[str, SliceFeature] 
    
    # 全局语义 (用于粗筛)
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
    """单行匹配证据"""
    slice_line: str      # 切片中的原始代码
    target_line: str     # 目标函数中的匹配代码
    line_no: int = Field(exclude=True) # 目标函数中的行号 (1-based), 内部使用，JSON不输出
    similarity: float    # 相似度

class AlignedTrace(BaseModel):
    """包含未匹配项的完整对齐轨迹"""
    slice_line: str
    target_line: Optional[str] = None
    line_no: Optional[int] = Field(default=None, exclude=True)
    similarity: float
    tag: Optional[str] = None # 'COMMON', 'VULN', 'FIX'

class MatchEvidence(BaseModel):
    """
    完整的匹配证据链（与 Methodology.tex §3.4 Signature Matching 对应）
    
    得分说明：
    - score_vuln/score_fix: 整体切片相似度（DP对齐后的总分）
    - score_feat_vuln/score_feat_fix: 局部特征相似度（仅 VULN/FIX 行）
    - confidence: 仅 VULNERABLE 时有效（= α*score_vuln + β*anchor_score）
    """
    verdict: Literal["VULNERABLE", "PATCHED", "UNKNOWN", "MISMATCH"]
    confidence: float  # 仅 VULNERABLE 时使用（二维加权得分）
    
    # 四分数体系（与论文一致）
    score_vuln: float       # 整体切片相似度（pre-patch vs target）
    score_fix: float        # 整体切片相似度（post-patch vs target）
    score_feat_vuln: float  # 局部特征相似度（仅 VULN 行）
    score_feat_fix: float   # 局部特征相似度（仅 FIX 行）
    
    # 全局对齐详情（用于调试和可视化）
    aligned_vuln_traces: List[AlignedTrace] = Field(default_factory=list)
    aligned_fix_traces: List[AlignedTrace] = Field(default_factory=list)

class SearchResultItem(BaseModel):
    """单条搜索结果"""
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
    code_content: Optional[str] = None # 用于 Phase 4 传递完整代码
    
class VulnerabilityFinding(BaseModel):
    """
    Phase 4 最终产出：确认存在的漏洞
    
    Per Methodology.tex Section 4.4: Semantic Constraint Verification
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
    analysis_report: str = Field(description="Exploitation Report per Methodology.tex")
    is_vulnerable: bool
    
    # [New] Verdict Category (per Methodology.tex Section 4.4.3)
    verdict_category: str = Field(
        default="UNKNOWN",
        description="One of: 'VULNERABLE', 'SAFE-Blocked', 'SAFE-Mismatch', 'SAFE-Unreachable', 'UNKNOWN'"
    )
    
    # Context information
    involved_functions: List[str] = Field(default_factory=list, description="Functions involved in the vulnerability chain.")
    peer_functions: List[str] = Field(default_factory=list, description="Peer functions used as context during analysis.")
    
    # Evidence chain (per Methodology.tex Evidence Schema)
    # Using Any type to avoid circular imports with StepAnalysis from verifier.py
    origin: Optional[Any] = Field(default=None, description="Origin anchor: where vulnerable state is created")
    impact: Optional[Any] = Field(default=None, description="Impact anchor: where vulnerability is triggered")
    defense_step: Optional[Any] = Field(default=None, description="Defense mechanism location (if SAFE-Blocked)")
    defense_status: Optional[str] = Field(default=None, description="Constraint outcomes: C_cons, C_reach, C_def status")
    trace: Optional[List[Any]] = Field(default=None, description="Execution trace from Origin to Impact")
    
class TargetFunctionInfo(BaseModel):
    name: str
    code: str
    start_line: int
    end_line: int

class VerificationResult(BaseModel):
    is_vulnerable: bool = Field(description="True if the target code contains the vulnerability.")
    confidence: float = Field(description="0.0 to 1.0 confidence score.")
    analysis_chain: str = Field(description="Step-by-step reasoning linking logic to code.")