import operator
from typing import List, TypedDict, Annotated, Optional, Dict, Any

from core.models import (
    AtomicPatch, PatchFeatures, TaxonomyFeature, VulnerabilityFinding,
    SliceFeature, SemanticFeature, SearchResultItem
)

def keep_original(old_value: Any, new_value: Any) -> Any:
    """
    Conflict resolution strategy: When multiple Workers write back the same commit_message at the same time,
    keep the original value (or any one), ignore conflicts.
    """
    if old_value is None or old_value == '' or old_value == 0:
        return new_value
    return old_value

# core/state.py (Append)

class BatchInputItem(TypedDict):
    repo: str
    vul_id: str
    fixed_commit_sha: str

class BatchState(TypedDict):
    inputs: List[dict]
    repo_base_path: str
    output_base_dir: str # New: Total output directory
    start_phase: Annotated[int, keep_original]
    end_phase: Annotated[int, keep_original]
    force_execution: Annotated[bool, keep_original]
    mode: Annotated[str, keep_original]
    # Simple counter for final statistics
    completed_ids: Annotated[List[str], operator.add]

class WorkflowState(TypedDict):
    # --- Basic Info ---
    repo_name: str
    repo_path: Annotated[str, keep_original]
    output_dir: str
    result_dir: str
    ablation: str
    vul_id: Annotated[str, keep_original]
    mode: Annotated[str, keep_original] # 'repo' or 'benchmark'
    commit_hash: Annotated[str, keep_original]
    commit_message: Annotated[str, keep_original]
    start_phase: int # 1, 2, 3, 4
    end_phase: int   # 1, 2, 3, 4
    force_execution: bool
    lang: str
    # --- Phase 3.1 Artifacts ---
    atomic_patches: List[AtomicPatch]        # Original patches
    grouped_patches: List[List[AtomicPatch]] # Grouped patch list
    
    # --- Phase 2: Feature Extraction ---
    # [Aggregation] Features extracted from all patch groups
    analyzed_features: Annotated[List[PatchFeatures], operator.add]
    
    # --- Phase 3: Search ---
    # [Aggregation] All search candidates found (MatchEvidence)
    search_candidates: Annotated[List[SearchResultItem], operator.add]
    
    # --- Phase 4: Confirmation ---
    # [Aggregation] Final confirmation results
    final_findings: Annotated[List[VulnerabilityFinding], operator.add]
    
    # --- Phase 4 Synchronization Control ---
    p4_total_tasks: int  # Total tasks set by dispatcher
    p4_done_markers: Annotated[List[int], operator.add]  # Add [1] when each subgraph is completed
    
    errors: Annotated[List[str], operator.add]
    
# ==========================================
# 2. Feature Extraction Sub-state (Phase 2 Worker)
# ==========================================
class PatchExtractionState(TypedDict):
    group_id: str
    vul_id: str  # CVE ID (e.g., "CVE-2022-28463")
    patches: List[AtomicPatch]
    commit_hash: str
    commit_message: str
    repo_path: str
    
    # Intermediate artifacts
    taxonomy: Optional[TaxonomyFeature]
    slices: Optional[Dict[str, SliceFeature]]
    semantics: Optional[SemanticFeature]
    
    # Output to main graph
    analyzed_features: List[PatchFeatures]

# ==========================================
# 3. Validation Sub-state (Phase 4 Worker)
# ==========================================
class VerificationState(TypedDict):
    mode: str # 'repo' or 'benchmark'
    vul_id: str
    candidates: List[SearchResultItem]  # A group of candidates with the same group_id
    feature_context: PatchFeatures # Provide context to LLM
    final_findings: List[VulnerabilityFinding]
    p4_done_markers: List[int]  # Subgraph completion marker