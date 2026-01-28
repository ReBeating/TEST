import operator
from typing import List, TypedDict, Annotated, Optional, Dict, Any

from core.models import (
    AtomicPatch, PatchFeatures, TaxonomyFeature, VulnerabilityFinding,
    SliceFeature, SemanticFeature, SearchResultItem
)

def keep_original(old_value: Any, new_value: Any) -> Any:
    """
    冲突解决策略：当多个 Worker 同时写回相同的 commit_message 时，
    只保留原值（或任意一个），忽略冲突。
    """
    if old_value is None or old_value == '' or old_value == 0:
        return new_value
    return old_value

# core/state.py (追加)

class BatchInputItem(TypedDict):
    repo: str
    vul_id: str
    fixed_commit_sha: str

class BatchState(TypedDict):
    inputs: List[dict]
    repo_base_path: str
    output_base_dir: str # 新增：总输出目录
    start_phase: Annotated[int, keep_original]
    end_phase: Annotated[int, keep_original]
    force_execution: Annotated[bool, keep_original]
    mode: Annotated[str, keep_original]
    # 简单的计数器，用于最终统计
    completed_ids: Annotated[List[str], operator.add]

class WorkflowState(TypedDict):
    # --- 基础信息 ---
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
    # --- Phase 3.1 产物 ---
    atomic_patches: List[AtomicPatch]        # 原始补丁
    grouped_patches: List[List[AtomicPatch]] # 分组后的补丁列表
    
    # --- Phase 2: Feature Extraction ---
    # [聚合] 所有补丁组提取出的特征
    analyzed_features: Annotated[List[PatchFeatures], operator.add]
    
    # --- Phase 3: Search ---
    # [聚合] 搜索到的所有候选 (MatchEvidence)
    search_candidates: Annotated[List[SearchResultItem], operator.add]
    
    # --- Phase 4: Confirmation ---
    # [聚合] 最终确认结果
    final_findings: Annotated[List[VulnerabilityFinding], operator.add]
    
    # --- Phase 4 同步控制 ---
    p4_total_tasks: int  # dispatcher 设置的总任务数
    p4_done_markers: Annotated[List[int], operator.add]  # 每个子图完成时添加 [1]
    
    errors: Annotated[List[str], operator.add]
    
# ==========================================
# 2. 特征提取子状态 (Phase 2 Worker)
# ==========================================
class PatchExtractionState(TypedDict):
    group_id: str
    vul_id: str  # CVE ID (e.g., "CVE-2022-28463")
    patches: List[AtomicPatch]
    commit_hash: str
    commit_message: str
    repo_path: str
    
    # 中间产物
    taxonomy: Optional[TaxonomyFeature]
    slices: Optional[Dict[str, SliceFeature]]
    semantics: Optional[SemanticFeature]
    
    # 输出到主图
    analyzed_features: List[PatchFeatures]

# ==========================================
# 3. 验证子状态 (Phase 4 Worker)
# ==========================================
class VerificationState(TypedDict):
    mode: str # 'repo' or 'benchmark'
    vul_id: str
    candidates: List[SearchResultItem]  # 一组同 group_id 的候选
    feature_context: PatchFeatures # 提供上下文给 LLM
    final_findings: List[VulnerabilityFinding]
    p4_done_markers: List[int]  # 子图完成标记