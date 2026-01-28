import os
import argparse
import uuid
from typing import List, Dict
from collections import defaultdict

from langgraph.graph import StateGraph, END
from langgraph.types import Send

from core.state import WorkflowState, PatchExtractionState, VerificationState
from core.checkpoint import CheckpointManager
from core.models import PatchFeatures

# 引入节点逻辑
from preprocessing.denoising import preprocessing_node
from preprocessing.grouping import grouping_node
from extraction.taxonomy import taxonomy_node
from extraction.slicer import slicing_node
from extraction.semantics import semantic_node
from search.matcher import matching_node 
from search.verifier import validation_node

from dotenv import load_dotenv

from core.configs import REPO_DIR_PATH, OUTPUT_DIR_PATH

# ==============================================================================
# 1. 节点定义 (Nodes)
# ==============================================================================

# ==============================================================================
# 1. 节点定义 (Nodes) - 修改 loader_node
# ==============================================================================

def loader_node(state: WorkflowState):
    """
    智能加载器：
    1. 支持断点续跑。
    2. 检测到结果文件存在且 force=False 时，自动加载数据并跳过该阶段（快进）。
    """
    vul_id = state["vul_id"]
    current_phase = state["start_phase"]
    end_phase = state["end_phase"]
    mode = state["mode"]
    output_dir = state["output_dir"]
    force = state.get("force_execution", False)
    print(f'Force execution: {force}')
    updates = {} # 用于更新 State 的数据
    
    print(f"[*] Pipeline Init: {vul_id} | Request Start: P{current_phase} | Force: {force}")

    # 定义检查和加载逻辑的循环
    # 我们尝试从 current_phase 开始快进，直到 end_phase
    # 注意：我们检查的是 "当前阶段的产出文件" 是否存在
    
    while current_phase <= end_phase:
        skip_current = False
        
        # --- Check Phase 1 Output ---
        if current_phase == 1:
            path = f"{vul_id}_phase1.pkl"
            full_path = os.path.join(output_dir, path)
            if os.path.exists(full_path) and not force:
                print(f"    [Skip] Phase 1 result found. Loading data...")
                data = CheckpointManager.load_pkl(path, output_dir)
                updates["grouped_patches"] = data.get("grouped_patches", [])
                updates["commit_message"] = data.get("commit_message", "")
                skip_current = True
            elif os.path.exists(full_path) and force:
                print(f"    [Force] Phase 1 result found but force=True. Will re-run.")
        
        # --- Check Phase 2 Output ---
        elif current_phase == 2:
            # 进入 P2 前必须确保 P1 数据在内存中 (如果是从 P1 跳过来的，updates里已经有了；如果是直接从 P2 start 的，需要补加载)
            # 但这里的逻辑是：如果 P2 产物存在，我们加载 P2 产物并跳过 P2 执行
            path = f"{vul_id}_phase2.pkl"
            full_path = os.path.join(output_dir, path)
            
            # 如果仅仅是从 P1 快进过来，还没加载 P1 数据用于 P2 运行？ 
            # 不需要，因为如果 P2 也跳过，就不需要 P1 数据。
            # 如果 P2 不跳过，state 会保留 P1 的 update。
            
            # 只有当我们需要真正运行 P2 时，才需要从 P1.pkl 恢复（如果不是刚才加载的）
            # 这里先判断是否跳过 P2
            if os.path.exists(full_path) and not force:
                print(f"    [Skip] Phase 2 result found. Loading data...")
                data = CheckpointManager.load_pkl(path, output_dir)
                updates["analyzed_features"] = data.get("analyzed_features", [])
                skip_current = True
            else:
                # 决定运行 P2。如果 updates 里没有 P1 数据（即直接从 P2 启动），需要补加载前序依赖
                if "grouped_patches" not in updates: 
                    p1_path = f"{vul_id}_phase1.pkl"
                    if os.path.exists(os.path.join(output_dir, p1_path)):
                         p1_data = CheckpointManager.load_pkl(p1_path, output_dir)
                         updates["grouped_patches"] = p1_data.get("grouped_patches")
                         updates["commit_message"] = p1_data.get("commit_message", "")

        # --- Check Phase 3 Output ---
        elif current_phase == 3:
            path = f"{vul_id}_{mode}_phase3.pkl"
            full_path = os.path.join(output_dir, path)
            
            if os.path.exists(full_path) and not force:
                print(f"    [Skip] Phase 3 result found. Loading data...")
                data = CheckpointManager.load_pkl(path, output_dir)
                updates["search_candidates"] = data.get("search_candidates", [])
                skip_current = True
            else:
                # 决定运行 P3。需确保 P2 数据存在
                if "analyzed_features" not in updates:
                    p2_path = f"{vul_id}_phase2.pkl"
                    if os.path.exists(os.path.join(output_dir, p2_path)):
                        updates["analyzed_features"] = CheckpointManager.load_pkl(p2_path, output_dir).get("analyzed_features", [])

        # --- Check Phase 4 Output ---
        elif current_phase == 4:
            path = f"{vul_id}_{mode}_phase4.pkl"
            full_path = os.path.join(output_dir, path)
            
            if os.path.exists(full_path) and not force:
                print(f"    [Skip] Phase 4 result found. Pipeline Done.")
                data = CheckpointManager.load_pkl(path, output_dir)
                updates["final_findings"] = data.get("final_findings", [])
                skip_current = True
            else:
                # 决定运行 P4。需要 P2 (features) 和 P3 (candidates)
                if "analyzed_features" not in updates:
                    p2_path = f"{vul_id}_phase2.pkl"
                    if os.path.exists(os.path.join(output_dir, p2_path)):
                        updates["analyzed_features"] = CheckpointManager.load_pkl(p2_path, output_dir).get("analyzed_features", [])
                if "search_candidates" not in updates:
                    p3_path = f"{vul_id}_{mode}_phase3.pkl"
                    if os.path.exists(os.path.join(output_dir, p3_path)):
                        updates["search_candidates"] = CheckpointManager.load_pkl(p3_path, output_dir).get("search_candidates", [])

        # --- 决策 ---
        if skip_current:
            current_phase += 1 # 快进到下一阶段
        else:
            break # 无法跳过当前阶段，停止快进，准备执行
            
    # 更新 state 中的 start_phase，这样 router 就会把我们发配到正确的地方
    updates["start_phase"] = current_phase
    print(f"[*] Pipeline Logic: Jumping to Phase {current_phase}")
    
    return updates

# --- Dispatchers (用于 Map) ---

def extraction_dispatcher(state: WorkflowState):
    tasks = []
    for group in state["grouped_patches"]:
        tasks.append(Send("phase2_extraction", {
            "group_id": str(uuid.uuid4())[:8],
            "vul_id": state["vul_id"],  # Pass vul_id for metadata loading
            "patches": group,
            "commit_hash": state["commit_hash"],
            "commit_message": state["commit_message"],
            "repo_path": state["repo_path"]
        }))
    return tasks

def validation_dispatcher(state: WorkflowState):
    tasks = []
    features_map = {f.group_id: f for f in state["analyzed_features"]}
    # 按 group_id 聚合
    group_map = defaultdict(list)
    for cand in state["search_candidates"]:
        group_map[cand.group_id].append(cand)
    for group_id, cands in group_map.items():
        feat = features_map.get(group_id)
        if feat:
            # [修改] 不再此处筛选，全部送入 Phase 4，由 validation_node 内部根据 verdict 和 confidence 筛选
            tasks.append(Send("phase4_validation", {
                "candidates": cands,
                "feature_context": feat,
                "mode": state["mode"],
                "vul_id": state["vul_id"],
            }))
    return tasks


def _calc_p4_expected_tasks(state: WorkflowState) -> int:
    """计算 Phase 4 预期的子图任务数量（与 validation_dispatcher 逻辑一致）"""
    features_map = {f.group_id: f for f in state.get("analyzed_features", [])}
    group_map = defaultdict(list)
    for cand in state.get("search_candidates", []):
        group_map[cand.group_id].append(cand)
    count = 0
    for group_id in group_map.keys():
        if group_id in features_map:
            count += 1
    return count

# --- Finalizers (用于 Reduce & Save) ---

def extraction_subgraph_finalizer(state: PatchExtractionState):
    # 子图内部的打包节点
    feat = PatchFeatures(
        group_id=state["group_id"],
        patches=state["patches"],
        commit_message=state["commit_message"],
        taxonomy=state["taxonomy"],
        slices=state["slices"],
        semantics=state["semantics"]
    )
    return {"analyzed_features": [feat]}

def validation_subgraph_finalizer(state: VerificationState):
    # 子图内部的打包节点，同时返回完成标记用于同步
    return {
        "final_findings": state.get("final_findings", []),
        "p4_done_markers": [1]  # 每个子图完成时添加一个标记
    }

def phase1_checkpoint_node(state: WorkflowState):
    """
    [新增] Phase 1 结束后的收口节点：负责保存和日志。
    LangGraph 会在所有并行子图完成后，且数据聚合到 state 后执行此节点。
    """
    count = len(state.get('grouped_patches', []))
    print(f"[*] Phase 1 Completed. Total patch groups: {count}")
    CheckpointManager.save_pkl(state, f"{state['vul_id']}_phase1.pkl", state["output_dir"])
    return {} # 不修改状态

def phase2_checkpoint_node(state: WorkflowState):
    """
    [新增] Phase 2 结束后的收口节点：负责保存和日志。
    LangGraph 会在所有并行子图完成后，且数据聚合到 state 后执行此节点。
    """
    count = len(state.get('analyzed_features', []))
    print(f"[*] Phase 2 Completed. Total features: {count}")
    CheckpointManager.save_pkl(state, f"{state['vul_id']}_phase2.pkl", state["output_dir"])
    CheckpointManager.save_json(state.get('analyzed_features', []), f"{state['vul_id']}_features.json", state["result_dir"])
    return {} # 不修改状态

def phase3_checkpoint_node(state: WorkflowState):
    """
    [新增] Phase 3 结束后的收口节点。
    """
    count = len(state.get('search_candidates', []))
    print(f"[*] Phase 3 Completed. Total candidates: {count}")
    CheckpointManager.save_pkl(state, f"{state['vul_id']}_{state['mode']}_phase3.pkl", state["output_dir"])
    CheckpointManager.save_json(state.get('search_candidates', []), f"{state['vul_id']}_{state['mode']}_candidates.json", state["result_dir"])
    return {}

def phase4_checkpoint_node(state: WorkflowState):
    """
    [新增] Phase 4 结束后的收口节点。
    使用计数机制确保只在所有子图完成后才保存结果，避免并发子图导致的覆盖问题。
    """
    # 计算预期任务数和已完成任务数
    expected = _calc_p4_expected_tasks(state)
    done = len(state.get('p4_done_markers', []))
    
    # 如果还有子图未完成，仅打印进度，不保存
    if done < expected:
        print(f"[*] Phase 4 Progress: {done}/{expected} subgraphs completed")
        return {}
    
    # 所有子图都完成，执行最终保存
    count = len(state.get('final_findings', []))
    print(f"[*] Phase 4 Completed. Total findings: {count}")
    CheckpointManager.save_pkl(state, f"{state['vul_id']}_{state['mode']}_phase4.pkl", state["output_dir"])
    CheckpointManager.save_json(state.get('final_findings', []), f"{state['vul_id']}_{state['mode']}_findings.json", state["result_dir"])
    return {}

# ==============================================================================
# 2. 图构建逻辑
# ==============================================================================

def build_extraction_subgraph():
    worker = StateGraph(PatchExtractionState)
    worker.add_node("taxonomy", taxonomy_node)
    worker.add_node("slicing", slicing_node)
    worker.add_node("semantic", semantic_node)
    worker.add_node("finalizer", extraction_subgraph_finalizer)
    
    worker.set_entry_point("taxonomy")
    worker.add_edge("taxonomy", "slicing")
    worker.add_edge("slicing", "semantic")
    worker.add_edge("semantic", "finalizer")
    worker.add_edge("finalizer", END)
    return worker.compile()

def build_validation_subgraph():
    worker = StateGraph(VerificationState)
    worker.add_node("verifier", validation_node)
    worker.add_node("finalizer", validation_subgraph_finalizer)
    
    worker.set_entry_point("verifier")
    worker.add_edge("verifier", "finalizer")
    worker.add_edge("finalizer", END)
    return worker.compile()

def build_pipeline():
    workflow = StateGraph(WorkflowState)
    
    # --- 注册节点 ---
    workflow.add_node("loader", loader_node)
    
    # P1
    workflow.add_node("phase1_preprocess", preprocessing_node) 
    workflow.add_node("phase1_grouping", grouping_node)
    workflow.add_node("phase1_checkpoint", phase1_checkpoint_node) # 新增收口点
    
    # P2
    workflow.add_node("phase2_extraction", build_extraction_subgraph())
    workflow.add_node("phase2_checkpoint", phase2_checkpoint_node) # 新增收口点
    
    # P3
    workflow.add_node("phase3_search", matching_node)
    workflow.add_node("phase3_checkpoint", phase3_checkpoint_node) # 新增收口点
    
    # P4
    workflow.add_node("phase4_validation", build_validation_subgraph())
    workflow.add_node("phase4_checkpoint", phase4_checkpoint_node) # 新增收口点
    
    # --- 流程编排 ---
    workflow.set_entry_point("loader")
    
    # 1. Router: After Loader
    def route_loader(state: WorkflowState):
        s = state.get("start_phase", 1)
        e = state.get("end_phase", 4)
        
        # [Fix] If the start phase calculated by loader exceeds end_phase, it means everything is already done.
        if s > e:
            print(f"[*] All requested phases are already completed (Start: {s}, End: {e}). Finishing.")
            return END
        
        if s <= 1: return "phase1_preprocess"
        if s == 2: 
            # 恢复时如果需要分发，必须在这里做，或者跳到一个专门的分发节点
            # 简单起见，如果从 P2 开始，我们复用 extraction_dispatcher
            return extraction_dispatcher(state)
        if s == 3: return "phase3_search"
        if s == 4: 
            tasks = validation_dispatcher(state)
            if not tasks: return "phase4_checkpoint"
            return tasks
        return END

    workflow.add_conditional_edges(
        "loader", 
        route_loader, 
        ["phase1_preprocess", "phase2_extraction", "phase3_search", "phase4_validation", "phase4_checkpoint", END]
    )
    
    # 2. P1 Flow
    workflow.add_edge("phase1_preprocess", "phase1_grouping")
    workflow.add_edge("phase1_grouping", "phase1_checkpoint")
    
    def route_p1(state):
        if state["end_phase"] < 2: return END
        return extraction_dispatcher(state) # Map to P2

    workflow.add_conditional_edges("phase1_checkpoint", route_p1, ["phase2_extraction", END])
    
    # 3. P2 Flow
    # P2 子图并发运行 -> 聚合 -> 进入 Checkpoint Node
    workflow.add_edge("phase2_extraction", "phase2_checkpoint")
    
    def route_p2(state):
        # 纯路由逻辑，不含副作用
        if state["end_phase"] < 3: return END
        return "phase3_search"

    workflow.add_conditional_edges("phase2_checkpoint", route_p2, {"phase3_search": "phase3_search", END: END})
    
    # 4. P3 Flow
    workflow.add_edge("phase3_search", "phase3_checkpoint")
    
    def route_p3(state):
        if state["end_phase"] < 4: return END
        tasks = validation_dispatcher(state) # Map to P4
        if not tasks: return "phase4_checkpoint" # Skip if empty
        return tasks

    workflow.add_conditional_edges(
        "phase3_checkpoint", 
        route_p3, 
        ["phase4_validation", "phase4_checkpoint", END]
    )
    
    # 5. P4 Flow
    workflow.add_edge("phase4_validation", "phase4_checkpoint")
    workflow.add_edge("phase4_checkpoint", END)
    
    return workflow.compile()

def setup_directories(base_output_dir: str, repo_name: str, vul_id: str):
    """
    构建分离式的目录路径：
    - Checkpoints: base/checkpoints/repo/vul_id
    - Results:     base/results/repo
    """
    # 1. Checkpoints 目录（保持深层结构，因为文件多且杂）
    ckpt_dir = os.path.join(base_output_dir, "checkpoints", repo_name, vul_id)
    
    # 2. Results 目录（扁平化结构，方便查看）
    # 这里我们把结果放在 outputs/results/repo_name/ 下
    res_dir = os.path.join(base_output_dir, "results", repo_name)
    
    # 3. 自动创建目录
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    
    return ckpt_dir, res_dir

# ==============================================================================
# 4. 运行入口
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', "--repo", required=True)
    parser.add_argument('-v', "--vul_id", required=True)
    parser.add_argument("-c", "--commit", required=True)
    parser.add_argument('-m', "--mode", type=str, default="repo", choices=["repo", "benchmark"])
    parser.add_argument("-s", "--start", type=int, default=1)
    parser.add_argument("-e", "--end", type=int, default=4)
    parser.add_argument('-o', '--output_dir', type=str, default=OUTPUT_DIR_PATH)
    parser.add_argument('-f', '--force', action='store_true', help="Force execution even if checkpoint exists")
    
    args = parser.parse_args()
    
    load_dotenv()
    
    repo_path = os.path.join(REPO_DIR_PATH, args.repo)
    checkpoint_dir, result_dir = setup_directories(args.output_dir, args.repo, args.vul_id)
    
    print(f"=== Starting Pipeline for {args.vul_id} ===")
    print(f"[*] Checkpoints Dir: {checkpoint_dir}")
    print(f"[*] Results Dir:     {result_dir}")
    
    # 初始化 State
    initial_state = {
        "repo_name": args.repo,
        "vul_id": args.vul_id,
        "commit_hash": args.commit,
        "output_dir": checkpoint_dir,
        "result_dir": result_dir,
        "mode": args.mode,
        "repo_path": repo_path,
        "start_phase": args.start,
        "end_phase": args.end,
        "force_execution": args.force,
        "lang": "c",  # 假设 C 语言
        # 列表字段初始化为空，防止 append 错误
        "atomic_patches": [],
        "grouped_patches": [],
        "analyzed_features": [],
        "search_candidates": [],
        "final_findings": [],
        # Phase 4 同步控制
        "p4_total_tasks": 0,
        "p4_done_markers": []
    }
    
    print(f"=== Starting Pipeline for {args.vul_id} ===")
    
    app = build_pipeline()
    # 增加 recursion_limit 因为 map-reduce 步骤多
    result = app.invoke(initial_state, {"recursion_limit": 100})
    
    print("=== Pipeline Finished ===")
    if args.end == 4:
        print(f"Findings: {len(result.get('final_findings', []))}")