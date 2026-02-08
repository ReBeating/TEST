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

# Import node logic
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
# 1. Node Definitions (Nodes)
# ==============================================================================

# ==============================================================================
# 1. Node Definitions (Nodes) - Modify loader_node
# ==============================================================================

def loader_node(state: WorkflowState):
    """
    Smart Loader:
    1. Supports resuming from breakpoints.
    2. Automatically loads data and skips the stage (fast-forward) if result file exists and force=False.
    """
    vul_id = state["vul_id"]
    current_phase = state["start_phase"]
    end_phase = state["end_phase"]
    mode = state["mode"]
    output_dir = state["output_dir"]
    force = state.get("force_execution", False)
    print(f'Force execution: {force}')
    updates = {} # Used to update State data
    
    print(f"[*] Pipeline Init: {vul_id} | Request Start: P{current_phase} | Force: {force}")

    # Define loop for check and load logic
    # We attempt to fast-forward from current_phase until end_phase
    # Note: We check if the "output file of the current phase" exists
    
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
            # Before entering P2, ensure P1 data is in memory (if jumped from P1, updates already has it; if starting directly from P2, need to reload)
            # The logic here is: if P2 output exists, we load P2 output and skip P2 execution
            path = f"{vul_id}_phase2.pkl"
            full_path = os.path.join(output_dir, path)
            
            # If just fast-forwarded from P1, and P1 data hasn't been loaded for P2 execution?
            # Not needed, because if P2 is also skipped, P1 data is not needed.
            # If P2 is not skipped, state will retain P1's update.

            # We only need to restore from P1.pkl when we actually need to run P2 (if not just loaded)
            # Check if P2 should be skipped first here
            if os.path.exists(full_path) and not force:
                print(f"    [Skip] Phase 2 result found. Loading data...")
                data = CheckpointManager.load_pkl(path, output_dir)
                updates["analyzed_features"] = data.get("analyzed_features", [])
                skip_current = True
            else:
                # Decided to run P2. If P1 data is not in updates (i.e., started directly from P2), need to reload preceding dependencies
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
                # Decided to run P3. Need to ensure P2 data exists
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
                # Decided to run P4. Need P2 (features) and P3 (candidates)
                if "analyzed_features" not in updates:
                    p2_path = f"{vul_id}_phase2.pkl"
                    if os.path.exists(os.path.join(output_dir, p2_path)):
                        updates["analyzed_features"] = CheckpointManager.load_pkl(p2_path, output_dir).get("analyzed_features", [])
                if "search_candidates" not in updates:
                    p3_path = f"{vul_id}_{mode}_phase3.pkl"
                    if os.path.exists(os.path.join(output_dir, p3_path)):
                        updates["search_candidates"] = CheckpointManager.load_pkl(p3_path, output_dir).get("search_candidates", [])

        # --- Decision ---
        if skip_current:
            current_phase += 1 # Fast-forward to next phase
        else:
            break # Cannot skip current phase, stop fast-forwarding, prepare to execute
            
    # Update start_phase in state so router dispatches us to the correct place
    updates["start_phase"] = current_phase
    print(f"[*] Pipeline Logic: Jumping to Phase {current_phase}")
    
    return updates

# --- Dispatchers (For Map) ---

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
    # Aggregate by group_id
    group_map = defaultdict(list)
    for cand in state["search_candidates"]:
        group_map[cand.group_id].append(cand)
    for group_id, cands in group_map.items():
        feat = features_map.get(group_id)
        if feat:
            # [Modify] Do not filter here, send all to Phase 4, validation_node filters internally based on verdict and confidence
            tasks.append(Send("phase4_validation", {
                "candidates": cands,
                "feature_context": feat,
                "mode": state["mode"],
                "vul_id": state["vul_id"],
            }))
    return tasks


def _calc_p4_expected_tasks(state: WorkflowState) -> int:
    """Calculate expected subgraph task count for Phase 4 (consistent with validation_dispatcher logic)"""
    features_map = {f.group_id: f for f in state.get("analyzed_features", [])}
    group_map = defaultdict(list)
    for cand in state.get("search_candidates", []):
        group_map[cand.group_id].append(cand)
    count = 0
    for group_id in group_map.keys():
        if group_id in features_map:
            count += 1
    return count

# --- Finalizers (For Reduce & Save) ---

def extraction_subgraph_finalizer(state: PatchExtractionState):
    # Packing node within subgraph
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
    # Packing node within subgraph, also returns completion marker for synchronization
    return {
        "final_findings": state.get("final_findings", []),
        "p4_done_markers": [1]  # Add a marker when each subgraph completes
    }

def phase1_checkpoint_node(state: WorkflowState):
    """
    [New] Closing node after Phase 1 ends: responsible for saving and logging.
    LangGraph executes this node after all parallel subgraphs complete and data is aggregated into state.
    """
    count = len(state.get('grouped_patches', []))
    print(f"[*] Phase 1 Completed. Total patch groups: {count}")
    CheckpointManager.save_pkl(state, f"{state['vul_id']}_phase1.pkl", state["output_dir"])
    return {} # Do not modify state

def phase2_checkpoint_node(state: WorkflowState):
    """
    [New] Closing node after Phase 2 ends: responsible for saving and logging.
    LangGraph executes this node after all parallel subgraphs complete and data is aggregated into state.
    """
    count = len(state.get('analyzed_features', []))
    print(f"[*] Phase 2 Completed. Total features: {count}")
    CheckpointManager.save_pkl(state, f"{state['vul_id']}_phase2.pkl", state["output_dir"])
    CheckpointManager.save_json(state.get('analyzed_features', []), f"{state['vul_id']}_features.json", state["result_dir"])
    return {} # Do not modify state

def phase3_checkpoint_node(state: WorkflowState):
    """
    [New] Closing node after Phase 3 ends.
    """
    count = len(state.get('search_candidates', []))
    print(f"[*] Phase 3 Completed. Total candidates: {count}")
    CheckpointManager.save_pkl(state, f"{state['vul_id']}_{state['mode']}_phase3.pkl", state["output_dir"])
    CheckpointManager.save_json(state.get('search_candidates', []), f"{state['vul_id']}_{state['mode']}_candidates.json", state["result_dir"])
    return {}

def phase4_checkpoint_node(state: WorkflowState):
    """
    [New] Closing node after Phase 4 ends.
    Use counting mechanism to ensure saving results only after all subgraphs complete, avoiding overwrite issues caused by concurrent subgraphs.
    """
    # Calculate expected task count and completed task count
    expected = _calc_p4_expected_tasks(state)
    done = len(state.get('p4_done_markers', []))
    
    # If there are unfinished subgraphs, only print progress, do not save
    if done < expected:
        print(f"[*] Phase 4 Progress: {done}/{expected} subgraphs completed")
        return {}
    
    # All subgraphs completed, execute final save
    count = len(state.get('final_findings', []))
    print(f"[*] Phase 4 Completed. Total findings: {count}")
    CheckpointManager.save_pkl(state, f"{state['vul_id']}_{state['mode']}_phase4.pkl", state["output_dir"])
    CheckpointManager.save_json(state.get('final_findings', []), f"{state['vul_id']}_{state['mode']}_findings.json", state["result_dir"])
    return {}

# ==============================================================================
# 2. Graph Construction Logic
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
    
    # --- Register Nodes ---
    workflow.add_node("loader", loader_node)
    
    # P1
    workflow.add_node("phase1_preprocess", preprocessing_node) 
    workflow.add_node("phase1_grouping", grouping_node)
    workflow.add_node("phase1_checkpoint", phase1_checkpoint_node) # New closing point
    
    # P2
    workflow.add_node("phase2_extraction", build_extraction_subgraph())
    workflow.add_node("phase2_checkpoint", phase2_checkpoint_node) # New closing point
    
    # P3
    workflow.add_node("phase3_search", matching_node)
    workflow.add_node("phase3_checkpoint", phase3_checkpoint_node) # New closing point
    
    # P4
    workflow.add_node("phase4_validation", build_validation_subgraph())
    workflow.add_node("phase4_checkpoint", phase4_checkpoint_node) # New closing point
    
    # --- Workflow Orchestration ---
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
            # If distribution is needed when resuming, it must be done here or jump to a dedicated dispatcher node
            # For simplicity, if starting from P2, we reuse extraction_dispatcher
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
    # P2 Subgraphs run concurrently -> Aggregate -> Enter Checkpoint Node
    workflow.add_edge("phase2_extraction", "phase2_checkpoint")
    
    def route_p2(state):
        # Pure routing logic, no side effects
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
    Build separated directory paths:
    - Checkpoints: base/checkpoints/repo/vul_id
    - Results:     base/results/repo
    """
    # 1. Checkpoints Directory (Keep deep structure as there are many diverse files)
    ckpt_dir = os.path.join(base_output_dir, "checkpoints", repo_name, vul_id)
    
    # 2. Results Directory (Flat structure for easy viewing)
    # Here we place results under outputs/results/repo_name/
    res_dir = os.path.join(base_output_dir, "results", repo_name)
    
    # 3. Automatically create directories
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    
    return ckpt_dir, res_dir

# ==============================================================================
# 4. Run Entry Point
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
    
    # Initialize State
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
        "lang": "c",  # Assume C language
        # Initialize list fields as empty to prevent append errors
        "atomic_patches": [],
        "grouped_patches": [],
        "analyzed_features": [],
        "search_candidates": [],
        "final_findings": [],
        # Phase 4 Synchronization Control
        "p4_total_tasks": 0,
        "p4_done_markers": []
    }
    
    print(f"=== Starting Pipeline for {args.vul_id} ===")
    
    app = build_pipeline()
    # Increase recursion_limit due to many map-reduce steps
    result = app.invoke(initial_state, {"recursion_limit": 100})
    
    print("=== Pipeline Finished ===")
    if args.end == 4:
        print(f"Findings: {len(result.get('final_findings', []))}")