import os
import argparse
import uuid
from typing import List, Dict, Any
from collections import defaultdict

from langgraph.graph import StateGraph, END
from langgraph.types import Send

from core.state import WorkflowState, PatchExtractionState, VerificationState
from core.checkpoint import CheckpointManager
from core.models import PatchFeatures

# Import Standard Nodes
from preprocessing.denoising import preprocessing_node, baseline_phase1_node
from preprocessing.grouping import grouping_node
from extraction.taxonomy import taxonomy_node
from extraction.slicer import slicing_node, baseline_extraction_node
from extraction.semantics import semantic_node
from search.matcher import matching_node 
from search.verifier import validation_node, baseline_validation_node

from dotenv import load_dotenv
from core.configs import REPO_DIR_PATH, OUTPUT_DIR_PATH

# ==============================================================================
# 0. Baseline Nodes (Ablation Placeholders)
# ==============================================================================

# ==============================================================================
# 1. Helper Nodes (Copied from run.py for independence)
# ==============================================================================

def loader_node(state: WorkflowState):
    """
    Smart Loader: Checks for existings checkpoints to skip phases.
    (Simplified copy from run.py)
    """
    vul_id = state["vul_id"]
    current_phase = state["start_phase"]
    end_phase = state["end_phase"]
    mode = state["mode"]
    output_dir = state["output_dir"]
    force = state.get("force_execution", False)
    updates = {} 

    ablation = state.get("ablation", "none")
    
    # [Important] If ablation is active (e.g. phase4), check output_dir (checkpoints_ablation4) first.
    # If not found there, we MUST look into the STANDARD checkpoints directory (checkpoints) to reuse prior work!
    # Because output_dir is already set to checkpoints_ablation4 by setup_directories.
    
    # Resolve the path to the STANDARD checkpoints directory
    # output_dir is currently .../checkpoints_ablation4/repo/vul_id
    # We need .../checkpoints/repo/vul_id
    base_checkpoints_dir = output_dir
    if ablation != "none":
        # Replace the last occurrence of 'checkpoints_ablationX' with 'checkpoints'
        # Path manipulation: 
        # output_dir = /path/to/checkpoints_ablation4/linux/CVE-123
        parts = output_dir.split(os.sep)
        try:
            # Find the index of the ablation folder
            ablation_folder_name = f"checkpoints_{ablation.replace('phase', 'ablation')}"
            if ablation_folder_name in parts:
                idx = parts.index(ablation_folder_name)
                parts[idx] = "checkpoints"
                base_checkpoints_dir = os.sep.join(parts)
        except:
            print("    [Warning] Could not resolve standard checkpoints dir automatically.")

    print(f"[*] Pipeline Init: {vul_id} | Start: P{current_phase} | Ablation: {ablation}")
    print(f"    Standard Checkpoints Source: {base_checkpoints_dir}")

    while current_phase <= end_phase:
        skip_current = False
        
        # Phase 1
        if current_phase == 1:
            p1_suffix = f"_{ablation}" if ablation == "phase1" else ""
            p1_filename = f"{vul_id}_phase1.pkl"
            
            # Check output_dir first (if we already ran this ablation)
            if os.path.exists(os.path.join(output_dir, p1_filename)) and not force:
                print(f"    [Skip] Phase 1 found in ablation dir.")
                data = CheckpointManager.load_pkl(p1_filename, output_dir)
                updates["grouped_patches"] = data.get("grouped_patches", [])
                skip_current = True
            # If not, and this phase is NOT being ablated (e.g. we are ablating P4, so P1 is standard), check standard dir
            elif ablation != "phase1" and not force:
                 standard_p1 = f"{vul_id}_phase1.pkl"
                 if os.path.exists(os.path.join(base_checkpoints_dir, standard_p1)):
                    print(f"    [Skip] Phase 1 found in standard dir (Reusing).")
                    data = CheckpointManager.load_pkl(standard_p1, base_checkpoints_dir) # Load from standard
                    updates["grouped_patches"] = data.get("grouped_patches", [])
                    skip_current = True
        
        # Phase 2
        elif current_phase == 2:
            p2_suffix = f"_{ablation}" if ablation == "phase2" else ""
            p2_filename = f"{vul_id}_phase2.pkl"
            
            if os.path.exists(os.path.join(output_dir, p2_filename)) and not force:
                print(f"    [Skip] Phase 2 found in ablation dir.")
                data = CheckpointManager.load_pkl(p2_filename, output_dir)
                updates["analyzed_features"] = data.get("analyzed_features", [])
                skip_current = True
            elif ablation != "phase2" and not force:
                 standard_p2 = f"{vul_id}_phase2.pkl"
                 if os.path.exists(os.path.join(base_checkpoints_dir, standard_p2)):
                    print(f"    [Skip] Phase 2 found in standard dir (Reusing).")
                    data = CheckpointManager.load_pkl(standard_p2, base_checkpoints_dir)
                    updates["analyzed_features"] = data.get("analyzed_features", [])
                    skip_current = True
            
            # If we decide to RUN Phase 2 (not skipping), we need P1 data loaded
            if not skip_current:
                # Load P1 from ablation dir (if ablated) OR standard dir 
                target_p1_dir = output_dir if ablation == "phase1" else base_checkpoints_dir
                target_p1_file = f"{vul_id}_{ablation}_phase1.pkl" if ablation == "phase1" else f"{vul_id}_phase1.pkl"
                
                if "grouped_patches" not in updates:
                     if os.path.exists(os.path.join(target_p1_dir, target_p1_file)):
                        p1 = CheckpointManager.load_pkl(target_p1_file, target_p1_dir)
                        updates["grouped_patches"] = p1.get("grouped_patches")

        # Phase 3
        elif current_phase == 3:
            p3_filename = f"{vul_id}_{mode}_phase3.pkl" 
            
            if os.path.exists(os.path.join(output_dir, p3_filename)) and not force:
                print(f"    [Skip] Phase 3 found in ablation dir.") # Unlikely unless copied
                data = CheckpointManager.load_pkl(p3_filename, output_dir)
                updates["search_candidates"] = data.get("search_candidates", [])
                skip_current = True
            # Reuse Standard Phase 3
            elif not force:
                 if os.path.exists(os.path.join(base_checkpoints_dir, p3_filename)):
                    print(f"    [Skip] Phase 3 found in standard dir (Reusing).")
                    data = CheckpointManager.load_pkl(p3_filename, base_checkpoints_dir)
                    updates["search_candidates"] = data.get("search_candidates", [])
                    skip_current = True

            if not skip_current:
                 # Ensure P2 exists
                 target_p2_dir = output_dir if ablation == "phase2" else base_checkpoints_dir
                 target_p2_file = f"{vul_id}_{ablation}_phase2.pkl" if ablation == "phase2" else f"{vul_id}_phase2.pkl"
                 if "analyzed_features" not in updates:
                    p2 = CheckpointManager.load_pkl(target_p2_file, target_p2_dir)
                    updates["analyzed_features"] = p2.get("analyzed_features", [])

        # Phase 4
        elif current_phase == 4:
            p4_suffix = f"_{ablation}" if ablation != "none" else ""
            p4_filename = f"{vul_id}_{mode}_phase4.pkl"
            
            if os.path.exists(os.path.join(output_dir, p4_filename)) and not force:
                print(f"    [Skip] Phase 4 result found ({p4_filename}).")
                skip_current = True
            else:
                # Load dependencies for RUNNING Phase 4
                # P2
                target_p2_dir = output_dir if ablation == "phase2" else base_checkpoints_dir
                target_p2_file = f"{vul_id}_{ablation}_phase2.pkl" if ablation == "phase2" else f"{vul_id}_phase2.pkl"
                if "analyzed_features" not in updates:
                    p2 = CheckpointManager.load_pkl(target_p2_file, target_p2_dir)
                    updates["analyzed_features"] = p2.get("analyzed_features", [])
                
                # P3
                # Assuming P3 never ablated (standard)
                target_p3_dir = base_checkpoints_dir
                target_p3_file = f"{vul_id}_{mode}_phase3.pkl"
                if "search_candidates" not in updates:
                    p3 = CheckpointManager.load_pkl(target_p3_file, target_p3_dir)
                    updates["search_candidates"] = p3.get("search_candidates", [])

        if skip_current:
            current_phase += 1
        else:
            break
            
    updates["start_phase"] = current_phase
    print(f"[*] Pipeline Logic: Jumping to Phase {current_phase}")
    return updates

def extraction_dispatcher(state: WorkflowState):
    tasks = []
    # If using baseline phase 1, grouped_patches structure might differ?
    # Provided baseline_phase1 returns List[List[Patch]], it's compatible.
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
    group_map = defaultdict(list)
    for cand in state["search_candidates"]:
        group_map[cand.group_id].append(cand)
    for group_id, cands in group_map.items():
        feat = features_map.get(group_id)
        if feat:
            tasks.append(Send("phase4_validation", {
                "candidates": cands,
                "feature_context": feat,
                "mode": state["mode"],
                "vul_id": state["vul_id"],
            }))
    return tasks

def extraction_subgraph_finalizer(state: PatchExtractionState):
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
    return {"final_findings": state.get("final_findings", [])}

def phase1_checkpoint_node(state: WorkflowState):
    ablation = state.get("ablation", "none")
    p1_suffix = f"_{ablation}" if ablation == "phase1" else ""
    p1_filename = f"{state['vul_id']}_phase1.pkl"
    
    count = len(state.get('grouped_patches', []))
    print(f"[*] Phase 1 Completed. Total patch groups: {count}")
    CheckpointManager.save_pkl(state, p1_filename, state["output_dir"])
    return {}

def phase2_checkpoint_node(state: WorkflowState):
    ablation = state.get("ablation", "none")
    p2_suffix = f"_{ablation}" if ablation == "phase2" else ""
    p2_filename = f"{state['vul_id']}_phase2.pkl"
    p2_json = f"{state['vul_id']}_features.json"
    
    count = len(state.get('analyzed_features', []))
    print(f"[*] Phase 2 Completed. Total features: {count}")
    CheckpointManager.save_pkl(state, p2_filename, state["output_dir"])
    if ablation == "phase2":
        CheckpointManager.save_json(state.get('analyzed_features', []), p2_json, state["result_dir"])
    return {}

def phase3_checkpoint_node(state: WorkflowState):
    # Usually no ablation for P3, but if P2 was ablated, P3 candidates might differ?
    # For now, keep standard naming unless we explicitly add P3 ablation later.
    count = len(state.get('search_candidates', []))
    print(f"[*] Phase 3 Completed. Total candidates: {count}")
    CheckpointManager.save_pkl(state, f"{state['vul_id']}_{state['mode']}_phase3.pkl", state["output_dir"])
    return {}

def phase4_checkpoint_node(state: WorkflowState):
    ablation = state.get("ablation", "none")
    p4_suffix = f"_{ablation}" if ablation != "none" else ""
    
    p4_filename = f"{state['vul_id']}_{state['mode']}_phase4.pkl"
    p4_json = f"{state['vul_id']}_{state['mode']}_findings.json"
    
    count = len(state.get('final_findings', []))
    print(f"[*] Phase 4 Completed. Total findings: {count}")
    CheckpointManager.save_pkl(state, p4_filename, state["output_dir"])
    CheckpointManager.save_json(state.get('final_findings', []), p4_json, state["result_dir"])
    return {}

# ==============================================================================
# 2. Graph Construction with Ablation Logic
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

def build_validation_subgraph(use_baseline_phase4: bool):
    worker = StateGraph(VerificationState)
    
    if use_baseline_phase4:
        worker.add_node("verifier", baseline_validation_node)
    else:
        worker.add_node("verifier", validation_node)
        
    worker.add_node("finalizer", validation_subgraph_finalizer)
    
    worker.set_entry_point("verifier")
    worker.add_edge("verifier", "finalizer")
    worker.add_edge("finalizer", END)
    return worker.compile()

def build_ablation_pipeline(ablation_type: str):
    """
    Constructs the pipeline based on the selected ablation type.
    ablation_type: 'phase1', 'phase2', 'phase4', or 'none' (standard)
    """
    print(f"[*] Building Pipeline with Ablation: {ablation_type}")
    
    workflow = StateGraph(WorkflowState)
    workflow.add_node("loader", loader_node)
    
    # --- P1 Construction ---
    if ablation_type == 'phase1':
        # Baseline phase1: skip preprocessing, only add grouping node
        workflow.add_node("phase1_preprocess", baseline_phase1_node)
    else:
        workflow.add_node("phase1_preprocess", preprocessing_node)
        workflow.add_node("phase1_grouping", grouping_node)
    workflow.add_node("phase1_checkpoint", phase1_checkpoint_node)
    
    # --- P2 Construction ---
    if ablation_type == 'phase2':
        # Baseline Phase 2: Use Agent-based total extraction (integrated slicing/semantics)
        workflow.add_node("phase2_extraction", baseline_extraction_node)
    else:
        # Standard Phase 2: Taxonomy -> Slicing -> Semantics sub-graph
        workflow.add_node("phase2_extraction", build_extraction_subgraph())
        
    workflow.add_node("phase2_checkpoint", phase2_checkpoint_node)
    
    # --- P3 Construction ---
    # (Assuming no baseline for P3 requested yet, using standard matching)
    workflow.add_node("phase3_search", matching_node)
    workflow.add_node("phase3_checkpoint", phase3_checkpoint_node)
    
    # --- P4 Construction ---
    use_baseline_p4 = (ablation_type == 'phase4')
    workflow.add_node("phase4_validation", build_validation_subgraph(use_baseline_p4))
    workflow.add_node("phase4_checkpoint", phase4_checkpoint_node)

    # --- Edge Routing (Same as Standard) ---
    workflow.set_entry_point("loader")

    def route_loader(state: WorkflowState):
        s = state.get("start_phase", 1)
        e = state.get("end_phase", 4)
        
        # [Fix] If the start phase calculated by loader exceeds end_phase, it means everything is already done.
        if s > e:
            print(f"[*] All requested phases are already completed (Start: {s}, End: {e}). Finishing.")
            return END
            
        if s <= 1: return "phase1_preprocess"
        if s == 2: return extraction_dispatcher(state)
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
    
    if ablation_type == 'phase1':
        # Directly connect grouping to checkpoint
        workflow.add_edge("phase1_preprocess", "phase1_checkpoint")
    else:
        workflow.add_edge("phase1_preprocess", "phase1_grouping")
        workflow.add_edge("phase1_grouping", "phase1_checkpoint")
    def route_p1(state):
        if state["end_phase"] < 2: return END
        return extraction_dispatcher(state) 
    workflow.add_conditional_edges("phase1_checkpoint", route_p1, ["phase2_extraction", END])
    
    workflow.add_edge("phase2_extraction", "phase2_checkpoint")
    
    def route_p2(state):
        if state["end_phase"] < 3: return END
        return "phase3_search"

    workflow.add_conditional_edges("phase2_checkpoint", route_p2, {"phase3_search": "phase3_search", END: END})
    
    workflow.add_edge("phase3_search", "phase3_checkpoint")
    
    def route_p3(state):
        if state["end_phase"] < 4: return END
        tasks = validation_dispatcher(state)
        if not tasks: return "phase4_checkpoint"
        return tasks

    workflow.add_conditional_edges(
        "phase3_checkpoint", 
        route_p3, 
        ["phase4_validation", "phase4_checkpoint", END]
    )
    
    workflow.add_edge("phase4_validation", "phase4_checkpoint")
    workflow.add_edge("phase4_checkpoint", END)
    
    return workflow.compile()

def setup_directories(base_output_dir: str, repo_name: str, vul_id: str, ablation: str = "none"):
    # Determine base folder names
    ckpt_folder_name = "checkpoints"
    res_folder_name = "results"
    
    if ablation != "none":
        # e.g., checkpoints_phase4 -> user asked for checkpoints_ablation4? 
        # User request: "checkpoints_ablation4", "results_ablation4"
        # However, ablation arg is 'phase4' (from choices).
        # Let's map 'phase4' -> 'ablation4' to match your requested naming convention.
        suffix = ablation.replace("phase", "ablation") 
        ckpt_folder_name = f"checkpoints_{suffix}"
        res_folder_name = f"results_{suffix}"

    ckpt_dir = os.path.join(base_output_dir, ckpt_folder_name, repo_name, vul_id)
    res_dir = os.path.join(base_output_dir, res_folder_name, repo_name)
    
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    return ckpt_dir, res_dir

# ==============================================================================
# Main Execution
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Ablation Studies for Vulnerability Analysis")
    parser.add_argument('-r', "--repo", required=True)
    parser.add_argument('-v', "--vul_id", required=True)
    parser.add_argument("-c", "--commit", required=True)
    parser.add_argument('-m', "--mode", type=str, default="repo", choices=["repo", "benchmark"])
    parser.add_argument("-s", "--start", type=int, default=1)
    parser.add_argument("-e", "--end", type=int, default=4)
    parser.add_argument('-o', '--output_dir', type=str, default=OUTPUT_DIR_PATH)
    parser.add_argument('-f', '--force', action='store_true', help="Force execution even if checkpoint exists")
    
    # New Argument for Ablation
    parser.add_argument('--ablation', type=str, default='none', 
                        choices=['none', 'phase1', 'phase2', 'phase4'],
                        help="Select ablation experiment type")
    
    args = parser.parse_args()
    load_dotenv()
    
    repo_path = os.path.join(REPO_DIR_PATH, args.repo)
    checkpoint_dir, result_dir = setup_directories(args.output_dir, args.repo, args.vul_id, args.ablation)
    
    # Append ablation suffix to result filenames to avoid overwriting standard runs
    if args.ablation != 'none':
        print(f"!!! ABLATION MODE: {args.ablation} !!!")
        # You might want to adjust checkpoint_dir or result_dir here if you want separate storage
        # For now, we keep same checkpoints (to reuse standard P1/P2/P3 outputs)
        # but maybe we should ensure we don't pollute standard checkpoints with baseline data?
        # User said "Phases 1-3 are reused", implying we reuse standard data.
        pass

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
        "ablation": args.ablation, # Inject into state key matching TypedDict
        "lang": "c",
        "atomic_patches": [],
        "grouped_patches": [],
        "analyzed_features": [],
        "search_candidates": [],
        "final_findings": []
    }
    
    app = build_ablation_pipeline(args.ablation)
    result = app.invoke(initial_state, {"recursion_limit": 100})
    print("=== Ablation Run Finished ===")
