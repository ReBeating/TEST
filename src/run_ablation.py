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
    Smart Loader: Checks for existing checkpoints to skip phases.
    
    For ablation experiments:
    - Phase 1 & 3: Always use standard version (can reuse across all ablations)
    - Phase 2 & 4: May use baseline or full version depending on ablation type
    """
    vul_id = state["vul_id"]
    current_phase = state["start_phase"]
    end_phase = state["end_phase"]
    mode = state["mode"]
    output_dir = state["output_dir"]
    force = state.get("force_execution", False)
    updates = {}

    ablation = state.get("ablation", "none")
    
    # Determine ablation configuration
    ablation_map = {
        'ablation1': {'p2': 'baseline', 'p4': 'baseline'},
        'plain': {'p2': 'baseline', 'p4': 'baseline'},
        'ablation2': {'p2': 'full', 'p4': 'baseline'},
        'with_extraction': {'p2': 'full', 'p4': 'baseline'},
        'ablation3': {'p2': 'baseline', 'p4': 'full'},
        'with_validation': {'p2': 'baseline', 'p4': 'full'},
        'ablation4': {'p2': 'full', 'p4': 'full'},
        'full': {'p2': 'full', 'p4': 'full'},
        'none': {'p2': 'full', 'p4': 'full'},
        'phase2': {'p2': 'baseline', 'p4': 'full'},
        'phase4': {'p2': 'full', 'p4': 'baseline'},
    }
    config = ablation_map.get(ablation, {'p2': 'full', 'p4': 'full'})
    
    # Resolve the path to the STANDARD checkpoints directory for P1/P3 reuse
    base_checkpoints_dir = output_dir
    if ablation not in ["none", "full", "ablation4"]:
        # Need to find standard checkpoints for P1/P3 reuse
        parts = output_dir.split(os.sep)
        for i, part in enumerate(parts):
            if part.startswith("checkpoints_ablation"):
                parts[i] = "checkpoints"
                base_checkpoints_dir = os.sep.join(parts)
                break

    print(f"[*] Pipeline Init: {vul_id} | Start: P{current_phase} | Ablation: {ablation}")
    if base_checkpoints_dir != output_dir:
        print(f"    Standard Checkpoints: {base_checkpoints_dir}")

    while current_phase <= end_phase:
        skip_current = False
        
        # Phase 1 (Always Full - can reuse from standard checkpoints)
        if current_phase == 1:
            p1_filename = f"{vul_id}_phase1.pkl"
            
            # Try current dir first
            if os.path.exists(os.path.join(output_dir, p1_filename)) and not force:
                print(f"    [Skip] Phase 1 found in current dir.")
                data = CheckpointManager.load_pkl(p1_filename, output_dir)
                updates["grouped_patches"] = data.get("grouped_patches", [])
                skip_current = True
            # Try standard dir for reuse
            elif not force and os.path.exists(os.path.join(base_checkpoints_dir, p1_filename)):
                print(f"    [Skip] Phase 1 found in standard dir (Reusing).")
                data = CheckpointManager.load_pkl(p1_filename, base_checkpoints_dir)
                updates["grouped_patches"] = data.get("grouped_patches", [])
                skip_current = True
        
        # Phase 2 (May be baseline or full depending on ablation)
        elif current_phase == 2:
            p2_filename = f"{vul_id}_phase2.pkl"
            
            # Try current dir first
            if os.path.exists(os.path.join(output_dir, p2_filename)) and not force:
                print(f"    [Skip] Phase 2 found in current dir.")
                data = CheckpointManager.load_pkl(p2_filename, output_dir)
                updates["analyzed_features"] = data.get("analyzed_features", [])
                skip_current = True
            # If using full P2, can try to reuse from standard checkpoints
            elif config['p2'] == 'full' and not force:
                if os.path.exists(os.path.join(base_checkpoints_dir, p2_filename)):
                    print(f"    [Skip] Phase 2 (full) found in standard dir (Reusing).")
                    data = CheckpointManager.load_pkl(p2_filename, base_checkpoints_dir)
                    updates["analyzed_features"] = data.get("analyzed_features", [])
                    skip_current = True
            
            # If we decide to RUN Phase 2, ensure P1 data is loaded
            if not skip_current and "grouped_patches" not in updates:
                p1_file = f"{vul_id}_phase1.pkl"
                # Try current dir first, then standard
                for check_dir in [output_dir, base_checkpoints_dir]:
                    if os.path.exists(os.path.join(check_dir, p1_file)):
                        p1 = CheckpointManager.load_pkl(p1_file, check_dir)
                        updates["grouped_patches"] = p1.get("grouped_patches", [])
                        break

        # Phase 3 (Always Full - can reuse from standard checkpoints)
        elif current_phase == 3:
            p3_filename = f"{vul_id}_{mode}_phase3.pkl"
            
            # Try current dir first
            if os.path.exists(os.path.join(output_dir, p3_filename)) and not force:
                print(f"    [Skip] Phase 3 found in current dir.")
                data = CheckpointManager.load_pkl(p3_filename, output_dir)
                updates["search_candidates"] = data.get("search_candidates", [])
                skip_current = True
            # Try standard dir for reuse
            elif not force and os.path.exists(os.path.join(base_checkpoints_dir, p3_filename)):
                print(f"    [Skip] Phase 3 found in standard dir (Reusing).")
                data = CheckpointManager.load_pkl(p3_filename, base_checkpoints_dir)
                updates["search_candidates"] = data.get("search_candidates", [])
                skip_current = True

            # If we decide to RUN Phase 3, ensure P2 data is loaded
            if not skip_current and "analyzed_features" not in updates:
                p2_file = f"{vul_id}_phase2.pkl"
                # Try current dir first, then standard
                for check_dir in [output_dir, base_checkpoints_dir]:
                    if os.path.exists(os.path.join(check_dir, p2_file)):
                        p2 = CheckpointManager.load_pkl(p2_file, check_dir)
                        updates["analyzed_features"] = p2.get("analyzed_features", [])
                        break

        # Phase 4 (May be baseline or full depending on ablation)
        elif current_phase == 4:
            p4_filename = f"{vul_id}_{mode}_phase4.pkl"
            
            # Try current dir first
            if os.path.exists(os.path.join(output_dir, p4_filename)) and not force:
                print(f"    [Skip] Phase 4 found in current dir.")
                skip_current = True
            # If using full P4, can try to reuse from standard checkpoints
            elif config['p4'] == 'full' and not force:
                if os.path.exists(os.path.join(base_checkpoints_dir, p4_filename)):
                    print(f"    [Skip] Phase 4 (full) found in standard dir (Reusing).")
                    skip_current = True
            
            # If we decide to RUN Phase 4, ensure P2 and P3 data are loaded
            if not skip_current:
                # Load P2
                if "analyzed_features" not in updates:
                    p2_file = f"{vul_id}_phase2.pkl"
                    for check_dir in [output_dir, base_checkpoints_dir]:
                        if os.path.exists(os.path.join(check_dir, p2_file)):
                            p2 = CheckpointManager.load_pkl(p2_file, check_dir)
                            updates["analyzed_features"] = p2.get("analyzed_features", [])
                            break
                
                # Load P3
                if "search_candidates" not in updates:
                    p3_file = f"{vul_id}_{mode}_phase3.pkl"
                    for check_dir in [output_dir, base_checkpoints_dir]:
                        if os.path.exists(os.path.join(check_dir, p3_file)):
                            p3 = CheckpointManager.load_pkl(p3_file, check_dir)
                            updates["search_candidates"] = p3.get("search_candidates", [])
                            break

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
    """Phase 1 checkpoint - always uses full version"""
    p1_filename = f"{state['vul_id']}_phase1.pkl"
    
    count = len(state.get('grouped_patches', []))
    print(f"[*] Phase 1 Completed. Total patch groups: {count}")
    CheckpointManager.save_pkl(state, p1_filename, state["output_dir"])
    return {}

def phase2_checkpoint_node(state: WorkflowState):
    """Phase 2 checkpoint - may be baseline or full depending on ablation"""
    p2_filename = f"{state['vul_id']}_phase2.pkl"
    p2_json = f"{state['vul_id']}_features.json"
    
    count = len(state.get('analyzed_features', []))
    print(f"[*] Phase 2 Completed. Total features: {count}")
    CheckpointManager.save_pkl(state, p2_filename, state["output_dir"])
    # Always save JSON for easier inspection
    CheckpointManager.save_json(state.get('analyzed_features', []), p2_json, state["result_dir"])
    return {}

def phase3_checkpoint_node(state: WorkflowState):
    """Phase 3 checkpoint - always uses full version"""
    count = len(state.get('search_candidates', []))
    print(f"[*] Phase 3 Completed. Total candidates: {count}")
    CheckpointManager.save_pkl(state, f"{state['vul_id']}_{state['mode']}_phase3.pkl", state["output_dir"])
    return {}

def phase4_checkpoint_node(state: WorkflowState):
    """Phase 4 checkpoint - may be baseline or full depending on ablation"""
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
    
    Ablation Design (2x2 Combinations):
    - Phase 1 & 3: Always use full version (no ablation)
    - Phase 2: baseline or full
    - Phase 4: baseline or full
    
    ablation_type options:
    - 'ablation1' or 'plain': P2 baseline + P4 baseline (最简版本)
    - 'ablation2' or 'with_extraction': P2 full + P4 baseline (+提取)
    - 'ablation3' or 'with_validation': P2 baseline + P4 full (+验证)
    - 'ablation4' or 'full' or 'none': P2 full + P4 full (完整版)
    
    Legacy options (for backward compatibility):
    - 'phase2': Same as ablation3 (P2 baseline)
    - 'phase4': Same as ablation2 (P4 baseline)
    """
    print(f"[*] Building Pipeline with Ablation: {ablation_type}")
    
    # Map ablation types to P2/P4 configuration
    ablation_map = {
        'ablation1': {'p2': 'baseline', 'p4': 'baseline'},  # plain
        'plain': {'p2': 'baseline', 'p4': 'baseline'},
        'ablation2': {'p2': 'full', 'p4': 'baseline'},      # +提取
        'with_extraction': {'p2': 'full', 'p4': 'baseline'},
        'ablation3': {'p2': 'baseline', 'p4': 'full'},      # +验证
        'with_validation': {'p2': 'baseline', 'p4': 'full'},
        'ablation4': {'p2': 'full', 'p4': 'full'},          # 完整
        'full': {'p2': 'full', 'p4': 'full'},
        'none': {'p2': 'full', 'p4': 'full'},
        # Legacy compatibility
        'phase2': {'p2': 'baseline', 'p4': 'full'},  # Only P2 baseline
        'phase4': {'p2': 'full', 'p4': 'baseline'},  # Only P4 baseline
    }
    
    config = ablation_map.get(ablation_type, {'p2': 'full', 'p4': 'full'})
    use_baseline_p2 = (config['p2'] == 'baseline')
    use_baseline_p4 = (config['p4'] == 'baseline')
    
    print(f"    [Config] Phase 2: {'BASELINE' if use_baseline_p2 else 'FULL'}, Phase 4: {'BASELINE' if use_baseline_p4 else 'FULL'}")
    
    workflow = StateGraph(WorkflowState)
    workflow.add_node("loader", loader_node)
    
    # --- P1 Construction (Always Full) ---
    workflow.add_node("phase1_preprocess", preprocessing_node)
    workflow.add_node("phase1_grouping", grouping_node)
    workflow.add_node("phase1_checkpoint", phase1_checkpoint_node)
    
    # --- P2 Construction (Baseline or Full) ---
    if use_baseline_p2:
        # Baseline Phase 2: Use Agent-based total extraction (integrated slicing/semantics)
        workflow.add_node("phase2_extraction", baseline_extraction_node)
    else:
        # Standard Phase 2: Taxonomy -> Slicing -> Semantics sub-graph
        workflow.add_node("phase2_extraction", build_extraction_subgraph())
        
    workflow.add_node("phase2_checkpoint", phase2_checkpoint_node)
    
    # --- P3 Construction (Always Full) ---
    workflow.add_node("phase3_search", matching_node)
    workflow.add_node("phase3_checkpoint", phase3_checkpoint_node)
    
    # --- P4 Construction (Baseline or Full) ---
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
    
    # P1 always uses full version (preprocessing + grouping)
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
    """
    Create directory structure for ablation experiments.
    
    Directory naming convention:
    - none/full/ablation4: checkpoints/, results/ (standard full version)
    - ablation1/plain: checkpoints_ablation1/, results_ablation1/
    - ablation2/with_extraction: checkpoints_ablation2/, results_ablation2/
    - ablation3/with_validation: checkpoints_ablation3/, results_ablation3/
    """
    # Determine base folder names
    ckpt_folder_name = "checkpoints"
    res_folder_name = "results"
    
    # Map ablation type to directory suffix
    ablation_dir_map = {
        'ablation1': 'ablation1',
        'plain': 'ablation1',
        'ablation2': 'ablation2',
        'with_extraction': 'ablation2',
        'ablation3': 'ablation3',
        'with_validation': 'ablation3',
        'ablation4': '',  # Use standard directories
        'full': '',
        'none': '',
        # Legacy compatibility
        'phase2': 'ablation3',  # P2 baseline
        'phase4': 'ablation2',  # P4 baseline
    }
    
    suffix = ablation_dir_map.get(ablation, '')
    if suffix:
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
    
    # Ablation Experiment Argument
    # 2x2 Combinations: P2 (baseline/full) × P4 (baseline/full)
    parser.add_argument('--ablation', type=str, default='none',
                        choices=['none', 'full',
                                 'ablation1', 'plain',           # P2=baseline, P4=baseline
                                 'ablation2', 'with_extraction', # P2=full, P4=baseline
                                 'ablation3', 'with_validation', # P2=baseline, P4=full
                                 'ablation4',                     # P2=full, P4=full (same as none/full)
                                 'phase2', 'phase4'],            # Legacy options
                        help="Select ablation experiment type: "
                             "ablation1 (plain), ablation2 (+extraction), "
                             "ablation3 (+validation), ablation4/none (full)")
    
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
