import networkx as nx
from typing import List, Dict, Set
from collections import defaultdict
from dataclasses import dataclass, field

# Tree-sitter
try:
    from tree_sitter import Language, Parser
    import tree_sitter_c # Use c parser to parse C code
    import tree_sitter_cpp # Use cpp parser to parse C code for best compatibility
except ImportError:
    tree_sitter_cpp = None

from core.models import AtomicPatch
from core.state import WorkflowState

# ==========================================
# 0. Internal Data Structures
# ==========================================

@dataclass
class HunkPatch:
    """[Internal] Temporary unit used for Hunk level grouping"""
    parent_id: str          # ID of the source AtomicPatch
    file_path: str
    function_name: str
    content: str            # specific code of Hunk (+/- lines)
    start_line: int         # Relative line number
    
    # Feature cache
    tokens: Set[str] = field(default_factory=set)
    members: Set[str] = field(default_factory=set)
    
    @property
    def id(self):
        # Unique identifier: Parent ID + Start Line
        return f"{self.parent_id}::L{self.start_line}"

# ==========================================
# 1. Basic Tool Components
# ==========================================

class TreeSitterAnalyzer:
    def __init__(self, lang : str = "c"):
        self.parser = None
        if lang == "c":
            self.LANGUAGE = Language(tree_sitter_c.language())
        else:
            self.LANGUAGE = Language(tree_sitter_cpp.language())
        self.parser = Parser(self.LANGUAGE)

    def extract_features(self, diff_content: str) -> Dict[str, Set[str]]:
        if not self.parser: return {"calls": set(), "tokens": set(), "members": set()}

        source_code = "\n".join([
            line[1:] for line in diff_content.splitlines() 
            if line.startswith('+') or line.startswith(' ') or line.startswith('-')
        ])
        code_bytes = source_code.encode('utf8')
        
        tree = self.parser.parse(code_bytes)
        calls, tokens, members = set(), set(), set()

        def get_nodes(captures):
            if isinstance(captures, dict):
                for nodes in captures.values():
                    for n in nodes: yield n
            elif isinstance(captures, list):
                for item in captures:
                    yield item[0] if isinstance(item, (list, tuple)) else item

        # Calls
        q_call = self.LANGUAGE.query("""(call_expression function: (identifier) @func)""")
        for n in get_nodes(q_call.captures(tree.root_node)):
            calls.add(code_bytes[n.start_byte:n.end_byte].decode('utf8'))
        
        # Tokens
        q_token = self.LANGUAGE.query("""(identifier) @id""")
        for n in get_nodes(q_token.captures(tree.root_node)):
            tokens.add(code_bytes[n.start_byte:n.end_byte].decode('utf8'))

        # Struct members (used for data coupling)
        q_member = self.LANGUAGE.query("""(field_expression field: (field_identifier) @f)""")
        for n in get_nodes(q_member.captures(tree.root_node)):
            members.add(code_bytes[n.start_byte:n.end_byte].decode('utf8'))

        return {"calls": calls, "tokens": tokens, "members": members}

# ==========================================
# 2. Static Grouping Logic
# ==========================================

class BaseGrouper:
    def __init__(self, lang: str = "c"):
        self.ts = TreeSitterAnalyzer(lang=lang)

    @staticmethod
    def _extract_diff_lines(diff_text: str) -> tuple:
        """
        Extract deleted and added lines from diff text (remove prefix, after strip)
        Return: (del_lines: Set[str], add_lines: Set[str])
        """
        del_lines = set()
        add_lines = set()
        for line in diff_text.splitlines():
            if line.startswith('-') and not line.startswith('---'):
                content = line[1:].strip()
                if content:  # Ignore empty lines
                    del_lines.add(content)
            elif line.startswith('+') and not line.startswith('+++'):
                content = line[1:].strip()
                if content:  # Ignore empty lines
                    add_lines.add(content)
        return del_lines, add_lines

    def _is_code_movement(self, item1, item2) -> bool:
        """
        Static detection of code movement: code deleted in one place is added in another.
        Code movement should stay together, not be split.
        """
        diff1 = item1.clean_diff if hasattr(item1, 'clean_diff') else item1.content
        diff2 = item2.clean_diff if hasattr(item2, 'clean_diff') else item2.content
        
        del1, add1 = self._extract_diff_lines(diff1)
        del2, add2 = self._extract_diff_lines(diff2)
        
        # Code deleted in one place, added in other -> Code movement
        return bool((del1 & add2) or (add1 & del2))

    def _is_parallel_instance(self, item1, item2) -> bool:
        """
        Static detection of parallel instances:
        1. Functions with same name in different files -> Parallel instances (e.g. same name functions in IPv4/IPv6/ARP)
        2. Same code deleted or added in both places -> Parallel instances (regardless of file)
        Parallel instances should be split into independent search units.
        """
        # Get function name and file path
        name1 = item1.function_name if hasattr(item1, 'function_name') else ''
        name2 = item2.function_name if hasattr(item2, 'function_name') else ''
        path1 = item1.file_path if hasattr(item1, 'file_path') else ''
        path2 = item2.file_path if hasattr(item2, 'file_path') else ''
        
        # Rule 1: Functions with same name in different files -> Parallel instances
        # (Same file cannot have functions with same name, so this rule is valid only for cross-file)
        if name1 and name1 == name2 and path1 != path2:
            return True
        
        # Rule 2: Same code deleted or added in both places -> Parallel instances
        # (Regardless of file, if different functions made same modification, they are independent fix points)
        diff1 = item1.clean_diff if hasattr(item1, 'clean_diff') else item1.content
        diff2 = item2.clean_diff if hasattr(item2, 'clean_diff') else item2.content
        
        del1, add1 = self._extract_diff_lines(diff1)
        del2, add2 = self._extract_diff_lines(diff2)
        
        same_del = del1 & del2
        same_add = add1 & add2
        
        return bool(same_del or same_add)

    def _path_affinity(self, p1, p2):
        parts_a = p1.strip('/').split('/'); parts_b = p2.strip('/').split('/')
        common = 0
        for a, b in zip(parts_a, parts_b):
            if a == b: common += 1
            else: break
        return common / max(len(parts_a), len(parts_b)) if parts_a else 0

    def _refine_via_coloring(self, group_items: list, graph: nx.Graph, id_extractor) -> List[list]:
        """
        Splitting logic based on static analysis (All static, no LLM)
        
        Core principles:
        1. Call Link is strong dependency, never split
        2. Code movement (one del one add) should stay together
        3. Parallel instances (name conflict or same del same add) should be split
        4. File path is only auxiliary reference, used to resolve attribution of same name functions
        """
        if len(group_items) < 2: 
            return [group_items]
        
        item_map = {id_extractor(item): item for item in group_items}
        ids = list(item_map.keys())
        
        # ========== Pre-analysis: Detect cross-file Call Link and name conflicts ==========
        has_cross_file_call = False
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id1, id2 = ids[i], ids[j]
                if graph.has_edge(id1, id2):
                    edge_data = graph.get_edge_data(id1, id2)
                    if edge_data.get('type') == 'call':
                        path1 = item_map[id1].file_path if hasattr(item_map[id1], 'file_path') else ''
                        path2 = item_map[id2].file_path if hasattr(item_map[id2], 'file_path') else ''
                        if path1 and path2 and path1 != path2:
                            has_cross_file_call = True
                            break
            if has_cross_file_call:
                break
        
        # Detect same name function conflicts
        name_to_paths = defaultdict(set)
        for item in group_items:
            name = item.function_name if hasattr(item, 'function_name') else ''
            path = item.file_path if hasattr(item, 'file_path') else ''
            if name and path:
                name_to_paths[name].add(path)
        
        has_name_conflict = any(len(paths) > 1 for paths in name_to_paths.values())
        
        # ========== Fast Path: Parallel Fix Scenario ==========
        # Conflict with same name, and no cross-file call -> Each file is independent, group by file directly
        if has_name_conflict and not has_cross_file_call:
            path_groups = defaultdict(list)
            for item in group_items:
                path = item.file_path if hasattr(item, 'file_path') else ''
                path_groups[path].append(item)
            return [g for g in path_groups.values() if g]
        
        # ========== Regular Path: Graph Coloring Splitting ==========
        conflict_graph = nx.Graph()
        has_conflict = False

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id1, id2 = ids[i], ids[j]
                item1, item2 = item_map[id1], item_map[id2]

                # Check edge type, distinguish "strong dependency" and "weak dependency"
                if graph.has_edge(id1, id2):
                    edge_data = graph.get_edge_data(id1, id2)
                    edge_type = edge_data.get('type', 'unknown')
                    
                    # Strong dependency (Call): Never split, skip check
                    if edge_type == 'call':
                        continue
                
                # 1. Code movement detection: same code deleted and added -> KEEP
                if self._is_code_movement(item1, item2):
                    continue
                
                # 2. Parallel instance detection -> SPLIT
                #    - Same function name in different files (static analysis cannot determine which is called, treat as independent instances)
                #    - Same deletion or same addition of code (independently fixing the same issue)
                if self._is_parallel_instance(item1, item2):
                    conflict_graph.add_edge(id1, id2)
                    has_conflict = True
        
        if not has_conflict: 
            return [group_items]

        # Graph coloring split
        coloring = nx.coloring.greedy_color(conflict_graph)
        color_groups = defaultdict(list)
        for pid, color in coloring.items():
            color_groups[color].append(item_map[pid])
        
        # Neutral node handling: Distinguish between "called helper" and "independent parallel instance"
        neutral_items = [item for item in group_items if id_extractor(item) not in conflict_graph.nodes()]
        
        final_subgroups = []
        for core_items in color_groups.values():
            group_ids = set(id_extractor(item) for item in core_items)
            group_with_helpers = list(core_items)
            
            # Iteratively absorb neutral nodes: including those with call relationships with group members, and those with call relationships with already absorbed neutral nodes
            changed = True
            while changed:
                changed = False
                for neutral in neutral_items:
                    neutral_id = id_extractor(neutral)
                    if neutral_id in group_ids:
                        continue  # Already in group
                    
                    # Check if any group member calls this neutral (or neutral calls a group member)
                    has_call_relation = False
                    for member_id in group_ids:
                        if graph.has_edge(neutral_id, member_id):
                            edge_data = graph.get_edge_data(neutral_id, member_id)
                            if edge_data.get('type') == 'call':
                                has_call_relation = True
                                break
                    
                    if has_call_relation:
                        group_with_helpers.append(neutral)
                        group_ids.add(neutral_id)
                        changed = True  # Continue iteration, likely more neutral nodes to join
            
            final_subgroups.append(group_with_helpers)
        
        # Handle neutral nodes not absorbed by any group
        absorbed_neutrals = set()
        for subgroup in final_subgroups:
            for item in subgroup:
                absorbed_neutrals.add(id_extractor(item))
        
        # Independent neutral nodes should be in the same group if they have call relationships
        remaining_neutrals = [n for n in neutral_items if id_extractor(n) not in absorbed_neutrals]
        if remaining_neutrals:
            # Build subgraph for remaining neutral nodes, group by connected components
            remaining_map = {id_extractor(n): n for n in remaining_neutrals}
            
            sub_graph = nx.Graph()
            for n in remaining_neutrals:
                sub_graph.add_node(id_extractor(n))
            
            for n1 in remaining_neutrals:
                n1_id = id_extractor(n1)
                for n2 in remaining_neutrals:
                    n2_id = id_extractor(n2)
                    if n1_id != n2_id and graph.has_edge(n1_id, n2_id):
                        edge_data = graph.get_edge_data(n1_id, n2_id)
                        if edge_data.get('type') == 'call':
                            sub_graph.add_edge(n1_id, n2_id)
            
            for component in nx.connected_components(sub_graph):
                final_subgroups.append([remaining_map[nid] for nid in component])
            
        return final_subgroups

# ==========================================
# 3. Step 1: Function level grouping (Function Grouper)
# ==========================================

class FunctionGrouper(BaseGrouper):
    def group(self, patches: List[AtomicPatch]) -> List[List[AtomicPatch]]:
        print(f"  >> [Step 1] Function Grouping ({len(patches)} patches)...")
        func_map = defaultdict(list)
        member_map = defaultdict(list)
        patch_map = {p.id: p for p in patches}

        # Feature extraction
        for p in patches:
            feats = self.ts.extract_features(p.clean_diff)
            p.calls = feats['calls']
            p.tokens = feats['tokens']
            p.metadata['_members'] = feats['members']
            
            func_map[p.function_name].append(p)
            for m in feats['members']: 
                member_map[m].append(p)

        G = nx.Graph()
        for p in patches: G.add_node(p.id)

        # Networking (Attraction)
        for p in patches:
            # Call Link: Prioritize same file, only real cross-file dependencies are connected across files
            for call in p.calls:
                if call in func_map:
                    candidates = [c for c in func_map[call] if c.id != p.id]
                    if candidates:
                        # Check if this function is modified in multiple files (parallel instances)
                        callee_files = set(c.file_path for c in candidates)
                        same_file_cands = [c for c in candidates if c.file_path == p.file_path]
                        
                        if same_file_cands:
                            # Candidates in the same file, choose the one in the same file
                            best_cand = same_file_cands[0]
                            G.add_edge(p.id, best_cand.id, type='call')
                        elif len(callee_files) == 1:
                            # Only one file has this function (real cross-file dependency)
                            best_cand = candidates[0]
                            G.add_edge(p.id, best_cand.id, type='call')
                        # else: Multiple files have same-named functions but current file doesn't -> Do not connect
                        # (Current file likely has this function too, just not modified, it is an internal call)
            # Data Link: Establish data coupling edges only within the same file
            # Cross-file data coupling is usually a feature of parallel instances, should not connect
            for m in p.metadata['_members']:
                if m in member_map:
                    for cand in member_map[m]:
                        # Only establish data coupling edges for same file
                        if cand.id != p.id and cand.file_path == p.file_path and not G.has_edge(p.id, cand.id):
                            G.add_edge(p.id, cand.id, type='data')

        # Preliminary grouping
        groups = [ [patch_map[n] for n in c] for c in nx.connected_components(G) ]
        
        # Split (Refinement)
        refined_groups = []
        for g in groups:
            refined_groups.extend(self._refine_via_coloring(g, G, lambda x: x.id))
            
        return refined_groups

# ==========================================
# 4. Step 2: Hunk level grouping
# ==========================================

class HunkGrouper(BaseGrouper):
    def group(self, function_groups: List[List[AtomicPatch]]) -> List[List[AtomicPatch]]:
        print(f"  >> [Step 2] Hunk Grouping on {len(function_groups)} groups...")
        final_output = []

        for f_group in function_groups:
            # 1. Decomposition
            hunks = self._decompose(f_group)
            if not hunks: 
                continue
            
            if len(hunks) == 1:
                final_output.append(self._merge(hunks, f_group))
                continue

            # 2. Build graph
            G = self._build_graph(hunks)
            
            # 3. Split (Refinement)
            sub_hunk_groups = self._refine_via_coloring(hunks, G, lambda x: x.id)
            
            # 4. Reduction
            for sub_group in sub_hunk_groups:
                final_output.append(self._merge(sub_group, f_group))
        
        return final_output

    def _decompose(self, patches: List[AtomicPatch]) -> List[HunkPatch]:
        hunks = []
        for p in patches:
            # Split based on @@, if not present then treat whole as one Hunk
            lines = p.clean_diff.splitlines()
            buffer = []
            start = 0
            
            for i, line in enumerate(lines):
                if line.startswith('@@'):
                    if buffer:
                        hunks.append(self._make_hunk(p, buffer, start))
                    buffer = [line]
                    start = i
                else:
                    buffer.append(line)
            if buffer:
                hunks.append(self._make_hunk(p, buffer, start))
        return hunks

    def _make_hunk(self, p: AtomicPatch, lines: List[str], start: int) -> HunkPatch:
        content = "\n".join(lines)
        feats = self.ts.extract_features(content)
        return HunkPatch(
            parent_id=p.id,
            file_path=p.file_path,
            function_name=p.function_name,
            content=content,
            start_line=start,
            tokens=feats['tokens'],
            members=feats['members']
        )

    def _build_graph(self, hunks: List[HunkPatch]) -> nx.Graph:
        G = nx.Graph()
        for h in hunks: G.add_node(h.id)
        
        for i, h1 in enumerate(hunks):
            for j in range(i+1, len(hunks)):
                h2 = hunks[j]
                
                # Within same function: Data coupling via shared Tokens
                if h1.function_name == h2.function_name and h1.parent_id == h2.parent_id:
                    # Shared Token (Non-empty)
                    if h1.tokens & h2.tokens:
                        G.add_edge(h1.id, h2.id, type='data_intra')
                
                # Cross-function: Data coupling (only within same file)
                else:
                    if h1.file_path == h2.file_path and h1.members & h2.members:
                        G.add_edge(h1.id, h2.id, type='data_inter')
        return G

    def _merge(self, hunks: List[HunkPatch], originals: List[AtomicPatch]) -> List[AtomicPatch]:
        """Merge Hunk groups back into AtomicPatch list"""
        original_map = {p.id: p for p in originals}
        func_bucket = defaultdict(list)
        
        for h in hunks:
            func_bucket[h.parent_id].append(h)
            
        result = []
        for pid, h_list in func_bucket.items():
            orig = original_map[pid]
            h_list.sort(key=lambda x: x.start_line)
            merged_diff = "\n".join([h.content for h in h_list])
            
            new_p = AtomicPatch(
                file_path=orig.file_path,
                function_name=orig.function_name,
                clean_diff=merged_diff,
                raw_diff=orig.raw_diff,
                change_type=orig.change_type,
                old_code=orig.old_code,
                new_code=orig.new_code,
                metadata=orig.metadata.copy(),
                start_line_old=orig.start_line_old,
                start_line_new=orig.start_line_new
            )
            # Save calculated features for later use
            new_p.calls = set().union(*[self.ts.extract_features(h.content)['calls'] for h in h_list])
            new_p.tokens = set().union(*[h.tokens for h in h_list])
            
            result.append(new_p)
        return result

# ==========================================
# 5. LangGraph Node
# ==========================================

def grouping_node(state: WorkflowState) -> Dict:
    import os
    print(f"--- Phase 2: Grouping {len(state['atomic_patches'])} patches ---")
    
    if not state['atomic_patches']:
        return {"grouped_patches": []}

    # 1. Function level grouping
    f_grouper = FunctionGrouper(lang=state.get('lang', 'cpp'))
    func_groups = f_grouper.group(state['atomic_patches'])
    
    # 2. Hunk level refinement
    h_grouper = HunkGrouper(lang=state.get('lang', 'cpp'))
    final_groups = h_grouper.group(func_groups)
    
    print(f"    >> Final Groups: {len(final_groups)}")
    for i, g in enumerate(final_groups):
        # Output format: file_basename::function_name
        items = [f"{os.path.basename(p.file_path)}::{p.function_name}" for p in g]
        print(f"       Group {i}: {items}")
    return {"grouped_patches": final_groups}
