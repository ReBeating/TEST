import networkx as nx
from typing import List, Dict, Set
from collections import defaultdict
from dataclasses import dataclass, field

# Tree-sitter
try:
    from tree_sitter import Language, Parser
    import tree_sitter_c # 使用 c parser 解析 C 代码
    import tree_sitter_cpp # 使用 cpp parser 解析 C 代码兼容性最好
except ImportError:
    tree_sitter_cpp = None

from core.models import AtomicPatch
from core.state import WorkflowState

# ==========================================
# 0. 内部数据结构 (Internal Data Structures)
# ==========================================

@dataclass
class HunkPatch:
    """[Internal] Hunk 级分组使用的临时单元"""
    parent_id: str          # 来源 AtomicPatch 的 ID
    file_path: str
    function_name: str
    content: str            # Hunk 的具体代码 (+/- 行)
    start_line: int         # 相对行号
    
    # 特征缓存
    tokens: Set[str] = field(default_factory=set)
    members: Set[str] = field(default_factory=set)
    
    @property
    def id(self):
        # 唯一标识符：父ID + 起始行
        return f"{self.parent_id}::L{self.start_line}"

# ==========================================
# 1. 基础工具组件 (Analyzer)
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

        # Struct Members (用于数据耦合)
        q_member = self.LANGUAGE.query("""(field_expression field: (field_identifier) @f)""")
        for n in get_nodes(q_member.captures(tree.root_node)):
            members.add(code_bytes[n.start_byte:n.end_byte].decode('utf8'))

        return {"calls": calls, "tokens": tokens, "members": members}

# ==========================================
# 2. 静态分组逻辑 (Base Logic)
# ==========================================

class BaseGrouper:
    def __init__(self, lang: str = "c"):
        self.ts = TreeSitterAnalyzer(lang=lang)

    @staticmethod
    def _extract_diff_lines(diff_text: str) -> tuple:
        """
        从 diff 文本中提取删除行和添加行（去除前缀，strip 后）
        返回: (del_lines: Set[str], add_lines: Set[str])
        """
        del_lines = set()
        add_lines = set()
        for line in diff_text.splitlines():
            if line.startswith('-') and not line.startswith('---'):
                content = line[1:].strip()
                if content:  # 忽略空行
                    del_lines.add(content)
            elif line.startswith('+') and not line.startswith('+++'):
                content = line[1:].strip()
                if content:  # 忽略空行
                    add_lines.add(content)
        return del_lines, add_lines

    def _is_code_movement(self, item1, item2) -> bool:
        """
        静态检测代码移动：一处删除的代码在另一处被添加。
        代码移动应保持在一起，不应被拆分。
        """
        diff1 = item1.clean_diff if hasattr(item1, 'clean_diff') else item1.content
        diff2 = item2.clean_diff if hasattr(item2, 'clean_diff') else item2.content
        
        del1, add1 = self._extract_diff_lines(diff1)
        del2, add2 = self._extract_diff_lines(diff2)
        
        # 一处删除的代码，另一处添加了 → 代码移动
        return bool((del1 & add2) or (add1 & del2))

    def _is_parallel_instance(self, item1, item2) -> bool:
        """
        静态检测平行实例：
        1. 同名函数在不同文件中 → 平行实例（如 IPv4/IPv6/ARP 的同名函数）
        2. 两处同时删除或新增相同代码 → 平行实例（无论同文件还是跨文件）
        平行实例应被拆分为独立的搜索单元。
        """
        # 获取函数名和文件路径
        name1 = item1.function_name if hasattr(item1, 'function_name') else ''
        name2 = item2.function_name if hasattr(item2, 'function_name') else ''
        path1 = item1.file_path if hasattr(item1, 'file_path') else ''
        path2 = item2.file_path if hasattr(item2, 'file_path') else ''
        
        # 规则 1: 同名函数在不同文件中 → 平行实例
        # （同一文件内不可能有同名函数，所以这条规则只对跨文件有效）
        if name1 and name1 == name2 and path1 != path2:
            return True
        
        # 规则 2: 两处都删除相同代码 或 两处都添加相同代码 → 平行实例
        # （无论同文件还是跨文件，只要是不同函数做了相同修改，就是独立的修复点）
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
        基于静态分析的分裂逻辑（全静态，无 LLM）
        
        核心原则：
        1. Call Link 是强依赖，绝对不拆
        2. 代码移动（一删一加）应保持在一起
        3. 平行实例（同名冲突 或 同删同加）应拆分
        4. 文件路径只是辅助参考，用于解决同名函数的归属问题
        """
        if len(group_items) < 2: 
            return [group_items]
        
        item_map = {id_extractor(item): item for item in group_items}
        ids = list(item_map.keys())
        
        # ========== 预分析：检测跨文件 Call Link 和同名冲突 ==========
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
        
        # 检测同名函数冲突
        name_to_paths = defaultdict(set)
        for item in group_items:
            name = item.function_name if hasattr(item, 'function_name') else ''
            path = item.file_path if hasattr(item, 'file_path') else ''
            if name and path:
                name_to_paths[name].add(path)
        
        has_name_conflict = any(len(paths) > 1 for paths in name_to_paths.values())
        
        # ========== 快速路径：Parallel Fix 场景 ==========
        # 存在同名冲突，且没有跨文件调用 → 每个文件是独立的，直接按文件分组
        if has_name_conflict and not has_cross_file_call:
            path_groups = defaultdict(list)
            for item in group_items:
                path = item.file_path if hasattr(item, 'file_path') else ''
                path_groups[path].append(item)
            return [g for g in path_groups.values() if g]
        
        # ========== 常规路径：图着色分裂 ==========
        conflict_graph = nx.Graph()
        has_conflict = False

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id1, id2 = ids[i], ids[j]
                item1, item2 = item_map[id1], item_map[id2]

                # 检查边的类型，区分"强依赖"和"弱依赖"
                if graph.has_edge(id1, id2):
                    edge_data = graph.get_edge_data(id1, id2)
                    edge_type = edge_data.get('type', 'unknown')
                    
                    # 强依赖 (Call): 绝对不拆，跳过检查
                    if edge_type == 'call':
                        continue
                
                # 1. 代码移动检测：一删一加相同代码 → KEEP
                if self._is_code_movement(item1, item2):
                    continue
                
                # 2. 平行实例检测 → SPLIT
                #    - 同名函数在不同文件（静态无法确定调用哪个，视为独立实例）
                #    - 同删或同加相同代码（独立修复同一问题）
                if self._is_parallel_instance(item1, item2):
                    conflict_graph.add_edge(id1, id2)
                    has_conflict = True
        
        if not has_conflict: 
            return [group_items]

        # 图着色分裂 (Graph Coloring Split)
        coloring = nx.coloring.greedy_color(conflict_graph)
        color_groups = defaultdict(list)
        for pid, color in coloring.items():
            color_groups[color].append(item_map[pid])
        
        # 中立节点处理：区分 "被调用的 helper" 和 "独立的平行实例"
        neutral_items = [item for item in group_items if id_extractor(item) not in conflict_graph.nodes()]
        
        final_subgroups = []
        for core_items in color_groups.values():
            group_ids = set(id_extractor(item) for item in core_items)
            group_with_helpers = list(core_items)
            
            # 迭代吸纳中立节点：包括和组内成员有调用关系的，以及和已吸纳中立节点有调用关系的
            changed = True
            while changed:
                changed = False
                for neutral in neutral_items:
                    neutral_id = id_extractor(neutral)
                    if neutral_id in group_ids:
                        continue  # 已经在组内
                    
                    # 检查是否有组内成员调用了这个 neutral（或 neutral 调用了组内成员）
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
                        changed = True  # 继续迭代，可能还有其他中立节点需要加入
            
            final_subgroups.append(group_with_helpers)
        
        # 处理没有被任何组吸纳的中立节点
        absorbed_neutrals = set()
        for subgroup in final_subgroups:
            for item in subgroup:
                absorbed_neutrals.add(id_extractor(item))
        
        # 独立的中立节点之间如果有调用关系，应该在同一组
        remaining_neutrals = [n for n in neutral_items if id_extractor(n) not in absorbed_neutrals]
        if remaining_neutrals:
            # 对剩余中立节点建立子图，按连通分量分组
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
# 3. 第一步：函数级分组 (Function Grouper)
# ==========================================

class FunctionGrouper(BaseGrouper):
    def group(self, patches: List[AtomicPatch]) -> List[List[AtomicPatch]]:
        print(f"  >> [Step 1] Function Grouping ({len(patches)} patches)...")
        func_map = defaultdict(list)
        member_map = defaultdict(list)
        patch_map = {p.id: p for p in patches}

        # 特征提取
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

        # 连线 (Attraction)
        for p in patches:
            # Call Link: 优先同文件，只有真正的跨文件依赖才跨文件连接
            for call in p.calls:
                if call in func_map:
                    candidates = [c for c in func_map[call] if c.id != p.id]
                    if candidates:
                        # 检查这个函数是否在多个文件中被修改（平行实例）
                        callee_files = set(c.file_path for c in candidates)
                        same_file_cands = [c for c in candidates if c.file_path == p.file_path]
                        
                        if same_file_cands:
                            # 同文件有候选者，选择同文件的
                            best_cand = same_file_cands[0]
                            G.add_edge(p.id, best_cand.id, type='call')
                        elif len(callee_files) == 1:
                            # 只有一个文件有这个函数（真正的跨文件依赖）
                            best_cand = candidates[0]
                            G.add_edge(p.id, best_cand.id, type='call')
                        # else: 多个文件有同名函数但当前文件没有 → 不连接
                        # （当前文件很可能也有这个函数，只是没被修改，是内部调用）
            # Data Link: 只在同文件内建立数据耦合边
            # 跨文件的数据耦合通常是平行实例的特征，不应该连接
            for m in p.metadata['_members']:
                if m in member_map:
                    for cand in member_map[m]:
                        # 只有同文件才建立数据耦合边
                        if cand.id != p.id and cand.file_path == p.file_path and not G.has_edge(p.id, cand.id):
                            G.add_edge(p.id, cand.id, type='data')

        # 初步分组
        groups = [ [patch_map[n] for n in c] for c in nx.connected_components(G) ]
        
        # 分裂 (Refinement)
        refined_groups = []
        for g in groups:
            refined_groups.extend(self._refine_via_coloring(g, G, lambda x: x.id))
            
        return refined_groups

# ==========================================
# 4. 第二步：Hunk 级分组 (Hunk Grouper)
# ==========================================

class HunkGrouper(BaseGrouper):
    def group(self, function_groups: List[List[AtomicPatch]]) -> List[List[AtomicPatch]]:
        print(f"  >> [Step 2] Hunk Grouping on {len(function_groups)} groups...")
        final_output = []

        for f_group in function_groups:
            # 1. 分解
            hunks = self._decompose(f_group)
            if not hunks: 
                continue
            
            if len(hunks) == 1:
                final_output.append(self._merge(hunks, f_group))
                continue

            # 2. 建图
            G = self._build_graph(hunks)
            
            # 3. 分裂 (Refinement)
            sub_hunk_groups = self._refine_via_coloring(hunks, G, lambda x: x.id)
            
            # 4. 归约
            for sub_group in sub_hunk_groups:
                final_output.append(self._merge(sub_group, f_group))
        
        return final_output

    def _decompose(self, patches: List[AtomicPatch]) -> List[HunkPatch]:
        hunks = []
        for p in patches:
            # 基于 @@ 拆分，如果不存在则整体作为一个 Hunk
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
                
                # 同函数内: Data coupling via shared tokens
                if h1.function_name == h2.function_name and h1.parent_id == h2.parent_id:
                    # 共享 Token (非空)
                    if h1.tokens & h2.tokens:
                        G.add_edge(h1.id, h2.id, type='data_intra')
                
                # 跨函数: Data Coupling (只在同文件内)
                else:
                    if h1.file_path == h2.file_path and h1.members & h2.members:
                        G.add_edge(h1.id, h2.id, type='data_inter')
        return G

    def _merge(self, hunks: List[HunkPatch], originals: List[AtomicPatch]) -> List[AtomicPatch]:
        """将 Hunk 组重新合并为 AtomicPatch 列表"""
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
            # 存回计算好的特征供后续使用
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

    # 1. 函数级分组
    f_grouper = FunctionGrouper(lang=state.get('lang', 'cpp'))
    func_groups = f_grouper.group(state['atomic_patches'])
    
    # 2. Hunk 级细化
    h_grouper = HunkGrouper(lang=state.get('lang', 'cpp'))
    final_groups = h_grouper.group(func_groups)
    
    print(f"    >> Final Groups: {len(final_groups)}")
    for i, g in enumerate(final_groups):
        # 输出格式: file_basename::function_name
        items = [f"{os.path.basename(p.file_path)}::{p.function_name}" for p in g]
        print(f"       Group {i}: {items}")
    return {"grouped_patches": final_groups}
