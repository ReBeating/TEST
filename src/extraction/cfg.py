import tree_sitter_c
import tree_sitter_cpp
from tree_sitter import Language, Parser, Node
import networkx as nx
from enum import Enum
from typing import List, Set, Optional, Tuple, Dict, Any
from pydantic import BaseModel, Field

# --- 数据模型 ---

class CFGNodeType(str, Enum):
    ENTRY = "ENTRY"
    EXIT = "EXIT"
    STATEMENT = "STATEMENT"
    PREDICATE = "PREDICATE"       # if/while/for 的条件
    MERGE = "MERGE"               # 虚拟节点，用于汇聚
    NO_OP = "NO_OP"               # 空操作
    
    # 结构特定
    SWITCH_HEAD = "SWITCH_HEAD"
    CASE_LABEL = "CASE_LABEL"
    DEFAULT_LABEL = "DEFAULT_LABEL"

class CFGEdgeType(str, Enum):
    FLOW = "FLOW"                 # 顺序流
    TRUE = "TRUE"                 # 条件真
    FALSE = "FALSE"               # 条件假
    JUMP = "JUMP"                 # goto/break/continue/return
    FALLTHROUGH = "FALLTHROUGH"   # Switch case 穿透
    BACK_EDGE = "BACK_EDGE"       # 循环回边
    
    SWITCH_MATCH = "SWITCH_MATCH"       # Switch 头匹配到 Case
    SWITCH_DEFAULT = "SWITCH_DEFAULT"   # Switch 头匹配到 Default

class CFGNode(BaseModel):
    id: str
    type: CFGNodeType
    code: str
    start_line: int
    end_line: int
    ast_type: str
    defs: Dict[str, Set[str]] = Field(default_factory=dict)
    uses: Dict[str, Set[str]] = Field(default_factory=dict)

    def __hash__(self):
        return hash(self.id)

# --- CFG 构建器 ---

class CFGBuilder:
    def __init__(self, lang: str = "c"):
        if lang == "c":
            self.language = Language(tree_sitter_c.language())
        elif lang == "cpp":
            self.language = Language(tree_sitter_cpp.language())
        else:
            raise ValueError(f"Unsupported language: {lang}")
        self.parser = Parser(self.language)
        self.cfg = nx.DiGraph()
        self.source_bytes = b""
        
        # 状态管理
        self.node_counter = 0
        self.labels: Dict[str, str] = {}    # label_name -> node_id
        self.gotos: List[Tuple[str, str]] = [] # (src_node_id, label_name)
        
        # Tree-sitter query
        self.query_ids = self.language.query("(identifier) @id")
        
        self.symbol_table: Dict[str, str] = {} # var_name -> type_string
    
    TERMINAL_FUNCTIONS = {"panic", "exit", "die", "BUG", "BUG_ON", "assert", "kfree_skb"}
    
    def build(self, code_str: str, target_line: Optional[int] = None) -> nx.DiGraph:
        # [FIX 1] 彻底重置所有状态，防止跨函数污染
        self.node_counter = 0
        self.labels.clear()
        self.gotos.clear()
        self.symbol_table.clear()
        self.cfg = nx.DiGraph()
        
        self.source_bytes = code_str.encode('utf8')
        tree = self.parser.parse(self.source_bytes)
        
        # 寻找函数定义
        root = tree.root_node
        func_def = None
        
        # Two modes: 
        # 1. Target Line provided: Find specific function covering that line
        # 2. No target: Default to first function (legacy behavior) 
        
        candidates = []
        for child in root.children:
            if child.type == 'function_definition':
                candidates.append(child)
        
        if target_line:
            for cand in candidates:
                # Tree-sitter uses 0-based indexing
                # node.start_point is (row, col)
                start_row = cand.start_point[0] + 1
                end_row = cand.end_point[0] + 1
                if start_row <= target_line <= end_row:
                    func_def = cand
                    break
            # If not found (e.g. global var or macro), maybe use the nearest one or fail?
            # Let's fallback to nothing if strictly requested? 
            # Or print warning. For now, strict.
        else:
            if candidates:
                func_def = candidates[0]
        
        if func_def:
            # 提取函数头作为 Entry
            declarator = func_def.child_by_field_name('declarator')
            params = self._extract_parameters(declarator)
            entry_code = self._get_text(declarator) if declarator else "ENTRY"
            
            self.entry_id = self._create_node(
                declarator if declarator else func_def, 
                CFGNodeType.ENTRY, 
                code=entry_code,
                extra_defs=params
            )
            
            self.exit_id = self._create_node(None, CFGNodeType.EXIT, "EXIT")
            
            # 处理函数体
            body = func_def.child_by_field_name('body')
            body_entry, body_exits = self._process_node(body, context=None)
            
            if body_entry:
                self.cfg.add_edge(self.entry_id, body_entry, type=CFGEdgeType.FLOW)
                for ex in body_exits:
                    self.cfg.add_edge(ex, self.exit_id, type=CFGEdgeType.FLOW)
            else:
                self.cfg.add_edge(self.entry_id, self.exit_id, type=CFGEdgeType.FLOW)
        
        else:
            # Fallback: 代码片段
            self.entry_id = self._create_node(None, CFGNodeType.ENTRY, "ENTRY")
            self.exit_id = self._create_node(None, CFGNodeType.EXIT, "EXIT")
            body_entry, body_exits = self._process_node(root, context=None)
            
            if body_entry:
                self.cfg.add_edge(self.entry_id, body_entry, type=CFGEdgeType.FLOW)
                for ex in body_exits:
                    self.cfg.add_edge(ex, self.exit_id, type=CFGEdgeType.FLOW)
            else:
                self.cfg.add_edge(self.entry_id, self.exit_id, type=CFGEdgeType.FLOW)

        self._resolve_gotos()
        self.cfg.graph['entry'] = self.entry_id
        self.cfg.graph['exit'] = self.exit_id
        return self.cfg

    # --- 递归处理 ---

    def _process_node(self, node: Node, context: dict) -> Tuple[Optional[str], List[str]]:
        if node is None: return None, []
        
        if node.type in ('comment', 'preproc_def', 'preproc_include'):
            return None, []
        
        if node.type == 'else_clause':
            # else_clause 的结构通常是: "else" statement
            # 我们需要找到那个 statement 子节点并递归处理
            for child in node.children:
                if child.type not in ('else', 'comment'):
                    return self._process_node(child, context)
            return None, []
        
        # 1. 容器类
        if node.type in ('translation_unit', 'compound_statement'):
            # 过滤掉标点和注释
            children = [c for c in node.children if c.type not in ('{', '}', 'comment')]
            if not children: return None, []

            first_entry = None
            current_exits = []
            
            i = 0
            while i < len(children):
                child = children[i]
                next_child = children[i+1] if i + 1 < len(children) else None
                
                # --- 使用启发式判断 ---
                is_macro, macro_name = self._looks_like_loop_macro(child, next_child)
                
                if is_macro:
                    # 命中！这是一个宏循环
                    # child 是 Header (Init/Cond), next_child 是 Body
                    body_node = next_child
                    
                    # 1. 处理宏循环
                    entry, exits = self._process_macro_loop(child, body_node, context, macro_name)
                    
                    # 2. 关键：跳过下一个节点，因为它是 Body，已经被融合进宏逻辑了
                    i += 2
                else:
                    # 普通节点
                    entry, exits = self._process_node(child, context)
                    i += 1

                # --- 连接逻辑 (不变) ---
                if not entry: continue # 可能是声明或空语句
                
                if first_entry is None: first_entry = entry
                
                if current_exits:
                    for ex in current_exits:
                        self.cfg.add_edge(ex, entry, type=CFGEdgeType.FLOW)
                
                current_exits = exits
            
            return first_entry, current_exits

        # 2. 函数定义 (如果在内部)
        if node.type == 'function_definition':
            body = node.child_by_field_name('body')
            return self._process_node(body, context)

        # 3. 表达式/声明
        if node.type in ('expression_statement', 'declaration'):
            # 检查是否是 Call
            is_terminal = False
            if node.type == 'expression_statement':
                child = node.children[0]
                if child.type == 'call_expression':
                    func_node = child.child_by_field_name('function')
                    if func_node:
                        func_name = self._get_text(func_node)
                        if any(t in func_name for t in self.TERMINAL_FUNCTIONS):
                            is_terminal = True
            
            nid = self._create_node(node, CFGNodeType.STATEMENT)
            
            if is_terminal:
                self.cfg.add_edge(nid, self.exit_id, type=CFGEdgeType.JUMP)
                return nid, [] # 没有出口，流程在此终结
            
            return nid, [nid]

        # 4. If
        if node.type == 'if_statement':
            cond_node = node.child_by_field_name('condition')
            then_node = node.child_by_field_name('consequence')
            else_node = node.child_by_field_name('alternative')

            cond_entry, true_exits, false_exits = self._process_condition(cond_node)
            all_exits = []

            then_entry, then_exits = self._process_node(then_node, context)
            if then_entry:
                for te in true_exits:
                    self.cfg.add_edge(te, then_entry, type=CFGEdgeType.TRUE)
                all_exits.extend(then_exits)
            else:
                all_exits.extend(true_exits)

            if else_node:
                else_entry, else_exits_block = self._process_node(else_node, context)
                if else_entry:
                    for fe in false_exits:
                        self.cfg.add_edge(fe, else_entry, type=CFGEdgeType.FALSE)
                    all_exits.extend(else_exits_block)
                else:
                    all_exits.extend(false_exits)
            else:
                all_exits.extend(false_exits)

            return cond_entry, all_exits

        # 5. Loops
        # 5.1 While
        if node.type == 'while_statement':
            cond_node = node.child_by_field_name('condition')
            body_node = node.child_by_field_name('body')
            
            cond_entry, true_exits, false_exits = self._process_condition(cond_node)
            
            loop_breaks = []
            # 嵌套关键：使用新的 break list，但继承/覆盖 continue target
            new_ctx = self._update_context(context, break_targets=loop_breaks, continue_target=cond_entry)
            
            body_entry, body_exits = self._process_node(body_node, new_ctx)
            
            if body_entry:
                for te in true_exits:
                    self.cfg.add_edge(te, body_entry, type=CFGEdgeType.TRUE)
                for be in body_exits:
                    self.cfg.add_edge(be, cond_entry, type=CFGEdgeType.BACK_EDGE)
            else:
                for te in true_exits:
                    self.cfg.add_edge(te, cond_entry, type=CFGEdgeType.BACK_EDGE)

            return cond_entry, false_exits + loop_breaks

        # 5.2 Do-While
        if node.type == 'do_statement':
            body_node = node.child_by_field_name('body')
            cond_node = node.child_by_field_name('condition')
            
            cond_entry, true_exits, false_exits = self._process_condition(cond_node)
            
            loop_breaks = []
            new_ctx = self._update_context(context, break_targets=loop_breaks, continue_target=cond_entry)
            
            body_entry, body_exits = self._process_node(body_node, new_ctx)
            
            if not body_entry:
                body_entry = self._create_node(None, CFGNodeType.NO_OP, "DO_EMPTY")
                body_exits = [body_entry]
            
            for be in body_exits:
                self.cfg.add_edge(be, cond_entry, type=CFGEdgeType.FLOW)
            
            for te in true_exits:
                self.cfg.add_edge(te, body_entry, type=CFGEdgeType.BACK_EDGE)
                
            return body_entry, false_exits + loop_breaks

        # 5.3 For
        if node.type == 'for_statement':
            init_node = node.child_by_field_name('initializer')
            cond_node = node.child_by_field_name('condition')
            update_node = node.child_by_field_name('update')
            body_node = node.child_by_field_name('body')
            
            init_entry, init_exits = self._process_node(init_node, context)
            
            if cond_node:
                cond_entry, true_exits, false_exits = self._process_condition(cond_node)
            else:
                cond_entry = self._create_node(None, CFGNodeType.PREDICATE, "TRUE")
                true_exits = [cond_entry]
                false_exits = []
            
            if init_entry:
                for ie in init_exits:
                    self.cfg.add_edge(ie, cond_entry, type=CFGEdgeType.FLOW)
            
            update_entry, update_exits = (None, [])
            if update_node:
                update_entry, update_exits = self._process_node(update_node, context)
                for ue in update_exits:
                    self.cfg.add_edge(ue, cond_entry, type=CFGEdgeType.BACK_EDGE)
            
            continue_target = update_entry if update_entry else cond_entry
            loop_breaks = []
            new_ctx = self._update_context(context, break_targets=loop_breaks, continue_target=continue_target)
            
            body_entry, body_exits = self._process_node(body_node, new_ctx)
            
            if body_entry:
                for te in true_exits:
                    self.cfg.add_edge(te, body_entry, type=CFGEdgeType.TRUE)
                target = update_entry if update_entry else cond_entry
                edge_type = CFGEdgeType.FLOW if update_entry else CFGEdgeType.BACK_EDGE
                for be in body_exits:
                    self.cfg.add_edge(be, target, type=edge_type)
            else:
                target = update_entry if update_entry else cond_entry
                edge_type = CFGEdgeType.FLOW if update_entry else CFGEdgeType.BACK_EDGE
                for te in true_exits:
                    self.cfg.add_edge(te, target, type=edge_type)
                    
            entry_point = init_entry if init_entry else cond_entry
            return entry_point, false_exits + loop_breaks

        # 6. Switch
        if node.type == 'switch_statement':
            cond_node = node.child_by_field_name('condition')
            body_node = node.child_by_field_name('body')
            
            head_id = self._create_node(cond_node, CFGNodeType.SWITCH_HEAD)
            
            switch_breaks = []
            switch_state = {"last_exits": []} 
            # 嵌套关键：break 指向 switch_breaks，continue 保持不变 (继承自外层 Loop)
            new_ctx = self._update_context(context, break_targets=switch_breaks, switch_state=switch_state)
            
            children = [c for c in body_node.children if c.type not in ('{', '}', 'comment')]
            
            for child in children:
                if child.type == 'case_statement':
                    # Label
                    label_text = self._get_text(child).split(':')[0]
                    label_id = self._create_node(child, CFGNodeType.CASE_LABEL, code=label_text)
                    
                    self.cfg.add_edge(head_id, label_id, type=CFGEdgeType.SWITCH_MATCH)
                    
                    for prev_ex in switch_state["last_exits"]:
                        self.cfg.add_edge(prev_ex, label_id, type=CFGEdgeType.FALLTHROUGH)
                    
                    case_entry, case_exits = self._process_case_body(child, new_ctx)
                    
                    if case_entry:
                        self.cfg.add_edge(label_id, case_entry, type=CFGEdgeType.FLOW)
                        switch_state["last_exits"] = case_exits
                    else:
                        switch_state["last_exits"] = [label_id]
            
            return head_id, switch_breaks + switch_state["last_exits"]

        # 7. Jumps
        if node.type == 'return_statement':
            nid = self._create_node(node, CFGNodeType.STATEMENT, "RETURN")
            self.cfg.add_edge(nid, self.exit_id, type=CFGEdgeType.JUMP)
            return nid, []
            
        if node.type == 'break_statement':
            nid = self._create_node(node, CFGNodeType.STATEMENT, "BREAK")
            if context and 'break_targets' in context:
                context['break_targets'].append(nid)
            return nid, [] 

        if node.type == 'continue_statement':
            nid = self._create_node(node, CFGNodeType.STATEMENT, "CONTINUE")
            if context and 'continue_target' in context:
                self.cfg.add_edge(nid, context['continue_target'], type=CFGEdgeType.JUMP)
            return nid, [] 

        if node.type == 'goto_statement':
            nid = self._create_node(node, CFGNodeType.STATEMENT, "GOTO")
            label_node = node.child_by_field_name('label')
            if label_node:
                self.gotos.append((nid, self._get_text(label_node)))
            return nid, [] 

        if node.type == 'labeled_statement':
            label_node = node.child_by_field_name('label')
            label_text = self._get_text(label_node)
            label_id = self._create_node(node, CFGNodeType.STATEMENT, f"LABEL {label_text}")
            self.labels[label_text] = label_id
            
            stmt = node.child_by_field_name('statement')
            if not stmt: # Robust check
                found_colon = False
                for child in node.children:
                    if child.type == ':': found_colon = True; continue
                    if found_colon and child.type != 'comment': stmt = child; break
            
            if stmt:
                entry, exits = self._process_node(stmt, context)
                if entry:
                    self.cfg.add_edge(label_id, entry, type=CFGEdgeType.FLOW)
                    return label_id, exits
            return label_id, [label_id]

        nid = self._create_node(node, CFGNodeType.STATEMENT)
        return nid, [nid]

    def _process_macro_loop(self, header_node: Node, body_node: Node, context: dict, macro_name: str):
        """
        处理 list_for_each_entry 等宏。
        Header: 宏调用本身 (充当 Init + Cond + Step)
        Body: 宏后面的语句块
        """
        # 1. 创建 Header 节点
        # 这里将其标记为 PREDICATE，因为它决定了循环是否继续
        # 或者增加一个新的类型 MACRO_LOOP_HEAD
        header_id = self._create_node(
            header_node, 
            CFGNodeType.PREDICATE, # 或 LOOP_COND
            code=f"MACRO: {self._get_text(header_node).splitlines()[0]}"
        )
        
        # 2. 准备 Context (处理 Body 内的 break/continue)
        # Continue 跳回 Header
        # Break 跳出 Loop
        loop_breaks = []
        new_ctx = self._update_context(context, break_targets=loop_breaks, continue_target=header_id)
        
        # 3. 处理 Body
        body_entry, body_exits = self._process_node(body_node, new_ctx)
        
        # 4. 构建拓扑
        # True: Header -> Body
        if body_entry:
            self.cfg.add_edge(header_id, body_entry, type=CFGEdgeType.TRUE)
            
            # Back Edge: Body -> Header
            for be in body_exits:
                self.cfg.add_edge(be, header_id, type=CFGEdgeType.BACK_EDGE)
        else:
            # 空 Body，死循环或纯副作用
            # Header -> Header
            self.cfg.add_edge(header_id, header_id, type=CFGEdgeType.BACK_EDGE)
            
        # False Exits: Header 本身也是退出点 (当列表遍历完时)
        # 所以该结构的出口包含：
        # 1. 正常循环结束 (Header -> Next) -> 我们把 header_id 返回作为出口之一
        # 2. Break (Break -> Next)
        return header_id, [header_id] + loop_breaks
    
    def _looks_like_loop_macro(self, node: Node, next_node: Optional[Node]) -> Tuple[bool, str]:
        """
        启发式判断：
        1. 名字包含 'for_each' (不区分大小写)
        2. 结构上看起来像函数调用
        3. (关键) 后面紧跟了一个节点作为 Body
        
        Returns: (is_loop, macro_name)
        """
        if not next_node:
            return False, ""

        # 1. 提取函数名
        macro_name = ""
        
        # 情况 A: 直接是 call_expression
        if node.type == 'call_expression':
            func_node = node.child_by_field_name('function')
            if func_node: macro_name = self._get_text(func_node)
            
        # 情况 B: expression_statement 包裹 call_expression (最常见)
        elif node.type == 'expression_statement':
            # 通常结构: expression_statement -> call_expression
            # 我们只看第一个 child
            if node.child_count > 0:
                child = node.children[0]
                if child.type == 'call_expression':
                    func_node = child.child_by_field_name('function')
                    if func_node: macro_name = self._get_text(func_node)

        if not macro_name:
            return False, ""

        # 2. 命名检查 (Heuristic 1)
        name_lower = macro_name.lower()
        # 涵盖 for_each, foreach, list_entry_loop 等变体
        is_name_match = "for_each" in name_lower or "foreach" in name_lower
        
        if not is_name_match:
            return False, ""

        # 3. 结构检查 (Heuristic 2) - 也可以在这里加更细致的校验
        # 既然 next_node 存在，且名字像循环，我们大概率可以认为它是
        # 甚至可以检查 next_node 是不是 compound_statement，但 C 语言允许单行循环，所以只要有 next 就行
        
        return True, macro_name
    
    def _process_case_body(self, node: Node, context: dict) -> Tuple[Optional[str], List[str]]:
        stmts = []
        skip = True
        for child in node.children:
            if child.type == ':': skip = False; continue
            if not skip: stmts.append(child)
        
        if not stmts: return None, []
        
        first_entry = None
        current_exits = []
        
        for child in stmts:
            entry, exits = self._process_node(child, context)
            if not entry: continue
            if first_entry is None: first_entry = entry
            if current_exits:
                for ex in current_exits: self.cfg.add_edge(ex, entry, type=CFGEdgeType.FLOW)
            current_exits = exits
        return first_entry, current_exits

    def _process_condition(self, node: Node) -> Tuple[str, List[str], List[str]]:
        if node.type == 'parenthesized_expression': # 处理 (a && b)
            return self._process_condition(node.children[1])
            
        if node.type == 'binary_expression':
            op = node.child_by_field_name('operator').type
            left = node.child_by_field_name('left')
            right = node.child_by_field_name('right')
            
            if op == '&&':
                l_entry, l_true, l_false = self._process_condition(left)
                r_entry, r_true, r_false = self._process_condition(right)
                for ex in l_true: self.cfg.add_edge(ex, r_entry, type=CFGEdgeType.TRUE)
                return l_entry, r_true, l_false + r_false
            
            elif op == '||':
                l_entry, l_true, l_false = self._process_condition(left)
                r_entry, r_true, r_false = self._process_condition(right)
                for ex in l_false: self.cfg.add_edge(ex, r_entry, type=CFGEdgeType.FALSE)
                return l_entry, l_true + r_true, r_false

        nid = self._create_node(node, CFGNodeType.PREDICATE)
        return nid, [nid], [nid]

    def _create_node(self, node: Optional[Node], type: CFGNodeType, code: str = None, extra_defs: Set[str] = None) -> str:
        nid = f"node_{self.node_counter}"
        self.node_counter += 1
        
        if code is None and node:
            # 提取单行代码并截断，防止 code 字段过长
            code = self._get_text(node).split('\n')[0][:50]
        
        start_line = node.start_point[0] + 1 if node else 0
        end_line = node.end_point[0] + 1 if node else 0
        ast_type = node.type if node else "virtual"
        
        # 初始为空字典
        defs: Dict[str, Set[str]] = {}
        uses: Dict[str, Set[str]] = {}

        # --- 针对不同类型的节点处理 Def/Use ---
        if type == CFGNodeType.ENTRY:
            # 入口节点：将函数参数标记为初始定值
            # 此时参数既提供了初始值 (VALUE)，也代表了资源生命周期的起点 (STATE)
            if extra_defs:
                for v in extra_defs:
                    defs[v] = {"VALUE", "STATE"}
            uses = {} # Entry 节点不使用任何变量

        elif type in (CFGNodeType.EXIT, CFGNodeType.MERGE, CFGNodeType.NO_OP):
            # 虚拟节点：不产生任何数据流
            defs = {}
            uses = {}

        else:
            # 普通代码节点：调用修改后的提取函数
            # defs/uses 现在都是 Dict[str, Set[str]]
            defs, uses = self._extract_def_use(node) if node else ({}, {})
            
            # 如果有额外的定义（例如通过参数传入的特定标志）
            if extra_defs:
                for v in extra_defs:
                    # 默认标记为 VALUE 定义
                    defs.setdefault(v, set()).add("VALUE")
        
        # 创建节点对象
        cfg_node = CFGNode(
            id=nid, 
            type=type, 
            code=code or "", 
            start_line=start_line, 
            end_line=end_line,
            ast_type=ast_type, 
            defs=defs, 
            uses=uses
        )
        
        # 将节点存入 NetworkX 图中
        self.cfg.add_node(nid, **cfg_node.dict())
        return nid

    def _get_text(self, node: Node) -> str:
        return self.source_bytes[node.start_byte:node.end_byte].decode('utf8')

    def _resolve_gotos(self):
        for src_id, label_name in self.gotos:
            if label_name in self.labels:
                self.cfg.add_edge(src_id, self.labels[label_name], type=CFGEdgeType.JUMP)

    def _update_context(self, old_ctx, **kwargs):
        ctx = old_ctx.copy() if old_ctx else {}
        ctx.update(kwargs)
        return ctx

    def _extract_parameters(self, declarator: Node) -> Set[str]:
        params = set()
        if not declarator: return params
        
        param_list = None
        # 寻找 parameter_list
        # 注意：函数指针或复杂声明可能层级较深，这里简化处理
        stack = [declarator]
        while stack:
            curr = stack.pop()
            if curr.type == 'parameter_list':
                param_list = curr
                break
            stack.extend(curr.children)
            
        if param_list:
            for param in param_list.children:
                if param.type == 'parameter_declaration':
                    # 提取类型
                    type_node = param.child_by_field_name('type')
                    type_str = self._get_text(type_node) if type_node else "void"
                    
                    # 提取变量名
                    decl = param.child_by_field_name('declarator')
                    if decl:
                        # declarator 可能是 "pointer_declarator" -> "*name"
                        # 我们需要最里面的 identifier
                        var_name = self._get_text(decl)
                        # 如果包含 *，说明是指针类型
                        if "*" in var_name or decl.type == 'pointer_declarator':
                            type_str += "*"
                            # 清洗变量名，去除 *
                            var_name = var_name.replace("*", "").strip()
                        
                        params.add(var_name)
                        # [FIX] 填充符号表
                        self.symbol_table[var_name] = type_str
                        
        return params

    def _is_resource_type(self, type_str: str) -> bool:
        # 启发式判断：指针、结构体指针、特定的内核句柄
        if "*" in type_str: return True
        resource_keywords = {"fd", "handle", "lock", "socket", "request"}
        return any(kw in type_str.lower() for kw in resource_keywords)
    
    def _extract_def_use(self, node: Node) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
        local_defs: Dict[str, Set[str]] = {}
        local_uses: Dict[str, Set[str]] = {}

        def get_all_ids(n: Node) -> Set[str]:
            ids = set()
            if not n: return ids
            captures = self.query_ids.captures(n)
            items = captures.get('id', []) if isinstance(captures, dict) else [c[0] for c in captures]
            for x in items: ids.add(self._get_text(x))
            return ids

        def visit(n: Node, is_call_arg: bool = False, is_deref: bool = False):
            if n is None: return
            
            # 1. 处理赋值 (a = b)
            if n.type == 'assignment_expression':
                left = n.child_by_field_name('left')
                right = n.child_by_field_name('right')
                operator_node = n.child_by_field_name('operator')
                op_text = self._get_text(operator_node) if operator_node else "="
                
                # [DEF]
                def_path = self._get_var_path(left)
                if def_path:
                    local_defs.setdefault(def_path, set()).add("VALUE")
                
                # [USE] 右值
                visit(right, is_call_arg=False, is_deref=False)
                
                # [USE] 左值复杂结构处理
                if left.type != 'identifier':
                    if left.type == 'subscript_expression':
                        # [FIX] 字段名改为 argument
                        visit(left.child_by_field_name('argument'), is_deref=True)
                        visit(left.child_by_field_name('index'), is_deref=False)
                    elif left.type == 'pointer_expression':
                        visit(left.child_by_field_name('argument'), is_deref=True)
                    elif left.type == 'field_expression':
                        visit(left.child_by_field_name('argument'), is_deref=True)
                    else:
                        visit(left, is_deref=False)

                if op_text != "=":
                    if def_path:
                        local_uses.setdefault(def_path, set()).add("VALUE")
                return
                        
            # 2. 函数调用
            elif n.type == 'call_expression':
                args = n.child_by_field_name('arguments')
                if args:
                    for arg in args.children:
                        if arg.type not in (',', '(', ')', 'comment'):
                            arg_path = self._get_var_path(arg)
                            # 清洗得到基变量名用于查表
                            base_var = arg_path.split('->')[0].split('.')[0].replace('*', '').replace('&', '').split('[')[0].strip()
                            var_type = self.symbol_table.get(base_var, "")
                            
                            is_resource = self._is_resource_type(var_type)
                            is_pointer = "*" in var_type or "struct" in var_type # 简化判断
                            
                            # Use (Argument passing)
                            if arg_path:
                                local_uses.setdefault(arg_path, set()).add("VALUE")
                            
                            # Def (Side effects)
                            # 如果是取地址 &a，或者是指针变量 p，视为 Def
                            is_address_of = arg.type in ('unary_expression', 'pointer_expression') and self._get_text(arg.child_by_field_name('operator')) == '&'
                            
                            if is_pointer or is_address_of:
                                # [重要] 标记为 VALUE DEF，防止切片在 func(a) 处断裂
                                if arg_path:
                                    local_defs.setdefault(arg_path, set()).add("VALUE")
                            
                            if is_resource:
                                if arg_path:
                                    local_uses[arg_path].add("STATE")
                                    local_defs.setdefault(arg_path, set()).add("STATE")
                            
                            visit(arg, is_call_arg=True, is_deref=False)
                return
            
            # 3. 指针/数组/成员访问 (右值)
            elif n.type in ('pointer_expression', 'unary_expression'): 
                op_node = n.child_by_field_name('operator')
                arg = n.child_by_field_name('argument')
                op_text = self._get_text(op_node) if op_node else ""
                
                # Case A: 解引用 (*p)
                if op_text == '*':
                    visit(arg, is_call_arg=is_call_arg, is_deref=True)
                    return

                # Case B: 取地址 (&a)
                elif op_text == '&':
                    # 关键逻辑：如果 &a 作为函数参数传入，视为对 a 的赋值 (Value Def)
                    # 例如: scanf("%d", &a) -> Def a
                    if is_call_arg:
                        def_path = self._get_var_path(arg)
                        if def_path:
                            local_defs.setdefault(def_path, set()).add("VALUE")
                            
                            # 注意：如果是 &obj->lock，这同时也是对 lock 的 STATE 使用/修改
                            # 如果 arg 是资源类型，我们已经在 call_expression 层处理了 STATE 标记
                            # 这里主要补全 VALUE DEF
                    
                    # 无论是否是参数，&a 都意味着我们需要用到 a 的地址 (Value Use)
                    # visit arg 时 is_deref=False (因为只是取地址，没读内存)
                    visit(arg, is_call_arg=is_call_arg, is_deref=False)
                    return
                
                # Case C: 其他一元运算 (!a, -a, ~a)
                else:
                    visit(arg, is_call_arg=is_call_arg, is_deref=is_deref)
                    return

            elif n.type == 'subscript_expression': 
                # [FIX] 字段名改为 argument
                visit(n.child_by_field_name('argument'), is_call_arg=is_call_arg, is_deref=True)
                visit(n.child_by_field_name('index'), is_call_arg=is_call_arg, is_deref=False)
                return

            elif n.type == 'field_expression': 
                visit(n.child_by_field_name('argument'), is_call_arg=is_call_arg, is_deref=True)
                return
                
            # 4. 更新 (i++)
            elif n.type == 'update_expression':
                arg = n.child_by_field_name('argument')
                path = self._get_var_path(arg)
                if path:
                    local_defs.setdefault(path, set()).add("VALUE")
                    local_uses.setdefault(path, set()).add("VALUE")
                return
            
            # 5. 声明
            elif n.type == 'declaration':
                base_type = self._get_type_str(n)
                for child in n.children:
                    if child.type == 'init_declarator':
                        decl = child.child_by_field_name('declarator')
                        val = child.child_by_field_name('value')
                        
                        var_name = self._get_text(decl)
                        full_type = base_type
                        if decl.type == 'pointer_declarator':
                            full_type += "*"
                            inner = decl.child_by_field_name('declarator')
                            if inner: var_name = self._get_text(inner)
                        
                        self.symbol_table[var_name] = full_type
                        local_defs.setdefault(var_name, set()).add("VALUE")
                        
                        if val: visit(val)
                        
                    elif child.type in ('identifier', 'pointer_declarator'):
                        var_name = self._get_text(child)
                        full_type = base_type
                        if child.type == 'pointer_declarator':
                            full_type += "*"
                            inner = child.child_by_field_name('declarator')
                            if inner: var_name = self._get_text(inner)
                        
                        self.symbol_table[var_name] = full_type
                        local_defs.setdefault(var_name, set()).add("VALUE")
                return
            
            # 6. 标识符
            elif n.type == 'identifier':
                var_name = self._get_text(n)
                var_type = self.symbol_table.get(var_name, "")
                is_resource = self._is_resource_type(var_type)
                
                local_uses.setdefault(var_name, set()).add("VALUE")
                
                if is_resource and (is_call_arg or is_deref):
                    local_uses[var_name].add("STATE")
                    if is_call_arg:
                        local_defs.setdefault(var_name, set()).add("STATE")
                return
            
            # 7. 递归
            else:
                for child in n.children:
                    if child.type not in ('comment', ';', '{', '}', '(', ')'):
                        visit(child, is_call_arg=is_call_arg, is_deref=is_deref)

        start_node = node
        if node.type == 'expression_statement':
            for c in node.children:
                if c.type not in (';', 'comment'):
                    start_node = c
                    break
        
        visit(start_node)
        return local_defs, local_uses
    
    def _get_var_path(self, node: Node) -> str:
        """递归提取完整变量路径 (e.g. obj->member, *p, arr[i])"""
        if not node: return ""
        text = self._get_text(node)
        
        if node.type == 'identifier':
            return text
        
        elif node.type == 'field_expression':
            arg = node.child_by_field_name('argument')
            field = node.child_by_field_name('field')
            return f"{self._get_var_path(arg)}->{self._get_text(field)}"
        
        # [FIX] 需要检查操作符，区分 *p (解引用) 和 &a (取地址)
        elif node.type in ('pointer_expression', 'unary_expression'):
            op = node.child_by_field_name('operator')
            arg = node.child_by_field_name('argument')
            op_text = self._get_text(op) if op else ""
            
            if op_text == '*':
                return f"*{self._get_var_path(arg)}"
            elif op_text == '&':
                # 取地址通常不作为变量路径的一部分 (def &a 其实是 def a)
                # 或者返回 &a 也可以，取决于你的符号表怎么存
                # 这里建议：返回去掉 & 的路径，因为我们追踪的是变量本身
                return self._get_var_path(arg)
                
        elif node.type == 'subscript_expression':
            arr = node.child_by_field_name('argument')
            return self._get_var_path(arr)
        
        return text

    def _get_type_str(self, node: Node) -> str:
        """从声明节点提取类型字符串 (增强版)"""
        if not node: return ""
        # 收集所有相关的类型描述符
        parts = []
        # 遍历 type 节点的子节点 (处理 const unsigned int 等)
        type_node = node.child_by_field_name('type')
        if type_node:
            # 如果是 primitive_type 或 struct_specifier，直接取文本
            return self._get_text(type_node)
        
        # Fallback: 线性扫描 (针对复杂的声明结构)
        for child in node.children:
            if child.type in ('type_identifier', 'primitive_type', 'struct_specifier', 
                              'union_specifier', 'enum_specifier', 'type_qualifier', 'sized_type_specifier'):
                parts.append(self._get_text(child))
            # 碰到 declarator 就停止
            if child.type == 'declarator':
                break
        return " ".join(parts) if parts else "unknown"