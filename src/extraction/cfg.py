import tree_sitter_c
import tree_sitter_cpp
from tree_sitter import Language, Parser, Node
import networkx as nx
from enum import Enum
from typing import List, Set, Optional, Tuple, Dict, Any
from pydantic import BaseModel, Field

# --- Data Models ---

class CFGNodeType(str, Enum):
    ENTRY = "ENTRY"
    EXIT = "EXIT"
    STATEMENT = "STATEMENT"
    PREDICATE = "PREDICATE"       # Condition for if/while/for
    MERGE = "MERGE"               # Virtual node for convergence
    NO_OP = "NO_OP"               # No operation
    
    # Structure specific
    SWITCH_HEAD = "SWITCH_HEAD"
    CASE_LABEL = "CASE_LABEL"
    DEFAULT_LABEL = "DEFAULT_LABEL"

class CFGEdgeType(str, Enum):
    FLOW = "FLOW"                 # Sequential flow
    TRUE = "TRUE"                 # Condition true
    FALSE = "FALSE"               # Condition false
    JUMP = "JUMP"                 # goto/break/continue/return
    FALLTHROUGH = "FALLTHROUGH"   # Switch case fallthrough
    BACK_EDGE = "BACK_EDGE"       # Loop back edge
    
    SWITCH_MATCH = "SWITCH_MATCH"       # Switch head matches Case
    SWITCH_DEFAULT = "SWITCH_DEFAULT"   # Switch head matches Default

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

# --- CFG Builder ---

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
        
        # State management
        self.node_counter = 0
        self.labels: Dict[str, str] = {}    # label_name -> node_id
        self.gotos: List[Tuple[str, str]] = [] # (src_node_id, label_name)
        
        # Tree-sitter query
        self.query_ids = self.language.query("(identifier) @id")
        
        self.symbol_table: Dict[str, str] = {} # var_name -> type_string
    
    TERMINAL_FUNCTIONS = {"panic", "exit", "die", "BUG", "BUG_ON", "assert", "kfree_skb"}
    
    def build(self, code_str: str, target_line: Optional[int] = None) -> nx.DiGraph:
        # [FIX 1] Thoroughly reset all states to prevent cross-function contamination
        self.node_counter = 0
        self.labels.clear()
        self.gotos.clear()
        self.symbol_table.clear()
        self.cfg = nx.DiGraph()
        
        self.source_bytes = code_str.encode('utf8')
        tree = self.parser.parse(self.source_bytes)
        
        # Find function definition
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
            # Extract function header as Entry
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
            
            # Handle function body
            body = func_def.child_by_field_name('body')
            body_entry, body_exits = self._process_node(body, context=None)
            
            if body_entry:
                self.cfg.add_edge(self.entry_id, body_entry, type=CFGEdgeType.FLOW)
                for ex in body_exits:
                    self.cfg.add_edge(ex, self.exit_id, type=CFGEdgeType.FLOW)
            else:
                self.cfg.add_edge(self.entry_id, self.exit_id, type=CFGEdgeType.FLOW)
        
        else:
            # Fallback: Code fragment
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

    # --- Recursive Processing ---

    def _process_node(self, node: Node, context: dict) -> Tuple[Optional[str], List[str]]:
        if node is None: return None, []
        
        if node.type in ('comment', 'preproc_def', 'preproc_include'):
            return None, []
        
        if node.type == 'else_clause':
            # else_clause definition is usually: "else" statement
            # We need to find that statement child node and process recursively
            for child in node.children:
                if child.type not in ('else', 'comment'):
                    return self._process_node(child, context)
            return None, []
        
        # 1. Containers
        if node.type in ('translation_unit', 'compound_statement'):
            # Filter out punctuation and comments
            children = [c for c in node.children if c.type not in ('{', '}', 'comment')]
            if not children: return None, []

            first_entry = None
            current_exits = []
            
            i = 0
            while i < len(children):
                child = children[i]
                next_child = children[i+1] if i + 1 < len(children) else None
                
                # --- Heuristic Check ---
                is_macro, macro_name = self._looks_like_loop_macro(child, next_child)
                
                if is_macro:
                    # Hit! This is a macro loop
                    # child is Header (Init/Cond), next_child is Body
                    body_node = next_child
                    
                    # 1. Process macro loop
                    entry, exits = self._process_macro_loop(child, body_node, context, macro_name)
                    
                    # 2. Key: Skip next node, because it's Body, already fused in macro logic
                    i += 2
                else:
                    # Normal node
                    entry, exits = self._process_node(child, context)
                    i += 1

                # --- Connection Logic (Unchanged) ---
                if not entry: continue # Could be declaration or empty statement
                
                if first_entry is None: first_entry = entry
                
                if current_exits:
                    for ex in current_exits:
                        self.cfg.add_edge(ex, entry, type=CFGEdgeType.FLOW)
                
                current_exits = exits
            
            return first_entry, current_exits

        # 2. Function definition (if internal)
        if node.type == 'function_definition':
            body = node.child_by_field_name('body')
            return self._process_node(body, context)

        # 3. Expression/Declaration
        if node.type in ('expression_statement', 'declaration'):
            # Check if it is Call
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
                return nid, [] # No exit, flow terminates here
            
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
            # Nested Key: Use new break list, but inherit/overwrite continue target
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
            # Nested Key: break points to switch_breaks, continue remains unchanged (inherited from outer Loop)
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
        Handle list_for_each_entry and similar macros.
        Header: The macro call itself (Acts as Init + Cond + Step)
        Body: The statement block following the macro
        """
        # 1. Create Header Node
        # Mark it as PREDICATE because it decides if the loop continues
        # Or add a new type MACRO_LOOP_HEAD
        header_id = self._create_node(
            header_node, 
            CFGNodeType.PREDICATE, # Or LOOP_COND
            code=f"MACRO: {self._get_text(header_node).splitlines()[0]}"
        )
        
        # 2. Prepare Context (Handle break/continue inside Body)
        # Continue jumps back to Header
        # Break jumps out of Loop
        loop_breaks = []
        new_ctx = self._update_context(context, break_targets=loop_breaks, continue_target=header_id)
        
        # 3. Process Body
        body_entry, body_exits = self._process_node(body_node, new_ctx)
        
        # 4. Build Topology
        # True: Header -> Body
        if body_entry:
            self.cfg.add_edge(header_id, body_entry, type=CFGEdgeType.TRUE)
            
            # Back Edge: Body -> Header
            for be in body_exits:
                self.cfg.add_edge(be, header_id, type=CFGEdgeType.BACK_EDGE)
        else:
            # Empty Body, infinite loop or pure side effect
            # Header -> Header
            self.cfg.add_edge(header_id, header_id, type=CFGEdgeType.BACK_EDGE)
            
        # False Exits: Header itself is also an exit point (when list traversal completes)
        # So the exits of this structure include:
        # 1. Normal loop end (Header -> Next) -> We return header_id as one of the exits
        # 2. Break (Break -> Next)
        return header_id, [header_id] + loop_breaks
    
    def _looks_like_loop_macro(self, node: Node, next_node: Optional[Node]) -> Tuple[bool, str]:
        """
        Heuristic check:
        1. Name contains 'for_each' (case-insensitive)
        2. Structurally looks like a function call
        3. (Critical) Followed immediately by a node serving as Body
        
        Returns: (is_loop, macro_name)
        """
        if not next_node:
            return False, ""

        # 1. Extract function name
        macro_name = ""
        
        # Case A: Directly a call_expression
        if node.type == 'call_expression':
            func_node = node.child_by_field_name('function')
            if func_node: macro_name = self._get_text(func_node)
            
        # Case B: expression_statement wraps call_expression (Most common)
        elif node.type == 'expression_statement':
            # Usually structure: expression_statement -> call_expression
            # We only look at the first child
            if node.child_count > 0:
                child = node.children[0]
                if child.type == 'call_expression':
                    func_node = child.child_by_field_name('function')
                    if func_node: macro_name = self._get_text(func_node)

        if not macro_name:
            return False, ""

        # 2. Naming Check (Heuristic 1)
        name_lower = macro_name.lower()
        # Covers for_each, foreach, list_entry_loop variants
        is_name_match = "for_each" in name_lower or "foreach" in name_lower
        
        if not is_name_match:
            return False, ""

        # 3. Structure Check (Heuristic 2) - can add more detailed validation here
        # Since next_node exists, and name looks like loop, we assume it is highly probable
        # Can even check if next_node is compound_statement, but C allows single-line loop, so just having next is enough
        
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
        if node.type == 'parenthesized_expression': # Handle (a && b)
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
            # Extract single line code and truncate to prevent code field from being too long
            code = self._get_text(node).split('\n')[0][:50]
        
        start_line = node.start_point[0] + 1 if node else 0
        end_line = node.end_point[0] + 1 if node else 0
        ast_type = node.type if node else "virtual"
        
        # Initial empty dictionary
        defs: Dict[str, Set[str]] = {}
        uses: Dict[str, Set[str]] = {}

        # --- Handle Def/Use for different node types ---
        if type == CFGNodeType.ENTRY:
            # Entry node: Mark function parameters as initial definitions
            # Parameters provide both initial value (VALUE) and represent resource lifecycle start (STATE)
            if extra_defs:
                for v in extra_defs:
                    defs[v] = {"VALUE", "STATE"}
            uses = {} # Entry node does not use any variables

        elif type in (CFGNodeType.EXIT, CFGNodeType.MERGE, CFGNodeType.NO_OP):
            # Virtual node: No data flow
            defs = {}
            uses = {}

        else:
            # Normal code node: Call modified extraction function
            # defs/uses are now Dict[str, Set[str]]
            defs, uses = self._extract_def_use(node) if node else ({}, {})
            
            # If there are extra definitions (e.g., specific flags passed via parameters)
            if extra_defs:
                for v in extra_defs:
                    # Default marked as VALUE definition
                    defs.setdefault(v, set()).add("VALUE")
        
        # Create node object
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
        
        # Add node to NetworkX graph
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
        # Find parameter_list
        # Note: Function pointers or complex declarations might be deep, simplified handling here
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
                    # Extract type
                    type_node = param.child_by_field_name('type')
                    type_str = self._get_text(type_node) if type_node else "void"
                    
                    # Extract variable name
                    decl = param.child_by_field_name('declarator')
                    if decl:
                        # declarator might be "pointer_declarator" -> "*name"
                        # We need the innermost identifier
                        var_name = self._get_text(decl)
                        # If contains *, it means pointer type
                        if "*" in var_name or decl.type == 'pointer_declarator':
                            type_str += "*"
                            # Clean variable name, remove *
                            var_name = var_name.replace("*", "").strip()
                        
                        params.add(var_name)
                        # [FIX] Fill symbol table
                        self.symbol_table[var_name] = type_str
                        
        return params

    def _is_resource_type(self, type_str: str) -> bool:
        # Heuristic judgment: pointers, structure pointers, specific kernel handles
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
            
            # 1. Handle assignment (a = b)
            if n.type == 'assignment_expression':
                left = n.child_by_field_name('left')
                right = n.child_by_field_name('right')
                operator_node = n.child_by_field_name('operator')
                op_text = self._get_text(operator_node) if operator_node else "="
                
                # [DEF]
                def_path = self._get_var_path(left)
                if def_path:
                    local_defs.setdefault(def_path, set()).add("VALUE")
                
                # [USE] R-value
                visit(right, is_call_arg=False, is_deref=False)
                
                # [USE] L-value complex structure handling
                if left.type != 'identifier':
                    if left.type == 'subscript_expression':
                        # [FIX] Field name changed to argument
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
                        
            # 2. Function calls
            elif n.type == 'call_expression':
                args = n.child_by_field_name('arguments')
                if args:
                    for arg in args.children:
                        if arg.type not in (',', '(', ')', 'comment'):
                            arg_path = self._get_var_path(arg)
                            # Clean to get base variable name for table lookup
                            base_var = arg_path.split('->')[0].split('.')[0].replace('*', '').replace('&', '').split('[')[0].strip()
                            var_type = self.symbol_table.get(base_var, "")
                            
                            is_resource = self._is_resource_type(var_type)
                            is_pointer = "*" in var_type or "struct" in var_type # Simplified check
                            
                            # Use (Argument passing)
                            if arg_path:
                                local_uses.setdefault(arg_path, set()).add("VALUE")
                            
                            # Def (Side effects)
                            # If address-of &a, or pointer variable p, treat as Def
                            is_address_of = arg.type in ('unary_expression', 'pointer_expression') and self._get_text(arg.child_by_field_name('operator')) == '&'
                            
                            if is_pointer or is_address_of:
                                # [Important] Mark as VALUE DEF to prevent slice from breaking at func(a)
                                if arg_path:
                                    local_defs.setdefault(arg_path, set()).add("VALUE")
                            
                            if is_resource:
                                if arg_path:
                                    local_uses[arg_path].add("STATE")
                                    local_defs.setdefault(arg_path, set()).add("STATE")
                            
                            visit(arg, is_call_arg=True, is_deref=False)
                return
            
            # 3. Pointer/Array/Member access (R-value)
            elif n.type in ('pointer_expression', 'unary_expression'): 
                op_node = n.child_by_field_name('operator')
                arg = n.child_by_field_name('argument')
                op_text = self._get_text(op_node) if op_node else ""
                
                # Case A: Dereference (*p)
                if op_text == '*':
                    visit(arg, is_call_arg=is_call_arg, is_deref=True)
                    return

                # Case B: Address-of (&a)
                elif op_text == '&':
                    # Key Logic: If &a is passed as function argument, treat as assignment to a (Value Def)
                    # Example: scanf("%d", &a) -> Def a
                    if is_call_arg:
                        def_path = self._get_var_path(arg)
                        if def_path:
                            local_defs.setdefault(def_path, set()).add("VALUE")
                            
                            # Note: If &obj->lock, this is also STATE USE/MODIFICATION of lock
                            # If arg is resource type, we already handled STATE marking at call_expression level
                            # Here mainly completing VALUE DEF
                    
                    # Regardless of whether it's an argument, &a means we need to use address of a (Value Use)
                    # When visiting arg, is_deref=False (Because just taking address, not reading memory)
                    visit(arg, is_call_arg=is_call_arg, is_deref=False)
                    return
                
                # Case C: Other unary operations (!a, -a, ~a)
                else:
                    visit(arg, is_call_arg=is_call_arg, is_deref=is_deref)
                    return

            elif n.type == 'subscript_expression': 
                # [FIX] Field name changed to argument
                visit(n.child_by_field_name('argument'), is_call_arg=is_call_arg, is_deref=True)
                visit(n.child_by_field_name('index'), is_call_arg=is_call_arg, is_deref=False)
                return

            elif n.type == 'field_expression': 
                visit(n.child_by_field_name('argument'), is_call_arg=is_call_arg, is_deref=True)
                return
                
            # 4. Updates (i++)
            elif n.type == 'update_expression':
                arg = n.child_by_field_name('argument')
                path = self._get_var_path(arg)
                if path:
                    local_defs.setdefault(path, set()).add("VALUE")
                    local_uses.setdefault(path, set()).add("VALUE")
                return
            
            # 5. Declarations
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
            
            # 6. Identifiers
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
            
            # 7. Recursion
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
        """Recursively extract full variable path (e.g. obj->member, *p, arr[i])"""
        if not node: return ""
        text = self._get_text(node)
        
        if node.type == 'identifier':
            return text
        
        elif node.type == 'field_expression':
            arg = node.child_by_field_name('argument')
            field = node.child_by_field_name('field')
            return f"{self._get_var_path(arg)}->{self._get_text(field)}"
        
        # [FIX] Need to check operator to distinguish *p (dereference) and &a (address-of)
        elif node.type in ('pointer_expression', 'unary_expression'):
            op = node.child_by_field_name('operator')
            arg = node.child_by_field_name('argument')
            op_text = self._get_text(op) if op else ""
            
            if op_text == '*':
                return f"*{self._get_var_path(arg)}"
            elif op_text == '&':
                # Address-of usually not part of variable path (def &a is actually def a)
                # Or return &a works too, depends on how your symbol table stores it
                # Suggestion here: return path without &, because we track the variable itself
                return self._get_var_path(arg)
                
        elif node.type == 'subscript_expression':
            arr = node.child_by_field_name('argument')
            return self._get_var_path(arr)
        
        return text

    def _get_type_str(self, node: Node) -> str:
        """Extract type string from declaration node (Enhanced)"""
        if not node: return ""
        # Collect all relevant type descriptors
        parts = []
        # Iterate over children of type node (Handle const unsigned int, etc.)
        type_node = node.child_by_field_name('type')
        if type_node:
            # If primitive_type or struct_specifier, get text directly
            return self._get_text(type_node)
        
        # Fallback: Linear scan (For complex declaration structures)
        for child in node.children:
            if child.type in ('type_identifier', 'primitive_type', 'struct_specifier', 
                              'union_specifier', 'enum_specifier', 'type_qualifier', 'sized_type_specifier'):
                parts.append(self._get_text(child))
            # Stop when encountering declarator
            if child.type == 'declarator':
                break
        return " ".join(parts) if parts else "unknown"