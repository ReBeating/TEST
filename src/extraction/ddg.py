import networkx as nx
from typing import Dict, Set, Tuple, Any
from collections import deque

class DDGBuilder:
    def __init__(self, cfg: nx.DiGraph):
        self.cfg = cfg
        # Recommend using MultiDiGraph in case of multiple dependencies between the same pair of nodes
        self.ddg = nx.MultiDiGraph()
        for n, d in cfg.nodes(data=True):
            self.ddg.add_node(n, **d)

    def build(self) -> nx.MultiDiGraph:
        # 1. Pre-calculate Gen and Kill (Adapt to dict format)
        gen = {}
        kill = {}
        entry_id = self.cfg.graph.get('entry')

        for n, data in self.cfg.nodes(data=True):
            # defs format: {"var_name": {"VALUE", "STATE"}}
            defs_map = data.get('defs', {})
            var_names = set(defs_map.keys())
            
            gen[n] = {(v, n) for v in var_names}
            kill[n] = var_names

        # 2. Iteratively solve Reaching Definitions (RD)
        in_sets = {n: set() for n in self.cfg.nodes()}
        out_sets = {n: set() for n in self.cfg.nodes()}
        
        if entry_id:
            out_sets[entry_id] = gen[entry_id]

        # Optimization: Use deque and reverse post-order sort to improve convergence speed
        worklist = deque(self.cfg.nodes())
        while worklist:
            n = worklist.popleft()
            
            new_in = set()
            for p in self.cfg.predecessors(n):
                new_in.update(out_sets[p])
            in_sets[n] = new_in 
            
            survivors = {item for item in new_in if item[0] not in kill[n]}
            new_out = gen[n].union(survivors)
            
            if new_out != out_sets[n]:
                out_sets[n] = new_out
                worklist.extend([s for s in self.cfg.successors(n) if s not in worklist])

        # 3. Build data dependence edges
        for n, data in self.cfg.nodes(data=True):
            uses_map = data.get('uses', {})
            
            for var_name, roles in uses_map.items():
                # Find definition points for var_name reaching current node n
                reaching_defs = [src for (v, src) in in_sets[n] if v == var_name]
                
                for src_id in reaching_defs:
                    src_node_defs = self.cfg.nodes[src_id].get('defs', {})
                    src_roles = src_node_defs.get(var_name, set())
                    
                    # Scene 1: Value Flow
                    if "VALUE" in src_roles and "VALUE" in roles:
                        self.ddg.add_edge(src_id, n, relationship='DATA', 
                                        flow_type='VALUE', var=var_name)
                    
                    # Scene 2: State Flow
                    # Core Logic: As long as current node is STATE USE (e.g. free or dereference),
                    # No matter if predecessor is VALUE DEF (malloc) or STATE DEF (giveback),
                    # Establish state dependence edge.
                    if "STATE" in roles:
                        self.ddg.add_edge(src_id, n, relationship='DATA', 
                                        flow_type='STATE', var=var_name)
                                        
                # Enhancement: Handle implicit inputs (global variables, parameters)
                if not reaching_defs and entry_id and n != entry_id:
                    # Exclude constants and keywords
                    if var_name not in ('NULL', 'true', 'false', 'stdin', 'stdout', 'stderr'):
                        # [Tweak] Decide implicit edge type based on role of current Use
                        # If current is STATE USE, it means the state of this external variable is changed/used
                        if "STATE" in roles:
                            self.ddg.add_edge(entry_id, n, relationship='DATA', 
                                            flow_type='STATE', var=var_name, implicit=True)
                        
                        # Default always add VALUE edge (because any operation needs to read value)
                        # Or check if "VALUE" in roles
                        if "VALUE" in roles:
                            self.ddg.add_edge(entry_id, n, relationship='DATA', 
                                            flow_type='VALUE', var=var_name, implicit=True)

        return self.ddg