import networkx as nx
from typing import Dict, Set, Tuple, Any
from collections import deque

class DDGBuilder:
    def __init__(self, cfg: nx.DiGraph):
        self.cfg = cfg
        # 建议使用 MultiDiGraph 以防同一对节点间存在多种依赖
        self.ddg = nx.MultiDiGraph()
        for n, d in cfg.nodes(data=True):
            self.ddg.add_node(n, **d)

    def build(self) -> nx.MultiDiGraph:
        # 1. 预计算 Gen 和 Kill (适配字典格式)
        gen = {}
        kill = {}
        entry_id = self.cfg.graph.get('entry')

        for n, data in self.cfg.nodes(data=True):
            # defs 格式: {"var_name": {"VALUE", "STATE"}}
            defs_map = data.get('defs', {})
            var_names = set(defs_map.keys())
            
            gen[n] = {(v, n) for v in var_names}
            kill[n] = var_names

        # 2. 迭代求解到达定值 (RD)
        in_sets = {n: set() for n in self.cfg.nodes()}
        out_sets = {n: set() for n in self.cfg.nodes()}
        
        if entry_id:
            out_sets[entry_id] = gen[entry_id]

        # 优化：使用 deque 和逆后序排序提高收敛速度
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

        # 3. 建立数据依赖边
        for n, data in self.cfg.nodes(data=True):
            uses_map = data.get('uses', {})
            
            for var_name, roles in uses_map.items():
                # 寻找到达当前节点 n 的关于 var_name 的定值点
                reaching_defs = [src for (v, src) in in_sets[n] if v == var_name]
                
                for src_id in reaching_defs:
                    src_node_defs = self.cfg.nodes[src_id].get('defs', {})
                    src_roles = src_node_defs.get(var_name, set())
                    
                    # 场景 1: 值的传递 (Value Flow)
                    if "VALUE" in src_roles and "VALUE" in roles:
                        self.ddg.add_edge(src_id, n, relationship='DATA', 
                                        flow_type='VALUE', var=var_name)
                    
                    # 场景 2: 状态的传递 (State Flow)
                    # 核心逻辑：只要当前节点是 STATE USE (如 free 或 解引用)，
                    # 无论前序是 VALUE DEF (malloc) 还是 STATE DEF (giveback)，
                    # 都建立状态依赖边。
                    if "STATE" in roles:
                        self.ddg.add_edge(src_id, n, relationship='DATA', 
                                        flow_type='STATE', var=var_name)
                                        
                # 增强：处理隐式输入 (全局变量、参数)
                if not reaching_defs and entry_id and n != entry_id:
                    # 排除常量和关键字
                    if var_name not in ('NULL', 'true', 'false', 'stdin', 'stdout', 'stderr'):
                        # [微调] 根据当前 Use 的角色决定隐式边的类型
                        # 如果当前是 STATE USE，说明这个外部变量的状态被改变/使用了
                        if "STATE" in roles:
                            self.ddg.add_edge(entry_id, n, relationship='DATA', 
                                            flow_type='STATE', var=var_name, implicit=True)
                        
                        # 默认总是添加 VALUE 边（因为任何操作都需要读取值）
                        # 或者检查 if "VALUE" in roles
                        if "VALUE" in roles:
                            self.ddg.add_edge(entry_id, n, relationship='DATA', 
                                            flow_type='VALUE', var=var_name, implicit=True)

        return self.ddg