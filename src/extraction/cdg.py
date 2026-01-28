import networkx as nx
from typing import Dict, Set, List, Tuple

class CDGBuilder:
    def __init__(self, cfg: nx.DiGraph):
        self.cfg = cfg
        self.cdg = nx.DiGraph()
        # 复制所有节点到 CDG (保持 ID 和属性)
        for node, data in cfg.nodes(data=True):
            self.cdg.add_node(node, **data)
            
    def build(self) -> nx.DiGraph:
        """
        构建控制依赖图 (CDG)
        """
        # 1. 获取入口和出口
        # 注意：必须确保 CFG 只有一个虚拟 EXIT 节点，且所有路径最终汇聚于此
        exit_id = self.cfg.graph.get('exit')
        if not exit_id or exit_id not in self.cfg:
            raise ValueError("CFG must have a valid 'exit' node marked in graph['exit']")

        # 2. 构建反向图 (Reverse CFG)
        # Post-Dominator on CFG == Dominator on Reverse CFG
        r_cfg = self.cfg.reverse()
        
        # 3. 计算后支配树 (PDT)
        # networkx.immediate_dominators 返回字典: {node: dominator}
        # 在 r_cfg 中，start node 是原图的 exit
        try:
            # immediate_dominators 要求从 start_node 可达所有其他节点
            # 如果有死代码（不可达 EXIT），这些节点不会出现在 pdoms 中
            pdoms = nx.immediate_dominators(r_cfg, exit_id)
        except Exception as e:
            print(f"[CDG] Error computing post-dominators: {e}")
            # Fallback: 仅对能到达 exit 的子图计算
            reachable = nx.ancestors(self.cfg, exit_id)
            reachable.add(exit_id)
            sub_r_cfg = r_cfg.subgraph(reachable)
            pdoms = nx.immediate_dominators(sub_r_cfg, exit_id)

        # 4. 计算控制依赖
        # 遍历 CFG 中的每一条边 (A -> B)
        for u, v, data in self.cfg.edges(data=True):
            # 忽略不可达 EXIT 的节点（死代码没有定义的控制依赖）
            if u not in pdoms or v not in pdoms:
                continue
            
            # 定义：如果 v 不后支配 u (即 v 不是 u 在 PDT 中的祖先)
            # 在 PDT 字典表示中，pdoms[u] 是 u 的 parent
            # 检查 v 是否后支配 u: 检查 v 是否出现在 u 的 pdom 链条上
            # 但有一个更简单的算法 (Ferrante et al.):
            
            # 算法核心：
            # 对每条边 (u, v)，如果 v 不是 u 的直接后支配者 (Strict Post-Dominator)
            # 或者更准确地说：如果 v 不后支配 u (即 u -> v 是导致流向分叉的关键边)
            # 实际上，只要是分支边 (TRUE/FALSE/SWITCH)，v 通常都不后支配 u
            
            # 我们查找从 v 到 ipdom(u) 的 PDT 路径 (不包含 ipdom(u))
            # 路径上的所有节点 w 都是 u 的控制依赖节点
            
            # 找到 u 的最近后支配者 (IPDOM)
            u_ipdom = pdoms.get(u)
            
            # 游标 runner 从 v 开始沿着 PDT 向上爬
            runner = v
            
            # 停止条件：
            # 1. runner == u_ipdom (到达了汇聚点)
            # 2. runner == None (到达 EXIT 根节点)
            # 3. 避免死循环 (虽然 PDT 是树，不应该有环)
            
            # 特殊情况：自循环 A->A。A 依赖于 A。此时 runner=A, u_ipdom=A的parent。
            # runner != u_ipdom 成立，添加 A->A 依赖。
            
            while runner != u_ipdom and runner is not None:
                # 添加控制依赖边: u (Controller) -> runner (Dependent)
                # 边的属性：沿用了 CFG 边的属性 (例如 'TRUE' 或 'FALSE')
                # 这表示：runner 只有在 u 取该分支时才执行
                
                # 避免重复添加 (虽然 nx 会处理，但为了逻辑清晰)
                if not self.cdg.has_edge(u, runner):
                    label = data.get('type', 'FLOW')
                    # 只有分支类型的边才有意义标记 True/False
                    if label in ('FLOW', 'BACK_EDGE', 'FALLTHROUGH'):
                        # 对于 Loop Back Edge (Body -> Header)，Body 依赖于 Header (True)
                        # 但反过来，Header 依赖于 Body 吗？不。
                        # 通常我们只关心 PREDICATE 产生的依赖
                        label = 'CONTROL' 
                    
                    self.cdg.add_edge(u, runner, type=label)
                
                # 向上爬 PDT
                if runner == exit_id:
                    break # 到达根节点
                runner = pdoms.get(runner)

        return self.cdg