import networkx as nx
# 假设你的模块路径如下，根据实际情况调整
from extraction.cfg import CFGBuilder
from extraction.cdg import CDGBuilder
from extraction.ddg import DDGBuilder

class PDGBuilder:
    def __init__(self, code_str: str, lang: str = "c"):
        self.code_str = code_str
        self.lang = lang
        
    def build(self, target_line: int = None) -> nx.MultiDiGraph:
        # 1. 构建 CFG
        cfg_builder = CFGBuilder(self.lang)
        cfg = cfg_builder.build(self.code_str, target_line=target_line)
        
        # 2. 构建 CDG
        cdg_builder = CDGBuilder(cfg)
        cdg = cdg_builder.build()
        
        # 3. 构建 DDG
        ddg_builder = DDGBuilder(cfg)
        ddg = ddg_builder.build()
        
        # 4. 融合为 MultiDiGraph
        pdg = nx.MultiDiGraph()
        
        # 复制节点
        for node, data in cfg.nodes(data=True):
            pdg.add_node(node, **data)
            
        # 添加控制依赖边
        for u, v, data in cdg.edges(data=True):
            pdg.add_edge(u, v, relationship='CONTROL', type=data.get('type'))
            
        # 添加数据依赖边 (核心修正)
        for u, v, data in ddg.edges(data=True):
            pdg.add_edge(
                u, v, 
                relationship='DATA', 
                var=data.get('var'), 
                implicit=data.get('implicit', False),
                flow_type=data.get('flow_type', 'VALUE') # [FIX] 必须保留流类型
            )
            
        return pdg