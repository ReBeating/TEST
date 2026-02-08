import networkx as nx
# Assuming your module path is as follows, adjust according to actual situation
from extraction.cfg import CFGBuilder
from extraction.cdg import CDGBuilder
from extraction.ddg import DDGBuilder

class PDGBuilder:
    def __init__(self, code_str: str, lang: str = "c"):
        self.code_str = code_str
        self.lang = lang
        
    def build(self, target_line: int = None) -> nx.MultiDiGraph:
        # 1. Build CFG
        cfg_builder = CFGBuilder(self.lang)
        cfg = cfg_builder.build(self.code_str, target_line=target_line)
        
        # 2. Build CDG
        cdg_builder = CDGBuilder(cfg)
        cdg = cdg_builder.build()
        
        # 3. Build DDG
        ddg_builder = DDGBuilder(cfg)
        ddg = ddg_builder.build()
        
        # 4. Merge into MultiDiGraph
        pdg = nx.MultiDiGraph()
        
        # Copy nodes
        for node, data in cfg.nodes(data=True):
            pdg.add_node(node, **data)
            
        # Add control dependence edges
        for u, v, data in cdg.edges(data=True):
            pdg.add_edge(u, v, relationship='CONTROL', type=data.get('type'))
            
        # Add data dependence edges (Core fix)
        for u, v, data in ddg.edges(data=True):
            pdg.add_edge(
                u, v, 
                relationship='DATA', 
                var=data.get('var'), 
                implicit=data.get('implicit', False),
                flow_type=data.get('flow_type', 'VALUE') # [FIX] Must preserve flow type
            )
            
        return pdg