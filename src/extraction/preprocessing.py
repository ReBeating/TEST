import re
from typing import List, Dict
from core.models import AtomicPatch, TaxonomyFeature, SlicingInstruction

class CodeAligner:
    """负责代码行的模糊匹配"""
    
    @staticmethod
    def normalize_string(s: str) -> str:
        """移除空白字符，用于模糊匹配"""
        return re.sub(r'\s+', '', s).strip()

    @staticmethod
    def find_line_number(source_code: str, target_snippet: str) -> int:
        """
        在源代码中寻找目标片段的行号 (1-based)
        策略：
        1. 单行精确匹配
        2. 单行模糊匹配
        3. 跨行模糊匹配 (针对被换行符切断的长语句)
        """
        if not source_code or not target_snippet:
            return -1
            
        lines = source_code.splitlines()
        target_norm = CodeAligner.normalize_string(target_snippet)
        
        # 1. 尝试直接包含匹配 (最快)
        for i, line in enumerate(lines):
            if target_snippet in line:
                return i + 1
        
        # 2. 尝试去空格匹配 (鲁棒)
        if not target_norm:
            return -1
            
        for i, line in enumerate(lines):
            line_norm = CodeAligner.normalize_string(line)
            if target_norm in line_norm:
                return i + 1
        
        # 3. 跨行模糊匹配 (新增)
        # 将源码和目标都归一化为无空格字符串，但保留源码的行号映射
        full_norm = ""
        line_map = [] # index -> line_number
        
        for i, line in enumerate(lines):
            norm = CodeAligner.normalize_string(line)
            start_idx = len(full_norm)
            full_norm += norm
            # 记录这一段归一化字符对应的行号
            # 简单起见，这段字符的每个位置都映射到行 i+1
            line_map.extend([i + 1] * len(norm))
            
        idx = full_norm.find(target_norm)
        if idx != -1:
            return line_map[idx]
            
        return -1

class PatchAnalyzer:
    """分析补丁特征，决定主切片方向 (参考 PDF Step 0)"""
    
    @staticmethod
    def determine_primary_version(diff_text: str) -> str:
        """
        根据 Diff 统计决定主切片侧:
        - Additive (新增为主) -> NEW (Post-Patch)
        - Subtractive (删除为主) -> OLD (Pre-Patch)
        - Modificative (修改) -> NEW (默认)
        """
        added_lines = 0
        removed_lines = 0
        
        for line in diff_text.splitlines():
            if line.startswith('+') and not line.startswith('+++'):
                added_lines += 1
            elif line.startswith('-') and not line.startswith('---'):
                removed_lines += 1
        
        # 启发式阈值：如果删除行数远多于新增，且新增很少，视为删除型补丁
        # 例如：移除了一个 buggy function call
        if removed_lines > 0 and added_lines == 0:
            return "OLD"
        
        # 默认偏向修复后的逻辑 (Fix View)
        return "NEW"

class SlicePreprocessor:
    """Phase 3.2.2 的前置处理器"""
    
    def __init__(self, patch: AtomicPatch, taxonomy: TaxonomyFeature):
        self.patch = patch
        self.taxonomy = taxonomy
        self.aligner = CodeAligner()

    def generate_instructions(self) -> Dict[str, List[SlicingInstruction]]:
        """
        生成切片指令：
        策略 1 (Physical): 从 Patch Diff 中提取被修改的行作为直接锚点。
        策略 2 (Semantic): 从 Hypothesis.expected_key_lines 中提取关键语义行。
        """
        instructions_map = {}
        patch = self.patch
        instructions = []
        
        seen_lines_old = set()
        seen_lines_new = set()

        # Helper to add instruction uniquely
        def add_instr(ver, line, content, origin):
            seen = seen_lines_new if ver == "NEW" else seen_lines_old
            if line in seen: return
            seen.add(line)
            
            instructions.append(SlicingInstruction(
                function_name=patch.function_name,
                target_version=ver,
                line_number=line,
                code_content=content.strip(),
                description=origin,
                strategy="bidirectional" # Default strategy
            ))

        # --- Strategy 1: Diff Anchors (The Ground Truth) ---
        diff_text = patch.clean_diff or patch.raw_diff
        if diff_text:
            for line in diff_text.splitlines():
                # Ignore metadata headers
                if line.startswith(('---', '+++', 'index', 'diff')): continue
                
                content = line[1:].strip()
                # Simple noise filter for comments/empty
                if not content or content.startswith(('//', '/*', '*', '#')): continue 
                
                if line.startswith('-'):
                    # Deleted line -> Should exist in OLD
                    ln = self.aligner.find_line_number(patch.old_code, content)
                    if ln > 0: add_instr("OLD", ln, content, "Diff-Deleted")
                
                elif line.startswith('+'):
                    # Added line -> Should exist in NEW
                    ln = self.aligner.find_line_number(patch.new_code, content)
                    if ln > 0: add_instr("NEW", ln, content, "Diff-Added")

        # --- Strategy 2: Semantic Anchors (From Anchor Roles) ---
        # Note: We now derive semantic hints from origin_roles and impact_roles
        # These provide conceptual guidance for anchor identification
        semantic_roles = []
        if self.taxonomy.origin_roles:
            semantic_roles.extend([role.value for role in self.taxonomy.origin_roles])
        if self.taxonomy.impact_roles:
            semantic_roles.extend([role.value for role in self.taxonomy.impact_roles])
        
        for role_str in semantic_roles:
            clean_key = role_str.strip().lower()
            if not clean_key: continue

            # 1. Re-align for OLD (Vuln representation) using content search
            #    User Note: LLM line numbers might be hallucinated or absolute-offsets not matching local string.
            #    We search the actual content in old_code to get the correct valid line number (1-based relative).
            ln_old = self.aligner.find_line_number(patch.old_code, clean_key)
            if ln_old > 0:
                add_instr("OLD", ln_old, clean_key, "Semantic-Role-Anchor")
            
            # 2. Try NEW (Fix representation) - Search textually
            ln_new = self.aligner.find_line_number(patch.new_code, clean_key)
            if ln_new > 0:
                add_instr("NEW", ln_new, clean_key, "Semantic-Role-Anchor")

        # Assign to function map
        if instructions:
            instructions_map[patch.function_name] = instructions
        else:
             print(f"    [Warn] No instructions generated for {patch.function_name} (checked Diff & Hypothesis)")
            
        return instructions_map