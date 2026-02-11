"""
Vulnerability Report Generation (Paper §3.1.4)

This module transforms the conceptual semantic hypothesis into a concrete
vulnerability report by grounding it with typed anchors and extracted slices.

Per Paper Definition 3, the VulnerabilityReport captures:
- Vulnerability type: The identified category
- Root cause: High-level description of the defect
- Attack chain: Step-by-step trace along the typed anchor chain with line numbers
- Patch defense: What the patch modifies and why

Evidence Grounding Tasks:
1. Anchor Mapping: Link typed anchors to specific code statements
2. Path Instantiation: Trace data/control flow through slices along the constraint chain
3. Fix Grounding: Identify which specific code changes break the attack chain
"""

from core.state import PatchExtractionState
from core.models import (
    SemanticFeature, SliceFeature, TaxonomyFeature,
    EvidenceRef
)
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, List
import os
from langchain_openai import ChatOpenAI
import re
from pydantic import BaseModel


class SemanticExtractor:
    def __init__(self):
        self.llm = ChatOpenAI(
            base_url=os.getenv("API_BASE"),
            api_key=os.getenv("API_KEY"),
            model=os.getenv("MODEL_NAME", "gpt-4o"), 
            temperature=0
        )
        
        # §3.2.3: Vulnerability Report Generation - Evidence Grounding Prompt
        # Only 3 tasks as per Methodology
        self.analysis_prompt = ChatPromptTemplate.from_template(
            """
            You are a Senior Security Researcher performing **Evidence Grounding** for a vulnerability report.
            Your task: Transform the conceptual semantic hypothesis into a concrete vulnerability report by mapping it to typed anchors and extracted slices.
            
            ### Context
            - **Vulnerability Type**: {vuln_type}
            - **Type Confidence**: {type_confidence}
            
            ### Typed Anchor Specification (from Vulnerability Category)
            - **Expected Anchor Types**: {anchor_types}
            - **Constraint Chain**: {chain_description}
            
            ### Input Data
            
            1. **Semantic Hypothesis** (Conceptual):
            - **Root Cause**: {hyp_root_cause}
            - **Attack Chain**: {hyp_attack_chain}
            - **Patch Defense**: {hyp_patch_defense}
            
            2. **Typed Anchors** (with types and line locations):
            {validated_anchors}
            
            3. **Vulnerability & Patch Slices** (extracted code):
            {slices_content}
            
            4. **Commit Message** (Developer Intent):
            {commit_message}
            
            ---
            
            ### Evidence Grounding Tasks
            
            #### Task 1: Anchor Mapping
            Link typed anchors to specific code statements:
            - For each anchor type in the chain ({anchor_types}), verify the code statement
            - Cite exact line numbers from the slices
            - Verify variable bindings and semantic roles match the hypothesis
            
            #### Task 2: Path Instantiation
            Trace data/control flow through slices along the constraint chain:
            - Build a step-by-step trace with evidence IDs: "func:pre:123", "func:post:456"
            - Follow the chain: {chain_description}
            - Cite intermediate statements (line numbers, functions, variables)
            - For inter-procedural cases, include call chains across functions
            
            #### Task 3: Fix Grounding
            Identify which specific code changes break the attack chain:
            - Cite exact blocking points (line numbers in post-patch slices)
            - Explain why each change prevents the vulnerability
            - Categorize defense type: validation check, state reset, resource cleanup, etc.
            
            ---
            
            ### Output Format (Structured Evidence)
            Return JSON with structured evidence references:
            {{
                "anchor_mapping": {{
                    "anchors": [
                        {{"func": "...", "version": "pre/post", "line": 123, "code": "...", "anchor_type": "alloc/dealloc/use/source/sink/...", "reasoning": "..."}}
                    ]
                }},
                "attack_chain": {{
                    "steps": [
                        {{"step": "Line 123: alloc returns null", "evidence_ids": ["func:pre:123"]}},
                        {{"step": "Line 130: null flows to ptr", "evidence_ids": ["func:pre:130"]}},
                        {{"step": "Line 456: ptr dereferenced without check", "evidence_ids": ["func:pre:456"]}}
                    ]
                }},
                "fix_effect": {{
                    "blocking_points": [
                        {{"description": "Null check added", "evidence_ids": ["func:post:125"], "defense_type": "validation"}}
                    ],
                    "patch_defense": "...",
                    "security_guarantee": "..."
                }},
                "summaries": {{
                    "root_cause": "One-sentence technical summary with concrete evidence",
                    "attack_chain": "Anchor chain flow with line references",
                    "patch_defense": "High-level remediation description with blocking point references"
                }}
            }}
            """
        )

        
    # §3.2.3: Helper to format validated anchors for LLM prompt
    def _format_validated_anchors(self, slices_map: Dict[str, SliceFeature]) -> str:
        """
        Format typed anchors from SliceFeature (pre_anchors, post_anchors)
        into a structured text for LLM consumption.
        """
        lines = []
        for func_name, feat in slices_map.items():
            lines.append(f"=== Function: {func_name} ===")
            
            # Typed anchors (pre-patch)
            if feat.pre_anchors:
                lines.append("  Pre-Patch Typed Anchors:")
                for a in feat.pre_anchors:
                    lines.append(f"    [{a.type.value}] Line {a.line_number}: {a.code_snippet}")
            
            # Typed anchors (post-patch, for fix grounding)
            if feat.post_anchors:
                lines.append("  Post-Patch Typed Anchors:")
                for a in feat.post_anchors:
                    lines.append(f"    [{a.type.value}] Line {a.line_number}: {a.code_snippet}")
            
            lines.append("")
        
        return "\n".join(lines)

    
    def analyze(self, slices_map: Dict[str, SliceFeature], taxonomy: TaxonomyFeature, commit_message: str) -> SemanticFeature:
        """
        §3.2.3: Vulnerability Report Generation
        
        Transform conceptual semantic hypothesis (from §3.2.1) into concrete vulnerability report
        by grounding it with validated anchors and extracted slices (from §3.2.2).
        
        Per Methodology Definition 3, the report captures:
        - Vulnerability type (from taxonomy)
        - Root cause (grounded with evidence)
        - Attack path (step-by-step Origin→Impact trace with evidence IDs)
        - Fix mechanism (specific code changes with blocking points)
        """
        
        # §3.2.3: Build evidence index from slices (all lines become checkable evidence)
        def _parse_slice_evidence(func_name: str, version: str, slice_text: str) -> Dict[str, EvidenceRef]:
            """Build EvidenceRef index from slice lines: '[ 123] code...'"""
            ev: Dict[str, EvidenceRef] = {}
            if not slice_text:
                return ev
            for raw in slice_text.splitlines():
                m = re.match(r'^\[\s*(\d+)\]\s*(.*)$', raw.strip())
                if not m:
                    continue
                ln = int(m.group(1))
                code = m.group(2)
                evidence_id = f"{func_name}:{version}:{ln}"
                ev[evidence_id] = EvidenceRef(
                    evidence_id=evidence_id,
                    func_name=func_name,
                    version=version,  # type: ignore[arg-type]
                    line_number=ln,
                    code=code,
                )
            return ev
        
        # Build evidence index (all slice lines)
        evidence_index: Dict[str, EvidenceRef] = {}
        for func_name, feat in slices_map.items():
            evidence_index.update(_parse_slice_evidence(func_name, "pre", feat.s_pre))
            evidence_index.update(_parse_slice_evidence(func_name, "post", feat.s_post))
        
        # §3.2.3: Format validated anchors for LLM (Task 1: Anchor Mapping input)
        validated_anchors_text = self._format_validated_anchors(slices_map)
        
        # §3.2.3: Format slices for LLM (Task 2: Path Instantiation input)
        slices_text = []
        for func, feat in slices_map.items():
            section = (
                f"=== Function: {func} ===\n"
                f"[Pre-Patch Code (Vulnerable)]:\n{feat.s_pre}\n\n"
                f"[Post-Patch Code (Fixed)]:\n{feat.s_post}\n"
            )
            slices_text.append(section)
        full_context = "\n".join(slices_text)
        
        # Get semantic hypothesis from taxonomy (§3.2.1 output)
        hyp_root = taxonomy.root_cause if taxonomy.root_cause else "N/A"
        hyp_attack = taxonomy.attack_chain if taxonomy.attack_chain else "N/A"
        hyp_fix = taxonomy.patch_defense if taxonomy.patch_defense else "N/A"
        
        # Format anchor types and constraint chain from taxonomy
        anchor_types_str = ", ".join(at.value if hasattr(at, 'value') else str(at) for at in taxonomy.anchor_types) if taxonomy.anchor_types else "N/A"
        chain_description = taxonomy.chain_description if taxonomy.chain_description else "N/A"
        
        # §3.2.3: Define structured LLM output matching Methodology requirements
        class AnchorMappingItem(BaseModel):
            func: str
            version: str  # "pre" or "post"
            line: int
            code: str
            role: str  # e.g., "Alloc", "Def", "Use", "Deref"
            reasoning: str
        
        class AnchorMapping(BaseModel):
            anchors: List[AnchorMappingItem]
        
        class AttackChainStep(BaseModel):
            step: str
            evidence_ids: List[str]
        
        class AttackChainOutput(BaseModel):
            steps: List[AttackChainStep]
        
        class BlockingPoint(BaseModel):
            description: str
            evidence_ids: List[str]
            defense_type: str  # e.g., "validation", "state_reset", "resource_cleanup"
        
        class FixEffectOutput(BaseModel):
            blocking_points: List[BlockingPoint]
            patch_defense: str
            security_guarantee: str
        
        class SummariesOutput(BaseModel):
            root_cause: str
            attack_chain: str
            patch_defense: str
        
        # Complete output structure (simplified to match Methodology 3 tasks)
        class EvidenceGroundingOutput(BaseModel):
            anchor_mapping: AnchorMapping
            attack_chain: AttackChainOutput
            fix_effect: FixEffectOutput
            summaries: SummariesOutput
        
        # §3.2.3: Invoke LLM for evidence grounding
        grounding_result: EvidenceGroundingOutput = (
            self.analysis_prompt | self.llm.with_structured_output(EvidenceGroundingOutput)
        ).invoke({
            "vuln_type": taxonomy.vuln_type.value,
            "type_confidence": taxonomy.type_confidence.value,
            "anchor_types": anchor_types_str,
            "chain_description": chain_description,
            "hyp_root_cause": hyp_root,
            "hyp_attack_chain": hyp_attack,
            "hyp_patch_defense": hyp_fix,
            "validated_anchors": validated_anchors_text,
            "slices_content": full_context,
            "commit_message": commit_message,
        })
        
        # §3.2.3: Return SemanticFeature (Definition 3, simplified structure)
        return SemanticFeature(
            # Component 1: Vulnerability Type & Root Cause
            vuln_type=taxonomy.vuln_type,
            cwe_id=taxonomy.cwe_id,
            cwe_name=taxonomy.cwe_name,
            root_cause=grounding_result.summaries.root_cause,
            
            # Component 2: Attack Chain (with evidence references embedded in text)
            attack_chain=grounding_result.summaries.attack_chain,
            
            # Component 3: Patch Defense
            patch_defense=grounding_result.summaries.patch_defense,
            
            # Evidence Index (for verification phase)
            evidence_index=evidence_index
        )


def semantic_node(state: PatchExtractionState) -> Dict:
    extractor = SemanticExtractor()
    # Pass taxonomy, internal reading of hypothesis and anchor roles
    semantic_result = extractor.analyze(state['slices'], state['taxonomy'], state['commit_message'])
    print(f'[semantic_node] Extracted semantics: \n{semantic_result}')
    return {"semantics": semantic_result}
