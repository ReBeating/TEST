"""
§3.1.2 Vulnerability Type Determination

Implements the VERDICT paper's type determination methodology:
  Phase 1: CWE direct mapping from NVD/OSV metadata (85.7% coverage)
  Phase 2: LLM root cause analysis with Numeric priority rule
  Phase 3: Hypothesis generation with type knowledge injection

Key principle: Classification is based on ROOT CAUSE, not manifestation.
If a computation step produces an incorrect value flowing to a sink,
classify as Numeric-Domain even if the symptom is OOB/Access.

Numeric-Domain chain model: source →d computation →d sink
"""

import os
import re
import json
import time
from typing import Dict, List, Optional, Any, Tuple

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from core.state import PatchExtractionState
from core.models import (
    TaxonomyFeature,
    AtomicPatch,
    GeneralVulnType,
    TypeConfidence,
)
from core.categories import (
    VulnerabilityKnowledgeBase,
    AnchorType,
    AnchorLocatability,
    AssumptionType,
    MajorCategory
)
from core.configs import VUL_METADATA_PATH


# ==============================================================================
# Connection retry utility
# ==============================================================================

def retry_on_connection_error(func, max_retries=3, initial_delay=2.0, backoff_factor=2.0):
    """
    Retry wrapper for LLM calls with exponential backoff.
    
    Args:
        func: Callable that performs LLM invocation
        max_retries: Maximum number of retry attempts (default: 3)
        initial_delay: Initial delay in seconds before first retry (default: 2.0)
        backoff_factor: Multiplier for delay between retries (default: 2.0)
        
    Returns:
        Result of func() if successful
        
    Raises:
        Last exception if all retries fail
    """
    last_exception = None
    delay = initial_delay
    
    for attempt in range(1, max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_exception = e
            error_str = str(e).lower()
            
            is_connection_error = any(keyword in error_str for keyword in [
                'connection', 'timeout', 'timed out', 'network',
                'refused', 'reset', 'broken pipe', 'unreachable'
            ])
            
            if is_connection_error and attempt < max_retries:
                print(f"    [Retry] Connection error on attempt {attempt}/{max_retries}: {e}")
                print(f"    [Retry] Waiting {delay:.1f}s before retry...")
                time.sleep(delay)
                delay *= backoff_factor
            else:
                raise
    
    raise last_exception


# ==============================================================================
# Metadata loading
# ==============================================================================

def load_metadata(cve_id: str) -> Optional[Dict[str, Any]]:
    """Load CVE metadata from the vulnerability metadata JSON file."""
    try:
        if not os.path.exists(VUL_METADATA_PATH):
            return None
        with open(VUL_METADATA_PATH, 'r', encoding='utf-8') as f:
            all_metadata = json.load(f)
        return all_metadata.get(cve_id)
    except Exception as e:
        print(f"    [Warning] Failed to load metadata for {cve_id}: {e}")
        return None


def parse_cwe_id(cwe_str: Optional[str]) -> Optional[int]:
    """
    Parse a CWE string (e.g., 'CWE-190') into its numeric ID (190).
    Returns None if parsing fails.
    """
    if not cwe_str:
        return None
    match = re.search(r'CWE-(\d+)', str(cwe_str))
    if match:
        return int(match.group(1))
    # Try bare number
    try:
        return int(str(cwe_str).strip())
    except (ValueError, TypeError):
        return None


# ==============================================================================
# NOTE: map_anchor_type_to_role() and get_anchor_spec() have been REMOVED.
# Per paper §3.1.3, the Origin/Impact binary classification is replaced by
# typed anchors (AnchorType from categories.py). Downstream consumers now
# access anchor types and constraints directly via TaxonomyFeature.anchor_types
# and TaxonomyFeature.constraint (which delegate to VulnerabilityKnowledgeBase).
# ==============================================================================


# ==============================================================================
# Context builder
# ==============================================================================

class ContextBuilder:
    """Build LLM context from patches."""
    
    @staticmethod
    def build_analysis_context(patches: List[AtomicPatch], max_chars: int = 20000) -> str:
        """
        Build analysis context: Diff (focus on changes) + Full Old Code (semantic context).
        """
        context_parts = []
        
        for p in patches:
            diff_text = p.clean_diff if p.clean_diff else p.raw_diff
            full_code = p.old_code if p.old_code else "N/A (New Function?)"
            
            start_line = p.start_line_old if p.start_line_old is not None else 1
            numbered_lines = []
            for idx, line in enumerate(full_code.splitlines()):
                numbered_lines.append(f"[{start_line + idx}] {line}")
            full_code_numbered = "\n".join(numbered_lines)

            section = (
                f"=== File: {p.file_path} | Function: {p.function_name} ===\n"
                f"--- [PART 1: The Patch Diff] (Focus on what CHANGED) ---\n"
                f"{diff_text}\n\n"
                f"--- [PART 2: Full Vulnerable Function] (Focus on Logic Flow & Sinks) ---\n"
                f"{full_code_numbered}\n"
                f"{'='*60}\n"
            )
            context_parts.append(section)
            
        return "\n".join(context_parts)


# ==============================================================================
# LLM structured output models
# ==============================================================================

class RootCauseAnalysis(BaseModel):
    """Phase 2 LLM output: root cause analysis with Numeric priority check."""
    
    # Root cause analysis
    root_cause_mechanism: str = Field(
        description="Specific description of the fundamental defect mechanism"
    )
    
    # Numeric determination - the core of §3.1.2
    involves_numeric_computation: bool = Field(
        description=(
            "Does the vulnerability chain contain a COMPUTATION step "
            "(arithmetic, type conversion, size calculation) that produces "
            "an incorrect value flowing to a sink? If YES, this is Numeric-Domain."
        )
    )
    numeric_evidence: Optional[str] = Field(
        default=None,
        description="If involves_numeric_computation=true, describe the computation and how it leads to the sink"
    )
    
    # Final classification
    final_major_category: str = Field(
        description="One of: Numeric-Domain, Access-Validity, Resource-Lifecycle, Control-Logic"
    )
    final_subtype: str = Field(
        description="Specific vulnerability subtype name from the Knowledge Base"
    )
    
    # Confidence and reasoning
    confidence: str = Field(
        description="High (CWE+code confirm), Medium (code analysis), Low (uncertain)"
    )
    reasoning: str = Field(
        description="Detailed reasoning for the classification"
    )


class HypothesisResult(BaseModel):
    """Phase 3 output: vulnerability hypothesis."""
    root_cause: str
    attack_chain: str
    patch_defense: str


# ==============================================================================
# §3.1.2 TypeDeterminer — Main classifier
# ==============================================================================

class TypeDeterminer:
    """
    §3.1.2 Vulnerability Type Determination.
    
    Three-phase approach:
      Phase 1: CWE direct mapping from metadata → initial category (no LLM)
      Phase 2: LLM root cause analysis with Numeric priority rule (1 LLM call)
      Phase 3: Hypothesis generation with type knowledge injection (1 LLM call)
    
    Core principle: Classification by ROOT CAUSE, not manifestation.
    Numeric-Domain chain: source →d computation →d sink
    If a computation step produces an incorrect value flowing to a sink,
    classify as Numeric-Domain even if the symptom is OOB write/read.
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(
            base_url=os.getenv("API_BASE"),
            api_key=os.getenv("API_KEY"),
            model=os.getenv("MODEL_NAME", "gpt-4o"),
            temperature=0
        )
        
        # Phase 3: Hypothesis generation prompt
        self.hypothesis_prompt = ChatPromptTemplate.from_template(
            """You are an Expert C/C++ Security Researcher generating a semantic hypothesis for a {vuln_type} vulnerability.

### Expert Knowledge for {vuln_type}
**Typical Vulnerability Pattern**:
{vulnerability_chain}

**Key Anchors to Look For**:
{anchors_desc}

These anchors define the structural pattern of the vulnerability.
- **Concrete**: Has specific location and clear semantics (e.g., malloc, free). Can be sliced and searched.
- **Assumed**: Has specific location but needs assumption about semantics (e.g., function parameter assumed controllable).
- **Conceptual**: No specific location, purely inferred existence. Cannot be sliced or searched.

### Task
Based on the above expert knowledge, generate a CONCEPTUAL hypothesis for THIS SPECIFIC patch:

1. **Root Cause**: What fundamental defect enabled the vulnerability?
   - Be specific but conceptual (e.g., "missing overflow check before size computation")
   - Do NOT use concrete line numbers

2. **Attack Path**: How does the vulnerability manifest in THIS patch?
   - Follow the anchor chain from the expert knowledge
   - Adapt to this specific instance
   - Be conceptual, describing the logical flow, NOT concrete line numbers

3. **Fix Mechanism**: What does the patch do to prevent the vulnerability?
   - Explain the defense mechanism

### Input

**CVE Description** (if available):
{description}

**Commit Message**:
{commit_message}

**Patch Context**:
{diff_content}

### Output
Return a JSON object with:
- root_cause: string (conceptual)
- attack_chain: string (conceptual, specific to this patch, no line numbers)
- patch_defense: string (conceptual)"""
        )
    
    # =========================================================================
    # Main entry point
    # =========================================================================
    
    def determine_type(
        self,
        patches: List[AtomicPatch],
        commit_msg: str,
        vul_id: str = None
    ) -> TaxonomyFeature:
        """
        Main entry: complete vulnerability type determination + hypothesis generation.
        
        Args:
            patches: List of atomic patches for this vulnerability
            commit_msg: Commit message
            vul_id: CVE/vulnerability ID for metadata lookup
            
        Returns:
            TaxonomyFeature with all fields populated (backward-compatible)
        """
        # Phase 1: Metadata + CWE initial mapping
        metadata = load_metadata(vul_id) if vul_id else None
        initial_category, cwe_id_str, cwe_name, description = self._phase1_cwe_mapping(metadata)
        
        # Build code context
        diff_context = ContextBuilder.build_analysis_context(patches)
        
        # Fast path: CWE is directly Numeric + no contradicting evidence
        if initial_category and self._is_clear_numeric(initial_category, cwe_id_str):
            category_name = initial_category
            confidence = TypeConfidence.HIGH
            reasoning = f"[CWE Direct Mapping] {cwe_id_str} → {category_name} (Numeric fast path)"
            numeric_priority = False  # Was already Numeric, no override needed
            print(f"    [Phase 1] Numeric fast path: {cwe_id_str} → {category_name}")
        else:
            # Phase 2: LLM Root Cause Analysis + Numeric priority + final classification
            category_name, confidence, reasoning, numeric_priority = self._phase2_root_cause_analysis(
                initial_category=initial_category,
                cwe_id_str=cwe_id_str,
                cwe_name=cwe_name,
                description=description,
                commit_msg=commit_msg,
                diff_context=diff_context
            )
        
        # Phase 3: Hypothesis generation with type knowledge injection
        category = VulnerabilityKnowledgeBase.get_category(category_name)
        hypothesis = self._phase3_hypothesis(
            category=category,
            category_name=category_name,
            description=description,
            commit_msg=commit_msg,
            diff_context=diff_context
        )
        
        # Build backward-compatible TaxonomyFeature
        return self._build_taxonomy_feature(
            category_name=category_name,
            category=category,
            confidence=confidence,
            cwe_id_str=cwe_id_str,
            cwe_name=cwe_name,
            reasoning=reasoning,
            hypothesis=hypothesis,
            numeric_priority=numeric_priority
        )
    
    # =========================================================================
    # Phase 1: CWE Direct Mapping
    # =========================================================================
    
    def _phase1_cwe_mapping(
        self, metadata: Optional[Dict[str, Any]]
    ) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        """
        Phase 1: Extract metadata and perform CWE direct mapping.
        
        Returns:
            (initial_category_name, cwe_id_str, cwe_name, description)
            initial_category_name is None if CWE is not in KnowledgeBase mapping.
        """
        if not metadata:
            print(f"    [Phase 1] No metadata available")
            return None, None, None, None
        
        cwe_id_str = metadata.get('cwe_ids', [None])[0] if metadata.get('cwe_ids') else None
        cwe_name = metadata.get('cwe_names', [None])[0] if metadata.get('cwe_names') else None
        description = metadata.get('description', '')
        
        print(f"    [Phase 1] Metadata loaded (CWE: {cwe_id_str})")
        
        # Try CWE direct mapping
        cwe_numeric = parse_cwe_id(cwe_id_str)
        if cwe_numeric is not None:
            category_name = VulnerabilityKnowledgeBase.get_category_name_by_cwe(cwe_numeric)
            if category_name:
                print(f"    [Phase 1] CWE {cwe_id_str} → initial mapping: {category_name}")
                return category_name, cwe_id_str, cwe_name, description
            else:
                print(f"    [Phase 1] CWE {cwe_id_str} not in KnowledgeBase mapping")
        
        return None, cwe_id_str, cwe_name, description
    
    def _is_clear_numeric(self, initial_category: str, cwe_id_str: Optional[str]) -> bool:
        """
        Fast-path check: is the CWE directly a Numeric-Domain type?
        
        If CWE maps to Integer Overflow / Integer Underflow / Divide By Zero / Type Confusion,
        skip Phase 2 LLM call. These CWEs are unambiguously Numeric.
        """
        cwe_numeric = parse_cwe_id(cwe_id_str)
        if cwe_numeric is None:
            return False
        return VulnerabilityKnowledgeBase.is_numeric_cwe(cwe_numeric)
    
    # =========================================================================
    # Phase 2: LLM Root Cause Analysis with Numeric Priority
    # =========================================================================
    
    def _phase2_root_cause_analysis(
        self,
        initial_category: Optional[str],
        cwe_id_str: Optional[str],
        cwe_name: Optional[str],
        description: Optional[str],
        commit_msg: str,
        diff_context: str
    ) -> Tuple[str, TypeConfidence, str, bool]:
        """
        Core phase: one LLM call for root cause analysis + Numeric priority + classification.
        
        Returns:
            (category_name, confidence, reasoning, numeric_priority_applied)
        """
        print(f"    [Phase 2] LLM root cause analysis...")
        
        # Inject KnowledgeBase type descriptions
        type_descriptions = VulnerabilityKnowledgeBase.get_type_descriptions_for_llm()
        valid_names = VulnerabilityKnowledgeBase.get_all_category_names()
        valid_names_str = ", ".join([f'"{n}"' for n in valid_names if n != "Unknown"])
        
        # Build initial mapping hint
        initial_hint = ""
        if initial_category:
            cat = VulnerabilityKnowledgeBase.get_category(initial_category)
            initial_hint = f"""
## CWE Initial Mapping (PRELIMINARY — may be overridden by root cause analysis)
CWE {cwe_id_str} ({cwe_name}) initially maps to: **{initial_category}**
Major category: {cat.major_category.value}
This is the MANIFESTATION type. Your job is to find the TRUE ROOT CAUSE type.
If the root cause involves numeric computation, override to Numeric-Domain.
"""
        
        prompt = f"""You are an expert C/C++ security researcher. Analyze this vulnerability patch to determine the TRUE root cause type.

## CRITICAL RULE: Numeric Priority
The Numeric-Domain chain model is: source →d computation →d sink.
If there is a COMPUTATION step (arithmetic, type conversion, size calculation) that produces an
incorrect value flowing to a sink (array access, malloc, memcpy), classify as **Numeric-Domain**.
The array/memory access is just the SINK of the numeric chain — it does NOT make it Access-Validity.

## Key Question: "Is the value reaching the sink produced by a COMPUTATION?"
- YES → Numeric-Domain (the access is just the sink)
- NO → Classify by manifestation (Access-Validity / Resource-Lifecycle / Control-Logic)

## Numeric-Domain Indicators (ANY → Numeric-Domain)
1. Value reaching sink was produced by arithmetic (e.g., `a + b`, `a * b`, `offset + delta`)
2. Value went through type conversion (e.g., `(int)size_t_value`, signed/unsigned mismatch)
3. Value was computed from multiple sources (e.g., `size = header_len + data_len`)
4. Patch adds overflow/range check on a COMPUTED value before it reaches sink
5. Patch fixes the computation itself (changing arithmetic, widening type)
6. Index/offset used in array access was COMPUTED, not a direct loop counter or constant
7. Patch adds bounds check like `if (idx < len)` BUT idx came from a computation

## NOT Numeric-Domain
- Index is a direct value (simple loop `i++`, constant, direct param) with NO computation → Access-Validity
- NULL pointer dereference (no computation) → Access-Validity
- Resource lifecycle (free, cleanup) → Resource-Lifecycle
- Auth/lock checks → Control-Logic
{initial_hint}

## Available Types (from Knowledge Base)
{type_descriptions}

## IMPORTANT: final_subtype MUST be exactly one of: {valid_names_str}

## CVE Metadata
- CWE: {cwe_id_str or 'N/A'} - {cwe_name or 'N/A'}
- Description: {description or 'N/A'}

## Commit Message
{commit_msg}

## Patch Context
{diff_context}

## Task
1. Analyze the patch to understand WHAT was actually fixed
2. Trace the data flow: identify if a COMPUTATION step produces the value reaching the sink
3. Apply the Numeric priority rule
4. Choose the most specific type — final_subtype MUST be an exact name from Available Types"""
        
        try:
            llm = self.llm.with_structured_output(RootCauseAnalysis)
            result: RootCauseAnalysis = retry_on_connection_error(
                lambda: llm.invoke(prompt),
                max_retries=3
            )
            
            # Validate category name against KnowledgeBase
            category_name = self._validate_category_name(result.final_subtype)
            
            # Determine if Numeric priority was applied (overrode initial mapping)
            numeric_priority = (
                result.involves_numeric_computation
                and initial_category is not None
                and VulnerabilityKnowledgeBase.get_category(initial_category).major_category != MajorCategory.NUMERIC_DOMAIN
            )
            
            # Determine confidence
            if cwe_id_str and cwe_id_str != 'N/A':
                confidence = TypeConfidence.HIGH if result.confidence == "High" else TypeConfidence.MEDIUM
            else:
                confidence = TypeConfidence.MEDIUM if result.confidence != "Low" else TypeConfidence.LOW
            
            # Build reasoning
            reasoning = f"[Root Cause] {result.root_cause_mechanism}"
            if result.involves_numeric_computation:
                reasoning += f"\n[Numeric Chain Detected] {result.numeric_evidence or 'computation → sink pattern found'}"
            if numeric_priority:
                reasoning += f"\n[Numeric Priority Override] Initial CWE mapping was {initial_category}, overridden to {category_name}"
            reasoning += f"\n[Classification] {result.reasoning}"
            
            print(f"    [Phase 2] Result: {category_name} (Confidence: {confidence.value})")
            if numeric_priority:
                print(f"    [Phase 2] ⚡ Numeric priority applied: {initial_category} → {category_name}")
            
            return category_name, confidence, reasoning, numeric_priority
            
        except Exception as e:
            print(f"    [Phase 2] LLM analysis failed: {e}")
            # Fallback: use initial category if available, otherwise Unknown
            fallback = initial_category or "Unknown"
            return fallback, TypeConfidence.LOW, f"[Fallback] LLM failed: {e}", False
    
    def _validate_category_name(self, name: str) -> str:
        """
        Validate that LLM output matches a registered KnowledgeBase category name.
        Falls back to fuzzy matching, then to 'Unknown'.
        """
        valid_names = VulnerabilityKnowledgeBase.get_all_category_names()
        
        # Exact match
        if name in valid_names:
            return name
        
        # Case-insensitive match
        name_lower = name.lower().strip()
        for valid in valid_names:
            if valid.lower() == name_lower:
                return valid
        
        # Fuzzy: check if the LLM output contains a valid name
        for valid in valid_names:
            if valid.lower() in name_lower or name_lower in valid.lower():
                print(f"    [Validation] Fuzzy match: '{name}' → '{valid}'")
                return valid
        
        # Try mapping via GeneralVulnType enum values
        for vt in GeneralVulnType:
            if vt.value.lower() == name_lower:
                # Check if this enum value maps to a KB category
                cat = VulnerabilityKnowledgeBase.get_category(vt.value)
                if cat.__class__.__name__ != 'UnknownViolation':
                    return vt.value
        
        print(f"    [Validation] Could not match '{name}' to any KB category, using 'Unknown'")
        return "Unknown"
    
    # =========================================================================
    # Phase 3: Hypothesis Generation
    # =========================================================================
    
    def _phase3_hypothesis(
        self,
        category,  # VulnerabilityCategory instance
        category_name: str,
        description: Optional[str],
        commit_msg: str,
        diff_context: str
    ) -> HypothesisResult:
        """
        Phase 3: Generate root_cause / attack_chain / patch_defense hypothesis
        with type knowledge injection from VulnerabilityKnowledgeBase.
        """
        print(f"    [Phase 3] Generating hypothesis with {category_name} knowledge...")
        
        # Build vulnerability chain from category anchors
        chain_parts = [f"{a.description} ({a.type.value})" for a in category.anchors]
        vulnerability_chain = (
            f"{category.description}\nChain: " + " → ".join(chain_parts)
            if chain_parts
            else category.description
        )
        
        # Build anchor description with locatability
        anchors_desc = "\n".join([
            f"- **{a.type.value}** ({a.locatability.value}): {a.description}"
            for a in category.anchors
        ]) if category.anchors else "No specific anchors defined for this type."
        
        try:
            hyp_llm = self.llm.with_structured_output(HypothesisResult)
            hyp_chain = self.hypothesis_prompt | hyp_llm
            
            result: HypothesisResult = retry_on_connection_error(
                lambda: hyp_chain.invoke({
                    "vuln_type": category_name,
                    "vulnerability_chain": vulnerability_chain,
                    "anchors_desc": anchors_desc,
                    "description": description or "N/A",
                    "commit_message": commit_msg,
                    "diff_content": diff_context
                }),
                max_retries=3
            )
            
            print(f"    [Phase 3] Hypothesis generated")
            return result
            
        except Exception as e:
            print(f"    [Phase 3] Hypothesis generation failed: {e}")
            return HypothesisResult(
                root_cause="Unknown - hypothesis generation failed",
                attack_chain="Unknown vulnerability pattern",
                patch_defense="Unknown fix mechanism"
            )
    
    # =========================================================================
    # Build backward-compatible TaxonomyFeature
    # =========================================================================
    
    def _build_taxonomy_feature(
        self,
        category_name: str,
        category,  # VulnerabilityCategory instance
        confidence: TypeConfidence,
        cwe_id_str: Optional[str],
        cwe_name: Optional[str],
        reasoning: str,
        hypothesis: HypothesisResult,
        numeric_priority: bool
    ) -> TaxonomyFeature:
        """
        Build TaxonomyFeature with all legacy fields populated for backward compatibility.
        """
        # Map category_name to GeneralVulnType enum
        vuln_type = self._category_name_to_vuln_type(category_name)
        
        return TaxonomyFeature(
            # Primary type identifier (from categories.py KB)
            category_name=category_name,
            # Legacy compatibility
            vuln_type=vuln_type,
            type_confidence=confidence,
            cwe_id=cwe_id_str if cwe_id_str and cwe_id_str != "N/A" else None,
            cwe_name=cwe_name if cwe_name and cwe_name != "N/A" else None,
            # NOTE: origin_roles/impact_roles removed — use taxonomy.anchor_types
            # and taxonomy.constraint from categories.py KB instead
            root_cause=hypothesis.root_cause,
            attack_chain=hypothesis.attack_chain,
            patch_defense=hypothesis.patch_defense,
            reasoning=reasoning,
            numeric_priority_applied=numeric_priority,
        )
    
    def _category_name_to_vuln_type(self, category_name: str) -> GeneralVulnType:
        """
        Map KB category name to GeneralVulnType enum.
        Handles name mismatches between KB and enum.
        """
        # Direct match by value
        for vt in GeneralVulnType:
            if vt.value == category_name:
                return vt
        
        # Special mappings for KB names that differ from enum values
        special_map = {
            "Missing Authorization": GeneralVulnType.AUTHORIZATION_BYPASS,
        }
        if category_name in special_map:
            return special_map[category_name]
        
        return GeneralVulnType.UNKNOWN
    
    # NOTE: _build_legacy_roles() has been REMOVED.
    # Origin/Impact role splitting was a lossy simplification.
    # Downstream consumers now use typed anchors from categories.py directly.


# ==============================================================================
# Legacy TaxonomyClassifier — kept as thin wrapper for any direct imports
# ==============================================================================

class TaxonomyClassifier:
    """
    Legacy wrapper. Delegates to TypeDeterminer.
    Kept for backward compatibility with any code that imports TaxonomyClassifier directly.
    """
    
    def __init__(self):
        self._determiner = TypeDeterminer()
    
    def classify_group(
        self, patches: List[AtomicPatch], commit_msg: str, vul_id: str = None
    ) -> TaxonomyFeature:
        return self._determiner.determine_type(patches, commit_msg, vul_id)


# ==============================================================================
# Pipeline node entry function
# ==============================================================================

def taxonomy_node(state: PatchExtractionState):
    """
    §3.1.2 Pipeline node: Vulnerability Type Determination.
    
    Reads: patches, commit_message, vul_id
    Writes: taxonomy (TaxonomyFeature)
    """
    vul_id = state.get('vul_id', 'N/A')
    print(f"  [Taxonomy §3.1.2] Processing Group {state['group_id']} (VUL: {vul_id})...")
    
    determiner = TypeDeterminer()
    feature_result = determiner.determine_type(
        patches=state['patches'],
        commit_msg=state['commit_message'],
        vul_id=vul_id if vul_id != 'N/A' else None
    )
    
    print(f"    → Type: {feature_result.vuln_type.value} (Confidence: {feature_result.type_confidence.value})")
    print(f"    → Category: {feature_result.category_name or feature_result.vuln_type.value}")
    print(f"    → CWE: {feature_result.cwe_id or 'N/A'}")
    print(f"    → Anchor Types: {feature_result.anchor_types}")
    print(f"    → Chain: {feature_result.chain_description}")
    if feature_result.numeric_priority_applied:
        print(f"    → ⚡ Numeric Priority Override Applied")
    return {"taxonomy": feature_result}
