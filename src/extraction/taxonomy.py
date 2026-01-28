import os
import json
import time
import uuid
from typing import Dict, List, Optional, Any
from pydantic import BaseModel

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from core.state import PatchExtractionState
from core.models import (
    TaxonomyFeature,
    AtomicPatch,
    GeneralVulnType,
    TypeConfidence,
    AnchorRole
)
from core.configs import VUL_METADATA_PATH


# ==============================================================================
# 重连重试工具函数
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
            
            # Check if it's a connection-related error
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
                # Not a connection error or last attempt, raise immediately
                raise
    
    # All retries exhausted
    raise last_exception

# --- 锚点规范映射表 ---

ANCHOR_SPECIFICATIONS: Dict[GeneralVulnType, Dict[str, Any]] = {
    # 1. Memory Safety (7)
    GeneralVulnType.USE_AFTER_FREE: {
        "origin_roles": [AnchorRole.FREE],
        "impact_roles": [AnchorRole.USE, AnchorRole.DEREF],
        "vulnerability_chain": "Free(ptr) → Use(ptr): Track pointer propagation from deallocation to use"
    },
    GeneralVulnType.DOUBLE_FREE: {
        "origin_roles": [AnchorRole.FREE],
        "impact_roles": [AnchorRole.FREE],
        "vulnerability_chain": "Free(ptr) → Free(ptr) again: Detect same pointer freed multiple times"
    },
    GeneralVulnType.BUFFER_OVERFLOW: {
        "origin_roles": [AnchorRole.SOURCE, AnchorRole.COMPUTE],
        "impact_roles": [AnchorRole.SINK, AnchorRole.OVERFLOW],
        "vulnerability_chain": "Unbounded Input/Computation → Buffer Write: Track size calculation from source to sink"
    },
    GeneralVulnType.OUT_OF_BOUNDS_READ: {
        "origin_roles": [AnchorRole.SOURCE, AnchorRole.COMPUTE],
        "impact_roles": [AnchorRole.ACCESS],
        "vulnerability_chain": "Unchecked Index → Array/Buffer Access: Track index propagation and bound validation"
    },
    GeneralVulnType.MEMORY_LEAK: {
        "origin_roles": [AnchorRole.ALLOC],
        "impact_roles": [AnchorRole.LEAK],
        "vulnerability_chain": "Alloc → [Error Path] → return without Free: Find paths from allocation to return that miss deallocation"
    },
    GeneralVulnType.NULL_POINTER_DEREFERENCE: {
        "origin_roles": [AnchorRole.DEF, AnchorRole.ASSIGN],
        "impact_roles": [AnchorRole.DEREF, AnchorRole.USE],
        "vulnerability_chain": "Null Assignment/Return → Dereference: Track pointer flow from null source to dereference"
    },
    GeneralVulnType.UNINITIALIZED_USE: {
        "origin_roles": [AnchorRole.DEF],
        "impact_roles": [AnchorRole.USE],
        "vulnerability_chain": "Variable Declaration → Use without Initialization: Find paths where variable is used before being initialized"
    },
    
    # 2. Concurrency (2)
    GeneralVulnType.RACE_CONDITION: {
        "origin_roles": [AnchorRole.CHECK, AnchorRole.ACCESS],
        "impact_roles": [AnchorRole.USE, AnchorRole.CORRUPTION],
        "vulnerability_chain": "Thread A: Check → [Thread B: Concurrent Modify] → Thread A: Use: Analyze concurrent execution interleaving"
    },
    GeneralVulnType.DEADLOCK: {
        "origin_roles": [AnchorRole.LOCK],
        "impact_roles": [AnchorRole.LOCK],
        "vulnerability_chain": "Lock A → Lock B vs. Lock B → Lock A: Detect circular lock dependency"
    },
    
    # 3. Numeric & Type (3)
    GeneralVulnType.INTEGER_OVERFLOW: {
        "origin_roles": [AnchorRole.SOURCE, AnchorRole.COMPUTE],
        "impact_roles": [AnchorRole.OVERFLOW, AnchorRole.SINK],
        "vulnerability_chain": "Unchecked Arithmetic → Overflow → Downstream Use: Track integer value flow and range constraints"
    },
    GeneralVulnType.DIVIDE_BY_ZERO: {
        "origin_roles": [AnchorRole.SOURCE, AnchorRole.DEF],
        "impact_roles": [AnchorRole.DIVIDE, AnchorRole.CRASH],
        "vulnerability_chain": "Zero Value → Division Operation: Track divisor value from source to division"
    },
    GeneralVulnType.TYPE_CONFUSION: {
        "origin_roles": [AnchorRole.DEF, AnchorRole.ASSIGN],
        "impact_roles": [AnchorRole.USE, AnchorRole.CORRUPTION],
        "vulnerability_chain": "Type Cast/Union → Mismatched Use: Track object type transitions and accesses"
    },
    
    # 4. Logic & Access Control (4)
    GeneralVulnType.AUTHENTICATION_BYPASS: {
        "origin_roles": [AnchorRole.CHECK],
        "impact_roles": [AnchorRole.ACCESS],
        "vulnerability_chain": "Flawed Auth Check → Protected Operation: Find control flow paths bypassing authentication"
    },
    GeneralVulnType.PRIVILEGE_ESCALATION: {
        "origin_roles": [AnchorRole.CHECK],
        "impact_roles": [AnchorRole.ACCESS],
        "vulnerability_chain": "Privilege Check → Privileged Operation: Find paths where unprivileged code reaches privileged operations"
    },
    GeneralVulnType.AUTHORIZATION_BYPASS: {
        "origin_roles": [AnchorRole.CHECK],
        "impact_roles": [AnchorRole.ACCESS],
        "vulnerability_chain": "Permission Check → Protected Resource: Find control flow bypassing authorization checks"
    },
    GeneralVulnType.LOGIC_ERROR: {
        "origin_roles": [AnchorRole.CHECK, AnchorRole.BRANCH],
        "impact_roles": [AnchorRole.USE, AnchorRole.CORRUPTION],
        "vulnerability_chain": "Incorrect Logic/Condition → Unexpected Behavior: Analyze control flow and state transitions"
    },
    
    # 5. Input & Data (5)
    GeneralVulnType.INJECTION: {
        "origin_roles": [AnchorRole.SOURCE],
        "impact_roles": [AnchorRole.SINK],
        "vulnerability_chain": "Untrusted Input → Command/Query Execution: Track taint flow from input to interpreter sink"
    },
    GeneralVulnType.PATH_TRAVERSAL: {
        "origin_roles": [AnchorRole.SOURCE],
        "impact_roles": [AnchorRole.ACCESS],
        "vulnerability_chain": "User-Controlled Path → File/Directory Access: Track path string flow and sanitization"
    },
    GeneralVulnType.IMPROPER_VALIDATION: {
        "origin_roles": [AnchorRole.SOURCE],
        "impact_roles": [AnchorRole.USE, AnchorRole.SINK],
        "vulnerability_chain": "Unvalidated Input → Downstream Use: Track input propagation and validation points"
    },
    GeneralVulnType.INFORMATION_EXPOSURE: {
        "origin_roles": [AnchorRole.SOURCE, AnchorRole.DEF],
        "impact_roles": [AnchorRole.LEAK, AnchorRole.SINK],
        "vulnerability_chain": "Sensitive Data → Unintended Output/Log: Track data flow from sensitive source to exposure point"
    },
    GeneralVulnType.CRYPTOGRAPHIC_ISSUE: {
        "origin_roles": [AnchorRole.SOURCE, AnchorRole.COMPUTE],
        "impact_roles": [AnchorRole.USE, AnchorRole.SINK],
        "vulnerability_chain": "Weak Crypto/Key Management → Security Breach: Analyze cryptographic operations and key handling"
    },
    
    # 6. Resource & Execution (4)
    GeneralVulnType.INFINITE_LOOP: {
        "origin_roles": [AnchorRole.BRANCH, AnchorRole.CHECK],
        "impact_roles": [AnchorRole.BRANCH],
        "vulnerability_chain": "Loop Condition → Infinite Iteration: Analyze loop termination conditions"
    },
    GeneralVulnType.RECURSION_ERROR: {
        "origin_roles": [AnchorRole.BRANCH, AnchorRole.CHECK],
        "impact_roles": [AnchorRole.CRASH],
        "vulnerability_chain": "Recursive Call → Stack Exhaustion: Track recursion depth and termination conditions"
    },
    GeneralVulnType.RESOURCE_EXHAUSTION: {
        "origin_roles": [AnchorRole.ACQUIRE, AnchorRole.ALLOC],
        "impact_roles": [AnchorRole.CRASH, AnchorRole.CORRUPTION],
        "vulnerability_chain": "Unbounded Resource Allocation → System Exhaustion: Track resource acquisition without limits"
    },
    GeneralVulnType.RESOURCE_LEAK: {
        "origin_roles": [AnchorRole.ACQUIRE],
        "impact_roles": [AnchorRole.LEAK],
        "vulnerability_chain": "Acquire → [Error Path] → return without Release: Find paths missing resource cleanup"
    },
    
    # 7. Fallback (2)
    GeneralVulnType.UNKNOWN: {
        "origin_roles": [],
        "impact_roles": [],
        "vulnerability_chain": "Unknown vulnerability pattern: Manual analysis required"
    },
    GeneralVulnType.OTHER: {
        "origin_roles": [],
        "impact_roles": [],
        "vulnerability_chain": "Non-standard vulnerability pattern: Requires custom analysis strategy"
    },
}

# --- Helper Functions ---

def load_metadata(cve_id: str) -> Optional[Dict[str, Any]]:
    """Load CVE metadata"""
    try:
        if not os.path.exists(VUL_METADATA_PATH):
            return None
        with open(VUL_METADATA_PATH, 'r', encoding='utf-8') as f:
            all_metadata = json.load(f)
        return all_metadata.get(cve_id)
    except Exception as e:
        print(f"    [Warning] Failed to load metadata for {cve_id}: {e}")
        return None

def get_anchor_spec(vuln_type: GeneralVulnType) -> Dict[str, Any]:
    """Get anchor specification for vulnerability type"""
    return ANCHOR_SPECIFICATIONS.get(vuln_type, ANCHOR_SPECIFICATIONS[GeneralVulnType.UNKNOWN])

# --- Context Builder ---

class ContextBuilder:
    """Build LLM context from patches"""
    
    @staticmethod
    def build_analysis_context(patches: List[AtomicPatch], max_chars: int = 20000) -> str:
        """
        Build analysis context: Provide both Diff (focus on changes) and Full Old Code (provide semantic context).
        """
        context_parts = []
        
        for p in patches:
            # 1. Diff (to understand Fix intent)
            diff_text = p.clean_diff if p.clean_diff else p.raw_diff
            
            # 2. Full old code (to locate Sink and understand context)
            full_code = p.old_code if p.old_code else "N/A (New Function?)"
            
            # Add line numbers to full code
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

# --- Core Classifier ---

class TypeDeterminationResult(BaseModel):
    """Step 1 output"""
    vuln_type: GeneralVulnType
    type_confidence: TypeConfidence
    cwe_id: Optional[str] = None
    cwe_name: Optional[str] = None
    reasoning: str

class HypothesisResult(BaseModel):
    """Step 2 output"""
    root_cause: str
    attack_path: str
    fix_mechanism: str

class TaxonomyClassifier:
    """Two-step classifier with expert knowledge injection"""
    
    def __init__(self):
        # Configure LLM
        self.llm = ChatOpenAI(
            base_url=os.getenv("API_BASE"),
            api_key=os.getenv("API_KEY"),
            model=os.getenv("MODEL_NAME", "gpt-4o"),
            temperature=0
        )
        
        # Step 1: Type Determination Prompt
        self.type_determination_prompt = ChatPromptTemplate.from_template(
            """
            You are an Expert C/C++ Security Researcher performing vulnerability type determination.
            
            ### Task
            Analyze the provided patch to determine the vulnerability type.
            
            ### Guidelines
            - **Use All Available Information**: Leverage CVE metadata (if provided), commit message, and code analysis
            - **Metadata as Hints**: CVE metadata (CWE, description) is usually accurate but can be generic or incorrect
            - **Code is Ground Truth**: If code clearly shows different vulnerability type than metadata suggests, trust the code
            
            ### Input
            
            **CVE Metadata** (Optional - may be N/A if unavailable):
            - CVE ID: {cve_id}
            - CWE ID: {cwe_id}
            - CWE Name: {cwe_name}
            - Description: {description}
            
            **Commit Message**:
            {commit_message}
            
            **Patch Context**:
            {diff_content}
            
            ### Vulnerability Types
            Choose from: {vuln_types}
            
            ### Output
            Return a JSON object with:
            - vuln_type: The most specific vulnerability type from the list above
            - type_confidence: "High" if you have strong ground truth (CVE metadata with clear type), "Medium" if inferred from code only, "Low" if uncertain
            - cwe_id: Optional, from metadata or inferred
            - cwe_name: Optional
            - reasoning: Explain your confidence level and any deviations from metadata
            """
        )
        
        # Step 2: Hypothesis Generation Prompt (with expert knowledge injection)
        self.hypothesis_generation_prompt = ChatPromptTemplate.from_template(
            """
            You are an Expert C/C++ Security Researcher generating a semantic hypothesis for a {vuln_type} vulnerability.
            
            ### Expert Knowledge for {vuln_type}
            **Typical Vulnerability Pattern**:
            {vulnerability_chain}
            
            **Key Anchor Roles to Look For**:
            - **Origin (vulnerability source)**: {origin_roles}
            - **Impact (exploitation point)**: {impact_roles}
            
            These anchor roles indicate what types of operations to focus on when analyzing the vulnerability chain.
            
            ### Task
            Based on the above expert knowledge, generate a CONCEPTUAL hypothesis for THIS SPECIFIC patch:
            
            1. **Root Cause**: What fundamental defect enabled the vulnerability?
               - Be specific but conceptual (e.g., "missing null check before dereference")
               - Do NOT use concrete line numbers
            
            2. **Attack Path**: How does the vulnerability manifest in THIS patch?
               - Follow the Origin→Impact pattern from the expert knowledge
               - Adapt to this specific instance (e.g., "allocation failure → null return → unchecked dereference")
               - Be conceptual, describing the logical flow, NOT concrete line numbers
            
            3. **Fix Mechanism**: What does the patch do to prevent the vulnerability?
               - Explain the defense mechanism (e.g., "adds null check after allocation")
            
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
            - attack_path: string (conceptual, specific to this patch, no line numbers)
            - fix_mechanism: string (conceptual)
            """
        )

    def classify_group(self, patches: List[AtomicPatch], commit_msg: str, vul_id: str = None) -> TaxonomyFeature:
        """Two-step classification with expert knowledge injection"""
        
        # 1. Use provided vul_id directly (no need to extract from group_id)
        cve_id = vul_id
        metadata = load_metadata(cve_id) if cve_id else None
        
        # 2. Prepare metadata fields (optional)
        if metadata:
            cve_id_str = cve_id
            cwe_id = metadata.get('cwe_ids', [None])[0] if metadata.get('cwe_ids') else None
            cwe_name = metadata.get('cwe_names', [None])[0] if metadata.get('cwe_names') else None
            description = metadata.get('description', '')
            print(f"    [Metadata] Loaded CVE {cve_id_str} metadata (CWE: {cwe_id})")
        else:
            cve_id_str = "N/A"
            cwe_id = "N/A"
            cwe_name = "N/A"
            description = "N/A"
            print(f"    [Metadata] No metadata available for CVE {cve_id or 'N/A'}")
        
        # Build context
        diff_context = ContextBuilder.build_analysis_context(patches)
        
        try:
            # === Step 1: Type Determination ===
            print(f"    [Step 1] Determining vulnerability type...")
            type_llm = self.llm.with_structured_output(TypeDeterminationResult)
            type_chain = self.type_determination_prompt | type_llm
            
            vuln_types_str = ", ".join([vt.value for vt in GeneralVulnType])
            type_result: TypeDeterminationResult = retry_on_connection_error(
                lambda: type_chain.invoke({
                    "cve_id": cve_id_str,
                    "cwe_id": cwe_id or "N/A",
                    "cwe_name": cwe_name or "N/A",
                    "description": description or "N/A",
                    "commit_message": commit_msg,
                    "diff_content": diff_context,
                    "vuln_types": vuln_types_str
                }),
                max_retries=3
            )
            
            print(f"    [Step 1] Determined type: {type_result.vuln_type.value} (Confidence: {type_result.type_confidence.value})")
            
            # === Step 2: Hypothesis Generation (with expert knowledge injection) ===
            print(f"    [Step 2] Generating hypothesis with expert knowledge...")
            anchor_spec = get_anchor_spec(type_result.vuln_type)
            vulnerability_chain = anchor_spec['vulnerability_chain']
            
            hyp_llm = self.llm.with_structured_output(HypothesisResult)
            hyp_chain = self.hypothesis_generation_prompt | hyp_llm
            
            # Format anchor roles for prompt
            origin_roles_str = ", ".join([r.value for r in anchor_spec['origin_roles']]) if anchor_spec['origin_roles'] else "N/A"
            impact_roles_str = ", ".join([r.value for r in anchor_spec['impact_roles']]) if anchor_spec['impact_roles'] else "N/A"
            
            hyp_result: HypothesisResult = retry_on_connection_error(
                lambda: hyp_chain.invoke({
                    "vuln_type": type_result.vuln_type.value,
                    "vulnerability_chain": vulnerability_chain,
                    "origin_roles": origin_roles_str,
                    "impact_roles": impact_roles_str,
                    "description": description or "N/A",
                    "commit_message": commit_msg,
                    "diff_content": diff_context
                }),
                max_retries=3
            )
            
            print(f"    [Step 2] Hypothesis generated")
            
            # === Combine results ===
            # If metadata exists, preserve original CWE
            final_cwe_id = cwe_id if (metadata and cwe_id and cwe_id != "N/A") else type_result.cwe_id
            final_cwe_name = cwe_name if (metadata and cwe_name and cwe_name != "N/A") else type_result.cwe_name
            
            result = TaxonomyFeature(
                vuln_type=type_result.vuln_type,
                type_confidence=type_result.type_confidence,
                cwe_id=final_cwe_id,
                cwe_name=final_cwe_name,
                origin_roles=anchor_spec['origin_roles'],
                impact_roles=anchor_spec['impact_roles'],
                root_cause=hyp_result.root_cause,
                attack_path=hyp_result.attack_path,
                fix_mechanism=hyp_result.fix_mechanism,
                reasoning=type_result.reasoning
            )
            
            if metadata:
                print(f"    [Metadata] Used CVE {cve_id_str} metadata (CWE: {final_cwe_id})")
            
            return result
            
        except Exception as e:
            print(f"    [Error] Classification failed: {e}, falling back to UNKNOWN")
            return self._create_fallback(cwe_id, cwe_name)
    
    def _create_fallback(self, cwe_id: Optional[str], cwe_name: Optional[str]) -> TaxonomyFeature:
        """Create fallback result (Low Confidence)"""
        
        return TaxonomyFeature(
            vuln_type=GeneralVulnType.UNKNOWN,
            type_confidence=TypeConfidence.LOW,
            cwe_id=cwe_id if cwe_id != "N/A" else None,
            cwe_name=cwe_name if cwe_name != "N/A" else None,
            origin_roles=[],
            impact_roles=[],
            root_cause="Unknown - requires manual analysis",
            attack_path="Unknown vulnerability pattern",
            fix_mechanism="Unknown fix mechanism",
            reasoning="Fallback (LOW): LLM analysis failed, unable to determine specific vulnerability type"
        )

# --- Node Entry Function ---

def taxonomy_node(state: PatchExtractionState):
    vul_id = state.get('vul_id', 'N/A')
    print(f"  [Taxonomy] Processing Group {state['group_id']} (VUL: {vul_id})...")
    
    classifier = TaxonomyClassifier()
    feature_result = classifier.classify_group(
        patches=state['patches'],
        commit_msg=state['commit_message'],
        vul_id=vul_id if vul_id != 'N/A' else None
    )
    
    print(f"    → Type: {feature_result.vuln_type.value} (Confidence: {feature_result.type_confidence.value})")
    print(f"    → CWE: {feature_result.cwe_id or 'N/A'}")
    print(f"    → Origin Roles: {[r.value for r in feature_result.origin_roles]}")
    print(f"    → Impact Roles: {[r.value for r in feature_result.impact_roles]}")
    
    return {"taxonomy": feature_result}
