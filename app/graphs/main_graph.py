import os
import json
import logging
import secrets
import csv
from typing import TypedDict, List, Dict, Any, Literal, Optional
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Import all individual functional components built in earlier stages
from app.tools.compliance_tool import ComplianceEngine
from app.tools.csr_sql import execute_select
from app.tools.boq_temp_store import normalize_user_id_for_filename, set_latest_csv
from app.tools.rfi_tool import rfi_draft
from app.tools.chat_tool import general_chat

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MainOrchestrator")

# 1. Define the Unified State Workspace
class AgentState(TypedDict):
    query: str
    user_id: str
    intent: str  # 'COMPLIANCE' | 'COST' | 'RFI' | 'CHAT'
    final_output: str
    citations: List[str]
    sources: List[str]
    compliance_status: str  # set only on COMPLIANCE route
    boq_csv_available: bool

# Initialize Shared System LLM Interface
shared_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)
# Remove or comment out this global line:
# compliance_engine = ComplianceEngine()

# Add this global placeholder instead:
_compliance_engine_instance = None

def get_compliance_engine() -> ComplianceEngine:
    """Lazily instantiates the compliance engine only when called, preventing import-time locks."""
    global _compliance_engine_instance
    if _compliance_engine_instance is None:
        logger.info("Initializing Qdrant Local Client connection safely within execution context...")
        _compliance_engine_instance = ComplianceEngine()
    return _compliance_engine_instance


# ==========================================
# NODE 1: INTENT CLASSIFICATION GATEWAY
# ==========================================
def classify_intent_node(state: AgentState) -> Dict[str, Any]:
    """Analyzes user engineering inquiries and classifies intent for routing."""
    query_text = state["query"]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are the structural engineering router engine for a civil design co-pilot.\n"
            "Classify the incoming user query into exactly ONE of these categories:\n"
            "- COMPLIANCE: If checking designs, configurations, spacing, grades, limits against IS codes.\n"
            "- COST: If asking for estimates, budgets, BOQ line-item pricing, bills of quantities.\n"
            "- RFI: If explicitly asking to draft a Request for Information document for a failure/discrepancy.\n"
            "- CHAT: For greetings, off-scope questions, or general conversation.\n\n"
            "Respond with EXACTLY one word from the list above. Do not include punctuation or explanations."
        )),
        ("user", "{query}")
    ])
    
    messages = prompt.format_messages(query=query_text)
    response = shared_llm.invoke(messages)
    predicted_intent = response.content.strip().upper()
    
    # Valid validation fallback array check
    if predicted_intent not in ["COMPLIANCE", "COST", "RFI", "CHAT"]:
        predicted_intent = "CHAT"
        
    logger.info(f"Route Determination Strategy Identified: {predicted_intent}")
    return {"intent": predicted_intent}

# ==========================================
# NODE 2A: COMPLIANCE AUDITING ROUTE
# ==========================================
def compliance_execution_node(state: AgentState) -> Dict[str, Any]:
    query_text = state["query"]
    engine = get_compliance_engine()

    # Execute the RAG lookup and analysis
    assessment = engine.evaluate_compliance(query_text)

    status = assessment.get("compliance_status", "UNKNOWN")
    answer = assessment.get("answer", "")

    status_prefix = {
        "COMPLIANT_DETERMINED": "Assessment based on retrieved IS code clauses.\n\n",
        "INSUFFICIENT_DATA": "Insufficient evidence in retrieved clauses to verify compliance.\n\n",
        "UNKNOWN": "No matching clauses were retrieved from the standards library.\n\n",
    }.get(status, "")

    return {
        "final_output": status_prefix + answer,
        "citations": assessment.get("applied_citations", []),
        "sources": assessment.get("source_documents", []),
        "compliance_status": status,
    }

# ==========================================
# NODE 2B: SEMANTIC BOQ COST ESTIMATION ROUTE
# ==========================================
def cost_estimation_node(state: AgentState) -> Dict[str, Any]:
    """NLP-to-SQL: generate BOQ query as SQL over CSR SQLite table, then write a temp CSV for download."""
    query_text = state["query"]
    user_id = (state.get("user_id") or "anonymous").strip() or "anonymous"

    def _extract_sql(text: str) -> str:
        return (text or "").replace("```sql", "").replace("```", "").strip()

    def _is_safe_select_sql(sql: str) -> Optional[str]:
        s = (sql or "").strip()
        if not s:
            return "empty SQL"
        # Allow only SELECT / WITH queries, no multiple statements.
        lowered = s.lower()
        if ";" in s.strip().rstrip(";"):
            return "multiple statements are not allowed"
        if not (lowered.startswith("select") or lowered.startswith("with")):
            return "only SELECT/WITH queries are allowed"
        blocked = ["insert", "update", "delete", "drop", "alter", "create", "attach", "detach", "pragma", "vacuum"]
        if any(f" {kw} " in f" {lowered} " for kw in blocked):
            return "non-select keyword detected"
        if "csr_rates" not in lowered:
            return "query must reference csr_rates"
        return None

    sql_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are an NLP-to-SQL engine for a BOQ (Bill of Quantities) cost estimate.\n"
                    "Generate a SINGLE SQLite SELECT query over this table:\n\n"
                    "Table: csr_rates(item_code TEXT, description TEXT, unit TEXT, rate_inr REAL)\n\n"
                    "Return columns exactly with these aliases:\n"
                    "- item_code\n"
                    "- quantity\n"
                    "- description\n"
                    "- unit\n"
                    "- unit_rate_inr\n"
                    "- total_cost_inr\n\n"
                    "Rules:\n"
                    "- Output ONLY raw SQL (no markdown, no commentary).\n"
                    "- Use WITH requested(item_code, quantity) AS (VALUES ... ) to encode quantities from the user.\n"
                    "- LEFT JOIN csr_rates so missing item codes still appear with NULL rates/description/unit.\n"
                    "- total_cost_inr must be quantity * unit_rate_inr.\n"
                    "- Do not use any non-SELECT statements.\n"
                ),
            ),
            ("user", "{query}"),
        ]
    )

    extracted = shared_llm.invoke(sql_prompt.format_messages(query=query_text))
    sql = _extract_sql(extracted.content)

    try:
        why_unsafe = _is_safe_select_sql(sql)
        if why_unsafe:
            raise ValueError(f"Unsafe SQL rejected: {why_unsafe}")

        rows = execute_select(sql)

        calculated = []
        unresolved = []
        total = 0.0

        for r in rows:
            code = (r.get("item_code") or "").strip().upper()
            qty = float(r.get("quantity") or 0.0)
            if qty <= 0:
                continue
            unit_rate = r.get("unit_rate_inr")
            if unit_rate is None:
                unresolved.append({"item_code": code, "quantity": qty, "status": "rate not available"})
                continue

            unit_rate_f = float(unit_rate)
            line_total = float(r.get("total_cost_inr") or (unit_rate_f * qty))
            total += line_total
            calculated.append(
                {
                    "item_code": code,
                    "description": (r.get("description") or "").strip(),
                    "unit": (r.get("unit") or "").strip(),
                    "quantity": qty,
                    "unit_rate_inr": unit_rate_f,
                    "total_cost_inr": line_total,
                }
            )

        # Write temp CSV and register it for download (also deletes prior temp file for this user)
        temp_dir = os.path.join("data", "temp", "boq")
        os.makedirs(temp_dir, exist_ok=True)
        csv_name = f"boq_{normalize_user_id_for_filename(user_id)}_{secrets.token_hex(8)}.csv"
        csv_path = os.path.join(temp_dir, csv_name)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["item_code", "description", "unit", "quantity", "unit_rate_inr", "total_cost_inr"],
            )
            writer.writeheader()
            for line in calculated:
                writer.writerow(line)
            for miss in unresolved:
                writer.writerow(
                    {
                        "item_code": miss["item_code"],
                        "description": "",
                        "unit": "",
                        "quantity": miss["quantity"],
                        "unit_rate_inr": "",
                        "total_cost_inr": "",
                    }
                )

        set_latest_csv(user_id=user_id, csv_path=csv_path)

        formatted_output = (
            "BOQ total\n"
            f"**Total Calculated Cost:** INR {total:,.2f}\n\n"
            "PRELIMINARY ESTIMATE DISCLAIMER: This evaluation is generated at a preliminary design stage and carries an expected variance of plus or minus 20%."
        )

        return {"final_output": formatted_output, "boq_csv_available": True}

    except Exception as e:
        formatted_output = f"❌ BOQ SQL processing failed. Error: {str(e)}"
        return {"final_output": formatted_output, "boq_csv_available": False}


# ==========================================
# NODE 2C: SEMANTIC RFI FORMATTING ROUTE
# ==========================================
def rfi_drafting_node(state: AgentState) -> Dict[str, Any]:
    """Extracts issue parameters text before invoking structural RFI generator."""
    query_text = state["query"]
    
    extractor_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "Extract the engineering issue narrative and structural standard code violation reference.\n"
            "Output valid JSON string matching exactly this format:\n"
            "{{\"issue_description\": \"Text narrative\", \"failed_clause_reference\": \"IS Code standard ID\"}}\n"
            "Return ONLY raw JSON text."
        )),
        ("user", "{query}")
    ])
    
    messages = extractor_prompt.format_messages(query=query_text)
    extracted_res = shared_llm.invoke(messages)
    
    try:
        cleaned_json = extracted_res.content.replace("```json", "").replace("```", "").strip()
        parsed_payload = json.loads(cleaned_json)
        
        tool_res = rfi_draft.invoke(parsed_payload)
        formatted_output = tool_res.get("generated_draft_text", "Failed to compile document text.")
    except Exception as e:
        formatted_output = f"❌ Failed to structure RFI payload components semantically. Error: {str(e)}"
        
    return {"final_output": formatted_output}


# ==========================================
# NODE 2D: GENERAL CHAT ROUTE
# ==========================================
def general_chat_node(state: AgentState) -> Dict[str, Any]:
    """Executes basic polite chat processing fallback loops."""
    query_text = state["query"]
    tool_res = general_chat.invoke({"user_query": query_text})
    return {"final_output": tool_res.get("message", "")}


# ==========================================
# LANGGRAPH CONDITIONAL ROUTING EDGE
# ==========================================
def route_intent_edge(state: AgentState) -> Literal["compliance", "cost", "rfi", "chat"]:
    """Reads state tracking parameters and commands the conditional directional switch."""
    intent = state["intent"]
    if intent == "COMPLIANCE":
        return "compliance"
    elif intent == "COST":
        return "cost"
    elif intent == "RFI":
        return "rfi"
    else:
        return "chat"

# ==========================================
# GRAPH WORKFLOW COMPILATION ASSEMBLY
# ==========================================
workflow = StateGraph(AgentState)

# 1. Register processing nodes
workflow.add_node("gateway_classifier", classify_intent_node)
workflow.add_node("compliance_runner", compliance_execution_node)
workflow.add_node("cost_runner", cost_estimation_node)
workflow.add_node("rfi_runner", rfi_drafting_node)
workflow.add_node("chat_runner", general_chat_node)

# 2. Establish entrypoint
workflow.set_entry_point("gateway_classifier")

# 3. Inject structural router conditions
workflow.add_conditional_edges(
    "gateway_classifier",
    route_intent_edge,
    {
        "compliance": "compliance_runner",
        "cost": "cost_runner",
        "rfi": "rfi_runner",
        "chat": "chat_runner"
    }
)

# 4. Terminate execution threads cleanly
workflow.add_edge("compliance_runner", END)
workflow.add_edge("cost_runner", END)
workflow.add_edge("rfi_runner", END)
workflow.add_edge("chat_runner", END)

# Compile into unified execution artifact
co_pilot_agent = workflow.compile()


if __name__ == "__main__":
    print("🚀 INITIALIZING COMPREHENSIVE INTEGRATION SUITE TESTING FOR CO-PILOT AGENT...\n")
    
    test_cases = [
        {
            "name": "1. Compliance Checking Test Case",
            "query": "Can high strength deformed bars like Fe500 grade steel be used for reinforcement in structural concrete design?"
        },
        {
            "name": "2. Natural Language Cost Estimation Test Case",
            "query": "Give me an estimate for 150 cum of CONC-M25 concrete, 450 sqm of formwork via FRM-SLAB, and 6200 kg of REINF-FE500D steel rebars."
        },
        {
            "name": "3. Construction RFI Generation Test Case",
            "query": "Draft an RFI for an on-site issue where core samples reveal beam compressive strengths hit only 19 MPa at 28 days violating IS 456."
        },
        {
            "name": "4. Out-Of-Scope Conversational Safeguard Test Case",
            "query": "Can you explain the baseline chemical properties of automotive synthetic rubber tires?"
        }
    ]
    
    for case in test_cases:
        print(f"==================================================")
        print(f"RUNNING TEST: {case['name']}")
        print(f"USER SAYS: '{case['query']}'")
        print(f"==================================================")
        
        state_input = {"query": case["query"]}
        result_state = co_pilot_agent.invoke(state_input)
        
        print(f"\n[DETERMINED INTENT ROUTE]: {result_state['intent']}")
        print(f"[AGENT FINAL RESPONSE]:\n{result_state['final_output']}\n")
        if result_state.get("citations"):
            print(f"[EXTRACTED CLAUSE REFERENCE CITATIONS]: {result_state['citations']}")
        print("\n\n")
