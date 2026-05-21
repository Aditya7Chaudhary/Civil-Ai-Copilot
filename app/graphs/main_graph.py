import os
import json
import logging
from typing import TypedDict, List, Dict, Any, Literal
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Import all individual functional components built in earlier stages
from app.tools.compliance_tool import ComplianceEngine
from app.tools.cost_tool import cost_estimate
from app.tools.rfi_tool import rfi_draft
from app.tools.chat_tool import general_chat

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MainOrchestrator")

# 1. Define the Unified State Workspace
class AgentState(TypedDict):
    query: str
    intent: str  # 'COMPLIANCE' | 'COST' | 'RFI' | 'CHAT'
    final_output: str
    citations: List[str]
    sources: List[str]

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

    # Ensure these keys EXACTLY match the fields defined in your AgentState schema
    return {
        "final_output": assessment.get("compliance_report", ""),
        "citations": assessment.get("citations", []),
        "sources": assessment.get("sources", [])
    }

# ==========================================
# NODE 2B: SEMANTIC BOQ COST ESTIMATION ROUTE
# ==========================================
def cost_estimation_node(state: AgentState) -> Dict[str, Any]:
    """Uses LLM to structure text inputs before querying the local CSR database."""
    query_text = state["query"]
    
    # Semantic extraction prompt to transform natural text into schema JSON structures
    extractor_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "Extract raw material items and quantities from the engineering text description.\n"
            "Map item names to exact allowed item codes in the database registry:\n"
            "- EXC-SOFT, EXC-HARD, EXC-ROCK, BACKFILL-SOIL\n"
            "- CONC-M10, CONC-M15, CONC-M20, CONC-M25, CONC-M30, CONC-M35, CONC-M40, CONC-M45, CONC-M50\n"
            "- FRM-FOUND, FRM-COL-RECT, FRM-COL-CIRC, FRM-BEAM, FRM-SLAB, FRM-WALL, FRM-STAIR\n"
            "- REINF-FE250, REINF-FE415, REINF-FE500, REINF-FE500D, REINF-FE550, REINF-FE600\n"
            "- PSC-M45, PSC-M50, PSC-M60, PSC-STRAND-127, PSC-ANCHOR-POST, PSC-DUCT-HDPE\n"
            "- ST-ROLLED-BEAM, ST-ROLLED-CHAN, ST-ROLLED-ANG, ST-BUILTUP-GIRD, ST-BOLT-HSFG, ST-WELD-BUTT\n\n"
            "Output valid JSON matching this schema: {{\"items\": [{{\"item_code\": \"CODE\", \"quantity\": 0.0}}]}}. "
            "Output ONLY raw JSON. No markdown enclosures."
        )),
        ("user", "{query}")
    ])
    
    messages = extractor_prompt.format_messages(query=query_text)
    extracted_res = shared_llm.invoke(messages)
    
    try:
        # Clean potential markdown backticks wrapped around raw structural text streaming responses
        cleaned_json = extracted_res.content.replace("```json", "").replace("```", "").strip()
        parsed_payload = json.loads(cleaned_json)
        
        # Invoke our functional Python tool over the parsed list payload
        tool_res = cost_estimate.invoke(parsed_payload)
        
        # Format a clean string response for Teams UI display
        formatted_output = (
            f"### 📊 Structural Cost Estimation Report\n"
            f"**Registry Base:** {tool_res['csr_edition_used']}\n"
            f"**Total Calculated Cost:** INR {tool_res['total_estimated_cost_inr']:,.2f}\n\n"
            f"#### Line Item Breakdown:\n"
        )
        for line in tool_res["calculated_line_items"]:
            formatted_output += f"- **{line['item_code']}**: {line['quantity']} {line['unit']} @ ₹{line['unit_rate_inr']}/unit → ₹{line['total_cost_inr']:,.2f}\n"
            
        if tool_res["unresolved_items"]:
            formatted_output += "\n⚠️ **Unresolved Items (Missing CSR Rates):**\n"
            for broken in tool_res["unresolved_items"]:
                formatted_output += f"- Code: {broken['item_code']} (Qty: {broken['quantity']})\n"
                
        formatted_output += f"\n*{tool_res['disclaimer']}*"
        
    except Exception as e:
        formatted_output = f"❌ Failed to parse itemization arrays semantically. Error: {str(e)}"
        
    return {"final_output": formatted_output}


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