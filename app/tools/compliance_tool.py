import os
import logging
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from app.tools.retrieval_tool import get_engineering_retriever
from app.tools.chunking import detect_standards_from_query

# Load environment variables from .env file explicitly
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ComplianceEngine")

class ComplianceEngine:
    def __init__(self):
        self.retriever = get_engineering_retriever()
        
        # Verify API Key existence before boot
        if not os.getenv("GROQ_API_KEY"):
            raise ValueError("CRITICAL: GROQ_API_KEY not found in environment or .env file.")

        # Initialize Groq Engine
        # Using Llama-3.3-70b-versatile for excellent engineering constraint validation
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.0,  # Strict adherence to retrieved context, no creativity
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        
        self.system_prompt = (
            "You are a rigid structural engineering compliance auditor specializing in Indian Standard Codes (IS Codes).\n"
            "Your core directive is to verify query configurations against the provided legal engineering clauses.\n\n"
            "CRITICAL OPERATING RULES:\n"
            "1. ONLY evaluate constraints using the provided text blocks. Do NOT use your own external background training data.\n"
            "2. If the retrieved evidence does not contain explicit parameters to answer the question, state exactly: "
            "'I could not find sufficient evidence within the retrieved standards to verify compliance.'\n"
            "3. Never guess, approximate, or extrapolate engineering limits. If a metric isn't directly mentioned, it doesn't exist.\n"
            "4. For every rule stated, format an absolute bold text inline citation to its matching Clause Reference ID.\n\n"
            "Retrieved Structural Reference Context:\n"
            "{context}\n"
        )
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", "Engineering Query: {query}")
        ])

    def _format_context_block(self, clauses: list) -> str:
        """Converts raw clause JSON metadata dictionaries into clean text blocks for the LLM prompt context."""
        formatted = []
        for c in clauses:
            formatted.append(
                f"--- START CLAUSE {c['clause_id']}: {c['title']} ---\n"
                f"{c['text']}\n"
                f"--- END CLAUSE {c['clause_id']} ---"
            )
        return "\n\n".join(formatted)

    def evaluate_compliance(self, query: str) -> dict:
        """Retrieves verified clause blocks and runs them through the strict verification prompt."""
        # 1. Execute Modulated Structural Query Search
        target_standards = detect_standards_from_query(query)
        retrieve_k = min(16, 10 + 3 * len(target_standards))
        evidence = self.retriever.retrieve_evidence(query, k=retrieve_k)
        
        # Short-circuit immediately if the retriever returns zero matches
        if not evidence["retrieved_clauses"]:
            return {
                "compliance_status": "UNKNOWN",
                "answer": "I could not find sufficient evidence within the retrieved standards to verify compliance.",
                "applied_citations": [],
                "source_documents": []
            }
            
        # 2. Build LLM Context Window Fields
        context_string = self._format_context_block(evidence["retrieved_clauses"])
        messages = self.prompt_template.format_messages(
            context=context_string,
            query=query
        )
        
        # 3. Generate Evaluation Answer Text via Groq
        response = self.llm.invoke(messages)
        answer_text = response.content
        
        # 4. Determine Compliance Verdict Status
        answer_lower = answer_text.lower()
        no_evidence_phrase = "could not find sufficient evidence"
        if no_evidence_phrase in answer_lower:
            status = "INSUFFICIENT_DATA"
        else:
            status = "COMPLIANT_DETERMINED"
            
        return {
            "compliance_status": status,
            "answer": answer_text,
            "applied_citations": evidence["citations"],
            "source_documents": evidence["sources"]
        }

if __name__ == "__main__":
    engine = ComplianceEngine()
    
    test_query = "Can high strength deformed bars like Fe500 grade steel be used for reinforcement in structural concrete design?"
    
    print(f"Executing Diagnostic Compliance Assessment via Groq for: '{test_query}'")
    result = engine.evaluate_compliance(test_query)
    ...
    
    print("\n--- EVALUATION VERDICT ---")
    print(f"Status: {result['compliance_status']}")
    print(f"Answer:\n{result['answer']}")
    print(f"Citations: {result['applied_citations']}")
    print(f"Documents: {result['source_documents']}")