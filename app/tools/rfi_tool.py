import os
import logging
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("RFIDraftTool")

class RFIDraftInput(BaseModel):
    issue_description: str = Field(description="Detailed narrative of the structural design anomaly or code contradiction discovered on-site.")
    failed_clause_reference: str = Field(description="The exact engineering standard code section or ID violated (e.g., IS 456 Cl. 26.5.2.1).")

@tool("rfi_draft", args_schema=RFIDraftInput)
def rfi_draft(issue_description: str, failed_clause_reference: str) -> dict:
    """Drafts a formal, contractual Request for Information document to clarify engineering discrepancies against structural standards."""
    
    # Check for API key existence before executing inference
    if not os.getenv("GROQ_API_KEY"):
        return {"error": "GROQ_API_KEY environment variable is missing."}
        
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert Principal Project Management Consultant drafting a highly professional, contractual engineering Request for Information (RFI).\n"
            "Maintain an objective, technical tone. Do NOT manufacture or fabricate external project specifications outside what the user tells you."
        )),
        ("user", "Draft an RFI for the following structural issue:\nIssue: {issue}\nViolated Clause/Context: {clause}")
    ])
    
    messages = prompt.format_messages(issue=issue_description, clause=failed_clause_reference)
    response = llm.invoke(messages)
    
    return {
        "rfi_document_type": "Standard Engineering Request for Information (RFI)",
        "associated_reference_clause": failed_clause_reference,
        "generated_draft_text": response.content
    }

if __name__ == "__main__":
    print("--- Testing Standalone RFI Drafting Tool ---")
    res = rfi_draft.invoke({
        "issue_description": "On-site core drill samples reveal clear concrete compressive strength degradation hitting only 18 MPa at 28 days.",
        "failed_clause_reference": "IS 456 Clause 6.1.1"
    })
    print(f"Draft:\n{res.get('generated_draft_text', 'Error')[:200]}...")