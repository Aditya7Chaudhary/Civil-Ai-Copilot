import logging
from pydantic import BaseModel, Field
from langchain_core.tools import tool

logger = logging.getLogger("GeneralChatTool")

class GeneralChatInput(BaseModel):
    user_query: str = Field(description="The generic out-of-scope question or standard conversational text from the user.")

@tool("general_chat", args_schema=GeneralChatInput)
def general_chat(user_query: str) -> dict:
    """Handles general conversational text, pleasantries, or politely redirects queries completely out of structural scope."""
    return {
        "response_type": "OUT_OF_SCOPE_REDIRECT",
        "message": (
            "Hello! I am your specialized Civil Engineering Co-Pilot AI assistant. "
            "I am strictly authorized to verify IS/BS structural code compliance checks, process BOQ cost estimations against CSR registries, or draft construction RFIs. "
            "Your query falls outside this technical scope. Please provide a structural or civil engineering configuration to audit."
        )
    }

if __name__ == "__main__":
    print("--- Testing Standalone General Chat Tool ---")
    res = general_chat.invoke({"user_query": "What is the capital of France?"})
    print(f"Message: {res['message']}")