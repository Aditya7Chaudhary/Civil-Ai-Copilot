import os
import csv
import logging
from typing import Dict, List, Any
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("CostEstimationTool")

class CostEstimateInput(BaseModel):
    items: List[Dict[str, Any]] = Field(
        description="A list of dictionaries containing extracted BOQ items. Each dict must have 'item_code' (e.g., M-25, ST-HIGH) and 'quantity' (float)."
    )

def lookup_csr_rate(item_code: str) -> Dict[str, Any]:
    """Helper to fetch a rate from the local CSR CSV registry."""
    csv_path = os.path.join("data", "csr", "csr_rates.csv")
    if not os.path.exists(csv_path):
        return {"error": "CSR registry database file not found."}
        
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["item_code"].strip().upper() == item_code.strip().upper():
                return {
                    "description": row["description"],
                    "unit": row["unit"],
                    "rate": float(row["rate_inr"])
                }
    return {"error": f"Item code '{item_code}' not found in CSR database."}

@tool("cost_estimate", args_schema=CostEstimateInput)
def cost_estimate(items: List[Dict[str, Any]]) -> dict:
    """Calculates preliminary structural cost estimates matching BOQ items against the standard CSR database."""
    total_project_cost = 0.0
    line_items_calculated = []
    missing_items = []
    
    for boq_item in items:
        code = boq_item.get("item_code", "").upper()
        qty = float(boq_item.get("quantity", 0.0))
        
        csr_data = lookup_csr_rate(code)
        
        if "error" in csr_data:
            # Enforce guardrail: State 'rate not available' rather than guessing rates
            missing_items.append({"item_code": code, "quantity": qty, "status": "rate not available"})
            continue
            
        line_cost = csr_data["rate"] * qty
        total_project_cost += line_cost
        
        line_items_calculated.append({
            "item_code": code,
            "description": csr_data["description"],
            "unit": csr_data["unit"],
            "quantity": qty,
            "unit_rate_inr": csr_data["rate"],
            "total_cost_inr": line_cost
        })
        
    return {
        "disclaimer": "PRELIMINARY ESTIMATE DISCLAIMER: This evaluation is generated at a preliminary design stage and carries an expected variance of ±20%.",
        "csr_edition_used": "CPWD Schedule of Rates (CSR) 2024 Reference Edition",
        "total_estimated_cost_inr": total_project_cost,
        "calculated_line_items": line_items_calculated,
        "unresolved_items": missing_items
    }

if __name__ == "__main__":
    print("--- Testing Standalone Cost Estimation Tool ---")
    mock_payload = [
        {"item_code": "M-25", "quantity": 150.0},
        {"item_code": "UNKNOWN-99", "quantity": 5.0}
    ]
    res = cost_estimate.invoke({"items": mock_payload})
    print(f"Total: INR {res['total_estimated_cost_inr']}")
    print(f"Unresolved: {res['unresolved_items']}")