import logging
from app.tools.compliance_tool import ComplianceEngine

logging.basicConfig(level=logging.INFO)

print("Running isolated compliance test...")
try:
    engine = ComplianceEngine()
    test_query = "What is the maximum slenderness ratio for a tension member under IS 800?"
    result = engine.evaluate_compliance(test_query)

    print("\n--- Output ---")
    print(f"Status: {result.get('compliance_status')}")
    print(f"Answer length: {len(result.get('answer', ''))}")
    print(f"Citations: {len(result.get('applied_citations', []))}")
    print(f"Sources: {result.get('source_documents')}")
    print("\nAnswer:")
    print(result.get("answer"))
    print("---")

except Exception as e:
    print(f"Error: {e}")
