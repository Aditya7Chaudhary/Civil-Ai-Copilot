import os
import sys

# 1. Dynamically locate the parent directory (project root) and append it to sys.path
# This ensures Python can see the 'app' directory from inside the 'test' folder
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 2. Now you can safely import your modules
import logging
from app.tools.compliance_tool import ComplianceEngine

logging.basicConfig(level=logging.INFO)

print("🔍 Initializing Compliance Engine Standalone Test inside Test Folder...")
try:
    engine = ComplianceEngine()
    
    # Change this query to test different clauses (e.g., IS 456 spacing, IS 800 slenderness, etc.)
    test_query = "What is the maximum slenderness ratio for a tension member under IS 800?"
    print(f"📡 Sending test query to compliance engine: '{test_query}'")
    
    result = engine.evaluate_compliance(test_query)
    
    print("\n📦 Raw Tool Output Captured:")
    print("---------------------------------")
    print(f"Compliance Report Length: {len(result.get('compliance_report', ''))} characters")
    print(f"Citations: {result.get('citations')}")
    print(f"Sources: {result.get('sources')}")
    print("\nFull Report text:")
    print(result.get('compliance_report'))
    print("---------------------------------")

except Exception as e:
    print(f"❌ Critical Failure during compliance execution: {str(e)}")
    import traceback
    traceback.print_exc()