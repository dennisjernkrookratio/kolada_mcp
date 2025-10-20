"""
Test script to verify search functionality works correctly
"""
import asyncio
import sys
from kolada_mcp.mcp_server import create_server

async def test_search():
    """Test the search function with timeseries commands"""
    
    server = create_server()
    
    # Get the search tool
    search_tool = None
    for tool in server._tools.values():
        if tool.name == "search":
            search_tool = tool
            break
    
    if not search_tool:
        print("ERROR: search tool not found!")
        return
    
    # Create a mock context
    class MockContext:
        def __init__(self):
            self.app_context = {
                "kpi_map": {},
                "municipality_map": {},
                "kpi_embeddings": None,
                "kpi_ids": []
            }
    
    ctx = MockContext()
    
    # Test 1: Single KPI timeseries
    print("\n=== Test 1: ts:kpi=N15504;muni=0188;years=2023 ===")
    result1 = await search_tool.fn("ts:kpi=N15504;muni=0188;years=2023", ctx)
    print(f"Result type: {type(result1)}")
    print(f"Result keys: {result1.keys() if isinstance(result1, dict) else 'N/A'}")
    if "results" in result1:
        print(f"Number of results: {len(result1['results'])}")
        if result1['results']:
            first_result = result1['results'][0]
            print(f"First result keys: {first_result.keys()}")
            print(f"Has 'id': {'id' in first_result}")
            print(f"Has 'title': {'title' in first_result}")
            print(f"Has 'text': {'text' in first_result}")
            print(f"Has 'url': {'url' in first_result}")
            print(f"Has 'rows': {'rows' in first_result}")
            if 'rows' in first_result:
                print(f"Number of rows: {len(first_result['rows'])}")
                if first_result['rows']:
                    print(f"First row: {first_result['rows'][0]}")
    else:
        print(f"Full result: {result1}")
    
    # Test 2: Multi-KPI timeseries
    print("\n\n=== Test 2: timeseries: muni=0188; years=2023; kpis=N15504 ===")
    result2 = await search_tool.fn("timeseries: muni=0188; years=2023; kpis=N15504", ctx)
    print(f"Result type: {type(result2)}")
    if "results" in result2:
        print(f"Number of results: {len(result2['results'])}")
        if result2['results']:
            first_result = result2['results'][0]
            print(f"First result has correct format: {all(k in first_result for k in ['id', 'title', 'text', 'url'])}")
    
    # Test 3: Verify the exact JSON structure
    print("\n\n=== Test 3: Full JSON output ===")
    import json
    print(json.dumps(result1, indent=2, default=str))

if __name__ == "__main__":
    asyncio.run(test_search())
