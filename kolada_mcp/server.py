import sys
import traceback
from mcp.server.fastmcp import FastMCP

from kolada_mcp.lifespan.context import app_lifespan
from kolada_mcp.tools.metadata_tools import list_operating_areas, get_kpis_by_operating_area, get_kpi_metadata, search_kpis
from kolada_mcp.tools.data_tools import fetch_kolada_data, analyze_kpi_across_municipalities
from kolada_mcp.tools.comparison_tools import compare_kpis
from kolada_mcp.prompts.entry_prompt import kolada_entry_point

# Instantiate FastMCP
mcp: FastMCP = FastMCP("KoladaServer", lifespan=app_lifespan)

# Register all tool functions
mcp.tool()(list_operating_areas)
mcp.tool()(get_kpis_by_operating_area)
mcp.tool()(get_kpi_metadata)
mcp.tool()(search_kpis)
mcp.tool()(fetch_kolada_data)
mcp.tool()(analyze_kpi_across_municipalities)
mcp.tool()(compare_kpis)

# Register the prompt
mcp.prompt()(kolada_entry_point)

if __name__ == "__main__":
    print("[Kolada MCP Main] Script starting...", file=sys.stderr)
    try:
        print(
            "[Kolada MCP Main] Calling mcp.run(transport='stdio')...", file=sys.stderr
        )
        mcp.run(transport="stdio")
        print("[Kolada MCP Main] mcp.run() finished unexpectedly.", file=sys.stderr)
    except Exception as e:
        print(
            f"[Kolada MCP Main] EXCEPTION caught around mcp.run(): {e}", file=sys.stderr
        )
        traceback.print_exc(file=sys.stderr)
    finally:
        print("[Kolada MCP Main] Script exiting.", file=sys.stderr)
