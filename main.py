#!/usr/bin/env python3
import sys
import traceback

from kolada_mcp import mcp

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
