import sys
from typing import cast
from mcp.server.fastmcp.server import Context
from kolada_mcp.models.types import KoladaLifespanContext

def safe_get_lifespan_context(
    ctx: Context,
) -> KoladaLifespanContext | None:
    """
    Safely retrieves the KoladaLifespanContext from the standard context structure.
    Returns None if the context is invalid or incomplete.
    """
    if (
        not ctx
        or not hasattr(ctx, "request_context")
        or not ctx.request_context
        or not hasattr(ctx.request_context, "lifespan_context")
        or not ctx.request_context.lifespan_context
    ):
        print("[Kolada MCP] Invalid or incomplete context structure.", file=sys.stderr)
        return None
    return cast(KoladaLifespanContext, ctx.request_context.lifespan_context)
