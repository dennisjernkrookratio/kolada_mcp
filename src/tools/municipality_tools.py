from mcp.server.fastmcp.server import Context
from utils.context import safe_get_lifespan_context  # type: ignore[Context]
from models.types import KoladaLifespanContext, KoladaMunicipality

async def list_municipalities(
    ctx: Context,  # type: ignore[Context]
    municipality_type: str = "K"
) -> list[dict[str, str]]:
    """
    **Purpose:** Returns a list of all municipalities or regions in the system,
    filtered by type. This is useful for getting valid municipality IDs and names
    that can be used with other tools.

    **Use Cases:**
    *   "List all municipalities (kommuner) in the system."
    *   "Get the IDs and names of all regions."
    *   "What are the valid municipality IDs I can use with other tools?"

    **Arguments:**
    *   `ctx` (Context): The server context (automatically injected by the MCP framework). You do not need to provide this.
    *   `municipality_type` (str, optional): Filter municipalities by their type.
        *   "K": Kommun (Municipality, default)
        *   "R": Region
        *   "L": Landsting (County Council - older term, often equivalent to Region)
        *   "": Empty string means "no filtering" - returns all types

    **Return Value:**
    A list of dictionaries, each containing:
    *   `id` (str): The municipality ID (e.g., "0180" for Stockholm)
    *   `name` (str): The municipality name (e.g., "Stockholm")
    *   If the context is invalid, returns a list with a single error entry.

    **Important Notes:**
    *   This tool accesses the server's cache, not the live Kolada API.
    *   The returned list is sorted by municipality ID.
    """
    lifespan_ctx: KoladaLifespanContext | None = safe_get_lifespan_context(ctx)
    if not lifespan_ctx:
        return [{"error": "Server context invalid or incomplete."}]
    
    municipality_map: dict[str, KoladaMunicipality] = lifespan_ctx.get("municipality_map", {})
    result = []
    for m_id, muni in municipality_map.items():
        # If municipality_type is provided (default "K"), match only those; allow empty string to mean "no filtering"
        if municipality_type and muni.get("type") != municipality_type:
            continue
        result.append({"id": m_id, "name": muni.get("title", f"Municipality {m_id}")})
    
    # Sort by municipality ID
    result.sort(key=lambda x: x["id"])
    return result
