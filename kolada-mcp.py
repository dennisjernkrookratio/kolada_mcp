import json
import sys
import traceback
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional, TypedDict, Union

import httpx

# Use the base Context provided by the framework for type hinting in tools
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.server import (
    Context as FastMCPContext,
)  # <--- Use this base context in tool signatures
from mcp.shared.context import RequestContext as BaseRequestContext
from pydantic import ConfigDict

###############################################################################
# 1) Typed structures for your Kolada data (Unchanged)
###############################################################################


class KoladaKpi(TypedDict, total=False):
    """
    Represents a single Key Performance Indicator (KPI) from Kolada.
    Kolada (Kommun- och landstingsdatabasen) provides ~6,500 KPIs
    covering Swedish municipalities and regions across various sectors
    like economy, schools, healthcare, environment, etc.
    """

    id: str  # The unique identifier for the KPI (e.g., "N00945")
    title: str  # The human-readable name of the KPI (e.g., "Population size")
    description: str  # A longer explanation of the KPI.
    operating_area: str  # The thematic category/categories (e.g., "Demographics", "Economy,Environment")


class KoladaLifespanContext(TypedDict):
    """
    Data cached in the server's memory at startup ('lifespan_context').
    This avoids repeatedly fetching static metadata from the Kolada API.
    """

    kpi_cache: list[KoladaKpi]  # A list of all KPI metadata objects.
    kpi_map: dict[
        str, KoladaKpi
    ]  # A mapping from KPI ID to the KPI object for quick lookups.
    operating_areas_summary: list[
        dict[str, Union[str, int]]
    ]  # Summary of areas and KPI counts.


###############################################################################
# 2) Define a typed request context (Unchanged)
#    This still defines the STRUCTURE of the lifespan_context part
###############################################################################


@dataclass
class KoladaRequestContext(BaseRequestContext[Any, KoladaLifespanContext]):
    """
    Typed request context. Defines that lifespan_context should hold
    KoladaLifespanContext data.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)


# --- KoladaContext Definition is no longer needed for tool signatures ---
# Using the base FastMCPContext in signatures is preferred for compatibility.
# Leaving this definition doesn't hurt if used elsewhere, but commenting out
# for clarity as it's not used in the tool signatures below.
# @dataclass
# class KoladaContext(FastMCPContext[Any, KoladaLifespanContext]):
#     """
#     Custom typed context class for Kolada tools. Inherits from FastMCPContext.
#     Provides convenient access to request_context which holds lifespan data.
#     """
#     model_config = ConfigDict(arbitrary_types_allowed=True)


###############################################################################
# 3) Server Lifespan: Fetch & Cache Kolada Metadata (Unchanged)
#    This function yields the KoladaLifespanContext dictionary, which populates
#    ctx.request_context.lifespan_context in the tools.
###############################################################################
BASE_URL = "https://api.kolada.se/v2"
KPI_PER_PAGE = 5000

# --- Helper functions (Unchanged) ---


def _group_kpis_by_operating_area(
    kpis: list[KoladaKpi],
) -> dict[str, list[KoladaKpi]]:
    """Groups KPIs by their 'operating_area' field."""
    grouped: dict[str, list[KoladaKpi]] = {}
    for kpi in kpis:
        operating_area_field = kpi.get("operating_area") or "Unknown"
        areas = [a.strip() for a in operating_area_field.split(",")]
        for area in areas:
            if area:
                grouped.setdefault(area, []).append(kpi)
    return grouped


def _get_operating_areas_summary(
    kpis: list[KoladaKpi],
) -> list[dict[str, Union[str, int]]]:
    """Generates a summary list of operating areas and their KPI counts."""
    grouped = _group_kpis_by_operating_area(kpis)
    areas_with_counts: list[dict[str, Union[str, int]]] = []
    for area in sorted(grouped.keys()):
        areas_with_counts.append(
            {"operating_area": area, "kpi_count": len(grouped[area])}
        )
    return areas_with_counts


# --- Lifespan Function (Unchanged, includes stderr prints) ---


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[KoladaLifespanContext]:
    """
    Initializes the Kolada MCP Server at startup. Includes stderr logging.
    Yields the dictionary that becomes ctx.request_context.lifespan_context.
    """
    print("[Kolada MCP Lifespan] Starting lifespan setup...", file=sys.stderr)
    kpi_list: list[KoladaKpi] = []
    print(
        "[Kolada MCP] Initializing: Fetching all KPI metadata from Kolada API...",
        file=sys.stderr,
    )
    async with httpx.AsyncClient() as client:
        next_url = f"{BASE_URL}/kpi?per_page={KPI_PER_PAGE}"
        while next_url:
            print(f"[Kolada MCP] Fetching page: {next_url}", file=sys.stderr)
            try:
                resp = await client.get(next_url, timeout=180.0)
                resp.raise_for_status()
                data = resp.json()
            except (
                httpx.RequestError,
                httpx.HTTPStatusError,
                json.JSONDecodeError,
            ) as e:
                print(
                    f"[Kolada MCP] CRITICAL ERROR fetching Kolada KPIs: {e}",
                    file=sys.stderr,
                )
                print(f"Failed URL: {next_url}", file=sys.stderr)
                raise RuntimeError(f"Failed to initialize Kolada KPI cache: {e}") from e

            values: list[KoladaKpi] | None = data.get("values")
            if isinstance(values, list):
                kpi_list.extend(values)
            next_url = data.get("next_page")

    print(
        f"[Kolada MCP] Fetched {len(kpi_list)} total KPIs from Kolada.", file=sys.stderr
    )

    # Build an ID -> KPI map
    kpi_map: dict[str, KoladaKpi] = {}
    for kpi_obj in kpi_list:
        k_id = kpi_obj.get("id")
        if k_id is not None:
            kpi_map[str(k_id)] = kpi_obj

    # Generate operating areas summary
    operating_areas_summary = _get_operating_areas_summary(kpi_list)
    print(
        f"[Kolada MCP] Identified {len(operating_areas_summary)} unique operating areas.",
        file=sys.stderr,
    )

    # This dictionary's structure matches KoladaLifespanContext
    context_data: KoladaLifespanContext = {
        "kpi_cache": kpi_list,
        "kpi_map": kpi_map,
        "operating_areas_summary": operating_areas_summary,
    }

    print("[Kolada MCP] Initialization complete. KPI metadata cached.", file=sys.stderr)
    print(
        f"[Kolada MCP Lifespan] Yielding context with {len(kpi_list)} KPIs (type KoladaLifespanContext)...",
        file=sys.stderr,
    )
    try:
        # Yield the data structure
        yield context_data
        print(
            "[Kolada MCP Lifespan] Post-yield (server shutting down)...",
            file=sys.stderr,
        )
    except Exception as e:
        print(
            f"[Kolada MCP Lifespan] Exception DURING yield/server run?: {e}",
            file=sys.stderr,
        )
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        print(
            "[Kolada MCP Lifespan] Entering finally block (shutdown).", file=sys.stderr
        )
        print("[Kolada MCP] Shutting down.", file=sys.stderr)
        pass


# --- Instantiate FastMCP (Unchanged) ---
# The lifespan function dictates the structure of lifespan_context
mcp = FastMCP("KoladaServer", lifespan=app_lifespan)

###############################################################################
# 4) Tools - Using Standard FastMCPContext Signature
###############################################################################


@mcp.tool()
async def list_operating_areas(
    ctx: FastMCPContext,  # <-- MODIFIED: Use standard framework context type
) -> list[dict[str, Union[str, int]]]:
    """
    **Step 1: Discover KPI Categories.**
    Retrieves a summary of all available 'operating areas' (thematic categories)
    for Kolada KPIs, along with the number of KPIs in each area.
    Use this tool first to understand the available categories before filtering KPIs.
    The data is sourced from the server's cache, populated at startup.

    Example Output:
    [
        {'operating_area': 'Demographics', 'kpi_count': 50},
        {'operating_area': 'Economy', 'kpi_count': 120},
        ...
    ]
    """
    # Defensive check for standard context structure
    # Check attributes exist before accessing to prevent AttributeErrors
    if (
        not ctx
        or not hasattr(ctx, "request_context")
        or not ctx.request_context
        or not hasattr(ctx.request_context, "lifespan_context")
        or not ctx.request_context.lifespan_context
    ):
        print(
            "Error: Invalid context structure received in list_operating_areas tool.",
            file=sys.stderr,
        )
        return [{"error": "Server context structure invalid or incomplete."}]

    # Access the lifespan data populated by the app_lifespan function
    # The actual object here IS the KoladaLifespanContext dictionary yielded above
    lifespan_ctx = ctx.request_context.lifespan_context

    # Use .get() for safe access to dictionary keys
    summary = lifespan_ctx.get("operating_areas_summary", [])
    if not summary:
        print("Warning: Operating areas summary is empty in context.", file=sys.stderr)
        return []
    return summary


@mcp.tool()
async def get_kpis_by_operating_area(
    operating_area: str,
    ctx: FastMCPContext,  # <-- MODIFIED: Use standard framework context type
) -> list[KoladaKpi]:
    """
    **Step 2: Filter KPIs by Category.**
    Retrieves a list of Kolada KPIs that belong to the specified 'operating_area'.
    Use this tool *after* identifying a relevant area using `list_operating_areas`.
    Provide the exact operating area name obtained from `list_operating_areas`.
    The data is sourced from the server's cache. Note that some KPIs might belong
    to multiple areas; this tool checks if the *specified* area is associated
    with the KPI.

    Args:
        operating_area: The exact name of the operating area to filter by.
        ctx: The server context (injected automatically).

    Returns:
        A list of KoladaKpi objects matching the area, or an empty list.
    """
    # Defensive check for standard context structure
    if (
        not ctx
        or not hasattr(ctx, "request_context")
        or not ctx.request_context
        or not hasattr(ctx.request_context, "lifespan_context")
        or not ctx.request_context.lifespan_context
    ):
        print(
            "Error: Invalid context structure received in get_kpis_by_operating_area tool.",
            file=sys.stderr,
        )
        return []  # Return empty list on context error

    lifespan_ctx = ctx.request_context.lifespan_context  # This is KoladaLifespanContext
    kpi_list = lifespan_ctx.get("kpi_cache", [])
    if not kpi_list:
        print("Warning: KPI cache is empty in context.", file=sys.stderr)
        return []

    target_area_lower = operating_area.lower().strip()
    matches: list[KoladaKpi] = []

    for kpi in kpi_list:
        area_field = kpi.get("operating_area", "").lower()
        kpi_areas = {a.strip() for a in area_field.split(",")}
        if target_area_lower in kpi_areas:
            matches.append(kpi)

    if not matches:
        print(
            f"Info: No KPIs found for operating area '{operating_area}'.",
            file=sys.stderr,
        )

    return matches


###############################################################################
# 5) Existing Tools - Using Standard FastMCPContext Signature
###############################################################################


@mcp.tool()
async def get_kpi_metadata(
    kpi_id: str,
    ctx: FastMCPContext,  # <-- MODIFIED: Use standard framework context type
) -> Union[KoladaKpi, dict[str, str]]:
    """
    Retrieves the cached metadata (title, description, operating area) for a
    *specific* Kolada KPI using its unique ID (e.g., "N00945").
    Kolada KPIs (Key Performance Indicators) represent various metrics for
    Swedish municipalities and regions.
    Use this tool when you have identified a specific KPI ID (e.g., from
    `get_kpis_by_operating_area` or `search_kpis`) and need its details.
    This tool accesses the server's cache, not the live Kolada API.

    Args:
        kpi_id: The unique identifier of the KPI.
        ctx: The server context (injected automatically).

    Returns:
        A KoladaKpi object if found, or an error dictionary.
    """
    # Defensive check for standard context structure
    if (
        not ctx
        or not hasattr(ctx, "request_context")
        or not ctx.request_context
        or not hasattr(ctx.request_context, "lifespan_context")
        or not ctx.request_context.lifespan_context
    ):
        print(
            "Error: Invalid context structure received in get_kpi_metadata tool.",
            file=sys.stderr,
        )
        return {"error": "Server context structure invalid or incomplete."}

    lifespan_ctx = ctx.request_context.lifespan_context  # This is KoladaLifespanContext
    kpi_map = lifespan_ctx.get("kpi_map", {})

    kpi_obj = kpi_map.get(str(kpi_id))

    if not kpi_obj:
        print(
            f"Info: KPI metadata request failed for ID '{kpi_id}'. Not found in cache.",
            file=sys.stderr,
        )
        return {"error": f"No KPI metadata found in cache for ID: {kpi_id}"}
    return kpi_obj


@mcp.tool()
async def search_kpis(
    keyword: str,
    ctx: FastMCPContext,  # <-- MODIFIED: Use standard framework context type
    limit: int = 20,
) -> list[KoladaKpi]:
    """
    Performs a simple keyword search within the *cached* titles and descriptions
    of all Kolada KPIs. This is useful for finding relevant KPIs when you don't
    know the exact ID or operating area.
    Returns a limited list of matching KPIs. Search is case-insensitive.

    Args:
        keyword: The term to search for in KPI titles and descriptions.
        limit: The maximum number of matching KPIs to return (default 20).
        ctx: The server context (injected automatically).

    Returns:
        A list of matching KoladaKpi objects (up to the limit).
    """
    # Defensive check for standard context structure
    if (
        not ctx
        or not hasattr(ctx, "request_context")
        or not ctx.request_context
        or not hasattr(ctx.request_context, "lifespan_context")
        or not ctx.request_context.lifespan_context
    ):
        print(
            "Error: Invalid context structure received in search_kpis tool.",
            file=sys.stderr,
        )
        return []  # Return empty list on context error

    lifespan_ctx = ctx.request_context.lifespan_context  # This is KoladaLifespanContext
    kpi_list = lifespan_ctx.get("kpi_cache", [])
    if not kpi_list:
        print("Warning: KPI cache is empty, cannot perform search.", file=sys.stderr)
        return []

    kw_lower = keyword.lower()
    matches: list[KoladaKpi] = []
    count = 0
    for kpi_obj in kpi_list:
        title = kpi_obj.get("title", "").lower()
        desc = kpi_obj.get("description", "").lower()
        if kw_lower in title or kw_lower in desc:
            matches.append(kpi_obj)
            count += 1
            if count >= limit:
                break
    if not matches:
        print(
            f"Info: Keyword search for '{keyword}' yielded no results.", file=sys.stderr
        )

    return matches


@mcp.tool()
async def fetch_kolada_data(
    kpi_id: str,
    municipality_id: str,
    ctx: FastMCPContext,  # <-- MODIFIED: Use standard framework context type (for consistency)
    year: Optional[str] = None,
) -> dict[str, Any]:
    """
    Fetches the *actual statistical data values* for a specific Kolada KPI
    and a specific Swedish municipality (identified by its ID, e.g., "1860" for Åsele).
    Optionally, you can specify a year or range of years.
    This tool calls the *live* Kolada API endpoint `/v2/data/kpi/.../municipality/...`.
    Use this tool *after* you have identified the specific KPI ID and municipality ID you need data for.

    Args:
        kpi_id: The unique ID of the Kolada KPI.
        municipality_id: The official ID of the Swedish municipality (see Kolada docs or `/v2/municipality` endpoint).
        year: Optional. A specific year (e.g., "2023") or a comma-separated list/range (e.g., "2020,2021,2022"). If None, fetches all available years.
        ctx: The server context (injected automatically, currently unused but available).

    Returns:
        A dictionary containing the response from the Kolada API, which typically includes
        'count', 'values' (list of data points per year/gender), etc., or an error dictionary
        if the API call fails.
    """
    # Context (ctx) is available if needed in the future, e.g., for logging or accessing lifespan data
    # if ctx and ctx.request_context:
    #     pass

    if not kpi_id or not municipality_id:
        return {"error": "kpi_id and municipality_id are required."}

    url = f"{BASE_URL}/data/kpi/{kpi_id}/municipality/{municipality_id}"
    params: dict[str, str] = {}
    if year:
        params["year"] = year

    async with httpx.AsyncClient() as client:
        try:
            print(
                f"Fetching Kolada data from: {url} with params: {params}",
                file=sys.stderr,
            )
            resp = await client.get(url, params=params, timeout=60.0)
            resp.raise_for_status()
            return resp.json()
        except httpx.RequestError as ex:
            error_msg = (
                f"Network error accessing Kolada API: {ex.__class__.__name__} - {ex}"
            )
            print(f"[Kolada MCP] {error_msg}", file=sys.stderr)
            return {
                "error": error_msg,
                "details": str(ex),
                "endpoint": url,
                "params": params,
            }
        except httpx.HTTPStatusError as ex:
            status_code = ex.response.status_code
            error_msg = (
                f"Kolada API returned error status {status_code} for {ex.request.url}"
            )
            print(f"[Kolada MCP] {error_msg}", file=sys.stderr)
            try:
                error_details = ex.response.json()
            except json.JSONDecodeError:
                error_details = ex.response.text
            if status_code == 404:
                error_msg = f"Kolada API Error: Not Found (404). Check if KPI ID '{kpi_id}' and Municipality ID '{municipality_id}' are valid for the given year(s) '{year or 'all'}'. Also check the API endpoint path."
            elif status_code >= 500:
                error_msg = f"Kolada API Error: Server Error ({status_code}). The Kolada service might be temporarily unavailable."
            else:
                error_msg = f"Kolada API Error: Client Error ({status_code}). There might be an issue with the request parameters."
            return {
                "error": error_msg,
                "status_code": status_code,
                "details": error_details,
                "endpoint": url,
                "params": params,
            }
        except json.JSONDecodeError as ex:
            error_msg = f"Failed to decode JSON response from Kolada API."
            print(f"[Kolada MCP] {error_msg}: {ex}", file=sys.stderr)
            return {
                "error": error_msg,
                "details": str(ex),
                "endpoint": url,
                "params": params,
            }
        except Exception as ex:
            error_msg = f"An unexpected error occurred during Kolada data fetch: {ex.__class__.__name__}"
            print(f"[Kolada MCP] {error_msg}: {ex}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return {
                "error": error_msg,
                "details": str(ex),
                "endpoint": url,
                "params": params,
            }


@mcp.tool()
async def get_kpi_list(
    ctx: FastMCPContext,  # <-- MODIFIED: Use standard framework context type
) -> list[KoladaKpi]:
    """
    (Potentially less useful with area filtering)
    Retrieves the *entire cached list* of all Kolada KPI metadata objects (~6500 items).
    Consider using `list_operating_areas` and `get_kpis_by_operating_area` or
    `search_kpis` for more targeted discovery instead of loading the full list.

    Returns:
        The full list of KoladaKpi objects from the server cache.
    """
    # Defensive check for standard context structure
    if (
        not ctx
        or not hasattr(ctx, "request_context")
        or not ctx.request_context
        or not hasattr(ctx.request_context, "lifespan_context")
        or not ctx.request_context.lifespan_context
    ):
        print(
            "Error: Invalid context structure received in get_kpi_list tool.",
            file=sys.stderr,
        )
        return []  # Return empty list on context error

    lifespan_ctx = ctx.request_context.lifespan_context  # This is KoladaLifespanContext
    kpi_list = lifespan_ctx.get("kpi_cache", [])
    if not kpi_list:
        print("Warning: KPI cache is empty when calling get_kpi_list.", file=sys.stderr)
    return kpi_list


###############################################################################
# 6) Prompt examples (Unchanged)
###############################################################################


@mcp.prompt()
def compare_multiple_municipalities(
    kpi_id: str, municipality_ids: list[str], year: Optional[str] = None
) -> str:
    """
    Generates a prompt instructing an LLM to compare a specific Kolada KPI
    across multiple municipalities using the available tools.
    """
    muns_list = "\n".join([f"  - {mid}" for mid in municipality_ids])
    year_instruction = (
        f"for the year {year}" if year else "for the most recent available year(s)"
    )
    return (
        f"## Prompt: Compare KPI '{kpi_id}' Across Municipalities\n\n"
        f"**Objective:** Compare the Kolada Key Performance Indicator (KPI) with ID `{kpi_id}` across the following municipalities:\n{muns_list}\n\n"
        "**Instructions:**\n"
        f"1.  First, use the `get_kpi_metadata` tool with the KPI ID `{kpi_id}` to understand what this KPI represents. Briefly state its title and operating area.\n"
        f"2.  Next, use the `fetch_kolada_data` tool **for each municipality ID** listed above. Use the KPI ID `{kpi_id}` and retrieve data {year_instruction}.\n"
        "3.  Present the fetched data in a clear table format. Include columns for Municipality ID, Year, Gender (if applicable), and Value.\n"
        "4.  Handle potential errors gracefully (e.g., if data is missing for a municipality/year, indicate that in the table).\n"
        "5.  Provide a brief summary comparing the values across the municipalities based on the table."
    )


@mcp.prompt()
def find_and_fetch_kpi_by_area(
    operating_area: str, municipality_id: str, keyword: Optional[str] = None
) -> str:
    """
    Generates a prompt guiding the LLM through the operating area filtering workflow.
    """
    keyword_instruction = (
        f" focusing on KPIs containing the keyword '{keyword}' in their title or description"
        if keyword
        else ""
    )

    return (
        f"## Prompt: Find and Fetch Data for a KPI in '{operating_area}' for Municipality '{municipality_id}'\n\n"
        f"**Objective:** Find a relevant KPI within the Kolada operating area '{operating_area}'{keyword_instruction}, and then fetch its data for municipality `{municipality_id}`.\n\n"
        "**Instructions (Follow these steps using the available tools):\n"
        f"1.  **(Optional but Recommended):** If unsure about the exact area name, you can first use `list_operating_areas` to see all available areas and their KPI counts.\n"
        f"2.  Use the `get_kpis_by_operating_area` tool with the `operating_area` parameter set to '{operating_area}'. This will give you a list of KPIs in that category.\n"
        f"3.  From the list returned in step 2, review the KPI titles and descriptions{keyword_instruction}. Select the *single most relevant* KPI ID that matches the objective.\n"
        f"4.  Use the `get_kpi_metadata` tool with the selected KPI ID to confirm its details (title, description). State the chosen KPI's title.\n"
        f"5.  Finally, use the `fetch_kolada_data` tool with the chosen KPI ID and the `municipality_id` '{municipality_id}'. Fetch data for the most recent available year(s).\n"
        "6.  Present the fetched data value(s) clearly, including the year and gender (if applicable).\n"
        "7.  If you cannot find a suitable KPI or if data is unavailable, clearly state that."
    )


###############################################################################
# 7) Main entry (Unchanged, includes stderr prints and exception handling)
###############################################################################

if __name__ == "__main__":
    print("[Kolada MCP Main] TOP LEVEL OF MAIN REACHED", file=sys.stderr)
    print("[Kolada MCP Main] Script starting...", file=sys.stderr)
    try:
        print(
            "[Kolada MCP Main] Calling mcp.run(transport='stdio')...", file=sys.stderr
        )
        # Run the MCP server over stdio by default
        mcp.run(transport="stdio")
        # This line should ideally not be reached if run() blocks correctly with stdio
        print("[Kolada MCP Main] mcp.run() finished unexpectedly.", file=sys.stderr)
    except Exception as e:
        # Catch any top-level exception during server run
        print(
            f"[Kolada MCP Main] EXCEPTION caught around mcp.run(): {e}", file=sys.stderr
        )
        traceback.print_exc(file=sys.stderr)
    finally:
        print("[Kolada MCP Main] Script exiting.", file=sys.stderr)
