import json
import statistics  # For median and mean calculation
import sys
import traceback
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, TypedDict

import httpx

# Use the base Context provided by the framework for type hinting in tools
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.server import Context

###############################################################################
# 1) Global Constants
###############################################################################
BASE_URL = "https://api.kolada.se/v2"
KPI_PER_PAGE = 5000

###############################################################################
# 2) Typed structures for your Kolada data
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


class KoladaMunicipality(TypedDict, total=False):
    """
    Represents a single municipality (or region) from Kolada.
    Each entry typically has:
      - 'id': Municipality ID (e.g., "0180" for Stockholm),
      - 'title': The human-readable municipality name (e.g., "Stockholm"),
      - 'type': "K" (kommun), "R" (region), "L" (landsting), etc.
    """

    id: str
    title: str
    type: str


class KoladaLifespanContext(TypedDict):
    """
    Data cached in the server's memory at startup ('lifespan_context').
    This avoids repeatedly fetching static metadata from the Kolada API.
    """

    # KPI data
    kpi_cache: list[KoladaKpi]  # A list of all KPI metadata objects.
    kpi_map: dict[str, KoladaKpi]  # Mapping from KPI ID -> KPI object for quick lookups
    operating_areas_summary: list[dict[str, str | int]]

    # Municipality data
    municipality_cache: list[
        KoladaMunicipality
    ]  # All municipality (or region) objects from Kolada
    municipality_map: dict[
        str, KoladaMunicipality
    ]  # Mapping from municipality_id -> municipality data


###############################################################################
# 3) Helper Functions
###############################################################################


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
) -> list[dict[str, str | int]]:
    """Generates a summary list of operating areas and their KPI counts."""
    grouped = _group_kpis_by_operating_area(kpis)
    areas_with_counts: list[dict[str, str | int]] = []
    for area in sorted(grouped.keys()):
        areas_with_counts.append(
            {"operating_area": area, "kpi_count": len(grouped[area])}
        )
    return areas_with_counts


async def _fetch_data_from_kolada(
    url: str, params: dict[str, Any] | None = None
) -> dict[str, Any]:
    """
    Helper function to fetch data from Kolada with consistent error handling.
    Now includes pagination support: if 'next_page' is present, we keep fetching
    subsequent pages and merge 'values' into one combined list.

    To avoid infinite loops, we track visited URLs in 'visited_urls'. If we see
    the same 'next_page' repeatedly, we break.
    """
    combined_values: list[dict[str, Any]] = []
    params = params or {}
    visited_urls = set()

    async with httpx.AsyncClient() as client:
        while url and url not in visited_urls:
            visited_urls.add(url)
            print(
                f"[Kolada MCP] Fetching page: {url} with params: {params}",
                file=sys.stderr,
            )
            try:
                resp = await client.get(url, params=params, timeout=60.0)
                resp.raise_for_status()  # Raises HTTPStatusError if status >= 400
                data = resp.json()
            except (
                httpx.RequestError,
                httpx.HTTPStatusError,
                json.JSONDecodeError,
            ) as ex:
                error_msg = f"Error accessing Kolada API: {ex}"
                print(f"[Kolada MCP] {error_msg}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                return {
                    "error": error_msg,
                    "details": str(ex),
                    "endpoint": url,
                    "params": params,
                }

            # If there's an error key, just return it
            if "error" in data:
                return data

            # Merge this page's values into combined
            page_values = data.get("values", [])
            combined_values.extend(page_values)

            # Check if there's a next_page
            next_url = data.get("next_page")
            if not next_url:
                # No more pages
                url = None
            else:
                # Prepare for next iteration
                url = next_url
                # Typically, next_page already has query params, so we reset:
                params = {}

    # Return a single merged structure
    return {
        "count": len(combined_values),
        "values": combined_values,
    }


def _safe_get_lifespan_context(ctx: Context) -> KoladaLifespanContext | None:  # type: ignore[Context]
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
    return ctx.request_context.lifespan_context  # type: ignore


###############################################################################
# 4) Server Lifespan: Fetch & Cache Kolada Metadata + Municipalities
###############################################################################


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[KoladaLifespanContext]:
    """
    Initializes the Kolada MCP Server at startup. Includes stderr logging.
    Yields the dictionary that becomes ctx.request_context.lifespan_context.
    """
    print("[Kolada MCP Lifespan] Starting lifespan setup...", file=sys.stderr)
    kpi_list: list[KoladaKpi] = []
    municipality_list: list[KoladaMunicipality] = []

    # ----------------------------------
    # 1) Fetch all KPI metadata
    # ----------------------------------
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
        f"[Kolada MCP] Fetched {len(kpi_list)} total KPIs from Kolada.",
        file=sys.stderr,
    )

    # ----------------------------------
    # 2) Fetch municipality data
    # ----------------------------------
    print("[Kolada MCP] Fetching municipality data...", file=sys.stderr)
    try:
        muni_resp = await _fetch_data_from_kolada(f"{BASE_URL}/municipality")
        if "error" in muni_resp:
            raise RuntimeError(
                f"Failed to initialize municipality cache: {muni_resp['error']}"
            )
        muni_values = muni_resp.get("values", [])
        # Each item is { "id": "...", "title": "...", "type": "K" }
        municipality_list = muni_values
        print(
            f"[Kolada MCP] Fetched {len(municipality_list)} municipalities/regions.",
            file=sys.stderr,
        )
    except Exception as e:
        print(
            f"[Kolada MCP] CRITICAL ERROR fetching municipality data: {e}",
            file=sys.stderr,
        )
        raise RuntimeError(f"Failed to initialize municipality cache: {e}") from e

    # Build an ID -> KPI map
    kpi_map: dict[str, KoladaKpi] = {}
    for kpi_obj in kpi_list:
        k_id = kpi_obj.get("id")
        if k_id is not None:
            kpi_map[str(k_id)] = kpi_obj

    # Build an ID -> municipality map
    municipality_map: dict[str, KoladaMunicipality] = {}
    for m_obj in municipality_list:
        m_id = m_obj.get("id")
        if m_id is not None:
            municipality_map[str(m_id)] = m_obj

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
        "municipality_cache": municipality_list,
        "municipality_map": municipality_map,
    }

    print("[Kolada MCP] Initialization complete. All data cached.", file=sys.stderr)
    print(
        f"[Kolada MCP Lifespan] Yielding context with {len(kpi_list)} KPIs and {len(municipality_list)} municipalities...",
        file=sys.stderr,
    )
    try:
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


###############################################################################
# 5) Instantiate FastMCP
###############################################################################
mcp = FastMCP("KoladaServer", lifespan=app_lifespan)


###############################################################################
# 6) Additional Tools (Unchanged from prior code, but included for completeness)
###############################################################################


@mcp.tool()
async def list_operating_areas(ctx: Context) -> list[dict[str, str | int]]:
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
    lifespan_ctx = _safe_get_lifespan_context(ctx)
    if not lifespan_ctx:
        return [{"error": "Server context structure invalid or incomplete."}]

    summary = lifespan_ctx.get("operating_areas_summary", [])
    if not summary:
        print("Warning: Operating areas summary is empty in context.", file=sys.stderr)
        return []
    return summary


@mcp.tool()
async def get_kpis_by_operating_area(
    operating_area: str,
    ctx: Context,
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
    lifespan_ctx = _safe_get_lifespan_context(ctx)
    if not lifespan_ctx:
        return []

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


@mcp.tool()
async def get_kpi_metadata(
    kpi_id: str,
    ctx: Context,
) -> KoladaKpi | dict[str, str]:
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
    lifespan_ctx = _safe_get_lifespan_context(ctx)
    if not lifespan_ctx:
        return {"error": "Server context structure invalid or incomplete."}

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
    ctx: Context,
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
    lifespan_ctx = _safe_get_lifespan_context(ctx)
    if not lifespan_ctx:
        return []

    kpi_list = lifespan_ctx.get("kpi_cache", [])
    if not kpi_list:
        print("Warning: KPI cache is empty, cannot perform search.", file=sys.stderr)
        return []

    kw_lower = keyword.lower()
    matches: list[KoladaKpi] = []
    for kpi_obj in kpi_list:
        title = kpi_obj.get("title", "").lower()
        desc = kpi_obj.get("description", "").lower()
        if kw_lower in title or kw_lower in desc:
            matches.append(kpi_obj)
            if len(matches) >= limit:
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
    ctx: Context,
    year: str | None = None,
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
        if the API call fails. Additionally, a 'municipality_name' is attached to each item
        in 'values' if possible, based on the cached municipality map.
    """
    if not kpi_id or not municipality_id:
        return {"error": "kpi_id and municipality_id are required."}

    lifespan_ctx = _safe_get_lifespan_context(ctx)
    if not lifespan_ctx:
        return {"error": "Server context structure invalid or incomplete."}

    municipality_map = lifespan_ctx["municipality_map"]

    url = f"{BASE_URL}/data/kpi/{kpi_id}/municipality/{municipality_id}"
    params: dict[str, str] = {}
    if year:
        params["year"] = year

    resp_data = await _fetch_data_from_kolada(url, params)
    if "error" in resp_data:
        return resp_data  # pass along the error structure

    # Attach municipality_name to each top-level item in resp_data['values']
    values_list = resp_data.get("values", [])
    for item in values_list:
        m_id = item.get("municipality")
        if m_id in municipality_map:
            item["municipality_name"] = municipality_map[m_id].get(
                "title", f"Kommun {m_id}"
            )
        else:
            item["municipality_name"] = f"Kommun {m_id}"

    return resp_data


###############################################################################
# 7) Refactored analyze_kpi_across_municipalities for Multi-Year Deltas
###############################################################################


def _parse_years_param(year_str: str) -> list[str]:
    """
    Parses a comma-separated string of years into a list (e.g. "2020,2021" -> ["2020","2021"]).
    If empty or invalid, returns an empty list.
    """
    if not year_str:
        return []
    parts = [y.strip() for y in year_str.split(",") if y.strip()]
    return parts


def _fetch_and_group_data_by_municipality(
    data: dict[str, Any],
    gender: str,
) -> dict[str, dict[str, float]]:
    """
    From a Kolada data response containing multiple years, extracts numeric values
    per (municipality, year) for the specified gender. Returns a structure:
    {
      "0180": { "2020": 123.0, "2021": 130.5 },
      "1860": { "2020": 90.0,  "2021": 95.0 },
      ...
    }
    Skips any entries where 'municipality' or 'period' is missing, or the value is invalid.
    """
    municipality_data: dict[str, dict[str, float]] = {}
    values_data = data.get("values", [])
    for item in values_data:
        municipality_id = item.get("municipality")
        period = item.get("period")
        if not municipality_id or not period:
            print(
                f"Warning: Skipping due to missing municipality_id or period: {item}",
                file=sys.stderr,
            )
            continue

        for value_item in item.get("values", []):
            if value_item.get("gender") == gender:
                val = value_item.get("value")
                if val is not None:
                    try:
                        numeric_value = float(val)
                        municipality_data.setdefault(municipality_id, {})[
                            period
                        ] = numeric_value
                    except (ValueError, TypeError):
                        print(
                            f"Warning: Could not convert '{val}' to float for municipality {municipality_id}, period {period}, gender {gender}.",
                            file=sys.stderr,
                        )
    return municipality_data


def _process_single_year_data(
    municipality_data: dict[str, dict[str, float]],
    municipality_map: dict[str, KoladaMunicipality],
    year: str,
    sort_order: str,
    limit: int,
    kpi_metadata: dict[str, Any],
    gender: str,
) -> dict[str, Any]:
    """
    Handles the original single-year case:
    - Sorts data by the value
    - Returns top/bottom/median sets and summary stats
    """
    processed_data: list[dict[str, Any]] = []
    all_values: list[float] = []

    for m_id, yearly_dict in municipality_data.items():
        if year in yearly_dict:
            numeric_value = yearly_dict[year]
            m_name = municipality_map.get(m_id, {}).get("title", f"Kommun {m_id}")
            processed_data.append(
                {
                    "municipality_id": m_id,
                    "municipality_name": m_name,
                    "year": year,
                    "value": numeric_value,
                }
            )
            all_values.append(numeric_value)

    if not processed_data:
        return {
            "error": f"No data available for the specified parameters (Year: {year}, Gender: {gender}).",
            "kpi_info": kpi_metadata,
            "selected_gender": gender,
            "selected_year": year,
            "municipalities_count": 0,
            "summary_stats": {},
            "top_municipalities": [],
            "bottom_municipalities": [],
            "median_municipalities": [],
        }

    is_descending = sort_order.lower() == "desc"
    sorted_data = sorted(
        processed_data,
        key=lambda x: (x["value"], x["municipality_id"]),
        reverse=is_descending,
    )
    municipalities_count = len(sorted_data)

    summary_stats: dict[str, float | int | None] = {
        "min": None,
        "max": None,
        "mean": None,
        "median": None,
        "count": municipalities_count,
    }

    if all_values:
        try:
            summary_stats["median"] = statistics.median(all_values)
            summary_stats["mean"] = statistics.mean(all_values)
            summary_stats["min"] = min(all_values)
            summary_stats["max"] = max(all_values)
        except statistics.StatisticsError as stat_err:
            print(
                f"Warning: Could not calculate statistics: {stat_err}", file=sys.stderr
            )

    safe_limit = (
        max(1, min(limit, municipalities_count)) if municipalities_count > 0 else 0
    )
    top_municipalities = sorted_data[:safe_limit]
    bottom_municipalities = sorted_data[-safe_limit:]
    bottom_municipalities.reverse()

    median_municipalities: list[dict[str, Any]] = []
    if municipalities_count > 0:
        n = municipalities_count
        median_rank_index_lower = (n - 1) // 2
        start_offset = safe_limit // 2
        median_start_index = max(0, median_rank_index_lower - start_offset)
        median_start_index = min(median_start_index, n - safe_limit)
        median_start_index = max(0, median_start_index)
        median_end_index = median_start_index + safe_limit
        median_municipalities = sorted_data[median_start_index:median_end_index]

    return {
        "kpi_info": kpi_metadata,
        "summary_stats": summary_stats,
        "top_municipalities": top_municipalities,
        "bottom_municipalities": bottom_municipalities,
        "median_municipalities": median_municipalities,
        "municipalities_count": municipalities_count,
        "selected_gender": gender,
        "selected_year": year,
        "sort_order": sort_order,
        "limit": limit,
    }


def _process_multi_year_data(
    municipality_data: dict[str, dict[str, float]],
    municipality_map: dict[str, KoladaMunicipality],
    years: list[str],
    sort_order: str,
    limit: int,
    kpi_metadata: dict[str, Any],
    gender: str,
    only_return_rate: bool,
) -> dict[str, Any]:
    """
    Handles multi-year logic:
    - For each municipality, compute earliest-latest difference
    - Sort/rank by the delta value
    - Return top/bottom/median sets for the change over the period
    """
    sorted_years = sorted(years)  # e.g., ["2020", "2021", "2022"]
    first_year, last_year = sorted_years[0], sorted_years[-1]

    processed_data: list[dict[str, Any]] = []
    all_deltas: list[float] = []

    for m_id, yearly_values in municipality_data.items():
        if first_year in yearly_values and last_year in yearly_values:
            earliest_val = yearly_values[first_year]
            latest_val = yearly_values[last_year]
            delta_value = latest_val - earliest_val
            m_name = municipality_map.get(m_id, {}).get("title", f"Kommun {m_id}")
            entry: dict[str, Any] = {
                "municipality_id": m_id,
                "municipality_name": m_name,
                "earliest_year": first_year,
                "latest_year": last_year,
                "earliest_value": earliest_val,
                "latest_value": latest_val,
                "delta_value": delta_value,
            }
            processed_data.append(entry)
            all_deltas.append(delta_value)

    if not processed_data:
        return {
            "error": f"No data found for multi-year analysis across {years} and gender={gender}. Possibly data is missing for earliest or latest year in each municipality.",
            "kpi_info": kpi_metadata,
            "selected_gender": gender,
            "selected_years": years,
            "municipalities_count": 0,
            "summary_stats": {},
            "top_municipalities": [],
            "bottom_municipalities": [],
            "median_municipalities": [],
        }

    # Sort by delta_value
    is_descending = sort_order.lower() == "desc"
    sorted_by_delta = sorted(
        processed_data,
        key=lambda x: (x["delta_value"], x["municipality_id"]),
        reverse=is_descending,
    )
    municipalities_count = len(sorted_by_delta)

    summary_stats: dict[str, float | int | None] = {
        "min_delta": None,
        "max_delta": None,
        "mean_delta": None,
        "median_delta": None,
        "count": municipalities_count,
    }

    if all_deltas:
        try:
            summary_stats["median_delta"] = statistics.median(all_deltas)
            summary_stats["mean_delta"] = statistics.mean(all_deltas)
            summary_stats["min_delta"] = min(all_deltas)
            summary_stats["max_delta"] = max(all_deltas)
        except statistics.StatisticsError as stat_err:
            print(
                f"Warning: Could not calculate statistics for multi-year: {stat_err}",
                file=sys.stderr,
            )

    safe_limit = (
        max(1, min(limit, municipalities_count)) if municipalities_count > 0 else 0
    )
    top_municipalities = sorted_by_delta[:safe_limit]
    bottom_municipalities = sorted_by_delta[-safe_limit:]
    bottom_municipalities.reverse()

    median_municipalities: list[dict[str, Any]] = []
    if municipalities_count > 0:
        n = municipalities_count
        median_rank_index_lower = (n - 1) // 2
        start_offset = safe_limit // 2
        median_start_index = max(0, median_rank_index_lower - start_offset)
        median_start_index = min(median_start_index, n - safe_limit)
        median_start_index = max(0, median_start_index)
        median_end_index = median_start_index + safe_limit
        median_municipalities = sorted_by_delta[median_start_index:median_end_index]

    # If only_return_rate is True, we do not return single-year stats, just the delta
    if only_return_rate:
        return {
            "kpi_info": kpi_metadata,
            "summary_stats": summary_stats,
            "top_municipalities": top_municipalities,
            "bottom_municipalities": bottom_municipalities,
            "median_municipalities": median_municipalities,
            "municipalities_count": municipalities_count,
            "selected_gender": gender,
            "selected_years": years,
            "sort_order": sort_order,
            "limit": limit,
            "multi_year_delta": True,
            "only_return_rate": True,
        }

    # Otherwise return both the delta-based ranking + flags
    return {
        "kpi_info": kpi_metadata,
        "summary_stats": summary_stats,
        "top_municipalities": top_municipalities,
        "bottom_municipalities": bottom_municipalities,
        "median_municipalities": median_municipalities,
        "municipalities_count": municipalities_count,
        "selected_gender": gender,
        "selected_years": years,
        "sort_order": sort_order,
        "limit": limit,
        "multi_year_delta": True,
        "only_return_rate": False,
    }


@mcp.tool()
async def analyze_kpi_across_municipalities(
    kpi_id: str,
    ctx: Context,
    year: str,
    sort_order: str = "desc",
    limit: int = 10,
    gender: str = "T",
    only_return_rate: bool = False,
) -> dict[str, Any]:
    """
    Analyserar ett KPI över alla kommuner och returnerar strukturerad data.
    Nu uppdaterat för att stödja flera år i ett enda anrop:

    1. Om enbart ETT år anges (t.ex. "2023"):
       - Hämtar data för detta år och returnerar samma statistik som tidigare
         (topp-, botten-, mediankommuner efter värdet).

    2. Om FLERA år anges (kommaseparerade, t.ex. "2020,2021,2022"):
       - Hämtar data för alla dessa år.
       - Beräknar förändring (delta_value) från det tidigaste året till det senaste,
         och returnerar topp-, botten- och medianlistor baserade på förändringen (dvs. delta_value).
       - Om parametern `only_return_rate=True`, returneras endast denna delta-baserade
         ranking och statistik, ej grunddata för enskilda år.

    Args:
        kpi_id: Det unika ID:t för KPI:n (t.ex. "N00530" för nöjdhet med förskola).
        year: Valfritt. Exempel:
            - "2023" för ett enskilt år,
            - "2020,2021,2022" för flera år.
            Om inget anges hämtas alla tillgängliga år (ej fullt implementerat här).
        sort_order: "asc" för stigande ordning, "desc" för fallande ordning (standard).
        limit: Antal kommuner att inkludera i topp-, botten- och medianlistorna (standard 10).
        gender: Kön att filtrera på ("K" för kvinnor, "M" för män, "T" för totalt).
        only_return_rate: Om True och flera år anges, returneras endast förändringsdata
            (top, bottom, median efter delta_value).
        ctx: Serverkontexten (injiceras automatiskt).

    Returns:
        Ett strukturerat dictionary med exempelvis:
        - 'kpi_info': Metadata om KPI:n (id, title, description, operating_area)
        - 'summary_stats': Statistiska mått
          (single-year: min, max, median, mean)
          (multi-year: min_delta, max_delta, median_delta, mean_delta)
        - 'top_municipalities': Topp-kommuner efter (value eller delta_value)
        - 'bottom_municipalities': Botten-kommuner (value/delta)
        - 'median_municipalities': Kommuner runt medianvärdet
        - 'municipalities_count': Antalet kommuner med data
        - 'selected_gender': "K", "M" eller "T"
        - 'selected_year': Om enskilt år
        - 'selected_years': Om flera år
        - 'multi_year_delta': Boolean, True när det är multi-year-analys
        - 'only_return_rate': Återspeglar funktionen om enbart delta ska returneras
        - 'error': Eventuellt felmeddelande
    """
    # Step 1: Fetch KPI metadata
    kpi_metadata_result = await get_kpi_metadata(kpi_id, ctx)
    if isinstance(kpi_metadata_result, dict) and "error" in kpi_metadata_result:
        return {
            "error": f"Failed to retrieve KPI metadata: {kpi_metadata_result['error']}",
            "kpi_id": kpi_id,
        }

    kpi_metadata = {
        "id": kpi_id,
        "title": kpi_metadata_result.get("title", ""),
        "description": kpi_metadata_result.get("description", ""),
        "operating_area": kpi_metadata_result.get("operating_area", ""),
    }

    # Step 2: Validate context & parse years
    lifespan_ctx = _safe_get_lifespan_context(ctx)
    if not lifespan_ctx:
        return {
            "error": "Server context structure invalid or incomplete.",
            "kpi_info": kpi_metadata,
        }
    municipality_map = lifespan_ctx["municipality_map"]

    year_list = _parse_years_param(year)

    # Step 3: Fetch Kolada data for the specified year(s)
    url = (
        f"{BASE_URL}/data/kpi/{kpi_id}/year/{year}"
        if year
        else f"{BASE_URL}/data/kpi/{kpi_id}"
    )
    kolada_data = await _fetch_data_from_kolada(url)
    if "error" in kolada_data:
        return {"error": kolada_data["error"], "kpi_info": kpi_metadata}

    # Step 4: Convert to municipality_data { m_id: {year: value} }
    municipality_data = _fetch_and_group_data_by_municipality(kolada_data, gender)

    # Step 5: Single-year vs Multi-year logic
    if len(year_list) <= 1:
        # Single-year scenario
        single_year = year_list[0] if year_list else (year or "Unknown")
        return _process_single_year_data(
            municipality_data=municipality_data,
            municipality_map=municipality_map,
            year=single_year,
            sort_order=sort_order,
            limit=limit,
            kpi_metadata=kpi_metadata,
            gender=gender,
        )
    else:
        # Multi-year scenario
        return _process_multi_year_data(
            municipality_data=municipality_data,
            municipality_map=municipality_map,
            years=year_list,
            sort_order=sort_order,
            limit=limit,
            kpi_metadata=kpi_metadata,
            gender=gender,
            only_return_rate=only_return_rate,
        )


###############################################################################
# 8) Prompt
###############################################################################


@mcp.prompt()
def kolada_entry_point() -> str:
    """
    Acts as a general entry point and guide for interacting with the Kolada MCP server.
    This prompt helps the LLM understand the available tools and devise a plan
    to answer the user's query about Swedish municipal and regional data.
    """
    return (
        "## Kolada MCP Server Interaction Guide\n\n"
        "**Objective:** You are interacting with the Kolada API via a set of tools. Kolada provides Key Performance Indicators (KPIs) for Swedish municipalities and regions across various sectors (e.g., demographics, economy, education, environment, healthcare).\n\n"
        "**Your Task:** Analyze the user's request carefully and use the available tools strategically to find the relevant information or perform the requested analysis. Think step-by-step about how to achieve the user's goal.\n\n"
        "**Available Tools & Common Use Cases:**\n\n"
        "1.  **`list_operating_areas()`:**\n"
        "    *   **Use When:** The user asks for the general *categories* or *themes* of data available.\n"
        "2.  **`get_kpis_by_operating_area(operating_area: str)`:**\n"
        "    *   **Use When:** The user wants to see *all KPIs within a specific category*.\n"
        "3.  **`search_kpis(keyword: str, limit: int = 20)`:**\n"
        "    *   **Use When:** The user is looking for KPIs related to a *specific topic or keyword*.\n"
        "4.  **`get_kpi_metadata(kpi_id: str)`:**\n"
        "    *   **Use When:** You have identified a *specific KPI ID* and need its *detailed description*.\n"
        "5.  **`fetch_kolada_data(kpi_id: str, municipality_id: str, year: str | None = None)`:**\n"
        "    *   **Use When:** The user wants the *actual data value(s)* for a *specific KPI* in a *specific municipality*.\n"
        "6.  **`analyze_kpi_across_municipalities(...)`:**\n"
        "    *   **Use When:** The user wants to *compare municipalities* for a *specific KPI* (supports multi-year analysis).\n\n"
        "**General Strategy & Workflow:**\n\n"
        "1. Understand the user's goal.\n"
        "2. If you need a KPI ID, find it (via `get_kpis_by_operating_area` or `search_kpis`).\n"
        "3. If data is municipality-specific, ensure you have the municipality ID.\n"
        "4. Use the appropriate fetch or analysis tool.\n"
        "5. Present the results clearly.\n"
        "6. If no data is found, let the user know.\n"
        "\n**Now, analyze the user's request and determine the best tool(s) and sequence to use.**"
    )


###############################################################################
# 9) Main entry
###############################################################################

if __name__ == "__main__":
    print("[Kolada MCP Main] TOP LEVEL OF MAIN REACHED", file=sys.stderr)
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
