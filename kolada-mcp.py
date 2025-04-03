import json
import os
import statistics  # For median and mean calculation
import sys
import traceback
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, TypedDict, cast

import httpx
import numpy as np
import polars as pl

# Use the base Context provided by the framework for type hinting in tools
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.server import Context
from sentence_transformers import SentenceTransformer

###############################################################################
# 1) Global Constants
###############################################################################
BASE_URL: str = "https://api.kolada.se/v2"
KPI_PER_PAGE: int = 5000
EMBEDDINGS_CACHE_FILE: str = "kpi_embeddings.npz"


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
    kpi_map: dict[str, KoladaKpi]  # Mapping from KPI ID -> KPI object
    operating_areas_summary: list[dict[str, str | int]]

    # Municipality data
    municipality_cache: list[KoladaMunicipality]
    municipality_map: dict[str, KoladaMunicipality]

    # Vector search additions
    sentence_model: SentenceTransformer  # The loaded embedding model
    kpi_embeddings: np.ndarray  # shape (num_kpis, embedding_dim)
    kpi_ids: list[str]  # KPI IDs in the same order as rows in kpi_embeddings


###############################################################################
# 3) Helper Functions
###############################################################################


def _group_kpis_by_operating_area(
    kpis: list[KoladaKpi],
) -> dict[str, list[KoladaKpi]]:
    """Groups KPIs by their 'operating_area' field."""
    grouped: dict[str, list[KoladaKpi]] = {}
    for kpi in kpis:
        operating_area_field: str = kpi.get("operating_area", "Unknown")
        areas: list[str] = [a.strip() for a in operating_area_field.split(",")]
        for area in areas:
            if area:
                if area not in grouped:
                    grouped[area] = []
                grouped[area].append(kpi)
    return grouped


def _get_operating_areas_summary(
    kpis: list[KoladaKpi],
) -> list[dict[str, str | int]]:
    """Generates a summary list of operating areas and their KPI counts."""
    grouped: dict[str, list[KoladaKpi]] = _group_kpis_by_operating_area(kpis)
    areas_with_counts: list[dict[str, str | int]] = []
    for area in sorted(grouped.keys()):
        area_dict: dict[str, str | int] = {
            "operating_area": area,
            "kpi_count": len(grouped[area]),
        }
        areas_with_counts.append(area_dict)
    return areas_with_counts


async def _fetch_data_from_kolada(url: str) -> dict[str, Any]:
    """
    Helper function to fetch data from Kolada with consistent error handling.
    Now includes pagination support: if 'next_page' is present, we keep fetching
    subsequent pages and merge 'values' into one combined list.
    """
    combined_values: list[dict[str, Any]] = []
    visited_urls: set[str] = set()

    this_url: str | None = url
    async with httpx.AsyncClient() as client:
        while this_url and this_url not in visited_urls:
            visited_urls.add(this_url)
            print(f"[Kolada MCP] Fetching page: {this_url}", file=sys.stderr)
            try:
                resp = await client.get(this_url, timeout=60.0)
                resp.raise_for_status()
                data: dict[str, Any] = resp.json()
            except (
                httpx.RequestError,
                httpx.HTTPStatusError,
                json.JSONDecodeError,
            ) as ex:
                error_msg: str = f"Error accessing Kolada API: {ex}"
                print(f"[Kolada MCP] {error_msg}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                return {"error": error_msg, "details": str(ex), "endpoint": this_url}

            if "error" in data:
                return data

            page_values: list[dict[str, Any]] = data.get("values", [])
            combined_values.extend(page_values)

            next_url: str | None = data.get("next_page")
            if not next_url:
                this_url = None
            else:
                this_url = next_url

    return {
        "count": len(combined_values),
        "values": combined_values,
    }


def _safe_get_lifespan_context(ctx: Context) -> KoladaLifespanContext | None:
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


def _parse_years_param(year_str: str) -> list[str]:
    """
    Parses a comma-separated string of years into a list (e.g. "2020,2021" -> ["2020","2021"]).
    If empty or invalid, returns an empty list.
    """
    if not year_str:
        return []
    parts: list[str] = [y.strip() for y in year_str.split(",") if y.strip()]
    return parts


def _fetch_and_group_data_by_municipality(
    data: dict[str, Any],
    gender: str,
) -> dict[str, dict[str, float]]:
    """
    From a Kolada data response containing multiple years, extracts numeric values
    per (municipality, year) for the specified gender. Returns a structure
    { "0180": { "2020": val, "2021": val}, ... }.
    """
    raw_rows: list[dict[str, Any]] = []

    values_data: list[dict[str, Any]] = data.get("values", [])
    for item in values_data:
        municipality_id: str | None = item.get("municipality")
        raw_period: int | str | None = item.get("period")
        if not municipality_id or raw_period is None:
            print(
                f"Warning: Skipping due to missing municipality_id or period: {item}",
                file=sys.stderr,
            )
            continue

        period_str: str = str(raw_period)

        for subval in item.get("values", []):
            row_gender: str | None = subval.get("gender")
            val: Any = subval.get("value")

            row: dict[str, Any] = {
                "municipality": municipality_id,
                "period": period_str,
                "gender": row_gender,
                "value": val,
            }
            raw_rows.append(row)

    df: pl.DataFrame = pl.from_dicts(raw_rows)
    if df.is_empty():
        empty_dict: dict[str, dict[str, float]] = {}
        return empty_dict

    df_filtered: pl.DataFrame = df.filter(
        (pl.col("gender") == gender) & (pl.col("value").is_not_null())
    ).drop(["gender"])

    if df_filtered.is_empty():
        empty_dict: dict[str, dict[str, float]] = {}
        return empty_dict

    df_cast: pl.DataFrame = df_filtered.with_columns(
        value=pl.col("value").cast(pl.Float64, strict=False)
    )

    municipality_dict: dict[str, dict[str, float]] = {}
    for row_data in df_cast.to_dicts():
        m_id: str = row_data["municipality"]
        p_str: str = row_data["period"]
        v_val: float = row_data["value"]
        if m_id not in municipality_dict:
            municipality_dict[m_id] = {}
        municipality_dict[m_id][p_str] = v_val

    return municipality_dict


###############################################################################
# 4) Shared Refactoring Helpers
###############################################################################


def _calculate_summary_stats(
    values: list[float], prefix: str = ""
) -> dict[str, float | int | None]:
    """
    Given a list of float values, computes min, max, mean, median, and count.
    Uses an optional prefix for keys (e.g., "" vs "delta_").
    """
    summary_stats: dict[str, float | int | None] = {
        f"{prefix}min": None,
        f"{prefix}max": None,
        f"{prefix}mean": None,
        f"{prefix}median": None,
        "count": len(values),
    }

    if values:
        try:
            summary_stats[f"{prefix}min"] = min(values)
            summary_stats[f"{prefix}max"] = max(values)
            summary_stats[f"{prefix}mean"] = statistics.mean(values)
            summary_stats[f"{prefix}median"] = statistics.median(values)
        except statistics.StatisticsError as stat_err:
            print(
                f"Warning: Could not calculate statistics: {stat_err}", file=sys.stderr
            )

    return summary_stats


def _rank_and_slice_municipalities(
    data: list[dict[str, Any]],
    sort_key: str,
    sort_order: str,
    limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Sorts a list of dict items by sort_key (e.g. 'value' or 'delta_value'),
    slices out top, bottom, and median sub-lists, and returns them.
    """
    is_descending: bool = sort_order.lower() == "desc"
    sorted_data: list[dict[str, Any]] = sorted(
        data,
        key=lambda x: (x.get(sort_key, 0.0), x.get("municipality_id", "")),
        reverse=is_descending,
    )
    count_data: int = len(sorted_data)
    if count_data == 0:
        empty_list: list[dict[str, Any]] = []
        return empty_list, empty_list, empty_list

    safe_limit: int = max(1, min(limit, count_data))

    top_municipalities: list[dict[str, Any]] = sorted_data[:safe_limit]
    bottom_municipalities: list[dict[str, Any]] = sorted_data[-safe_limit:]
    bottom_municipalities.reverse()

    median_municipalities: list[dict[str, Any]] = []
    if count_data > 0:
        n: int = count_data
        median_rank_index_lower: int = (n - 1) // 2
        start_offset: int = safe_limit // 2
        median_start_index: int = max(0, median_rank_index_lower - start_offset)
        median_start_index = min(median_start_index, n - safe_limit)
        median_start_index = max(0, median_start_index)
        median_end_index: int = median_start_index + safe_limit
        median_municipalities = sorted_data[median_start_index:median_end_index]

    return top_municipalities, bottom_municipalities, median_municipalities


###############################################################################
# 5) Single-year & Multi-year Data Processing (Unified)
###############################################################################


def _process_kpi_data(
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
    See docstring above in the original code: this function handles single-year
    or multi-year data, returning structured stats and top/bottom slices.
    """

    sorted_years: list[str] = sorted(years)
    is_multi_year: bool = len(sorted_years) > 1

    print(
        f"[Kolada MCP] Unified KPI processing. Requested years: {sorted_years}",
        file=sys.stderr,
    )
    print(
        f"[Kolada MCP] Processing data for {len(municipality_data)} municipalities.",
        file=sys.stderr,
    )

    full_municipality_list: list[dict[str, Any]] = []
    latest_values_list: list[float] = []
    delta_list: list[dict[str, Any]] = []
    delta_values: list[float] = []

    for m_id, yearly_values in municipality_data.items():
        available_years: list[str] = [y for y in sorted_years if y in yearly_values]
        if not available_years:
            continue

        earliest_year: str = available_years[0]
        latest_year: str = available_years[-1]
        earliest_val: float = yearly_values[earliest_year]
        latest_val: float = yearly_values[latest_year]

        m_name: str = municipality_map.get(m_id, {}).get("title", f"Kommun {m_id}")
        entry: dict[str, Any] = {
            "municipality_id": m_id,
            "municipality_name": m_name,
            "latest_year": latest_year,
            "latest_value": latest_val,
            "years_in_data": available_years,
        }
        full_municipality_list.append(entry)
        latest_values_list.append(latest_val)

        # Multi-year delta
        if len(available_years) >= 2:
            delta_value: float = latest_val - earliest_val
            entry["earliest_year"] = earliest_year
            entry["earliest_value"] = earliest_val
            entry["delta_value"] = delta_value
            delta_list.append(entry)
            delta_values.append(delta_value)

    if not full_municipality_list:
        return {
            "error": f"No data available for the specified parameters (Years: {years}, Gender: {gender}).",
            "kpi_info": kpi_metadata,
            "selected_gender": gender,
            "selected_years": years,
            "municipalities_count": 0,
            "summary_stats": {},
            "top_municipalities": [],
            "bottom_municipalities": [],
            "median_municipalities": [],
        }

    if only_return_rate:
        delta_top, delta_bottom, delta_median = _rank_and_slice_municipalities(
            delta_list, "delta_value", sort_order, limit
        )
        delta_stats: dict[str, float | int | None] = _calculate_summary_stats(
            delta_values
        )
        delta_summary_stats: dict[str, float | int | None] = {
            "min_delta": delta_stats["min"],
            "max_delta": delta_stats["max"],
            "mean_delta": delta_stats["mean"],
            "median_delta": delta_stats["median"],
            "count": delta_stats["count"],
        }
        return {
            "kpi_info": kpi_metadata,
            "summary_stats": delta_summary_stats,
            "top_municipalities": [],
            "bottom_municipalities": [],
            "median_municipalities": [],
            "municipalities_count": len(delta_list),
            "selected_gender": gender,
            "selected_years": years,
            "sort_order": sort_order,
            "limit": limit,
            "multi_year_delta": is_multi_year,
            "only_return_rate": True,
            "delta_municipalities": delta_list,
            "top_delta_municipalities": delta_top,
            "bottom_delta_municipalities": delta_bottom,
            "median_delta_municipalities": delta_median,
        }

    top_main, bottom_main, median_main = _rank_and_slice_municipalities(
        full_municipality_list, "latest_value", sort_order, limit
    )
    main_stats: dict[str, float | int | None] = _calculate_summary_stats(
        latest_values_list
    )
    summary_stats: dict[str, float | int | None] = {
        "min_latest": main_stats["min"],
        "max_latest": main_stats["max"],
        "mean_latest": main_stats["mean"],
        "median_latest": main_stats["median"],
        "count": main_stats["count"],
    }

    delta_top, delta_bottom, delta_median = _rank_and_slice_municipalities(
        delta_list, "delta_value", sort_order, limit
    )
    delta_stats: dict[str, float | int | None] = _calculate_summary_stats(delta_values)
    delta_summary_stats: dict[str, float | int | None] = {
        "min_delta": delta_stats["min"],
        "max_delta": delta_stats["max"],
        "mean_delta": delta_stats["mean"],
        "median_delta": delta_stats["median"],
        "count": delta_stats["count"],
    }

    return {
        "kpi_info": kpi_metadata,
        "summary_stats": summary_stats,
        "top_municipalities": top_main,
        "bottom_municipalities": bottom_main,
        "median_municipalities": median_main,
        "municipalities_count": len(full_municipality_list),
        "selected_gender": gender,
        "selected_years": years,
        "sort_order": sort_order,
        "limit": limit,
        "multi_year_delta": is_multi_year,
        "only_return_rate": False,
        "top_delta_municipalities": delta_top,
        "bottom_delta_municipalities": delta_bottom,
        "median_delta_municipalities": delta_median,
    }


###############################################################################
# 6) Server Lifespan: Fetch & Cache Kolada Metadata + Municipalities
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

    print(
        "[Kolada MCP] Initializing: Fetching all KPI metadata from Kolada API...",
        file=sys.stderr,
    )
    async with httpx.AsyncClient() as client:
        next_url: str | None = f"{BASE_URL}/kpi?per_page={KPI_PER_PAGE}"
        while next_url:
            print(f"[Kolada MCP] Fetching page: {next_url}", file=sys.stderr)
            try:
                resp = await client.get(next_url, timeout=180.0)
                resp.raise_for_status()
                data: dict[str, Any] = resp.json()
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

            values: list[KoladaKpi] = data.get("values", [])
            kpi_list.extend(values)
            next_url = data.get("next_page")

    print(
        f"[Kolada MCP] Fetched {len(kpi_list)} total KPIs from Kolada.", file=sys.stderr
    )

    print("[Kolada MCP] Fetching municipality data...", file=sys.stderr)
    try:
        muni_resp: dict[str, Any] = await _fetch_data_from_kolada(
            f"{BASE_URL}/municipality"
        )
        if "error" in muni_resp:
            raise RuntimeError(
                f"Failed to initialize municipality cache: {muni_resp['error']}"
            )
        muni_values: list[Any] = muni_resp.get("values", [])
        municipality_list = cast(list[KoladaMunicipality], muni_values)
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

    kpi_map: dict[str, KoladaKpi] = {}
    for kpi_obj in kpi_list:
        k_id: str | None = kpi_obj.get("id")
        if k_id is not None:
            kpi_map[k_id] = kpi_obj

    municipality_map: dict[str, KoladaMunicipality] = {}
    for m_obj in municipality_list:
        m_id: str | None = m_obj.get("id")
        if m_id is not None:
            municipality_map[m_id] = m_obj

    operating_areas_summary: list[dict[str, str | int]] = _get_operating_areas_summary(
        kpi_list
    )
    print(
        f"[Kolada MCP] Identified {len(operating_areas_summary)} unique operating areas.",
        file=sys.stderr,
    )

    # ----------------------------------------------------------------
    # Load or create embeddings for the simpler vector-based approach
    # ----------------------------------------------------------------
    print("[Kolada MCP] Loading SentenceTransformer model...", file=sys.stderr)
    sentence_model: SentenceTransformer = SentenceTransformer(
        "KBLab/sentence-bert-swedish-cased"
    )
    print("[Kolada MCP] Model loaded.", file=sys.stderr)

    all_kpis: list[KoladaKpi] = [k for k in kpi_list if "id" in k]
    kpi_ids_list: list[str] = []
    titles_list: list[str] = []

    for kpi_obj in all_kpis:
        k_id: str = kpi_obj["id"]
        title_str: str = kpi_obj.get("title", "")
        kpi_ids_list.append(k_id)
        titles_list.append(title_str)

    # Attempt to load cached .npz
    existing_embeddings: np.ndarray | None = None
    loaded_ids: list[str] = []
    if os.path.isfile(EMBEDDINGS_CACHE_FILE):
        print(
            f"[Kolada MCP] Found embeddings cache at {EMBEDDINGS_CACHE_FILE}",
            file=sys.stderr,
        )
        try:
            cache_data: dict[str, Any] = dict(
                np.load(EMBEDDINGS_CACHE_FILE, allow_pickle=True)
            )
            existing_embeddings = cache_data.get("embeddings", None)
            loaded_ids_arr: Any = cache_data.get("kpi_ids", None)
            if isinstance(loaded_ids_arr, np.ndarray):
                loaded_ids = loaded_ids_arr.tolist()
        except Exception as ex:
            print(f"[Kolada MCP] Failed to load .npz cache: {ex}", file=sys.stderr)
            existing_embeddings = None

    # Check if we can reuse the loaded embeddings
    embeddings: np.ndarray
    if (
        existing_embeddings is not None
        and len(loaded_ids) == len(kpi_ids_list)
        and set(loaded_ids) == set(kpi_ids_list)
    ):
        print("[Kolada MCP] Using existing cached embeddings.", file=sys.stderr)
        embeddings = existing_embeddings
    else:
        print(
            "[Kolada MCP] Generating new embeddings for all KPI titles...",
            file=sys.stderr,
        )
        embeddings = sentence_model.encode(
            titles_list, show_progress_bar=True, normalize_embeddings=True
        )

        # Save them
        try:
            np.savez(
                EMBEDDINGS_CACHE_FILE,
                embeddings=embeddings,
                kpi_ids=np.array(kpi_ids_list),
            )
            print("[Kolada MCP] Embeddings saved to disk.", file=sys.stderr)
        except Exception as ex:
            print(
                f"[Kolada MCP] WARNING: Failed to save embeddings: {ex}",
                file=sys.stderr,
            )

    # Create the final context data
    context_data: KoladaLifespanContext = {
        "kpi_cache": kpi_list,
        "kpi_map": kpi_map,
        "operating_areas_summary": operating_areas_summary,
        "municipality_cache": municipality_list,
        "municipality_map": municipality_map,
        "sentence_model": sentence_model,
        "kpi_embeddings": embeddings,
        "kpi_ids": kpi_ids_list,
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
# 7) Instantiate FastMCP
###############################################################################
mcp: FastMCP = FastMCP("KoladaServer", lifespan=app_lifespan)


###############################################################################
# 8) Additional Tools
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
    lifespan_ctx: KoladaLifespanContext | None = _safe_get_lifespan_context(ctx)
    if not lifespan_ctx:
        error_list: list[dict[str, str]] = [{"error": "Server context invalid."}]
        return error_list

    summary: list[dict[str, str | int]] = lifespan_ctx.get(
        "operating_areas_summary", []
    )
    if not summary:
        print("Warning: Operating areas summary is empty in context.", file=sys.stderr)
        empty_list: list[dict[str, str | int]] = []
        return empty_list
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
    lifespan_ctx: KoladaLifespanContext | None = _safe_get_lifespan_context(ctx)
    if not lifespan_ctx:
        empty_list: list[KoladaKpi] = []
        return empty_list

    kpi_list: list[KoladaKpi] = lifespan_ctx.get("kpi_cache", [])
    if not kpi_list:
        print("Warning: KPI cache is empty in context.", file=sys.stderr)
        empty_list: list[KoladaKpi] = []
        return empty_list

    target_area_lower: str = operating_area.lower().strip()
    matches: list[KoladaKpi] = []

    for kpi in kpi_list:
        area_field: str = kpi.get("operating_area", "").lower()
        kpi_areas: set[str] = {a.strip() for a in area_field.split(",")}
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
    lifespan_ctx: KoladaLifespanContext | None = _safe_get_lifespan_context(ctx)
    if not lifespan_ctx:
        return {"error": "Server context structure invalid or incomplete."}

    kpi_map: dict[str, KoladaKpi] = lifespan_ctx.get("kpi_map", {})
    kpi_obj: KoladaKpi | None = kpi_map.get(kpi_id)

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
    lifespan_ctx: KoladaLifespanContext | None = _safe_get_lifespan_context(ctx)
    if not lifespan_ctx:
        empty_list: list[KoladaKpi] = []
        return empty_list

    # --- Vector-based approach (while keeping the original docstring) ---
    model: SentenceTransformer = lifespan_ctx["sentence_model"]
    embeddings: np.ndarray = lifespan_ctx["kpi_embeddings"]
    kpi_ids: list[str] = lifespan_ctx["kpi_ids"]
    kpi_map: dict[str, KoladaKpi] = lifespan_ctx["kpi_map"]

    if embeddings.shape[0] == 0:
        print(
            "[Kolada MCP] No KPI embeddings found; returning empty list.",
            file=sys.stderr,
        )
        empty_list: list[KoladaKpi] = []
        return empty_list

    # 1) Embed user query
    query_vector: np.ndarray = model.encode([keyword], normalize_embeddings=True)
    query_vec: np.ndarray = query_vector[0]  # shape (embedding_dim,)

    # 2) Compute dot products with normalized embeddings
    #    (We assume embeddings is already normalized)
    sims: np.ndarray = embeddings @ query_vec  # shape (num_kpis,)

    # 3) Sort descending by similarity
    indices_sorted: np.ndarray = np.argsort(-sims)
    top_indices: np.ndarray = indices_sorted[:limit]

    results: list[KoladaKpi] = []
    for idx in top_indices:
        kpi_id: str = kpi_ids[idx]
        if kpi_id in kpi_map:
            results.append(kpi_map[kpi_id])

    return results


@mcp.tool()
async def fetch_kolada_data(
    kpi_id: str,
    municipality_id: str,
    ctx: Context,
    year: str | None = None,
    municipality_type: str = "K",
) -> dict[str, Any]:
    """
    Fetches the *actual statistical data values* for a specific Kolada KPI
    and a specific Swedish municipality (identified by its ID, e.g., "1860" for Åsele).
    Optionally, you can specify a year or range of years.
    This tool calls the *live* Kolada API endpoint `/v2/data/kpi/.../municipality/...`.
    Use this tool *after* you have identified the specific KPI ID and municipality ID you need data for.

    Args:
        kpi_id: The unique ID of the Kolada KPI.
        municipality_id: The official ID of the Swedish municipality.
        year: Optional. A specific year (e.g., "2023") or a comma-separated list/range
              (e.g., "2020,2021,2022"). If None, fetches all available years.
        municipality_type: (Default = "K"). Filter so that results only come from municipalities
              of this type ("K", "R", or "L"). If the requested municipality does not match
              this type, an error is returned.
        ctx: The server context (injected automatically, currently unused but available).

    Returns:
        A dictionary containing the response from the Kolada API, which typically includes
        'count', 'values' (list of data points per year/gender), etc., or an error dictionary
        if the API call fails. Additionally, a 'municipality_name' is attached to each item
        in 'values' if possible, based on the cached municipality map.
    """
    lifespan_ctx: KoladaLifespanContext | None = _safe_get_lifespan_context(ctx)
    if not lifespan_ctx:
        return {"error": "Server context structure invalid or incomplete."}

    if not kpi_id or not municipality_id:
        return {"error": "kpi_id and municipality_id are required."}

    municipality_map: dict[str, KoladaMunicipality] = lifespan_ctx["municipality_map"]
    if municipality_id not in municipality_map:
        return {"error": f"Municipality ID '{municipality_id}' not found in system."}

    actual_type: str | None = municipality_map[municipality_id].get("type", None)
    if actual_type != municipality_type:
        return {
            "error": f"Municipality '{municipality_id}' is type '{actual_type}', "
            f"but user requested type '{municipality_type}'."
        }

    url: str = f"{BASE_URL}/data/kpi/{kpi_id}/municipality/{municipality_id}"
    resp_data: dict[str, Any] = await _fetch_data_from_kolada(url)
    if "error" in resp_data:
        return resp_data

    values_list: list[dict[str, Any]] = resp_data.get("values", [])
    for item in values_list:
        m_id: str = item.get("municipality", "Unknown")
        if m_id in municipality_map:
            item["municipality_name"] = municipality_map[m_id].get(
                "title", f"Kommun {m_id}"
            )
        else:
            item["municipality_name"] = f"Kommun {m_id}"

    return resp_data


@mcp.tool()
async def analyze_kpi_across_municipalities(
    kpi_id: str,
    ctx: Context,
    year: str,
    sort_order: str = "desc",
    limit: int = 10,
    gender: str = "T",
    only_return_rate: bool = False,
    municipality_type: str = "K",
) -> dict[str, Any]:
    """
    Analyzes a KPI across all municipalities and returns structured data.
    Now updated to a unified approach for single-year & multi-year:

    1. If ONLY ONE year is specified ...
    2. If MULTIPLE years are specified ...
    ...
    (Docstring truncated for brevity, see above for full text.)
    """
    kpi_metadata_result: KoladaKpi | dict[str, str] = await get_kpi_metadata(
        kpi_id, ctx
    )
    kpi_metadata: dict[str, Any] = {
        "id": kpi_id,
        "title": kpi_metadata_result.get("title", ""),
        "description": kpi_metadata_result.get("description", ""),
        "operating_area": kpi_metadata_result.get("operating_area", ""),
    }

    print(
        f"[Kolada MCP] Analyzing KPI {kpi_id} ({kpi_metadata['title']}) across municipalities.",
        file=sys.stderr,
    )

    lifespan_ctx: KoladaLifespanContext | None = _safe_get_lifespan_context(ctx)
    if not lifespan_ctx:
        return {
            "error": "Server context structure invalid or incomplete.",
            "kpi_info": kpi_metadata,
        }

    municipality_map: dict[str, KoladaMunicipality] = lifespan_ctx["municipality_map"]
    year_list: list[str] = _parse_years_param(year)

    url: str
    if year:
        url = f"{BASE_URL}/data/kpi/{kpi_id}/year/{year}"
    else:
        url = f"{BASE_URL}/data/kpi/{kpi_id}"

    kolada_data: dict[str, Any] = await _fetch_data_from_kolada(url)
    if "error" in kolada_data:
        return {"error": kolada_data["error"], "kpi_info": kpi_metadata}

    print(
        f"[Kolada MCP] Fetched data for {len(kolada_data.get('values', []))} entries.",
        file=sys.stderr,
    )
    print(
        f"[Kolada MCP] Sample data: {list(kolada_data.get('values', [])[:5])}",
        file=sys.stderr,
    )

    municipality_data: dict[str, dict[str, float]] = (
        _fetch_and_group_data_by_municipality(kolada_data, gender)
    )
    print(
        f"[Kolada MCP] Fetched data for {len(municipality_data)} municipalities.",
        file=sys.stderr,
    )

    filtered_municipality_data: dict[str, dict[str, float]] = {}
    for m_id, yearly_vals in municipality_data.items():
        if m_id in municipality_map:
            actual_type: str = municipality_map[m_id].get("type", "")
            if actual_type == municipality_type:
                filtered_municipality_data[m_id] = yearly_vals

    return _process_kpi_data(
        municipality_data=filtered_municipality_data,
        municipality_map=municipality_map,
        years=year_list,
        sort_order=sort_order,
        limit=limit,
        kpi_metadata=kpi_metadata,
        gender=gender,
        only_return_rate=only_return_rate,
    )


@mcp.tool()
async def compare_kpis(
    kpi1_id: str,
    kpi2_id: str,
    year: str,
    ctx: Context,
    gender: str = "T",
    municipality_type: str = "K",
) -> dict[str, Any]:
    """
    Compare two Kolada KPIs across municipalities and compute correlations.

    The behavior differs depending on single vs multiple years ...
    (See original docstring above for full details.)
    """
    kpi1_meta: KoladaKpi | dict[str, str] = await get_kpi_metadata(kpi1_id, ctx)
    kpi2_meta: KoladaKpi | dict[str, str] = await get_kpi_metadata(kpi2_id, ctx)

    kpi1_info: dict[str, str] = {
        "id": kpi1_id,
        "title": kpi1_meta.get("title", ""),
        "description": kpi1_meta.get("description", ""),
        "operating_area": kpi1_meta.get("operating_area", ""),
    }
    kpi2_info: dict[str, str] = {
        "id": kpi2_id,
        "title": kpi2_meta.get("title", ""),
        "description": kpi2_meta.get("description", ""),
        "operating_area": kpi2_meta.get("operating_area", ""),
    }

    year_list: list[str] = _parse_years_param(year)
    is_multi_year: bool = len(year_list) > 1

    lifespan_ctx: KoladaLifespanContext | None = _safe_get_lifespan_context(ctx)
    if not lifespan_ctx:
        await ctx.error(
            "compare_kpis: Server context invalid or missing lifespan context."
        )
        return {
            "error": "Server context invalid.",
            "kpi1_info": kpi1_info,
            "kpi2_info": kpi2_info,
        }

    municipality_map: dict[str, KoladaMunicipality] = lifespan_ctx["municipality_map"]

    if year:
        url1: str = f"{BASE_URL}/data/kpi/{kpi1_id}/year/{year}"
    else:
        url1 = f"{BASE_URL}/data/kpi/{kpi1_id}"

    data_kpi1: dict[str, Any] = await _fetch_data_from_kolada(url1)
    if "error" in data_kpi1:
        await ctx.error(
            f"compare_kpis: Error fetching data for KPI1 '{kpi1_id}' at '{url1}': {data_kpi1['error']}"
        )
        return {
            "error": data_kpi1["error"],
            "kpi1_info": kpi1_info,
            "kpi2_info": kpi2_info,
        }

    municipality_data1: dict[str, dict[str, float]] = (
        _fetch_and_group_data_by_municipality(data_kpi1, gender)
    )

    if year:
        url2: str = f"{BASE_URL}/data/kpi/{kpi2_id}/year/{year}"
    else:
        url2 = f"{BASE_URL}/data/kpi/{kpi2_id}"

    data_kpi2: dict[str, Any] = await _fetch_data_from_kolada(url2)
    if "error" in data_kpi2:
        await ctx.error(
            f"compare_kpis: Error fetching data for KPI2 '{kpi2_id}' at '{url2}': {data_kpi2['error']}"
        )
        return {
            "error": data_kpi2["error"],
            "kpi1_info": kpi1_info,
            "kpi2_info": kpi2_info,
        }

    municipality_data2: dict[str, dict[str, float]] = (
        _fetch_and_group_data_by_municipality(data_kpi2, gender)
    )

    def filter_muni_type(
        data_dict: dict[str, dict[str, float]],
    ) -> dict[str, dict[str, float]]:
        result_dict: dict[str, dict[str, float]] = {}
        for m_id, year_values in data_dict.items():
            muni_obj: KoladaMunicipality | None = municipality_map.get(m_id)
            if muni_obj and muni_obj.get("type") == municipality_type:
                result_dict[m_id] = year_values
        return result_dict

    municipality_data1 = filter_muni_type(municipality_data1)
    municipality_data2 = filter_muni_type(municipality_data2)

    result: dict[str, Any] = {
        "kpi1_info": kpi1_info,
        "kpi2_info": kpi2_info,
        "selected_years": year_list,
        "gender": gender,
        "municipality_type": municipality_type,
        "multi_year": is_multi_year,
    }

    import statistics

    async def compute_pearson_correlation(
        x_vals: list[float], y_vals: list[float]
    ) -> float | None:
        if len(x_vals) < 2 or len(y_vals) < 2:
            return None
        try:
            return statistics.correlation(x_vals, y_vals)
        except (ValueError, statistics.StatisticsError) as exc:
            await ctx.warning(f"Failed to compute correlation: {exc}")
            return None

    if not is_multi_year:
        if not year_list:
            await ctx.warning("compare_kpis: No valid single year specified.")
            return {
                **result,
                "error": "No valid year specified for single-year analysis.",
            }

        single_year: str = year_list[0]
        x_vals: list[float] = []
        y_vals: list[float] = []
        cross_section_data: list[dict[str, Any]] = []

        for m_id, values_1 in municipality_data1.items():
            values_2: dict[str, float] | None = municipality_data2.get(m_id)
            if not values_2:
                continue
            if single_year in values_1 and single_year in values_2:
                k1_val: float = values_1[single_year]
                k2_val: float = values_2[single_year]
                x_vals.append(k1_val)
                y_vals.append(k2_val)

                cross_section_data.append(
                    {
                        "municipality_id": m_id,
                        "municipality_name": municipality_map.get(m_id, {}).get(
                            "title", f"Municipality {m_id}"
                        ),
                        "kpi1_value": k1_val,
                        "kpi2_value": k2_val,
                        "difference": k2_val - k1_val,
                    }
                )

        if not cross_section_data:
            await ctx.warning(
                f"compare_kpis: No overlapping data found for year {single_year}."
            )
            return {
                **result,
                "error": f"No overlapping data for single year {single_year}.",
            }

        overall_corr: float | None = await compute_pearson_correlation(x_vals, y_vals)
        result["overall_correlation"] = overall_corr

        cross_section_data.sort(key=lambda item: item["difference"])
        n_muni: int = len(cross_section_data)
        slice_limit: int = min(10, n_muni)
        median_start: int = max(0, (n_muni - 1) // 2 - (slice_limit // 2))
        median_end: int = min(median_start + slice_limit, n_muni)

        result["municipality_differences"] = cross_section_data
        result["top_difference_municipalities"] = list(
            reversed(cross_section_data[-slice_limit:])
        )
        result["bottom_difference_municipalities"] = cross_section_data[:slice_limit]
        result["median_difference_municipalities"] = cross_section_data[
            median_start:median_end
        ]

        return result

    big_x: list[float] = []
    big_y: list[float] = []
    municipality_correlations: list[dict[str, Any]] = []

    for m_id, values_1 in municipality_data1.items():
        values_2: dict[str, float] | None = municipality_data2.get(m_id)
        if not values_2:
            continue

        intersection_years: list[str] = sorted(
            set(values_1.keys()) & set(values_2.keys())
        )
        if not intersection_years:
            continue

        ts_x: list[float] = []
        ts_y: list[float] = []
        for y in intersection_years:
            ts_x.append(values_1[y])
            ts_y.append(values_2[y])
            big_x.append(values_1[y])
            big_y.append(values_2[y])

        muni_corr: float | None = await compute_pearson_correlation(ts_x, ts_y)
        if muni_corr is not None:
            municipality_correlations.append(
                {
                    "municipality_id": m_id,
                    "municipality_name": municipality_map.get(m_id, {}).get(
                        "title", f"Municipality {m_id}"
                    ),
                    "correlation": muni_corr,
                    "years_used": intersection_years,
                    "n_years": len(intersection_years),
                }
            )

    overall_corr: float | None = await compute_pearson_correlation(big_x, big_y)
    result["overall_correlation"] = overall_corr

    municipality_correlations.sort(key=lambda item: item["correlation"])
    n_corr: int = len(municipality_correlations)
    if n_corr == 0:
        await ctx.warning(
            "compare_kpis: No municipality had at least 2 overlapping years for both KPIs."
        )
        return {
            **result,
            "error": "No municipality had 2+ overlapping data points to compute correlation.",
        }

    slice_limit: int = min(10, n_corr)
    median_start: int = max(0, (n_corr - 1) // 2 - (slice_limit // 2))
    median_end: int = min(median_start + slice_limit, n_corr)

    result["municipality_correlations"] = municipality_correlations
    result["top_correlation_municipalities"] = list(
        reversed(municipality_correlations[-slice_limit:])
    )
    result["bottom_correlation_municipalities"] = municipality_correlations[
        :slice_limit
    ]
    result["median_correlation_municipalities"] = municipality_correlations[
        median_start:median_end
    ]

    return result


###############################################################################
# 9) Prompt
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
# 10) Main entry
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
