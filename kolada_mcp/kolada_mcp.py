import json
import os
import statistics  # For median and mean calculation
import sys
import traceback
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Required, TypedDict, cast

import httpx
import numpy as np
import numpy.typing as npt
import polars as pl

# Use the base Context provided by the framework for type hinting in tools
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.server import Context
from sentence_transformers import SentenceTransformer

from kolada_mcp.config import BASE_URL, EMBEDDINGS_CACHE_FILE, KPI_PER_PAGE

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

    id: Required[str]  # The unique identifier for the KPI (e.g., "N00945")
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
    kpi_embeddings: npt.NDArray[np.float32]  # The embeddings for all KPIs
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


def _safe_get_lifespan_context(
    ctx: Context,  # type: ignore[Context]
) -> KoladaLifespanContext | None:
    """
    Safely retrieves the KoladaLifespanContext from the standard context structure.
    Returns None if the context is invalid or incomplete.
    """
    if (
        not ctx
        or not hasattr(ctx, "request_context")  # type: ignore
        or not ctx.request_context  # type: ignore
        or not hasattr(ctx.request_context, "lifespan_context")  # type: ignore
        or not ctx.request_context.lifespan_context  # type: ignore
    ):
        print("[Kolada MCP] Invalid or incomplete context structure.", file=sys.stderr)
        return None
    return cast(KoladaLifespanContext, ctx.request_context.lifespan_context)  # type: ignore


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
    This function processes the KPI data for a given municipality and returns
    a summary of the results. It handles both single-year and multi-year data.
    It also calculates summary statistics and ranks the municipalities based
    on the specified sort order.
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

    k_id: str | None
    kpi_map: dict[str, KoladaKpi] = {}
    for kpi_obj in kpi_list:
        k_id = kpi_obj.get("id")
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
        "KBLab/sentence-bert-swedish-cased"  # type: ignore
    )  # type: ignore
    print("[Kolada MCP] Model loaded.", file=sys.stderr)

    all_kpis: list[KoladaKpi] = [k for k in kpi_list if "id" in k]
    kpi_ids_list: list[str] = []
    titles_list: list[str] = []

    for kpi_obj in all_kpis:
        k_id = kpi_obj["id"]
        title_str: str = kpi_obj.get("title", "")
        kpi_ids_list.append(k_id)
        titles_list.append(title_str)

    # Attempt to load cached .npz
    existing_embeddings: npt.NDArray[np.float32] | None = None
    loaded_ids = []
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
            loaded_ids_arr: npt.NDArray[np.str_] = cache_data.get("kpi_ids", [])
            loaded_ids = loaded_ids_arr.tolist()
        except Exception as ex:
            print(f"[Kolada MCP] Failed to load .npz cache: {ex}", file=sys.stderr)
        if existing_embeddings is None or existing_embeddings.size == 0:
            print(
                "[Kolada MCP] WARNING: No valid embeddings found in cache.",
                file=sys.stderr,
            )
            existing_embeddings = None

    # Check if we can reuse the loaded embeddings
    embeddings: npt.NDArray[np.float32] | None = None
    if (
        existing_embeddings is not None
        and existing_embeddings.size > 0
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
        embeddings = sentence_model.encode(  # type: ignore
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
async def list_operating_areas(ctx: Context) -> list[dict[str, str | int]]:  # type: ignore[Context]
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
        raise RuntimeError(
            "Server context invalid. Unable to retrieve operating areas."
            f" Context: {error_list}"
        )

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
    ctx: Context,  # type: ignore[Context]
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
    ctx: Context,  # type: ignore[Context]
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
    ctx: Context,  # type: ignore[Context]
    limit: int = 20,
) -> list[KoladaKpi]:
    """
    **Purpose:** Performs a semantic search for Kolada Key Performance Indicators (KPIs)
    based on a user-provided keyword or phrase. Instead of simple text matching,
    it uses vector embeddings to find KPIs whose titles are semantically related
    to the search term, even if the exact words don't match. This is useful for
    discovering relevant KPIs when the exact ID, title, or operating area is unknown.

    **Use Cases:**
    *   "Find KPIs related to 'school results'."
    *   "Search for indicators about 'environmental quality'."
    *   "Are there any KPIs measuring 'elderly care satisfaction'?"
    *   "Look up KPIs for 'unemployment rates'."
    *   (Used as a preliminary step before using tools like `analyze_kpi_across_municipalities` or `fetch_kolada_data` if the KPI ID is not known).

    **Arguments:**
    *   `keyword` (str): The search term or phrase describing the topic of interest. The tool will find KPIs with semantically similar titles. **Required.**
    *   `ctx` (Context): The server context (automatically injected by the MCP framework). You do not need to provide this.
    *   `limit` (int, optional): The maximum number of matching KPIs to return, ordered by relevance (highest relevance first). Default is 20.

    **Core Logic:**
    1.  Accesses the pre-loaded data from the server's lifespan context (`lifespan_ctx`), specifically:
        *   The `SentenceTransformer` model (e.g., `KBLab/sentence-bert-swedish-cased`).
        *   The pre-computed `kpi_embeddings` (a NumPy array where each row is the vector embedding of a KPI title).
        *   The list of `kpi_ids` corresponding to the rows in the embeddings array.
        *   The `kpi_map` (dictionary mapping KPI IDs to their full metadata objects).
    2.  Checks if embeddings are available. If not (e.g., failed during startup), returns an empty list.
    3.  **Embeds the User Query:** Takes the input `keyword` string and uses the loaded SentenceTransformer model to convert it into a numerical vector representation (embedding). This captures the semantic meaning of the keyword.
    4.  **Calculates Similarity:** Computes the cosine similarity between the user's query vector and *all* the pre-computed KPI title vectors stored in `kpi_embeddings`. Since the embeddings are pre-normalized during startup, this is efficiently done using a matrix-vector dot product (`embeddings @ query_vec`).
    5.  **Sorts by Relevance:** Sorts the results based on the calculated similarity scores in descending order. The indices of the most similar KPI embeddings are identified.
    6.  **Selects Top N:** Takes the top `limit` indices from the sorted list.
    7.  **Retrieves KPI Metadata:** Uses the top indices to look up the corresponding `kpi_ids` and then retrieves the full `KoladaKpi` metadata objects for those IDs from the `kpi_map`.
    8.  Returns the list of found `KoladaKpi` objects.

    **Return Value:**
    *   A list of `KoladaKpi` dictionaries (containing `id`, `title`, `description`, `operating_area`).
    *   The list is sorted by semantic relevance to the `keyword`, with the most relevant KPI appearing first.
    *   The list contains at most `limit` items.
    *   Returns an empty list (`[]`) if no relevant KPIs are found or if the embeddings cache is unavailable.

    **Important Notes:**
    *   This tool operates entirely on **cached data** loaded at server startup. It does **not** call the live Kolada API.
    *   The search is **semantic**, meaning it looks for related concepts, not just exact word matches. A search for "cars" might find KPIs about "vehicle traffic".
    *   The quality of the search results depends on the chosen SentenceTransformer model and the clarity/informativeness of the cached KPI titles.
    *   It searches primarily based on **KPI titles**. While descriptions are part of the metadata, the embeddings used for the search are generated *only* from the titles for efficiency.
    *   The default `limit` is 20, but can be adjusted if more or fewer results are needed.
    """
    lifespan_ctx: KoladaLifespanContext | None = _safe_get_lifespan_context(ctx)
    if not lifespan_ctx:
        empty_list: list[KoladaKpi] = []
        return empty_list

    # --- Vector-based approach (while keeping the original docstring) ---
    model: SentenceTransformer = lifespan_ctx["sentence_model"]
    embeddings: npt.NDArray[np.float32] = lifespan_ctx["kpi_embeddings"]
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
    query_vector: npt.NDArray[np.float32] = model.encode(  # type: ignore[encode]
        [keyword], normalize_embeddings=True
    )
    query_vec: npt.NDArray[np.float32] = query_vector[0]

    # 2) Compute dot products with normalized embeddings
    #    (We assume embeddings is already normalized)
    sims: npt.NDArray[np.float32] = embeddings @ query_vec

    # 3) Sort descending by similarity
    indices_sorted: npt.NDArray[np.int64] = np.argsort(-sims)
    top_indices: npt.NDArray[np.int64] = indices_sorted[:limit]

    results: list[KoladaKpi] = []
    for idx in top_indices:
        if kpi_ids[idx] in kpi_map:
            results.append(kpi_map[kpi_ids[idx]])

    return results


@mcp.tool()
async def fetch_kolada_data(
    kpi_id: str,
    municipality_id: str,
    ctx: Context,  # type: ignore[Context]
    year: str | None = None,
    municipality_type: str = "K",
) -> dict[str, Any]:
    """
    **Purpose:** Fetches the raw, specific statistical data points for a single
    Kolada Key Performance Indicator (KPI) within a single designated Swedish
    municipality or region. It allows specifying particular years or retrieving
    all available historical data points for that specific KPI/municipality pair.

    **Use Cases:**
    *   "What was the exact value for KPI [ID or name] in [Municipality Name or ID] for the year [YYYY]?"
    *   "Get all historical data points available for KPI [ID] in [Municipality ID]."
    *   "Retrieve the data for [KPI ID] in [Municipality ID] specifically for the years [YYYY1],[YYYY2]."
    *   (Used internally by other tools, but can be called directly if a very specific raw data point is needed).

    **Arguments:**
    *   `kpi_id` (str): The unique identifier of the Kolada KPI whose data is needed (e.g., "N00945"). Use `search_kpis` or `get_kpis_by_operating_area` if you don't have the ID. **Required.**
    *   `municipality_id` (str): The official unique identifier of the specific Swedish municipality or region (e.g., "0180" for Stockholm, "1480" for Göteborg). The server cache contains valid IDs. **Required.**
    *   `ctx` (Context): The server context (automatically injected by the MCP framework). You do not need to provide this.
    *   `year` (str | None, optional): Specifies the year(s) for which to fetch data.
        *   `None` (default): Fetches data for *all* available years for this KPI/municipality combination.
        *   Single Year (e.g., "2023"): Fetches data only for that specific year.
        *   Multiple Years (e.g., "2020,2021,2022"): Fetches data for the specified years.
    *   `municipality_type` (str, optional): Ensures the requested `municipality_id` actually corresponds to the expected type ("K", "R", or "L"). If the ID exists but its type in the server cache doesn't match this parameter, an error is returned *before* calling the Kolada API. Default is "K" (Kommun/Municipality).

    **Core Logic:**
    1.  Retrieves the cached Kolada context (`lifespan_ctx`).
    2.  Validates that `kpi_id` and `municipality_id` are provided.
    3.  Looks up the `municipality_id` in the cached `municipality_map`. If not found, returns an error.
    4.  Checks if the cached type of the `municipality_id` matches the `municipality_type` parameter. If they don't match, returns an error.
    5.  Constructs the specific Kolada API URL targeting the `/v2/data/kpi/{kpi_id}/municipality/{municipality_id}` endpoint. If `year` is provided, it appends `/year/{year}` to the URL.
    6.  Calls the internal `_fetch_data_from_kolada` helper function to make the **live call to the Kolada API**, handling potential errors and pagination (though pagination is less common for this specific endpoint).
    7.  If the API call returns an error, the error dictionary is returned immediately.
    8.  If the API call is successful, it iterates through the returned data points (in the `values` list of the response). For each data point, it attempts to add a `municipality_name` field by looking up the municipality ID (which should be the one requested) in the cached `municipality_map`.
    9.  Returns the dictionary received from Kolada (potentially augmented with `municipality_name` fields).

    **Return Value:**
    A dictionary representing the response from the Kolada API, typically structured as:
    *   `count` (int): The number of main data entries returned (usually 1, representing the municipality).
    *   `values` (list[dict]): A list containing the primary data structure(s). For this endpoint, it's usually a list with one dictionary representing the requested municipality. This dictionary typically contains:
        *   `kpi` (str): The KPI ID.
        *   `municipality` (str): The Municipality ID.
        *   `period` (int): The year for the data points within the 'values' sub-list.
        *   `municipality_name` (str): The human-readable name added by this tool from the cache.
        *   `values` (list[dict]): A sub-list containing the actual data points, often broken down by gender. Each dict here usually has:
            *   `gender` (str): "T", "M", or "K".
            *   `status` (str | None): Data status flag (e.g., None, "B", "M").
            *   `value` (float | int | None): The actual statistical value.
    *   (Other potential keys from Kolada API like `value_types`, etc.)
    *   `error` (str, optional): If an error occurred (e.g., invalid ID provided *to Kolada*, API down, type mismatch detected *before* API call), this key will contain an error message instead of `count` and `values`.

    **Important Notes:**
    *   This tool makes a **live call to the Kolada API** for each execution.
    *   It requires **specific, valid** `kpi_id` and `municipality_id`. Use other tools like `search_kpis` or `analyze_kpi_across_municipalities` if you need to explore or compare data more broadly.
    *   The exact structure of the returned `values` list and its sub-lists depends on the specific KPI and how Kolada structures its data (e.g., whether it includes gender breakdowns).
    *   The `municipality_type` check happens *before* the API call, preventing unnecessary requests if the provided ID is known to be of the wrong type based on the server's cache.
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
    if year:
        url += f"/year/{year}"
        
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
    ctx: Context,  # type: ignore[Context]
    year: str,
    sort_order: str = "desc",
    limit: int = 10,
    gender: str = "T",
    only_return_rate: bool = False,
    municipality_type: str = "K",
) -> dict[str, Any]:
    """
    **Purpose:** Analyzes a single Kolada Key Performance Indicator (KPI) across
    all relevant Swedish municipalities for one or more specified years. It
    provides overall summary statistics (min, max, mean, median) and lists
    of municipalities ranking highest, lowest, and around the median for the
    KPI's value. If multiple years are provided, it also calculates and ranks
    the change (delta) in the KPI value over the specified period for each
    municipality.

    **Use Cases:**
    *   "Which municipalities had the highest [KPI description, e.g., population] in year [YYYY]?"
    *   "Show the lowest performing municipalities for KPI [ID or name] in [YYYY], sorted ascending."
    *   "What are the average, minimum, and maximum values for [KPI description] across all municipalities in [YYYY]?"
    *   "Analyze KPI [ID or name] for the years [YYYY1],[YYYY2]. Which municipalities showed the largest increase?"
    *   "Get only the rate of change statistics for [KPI description] between [YYYY1] and [YYYY2]."
    *   "Compare regions (type R) based on their values for KPI [ID] in [YYYY]."

    **Arguments:**
    *   `kpi_id` (str): The unique identifier of the Kolada KPI to analyze (e.g., "N00945"). Use `search_kpis` or `get_kpis_by_operating_area` if you don't have the ID. **Required.**
    *   `ctx` (Context): The server context (automatically injected by the MCP framework). You do not need to provide this.
    *   `year` (str): Specifies the year(s) for the analysis.
        *   **Single Year:** Provide a single year (e.g., "2022"). Analysis focuses on the values in that year.
        *   **Multiple Years:** Provide a comma-separated list of years (e.g., "2020,2021,2022"). Analysis includes both the latest value within the range and the *change (delta)* between the earliest and latest available year within the range for each municipality.
        **Required.**
    *   `sort_order` (str, optional): Determines the sorting direction for rankings.
        *   "desc": Descending order (highest values first, default).
        *   "asc": Ascending order (lowest values first).
    *   `limit` (int, optional): The maximum number of municipalities to include in the 'top', 'bottom', and 'median' ranking lists (default is 10).
    *   `gender` (str, optional): Filters the data by gender before analysis.
        *   "T": Total (default)
        *   "M": Men
        *   "K": Women
    *   `only_return_rate` (bool, optional): If True **and** multiple years are specified, the returned results will *only* include statistics and rankings related to the *change (delta)* over the period. The statistics and rankings based on the absolute latest value will be omitted. Default is False. Has no effect if only a single year is provided.
    *   `municipality_type` (str, optional): Filters the analysis to include only municipalities of a specific type.
        *   "K": Kommun (Municipality, default)
        *   "R": Region
        *   "L": Landsting (County Council - older term, often equivalent to Region)
        The tool will only include municipalities matching this type in the analysis.

    **Core Logic:**
    1.  Retrieves metadata (title, description, etc.) for the specified `kpi_id` from the server cache.
    2.  Parses the `year` parameter into a list of years.
    3.  Constructs the appropriate URL and fetches the actual data values **from the live Kolada API** for the given `kpi_id` and `year`(s) across all municipalities.
    4.  Processes the raw API response: filters by the specified `gender`, extracts values, and groups them into a structure like `{ municipality_id: { year: value } }`.
    5.  Filters this grouped data to include only municipalities matching the specified `municipality_type`.
    6.  Performs the main analysis (`_process_kpi_data`):
        *   For each included municipality, identifies the value for the latest available year within the requested `year` range (`latest_value`).
        *   If multiple years were requested and data exists for at least two years for a municipality within that range, calculates the `delta_value` (`latest_value` - `earliest_value` in range).
        *   Calculates overall summary statistics (min, max, mean, median, count) across all included municipalities based on their `latest_value`.
        *   If delta values were calculated, calculates overall summary statistics for the `delta_value` across relevant municipalities.
        *   Ranks municipalities based on `latest_value` according to `sort_order` and extracts top, bottom, and median lists based on `limit`.
        *   If delta values were calculated, ranks municipalities based on `delta_value` and extracts top, bottom, and median lists for the delta.
    7.  Constructs and returns a detailed dictionary based on the analysis results and the `only_return_rate` flag.

    **Return Value:**
    A dictionary containing:
    *   `kpi_info` (dict): Metadata (id, title, description, area) for the analyzed KPI.
    *   `selected_years` (list[str]): The list of years used in the analysis.
    *   `selected_gender` (str): The gender filter used.
    *   `sort_order` (str): The sorting order used for rankings.
    *   `limit` (int): The limit used for the size of ranking lists.
    *   `multi_year_delta` (bool): True if multiple years were specified AND delta calculations were possible for at least one municipality.
    *   `only_return_rate` (bool): Reflects the value of the input parameter.
    *   `municipalities_count` (int): The number of municipalities included in the analysis after all filtering (gender, type, data availability).
    *   `summary_stats` (dict): Overall statistics (`min_latest`, `max_latest`, `mean_latest`, `median_latest`, `count`) based on the latest available value for each municipality. **Omitted if `only_return_rate` is True and `multi_year_delta` is True.**
    *   `top_municipalities` (list[dict]): List of municipalities (up to `limit`) with the highest `latest_value` (or lowest if `sort_order`="asc"). Each entry contains `municipality_id`, `municipality_name`, `latest_year`, `latest_value`, potentially `earliest_year`, `earliest_value`, `delta_value`. **Omitted if `only_return_rate` is True and `multi_year_delta` is True.**
    *   `bottom_municipalities` (list[dict]): List of municipalities (up to `limit`) with the lowest `latest_value` (or highest if `sort_order`="asc"). **Omitted if `only_return_rate` is True and `multi_year_delta` is True.**
    *   `median_municipalities` (list[dict]): List of municipalities (up to `limit`) around the median `latest_value`. **Omitted if `only_return_rate` is True and `multi_year_delta` is True.**
    *   `delta_summary_stats` (dict): Overall statistics (`min_delta`, `max_delta`, `mean_delta`, `median_delta`, `count`) based on the calculated change (delta) over the period. **Included only if `multi_year_delta` is True.**
    *   `top_delta_municipalities` (list[dict]): List of municipalities (up to `limit`) with the highest `delta_value` (largest increase, or decrease if `sort_order`="asc"). **Included only if `multi_year_delta` is True.**
    *   `bottom_delta_municipalities` (list[dict]): List of municipalities (up to `limit`) with the lowest `delta_value` (largest decrease, or increase if `sort_order`="asc"). **Included only if `multi_year_delta` is True.**
    *   `median_delta_municipalities` (list[dict]): List of municipalities (up to `limit`) around the median `delta_value`. **Included only if `multi_year_delta` is True.**
    *   `error` (str, optional): If an error occurred (e.g., API fetch failed, no data found for the parameters), this key will contain an error message.

    **Important Notes:**
    *   This tool makes a **live call to the Kolada API** to fetch the raw data, which might take some time depending on the KPI and number of years requested.
    *   The results depend heavily on data availability within Kolada for the specific KPI, years, gender, and municipality type. If no data matching the criteria is found, the counts will be zero and rankings empty.
    *   The `delta_value` and associated statistics/rankings are only calculated and returned (`multi_year_delta`=True) if multiple years are specified *and* at least one municipality has data for two or more years *within the requested range*.
    *   Using `only_return_rate=True` can simplify the output when you are specifically interested in the *change* over time rather than the absolute latest values.
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
    ctx: Context,  # type: ignore[Context]
    gender: str = "T",
    municipality_type: str = "K",
) -> dict[str, Any]:
    """
    **Purpose:** Compares two different Kolada Key Performance Indicators (KPIs)
    across Swedish municipalities for one or more specified years. It calculates
    either the difference or the correlation between the KPIs, depending on
    whether a single year or multiple years are provided.

    **Use Cases:**
    *   "How does KPI [A] correlate with KPI [B] across municipalities in year [YYYY]?"
    *   "Compare KPI [X] and KPI [Y] for the years [YYYY1],[YYYY2]. Which municipalities show the strongest positive/negative correlation?"
    *   "For year [YYYY], which municipalities have the largest difference between KPI [A] and KPI [B]?"
    *   "Analyze the relationship between [topic 1, e.g., unemployment] and [topic 2, e.g., education level] across municipalities over the period [YYYY1]-[YYYY2]." (Requires finding relevant KPI IDs first using `search_kpis`).

    **Arguments:**
    *   `kpi1_id` (str): The unique identifier of the first Kolada KPI (e.g., "N00945"). Use `search_kpis` or `get_kpis_by_operating_area` if you don't have the ID. **Required.**
    *   `kpi2_id` (str): The unique identifier of the second Kolada KPI. **Required.**
    *   `year` (str): Specifies the year(s) for the comparison.
        *   **Single Year:** Provide a single year (e.g., "2022"). The tool will calculate the *difference* (KPI2 - KPI1) for each municipality.
        *   **Multiple Years:** Provide a comma-separated list of years (e.g., "2020,2021,2022"). The tool will calculate the *Pearson correlation* between the time series of the two KPIs *within each municipality* that has data for at least two overlapping years.
        **Required.**
    *   `ctx` (Context): The server context (automatically injected by the MCP framework). You do not need to provide this.
    *   `gender` (str, optional): Filters the data by gender before comparison.
        *   "T": Total (default)
        *   "M": Men
        *   "K": Women
    *   `municipality_type` (str, optional): Filters the comparison to include only municipalities of a specific type.
        *   "K": Kommun (Municipality, default)
        *   "R": Region
        *   "L": Landsting (County Council - older term, often equivalent to Region)
        The tool will only include municipalities matching this type in the analysis.

    **Core Logic:**
    1.  Retrieves metadata for both `kpi1_id` and `kpi2_id` from the server cache.
    2.  Parses the `year` parameter to determine if it's a single-year or multi-year analysis.
    3.  Fetches the actual data values **from the live Kolada API** for BOTH `kpi1_id` and `kpi2_id` for the specified `year`(s).
    4.  Filters the fetched data based on the requested `gender` and `municipality_type`.
    5.  Identifies municipalities that have data for *both* KPIs for the relevant year(s).
    6.  **If Single Year:**
        *   Calculates the difference (`kpi2_value - kpi1_value`) for each common municipality.
        *   Calculates the overall Pearson correlation between the two KPIs across all common municipalities for that single year.
        *   Ranks municipalities based on the calculated difference.
    7.  **If Multiple Years:**
        *   For each common municipality with at least two overlapping data points, calculates the Pearson correlation between the time series of KPI1 and KPI2.
        *   Calculates the overall Pearson correlation using *all* available (municipality, year) data points combined.
        *   Ranks municipalities based on their individual time-series correlations.
    8.  Constructs and returns a detailed dictionary with the results.

    **Return Value:**
    A dictionary containing:
    *   `kpi1_info` (dict): Metadata (id, title, description, area) for the first KPI.
    *   `kpi2_info` (dict): Metadata for the second KPI.
    *   `selected_years` (list[str]): The list of years used in the analysis.
    *   `gender` (str): The gender filter used.
    *   `municipality_type` (str): The municipality type filter used.
    *   `multi_year` (bool): True if multiple years were analyzed, False otherwise.
    *   `overall_correlation` (float | None): The Pearson correlation coefficient calculated across all data points (either all municipalities in a single year, or all municipality-year pairs in a multi-year analysis). Can be `None` if insufficient data exists.
    *   **If `multi_year` is False (Single Year Analysis):**
        *   `municipality_differences` (list[dict]): A list of dictionaries, one per municipality with data, containing `municipality_id`, `municipality_name`, `kpi1_value`, `kpi2_value`, and `difference`. Sorted by difference.
        *   `top_difference_municipalities` (list[dict]): Top N municipalities with the largest positive difference (KPI2 > KPI1).
        *   `bottom_difference_municipalities` (list[dict]): Top N municipalities with the largest negative difference (KPI1 > KPI2).
        *   `median_difference_municipalities` (list[dict]): N municipalities around the median difference.
    *   **If `multi_year` is True (Multi-Year Analysis):**
        *   `municipality_correlations` (list[dict]): A list of dictionaries, one per municipality with sufficient data, containing `municipality_id`, `municipality_name`, `correlation` (the within-municipality time-series correlation), `years_used`, and `n_years`. Sorted by correlation.
        *   `top_correlation_municipalities` (list[dict]): Top N municipalities with the highest positive correlation.
        *   `bottom_correlation_municipalities` (list[dict]): Top N municipalities with the lowest (most negative) correlation.
        *   `median_correlation_municipalities` (list[dict]): N municipalities around the median correlation.
    *   `error` (str, optional): If an error occurred (e.g., API fetch failed, no overlapping data found), this key will contain an error message.

    **Important Notes:**
    *   This tool makes **live calls to the Kolada API** (potentially two separate calls for the data), which might take some time.
    *   Ensure you provide valid `kpi1_id` and `kpi2_id`.
    *   The analysis depends on data availability in Kolada for the selected KPIs, years, gender, and municipalities. Lack of overlapping data can lead to empty results or `None` for correlations.
    *   For multi-year analysis, a municipality is only included in the `municipality_correlations` list if it has data for *both* KPIs in at least *two* common years within the specified range.
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
