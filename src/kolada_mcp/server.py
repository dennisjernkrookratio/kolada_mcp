"""Kolada MCP server built on top of koladapy."""
from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, Iterable, List, Optional

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from koladapy.api import KoladaAPI
from koladapy.exceptions import KoladaAPIError

# Avoid tqdm progress bars when koladapy performs paginated requests
os.environ.setdefault("TQDM_DISABLE", "1")


def _parse_years_spec(spec: Optional[str]) -> Optional[List[int]]:
    """Parse user provided year spec into a list of integers."""
    if not spec:
        return None

    years: List[int] = []
    for token in (part.strip() for part in spec.split(",")):
        if not token:
            continue
        if "-" in token:
            start_str, end_str = (t.strip() for t in token.split("-", 1))
            if not start_str.isdigit() or not end_str.isdigit():
                raise ValueError(f"Invalid year range: {token}")
            start, end = int(start_str), int(end_str)
            if start > end:
                raise ValueError(f"Year range start must be <= end ({token})")
            years.extend(range(start, end + 1))
        else:
            if not token.isdigit():
                raise ValueError(f"Invalid year token: {token}")
            years.append(int(token))

    # Remove duplicates while preserving order
    seen = set()
    deduped: List[int] = []
    for year in years:
        if year not in seen:
            seen.add(year)
            deduped.append(year)

    return deduped or None


def _truncate(text: str, limit: int = 240) -> str:
    """Return a shortened version of text for summary fields."""
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


async def _to_thread(func, *args, **kwargs):
    """Run a blocking koladapy function in a worker thread."""
    return await asyncio.to_thread(func, *args, **kwargs)


@asynccontextmanager
async def kolada_lifespan(_mcp: FastMCP):
    """Create and clean up the Kolada API client."""
    api = KoladaAPI()
    try:
        yield {"api": api}
    finally:
        api.session.close()


server_instructions = (
    "Use the search tool to discover Kolada KPIs and municipalities. "
    "Then call the timeseries tool with the KPI id and municipality id to fetch yearly values."
)

mcp = FastMCP(
    name="Kolada MCP",
    instructions=server_instructions,
    lifespan=kolada_lifespan,
)


def _format_kpi_result(kpi: Dict[str, Any]) -> Dict[str, Any]:
    description = kpi.get("description") or ""
    return {
        "id": f"kpi:{kpi['id']}",
        "title": f"{kpi['title']} ({kpi['id']})",
        "text": _truncate(description) if description else "KPI in Kolada",
        "type": "kpi",
        "kpi_id": kpi["id"],
        "municipality_type": kpi.get("municipality_type"),
        "operating_area": kpi.get("operating_area"),
        "url": f"https://www.kolada.se/verktyg/fri-sokning/?kpis={kpi['id']}",
    }


def _format_municipality_result(municipality: Dict[str, Any]) -> Dict[str, Any]:
    muni_type = municipality.get("type") or ""
    muni_label = {
        "K": "Kommun",
        "L": "Region",
    }.get(muni_type, muni_type or "Municipality")
    return {
        "id": f"municipality:{municipality['id']}",
        "title": f"{municipality['title']} ({municipality['id']})",
        "text": f"{muni_label} listed in Kolada",
        "type": "municipality",
        "municipality_id": municipality["id"],
        "municipality_type": muni_type,
        "url": f"https://www.kolada.se/kommun/?municipality={municipality['id']}",
    }


@mcp.tool()
async def search(query: str, ctx: Context = None) -> Dict[str, List[Dict[str, Any]]]:
    """Search Kolada for KPIs and municipalities."""
    if not query or not query.strip():
        return {"results": []}

    if ctx is None:
        raise ToolError("Context unavailable")

    if ctx is None:
        raise ToolError("Context unavailable")

    api: KoladaAPI = ctx["api"]

    try:
        kpi_task = asyncio.create_task(
            _to_thread(api.search_kpis, query=query, as_dataframe=False)
        )
        muni_task = asyncio.create_task(
            _to_thread(api.get_municipalities, query=query, as_dataframe=False)
        )
        kpis, municipalities = await asyncio.gather(kpi_task, muni_task)
    except KoladaAPIError as exc:
        raise ToolError(f"Kolada API search failed: {exc}") from exc

    results: List[Dict[str, Any]] = []
    for kpi in (kpis or [])[:8]:
        results.append(_format_kpi_result(kpi))

    for municipality in (municipalities or [])[:8]:
        results.append(_format_municipality_result(municipality))

    return {"results": results}


def _extract_rows(entries: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for entry in entries:
        period = entry.get("period")
        if period is None:
            continue
        totals = next(
            (value for value in entry.get("values", []) if value.get("gender") == "T"),
            None,
        )
        if not totals:
            continue
        value = totals.get("value")
        if value is None:
            continue
        rows.append({"year": period, "value": value})

    rows.sort(key=lambda item: item["year"])
    return rows


@mcp.tool()
async def timeseries(
    kpi_id: str,
    municipality_id: str,
    years: Optional[str] = None,
    ctx: Context = None,
) -> Dict[str, Any]:
    """Fetch yearly KPI values for a municipality."""

    api: KoladaAPI = ctx["api"]

    try:
        year_list = _parse_years_spec(years)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc

    try:
        kpi_task = asyncio.create_task(_to_thread(api.get_kpi, kpi_id))
        municipality_task = asyncio.create_task(
            _to_thread(api.get_municipality, municipality_id)
        )
        values_task = asyncio.create_task(
            _to_thread(
                api.get_values,
                kpi_id=kpi_id,
                municipality_id=municipality_id,
                years=year_list,
            )
        )
        kpi, municipality, entries = await asyncio.gather(
            kpi_task, municipality_task, values_task
        )
    except KoladaAPIError as exc:
        raise ToolError(f"Kolada request failed: {exc}") from exc

    rows = _extract_rows(entries)
    latest = rows[-1] if rows else None

    return {
        "kpi": {
            "id": kpi.get("id"),
            "title": kpi.get("title"),
            "description": kpi.get("description"),
            "operating_area": kpi.get("operating_area"),
            "municipality_type": kpi.get("municipality_type"),
        },
        "municipality": {
            "id": municipality.get("id"),
            "name": municipality.get("title"),
            "type": municipality.get("type"),
        },
        "query": {
            "kpi_id": kpi_id,
            "municipality_id": municipality_id,
            "years": year_list,
        },
        "rows": rows,
        "latest": latest,
        "source_url": f"https://www.kolada.se/verktyg/fri-sokning/?kpis={kpi_id}",
    }


def main() -> None:
    """Run the Kolada MCP server using SSE or stdio transport."""

    transport = os.environ.get("MCP_TRANSPORT", "sse").lower().strip()

    if transport == "stdio":
        mcp.run()
        return

    port = int(os.environ.get("PORT", "8000"))
    mcp.run(transport="sse", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
