"""Smoke test for Kolada MCP tools using the live Kolada API.

The script exercises two core behaviours:
1. `search` returns at least one KPI when querying for "meritvärde".
2. `timeseries` returns yearly rows for the first KPI found and municipality 0188 (Stockholm).
"""

import asyncio
from typing import Dict, List

from kolada_mcp.server import kolada_lifespan, search, timeseries


class DummyCtx(dict):
    """Minimal dict-like context for invoking tools directly."""


TEST_QUERY = "meritvärde"
TEST_MUNICIPALITY_ID = "0188"  # Stockholm stad
TEST_YEARS = "2023"


def _summarise_results(results: List[Dict]) -> None:
    """Print a short summary of the returned search rows."""
    kpi_hits = [item for item in results if item.get("type") == "kpi"]
    muni_hits = [item for item in results if item.get("type") == "municipality"]

    print(f"search → total results: {len(results)} (kpi: {len(kpi_hits)}, municipalities: {len(muni_hits)})")
    if kpi_hits:
        sample = kpi_hits[0]
        print(f"  first KPI: {sample['title']}")
    if muni_hits:
        sample = muni_hits[0]
        print(f"  first municipality: {sample['title']}")


def main() -> None:
    async def run() -> None:
        async with kolada_lifespan() as data:
            ctx = DummyCtx(data)

            search_result = await search.fn(query=TEST_QUERY, ctx=ctx)
            results = search_result["results"]
            _summarise_results(results)
            assert results, "Expected at least one search result"

            first_kpi = next(item for item in results if item.get("type") == "kpi")
            series = await timeseries.fn(
                kpi_id=first_kpi["kpi_id"],
                municipality_id=TEST_MUNICIPALITY_ID,
                years=TEST_YEARS,
                ctx=ctx,
            )

            rows = series["rows"]
            latest = series.get("latest")
            print(f"timeseries → rows: {len(rows)}, latest: {latest}")
            assert rows, "Expected at least one timeseries row"
            assert latest, "Expected latest datapoint in timeseries response"

    asyncio.run(run())


if __name__ == "__main__":
    main()
