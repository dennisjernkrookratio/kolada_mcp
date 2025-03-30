#!/usr/bin/env python3
from typing import Any

import httpx
import yaml


def fetch_all_kpis(
    base_url: str = "https://api.kolada.se/v2/kpi", per_page: int = 5000
) -> list[dict[str, Any]]:
    """
    Fetch all KPI metadata from Kolada, handling pagination.
    Returns a list of KPI objects.
    """
    all_kpis: list[dict[str, Any]] = []
    next_url: str | None = f"{base_url}?per_page={per_page}"

    # Use a persistent httpx Client for efficiency
    with httpx.Client() as client:
        while next_url:
            response = client.get(next_url)
            response.raise_for_status()
            data: dict[str, Any] = response.json()

            kpis: list[dict[str, Any]] = data.get("values", [])
            all_kpis.extend(kpis)

            next_url = data.get("next_page")

    return all_kpis


def group_kpis_by_operating_area(
    kpis: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """
    Group the KPI list by their 'operating_area' field.
    Returns a dictionary where the key is the operating area and the value
    is a list of KPI dictionaries.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for kpi in kpis:
        operating_area: str = kpi.get("operating_area") or "Unknown"
        grouped.setdefault(operating_area, []).append(kpi)
    return grouped


def get_operating_areas_counts(kpis: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Return a list of dictionaries, where each dictionary contains:
      - 'operating_area'
      - 'count' (number of KPIs in that area)
    """
    grouped = group_kpis_by_operating_area(kpis)
    areas_with_counts: list[dict[str, Any]] = []
    for area, kpis_in_area in grouped.items():
        areas_with_counts.append({"operating_area": area, "count": len(kpis_in_area)})
    return areas_with_counts


def get_operating_areas_yaml(kpis: list[dict[str, Any]]) -> str:
    """
    Return a YAML string listing each operating area and the count of KPIs in that area.
    """
    areas_with_counts = get_operating_areas_counts(kpis)
    yaml_output: str = yaml.dump(
        areas_with_counts, sort_keys=False, default_flow_style=False, allow_unicode=True
    )
    return yaml_output


def filter_kpis_by_operating_areas(
    kpis: list[dict[str, Any]],
    operating_areas: list[str],
) -> list[dict[str, Any]]:
    """
    Given a list of KPIs and a list of operating areas, return only the KPIs
    whose 'operating_area' field matches (or contains) any of those areas.

    NOTE: The Kolada data sometimes has multiple areas in a single string
          (e.g. "Allmän regional utveckling,Regional utveckling").
          Here we split by comma and strip spaces to handle partial matches.
    """
    # Convert user-supplied list to a set for faster membership checks.
    areas_set = set(operating_areas)

    filtered_kpis: list[dict[str, Any]] = []
    for kpi in kpis:
        area_field = kpi.get("operating_area", "Unknown")
        # Split by comma to handle multiple areas in one field
        area_list = [a.strip() for a in area_field.split(",")]
        # Check if there's any overlap between the set of desired areas and the KPI's area list
        if areas_set.intersection(area_list):
            filtered_kpis.append(kpi)

    return filtered_kpis


def main() -> None:
    # 1. Fetch all KPI metadata
    kpis = fetch_all_kpis()

    # 2. Output a YAML list of all operating areas and the count of KPIs in each
    areas_yaml = get_operating_areas_yaml(kpis)
    print("Operating areas (with KPI counts) as YAML:\n")
    print(areas_yaml)


if __name__ == "__main__":
    main()
