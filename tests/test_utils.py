import statistics
from typing import Any

import numpy as np
import pytest

from kolada_mcp.kolada_mcp import (
    _calculate_summary_stats,
    _group_kpis_by_operating_area,
    _parse_years_param,
    _rank_and_slice_municipalities,
)


class TestParseYearsParam:
    def test_empty_string(self):
        assert _parse_years_param("") == []
    
    def test_single_year(self):
        assert _parse_years_param("2022") == ["2022"]
    
    def test_multiple_years(self):
        assert _parse_years_param("2020,2021,2022") == ["2020", "2021", "2022"]
    
    def test_whitespace_handling(self):
        assert _parse_years_param(" 2020, 2021 , 2022 ") == ["2020", "2021", "2022"]
    
    def test_empty_parts_filtered(self):
        assert _parse_years_param("2020,,2022") == ["2020", "2022"]


class TestCalculateSummaryStats:
    def test_empty_list(self):
        result = _calculate_summary_stats([])
        assert result["min"] is None
        assert result["max"] is None
        assert result["mean"] is None
        assert result["median"] is None
        assert result["count"] == 0
    
    def test_single_value(self):
        result = _calculate_summary_stats([5.0])
        assert result["min"] == 5.0
        assert result["max"] == 5.0
        assert result["mean"] == 5.0
        assert result["median"] == 5.0
        assert result["count"] == 1
    
    def test_multiple_values(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = _calculate_summary_stats(values)
        assert result["min"] == 1.0
        assert result["max"] == 5.0
        assert result["mean"] == 3.0
        assert result["median"] == 3.0
        assert result["count"] == 5
    
    def test_with_prefix(self):
        values = [1.0, 2.0, 3.0]
        result = _calculate_summary_stats(values, prefix="delta_")
        assert result["delta_min"] == 1.0
        assert result["delta_max"] == 3.0
        assert result["delta_mean"] == 2.0
        assert result["delta_median"] == 2.0
        assert result["count"] == 3  # Note: count doesn't get prefix


class TestGroupKpisByOperatingArea:
    def test_empty_list(self):
        assert _group_kpis_by_operating_area([]) == {}
    
    def test_single_kpi_single_area(self):
        kpis = [{"id": "N00945", "operating_area": "Demographics"}]
        result = _group_kpis_by_operating_area(kpis)
        assert len(result) == 1
        assert "Demographics" in result
        assert len(result["Demographics"]) == 1
        assert result["Demographics"][0]["id"] == "N00945"
    
    def test_multiple_kpis_multiple_areas(self):
        kpis = [
            {"id": "N00945", "operating_area": "Demographics"},
            {"id": "N00946", "operating_area": "Demographics"},
            {"id": "N07403", "operating_area": "Economy"},
        ]
        result = _group_kpis_by_operating_area(kpis)
        assert len(result) == 2
        assert "Demographics" in result
        assert "Economy" in result
        assert len(result["Demographics"]) == 2
        assert len(result["Economy"]) == 1
    
    def test_kpi_with_multiple_areas(self):
        kpis = [{"id": "N12345", "operating_area": "Economy,Environment"}]
        result = _group_kpis_by_operating_area(kpis)
        assert len(result) == 2
        assert "Economy" in result
        assert "Environment" in result
        assert len(result["Economy"]) == 1
        assert len(result["Environment"]) == 1
    
    def test_kpi_with_missing_area(self):
        kpis = [{"id": "N12345"}]  # No operating_area field
        result = _group_kpis_by_operating_area(kpis)
        assert len(result) == 1
        assert "Unknown" in result
        assert len(result["Unknown"]) == 1
    
    def test_kpi_with_empty_area(self):
        kpis = [{"id": "N12345", "operating_area": ""}]
        result = _group_kpis_by_operating_area(kpis)
        assert result == {}


class TestRankAndSliceMunicipalities:
    def test_empty_list(self):
        top, bottom, median = _rank_and_slice_municipalities([], "value", "desc", 5)
        assert top == []
        assert bottom == []
        assert median == []
    
    def test_single_item(self):
        data = [{"municipality_id": "0180", "value": 100}]
        top, bottom, median = _rank_and_slice_municipalities(data, "value", "desc", 5)
        assert len(top) == 1
        assert len(bottom) == 1
        assert len(median) == 1
        assert top[0]["value"] == 100
        assert bottom[0]["value"] == 100
        assert median[0]["value"] == 100
    
    def test_multiple_items_desc_order(self):
        data = [
            {"municipality_id": "0180", "value": 100},
            {"municipality_id": "1480", "value": 200},
            {"municipality_id": "0126", "value": 150},
            {"municipality_id": "0127", "value": 50},
            {"municipality_id": "0128", "value": 75},
        ]
        top, bottom, median = _rank_and_slice_municipalities(data, "value", "desc", 2)
        
        # Top should have highest values first
        assert len(top) == 2
        assert top[0]["value"] == 200
        assert top[1]["value"] == 150
        
        # Bottom should have lowest values first
        assert len(bottom) == 2
        assert bottom[0]["value"] == 50
        assert bottom[1]["value"] == 75
        
        # Median should have values around the middle
        assert len(median) == 2
        assert median[0]["value"] == 100  # The median value
    
    def test_multiple_items_asc_order(self):
        data = [
            {"municipality_id": "0180", "value": 100},
            {"municipality_id": "1480", "value": 200},
            {"municipality_id": "0126", "value": 150},
            {"municipality_id": "0127", "value": 50},
            {"municipality_id": "0128", "value": 75},
        ]
        top, bottom, median = _rank_and_slice_municipalities(data, "value", "asc", 2)
        
        # Top should have lowest values first (ascending)
        assert len(top) == 2
        assert top[0]["value"] == 50
        assert top[1]["value"] == 75
        
        # Bottom should have highest values first (ascending)
        assert len(bottom) == 2
        assert bottom[0]["value"] == 200
        assert bottom[1]["value"] == 150
        
        # Median should have values around the middle
        assert len(median) == 2
        assert median[0]["value"] == 100  # The median value
    
    def test_limit_larger_than_data(self):
        data = [
            {"municipality_id": "0180", "value": 100},
            {"municipality_id": "1480", "value": 200},
        ]
        top, bottom, median = _rank_and_slice_municipalities(data, "value", "desc", 5)
        assert len(top) == 2
        assert len(bottom) == 2
        assert len(median) == 2
