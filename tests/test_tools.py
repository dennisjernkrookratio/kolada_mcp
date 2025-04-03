import json
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from kolada_mcp.kolada_mcp import (
    analyze_kpi_across_municipalities,
    compare_kpis,
    fetch_kolada_data,
    get_kpi_metadata,
    get_kpis_by_operating_area,
    list_operating_areas,
    search_kpis,
)


@pytest.mark.asyncio
async def test_list_operating_areas(mock_context):
    """Test listing operating areas from the context"""
    result = await list_operating_areas(mock_context)
    
    # Verify the result matches what's in the mock context
    assert len(result) == 2
    assert {"operating_area": "Demographics", "kpi_count": 2} in result
    assert {"operating_area": "Economy", "kpi_count": 1} in result


@pytest.mark.asyncio
async def test_list_operating_areas_invalid_context():
    """Test handling of invalid context"""
    invalid_context = MagicMock()
    invalid_context.request_context = None
    
    with pytest.raises(RuntimeError) as excinfo:
        await list_operating_areas(invalid_context)
    
    assert "Server context invalid" in str(excinfo.value)


@pytest.mark.asyncio
async def test_get_kpis_by_operating_area(mock_context):
    """Test getting KPIs by operating area"""
    # Test with a valid operating area
    result = await get_kpis_by_operating_area("Demographics", mock_context)
    assert len(result) == 2
    assert result[0]["id"] == "N00945"
    assert result[1]["id"] == "N00946"
    
    # Test with a different operating area
    result = await get_kpis_by_operating_area("Economy", mock_context)
    assert len(result) == 1
    assert result[0]["id"] == "N07403"
    
    # Test with non-existent operating area
    result = await get_kpis_by_operating_area("NonExistent", mock_context)
    assert len(result) == 0


@pytest.mark.asyncio
async def test_get_kpi_metadata(mock_context):
    """Test getting KPI metadata by ID"""
    # Test with a valid KPI ID
    result = await get_kpi_metadata("N00945", mock_context)
    assert result["id"] == "N00945"
    assert result["title"] == "Population"
    assert result["operating_area"] == "Demographics"
    
    # Test with non-existent KPI ID
    result = await get_kpi_metadata("NonExistent", mock_context)
    assert "error" in result
    assert "No KPI metadata found" in result["error"]


@pytest.mark.asyncio
async def test_search_kpis(mock_context):
    """Test searching KPIs by keyword"""
    # Set up the dot product result for similarity calculation
    # This simulates the result of embeddings @ query_vec
    with patch("numpy.ndarray.__matmul__", return_value=np.array([0.8, 0.6, 0.3])):
        result = await search_kpis("population", mock_context, limit=2)
        
        # Verify the results are sorted by similarity (highest first)
        assert len(result) == 2
        assert result[0]["id"] == "N00945"  # Highest similarity
        assert result[1]["id"] == "N00946"  # Second highest


@pytest.mark.asyncio
async def test_search_kpis_empty_embeddings(mock_context):
    """Test search behavior with empty embeddings"""
    # Modify the mock context to have empty embeddings
    mock_context.request_context.lifespan_context["kpi_embeddings"] = np.array([], dtype=np.float32).reshape(0, 3)
    
    result = await search_kpis("population", mock_context)
    assert result == []


@pytest.mark.asyncio
async def test_fetch_kolada_data(mock_context, mock_httpx_client):
    """Test fetching data from Kolada API"""
    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        result = await fetch_kolada_data("N00945", "0180", mock_context)
        
        # Verify the result structure
        assert "count" in result
        assert "values" in result
        assert len(result["values"]) > 0
        
        # Check that municipality names were added
        assert "municipality_name" in result["values"][0]
        assert result["values"][0]["municipality_name"] == "Stockholm"


@pytest.mark.asyncio
async def test_fetch_kolada_data_invalid_municipality(mock_context):
    """Test handling of invalid municipality ID"""
    result = await fetch_kolada_data("N00945", "NonExistent", mock_context)
    
    assert "error" in result
    assert "Municipality ID 'NonExistent' not found" in result["error"]


@pytest.mark.asyncio
async def test_fetch_kolada_data_type_mismatch(mock_context):
    """Test handling of municipality type mismatch"""
    # Try to fetch data for a municipality but specify region type
    result = await fetch_kolada_data("N00945", "0180", mock_context, municipality_type="R")
    
    assert "error" in result
    assert "Municipality '0180' is type 'K'" in result["error"]


@pytest.mark.asyncio
async def test_analyze_kpi_across_municipalities(mock_context, mock_httpx_client):
    """Test analyzing KPI across municipalities"""
    # Mock the get_kpi_metadata function
    with patch("kolada_mcp.kolada_mcp.get_kpi_metadata", return_value={"id": "N00945", "title": "Population"}):
        # Mock the httpx client
        with patch("httpx.AsyncClient", return_value=mock_httpx_client):
            result = await analyze_kpi_across_municipalities(
                "N00945", mock_context, year="2022", limit=5
            )
            
            # Verify the result structure
            assert "kpi_info" in result
            assert result["kpi_info"]["id"] == "N00945"
            assert "summary_stats" in result
            assert "top_municipalities" in result
            assert "bottom_municipalities" in result
            assert "median_municipalities" in result


@pytest.mark.asyncio
async def test_compare_kpis(mock_context, mock_httpx_client):
    """Test comparing two KPIs"""
    # Mock the get_kpi_metadata function to return different metadata for different KPIs
    async def mock_get_kpi_metadata(kpi_id, ctx):
        if kpi_id == "N00945":
            return {"id": "N00945", "title": "Population"}
        else:
            return {"id": "N07403", "title": "Unemployment rate"}
    
    with patch("kolada_mcp.kolada_mcp.get_kpi_metadata", side_effect=mock_get_kpi_metadata):
        # Mock the httpx client
        with patch("httpx.AsyncClient", return_value=mock_httpx_client):
            # Mock the compute_pearson_correlation function
            with patch("kolada_mcp.kolada_mcp.compute_pearson_correlation", return_value=0.75):
                result = await compare_kpis(
                    "N00945", "N07403", "2022", mock_context
                )
                
                # Verify the result structure
                assert "kpi1_info" in result
                assert result["kpi1_info"]["id"] == "N00945"
                assert "kpi2_info" in result
                assert result["kpi2_info"]["id"] == "N07403"
                assert "overall_correlation" in result
