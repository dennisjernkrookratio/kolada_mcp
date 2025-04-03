import json
import os
from typing import Any, AsyncIterator, TypedDict, cast
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from mcp.server.fastmcp.server import Context

from kolada_mcp.config import BASE_URL


class MockLifespanContext(TypedDict):
    """Mock version of KoladaLifespanContext for testing"""
    kpi_cache: list[dict[str, Any]]
    kpi_map: dict[str, dict[str, Any]]
    operating_areas_summary: list[dict[str, str | int]]
    municipality_cache: list[dict[str, Any]]
    municipality_map: dict[str, dict[str, Any]]
    sentence_model: Any
    kpi_embeddings: np.ndarray
    kpi_ids: list[str]


@pytest.fixture
def mock_context() -> Context:
    """
    Creates a mock Context object with a populated lifespan_context
    that mimics the structure expected by the Kolada MCP tools.
    """
    # Create sample KPIs
    kpi1 = {
        "id": "N00945",
        "title": "Population",
        "description": "Total population count",
        "operating_area": "Demographics",
    }
    kpi2 = {
        "id": "N00946",
        "title": "Population density",
        "description": "Population per square km",
        "operating_area": "Demographics",
    }
    kpi3 = {
        "id": "N07403",
        "title": "Unemployment rate",
        "description": "Percentage of working age population unemployed",
        "operating_area": "Economy",
    }
    
    kpi_list = [kpi1, kpi2, kpi3]
    kpi_map = {kpi["id"]: kpi for kpi in kpi_list}
    
    # Create sample municipalities
    muni1 = {"id": "0180", "title": "Stockholm", "type": "K"}
    muni2 = {"id": "1480", "title": "Göteborg", "type": "K"}
    muni3 = {"id": "0126", "title": "Huddinge", "type": "K"}
    muni4 = {"id": "01", "title": "Stockholm Region", "type": "R"}
    
    muni_list = [muni1, muni2, muni3, muni4]
    muni_map = {muni["id"]: muni for muni in muni_list}
    
    # Create operating areas summary
    areas_summary = [
        {"operating_area": "Demographics", "kpi_count": 2},
        {"operating_area": "Economy", "kpi_count": 1},
    ]
    
    # Create mock embeddings
    mock_embeddings = np.array([
        [0.1, 0.2, 0.3],  # N00945
        [0.2, 0.3, 0.4],  # N00946
        [0.5, 0.6, 0.7],  # N07403
    ], dtype=np.float32)
    
    # Create mock sentence model
    mock_model = MagicMock()
    mock_model.encode = MagicMock(return_value=np.array([[0.3, 0.4, 0.5]]))
    
    # Create the lifespan context
    lifespan_context: MockLifespanContext = {
        "kpi_cache": kpi_list,
        "kpi_map": kpi_map,
        "operating_areas_summary": areas_summary,
        "municipality_cache": muni_list,
        "municipality_map": muni_map,
        "sentence_model": mock_model,
        "kpi_embeddings": mock_embeddings,
        "kpi_ids": ["N00945", "N00946", "N07403"],
    }
    
    # Create the request context
    request_context = MagicMock()
    request_context.lifespan_context = lifespan_context
    
    # Create the main context
    context = MagicMock(spec=Context)
    context.request_context = request_context
    
    return cast(Context, context)


@pytest.fixture
def mock_httpx_client():
    """
    Creates a mock httpx.AsyncClient that returns predefined responses
    for different Kolada API endpoints.
    """
    mock_client = AsyncMock()
    
    # Define response for KPI data
    kpi_response = MagicMock()
    kpi_response.status_code = 200
    kpi_response.json.return_value = {
        "count": 2,
        "values": [
            {
                "kpi": "N00945",
                "municipality": "0180",
                "period": 2022,
                "values": [
                    {"gender": "T", "value": 978770},
                    {"gender": "M", "value": 485000},
                    {"gender": "K", "value": 493770},
                ]
            },
            {
                "kpi": "N00945",
                "municipality": "1480",
                "period": 2022,
                "values": [
                    {"gender": "T", "value": 587000},
                    {"gender": "M", "value": 293500},
                    {"gender": "K", "value": 293500},
                ]
            }
        ]
    }
    
    # Define response for municipality data
    muni_response = MagicMock()
    muni_response.status_code = 200
    muni_response.json.return_value = {
        "count": 2,
        "values": [
            {"id": "0180", "title": "Stockholm", "type": "K"},
            {"id": "1480", "title": "Göteborg", "type": "K"}
        ]
    }
    
    # Configure the mock client to return different responses based on URL
    async def get_side_effect(url, **kwargs):
        if f"{BASE_URL}/data/kpi" in url:
            return kpi_response
        elif f"{BASE_URL}/municipality" in url:
            return muni_response
        else:
            # Default response for other URLs
            default_response = MagicMock()
            default_response.status_code = 200
            default_response.json.return_value = {"count": 0, "values": []}
            return default_response
    
    mock_client.get.side_effect = get_side_effect
    
    return mock_client


@pytest.fixture
def mock_sentence_transformer():
    """
    Creates a mock SentenceTransformer that returns predefined embeddings.
    """
    mock_model = MagicMock()
    
    def encode_side_effect(texts, **kwargs):
        # Return different embeddings based on input text
        if isinstance(texts, list) and len(texts) > 0:
            if "population" in texts[0].lower():
                return np.array([[0.1, 0.2, 0.3]])
            elif "economy" in texts[0].lower():
                return np.array([[0.4, 0.5, 0.6]])
            else:
                return np.array([[0.7, 0.8, 0.9]])
        return np.array([[0.0, 0.0, 0.0]])
    
    mock_model.encode.side_effect = encode_side_effect
    return mock_model
