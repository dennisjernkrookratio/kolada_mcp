import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from kolada_mcp.config import BASE_URL
from kolada_mcp.kolada_mcp import _fetch_data_from_kolada


@pytest.mark.asyncio
async def test_fetch_data_success(mock_httpx_client):
    """Test successful data fetching from Kolada API"""
    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        url = f"{BASE_URL}/data/kpi/N00945/municipality/0180"
        result = await _fetch_data_from_kolada(url)
        
        # Verify the client was called with the correct URL
        mock_httpx_client.get.assert_called_with(url, timeout=60.0)
        
        # Check the result structure
        assert "count" in result
        assert "values" in result
        assert isinstance(result["values"], list)


@pytest.mark.asyncio
async def test_fetch_data_with_pagination():
    """Test fetching data with pagination handling"""
    # Create a mock client that returns paginated responses
    mock_client = AsyncMock()
    
    # First response with a next_page link
    first_response = MagicMock()
    first_response.status_code = 200
    first_response.json.return_value = {
        "count": 2,
        "values": [{"id": "item1"}, {"id": "item2"}],
        "next_page": f"{BASE_URL}/next_page_url"
    }
    
    # Second response (final page)
    second_response = MagicMock()
    second_response.status_code = 200
    second_response.json.return_value = {
        "count": 1,
        "values": [{"id": "item3"}]
    }
    
    # Configure the mock to return different responses on successive calls
    mock_client.get.side_effect = [first_response, second_response]
    
    with patch("httpx.AsyncClient", return_value=mock_client):
        url = f"{BASE_URL}/some_endpoint"
        result = await _fetch_data_from_kolada(url)
        
        # Verify both pages were fetched
        assert mock_client.get.call_count == 2
        mock_client.get.assert_any_call(url, timeout=60.0)
        mock_client.get.assert_any_call(f"{BASE_URL}/next_page_url", timeout=60.0)
        
        # Check that values from both pages were combined
        assert len(result["values"]) == 3
        assert result["count"] == 3
        assert result["values"][0]["id"] == "item1"
        assert result["values"][2]["id"] == "item3"


@pytest.mark.asyncio
async def test_fetch_data_http_error():
    """Test handling of HTTP errors"""
    mock_client = AsyncMock()
    mock_client.get.side_effect = httpx.HTTPStatusError(
        "404 Not Found", 
        request=MagicMock(), 
        response=MagicMock(status_code=404)
    )
    
    with patch("httpx.AsyncClient", return_value=mock_client):
        url = f"{BASE_URL}/invalid_endpoint"
        result = await _fetch_data_from_kolada(url)
        
        # Verify error handling
        assert "error" in result
        assert "404 Not Found" in result["error"]
        assert result["endpoint"] == url


@pytest.mark.asyncio
async def test_fetch_data_request_error():
    """Test handling of request errors (e.g., network issues)"""
    mock_client = AsyncMock()
    mock_client.get.side_effect = httpx.RequestError("Connection error", request=MagicMock())
    
    with patch("httpx.AsyncClient", return_value=mock_client):
        url = f"{BASE_URL}/some_endpoint"
        result = await _fetch_data_from_kolada(url)
        
        # Verify error handling
        assert "error" in result
        assert "Connection error" in result["error"]


@pytest.mark.asyncio
async def test_fetch_data_json_decode_error():
    """Test handling of JSON decode errors"""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
    
    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response
    
    with patch("httpx.AsyncClient", return_value=mock_client):
        url = f"{BASE_URL}/some_endpoint"
        result = await _fetch_data_from_kolada(url)
        
        # Verify error handling
        assert "error" in result
        assert "Invalid JSON" in result["error"]


@pytest.mark.asyncio
async def test_fetch_data_api_error_response():
    """Test handling of API error responses"""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "error": "Invalid KPI ID",
        "details": "The KPI ID 'INVALID' does not exist"
    }
    
    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response
    
    with patch("httpx.AsyncClient", return_value=mock_client):
        url = f"{BASE_URL}/data/kpi/INVALID/municipality/0180"
        result = await _fetch_data_from_kolada(url)
        
        # Verify the error is passed through
        assert "error" in result
        assert result["error"] == "Invalid KPI ID"
        assert result["details"] == "The KPI ID 'INVALID' does not exist"
