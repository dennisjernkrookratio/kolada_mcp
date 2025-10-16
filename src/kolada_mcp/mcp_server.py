"""
MCP Server for ChatGPT Deep Research Integration

This server implements the Model Context Protocol (MCP) with search and fetch
capabilities designed to work with ChatGPT's deep research feature.

According to OpenAI's specification:
- search: Returns list of KPI search results
- fetch: Returns full KPI details and statistics
"""

import logging
import os
import json
from typing import Dict, List, Any

from fastmcp import FastMCP, Context

# Import existing Kolada functionality
from kolada_mcp.tools.metadata_tools import search_kpis as kolada_search_kpis, get_kpi_metadata
from kolada_mcp.tools.data_tools import fetch_kolada_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

server_instructions = """
This MCP server provides search and document retrieval capabilities for
Swedish municipal data from Kolada (kolada.se). Use the search tool to find
relevant Key Performance Indicators (KPIs) based on keywords, then use the
fetch tool to retrieve complete KPI details including metadata and statistics.
"""

def create_server():
    """Create and configure the MCP server with search and fetch tools."""
    
    # Initialize the FastMCP server with lifespan
    from kolada_mcp.lifespan.context import app_lifespan
    
    mcp = FastMCP(
        name="Kolada MCP Server",
        instructions=server_instructions,
        lifespan=app_lifespan
    )

    @mcp.tool()
    async def search(query: str, ctx: Context) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search for Swedish municipal Key Performance Indicators (KPIs) from Kolada.
        
        This tool searches through 6000+ KPIs to find semantically relevant matches
        using AI embeddings. Returns a list of search results with basic information.
        Use the fetch tool to get complete KPI details and statistics.

        Args:
            query: Search query string in Swedish or English. Natural language
                   queries work best. Examples: "arbetslöshet", "utbildningsnivå",
                   "förskoleplats", "skatteintäkter"

        Returns:
            Dictionary with 'results' key containing list of matching KPIs.
            Each result includes id, title, brief description, and URL.
        """
        if not query or not query.strip():
            logger.warning("Empty query received")
            return {"results": []}
        
        logger.info(f"MCP search request for query: '{query}'")
        
        try:
            # Use existing Kolada search functionality
            logger.info(f"Calling kolada_search_kpis with query='{query}', top_k=10")
            search_results = await kolada_search_kpis(keyword=query, ctx=ctx, limit=10)
            logger.info(f"kolada_search_kpis returned {len(search_results)} results")
            
            # Transform to OpenAI's required format
            results = []
            for kpi in search_results:
                kpi_id = kpi.get('id', '')
                title = kpi.get('title', 'Unknown KPI')
                description = kpi.get('description', '')
                
                # Create snippet from description
                snippet = description[:200] + "..." if len(description) > 200 else description
                
                result = {
                    "id": kpi_id,
                    "title": title,
                    "text": snippet,
                    "url": f"https://www.kolada.se/verktyg/fri-sokning/?kpis={kpi_id}"
                }
                results.append(result)
            
            logger.info(f"MCP search returned {len(results)} results")
            return {"results": results}
            
        except Exception as e:
            logger.error(f"Error in MCP search: {e}")
            return {"results": []}

    @mcp.tool()
    async def fetch(id: str, ctx: Context) -> Dict[str, Any]:
        """
        Retrieve complete KPI information and statistics by ID.
        
        This tool fetches comprehensive details for a specific KPI including
        metadata, description, unit of measurement, and recent statistics for
        major Swedish municipalities. Use this after finding relevant KPIs
        with the search tool to get complete information for analysis.

        Args:
            id: KPI identifier from Kolada (e.g., "N00941", "U00407")

        Returns:
            Complete KPI document with id, title, full description, unit,
            operating area, recent statistics, and URL for citation.

        Raises:
            ValueError: If the specified KPI ID is not found
        """
        if not id:
            logger.error("fetch called without KPI ID")
            raise ValueError("KPI ID is required")
        
        logger.info(f"MCP fetch request for KPI ID: {id}")
        
        try:
            # Get KPI metadata
            logger.info(f"Calling get_kpi_metadata for id={id}")
            metadata = await get_kpi_metadata(kpi_id=id, ctx=ctx)
            logger.info(f"get_kpi_metadata returned: {bool(metadata)}")
            
            if not metadata:
                logger.error(f"KPI {id} not found in metadata")
                raise ValueError(f"KPI {id} not found")
            
            # Note: Historical statistics would require separate API calls
            statistics_text = "\n\nNote: Historical statistics are available via separate queries."
            
            # Build complete document text
            title = metadata.get('title', f'KPI {id}')
            description = metadata.get('description', 'No description available')
            unit = metadata.get('unit', 'Unknown unit')
            operating_area = metadata.get('operating_area', 'Unknown area')
            
            full_text = f"""KPI: {title}

ID: {id}
Operating Area: {operating_area}
Unit of Measurement: {unit}

Description:
{description}
{statistics_text}

Data Source: Kolada (Swedish Association of Local Authorities and Regions)
"""
            
            result = {
                "id": id,
                "title": title,
                "text": full_text,
                "url": f"https://www.kolada.se/verktyg/fri-sokning/?kpis={id}",
                "metadata": {
                    "operating_area": operating_area,
                    "unit": unit,
                    "source": "kolada.se"
                }
            }
            
            logger.info(f"MCP fetch completed for KPI: {id}")
            return result
            
        except Exception as e:
            logger.error(f"Error in MCP fetch: {e}")
            raise ValueError(f"Could not fetch KPI {id}: {str(e)}")

    @mcp.tool()
    async def timeseries(
        kpi_id: str, 
        municipality_id: str, 
        ctx: Context,
        years: str | None = None
    ) -> Dict[str, Any]:
        """
        Fetch time series data for a specific KPI and municipality.
        
        This tool retrieves historical data points for a Key Performance Indicator
        across one or more years for a specific Swedish municipality. Use this after
        finding relevant KPIs with search to get actual statistics over time.

        Args:
            kpi_id: KPI identifier from Kolada (e.g., "N00941", "N02387")
            municipality_id: Municipality code (e.g., "0180" for Stockholm, 
                           "0188" for Norrtälje). Use 4 digits with leading zero.
            years: Optional year range (e.g., "2019-2023" or "2020,2021,2022").
                   If omitted, fetches all available years.

        Returns:
            Dictionary with time series data including years and values, plus
            metadata about the KPI and municipality.

        Examples:
            - Get latest 5 years: timeseries("N02387", "0188", years="2019-2023")
            - Get specific years: timeseries("N02348", "0188", years="2020,2022,2023")
            - Get all available: timeseries("N00941", "0180")
        """
        logger.info(f"MCP timeseries request: KPI={kpi_id}, Municipality={municipality_id}, Years={years}")
        
        try:
            # Parse years parameter
            year_param = None
            if years:
                # Handle range format "2019-2023"
                if "-" in years:
                    start, end = years.split("-")
                    year_list = [str(y) for y in range(int(start), int(end) + 1)]
                    year_param = ",".join(year_list)
                else:
                    # Handle comma-separated "2020,2021,2022"
                    year_param = years
            
            logger.info(f"Calling fetch_kolada_data with year_param={year_param}")
            
            # Fetch data using existing tool
            data = await fetch_kolada_data(
                kpi_id=kpi_id,
                municipality_id=municipality_id,
                ctx=ctx,
                year=year_param,
                municipality_type="K"
            )
            
            if "error" in data:
                logger.error(f"Error from fetch_kolada_data: {data['error']}")
                return {"error": data["error"], "kpi_id": kpi_id, "municipality_id": municipality_id}
            
            # Extract time series from response
            values_list = data.get("values", [])
            logger.info(f"Received {len(values_list)} value entries")
            
            if not values_list:
                return {
                    "kpi_id": kpi_id,
                    "municipality_id": municipality_id,
                    "municipality_name": "Unknown",
                    "rows": [],
                    "message": "No data available for the specified parameters"
                }
            
            # Build compact time series
            rows = []
            municipality_name = "Unknown"
            
            for entry in values_list:
                municipality_name = entry.get("municipality_name", municipality_name)
                period = entry.get("period")
                
                # Extract values (handle gender breakdown)
                for value_item in entry.get("values", []):
                    gender = value_item.get("gender", "T")
                    value = value_item.get("value")
                    
                    # Default to "T" (total) gender unless specified
                    if gender == "T" and value is not None:
                        rows.append({
                            "year": period,
                            "value": value,
                            "gender": gender
                        })
            
            # Sort by year
            rows.sort(key=lambda x: x["year"])
            
            logger.info(f"Returning {len(rows)} data points for {municipality_name}")
            
            result = {
                "kpi_id": kpi_id,
                "municipality_id": municipality_id,
                "municipality_name": municipality_name,
                "rows": rows,
                "count": len(rows)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in timeseries: {e}", exc_info=True)
            return {
                "error": str(e),
                "kpi_id": kpi_id,
                "municipality_id": municipality_id
            }

    return mcp


def main():
    """Main function to start the MCP server with SSE transport."""
    
    # Get port from environment (Cloud Run uses PORT)
    port = int(os.environ.get("PORT", 8000))
    
    logger.info(f"Starting Kolada MCP server with SSE transport")
    logger.info(f"Port: {port}")
    
    # Create the MCP server
    server = create_server()
    
    # Configure and start the server with SSE transport
    try:
        # Use FastMCP's built-in run method with SSE transport
        server.run(transport="sse", host="0.0.0.0", port=port)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


if __name__ == "__main__":
    main()
