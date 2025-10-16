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
