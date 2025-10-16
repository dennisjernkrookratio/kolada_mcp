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
        
        SPECIAL COMMANDS for time series data:
        - Format: "ts:kpi=<ID>;muni=<CODE>;years=<RANGE>"
        - Example: "ts:kpi=N02387;muni=0188;years=2019-2023"
        - Or natural language: "Norrtälje skolresultat senaste 5 åren"

        Args:
            query: Search query string in Swedish or English. Natural language
                   queries work best. Examples: "arbetslöshet", "utbildningsnivå",
                   "förskoleplats", "skatteintäkter"
                   
                   For time series: include municipality name + topic + years

        Returns:
            Dictionary with 'results' key containing list of matching KPIs
            or time series data if pattern matches.
        """
        if not query or not query.strip():
            logger.warning("Empty query received")
            return {"results": []}
        
        logger.info(f"MCP search request for query: '{query}'")
        
        # Check for timeseries command pattern: "ts:kpi=XXX;muni=YYY;years=ZZZ"
        import re
        ts_match = re.match(r'ts:kpi=([^;]+);muni=([^;]+);years=(.+)', query.strip(), re.IGNORECASE)
        
        if ts_match:
            kpi_id = ts_match.group(1).strip()
            muni_id = ts_match.group(2).strip()
            years = ts_match.group(3).strip()
            
            logger.info(f"Timeseries command detected: KPI={kpi_id}, Municipality={muni_id}, Years={years}")
            
            try:
                # Parse years
                year_param = None
                if "-" in years:
                    start, end = years.split("-")
                    year_list = [str(y) for y in range(int(start), int(end) + 1)]
                    year_param = ",".join(year_list)
                else:
                    year_param = years
                
                # Fetch data
                data = await fetch_kolada_data(
                    kpi_id=kpi_id,
                    municipality_id=muni_id,
                    ctx=ctx,
                    year=year_param,
                    municipality_type="K"
                )
                
                if "error" in data:
                    return {"results": [], "error": data["error"]}
                
                # Extract time series
                values_list = data.get("values", [])
                rows = []
                municipality_name = "Unknown"
                
                for entry in values_list:
                    municipality_name = entry.get("municipality_name", municipality_name)
                    period = entry.get("period")
                    
                    for value_item in entry.get("values", []):
                        if value_item.get("gender") == "T" and value_item.get("value") is not None:
                            rows.append({
                                "year": period,
                                "value": value_item.get("value")
                            })
                
                rows.sort(key=lambda x: x["year"])
                
                result = {
                    "kpi_id": kpi_id,
                    "name": f"Time series for KPI {kpi_id}",
                    "municipality_id": muni_id,
                    "municipality_name": municipality_name,
                    "rows": rows
                }
                
                logger.info(f"Returning timeseries with {len(rows)} data points")
                return {"results": [result]}
                
            except Exception as e:
                logger.error(f"Error in timeseries command: {e}", exc_info=True)
                return {"results": [], "error": str(e)}
        
        # Check for multi-KPI timeseries pattern: "timeseries: muni=XXX; years=YYY; kpis=A,B,C"
        multi_ts_match = re.match(
            r'timeseries:\s*muni=([^;]+);\s*years=([^;]+);\s*kpis=(.+)',
            query.strip(),
            re.IGNORECASE
        )
        
        if multi_ts_match:
            muni_id = multi_ts_match.group(1).strip()
            years = multi_ts_match.group(2).strip()
            kpis = [k.strip() for k in multi_ts_match.group(3).split(',')]
            
            logger.info(f"Multi-KPI timeseries command: Municipality={muni_id}, Years={years}, KPIs={kpis}")
            
            # Parse years
            year_param = None
            if "-" in years:
                start, end = years.split("-")
                year_list = [str(y) for y in range(int(start), int(end) + 1)]
                year_param = ",".join(year_list)
            else:
                year_param = years
            
            # Fetch data for each KPI
            results = []
            for kpi_id in kpis:
                try:
                    data = await fetch_kolada_data(
                        kpi_id=kpi_id,
                        municipality_id=muni_id,
                        ctx=ctx,
                        year=year_param,
                        municipality_type="K"
                    )
                    
                    if "error" not in data:
                        values_list = data.get("values", [])
                        rows = []
                        muni_name = "Unknown"
                        kpi_title = kpi_id
                        
                        for entry in values_list:
                            muni_name = entry.get("municipality_name", muni_name)
                            period = entry.get("period")
                            
                            for value_item in entry.get("values", []):
                                if value_item.get("gender") == "T" and value_item.get("value") is not None:
                                    rows.append({
                                        "year": period,
                                        "value": value_item.get("value")
                                    })
                        
                        rows.sort(key=lambda x: x["year"], reverse=True)
                        
                        # Try to get KPI title from metadata
                        try:
                            kpi_meta = await get_kpi_metadata(kpi_id, ctx)
                            kpi_title = kpi_meta.get("title", kpi_id)
                        except:
                            pass
                        
                        if rows:
                            results.append({
                                "kpi_id": kpi_id,
                                "name": kpi_title,
                                "municipality_id": muni_id,
                                "municipality_name": muni_name,
                                "rows": rows
                            })
                    else:
                        logger.warning(f"Error fetching KPI {kpi_id}: {data.get('error')}")
                
                except Exception as e:
                    logger.error(f"Error fetching KPI {kpi_id}: {e}")
                    continue
            
            logger.info(f"Returning {len(results)} time series for multi-KPI query")
            return {
                "results": results,
                "meta": {
                    "municipality_id": muni_id,
                    "municipality_name": results[0]["municipality_name"] if results else "Unknown"
                }
            }
        
        # Check for natural language pattern: "Norrtälje + skolresultat/meritvärde + senaste X år"
        # Broader pattern matching
        nl_match = re.search(
            r'(norrtälje|stockholm|göteborg|malmö|uppsala)',
            query.lower()
        )
        
        years_match = re.search(r'(senaste|sista)\s*(\d+)\s*(år|åren)', query.lower())
        topic_match = re.search(r'(skolresultat|meritvärde|betyg|behörighet|godkända|skol)', query.lower())
        
        nl_combined = nl_match and years_match and topic_match
        
        if nl_combined:
            municipality_name = nl_match.group(1)
            topic = topic_match.group(1) if topic_match else "skolresultat"
            num_years = int(years_match.group(2))
            
            logger.info(f"Natural language timeseries detected: {municipality_name} + {topic} + {num_years} years")
            
            # Map municipality names to codes
            muni_codes = {
                "norrtälje": "0188",
                "stockholm": "0180",
                "göteborg": "1480",
                "malmö": "1280",
                "uppsala": "0380"
            }
            
            muni_id = muni_codes.get(municipality_name)
            if not muni_id:
                logger.warning(f"Unknown municipality: {municipality_name}")
                return {"results": [], "error": f"Unknown municipality: {municipality_name}"}
            
            # Define relevant KPIs for school results (updated with correct IDs)
            school_kpis = {
                "meritvärde_läge": "N15504",    # Meritvärde åk 9 lägeskommun
                "meritvärde_hem": "N15507",     # Meritvärde åk 9 hemkommun
                "alla_ämnen": "N15560",         # Godkända i alla ämnen åk 9
                "svenska": "N02387",            # Betygspoäng svenska åk 9
                "engelska": "N02348",           # Betygspoäng engelska åk 9
                "matematik": "N02391",          # Betygspoäng matematik åk 9
                "behörighet": "N00956",         # Andel behöriga till gymnasiet
            }
            
            # Determine which KPIs to fetch
            kpi_ids = []
            if "meritvärde" in topic:
                kpi_ids = [
                    ("N15504", "Meritvärde åk 9 (lägeskommun)"),
                    ("N15507", "Meritvärde åk 9 (hemkommun)")
                ]
            elif "behörighet" in topic or "behöriga" in topic:
                kpi_ids = [("N00956", "Andel behöriga till gymnasiet")]
            elif "godkända" in topic or "alla ämnen" in topic:
                kpi_ids = [("N15560", "Godkända i alla ämnen åk 9")]
            else:
                # Fetch comprehensive school KPIs
                kpi_ids = [
                    ("N15504", "Meritvärde åk 9 (lägeskommun)"),
                    ("N15507", "Meritvärde åk 9 (hemkommun)"),
                    ("N15560", "Godkända i alla ämnen åk 9"),
                ]
            
            # Calculate year range
            from datetime import datetime
            current_year = datetime.now().year
            start_year = current_year - num_years
            years_str = f"{start_year}-{current_year - 1}"
            
            logger.info(f"Fetching {len(kpi_ids)} KPIs for {municipality_name} ({muni_id}), years {years_str}")
            
            # Fetch data for each KPI
            results = []
            for kpi_id, kpi_name in kpi_ids:
                try:
                    year_list = [str(y) for y in range(start_year, current_year)]
                    year_param = ",".join(year_list)
                    
                    data = await fetch_kolada_data(
                        kpi_id=kpi_id,
                        municipality_id=muni_id,
                        ctx=ctx,
                        year=year_param,
                        municipality_type="K"
                    )
                    
                    if "error" not in data:
                        values_list = data.get("values", [])
                        rows = []
                        muni_name = "Unknown"
                        
                        for entry in values_list:
                            muni_name = entry.get("municipality_name", muni_name)
                            period = entry.get("period")
                            
                            for value_item in entry.get("values", []):
                                if value_item.get("gender") == "T" and value_item.get("value") is not None:
                                    rows.append({
                                        "year": period,
                                        "value": value_item.get("value")
                                    })
                        
                        rows.sort(key=lambda x: x["year"], reverse=True)
                        
                        if rows:
                            results.append({
                                "kpi_id": kpi_id,
                                "name": kpi_name,
                                "municipality_id": muni_id,
                                "municipality_name": muni_name,
                                "rows": rows
                            })
                
                except Exception as e:
                    logger.error(f"Error fetching KPI {kpi_id}: {e}")
                    continue
            
            logger.info(f"Returning {len(results)} time series")
            return {
                "results": results,
                "meta": {
                    "municipality_id": muni_id,
                    "municipality_name": results[0]["municipality_name"] if results else "Unknown"
                }
            }
        
        # Default behavior: semantic KPI search
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
                municipality_id=muni_id,
                ctx=ctx,
                year=year_param,
                municipality_type="K"
            )
            
            if "error" in data:
                logger.error(f"Error from fetch_kolada_data: {data['error']}")
                return {"error": data["error"], "kpi_id": kpi_id, "municipality_id": muni_id}
            
            # Extract time series from response
            values_list = data.get("values", [])
            logger.info(f"Received {len(values_list)} value entries")
            
            if not values_list:
                return {
                    "kpi_id": kpi_id,
                    "municipality_id": muni_id,
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
                "municipality_id": muni_id,
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
                "municipality_id": muni_id
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
