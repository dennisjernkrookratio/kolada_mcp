# Kolada MCP Server

[![MCP badge](https://modelcontextprotocol.io/badge/kolada-mcp)](https://modelcontextprotocol.io)

> **Note:** This project is an independent, third-party implementation and is not endorsed by or affiliated with RKA (Council for the Promotion of Municipal Analysis).

The **Kolada MCP Server** enables seamless integration between Large Language Models (LLMs) and [Kolada](https://www.kolada.se/), Sweden’s comprehensive municipal and regional statistical database. It provides structured access to thousands of Key Performance Indicators (KPIs), facilitating rich, data-driven analysis, comparisons, and explorations of public sector statistics.

## Overview

Kolada MCP server acts as an intelligent middleware between LLM-based applications and the Kolada database, allowing users to easily query and analyze data related to Swedish municipalities and regions. With semantic search capabilities and robust analysis tools, the Kolada MCP Server significantly simplifies the task of navigating and interpreting the vast array of KPIs available in Kolada.

## Features

- **Semantic Search**: Find KPIs based on natural language descriptions.
- **Category Filtering**: Access KPIs grouped by thematic categories (e.g., demographics, economy, education).
- **Municipal & Regional Data Retrieval**: Fetch precise data points or historical time series.
- **Multi-Year Comparative Analysis**: Calculate changes in KPI performance over multiple years, over all municipalities, regions or landstings.
- **Cross-KPI Correlation**: Analyze relationships between different KPIs across municipalities or regions.

## Components

### Tools

1. `list_operating_areas`
   - Retrieve available KPI categories.

2. `get_kpis_by_operating_area`
   - List KPIs under a specific category.

3. `search_kpis`
   - Perform semantic searches to discover relevant KPIs.

4. `get_kpi_metadata`
   - Access detailed metadata for specific KPIs.

5. `fetch_kolada_data`
   - Obtain precise KPI values for specific municipalities or regions.

6. `analyze_kpi_across_municipalities`
   - Conduct in-depth analysis and comparisons of KPI performance across municipalities.

7. `compare_kpis`
   - Evaluate the correlation or difference between two KPIs.

8. `list_municipalities`
   - Returns a list of municipality IDs and names filtered by type (default is `"K"`). Passing an empty string for `municipality_type` returns municipalities of all types.


## Quick Start

Kolada MCP Server uses sensible defaults, with data fetched and cached on startup. No additional API keys or authentication are necessary to use Kolada’s open API.

### Cache

Kolada also with pre-cached dataset that lists all available KPIs and their metadata.
To use a fresh cache instead, simply delete the kpi_embeddings.npz file and restart the server.

# Installation
Using uv to install the Kolada MCP requirements is highly recommended. This ensures that all dependencies are installed in a clean environment. Simply run `uv sync` to install the required packages.

## Development and Testing

Run the Kolada MCP server locally in development mode with detailed debugging:

```bash
uv run mcp dev kolada-mcp.py
```

Then open the [MCP Inspector](https://github.com/modelcontextprotocol/inspector) at `http://localhost:5173` in your browser. Use the inspector interface to:

- Test individual tools.
- Inspect returned data.
- Debug server interactions.


## Claude Desktop


``` json
{
  "mcpServers": {
        "Kolada": {
        "command": "uv",
        "args": [
            "--directory",
            "[path to kolada-mcp directory]",
            "run",
            "kolada-mcp.py"
        ]
        }
    }
}
```



## Contributing

We welcome contributions! Report issues, suggest enhancements, or submit pull requests on GitHub.

## Disclaimer

Kolada MCP Server is independently developed and maintained. It is not officially endorsed by, affiliated with, or related to the Council for the Promotion of Municipal Analysis (RKA) or any governmental body.

## License

Kolada MCP Server is released under the [Apache License 2.0](LICENSE).

