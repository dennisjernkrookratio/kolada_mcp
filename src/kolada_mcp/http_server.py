"""
HTTP Server for Kolada MCP - enables ChatGPT integration
"""
import asyncio
import json
from typing import Any, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

from kolada_mcp.lifespan.context import app_lifespan
from kolada_mcp.tools.comparison_tools import compare_kpis
from kolada_mcp.tools.data_tools import (
    analyze_kpi_across_municipalities,
    fetch_kolada_data,
)
from kolada_mcp.tools.municipality_tools import list_municipalities, filter_municipalities_by_kpi
from kolada_mcp.tools.metadata_tools import (
    get_kpi_metadata,
    get_kpis_by_operating_area,
    list_operating_areas,
    search_kpis,
)

# Tool registry
TOOLS = {
    "list_operating_areas": list_operating_areas,
    "get_kpis_by_operating_area": get_kpis_by_operating_area,
    "get_kpi_metadata": get_kpi_metadata,
    "search_kpis": search_kpis,
    "fetch_kolada_data": fetch_kolada_data,
    "analyze_kpi_across_municipalities": analyze_kpi_across_municipalities,
    "compare_kpis": compare_kpis,
    "list_municipalities": list_municipalities,
    "filter_municipalities_by_kpi": filter_municipalities_by_kpi,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup application resources"""
    # Initialize app context using the existing lifespan
    async with app_lifespan(None) as context:
        app.state.context = context
        yield


# Create FastAPI app
app = FastAPI(
    title="Kolada MCP HTTP Server",
    description="HTTP endpoint for Kolada MCP Server - compatible with ChatGPT",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware for ChatGPT access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://chat.openai.com", "https://chatgpt.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ToolRequest(BaseModel):
    tool_name: str
    arguments: Dict[str, Any] = {}


class ToolResponse(BaseModel):
    success: bool
    result: Any = None
    error: str = None


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Kolada MCP HTTP Server",
        "version": "0.1.0",
        "description": "HTTP API for accessing Kolada statistical data",
        "endpoints": {
            "/tools": "List available tools",
            "/tool": "Execute a tool (POST)",
            "/health": "Health check"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.get("/tools")
async def list_tools():
    """List all available tools with their schemas"""
    tool_descriptions = {}
    
    for tool_name, tool_func in TOOLS.items():
        # Get function signature and docstring
        import inspect
        sig = inspect.signature(tool_func)
        doc = inspect.getdoc(tool_func) or "No description available"
        
        params = {}
        for param_name, param in sig.parameters.items():
            if param_name != "context":
                params[param_name] = {
                    "required": param.default == inspect.Parameter.empty,
                    "type": str(param.annotation) if param.annotation != inspect.Parameter.empty else "Any"
                }
        
        tool_descriptions[tool_name] = {
            "description": doc,
            "parameters": params
        }
    
    return tool_descriptions


@app.post("/tool")
async def execute_tool(request: ToolRequest) -> ToolResponse:
    """Execute a specific tool with given arguments"""
    tool_name = request.tool_name
    
    if tool_name not in TOOLS:
        raise HTTPException(
            status_code=404,
            detail=f"Tool '{tool_name}' not found. Available tools: {list(TOOLS.keys())}"
        )
    
    tool_func = TOOLS[tool_name]
    
    try:
        # Get context from app state
        context = app.state.context
        
        # Check if function expects context parameter
        import inspect
        sig = inspect.signature(tool_func)
        
        if "context" in sig.parameters:
            # Call with context
            if asyncio.iscoroutinefunction(tool_func):
                result = await tool_func(context=context, **request.arguments)
            else:
                result = tool_func(context=context, **request.arguments)
        else:
            # Call without context
            if asyncio.iscoroutinefunction(tool_func):
                result = await tool_func(**request.arguments)
            else:
                result = tool_func(**request.arguments)
        
        return ToolResponse(success=True, result=result)
    
    except TypeError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid arguments for tool '{tool_name}': {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error executing tool '{tool_name}': {str(e)}"
        )


@app.get("/.well-known/ai-plugin.json")
async def plugin_manifest():
    """OpenAI plugin manifest for ChatGPT integration"""
    return {
        "schema_version": "v1",
        "name_for_human": "Kolada Statistics",
        "name_for_model": "kolada",
        "description_for_human": "Access Swedish municipal and regional statistics from Kolada database.",
        "description_for_model": "Access comprehensive Swedish municipal and regional statistics from the Kolada database. Search for KPIs, retrieve data, analyze trends, and compare municipalities across thousands of indicators covering education, healthcare, environment, economy, and more.",
        "auth": {
            "type": "none"
        },
        "api": {
            "type": "openapi",
            "url": "/openapi.json"
        },
        "logo_url": "https://www.kolada.se/images/kolada-logo.png",
        "contact_email": "support@example.com",
        "legal_info_url": "https://github.com/aerugo/kolada-mcp"
    }


def main():
    """Run the HTTP server"""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)


if __name__ == "__main__":
    main()
