"""
Simple launcher for HTTP server - avoiding early imports
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == "__main__":
    from kolada_mcp.http_server import main
    main()
