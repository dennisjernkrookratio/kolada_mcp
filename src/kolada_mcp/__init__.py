def main():
    """Entry point for kolada-mcp stdio server"""
    from kolada_mcp.server import main as server_main
    server_main()

__all__ = ['main']
