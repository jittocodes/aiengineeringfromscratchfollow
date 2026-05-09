from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Demo")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.resource("greetings://{name}")
def get_greetings(name: str) -> str:
    """Get a personalized greeting"""
    return f"This is a personalized Greeting. Hello, {name}"