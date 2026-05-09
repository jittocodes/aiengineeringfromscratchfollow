from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from tavily import TavilyClient
from langchain_tavily import TavilySearch

load_dotenv()

tavily = TavilyClient()

@tool
def search(query: str) -> str:
    """Tool that searches the internet

    Args:
        query (str): The query to search for

    Returns:
        str: The search result
    """
    print(f"Searching for {query}")
    return tavily.search(query=query)



tools = [TavilySearch]
llm = ChatOllama(model="llama3.2:latest")

agent = create_agent(model=llm, tools=tools)

result = agent.invoke({"messages": HumanMessage(content="What is the weather in Tokyo")})

print(result)