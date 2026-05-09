from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from langchain_tavily import TavilySearch

from typing import List
from pydantic import BaseModel, Field

load_dotenv()
# tavily = TavilyClient()

# @tool
# def search(query: str) -> str:
#     """Tool that searches the internet

#     Args:
#         query (str): The query to search for

#     Returns:
#         str: The search result
#     """
#     print(f"Searching for {query}")
#     return tavily.search(query=query)

class Source(BaseModel):
    """Schema for a source used by the agent"""
    
    url:str = Field(description="The URL of the source")
    
class AgentResponse(BaseModel):
    """Schema for agent response with the answers and sources"""
    
    answer:str = Field(description="the agent's answer to the query")
    sources: List[Source] = Field(default_factory=list, description="The list of the sources used to generate the answers")

tools = [TavilySearch()]
# llm = ChatOllama(model="llama3.2:latest")
llm = ChatOpenAI(model='gpt-4.1')

agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

result = agent.invoke({"messages": HumanMessage(content="What is the weather in Tokyo")})

print(result)