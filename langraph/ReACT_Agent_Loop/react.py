from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv()


@tool
def triple(num: float) -> float:
    """_summary_

    Args:
        num (float): A number to triple function

    Returns:
        float: The triple of the input number
    """
    
    return float(num) * 3

@tool
def is_valid_number(num: int) -> bool:
    """Checks if a number is valid as per business use case or not.
    Args:
        
        Input: 
        num (int) : Number to check if its valid or not
        
        Output:
        bool: returns if the input is valid or not
    """
    
    return True if num < 50 else False

tools = [TavilySearch(max_results = 1), triple, is_valid_number]

llm = ChatOpenAI(
    model="gpt-4.1",
    temperature=0
).bind_tools(tools)