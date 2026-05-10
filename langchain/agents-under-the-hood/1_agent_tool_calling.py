from dotenv import load_dotenv
load_dotenv()

from langchain.tools import tool # noqa: E402
from langchain.chat_models import init_chat_model  # noqa: E402
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage # noqa: E402
from langsmith import traceable # noqa: E402

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"

@tool
def get_product_price(product: str) -> float:
    """Lookup the price of a product in the catalog."""
    print(f"    >> Executing get_product_price(product={product})")
    prices = {"laptop" : 1299.99, "headphones" : 149.95, "keyboard" : 89.50}

    return prices.get(product, 0)

@tool
def apply_discount(price: float, discount_tier) -> float:
    """Aply a discout tier to a price and return the final price
        Available Tier: Bronze, Silve and Gold
        """
    discount_percentages = {"bronze": 5, "silver": 10, "gold": 25}
    discount = discount_percentages.get(discount_tier.lower(), 0)

    return round(price * (1 - discount/100), 2)

@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}
    
    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    
    print(f"Question: {question}")
    print("=" * 60)
    
    messages = [
        SystemMessage(
            content="You are a helpful shopping assitant"
            "You have access to a product catalog tool"
            "and a discount tool \n\n"
        ),
        HumanMessage(
            content=question
        )
    ]
    
    for iteration in range(1, 30 + 1):
        print(f"\n Iteration: {iteration}")
         
        ai_message = llm_with_tools.invoke(messages)
         
        tool_calls = ai_message.tool_calls
         
        if not tool_calls:
           print(f"Answer is ready. Answer: {ai_message.content}")
           return ai_message.content
       
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")
        
        print(f" [Tool Selected] {tool_name} with args: {tool_args}")
        
        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        observation = tool_to_use.invoke(tool_args)
        
        print(f" [Tool Result] {observation}")
        
        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )
        
         
    pass

if __name__ == "__main__":   
    print("Langchain Agent")

    result = run_agent("What is the price of a laptop after applying a gold discount")