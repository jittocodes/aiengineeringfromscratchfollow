from dotenv import load_dotenv
load_dotenv()

import re
import inspect

import ollama
from langsmith import traceable # noqa: E402

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"

@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Lookup the price of a product in the catalog."""
    print(f"    >> Executing get_product_price(product={product})")
    prices = {"laptop" : 1299.99, "headphones" : 149.95, "keyboard" : 89.50}

    return prices.get(product, 0)

@traceable(run_type="tool")
def apply_discount(price: float, discount_tier) -> float:
    """Apply a discout tier to a price and return the final price
        Available Tier: Bronze, Silve and Gold
        """
    discount_percentages = {"bronze": 5, "silver": 10, "gold": 25}
    discount = discount_percentages.get(discount_tier.lower(), 0)

    return round(price * (1 - discount/100), 2)

tools = {
    "get_product_price" : get_product_price,
    "apply_discount" : apply_discount
}


@traceable(name="Ollama Chat", run_time="llm")
def ollama_chat_traced(messages):
    return ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)

@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools_dict = {
        "get_product_price" : get_product_price,
        "apply_discount" : apply_discount
    }


    
    print(f"Question: {question}")
    print("=" * 60)
    
    messages = [
        {"role" : "system", 
         "content" : (
             "You are a helpful shopping assitant"
            "You have access to a product catalog tool"
            "and a discount tool \n\n"
            "Rules:"  
            "1. NEVER GUESS OR ASSUME ANY PRODUCT PRICE"
            "You must call get_product_price first to get the real price.\n"
            "2. Only call apply_discount AFTER you have recieved a price from get_product_price. Pass the exact price"
            "returned by get_product_price. Do not pass a made up number.\n"    
            "3. Never calculate discounts yourself using math. Always use apply_discount tool.\n"
            "If the user does not specify a discount tier."
            "Ask them which tier to use - do NOT assume one." 
            "If the product is not available, Simply say the product is not listed."
         ),
        },
        {"role" : "user", "content" : question}
    ]
    
    for iteration in range(1, 30 + 1):
        print(f"\n Iteration: {iteration}")
         
        response = ollama_chat_traced(messages=messages)
        ai_message = response.message
        
        tool_calls = ai_message.tool_calls
         
        if not tool_calls:
           print(f"Answer is ready. Answer: {ai_message.content}")
           return ai_message.content
       
        tool_call = tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments
        
        print(f" [Tool Selected] {tool_name} with args: {tool_args}")
        
        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        observation = tool_to_use(**tool_args)
        
        print(f" [Tool Result] {observation}")
        
        messages.append(ai_message)
        messages.append(
            {
                "role" : "tool", 
                "content" : str(observation)
            }
        )
        
         
    pass

if __name__ == "__main__":   
    print("Langchain Agent")

    result = run_agent("What is the price of a Motorbike after applying a gold discount")