from dotenv import load_dotenv
load_dotenv()

from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langsmith import traceable

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
    discount_percentages = {"brozne": 5, "silver": 10, "gold": 25}
    discount = discount_percentages.get(discount_tier, 0)

    return round(price * (1 - discount/100), 2)

def run_agent(question: str):
    pass

if __name__ = "__main__":
    print("Langchain Agent")

    result = run_agent("What is the price of a laptop after applying a gold discount")