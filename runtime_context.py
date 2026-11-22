from dotenv import load_dotenv
from langchain.agents import create_agent
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime
from langchain.messages import HumanMessage
from pprint import pprint

load_dotenv()

@dataclass
class FoodContext:
    favorite_food:str = "Any paneer dish"
    least_favorite_food:str = "Fish"

@tool
def get_favorite_food(runtime: ToolRuntime[FoodContext]) -> str:
    """Get favorite food of the user"""
    return runtime.context.favorite_food

@tool
def get_least_favorite_food(runtime: ToolRuntime[FoodContext]) -> str:
    """Get least favorite food of the user"""
    return runtime.context.least_favorite_food

agent = create_agent(model="gpt-4o-mini", tools = [get_favorite_food, get_least_favorite_food], context_schema= FoodContext)

question = HumanMessage(content="Suggest me a dish with my favorite food")

response = agent.invoke({"messages" :[question]}, context = FoodContext)
pprint(response["messages"][-1].content)
