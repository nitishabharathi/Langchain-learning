from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from pprint import pprint


load_dotenv()

@tool
def square_root(x: float) -> float:
    """Returns square roots of a number"""
    return x ** 0.5

@tool
def sqaure(x: float) -> float:
    """Returns square of a number"""
    return x ** 2

square_root_agent = create_agent(model='gpt-4o-mini', tools=[square_root])
square_agent = create_agent(model="gpt-4o-mini", tools=[sqaure])

@tool
def call_square_root_agent(x: float) -> float:
    """Invoke square root agent if user asks for square root of a number"""
    response = square_root_agent.invoke({"messages":[HumanMessage(content = f"Calculate square root of {x}")]})
    return response['messages'][-1].content

@tool
def call_square_agent(x: float) -> float:
    """Invoke square agent if user asks for a square of a number"""
    response = square_agent.invoke({"messages": [HumanMessage(content = f"Calculate square of a {x}")]})
    return response['messages'][-1].content

main_agent = create_agent(model='gpt-4o-mini', tools = [call_square_agent, call_square_root_agent], system_prompt="You are a math agent who can calcualte square root or square of a number")

main_response = main_agent.invoke({"messages": [HumanMessage(content="What is the square root of 225")]})

pprint(main_response)