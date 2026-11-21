from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage
from tavily import TavilyClient
from pprint import pprint


load_dotenv()

@tool
def get_sports_info(query: str) -> str:
    """
    Get sports information from Travily API.
    """
    client = TavilyClient()
    return client.search(query)

system_prompt="You are a sports info agent, give the answer in just one line, if you don't know the answer, say 'I don't know'."

agent = create_agent(model="gpt-4o-mini", tools=[get_sports_info], system_prompt=system_prompt)

question = HumanMessage(content="When MS Dhoni retires from IPL?, give me the exact date")
response = agent.invoke({"messages":[question]})
pprint(response)