from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from pprint import pprint
from pydantic import BaseModel

load_dotenv()

model = init_chat_model(model="gpt-4o-mini", temperature=0.5)

system_prompt = """
You are a travel expert. Give information in the below format. keep only four-five pointers

Best month to visit: <Month>
Cheapest month to visit: <Month>
What are the attractions to visit: <Attractions>
Is it a baby friendly destination: <Yes/No>

"""
agent = create_agent(model=model, system_prompt=system_prompt)
question = HumanMessage(content="I am planning to visit London")

response = agent.invoke({"messages":[question]})
pprint(response)


### Structured Output Example

class PlaceInfo(BaseModel):
    best_month_to_visit: str
    cheapest_month_to_visit: str
    attractions_to_visit: str
    is_baby_friendly_destination: bool

agent = create_agent(model=model, system_prompt="You are a travel expert. Provide the information in the requested format", response_format=PlaceInfo)
response=agent.invoke({"messages":[question]})
pprint(response)