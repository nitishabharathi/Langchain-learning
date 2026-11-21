from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from pprint import pprint

load_dotenv()

system_prompt = "Answer the question if you know, else tell do not know"
agent = create_agent(model="gpt-4o-mini",checkpointer=InMemorySaver(), system_prompt=system_prompt)

question = HumanMessage(content="My favorite football club is Arsenal")
config = {"configurable":{"thread_id":"1"}}

response = agent.invoke({"messages": [question]}, config)
pprint(response["messages"][-1].content)

question1=HumanMessage(content="What is my favorite football club?")


response1 = agent.invoke({"messages":[question1]}, config)
pprint(response1["messages"][-1].content)

question2 = HumanMessage(content="What is my favorite cricket team?")
response2 = agent.invoke({"messages":[question2]}, config)
pprint(response2["messages"][-1].content)