from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.messages import AIMessage
from pprint import pprint

load_dotenv()  # Load environment variables from .env file

#temperature is set to 1 for more creative responses, and use_responses_api is set to True to utilize the Responses API for better performance.
model = init_chat_model(model="gpt-4o-mini", temperature=1, use_responses_api = False)
#model_response = model.invoke("When will MS Dhoni retire from IPL")
#print(model_response.content)


agent = create_agent(model = model)
response = agent.invoke({"messages" : [HumanMessage(content="When will MS Dhoni retire from IPL"),AIMessage(content="Dhoni retires in 2028"), HumanMessage(content="What is your source of confirmation?")]})
pprint(response)

