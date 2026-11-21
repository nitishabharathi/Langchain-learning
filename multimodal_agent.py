from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from pprint import pprint
import base64


load_dotenv()

with open("data/penguin.jpg", "rb") as image_file:
    image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

multimodal_question = HumanMessage(content=[
    {"type":"text", "text" :"Generate a motivational poem about the best part of this image"},
    {
    "type":"image", "base64": image_base64, "mime_type":"image/png"
    }
])


agent = create_agent(model="gpt-5-nano")
response = agent.invoke({"messages":[multimodal_question]})
pprint(response["messages"][-1].content)