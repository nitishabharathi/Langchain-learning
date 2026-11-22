from dataclasses import dataclass
from pprint import pprint
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_openai import OpenAIEmbeddings

load_dotenv()

recipes = [
    Document(
        page_content="Paneer Pallipalayam: Spicy coconut-roasted paneer cubes with curry leaves and red chillies."
    ),
    Document(
        page_content="Paneer Aranmani Varuval: Royal pan-roasted paneer tossed in aromatic South Indian spices."
    ),
    Document(
        page_content="Meen Kulambu: Tangy tamarind fish curry cooked with shallots and fenugreek."
    ),
]
vectorstore = Chroma.from_documents(recipes, embedding=OpenAIEmbeddings())


@dataclass
class FoodContext:
    vectorstore: Chroma
    favorite_food: str = "Any paneer dish"
    least_favorite_food: str = "Fish"



@tool
def search_matching_recipes(runtime: ToolRuntime[FoodContext]) -> str:
    """Search the recipe knowledge base for dishes matching the user's favorite food. But return only one"""
    fav = runtime.context.favorite_food
    avoid = runtime.context.least_favorite_food.lower()

    docs = runtime.context.vectorstore.similarity_search(fav, k=1)

    safe_recipes = [
        d.page_content for d in docs if avoid not in d.page_content.lower()
    ]
    return "\n".join(safe_recipes) if safe_recipes else "No matching recipes found."


agent = create_agent(
    model="gpt-4o-mini",
    tools=[search_matching_recipes],
    context_schema=FoodContext,
)

question = HumanMessage(
    content="Suggest me a dish with my favorite food from my recipes"
)


response = agent.invoke(
    {"messages": [question]},
    context=FoodContext(vectorstore=vectorstore),
)
pprint(response["messages"][-1].content)