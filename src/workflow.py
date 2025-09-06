import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from relevence_query_check import relevence_query_check 
from Retrieval import Retrieveal
from check_relevent_document import check_relevence_document
from llm_response import LLm_response
from response_irrelevent_query import response_irrelevent_Question
from langgraph.graph import StateGraph,START,END
from typing_extensions import TypedDict
from typing import List,Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class chatSate(TypedDict):
    chat_history:Annotated[list[BaseMessage],add_messages]
    answer: BaseMessage
    is_relevence_query:str
    is_relevent_document:str
    docs:List[str]

Graph = StateGraph(state_schema=chatSate)



Graph.add_node("relevet_query_check",relevence_query_check)
Graph.add_node("Retrieveal",Retrieveal)
Graph.add_node("check_relevence_document",check_relevence_document)
Graph.add_node("LLm_response",LLm_response)
Graph.add_node("response_irrelevent_Question",response_irrelevent_Question)


Graph.add_edge(START, "relevet_query_check")
Graph.add_conditional_edges(
    "relevet_query_check",
    lambda X: X['is_relevence_query'].strip().lower(),
    {
        "relevant": "Retrieveal",
        "memory": "LLm_response",
        "irrelevant": "response_irrelevent_Question"
    }
)
Graph.add_edge('Retrieveal','check_relevence_document')
Graph.add_conditional_edges("check_relevence_document",lambda X: "yes" if "yes" in X['is_relevent_document'].lower() else "no",{
        "yes": "LLm_response",
        "no": "response_irrelevent_Question"  } )
Graph.add_edge("LLm_response",END)
Graph.add_edge("response_irrelevent_Question",END)

conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

Workflow =Graph.compile(checkpointer=checkpointer)


from langchain_core.messages import HumanMessage
config = {'configurable': {'thread_id': 'thread-4'}}

response = Workflow.invoke(
    {'chat_history': [HumanMessage(content='when Speaker to act as President')]},
    config=config
)
print(response['answer'].content)

from PIL import Image
import io

# Assuming Workflow.get_graph().draw_mermaid_png() returns the raw image data (e.g., bytes)
image_data = Workflow.get_graph().draw_mermaid_png()

# Create a PIL Image object from the raw image data
img = Image.open(io.BytesIO(image_data))

# Save the image to a file
img.save("workflow_graph.png")