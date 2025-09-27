import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from relevence_query_check import relevance_query_check 
from Retrieval import Retrieveal
from check_relevent_document import check_relevence_document
from llm_response import LLm_response
from response_irrelevent_query import response_irrelevent_Question
from langgraph.graph import StateGraph,START,END
from typing_extensions import TypedDict
from typing import List,Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ChatSate(TypedDict):
    chat_history:Annotated[list[BaseMessage],add_messages]
    answer: BaseMessage
    is_relevance_query:str
    is_relevent_document:str
    docs:List[str]

Graph = StateGraph(state_schema=ChatSate)



Graph.add_node("relevance_query_check",relevance_query_check)
Graph.add_node("Retrieveal",Retrieveal)
Graph.add_node("check_relevence_document",check_relevence_document)
Graph.add_node("LLm_response",LLm_response)
Graph.add_node("response_irrelevent_Question",response_irrelevent_Question)


Graph.add_edge(START, "relevance_query_check")
Graph.add_conditional_edges(
    "relevance_query_check",
    lambda X: X['is_relevance_query'].strip().lower(),
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


def retrieve_all_threads():
  thread_list=set()
  for checkpoint in checkpointer.list(None):
    thread_list.add(checkpoint.config['configurable']['thread_id'])
  return list(thread_list)

def retrieve_all_threads_with_titles():
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    cursor.execute("SELECT thread_id, title FROM chat_titles")
    threads = {row[0]: row[1] for row in cursor.fetchall()}
    conn.close()
    return threads