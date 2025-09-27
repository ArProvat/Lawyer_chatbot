import streamlit as st
import uuid
from workflow import Workflow, retrieve_all_threads_with_titles
from title import title_generation , save_chat_title
from langchain_core.messages import HumanMessage, AIMessage

# ---------- Utility ----------
def generate_thread_id():
    return str(uuid.uuid4())


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(st.session_state["thread_id"])
    st.session_state["message_history"] = []


def add_thread(thread_id, title="New Chat"):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"][thread_id] = title
        save_chat_title(thread_id, "New Chat")


def load_conversation(thread_id):
    state = Workflow.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("chat_history", [])


# ---------- Session Setup ----------
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads_with_titles()


# ---------- Sidebar ----------
st.sidebar.title("Lawyer👨‍⚖️ Chatbot")

if st.sidebar.button("➕New Chat"):
    reset_chat()

st.sidebar.header("My Conversations")
for thread_id, title in st.session_state["chat_threads"].items():
    if st.sidebar.button(title, key=thread_id):
        st.session_state["thread_id"] = thread_id
        messages = load_conversation(thread_id)
        temp_messages = [
            {"role": "user" if isinstance(msg, HumanMessage) else "assistant", "content": msg.content}
            for msg in messages
        ]
        st.session_state["message_history"] = temp_messages

# ---------- Main UI ----------
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Type here")

if user_input:
    # Add user msg
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_turn",
    }

   
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Ai thinking...")
        full_ai_message = ""
    for message_chunk, metadata in Workflow.stream(
    {"chat_history": [HumanMessage(content=user_input)]},  # <-- input dict
    config=CONFIG,
    stream_mode="messages"
):
        if message_chunk.content and metadata.get('langgraph_node') == 'LLm_response':
            full_ai_message += message_chunk.content
            message_placeholder.markdown(full_ai_message)

    
    st.session_state['message_history'].append({'role': 'assistant', 'content': full_ai_message})

    if len(st.session_state["message_history"]) == 2:
        context = "\n".join(m["content"] for m in st.session_state["message_history"])
        title = title_generation(context)
        save_chat_title(st.session_state["thread_id"], title)
        st.session_state["chat_threads"][st.session_state["thread_id"]] = title
        st.rerun()