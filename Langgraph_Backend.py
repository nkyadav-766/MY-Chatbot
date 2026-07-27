from typing import TypedDict, Annotated
from dotenv import load_dotenv
import os

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# -----------------------------
# Load Environment Variables
# -----------------------------
load_dotenv()

hf_token = os.getenv("HF_TOKEN")

if hf_token is None:
    raise ValueError("HF_TOKEN not found in .env file")

# -----------------------------
# Load Hugging Face Model
# -----------------------------
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    huggingfacehub_api_token=hf_token,
    task="text-generation",
    max_new_tokens=512,
    temperature=0.7,
)

model = ChatHuggingFace(llm=llm)

# -----------------------------
# Define State
# -----------------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -----------------------------
# Chat Node
# -----------------------------
def chat_node(state: ChatState):
    response = model.invoke(state["messages"])
    return {"messages": [response]}

# -----------------------------
# Memory
# -----------------------------
checkpointer = MemorySaver()

# -----------------------------
# Build Graph
# -----------------------------
graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

