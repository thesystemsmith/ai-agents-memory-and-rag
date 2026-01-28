from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore
from langchain_core.runnables import RunnableConfig
import uuid


# llm
model = ChatOpenAI(
    model="qwen2.5:3b-instruct",
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)


# node
def call_model(
    state: MessagesState,
    config: RunnableConfig,
    *,
    store: BaseStore,
):
    user_id = config["configurable"]["user_id"]
    namespace = ("memories", user_id)

    # fetch memories
    memories = store.search(namespace, query=str(state["messages"][-1].content))
    info = "\n".join([d.value["data"] for d in memories])

    system_msg = f"You are a helpful assistant. User info: {info}"

    # detect "remember:" and store memory
    last_msg = state["messages"][-1].content.lower()
    if "remember:" in last_msg:
        memory = state["messages"][-1].content.split("remember:", 1)[1].strip()
        store.put(namespace, str(uuid.uuid4()), {"data": memory})

    # call local model
    response = model.invoke(
        [{"role": "system", "content": system_msg}] + state["messages"]
    )

    return {"messages": response}


# build graph
builder = StateGraph(MessagesState)
builder.add_node(call_model)
builder.add_edge(START, "call_model")

# short term mem
checkpointer = InMemorySaver()

# long term mem
store = InMemoryStore()

graph = builder.compile(
    checkpointer=checkpointer,
    store=store,
)


# demo
config1 = {
    "configurable": {
        "thread_id": "1",
        "user_id": "99",
    }
}

print("\n--- Turn 1: teach memory ---")
for chunk in graph.stream(
    {"messages": [{"role": "user", "content": "remember: my name is bobo"}]},
    config1,
    stream_mode="values",
):
    chunk["messages"][-1].pretty_print()


config2 = {
    "configurable": {
        "thread_id": "2",   # NEW THREAD
        "user_id": "99",    # SAME USER
    }
}

print("\n--- Turn 2: recall memory ---")
for chunk in graph.stream(
    {"messages": [{"role": "user", "content": "what is my name?"}]},
    config2,
    stream_mode="values",
):
    chunk["messages"][-1].pretty_print()
