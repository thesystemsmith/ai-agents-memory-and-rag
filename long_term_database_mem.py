from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph_store_mongodb import MongoDBStore
from langgraph.store.base import BaseStore
from langchain_core.runnables import RunnableConfig
import uuid


# LLM (local)
model = ChatOllama(
    model="qwen2.5:3b-instruct",
    temperature=0.0,
)

# MongoDB details
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "agent_memory_db"


# node logic (long-term memory)
def call_model(
    state: MessagesState,
    config: RunnableConfig,
    *,
    store: BaseStore,
):
    user_id = config["configurable"]["user_id"]
    namespace = ("memories", user_id)

    # retrieve memories
    memories = store.search(namespace, query=state["messages"][-1].content)
    remembered_info = "\n".join([m.value["data"] for m in memories])

    system_msg = (
        "You are a helpful assistant.\n"
        f"Long-term memory about user:\n{remembered_info}"
    )

    # store new memory
    last_msg = state["messages"][-1].content.lower()
    if "remember:" in last_msg:
        memory = last_msg.split("remember:", 1)[1].strip()
        store.put(namespace, str(uuid.uuid4()), {"data": memory})

    # llm invoke
    response = model.invoke(
        [{"role": "system", "content": system_msg}] + state["messages"]
    )
    return {"messages": response}


# graph builder
builder = StateGraph(MessagesState)
builder.add_node(call_model)
builder.add_edge(START, "call_model")


def test_memory_flow():
    # MUST use context managers
    with MongoDBStore.from_conn_string(MONGO_URI, db_name=DB_NAME) as store, \
         MongoDBSaver.from_conn_string(MONGO_URI, db_name=DB_NAME) as checkpointer:

        graph = builder.compile(checkpointer=checkpointer, store=store)

        print("\n--- THREAD 1: STORE MEMORY ---")
        config = {"configurable": {"thread_id": "1", "user_id": "99"}}

        for chunk in graph.stream(
            {"messages": [{"role": "user", "content": "Hi! Remember: my favorite bike is cruiser"}]},
            config,
            stream_mode="values",
        ):
            chunk["messages"][-1].pretty_print()

        print("\n--- THREAD 2: RETRIEVE MEMORY ---")
        config = {"configurable": {"thread_id": "2", "user_id": "99"}}

        for chunk in graph.stream(
            {"messages": [{"role": "user", "content": "What is my favorite bike?"}]},
            config,
            stream_mode="values",
        ):
            chunk["messages"][-1].pretty_print()


if __name__ == "__main__":
    test_memory_flow()
