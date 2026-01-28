from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage, AIMessage


# llm
model = ChatOpenAI(
    model = "qwen2.5:3b-instruct",
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    temperature = 0.0
)


# node
def call_model(state: MessagesState):
    """
    State contains a list of messages:
        state["messages"] → [... conversation history …]

    This function calls the LLM and appends its response.
    """
    response = model.invoke(state["messages"])
    return {"messages": response}


# build graph
builder = StateGraph(MessagesState)

builder.add_node("llm", call_model)
builder.add_edge(START, "llm")


# add memory
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)


config = {
    "configurable": {
        "thread_id": "thread_1"   # all messages in this thread are remembered
    }
}


# demo
def main():
    print("\n---- USER: hi! I'm Rashid ----\n")
    for chunk in graph.stream(
        {"messages": [{"role": "user", "content": "hi! I'm Rashid"}]},
        config=config,
        stream_mode="values",
    ):
        print(chunk["messages"][-1])

    # Second message: ask model to recall the name
    print("\n---- USER: what's my name? ----\n")
    for chunk in graph.stream(
        {"messages": [{"role": "user", "content": "what's my name?"}]},
        config=config,
        stream_mode="values",
    ):
        print(chunk["messages"][-1])
        
if __name__ == '__main__':
    main()
    
    