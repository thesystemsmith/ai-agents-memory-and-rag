from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.mongodb import MongoDBSaver

from langchain_core.runnables import RunnableConfig


# llm
model = ChatOpenAI(
    base_url='http://localhost:11434/v1', 
    api_key='ollama',                    
    model='qwen2.5:3b-instruct',
    temperature=0.0,
)


# node
def call_model(state: MessagesState):
    """
    Single node:
    - Takes conversation messages from state["messages"]
    - Asks the LLM
    - Returns new messages appended (LangGraph merges them)
    """
    response = model.invoke(state['messages'])
    return {'messages': response}


# build graph
def build_graph():
    graph = StateGraph(MessagesState)
    graph.add_node('chat', call_model)
    graph.add_edge(START, 'chat')
    return graph


# memory
MONGODB_URI = 'mongodb://localhost:27017'
DB_NAME = 'langgraph_short_term_demo' 

def main():
    # Create checkpointer that writes checkpoints into MongoDB
    with MongoDBSaver.from_conn_string(MONGODB_URI, DB_NAME) as checkpointer:
        # Build graph and compile with checkpointer
        builder = build_graph()
        graph = builder.compile(checkpointer=checkpointer)

        # Thread config:
        # All calls that share this thread_id will share the SAME memory
        config = RunnableConfig(
            configurable={
                'thread_id': 'user-1'
            }
        )

        # -------- Turn 1 --------
        print('\n--- Turn 1 ---\n')
        for chunk in graph.stream(
            {'messages': [{'role': 'user', 'content': "hi! I'm Bob"}]},
            config=config,
            stream_mode='values',
        ):
            # print only latest message each time
            chunk['messages'][-1].pretty_print()

        # -------- Turn 2 --------
        print('\n--- Turn 2 ---\n')
        for chunk in graph.stream(
            {'messages': [{'role': 'user', 'content': "what's my name?"}]},
            config=config,
            stream_mode='values',
        ):
            chunk['messages'][-1].pretty_print()

        print('\nDone. Memory is stored in MongoDB (short-term per thread_id).\n')


if __name__ == '__main__':
    main()