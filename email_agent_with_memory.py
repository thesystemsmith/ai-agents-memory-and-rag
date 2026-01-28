from __future__ import annotations          # Allows forward references in type hints (cleaner typing, fewer import issues)

import uuid                                 # Used to generate unique IDs (e.g., memory keys, message IDs)

from typing import TypedDict, Literal, Dict, Any, List  
# TypedDict: structured state shape
# Literal: restrict values (e.g., "respond" | "ignore")
# Dict, Any, List: generic typing for flexibility
from langchain_openai import ChatOpenAI     # LLM interface (OpenAI models)
from langchain_core.runnables import RunnableConfig  # Carries runtime metadata like user_id, thread_id, tracing
from langchain_core.tools import tool       # Decorator to expose Python functions as agent tools
from langgraph.graph import StateGraph, START, END  
# StateGraph: workflow graph
# START/END: graph entry and exit points
from langgraph.prebuilt import create_react_agent  # Creates a ReAct-style agent (reason + act using tools)
from langgraph.checkpoint.mongodb import MongoDBSaver  # Persists graph state (short-term memory) in MongoDB
# long-term store (needs: pip install langgraph-store-mongodb)
from langgraph.store.mongodb import MongoDBStore
from langgraph.store.base import BaseStore


# ------------------------------------------------------------
# 0) Local LLM (Ollama via OpenAI-compatible endpoint)
# ------------------------------------------------------------
# Requirements:
#   ollama serve
#   ollama pull qwen2.5:3b-instruct
llm = ChatOpenAI(
    model='qwen2.5:3b-instruct',
    base_url='http://localhost:11434/v1',
    api_key='ollama',  # dummy; Ollama ignores
    temperature=0.0,
)

MONGO_URI = 'mongodb://localhost:27017/'
DB_NAME = 'email_agent_demo'


# ------------------------------------------------------------
# 1) State = agent "working memory" (thread-level scratchpad)
# ------------------------------------------------------------
class State(TypedDict):
    email_input: dict
    triage_result: Literal['ignore', 'notify', 'respond']
    response_text: str
    
    
# ------------------------------------------------------------
# 2) Memory namespaces (3 kinds)
# ------------------------------------------------------------
def ns_examples(user_id: str):
    # EPISODIC MEMORY: past labeled emails
    return ('email_agent', user_id, 'examples')

def ns_prompts(user_id: str):
    # PROCEDURAL MEMORY: prompts/playbook the agent follows
    return ('email_agent', user_id, 'prompts')

def ns_facts(user_id: str):
    # SEMANTIC MEMORY: facts/context about clients/projects
    return ('email_agent', user_id, 'facts')


# ------------------------------------------------------------
# 3) Initialize memory (few-shot examples + default prompts)
# ------------------------------------------------------------
DEFAULT_TRIAGE_PROMPT = """You are an email triage assistant.
Classify the email as exactly one of: ignore, notify, respond.

Use examples below as guidance.

EXAMPLES:
{examples}

EMAIL:
From: {author}
To: {to}
Subject: {subject}
Body: {email_thread}

Return ONLY one word: ignore OR notify OR respond.
"""

DEFAULT_RESPONSE_PROMPT = """You are an email reply assistant.
Use tools if helpful.
Be concise and professional.
If you know relevant facts about the user/projects/clients, use them.
"""


def initialize_memory(store: BaseStore, user_id: str):
    # EPISODIC example: spam -> ignore
    store.put(
        ns_examples(user_id),
        'ex_spam',
        {
            'email': {
                'author': 'Spammy Marketer <spam@example.com>',
                'to': 'You <you@company.com>',
                'subject': 'BIG SALE!!!',
                'email_thread': 'Buy now and get 50% off!!!',
            },
            'label': 'ignore',
        },
    )

    # EPISODIC example: real client question -> respond
    store.put(
        ns_examples(user_id),
        'ex_client',
        {
            'email': {
                'author': 'Alice <alice@client.com>',
                'to': 'You <you@company.com>',
                'subject': 'Need help with API docs',
                'email_thread': 'Endpoints missing in docs. Can you help?',
            },
            'label': 'respond',
        },
    )

    # PROCEDURAL memory: store prompts/playbook
    store.put(ns_prompts(user_id), 'triage_prompt', {'text': DEFAULT_TRIAGE_PROMPT})
    store.put(ns_prompts(user_id), 'response_prompt', {'text': DEFAULT_RESPONSE_PROMPT})

    # SEMANTIC memory: store some facts (project context)
    store.put(ns_facts(user_id), 'fact_1', {'data': 'Project Phoenix API base URL is /v2.'})
    store.put(ns_facts(user_id), 'fact_2', {'data': 'Client Alice prefers short answers and ETA in replies.'})


# ------------------------------------------------------------
# 4) Helpers: format episodic examples for triage prompt
# ------------------------------------------------------------
def format_examples(items: List[Any]) -> str:
    out = []
    for it in items:
        email = it.value['email']
        label = it.value['label']
        out.append(
            f"From: {email['author']}\n"
            f"Subject: {email['subject']}\n"
            f"Body: {email['email_thread']}\n"
            f"Classification: {label}"
        )
    return '\n\n'.join(out) if out else '(no examples)'


# ------------------------------------------------------------
# 5) TRIAGE node (Episodic + Procedural memory)
# ------------------------------------------------------------
def triage_email(state: State, config: RunnableConfig, *, store: BaseStore) -> Dict[str, Any]:
    user_id = config['configurable']['user_id']
    email = state['email_input']

    # PROCEDURAL: load the triage prompt template (playbook)
    triage_prompt_item = store.get(ns_prompts(user_id), 'triage_prompt')
    triage_prompt = triage_prompt_item.value['text']

    # EPISODIC: retrieve similar examples (few-shot)
    examples = store.list(ns_examples(user_id), query=str(email))
    formatted = format_examples(examples)

    prompt = triage_prompt.format(examples=formatted, **email)

    resp = llm.invoke([{'role': 'user', 'content': prompt}]).content.strip().lower()

    # normalize output
    if 'respond' in resp:
        label = 'respond'
    elif 'notify' in resp:
        label = 'notify'
    else:
        label = 'ignore'

    return {'triage_result': label}


# ------------------------------------------------------------
# 6) Tools for response agent (Semantic memory tools + mock actions)
# ------------------------------------------------------------
@tool
def write_email(to: str, subject: str, content: str) -> str:
    """Draft/send an email (mock)."""
    print(f"\n[SEND EMAIL]\nTo: {to}\nSubject: {subject}\n\n{content}\n")
    return 'sent'

@tool
def check_calendar(day: str) -> str:
    """Check calendar availability (mock)."""
    return f'Available times on {day}: 9:00, 14:00, 16:00'

@tool
def remember_fact(user_id: str, text: str) -> str:
    """Store a semantic fact for this user."""
    # NOTE: store is not directly available inside tool in this minimal version.
    # We keep it simple: this tool will be injected via closure below.
    return 'not wired'


@tool
def search_facts(user_id: str, query: str) -> str:
    """Search semantic facts for this user."""
    return 'not wired'


def build_semantic_tools(store: BaseStore):
    # We bind store into tools using closures (simple + clear).
    @tool
    def remember_fact_bound(user_id: str, text: str) -> str:
        """Store a semantic fact for this user."""
        store.put(ns_facts(user_id), str(uuid.uuid4()), {'data': text})
        return 'stored'

    @tool
    def search_facts_bound(user_id: str, query: str) -> str:
        """Search semantic facts for this user."""
        hits = store.search(ns_facts(user_id), query=query)
        facts = [h.value['data'] for h in hits]
        return '\n'.join(facts) if facts else '(no facts found)'

    return [remember_fact_bound, search_facts_bound]



# ------------------------------------------------------------
# 7) RESPONSE agent node (Semantic + Procedural memory)
# ------------------------------------------------------------
def make_response_agent(store: BaseStore):
    tools = [write_email, check_calendar] + build_semantic_tools(store)

    def response_prompt(state: Dict[str, Any], config: RunnableConfig, store: BaseStore):
        user_id = config['configurable']['user_id']

        # PROCEDURAL: load response prompt from memory (playbook)
        p = store.get(ns_prompts(user_id), 'response_prompt').value['text']

        # SEMANTIC: fetch some relevant facts and inject as context
        email = state['email_input']
        hits = store.search(ns_facts(user_id), query=str(email))
        facts = '\n'.join([h.value['data'] for h in hits]) or '(none)'

        system = (
            f"{p}\n\n"
            f"Known relevant facts:\n{facts}\n\n"
            "Now draft a reply. Keep it short and useful."
        )

        # ReAct agent expects messages format
        return [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': f"Email:\n{email}"},
        ]

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=response_prompt,
        store=store,
        name='response_agent',
    )

def run_response_agent(state: State, config: RunnableConfig, *, store: BaseStore) -> Dict[str, Any]:
    agent = make_response_agent(store)
    out = agent.invoke({'email_input': state['email_input']}, config=config)
    # agent returns messages; we store final text in state
    final_text = out['messages'][-1].content
    return {'response_text': final_text}



# ------------------------------------------------------------
# 8) Routing
# ------------------------------------------------------------
def route_after_triage(state: State) -> str:
    return 'response' if state['triage_result'] == 'respond' else 'end'



# ------------------------------------------------------------
# 9) Build graph
# ------------------------------------------------------------
def build_graph():
    workflow = StateGraph(State)

    workflow.add_node('triage', triage_email)
    workflow.add_node('response', run_response_agent)

    workflow.add_edge(START, 'triage')
    workflow.add_conditional_edges('triage', route_after_triage, {'response': 'response', 'end': END})
    workflow.add_edge('response', END)

    return workflow



# ------------------------------------------------------------
# 10) Demo run (two emails)
# ------------------------------------------------------------
def main():
    user_id = '99'
    config = RunnableConfig(configurable={'user_id': user_id, 'thread_id': 'thread-1'})

    with MongoDBStore.from_conn_string(MONGO_URI, db_name=DB_NAME) as store, \
         MongoDBSaver.from_conn_string(MONGO_URI, db_name=DB_NAME) as checkpointer:

        # first time only (safe to run again; will overwrite keys)
        initialize_memory(store, user_id)

        graph = build_graph().compile(store=store, checkpointer=checkpointer)

        # Email 1: spam -> should ignore
        email1 = {
            'author': 'Spammy Marketer <spam@example.com>',
            'to': 'You <you@company.com>',
            'subject': 'BIG SALE!!!',
            'email_thread': 'Buy our product now and get 50% off!',
        }

        out1 = graph.invoke({'email_input': email1, 'triage_result': 'ignore', 'response_text': ''}, config=config)
        print('\n=== EMAIL 1 RESULT ===')
        print('triage_result:', out1['triage_result'])
        print('response_text:', out1['response_text'])

        # Email 2: client asks about API docs -> should respond
        email2 = {
            'author': 'Alice <alice@client.com>',
            'to': 'You <you@company.com>',
            'subject': 'Quick question about API documentation',
            'email_thread': 'Hi, I noticed some endpoints missing in docs. Can you help?',
        }

        config2 = RunnableConfig(configurable={'user_id': user_id, 'thread_id': 'thread-2'})
        out2 = graph.invoke({'email_input': email2, 'triage_result': 'ignore', 'response_text': ''}, config=config2)

        print('\n=== EMAIL 2 RESULT ===')
        print('triage_result:', out2['triage_result'])
        print('response_text:\n', out2['response_text'])


if __name__ == '__main__':
    main()