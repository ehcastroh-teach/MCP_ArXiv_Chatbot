"""
fastapi_arxiv_server.py

Wraps the arXiv chatbot behind a FastAPI web server with per-session conversation
memory.

Why this exists
---------------
A CLI serves one user, sequentially. A web API serves many users, concurrently,
each with their own conversation history. That's the deployment shape most
production chatbots take: a REST endpoint that a web/mobile frontend calls.

The two things this file adds on top of the base chatbot:

1. Session storage — a dict keyed by session_id, holding the messages list
   for that user's conversation. In production you'd use Redis or a database;
   this file uses an in-memory dict to keep the code readable.

2. Request handling — accept a JSON POST, run process_query, return the final
   text answer. Tool calls happen server-side, invisible to the client.

Running
-------
    pip install fastapi uvicorn
    uvicorn examples.fastapi_arxiv_server:app --reload

Then POST to http://localhost:8000/query with:
    {"session_id": "user-42", "query": "Find papers about LLMs"}

Session limits (real-world concerns)
------------------------------------
This code has no per-session token limits, no rate limiting, and no auth.
Before shipping anything like this to real users, add all three.
"""

import json
import os
from typing import Dict, List

import anthropic
import arxiv
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
PAPER_DIR = "papers"
load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

app = FastAPI(title="arXiv MCP Chatbot API", version="1.0")

# Simple in-memory session store. Key: session_id, Value: list of messages.
# WARNING: this is lost on server restart. Use Redis/DB for production.
SESSIONS: Dict[str, List[dict]] = {}


# ---------------------------------------------------------------------------
# Tool implementations (same as base)
# ---------------------------------------------------------------------------
def search_papers(topic: str, max_results: int = 5) -> List[str]:
    arxiv_client = arxiv.Client()
    search = arxiv.Search(query=topic, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
    papers = arxiv_client.results(search)

    path = os.path.join(PAPER_DIR, topic.lower().replace(" ", "_"))
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, "papers_info.json")

    try:
        with open(file_path, "r") as f:
            papers_info = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        papers_info = {}

    paper_ids = []
    for paper in papers:
        paper_id = paper.get_short_id()
        paper_ids.append(paper_id)
        papers_info[paper_id] = {
            "title": paper.title,
            "authors": [a.name for a in paper.authors],
            "summary": paper.summary,
            "pdf_url": paper.pdf_url,
            "published": str(paper.published.date()),
        }
    with open(file_path, "w") as f:
        json.dump(papers_info, f, indent=2)
    return paper_ids


def extract_info(paper_id: str) -> str:
    for item in os.listdir(PAPER_DIR):
        item_path = os.path.join(PAPER_DIR, item)
        if os.path.isdir(item_path):
            file_path = os.path.join(item_path, "papers_info.json")
            if os.path.isfile(file_path):
                try:
                    with open(file_path, "r") as f:
                        papers_info = json.load(f)
                        if paper_id in papers_info:
                            return json.dumps(papers_info[paper_id], indent=2)
                except (FileNotFoundError, json.JSONDecodeError):
                    continue
    return f"There's no saved information related to paper {paper_id}."


tools = [
    {
        "name": "search_papers",
        "description": "Search arXiv for papers on a topic and store their metadata.",
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "The topic to search for"},
                "max_results": {"type": "integer", "description": "Max results", "default": 5},
            },
            "required": ["topic"],
        },
    },
    {
        "name": "extract_info",
        "description": "Retrieve stored metadata for a paper by ID.",
        "input_schema": {
            "type": "object",
            "properties": {"paper_id": {"type": "string", "description": "arXiv ID"}},
            "required": ["paper_id"],
        },
    },
]

tool_functions = {"search_papers": search_papers, "extract_info": extract_info}


def execute_tool(tool_name: str, tool_args: dict) -> str:
    fn = tool_functions.get(tool_name)
    if fn is None:
        return f"Error: unknown tool {tool_name!r}"
    try:
        result = fn(**tool_args)
    except Exception as e:
        return f"Error calling {tool_name}: {type(e).__name__}: {e}"

    if result is None:
        return "The operation completed but didn't return any results."
    if isinstance(result, list):
        return ", ".join(str(x) for x in result)
    if isinstance(result, dict):
        return json.dumps(result, indent=2)
    return str(result)


# ---------------------------------------------------------------------------
# Session-aware conversation loop
# ---------------------------------------------------------------------------
def process_query(session_id: str, query: str) -> str:
    """Runs one query through the loop, appending to per-session history.

    Returns the concatenated text of all assistant text blocks."""
    messages = SESSIONS.setdefault(session_id, [])
    messages.append({"role": "user", "content": query})

    collected_text: List[str] = []

    while True:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2048,
            tools=tools,
            messages=messages,
        )

        for block in response.content:
            if block.type == "text":
                collected_text.append(block.text)

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            # LLM is done. Persist its final turn and return.
            messages.append({"role": "assistant", "content": response.content})
            break

        # Persist the LLM's turn and run tools
        messages.append({"role": "assistant", "content": response.content})
        tool_results = [
            {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": execute_tool(block.name, block.input),
            }
            for block in tool_uses
        ]
        messages.append({"role": "user", "content": tool_results})

    return "\n".join(collected_text)


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    session_id: str
    query: str


class QueryResponse(BaseModel):
    session_id: str
    answer: str
    turn_count: int


@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    """Process one user query in the context of a session's history."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")

    answer = process_query(req.session_id, req.query)
    return QueryResponse(
        session_id=req.session_id,
        answer=answer,
        turn_count=len(SESSIONS.get(req.session_id, [])),
    )


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    """Reset a user's conversation history."""
    SESSIONS.pop(session_id, None)
    return {"session_id": session_id, "cleared": True}


@app.get("/session/{session_id}/length")
def session_length(session_id: str):
    """How many turns are stored for this session? Useful for debugging."""
    return {"session_id": session_id, "message_count": len(SESSIONS.get(session_id, []))}


@app.get("/healthz")
def healthz():
    return {"status": "ok", "sessions": len(SESSIONS)}
