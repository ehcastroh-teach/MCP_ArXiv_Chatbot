"""
async_arxiv_chatbot.py

Async version of the arXiv chatbot with concurrent tool execution.

Why this exists
---------------
The base chatbot (arxiv_chatbot.py) runs tools one at a time. When the LLM
requests three tools in a single turn, we call them serially — total time is
the sum of every tool's latency.

This version uses asyncio to run them concurrently. For I/O-bound tools
(arXiv API, file reads, HTTP calls), you get near-linear speedup with almost
no extra code.

When to reach for this
----------------------
- Tools do network calls (arXiv, HTTP APIs, databases)
- The LLM often requests 2+ tools per turn
- You care about end-to-end latency

When NOT to bother
------------------
- Tools are CPU-bound (asyncio does nothing for you here — use multiprocessing)
- The LLM almost always calls one tool at a time (no concurrency to exploit)
- You're prototyping and speed doesn't matter yet
"""

import asyncio
import json
import os
from typing import List

import anthropic
import arxiv
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
PAPER_DIR = "papers"
load_dotenv()
async_client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


# ---------------------------------------------------------------------------
# Async tool implementations
# ---------------------------------------------------------------------------
# Note: the arxiv library is synchronous. We wrap sync calls in asyncio.to_thread
# so they don't block the event loop. This is the standard pattern for adopting
# async in a codebase that has sync dependencies.

async def search_papers(topic: str, max_results: int = 5) -> List[str]:
    """Search arXiv concurrently-safe: wraps the sync arxiv client in a thread."""
    return await asyncio.to_thread(_search_papers_sync, topic, max_results)


def _search_papers_sync(topic: str, max_results: int) -> List[str]:
    client = arxiv.Client()
    search = arxiv.Search(query=topic, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
    papers = client.results(search)

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


async def extract_info(paper_id: str) -> str:
    """Look up stored paper metadata. Async wrapper over sync file I/O."""
    return await asyncio.to_thread(_extract_info_sync, paper_id)


def _extract_info_sync(paper_id: str) -> str:
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


# ---------------------------------------------------------------------------
# Schemas and dispatcher
# ---------------------------------------------------------------------------
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
            "properties": {
                "paper_id": {"type": "string", "description": "arXiv ID"},
            },
            "required": ["paper_id"],
        },
    },
]

tool_functions = {
    "search_papers": search_papers,
    "extract_info": extract_info,
}


async def execute_tool(tool_name: str, tool_args: dict) -> str:
    """Execute a single tool and normalize the result."""
    fn = tool_functions[tool_name]
    result = await fn(**tool_args)

    if result is None:
        return "The operation completed but didn't return any results."
    if isinstance(result, list):
        return ", ".join(str(x) for x in result)
    if isinstance(result, dict):
        return json.dumps(result, indent=2)
    return str(result)


# ---------------------------------------------------------------------------
# The key idea: gather concurrent tool calls
# ---------------------------------------------------------------------------
# When one LLM response contains multiple tool_use blocks, we run them all
# concurrently with asyncio.gather. On I/O-bound tools, wall-clock latency
# drops from sum(latencies) to max(latencies).

async def execute_all_tools(tool_use_blocks) -> List[dict]:
    """Run every tool_use in the response concurrently. Preserves order."""
    tasks = [execute_tool(block.name, block.input) for block in tool_use_blocks]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    tool_results = []
    for block, result in zip(tool_use_blocks, results):
        content = result if isinstance(result, str) else f"Tool error: {result}"
        tool_results.append({
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": content,
        })
    return tool_results


# ---------------------------------------------------------------------------
# Async loop
# ---------------------------------------------------------------------------
async def process_query(query: str) -> None:
    messages = [{"role": "user", "content": query}]

    while True:
        response = await async_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2048,
            tools=tools,
            messages=messages,
        )

        # Collect and display text; collect tool_use blocks for concurrent dispatch
        text_blocks = [b for b in response.content if b.type == "text"]
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

        for block in text_blocks:
            print(block.text)

        if not tool_use_blocks:
            return  # LLM is done

        # Log what's about to run concurrently
        print(f"\n[Running {len(tool_use_blocks)} tool(s) concurrently: "
              f"{', '.join(b.name for b in tool_use_blocks)}]\n")

        # Preserve the LLM's turn in history
        messages.append({"role": "assistant", "content": response.content})

        # Concurrent execution — the win happens here
        tool_results = await execute_all_tools(tool_use_blocks)

        # Send all results back in one user message
        messages.append({"role": "user", "content": tool_results})


async def chat_loop() -> None:
    print("Async arxiv chatbot. Type 'quit' to exit.\n")
    loop = asyncio.get_event_loop()
    while True:
        query = await loop.run_in_executor(None, lambda: input("Query: ").strip())
        if query.lower() == "quit":
            break
        try:
            await process_query(query)
            print()
        except Exception as e:
            print(f"Error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    asyncio.run(chat_loop())
