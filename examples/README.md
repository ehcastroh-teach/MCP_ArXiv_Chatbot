# Advanced Examples

These files take the basic `arxiv_chatbot.py` pattern and elevate it to production-ready patterns you'll actually reach for in real applications. Each example is standalone and adds one specific capability on top of the base pattern from the lesson notebooks.

Read them in order — each builds on the previous one.

## Overview

| File | What it adds | Level |
|:-----|:-------------|:-----:|
| [`async_arxiv_chatbot.py`](async_arxiv_chatbot.py) | Concurrent tool execution using asyncio | ⭐⭐ |
| [`streaming_arxiv_chatbot.py`](streaming_arxiv_chatbot.py) | Token-by-token streaming responses | ⭐⭐ |
| [`fastapi_arxiv_server.py`](fastapi_arxiv_server.py) | Web API wrapper with session management | ⭐⭐⭐ |
| [`multi_tool_reflection.py`](multi_tool_reflection.py) | Self-reflecting agent that critiques its own answers | ⭐⭐⭐ |

## Why these examples

The base chatbot works, but real applications hit a few common walls:

- **Speed**: when the LLM calls three tools in a row, calling them serially is slow. `async_arxiv_chatbot.py` shows how to run them concurrently.
- **Perceived latency**: users hate silence. `streaming_arxiv_chatbot.py` shows the response as it's generated.
- **Multi-user**: a CLI serves one user; a web API serves many. `fastapi_arxiv_server.py` shows session-scoped conversations behind HTTP.
- **Quality**: LLM answers can be shallow. `multi_tool_reflection.py` shows a reflection loop that catches its own errors.

## Setup

All examples share the base dependencies from `requirements.txt` plus a few extras noted at the top of each file. Install extras as needed:

```bash
# Base (from repo root)
pip install -r requirements.txt

# For async example
pip install anthropic[bedrock,vertex]  # or just base anthropic — AsyncAnthropic is included

# For FastAPI example
pip install fastapi uvicorn

# For streaming example
# (uses base anthropic, no extras)
```

## Running

Each file has a `__main__` block. Run directly:

```bash
python examples/async_arxiv_chatbot.py
python examples/streaming_arxiv_chatbot.py
uvicorn examples.fastapi_arxiv_server:app --reload
python examples/multi_tool_reflection.py
```

## A note on progression

These examples are labelled ⭐⭐ or ⭐⭐⭐, but none of them are conceptually harder than the base notebook — they're just applying the same pattern to more demanding contexts. If you finished all three lesson notebooks, you have everything you need to read (and modify) these.
