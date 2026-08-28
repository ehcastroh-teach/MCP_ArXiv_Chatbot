"""
streaming_arxiv_chatbot.py

Streaming version — the LLM's response prints token-by-token instead of arriving
in one block at the end.

Why this matters
----------------
The base chatbot waits for the full response before printing anything. For long
answers, the user stares at a blinking cursor for 5-10 seconds. Streaming shows
the first tokens within ~200ms, dramatically improving perceived responsiveness.

The trade-off: streaming makes the code slightly harder because tool_use blocks
don't fully arrive one-token-at-a-time — you accumulate them across events, then
dispatch when the block is complete.

When to reach for this
----------------------
- Interactive UIs where perceived latency matters (CLIs, chat interfaces)
- Any user-facing application

When NOT to bother
------------------
- Batch processing (nobody's watching)
- You'll parse the whole response programmatically anyway
"""

import json
import os
from typing import List

import anthropic
import arxiv
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Setup (same as base)
# ---------------------------------------------------------------------------
PAPER_DIR = "papers"
load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


# ---------------------------------------------------------------------------
# Tools (same as base)
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
        "description": "Search arXiv for papers on a topic.",
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "The topic"},
                "max_results": {"type": "integer", "description": "Max results", "default": 5},
            },
            "required": ["topic"],
        },
    },
    {
        "name": "extract_info",
        "description": "Get details for a paper by ID.",
        "input_schema": {
            "type": "object",
            "properties": {"paper_id": {"type": "string", "description": "arXiv ID"}},
            "required": ["paper_id"],
        },
    },
]

tool_functions = {"search_papers": search_papers, "extract_info": extract_info}


def execute_tool(tool_name: str, tool_args: dict) -> str:
    fn = tool_functions[tool_name]
    result = fn(**tool_args)
    if isinstance(result, list):
        return ", ".join(str(x) for x in result)
    if isinstance(result, dict):
        return json.dumps(result, indent=2)
    return str(result) if result is not None else "Empty result."


# ---------------------------------------------------------------------------
# Streaming loop — this is where the interesting code lives
# ---------------------------------------------------------------------------
# The SDK's client.messages.stream() returns a context manager. Inside, we
# iterate .text_stream to get text as it arrives, and use .get_final_message()
# to get the fully-assembled response (with any tool_use blocks) at the end.
#
# The trick: we can print text tokens live, but tool_use blocks must be
# dispatched only after the response is fully assembled.

def process_query_streaming(query: str) -> None:
    messages = [{"role": "user", "content": query}]

    while True:
        with client.messages.stream(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2048,
            tools=tools,
            messages=messages,
        ) as stream:
            # Print text as it arrives — this is the streaming win
            for text_chunk in stream.text_stream:
                print(text_chunk, end="", flush=True)

            # After streaming ends, get the fully-assembled final message
            final = stream.get_final_message()

        print()  # newline after the streamed text

        tool_uses = [b for b in final.content if b.type == "tool_use"]
        if not tool_uses:
            return  # done — no more tools to call

        # Preserve LLM turn
        messages.append({"role": "assistant", "content": final.content})

        # Execute tools and package results
        tool_results = []
        for block in tool_uses:
            print(f"\n[Tool: {block.name}({block.input})]")
            result = execute_tool(block.name, block.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
            })

        messages.append({"role": "user", "content": tool_results})


def chat_loop() -> None:
    print("Streaming arxiv chatbot. Type 'quit' to exit.\n")
    while True:
        try:
            query = input("Query: ").strip()
            if query.lower() == "quit":
                break
            process_query_streaming(query)
            print()
        except KeyboardInterrupt:
            print("\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {type(e).__name__}: {e}")


if __name__ == "__main__":
    chat_loop()
