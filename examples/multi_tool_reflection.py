"""
multi_tool_reflection.py

Adds a self-reflection loop: after the LLM produces an answer, a second
"critic" pass looks for problems (missing citations, unsupported claims,
answering the wrong question). If it finds any, the loop runs again with the
critique fed back in.

Why this exists
---------------
Single-shot LLM answers are often shallow: they'll say "here are some papers"
without noting that two of them are duplicates, or answer a related question
instead of the one asked. A reflection pass catches this cheaply — you spend
one extra API call to significantly improve output quality.

The pattern
-----------
1. Answer normally (original process_query loop)
2. Pass the answer + the original question to a "critic" prompt
3. If the critic finds issues, re-run with the critique appended as guidance
4. Cap at N reflection rounds so you don't loop forever

When to reach for this
----------------------
- Answers must be factually grounded (research, medical, legal)
- The consequence of a bad answer is high (customer support, code changes)
- You have budget for the extra API calls

When NOT to bother
------------------
- Real-time chat where latency matters more than depth
- Simple lookups (the reflection pass has nothing useful to critique)
"""

import json
import os
from typing import List, Optional

import anthropic
import arxiv
from dotenv import load_dotenv

PAPER_DIR = "papers"
MAX_REFLECTIONS = 2  # how many critique-and-revise rounds before giving up

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
        pid = paper.get_short_id()
        paper_ids.append(pid)
        papers_info[pid] = {
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


def execute_tool(name: str, args: dict) -> str:
    result = tool_functions[name](**args)
    if isinstance(result, list):
        return ", ".join(str(x) for x in result)
    if isinstance(result, dict):
        return json.dumps(result, indent=2)
    return str(result) if result is not None else "Empty result."


# ---------------------------------------------------------------------------
# The standard answering loop
# ---------------------------------------------------------------------------
def answer_query(query: str, extra_guidance: Optional[str] = None) -> str:
    """Run one full answer pass. Returns the combined text of the final answer."""
    user_content = query
    if extra_guidance:
        user_content = f"{query}\n\n[Reviewer feedback from previous attempt]: {extra_guidance}"

    messages = [{"role": "user", "content": user_content}]
    collected: List[str] = []

    while True:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2048,
            tools=tools,
            messages=messages,
        )

        for block in response.content:
            if block.type == "text":
                collected.append(block.text)

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            return "\n".join(collected)

        messages.append({"role": "assistant", "content": response.content})
        tool_results = [
            {"type": "tool_result", "tool_use_id": b.id, "content": execute_tool(b.name, b.input)}
            for b in tool_uses
        ]
        messages.append({"role": "user", "content": tool_results})


# ---------------------------------------------------------------------------
# The reflection pass — the interesting part
# ---------------------------------------------------------------------------
# We use a separate, tool-less LLM call to critique the answer. Keeping it
# tool-less means the critic focuses on quality, not on gathering more info.

CRITIC_SYSTEM = """You are a careful reviewer of AI-generated answers about research papers.

Your job: read the original question and the AI's answer, then decide if the answer is good enough as-is or needs revision.

An answer needs revision if:
- It doesn't actually answer the specific question asked
- It makes claims about papers without citing paper IDs
- It repeats the same paper in a list without noting duplicates
- It hedges when the tools returned concrete information

Respond in this exact format:
STATUS: GOOD | NEEDS_REVISION
REASON: <one sentence — what's specifically wrong, or "answer is well-grounded and specific" if GOOD>
"""


def critique(query: str, answer: str) -> tuple:
    """Returns (status, reason). status is 'GOOD' or 'NEEDS_REVISION'."""
    critique_response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=300,
        system=CRITIC_SYSTEM,
        messages=[{
            "role": "user",
            "content": f"Original question:\n{query}\n\nAI's answer:\n{answer}"
        }],
    )

    text = "".join(b.text for b in critique_response.content if b.type == "text")
    status_line = next((l for l in text.splitlines() if l.startswith("STATUS:")), "STATUS: GOOD")
    reason_line = next((l for l in text.splitlines() if l.startswith("REASON:")), "REASON: (no reason given)")

    status = "NEEDS_REVISION" if "NEEDS_REVISION" in status_line else "GOOD"
    reason = reason_line.replace("REASON:", "").strip()
    return status, reason


# ---------------------------------------------------------------------------
# Orchestration: answer → critique → revise loop
# ---------------------------------------------------------------------------
def process_query_with_reflection(query: str) -> str:
    guidance: Optional[str] = None
    for round_num in range(MAX_REFLECTIONS + 1):
        answer = answer_query(query, extra_guidance=guidance)

        if round_num == MAX_REFLECTIONS:
            return answer  # out of rounds — return what we have

        status, reason = critique(query, answer)
        print(f"\n[Reflection round {round_num + 1}] {status}: {reason}\n")

        if status == "GOOD":
            return answer

        guidance = reason  # feed critique into next attempt

    return answer


def chat_loop():
    print("arXiv chatbot with self-reflection. Type 'quit' to exit.\n")
    while True:
        try:
            query = input("Query: ").strip()
            if query.lower() == "quit":
                break
            final = process_query_with_reflection(query)
            print(f"\n--- FINAL ANSWER ---\n{final}\n")
        except KeyboardInterrupt:
            print("\nInterrupted.")
            break
        except Exception as e:
            print(f"Error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    chat_loop()
