"""
arxiv_chatbot.py

An interactive chatbot that uses an LLM with tool support to help users
search for academic papers on arXiv and retrieve metadata about them.

Tools:
- search_papers: Query arXiv for papers on a given topic and store metadata.
- extract_info: Retrieve stored metadata for a specific paper by ID.

The chatbot runs in an interactive loop, processing user queries and
invoking tools when necessary.
"""

import arxiv
import json
import os
from typing import List
from dotenv import load_dotenv
import anthropic


# Root directory where paper metadata will be stored
PAPER_DIR = "papers"

# Anthropic API access
load_dotenv()
API_KEY = os.getenv("ANTHROPIC_API_KEY") 
client = anthropic.Anthropic(api_key=API_KEY)

# -----------------------------
# Tool Implementations
# -----------------------------
def search_papers(topic: str, max_results: int = 5) -> List[str]:
    """
    Search for papers on arXiv based on a topic and store their information.

    Args:
        topic (str): The topic/keyword to search for.
        max_results (int, optional): Maximum number of results to retrieve.
            Defaults to 5.

    Returns:
        List[str]: List of arXiv paper IDs found in the search.
    """
    client = arxiv.Client()

    search = arxiv.Search(
        query=topic,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    papers = client.results(search)

    path = os.path.join(PAPER_DIR, topic.lower().replace(" ", "_"))
    os.makedirs(path, exist_ok=True)

    file_path = os.path.join(path, "papers_info.json")

    try:
        with open(file_path, "r") as json_file:
            papers_info = json.load(json_file)
    except (FileNotFoundError, json.JSONDecodeError):
        papers_info = {}

    paper_ids = []
    for paper in papers:
        paper_id = paper.get_short_id()
        paper_ids.append(paper_id)

        papers_info[paper_id] = {
            "title": paper.title,
            "authors": [author.name for author in paper.authors],
            "summary": paper.summary,
            "pdf_url": paper.pdf_url,
            "published": str(paper.published.date()),
        }

    with open(file_path, "w") as json_file:
        json.dump(papers_info, json_file, indent=2)

    print(f"Results are saved in: {file_path}")
    return paper_ids


def extract_info(paper_id: str) -> str:
    """
    Retrieve stored metadata for a specific paper by ID.

    Args:
        paper_id (str): The arXiv short ID of the paper to look up.

    Returns:
        str: JSON-formatted string with the paper's metadata if found.
             Otherwise, a human-readable error message.
    """
    for item in os.listdir(PAPER_DIR):
        item_path = os.path.join(PAPER_DIR, item)
        if os.path.isdir(item_path):
            file_path = os.path.join(item_path, "papers_info.json")
            if os.path.isfile(file_path):
                try:
                    with open(file_path, "r") as json_file:
                        papers_info = json.load(json_file)
                        if paper_id in papers_info:
                            return json.dumps(papers_info[paper_id], indent=2)
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    print(f"Error reading {file_path}: {str(e)}")
                    continue
    return f"There's no saved information related to paper {paper_id}."


# -----------------------------
# Tool Schemas
# -----------------------------
tools = [
    {
        "name": "search_papers",
        "description": "Search for papers on arXiv based on a topic and store their information.",
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The topic to search for"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to retrieve",
                    "default": 5
                }
            },
            "required": ["topic"]
        }
    },
    {
        "name": "extract_info",
        "description": "Search for information about a specific paper across all topic directories.",
        "input_schema": {
            "type": "object",
            "properties": {
                "paper_id": {
                    "type": "string",
                    "description": "The ID of the paper to look for"
                }
            },
            "required": ["paper_id"]
        }
    }
]

# -----------------------------
# Tool Dispatcher
# -----------------------------
mapping_tool_function = {
    "search_papers": search_papers,
    "extract_info": extract_info
}

def execute_tool(tool_name: str, tool_args: dict) -> str:
    """
    Execute a registered tool function by name with provided arguments.

    Args:
        tool_name (str): The name of the tool to execute.
        tool_args (dict): Arguments to pass to the tool.

    Returns:
        str: String representation of the tool's result.
    """
    result = mapping_tool_function[tool_name](**tool_args)

    if result is None:
        return "The operation completed but didn't return any results."
    if isinstance(result, list):
        return ", ".join(result)
    if isinstance(result, dict):
        return json.dumps(result, indent=2)
    return str(result)

# -----------------------------
# Chatbot Core
# -----------------------------

def process_query(query: str) -> None:
    """
    Process a user query by interacting with the LLM and handling tool calls.
    """
    messages = [{'role': 'user', 'content': query}]

    response = client.messages.create(
        max_tokens=2024,
        model='claude-3-7-sonnet-20250219',
        tools=tools,
        messages=messages
    )

    process_query = True
    while process_query:
        assistant_content = []

        for content in response.content:
            if content.type == 'text':
                print(content.text)
                assistant_content.append(content)
                if len(response.content) == 1:
                    process_query = False

            elif content.type == 'tool_use':
                assistant_content.append(content)
                messages.append({'role': 'assistant', 'content': assistant_content})

                tool_id = content.id
                tool_args = content.input
                tool_name = content.name
                print(f"Calling tool {tool_name} with args {tool_args}")

                result = execute_tool(tool_name, tool_args)
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": result
                        }
                    ]
                })

                response = client.messages.create(
                    max_tokens=2024,
                    model='claude-3-7-sonnet-20250219',
                    tools=tools,
                    messages=messages
                )

                if len(response.content) == 1 and response.content[0].type == "text":
                    print(response.content[0].text)
                    process_query = False


def chat_loop() -> None:
    """
    Start an interactive command-line chat session with the chatbot.
    """
    print("Type your queries or 'quit' to exit.")
    while True:
        try:
            query = input("\nQuery: ").strip()
            if query.lower() == 'quit':
                break
            process_query(query)
            print("\n")
        except Exception as e:
            print(f"\nError: {str(e)}")

# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    chat_loop()
