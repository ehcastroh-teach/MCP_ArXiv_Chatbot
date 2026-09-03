<div align="center">
  <img src="images/thumbnails/mcp-arxiv-hero.svg" alt="Building MCP Applications: an arXiv chatbot" width="35%" />
</div>

# Building MCP Applications: An ArXiv Chatbot

This repository teaches you how to build applications where large language models can call external tools. Using the Model Context Protocol (MCP) pattern - schemas the model can read, a dispatcher that routes tool requests, and a client loop that threads results back into the conversation - you will implement a working chatbot that searches arXiv for research papers and retrieves their metadata. The material spans three lesson notebooks, three paired homework notebooks, four production-shaped examples, and a runnable `arxiv_chatbot.py`. Every design decision, from schema wording to loop termination to deployment shape, is covered with an explanation of why, not just what.

---

## Learning Objectives

By the end of this material, you will be able to:

- Define Python functions with the signatures, type annotations, and error semantics that make them reliable as LLM tools
- Write JSON schemas that describe tool inputs precisely enough for the model to call them correctly and consistently
- Build a dispatcher that maps schema name-plus-arguments requests to Python callables and normalizes return values into strings the model can read
- Route model responses that mix `text` and `tool_use` content blocks in a single response
- Manage conversation state across multi-turn tool loops, including the counterintuitive `role="user"` shape for tool results
- Trace a user query through the full sequence of `messages.create` API calls and understand when the loop terminates
- Extend a running MCP application with a new tool by following the three-step pattern: write function, write schema, register - no changes to the client loop
- Choose among four deployment shapes: CLI, web API, async, streaming, and self-reflection

---

## Data / File Dictionary

| File | Type | Description |
|------|------|-------------|
| `01_building_mcp_servers.ipynb` | Notebook | Lesson 1 - tool functions, JSON schemas, dispatcher, and isolated testing |
| `02_building_mcp_clients.ipynb` | Notebook | Lesson 2 - client initialization, message routing, conversation state, and the interactive loop |
| `03_mcp_in_practice.ipynb` | Notebook | Lesson 3 - end-to-end architecture, message-flow traces, tool design principles, extension pattern, and deployment patterns |
| `homework/building_mcp_servers_homework.ipynb` | Notebook | Practice writing tools, schemas, and a dispatcher on an offline dataset - no API key required |
| `homework/building_mcp_clients_homework.ipynb` | Notebook | Practice response classification, message construction, and loop termination against mocked responses - no API key required |
| `homework/mcp_in_practice_homework.ipynb` | Notebook | Capstone - design and build your own MCP application in a new domain, applying all three lessons |
| `arxiv_chatbot.py` | Script | Reference implementation combining server, client, and terminal user layer in one file |
| `examples/async_arxiv_chatbot.py` | Script | Concurrent tool execution using asyncio |
| `examples/streaming_arxiv_chatbot.py` | Script | Token-by-token response streaming |
| `examples/fastapi_arxiv_server.py` | Script | Per-session web API built around the same tools |
| `examples/multi_tool_reflection.py` | Script | Self-critique loop that asks the model to evaluate and refine its own answer |
| `images/` | Directory | Architecture, message-flow, response-routing, and tool-anatomy diagrams used in the notebooks |
| `papers/` | Directory | Persisted paper metadata written by `search_papers` at runtime, one subdirectory per topic |
| `requirements.txt` | Text | Python dependencies |

---

## Workflow Diagram

```
  Clone repo
      |
      v
  Install dependencies (requirements.txt)
      |
      v
  Open 01_building_mcp_servers.ipynb
      |
      +---> Define tool functions (search_papers, extract_info)
      |     Write JSON schemas
      |     Build dispatcher (execute_tool)
      |     Test tools in isolation
      |
  Open 02_building_mcp_clients.ipynb
      |
      +---> Initialize Anthropic client
      |     Route text and tool_use content blocks
      |     Manage messages list across turns
      |     Build process_query and chat_loop
      |
  Open 03_mcp_in_practice.ipynb
      |
      +---> Read full arxiv_chatbot.py end-to-end
      |     Trace API call sequences for real queries
      |     Apply tool design principles
      |     Extend system with a new tool (no loop changes)
      |     Choose a deployment pattern
      |
  Work through paired homework notebooks (no API key needed)
      |
      v
  Create .env with ANTHROPIC_API_KEY
      |
      v
  python arxiv_chatbot.py          (CLI chatbot)
  python examples/async_arxiv_chatbot.py
  python examples/streaming_arxiv_chatbot.py
  python examples/multi_tool_reflection.py
  uvicorn examples.fastapi_arxiv_server:app --reload
      |
      v
  Extend: add a new tool using the three-step pattern,
  or apply the full architecture to a different domain
```

---

## Step-by-Step Walkthrough

### Part 1 - Building the server: tools, schemas, and the dispatcher

An MCP application starts on the server side - the functions the model can call and the schema descriptions that let it discover them. Notebook 01 builds two tools: `search_papers` queries arXiv and writes paper metadata to disk organized by topic slug, and `extract_info` looks up stored metadata for a specific arXiv ID.

The design choices are deliberate and connected. Tools return strings rather than Python objects because the model reads text, not Python data structures. They return error strings instead of raising exceptions because a raised exception crashes the whole conversation loop before the model has a chance to explain the problem or try again. The `description` field in each schema is specific ("Search arXiv for academic papers on a topic") rather than vague ("A search tool") because the model uses that description to decide whether the tool is the right one for the current query - ambiguity leads to wrong tool calls. The `required` list in each schema includes only arguments that have no default, so the model is not asked to invent values for optional parameters.

The dispatcher, `execute_tool`, does three things: looks up the callable by name in a registry dict, invokes it with model-provided arguments, and normalizes the return value into a string. That normalization step - converting lists to comma-separated strings and dicts to JSON - is centralized here rather than scattered through the tools themselves, so every tool can focus on domain logic.

### Part 2 - Building the client: loop, state, and routing

The client owns the conversation. It sends each user query to the model together with the current tool schemas, walks the list of content blocks in the response, routes each block by type, and loops until the model returns a response with no `tool_use` blocks - that terminal response is the answer to show the user.

The `messages` list is the entire state of the conversation. The API is stateless: every call to `messages.create` must send the full history because the model has no memory between calls. Every exchange appends to the list - user queries, assistant responses (as a list of content block objects), and tool results. The counterintuitive detail: tool results carry `role="user"`, not `role="tool"`. The API models tool output as information flowing back into the model's reasoning from outside, the same structural category as a user typing a message. Getting this shape wrong silently produces bad results rather than a clear error.

Notebook 02 builds `process_query` step by step, starting with mock response objects that let the routing code run without any API call. This offline-first approach lets you verify routing logic without incurring API latency or spending tokens, and it produces unit-testable code.

### Part 3 - MCP in practice: architecture, message flow, and extension

Notebook 03 stitches the two previous layers into a coherent system. It reads through the complete `arxiv_chatbot.py` module section by section, then traces two specific queries at the API level - showing the exact sequence of `messages.create` calls, their arguments, and the response shapes.

The architectural insight is that the three layers (user, client, server) are logically independent even when they share a process. Swapping the terminal prompt for a FastAPI endpoint, or adding asyncio for concurrent tool execution, only changes the user layer. The tool functions, schemas, and dispatcher remain untouched. This is the payoff for the strict separation: new deployment shapes are small additions rather than rewrites.

The extension pattern is deliberately three steps: write the function, write its schema, add one line to the registry dict. Notebook 03 demonstrates this with a `save_papers_to_file` tool and confirms that `process_query` and `chat_loop` require no modification.

The deployment section covers four patterns: the CLI chatbot in `arxiv_chatbot.py`, a per-session web API in `examples/fastapi_arxiv_server.py`, concurrent and streaming variants in the async and streaming examples, and a self-critique loop in `examples/multi_tool_reflection.py`. Each pattern solves a specific problem - speed, perceived latency, multi-user state, or answer quality - and all share the same tool registry.

---

## How to Run

```bash
# 1. Clone this repository
git clone https://github.com/ehcastroh-teach/MCP_ArXiv_Chatbot.git
cd MCP_ArXiv_Chatbot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create a .env file with your Anthropic API key
echo 'ANTHROPIC_API_KEY="sk-..."' > .env

# 4. Open JupyterLab and work through the lessons in order
jupyter lab
#    Open 01_building_mcp_servers.ipynb, then 02, then 03.
#    The homework notebooks in homework/ run without an API key.

# 5. Run the reference chatbot
python arxiv_chatbot.py
#    Try: "Find papers about retrieval augmented generation"
#         "Tell me about the first one"
#         "quit"

# 6. Run the advanced examples (each adds one production capability)
python examples/async_arxiv_chatbot.py
python examples/streaming_arxiv_chatbot.py
python examples/multi_tool_reflection.py

# FastAPI web server (requires: pip install fastapi uvicorn)
uvicorn examples.fastapi_arxiv_server:app --reload
#    Then POST to http://127.0.0.1:8000/query with {"text": "...", "session_id": "..."}
```

---

## Key Concepts Glossary

| Term | Definition |
|------|------------|
| **MCP** | Model Context Protocol - the cross-provider pattern for exposing callable tools to an LLM, standardizing how schemas, dispatch, and results are structured |
| **tool** | A registered function the model can request to invoke, identified by a name, described by a schema, and executed by the dispatcher |
| **tool schema** | A JSON object describing one tool: its name, a natural-language description the model uses to decide when to call it, and an input_schema listing its arguments |
| **input_schema** | The JSON Schema fragment inside a tool schema that defines which arguments the model may pass, their types, and which are required |
| **tool_use block** | A content block in the model's response that requests a tool call; carries a `.name`, `.input` dict, and `.id` for correlation |
| **tool_result message** | The message shape used to send a dispatcher's return value back to the model; has `role="user"` because tool output is treated as external information flowing into the model |
| **dispatcher** | The function (`execute_tool`) that looks up a tool by name in a registry dict, invokes it with model-provided arguments, and normalizes the return value to a string |
| **content block** | One element of a model response - either `text` (free-form output for the user) or `tool_use` (a dispatch request for the client) |
| **process_query** | The client-side loop that drives one user query through one or more tool round-trips until the model returns a response containing only text blocks |
| **conversation state** | The `messages` list that accumulates every user turn, assistant response, and tool result; resent in full on every API call because the API is stateless |
| **idempotence** | Property of a tool that produces the same result on repeated calls with the same arguments; important when the model may retry on transient errors |
| **reflection loop** | A post-answer technique where the client asks the model to critique its own response and optionally regenerate a better one, trading latency for quality |
| **three-step extension pattern** | The protocol for adding a new tool without modifying the client loop: write the function, write its schema, add one entry to the registry dict |

---

## Further Reading

- Model Context Protocol specification (modelcontextprotocol.io)
- Anthropic Messages API - Tool use documentation
- JSON Schema specification (json-schema.org)
- FastAPI documentation
- arxiv Python client (lukasschwab/arxiv.py on GitHub)
- Anthropic tool_use and tool_result content block reference

---

## Credits and Acknowledgements

Tool-calling architecture and terminology follow the Model Context Protocol design as documented by Anthropic and community MCP contributors. The arXiv chatbot pattern is adapted from Anthropic's tool-use cookbook. Paper metadata is retrieved from arXiv.org via the `arxiv` Python client (Lukas Schwab).

---

## Contact

<div align="center">
  <img src="images/thumbnails/ehcastroh_teach_banner_flower.png" alt="ehcastroh" width="90" style="border-radius: 50%;" />

  <sub>ehcastroh</sub>

  <a href="https://github.com/ehcastroh">GitHub</a> · <a href="https://www.linkedin.com/in/ehcastroh/">LinkedIn</a>
</div>
