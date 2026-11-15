# Summary of `./app`

## Overview

The `app/` folder contains a self-contained, LLM-driven **agentic coding loop** built on LangGraph. It plans tool calls with a DeepSeek model (`deepseek-v3` via an OpenAI-compatible client), executes those tool calls against a sandboxed filesystem tool set, feeds the results back as history, and repeats until the model calls `finish_task`, the turn budget is exhausted, or the retry budget is blown.

It is a small, readable reference implementation of a JSON-tool-calling ReAct-style agent: no prose, no markdown fences — the model must reply with a JSON array of `{ "tool": ..., "args": {...} }` objects.

## Files

| File | Lines | Role |
|------|-------|------|
| `agent.py` | ~392 | Core graph: parsing, state, nodes, routing, prompt I/O, summary printing. |
| `helpers.py` | ~195 | Tool implementations + safety layer + the shared `OpenAI` client and constants. |
| `tryout1.py` | ~126 | Flat agent (no sub-tasks). Entry point #1. |
| `tryout2.py` | ~204 | Same agent **plus recursive `spawn_task` sub-agents** (bounded nesting). Entry point #2. |

There is also a generated `graph.mmd` in the repo root (Mermaid flowchart of the state graph), and the shared `DeepSeekAPI/` package + `server.py` live outside `app/`.

## Architecture (`agent.py`)

### Parsing layer
- `_strip_fences()` removes a single outer ```...``` wrapper.
- `_looks_degenerate()` rejects empty/1-char replies; anything starting with `[`/`{` or containing a fence is considered viable.
- `_normalize_calls()` validates the payload is a non-empty JSON array where every element is a dict with a `tool` key. An empty array `[]` is explicitly rejected (the model must call `finish_task` instead).
- `_try_decode()` = `json.loads` + normalize, never raises.
- `_extract_fenced_json()` fallback: scans all ```` ```json ```` / bare fences, requires **exactly one** valid tool-call payload; zero or multiple is an error (refuses to guess).
- `_parse_calls()` orchestrates: whole-reply fast path → fenced fallback.

### State (`AgentState`, TypedDict)
`user_request`, `parent_context`, `depth`, `history`, `attempts`, `turns`, `raw_response`, `calls`, `last_results`, `error`, `finished`, `summary`.

### Nodes
- **`plan_node`** (factory taking a `build_message` callable): builds the prompt from state, calls the LLM, parses tool calls. Detects the special rate-limit string `[消息发送过于频繁，请稍后重试]` and sleeps 10× the retry delay. Enforces that `finish_task` must be the **last** call; if so, marks `finished=True` and captures `summary`.
- **`wait_node`**: sleeps `RETRY_DELAY_SECONDS`, loops back to `plan`.
- **`execute_node`** (factory taking a `dispatch` callable): runs each call via `dispatch(name, args, depth)`, catches `KeyError` (unknown tool) and any exception into an error result, appends a `{calls, results}` turn to history.
- **`done_node`** / **`give_up_node`**: terminal printing nodes.

### Routing
- After `plan`: if calls → `execute`; else if turns/attempts exhausted → `give_up`; else → `wait` → back to `plan`.
- After `execute`: if `finished` → `done`; if turns exhausted → `give_up`; else → `plan`.

### Graph construction
`build_graph(build_message, dispatch, write_mermaid=False)` wires the `StateGraph`, compiles it, and optionally writes a Mermaid diagram to `graph.mmd`.

### Prompt I/O
- `read_request()` reads from stdin (piped or interactive). Interactive mode terminates on `END`, and typing `1` loads `prompt.txt` instead.
- `print_summary()` prints turn count, per-tool result statuses, and finished flag.

Constants (from `helpers.py`): `MAX_ATTEMPTS = 5`, `RETRY_DELAY_SECONDS = 1.0`, `MAX_TURNS = 25`; `agent.py` adds `MAX_DEPTH = 4`.

## Tools & safety (`helpers.py`)

The client is an `OpenAI` SDK instance pointed at a Lambda URL with a dummy key. Tool implementations:

- `list_dir(path=".")`, `read_file(path)` (UTF-8 only, returns full content + byte/line counts), `create_folder(path)`, `write_file(path, content)`, `edit_file(path, old, new, count=1)` (verbatim snippet replace; `count=0` = replace all), `append_file(path, content)`, `zip_folder(src_dir, zip_path)`, `comment(text)` (colored reasoning output), `finish_task(summary)`.

All tools are registered in `TOOL_EXECUTORS` for name-based dispatch.

**Safety:** `_safe_path()` resolves every path against `WORKDIR` and rejects anything escaping it via `os.path.commonpath`. Styling helpers `_status`, `_tool`, `_dim_json` add colored terminal output via `yachalk`.

## Entry points

### `tryout1.py` — flat agent
Defines `TOOLS_SPEC` (9 tools, no sub-tasks) and `RULES_SPEC`. `build_message()` assembles SYSTEM/RULES + USER REQUEST + HISTORY + INSTRUCTION. `dispatch()` is a direct `TOOL_EXECUTORS[name](**args)` lookup. Runs the graph at `depth=0` and prints the summary.

### `tryout2.py` — recursive agent
Same, but adds `SUBTASK_TOOL_SPEC` (a 10th tool, `spawn_task`) and `build_tools_spec(depth)` which hides `spawn_task` once `depth >= MAX_DEPTH`. `run_subtask()` builds a **fresh graph invocation** with an isolated history (only `task` + `context`), increments depth, and returns `{status, summary, turns}` — if the child never finishes it returns an error instead of a summary. `dispatch()` routes `spawn_task` to `run_subtask` and everything else to `TOOL_EXECUTORS`. Writes `graph.mmd` on startup.

## Control flow

```
START → plan → (calls?) → execute → (finished? → done : plan)
              ↓ no calls
            wait → plan
              ↓ retries/turns exhausted
            give_up
```

## Dependencies

The app relies on `langgraph` (graph runtime), `openai` (client), `yachalk` (ANSI styling), `python-dotenv` (env loading in `tryout2`/`server`), plus the project-level `requirements.txt` (Flask, gunicorn, wasmtime, streamdown helpers, etc.). Deployment via `dockerfile` (python:3.11-slim + venv + gunicorn + AWS Lambda adapter) targets `server.py`, not the agent scripts.

## Design notes / limitations

- Single-model planning call per turn; no retry with temperature variation.
- `[]` is treated as a protocol error rather than completion (forces explicit `finish_task`).
- Fenced-JSON fallback exists to tolerate models that wrap JSON in markdown despite instructions.
- Sub-agent nesting (tryout2) is what saves context tokens: children get tiny, isolated contexts and return only short summaries.
- These scripts are experiments (`tryout*`) sharing `agent.py`/`helpers.py`; they are not wired into the Flask server.
