from __future__ import annotations

import json
import os
import sys
import time
import zipfile
from typing import Any, TypedDict

from openai import OpenAI
from langgraph.graph import StateGraph, START, END

# ------------------------------------------------------------------ client
client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="sk-dummy-key",
)

MAX_ATTEMPTS        = 5
RETRY_DELAY_SECONDS = 1.0
MAX_TURNS           = 25
WORKDIR             = os.path.abspath(os.getcwd())


# ------------------------------------------------------------------ safety
def _safe_path(p: str) -> str:
    full = os.path.abspath(os.path.join(WORKDIR, p))
    if os.path.commonpath([WORKDIR, full]) != WORKDIR:
        raise ValueError(f"Path escapes workdir: {p!r}")
    return full


# ------------------------------------------------------------------ tools
def list_dir(path: str = ".") -> dict:
    full = _safe_path(path)
    if not os.path.isdir(full):
        return {"status": "error", "message": f"Not a directory: {path}"}
    entries = []
    for name in sorted(os.listdir(full)):
        child = os.path.join(full, name)
        entries.append({
            "name": name,
            "type": "dir" if os.path.isdir(child) else "file",
            "size": 0 if os.path.isdir(child) else os.path.getsize(child),
        })
    return {"status": "ok", "path": path, "entries": entries}


def read_file(path: str, max_bytes: int = 20_000) -> dict:
    full = _safe_path(path)
    if not os.path.isfile(full):
        return {"status": "error", "message": f"Not a file: {path}"}
    with open(full, "rb") as f:
        data = f.read(max_bytes)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return {"status": "error", "message": "Binary file; not readable as UTF-8"}
    return {
        "status": "ok",
        "path": path,
        "bytes_read": len(data),
        "truncated": os.path.getsize(full) > len(data),
        "content": text,
    }


def create_folder(path: str) -> dict:
    full = _safe_path(path)
    os.makedirs(full, exist_ok=True)
    return {"status": "ok", "path": path, "message": f"Folder ready: {path}"}


def write_file(path: str, content: str) -> dict:
    full = _safe_path(path)
    parent = os.path.dirname(full)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        f.write(content)
    return {
        "status": "ok",
        "path": path,
        "bytes_written": len(content.encode("utf-8")),
    }


def zip_folder(src_dir: str, zip_path: str) -> dict:
    src = _safe_path(src_dir)
    dst = _safe_path(zip_path)
    parent = os.path.dirname(dst)
    if parent:
        os.makedirs(parent, exist_ok=True)
    count = 0
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(src):
            for fname in files:
                full = os.path.join(root, fname)
                arc = os.path.relpath(full, src)
                zf.write(full, arc)
                count += 1
    return {"status": "ok", "zip": zip_path, "files_included": count}


def finish_task(summary: str = "") -> dict:
    return {"status": "ok", "message": "task complete", "summary": summary}


TOOL_EXECUTORS = {
    "list_dir":      list_dir,
    "read_file":     read_file,
    "create_folder": create_folder,
    "write_file":    write_file,
    "zip_folder":    zip_folder,
    "finish_task":   finish_task,
}


# ------------------------------------------------------------------ spec
TOOLS_SPEC = """
Available tools:

1. list_dir
   args: { "path": "<string>" }         # default "."
   Lists a directory. Use this FIRST to discover the folder layout.

2. read_file
   args: { "path": "<string>", "max_bytes"?: <int> }
   Reads a UTF-8 text file. Use before writing/deciding anything that depends
   on existing content.

3. create_folder
   args: { "path": "<string>" }
   Creates a folder (and any parent folders).

4. write_file
   args: { "path": "<string>", "content": "<string>" }
   Writes UTF-8 text to a file (creates parent folder if needed).

5. zip_folder
   args: { "src_dir": "<string>", "zip_path": "<string>" }
   Recursively zips everything under src_dir into a single .zip file.

6. finish_task
   args: { "summary"?: "<string>" }
   Call this WHEN AND ONLY WHEN the user's request has been fully satisfied.
   This is the ONLY way to end the task. Never emit an empty array [] as a
   "done" signal — always call finish_task instead.

RULES
- All paths are relative to the working directory and may not escape it.
- Prefer discovering state (list_dir / read_file) before mutating.
- Reply with ONLY a JSON array of { "tool": ..., "args": { ... } }.
- Do NOT reply with [] to finish. You MUST call finish_task to finish.
- finish_task must be the LAST call in the array; do not batch it with other
  mutating calls whose results you haven't seen yet.
"""


def build_single_user_message(state: "AgentState") -> str:
    """One user message containing everything the model needs this call."""
    parts: list[str] = []
    parts.append("=== SYSTEM / RULES ===")
    parts.append(TOOLS_SPEC.strip())

    parts.append("\n=== USER REQUEST ===")
    parts.append(state["user_request"])

    history = state.get("history", [])
    if history:
        parts.append(f"\n=== HISTORY ({len(history)} turn(s)) ===")
        for i, turn in enumerate(history, 1):
            parts.append(f"\n--- Turn {i}: calls ---")
            parts.append(json.dumps(turn.get("calls", []), indent=2))
            parts.append(f"--- Turn {i}: results ---")
            txt = json.dumps(turn.get("results", []), indent=2)
            if len(txt) > 4000:
                txt = txt
            parts.append(txt)

    parts.append(
        "\n=== INSTRUCTION ===\n"
        "Respond with ONLY a JSON array of tool calls for the NEXT step. "
        "When the task is complete, respond with a JSON array containing a "
        "single finish_task call. No prose, no markdown fences."
    )
    return "\n".join(parts)


# ------------------------------------------------------------------ state
class AgentState(TypedDict, total=False):
    user_request: str
    history: list[dict]           # [{"calls": [...], "results": [...]}, ...]
    attempts: int
    turns: int
    raw_response: str | None
    calls: list[dict] | None
    last_results: list[dict] | None
    error: str | None
    finished: bool


# ------------------------------------------------------------------ helpers
def _strip_fences(text: str) -> str:
    clean = text.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
        if clean.endswith("```"):
            clean = clean[:-3]
        clean = clean.strip()
    return clean


def _looks_degenerate(text: str | None) -> bool:
    if not text:
        return True
    s = text.strip()
    if len(s) <= 1:
        return True
    if not s.startswith("[") and not s.startswith("{"):
        return True
    return False


def _parse_calls(raw: str) -> list[dict]:
    clean = _strip_fences(raw)
    data = json.loads(clean)
    print(data)
    if isinstance(data, dict) and "calls" in data:
        data = data["calls"]
    if not isinstance(data, list):
        raise ValueError("Model did not return a JSON array of tool calls.")
    if data == []:
        raise ValueError(
            "Empty array is not a valid response. "
            "Call finish_task when done, otherwise emit tool calls."
        )
    for el in data:
        if not isinstance(el, dict) or "tool" not in el:
            raise ValueError(f"Malformed tool call element: {el!r}")
    return data


# ------------------------------------------------------------------ nodes
def plan_node(state: AgentState) -> AgentState:
    attempts = state.get("attempts", 0) + 1
    turns    = state.get("turns", 0) + 1
    print(f"\n[plan] turn {turns}, attempt {attempts}/{MAX_ATTEMPTS}")

    user_message = build_single_user_message(state)

    raw = None  # initialize before try
    try:
        response = client.chat.completions.create(
            model="deepseek-v3",
            messages=[{"role": "user", "content": user_message}],   # exactly one
        )
        print(response)
        raw = response.choices[0].message.content

    except Exception as e:

        print(f"[plan] API error: {e} \n Raw:{raw}")
        return {"attempts": attempts, "turns": turns,
                "error": f"api: {e}", "calls": None}
    # print(f"[plan] raw (truncated): {(raw or '')[:300]!r}")

    if _looks_degenerate(raw):
        return {"attempts": attempts, "turns": turns,
                "raw_response": raw, "error": "degenerate", "calls": None}

    try:
        calls = _parse_calls(raw)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"[plan] parse failed: {e}\n raw:{raw}")
        return {"attempts": attempts, "turns": turns,
                "raw_response": raw, "error": f"parse: {e}", "calls": None}

    # detect finish_task — it must be terminal
    if any(c.get("tool") == "finish_task" for c in calls):
        if calls[-1].get("tool") != "finish_task":
            return {"attempts": attempts, "turns": turns,
                    "raw_response": raw, "calls": None,
                    "error": "finish_task must be the last call"}
        print("[plan] finish_task called — finished.")
        return {"attempts": 0, "turns": turns, "calls": calls,
                "raw_response": raw, "error": None, "finished": True}

    print(f"[plan] parsed {len(calls)} call(s).")
    return {"attempts": 0, "turns": turns, "calls": calls,
            "raw_response": raw, "error": None, "finished": False}


def wait_node(state: AgentState) -> AgentState:
    print(f"[wait] sleeping {RETRY_DELAY_SECONDS}s...")
    time.sleep(RETRY_DELAY_SECONDS)
    return state


def execute_node(state: AgentState) -> AgentState:
    calls = state.get("calls") or []
    print(f"\n[execute] running {len(calls)} call(s)")
    results: list[dict] = []
    for i, call in enumerate(calls):
        name = call.get("tool")
        args = call.get("args", {}) or {}
        preview = json.dumps(args)
        print(f"  [{i+1}] {name}({preview})")
        fn = TOOL_EXECUTORS.get(name)
        if fn is None:
            results.append({"tool": name,
                            "result": {"status": "error",
                                       "message": "unknown tool"}})
            continue
        try:
            res = fn(**args)
        except Exception as e:
            res = {"status": "error", "message": str(e)}
        # Trim huge content so the single-message payload stays bounded
        results.append({"tool": name, "args": args, "result": res})
        print(f"      -> status={res.get('status')}")

    history = list(state.get("history", []))
    history.append({"calls": state.get("calls") or [], "results": results})
    return {"last_results": results, "history": history, "calls": None,
            "finished": state.get("finished", False)}


def done_node(state: AgentState) -> AgentState:
    print("\n[done]")
    return state


def give_up_node(state: AgentState) -> AgentState:
    print(f"\n[give-up] after {state.get('attempts')} attempt(s) on "
          f"turn {state.get('turns')}. last error: {state.get('error')}")
    return state


# ------------------------------------------------------------------ routing
def route_after_plan(state: AgentState) -> str:
    if state.get("finished") and state.get("calls"):
        return "execute"          # run finish_task, then execute routes to done
    if state.get("calls"):
        return "execute"
    if state.get("turns", 0) >= MAX_TURNS:
        return "give_up"
    if state.get("attempts", 0) >= MAX_ATTEMPTS:
        return "give_up"
    return "wait"


def route_after_execute(state: AgentState) -> str:
    if state.get("finished"):
        return "done"
    if state.get("turns", 0) >= MAX_TURNS:
        return "give_up"
    return "plan"


# ------------------------------------------------------------------ graph
def build_graph() -> Any:
    g = StateGraph(AgentState)

    g.add_node("plan", plan_node)
    g.add_node("wait", wait_node)
    g.add_node("execute", execute_node)
    g.add_node("done", done_node)
    g.add_node("give_up", give_up_node)

    g.add_edge(START, "plan")
    g.add_conditional_edges(
        "plan", route_after_plan,
        {"execute": "execute", "wait": "wait", "give_up": "give_up"},
    )
    g.add_edge("wait", "plan")
    g.add_conditional_edges(
        "execute", route_after_execute,
        {"plan": "plan", "done": "done", "give_up": "give_up"},
    )
    g.add_edge("done", END)
    g.add_edge("give_up", END)
    compiled=g.compile()
    merm= compiled.get_graph().draw_mermaid()

    with open("graph.mmd", "w") as f:
        f.write(merm)
    return compiled


# ------------------------------------------------------------------ main
# ------------------------------------------------------------------ main
def read_request() -> str:
    """
    Read the user request from stdin.

    Behaviour:
      - If stdin is piped/redirected (not a tty), read everything until EOF.
      - If interactive:
          * type the request on one or more lines
          * finish by entering a line containing only `END` (case-insensitive)
            OR by pressing Ctrl+Z then Enter on Windows / Ctrl+D on Unix.
    """
    # Piped / redirected input — just slurp it.
    if not sys.stdin.isatty():
        data = sys.stdin.read().strip()
        if not data:
            raise SystemExit("Empty request; aborting.")
        return data

    # Interactive input.
    print("Enter your request.")
    print("Finish with a line containing only 'END' "
          "(or Ctrl+Z+Enter on Windows / Ctrl+D on Unix).\n", flush=True)

    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            # Ctrl+Z+Enter (Win) or Ctrl+D (Unix) hit
            break
        if line.strip().upper() == "END":
            break
        lines.append(line)

    data = "\n".join(lines).strip()
    if not data:
        raise SystemExit("Empty request; aborting.")
    return data


if __name__ == "__main__":

    app = build_graph()

    user_request = read_request()
    print(f"\nWorkdir: {WORKDIR}")
    print(f"Request: {user_request[:200]}{'...' if len(user_request) > 200 else ''}\n")

    final_state = app.invoke({
        "user_request": user_request,
        "history": [],
        "attempts": 0,
        "turns": 0,
        "finished": False,
    })

    print("\n--- Summary ---")
    print(f"turns used: {final_state.get('turns')}")
    last = final_state.get("last_results") or []
    print(json.dumps(last, indent=2) if last else "(no tool results)")
    print(f"finished: {final_state.get('finished')}")
