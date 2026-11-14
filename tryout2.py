from __future__ import annotations

import json
import os
import sys
import time
import zipfile
from typing import Any, TypedDict

from yachalk import chalk
from openai import OpenAI
from langgraph.graph import StateGraph, START, END

# ------------------------------------------------------------------ client
client = OpenAI(
    base_url="https://xqxmedozyom24kmy522qsbacqu0qtiaq.lambda-url.ap-south-1.on.aws/v1",
    api_key="sk-dummy-key",
)

MAX_ATTEMPTS        = 5
RETRY_DELAY_SECONDS = 1.0
MAX_TURNS           = 25
MAX_DEPTH           = 4          # how deep nested tasks may go
WORKDIR             = os.path.abspath(os.getcwd())


# ------------------------------------------------------------------ style helpers
def _status(res: dict) -> str:
    st = res.get("status", "?")
    return {
        "ok":    chalk.green("ok"),
        "error": chalk.red("error"),
    }.get(st, chalk.yellow(st))


def _tool(name: str | None) -> str:
    return chalk.cyan.bold(name or "?")


def _dim_json(obj: Any, limit: int = 160) -> str:
    txt = json.dumps(obj, default=str)
    if len(txt) > limit:
        txt = txt
    return chalk.gray(txt)


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


def read_file(path: str, max_bytes: int = 20_000, offset: int = 0, limit: int | None = None) -> dict:
    full = _safe_path(path)
    if not os.path.isfile(full):
        return {"status": "error", "message": f"Not a file: {path}"}
    with open(full, "rb") as f:
        data = f.read()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return {"status": "error", "message": "Binary file; not readable as UTF-8"}

    total_bytes = len(data)
    total_lines = text.count("\n") + (0 if text.endswith("\n") or text == "" else 1)

    lines = text.splitlines(keepends=True)
    start = max(0, offset)
    end = len(lines) if limit is None else max(start, start + limit)
    window = "".join(lines[start:end])

    enc = window.encode("utf-8")
    byte_truncated = False
    if len(enc) > max_bytes:
        window = enc.decode("utf-8", errors="ignore")
        byte_truncated = True

    return {
        "status": "ok",
        "path": path,
        "bytes_read": len(window.encode("utf-8")),
        "total_bytes": total_bytes,
        "total_lines": total_lines,
        "offset": start,
        "truncated": byte_truncated or end < len(lines),
        "content": window,
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


def edit_file(path: str, old: str, new: str, count: int = 1) -> dict:
    full = _safe_path(path)
    if not os.path.isfile(full):
        return {"status": "error", "message": f"Not a file: {path}"}
    with open(full, "r", encoding="utf-8", newline="") as f:
        original = f.read()

    if old == "":
        return {"status": "error", "message": "`old` must be non-empty"}

    occurrences = original.count(old)
    if occurrences == 0:
        return {
            "status": "error",
            "message": "`old` snippet not found verbatim in file",
            "path": path,
        }

    limit = occurrences if count == 0 else count
    updated = original.replace(old, new, limit)
    replaced = min(limit, occurrences)

    with open(full, "w", encoding="utf-8", newline="") as f:
        f.write(updated)

    return {
        "status": "ok",
        "path": path,
        "replaced": replaced,
        "occurrences": occurrences,
        "bytes_written": len(updated.encode("utf-8")),
    }


def append_file(path: str, content: str) -> dict:
    full = _safe_path(path)
    parent = os.path.dirname(full)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(full, "a", encoding="utf-8") as f:
        f.write(content)
    return {"status": "ok", "path": path, "bytes_appended": len(content.encode("utf-8"))}


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
    "edit_file":     edit_file,
    "append_file":   append_file,
    "zip_folder":    zip_folder,
    "finish_task":   finish_task,
}


# ------------------------------------------------------------------ spec
BASE_TOOLS_SPEC = """
Available tools:

1. list_dir
   args: { "path": "<string>" }         # default "."
   Lists a directory. Use this FIRST to discover the folder layout.

2. read_file
   args: { "path": "<string>", "offset"?: <int>, "limit"?: <int>, "max_bytes"?: <int> }
   Reads a UTF-8 text file. `offset`/`limit` select a line window (0-based);
   `max_bytes` caps the returned content. Prefer this over reading the whole
   file when you only need a small region. Reads before writing/deciding.

3. create_folder
   args: { "path": "<string>" }
   Creates a folder (and any parent folders).

4. edit_file   <-- PREFERRED for small changes (token-saving)
   args: { "path": "<string>", "old": "<string>", "new": "<string>", "count"?: <int> }
   Replaces a snippet. `old` must match verbatim (whitespace included).
   `count` = max replacements (default 1; 0 = replace all). Only the matched
   region is rewritten — you do NOT resend the whole file. Use this to make
   targeted edits instead of rewriting entire files with write_file.

5. write_file
   args: { "path": "<string>", "content": "<string>" }
   Writes the FULL contents of a UTF-8 file (creates parent folder if needed).
   Use only for new files or full rewrites; prefer edit_file for small changes.

6. append_file
   args: { "path": "<string>", "content": "<string>" }
   Appends UTF-8 text to the end of a file (creates it if needed).

7. zip_folder
   args: { "src_dir": "<string>", "zip_path": "<string>" }
   Recursively zips everything under src_dir into a single .zip file.

8. finish_task
   args: { "summary"?: "<string>" }
   Call this WHEN AND ONLY WHEN the current (sub)task is fully satisfied.
   Inside a subtask this finishes ONLY that subtask and returns its summary
   to the parent — it does NOT end the whole agent.
   This is the ONLY way to end a task. Never emit an empty array [] as a
   "done" signal — always call finish_task instead.
"""

SUBTASK_TOOL_SPEC = """
9. spawn_task   <-- NEST A SUBTASK (isolated context, saves input tokens)
   args: { "task": "<string>", "context"?: "<string>" }
   Runs a self-contained sub-agent to accomplish `task`. The sub-agent gets
   ONLY the `task` text plus an optional `context` string you supply — it
   does NOT see this conversation's history. That isolation is what saves
   input tokens: do the smaller job with a fresh, tiny context and get back
   a short summary string. It shares the same working directory and tools.
   Returns: { "status": "ok", "summary": "..." } (or an error).
   Use it whenever a chunk of work is independently describable. Do NOT use
   it for trivial single tool calls. Nesting is allowed up to MAX_DEPTH.
"""


def build_tools_spec(depth: int) -> str:
    spec = BASE_TOOLS_SPEC
    if depth < MAX_DEPTH:
        spec += SUBTASK_TOOL_SPEC
    else:
        spec += (
            "\n(Note: you are at the maximum nesting depth; spawn_task is "
            "unavailable at this level.)\n"
        )
    return spec.strip()


RULES_SPEC = """
RULES
- All paths are relative to the working directory and may not escape it.
- Prefer discovering state (list_dir / read_file) before mutating.
- To save output tokens, make targeted edits with edit_file instead of
  rewriting whole files. Keep `old`/`new` snippets as small as possible while
  still being unique in the file.
- Prefer spawn_task for any self-contained sub-job: give it a short task
  description and only the minimal context it needs. Its history stays
  isolated, so the parent context stays small.
- Reply with ONLY a JSON array of { "tool": ..., "args": { ... } }.
- Do NOT reply with [] to finish. You MUST call finish_task to finish.
- finish_task must be the LAST call in the array; do not batch it with other
  mutating calls whose results you haven't seen yet.
"""


def build_single_user_message(state: "AgentState") -> str:
    """One user message containing everything the model needs this call."""
    depth = state.get("depth", 0)
    parts: list[str] = []
    parts.append("=== SYSTEM / RULES ===")
    parts.append(build_tools_spec(depth))
    parts.append(RULES_SPEC.strip())

    parts.append("\n=== TASK ===")
    parts.append(state["user_request"])

    parent_ctx = state.get("parent_context") or ""
    if parent_ctx:
        parts.append("\n=== CONTEXT (from parent task) ===")
        parts.append(parent_ctx)

    if depth:
        parts.append(f"\n=== NESTING ===\nYou are a subtask at depth {depth}/{MAX_DEPTH}.")

    history = state.get("history", [])
    if history:
        parts.append(f"\n=== HISTORY ({len(history)} turn(s)) ===")
        for i, turn in enumerate(history, 1):
            parts.append(f"\n--- Turn {i}: calls ---")
            parts.append(json.dumps(turn.get("calls", []), indent=2))
            parts.append(f"--- Turn {i}: results ---")
            txt = json.dumps(turn.get("results", []), indent=2)
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
    parent_context: str
    depth: int
    history: list[dict]           # [{"calls": [...], "results": [...]}, ...]
    attempts: int
    turns: int
    raw_response: str | None
    calls: list[dict] | None
    last_results: list[dict] | None
    error: str | None
    finished: bool
    summary: str


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


# ------------------------------------------------------------------ subtask runner
def run_subtask(task: str, context: str = "", parent_depth: int = 0) -> dict:
    """Run a nested agent with an isolated context and return its summary."""
    child_depth = parent_depth + 1
    if child_depth > MAX_DEPTH:
        return {"status": "error", "message": f"max nesting depth {MAX_DEPTH} exceeded"}
    if not task.strip():
        return {"status": "error", "message": "spawn_task requires a non-empty `task`"}

    print(
        "\n"
        + chalk.bg_yellow.black.bold(" subtask ")
        + chalk.gray(f" depth {child_depth}/{MAX_DEPTH}")
    )
    preview = task
    print(chalk.yellow("task: ") + chalk.white(preview))

    app = build_graph()
    final = app.invoke({
        "user_request": task,
        "parent_context": context,
        "depth": child_depth,
        "history": [],
        "attempts": 0,
        "turns": 0,
        "finished": False,
    })

    ok = bool(final.get("finished"))
    summary = final.get("summary") or ""
    if not ok and not summary:
        summary = f"subtask ended without finish (error={final.get('error')})"

    print(
        chalk.bg_yellow.black.bold(" subtask done ")
        + " "
        + (chalk.green("ok") if ok else chalk.red("unfinished"))
    )
    return {
        "status": "ok" if ok else "error",
        "summary": summary,
        "turns": final.get("turns", 0),
    }


# ------------------------------------------------------------------ nodes
def plan_node(state: AgentState) -> AgentState:
    attempts = state.get("attempts", 0) + 1
    turns    = state.get("turns", 0) + 1
    depth    = state.get("depth", 0)

    header = (
        chalk.bg_blue.white.bold(" plan ")
        + " "
        + chalk.blue(f"turn {turns}/{MAX_TURNS}")
        + chalk.gray(" · ")
        + chalk.yellow(f"attempt {attempts}/{MAX_ATTEMPTS}")
        + chalk.gray(" · ")
        + chalk.magenta(f"depth {depth}/{MAX_DEPTH}")
    )
    print("\n" + header)

    user_message = build_single_user_message(state)

    raw = None
    try:
        response = client.chat.completions.create(
            model="deepseek-v3",
            messages=[{"role": "user", "content": user_message}],
        )
        raw = response.choices[0].message.content

    except Exception as e:
        print(chalk.red(f"[plan] API error: {e}") + chalk.gray(f" raw={raw!r}"))
        return {"attempts": attempts, "turns": turns,
                "error": f"api: {e}", "calls": None}

    if _looks_degenerate(raw):
        print(chalk.yellow("[plan] degenerate response"))
        return {"attempts": attempts, "turns": turns,
                "raw_response": raw, "error": "degenerate", "calls": None}

    try:
        calls = _parse_calls(raw)
    except (json.JSONDecodeError, ValueError) as e:
        print(chalk.red(f"[plan] parse failed: {e}"))

        print(chalk.gray(f"       raw: {(raw or '')[:200]!r}"))
        if raw==f"[消息发送过于频繁，请稍后重试":
            time.sleep(RETRY_DELAY_SECONDS*10)

        return {"attempts": attempts, "turns": turns,
                "raw_response": raw, "error": f"parse: {e}", "calls": None}

    if any(c.get("tool") == "finish_task" for c in calls):
        if calls[-1].get("tool") != "finish_task":
            print(chalk.red("[plan] finish_task must be the last call"))
            return {"attempts": attempts, "turns": turns,
                    "raw_response": raw, "calls": None,
                    "error": "finish_task must be the last call"}
        summary = (calls[-1].get("args") or {}).get("summary", "")
        print(chalk.green("✔ finish_task called — finished."))
        return {"attempts": 0, "turns": turns, "calls": calls,
                "raw_response": raw, "error": None, "finished": True,
                "summary": summary}

    print(chalk.gray(f"[plan] parsed {len(calls)} call(s)."))
    return {"attempts": 0, "turns": turns, "calls": calls,
            "raw_response": raw, "error": None, "finished": False}


def wait_node(state: AgentState) -> AgentState:
    print(chalk.gray(f"⏳ waiting {RETRY_DELAY_SECONDS}s…"))
    time.sleep(RETRY_DELAY_SECONDS)
    return state


def execute_node(state: AgentState) -> AgentState:
    calls = state.get("calls") or []
    depth = state.get("depth", 0)
    print("\n" + chalk.bg_magenta.white.bold(f" execute · {len(calls)} call(s) "))
    results: list[dict] = []
    for i, call in enumerate(calls):
        name = call.get("tool")
        args = call.get("args", {}) or {}
        print(f"  {chalk.gray(f'[{i+1}/{len(calls)}]')} {_tool(name)} {_dim_json(args)}")

        if name == "spawn_task":
            task    = str(args.get("task", ""))
            context = str(args.get("context", "") or "")
            res = run_subtask(task, context, parent_depth=depth)
            results.append({"tool": name, "args": args, "result": res})
            mark = "✔" if res.get("status") == "ok" else "✖"
            color = chalk.green if res.get("status") == "ok" else chalk.red
            print(f"      {color(mark)} {_status(res)}")
            continue

        fn = TOOL_EXECUTORS.get(name)
        if fn is None:
            results.append({"tool": name,
                            "result": {"status": "error",
                                       "message": "unknown tool"}})
            print(f"      {chalk.red('✖')} unknown tool")
            continue
        try:
            res = fn(**args)
        except Exception as e:
            res = {"status": "error", "message": str(e)}

        results.append({"tool": name, "args": args, "result": res})
        mark = "✔" if res.get("status") == "ok" else "✖"
        color = chalk.green if res.get("status") == "ok" else chalk.red
        print(f"      {color(mark)} {_status(res)}")

    history = list(state.get("history", []))
    history.append({"calls": state.get("calls") or [], "results": results})
    return {"last_results": results, "history": history, "calls": None,
            "finished": state.get("finished", False)}


def done_node(state: AgentState) -> AgentState:
    print("\n" + chalk.green.bold("✔ done"))
    return state


def give_up_node(state: AgentState) -> AgentState:
    print(
        "\n"
        + chalk.bg_red.white.bold(" give-up ")
        + " "
        + chalk.red(
            f"after {state.get('attempts')} attempt(s) on turn {state.get('turns')}"
        )
        + chalk.gray(f" · last error: {state.get('error')}")
    )
    return state


# ------------------------------------------------------------------ routing
def route_after_plan(state: AgentState) -> str:
    if state.get("finished") and state.get("calls"):
        return "execute"
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
def read_request() -> str:
    if not sys.stdin.isatty():
        data = sys.stdin.read().strip()
        if not data:
            raise SystemExit("Empty request; aborting.")
        return data

    print(chalk.blue("Enter your request."))
    print(chalk.gray(
        "Finish with a line containing only 'END' "
        "(or Ctrl+Z+Enter on Windows / Ctrl+D on Unix)."
    ) + "\n")

    lines: list[str] = []
    while True:
        try:
            line = input(chalk.cyan("> ") if not lines else chalk.gray("… "))
        except EOFError:
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

    print("\n" + chalk.bg_black.white.bold(" agent ") +
          chalk.gray(f"  workdir = {WORKDIR}"))
    preview = user_request
    print(chalk.blue("request: ") + chalk.white(preview))
    print()

    final_state = app.invoke({
        "user_request": user_request,
        "parent_context": "",
        "depth": 0,
        "history": [],
        "attempts": 0,
        "turns": 0,
        "finished": False,
    })

    print("\n" + chalk.bg_black.white.bold(" summary "))
    print(chalk.blue("turns used: ") + chalk.white(str(final_state.get("turns"))))

    last = final_state.get("last_results") or []
    if last:
        for r in last:
            tool = r.get("tool", "?")
            res  = r.get("result", {})
            print(
                f"  {chalk.cyan(tool):<14} "
                f"{_status(res)} "
                + chalk.gray(str(res.get("message", res.get("summary", ""))))
            )
    else:
        print(chalk.gray("(no tool results)"))

    finished = final_state.get("finished")
    print(
        chalk.blue("finished:   ")
        + (chalk.green("yes") if finished else chalk.red("no"))
    )
