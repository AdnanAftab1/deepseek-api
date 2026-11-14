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
WORKDIR             = os.path.abspath(os.getcwd())

# Default per-read byte cap; can be overridden per call via max_bytes
DEFAULT_READ_BYTES  = 20_000


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
        txt = txt[:limit] + "…"
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


def read_file(
    path: str,
    max_bytes: int = DEFAULT_READ_BYTES,
    start: int = 0,
    start_line: int | None = None,
    end_line: int | None = None,
    offset: int = 0,
) -> dict:
    """Read a selected portion of a UTF-8 text file to save token output.

    Args:
        path:        relative file path.
        max_bytes:   hard cap on the number of bytes read for this slice.
        start:       byte offset to begin reading from (default 0).
        start_line:  1-based first line to include (inclusive). If given,
                     overrides ``start`` (byte offset).
        end_line:    1-based last line to include (inclusive). If given
                     together with ``start_line``, restricts the slice to
                     that line range.
        offset:      number of lines to skip before starting (alternative
                     0-based line offset when start_line isn't used).

    Returns a dict with:
        status, path, bytes_read, start_byte, end_byte, total_bytes,
        truncated (True if bytes remain after this slice),
        start_line, end_line (actual 1-based line numbers of the slice),
        total_lines (if computed), content, next_start (byte offset to
        continue reading, or None if EOF).
    """
    full = _safe_path(path)
    if not os.path.isfile(full):
        return {"status": "error", "message": f"Not a file: {path}"}

    total_bytes = os.path.getsize(full)

    # --- line-based selection -------------------------------------------
    if start_line is not None or end_line is not None or offset:
        try:
            with open(full, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            return {"status": "error",
                    "message": "Binary file; not readable as UTF-8"}
        total_lines = len(lines)

        first = (start_line - 1) if start_line is not None else offset
        if first < 0:
            first = 0
        last = end_line if end_line is not None else total_lines
        if last > total_lines:
            last = total_lines
        if last < first:
            last = first

        selected = lines[first:last]
        text = "".join(selected)
        raw = text.encode("utf-8")

        # Enforce byte cap
        truncated = False
        if len(raw) > max_bytes:
            raw = raw[:max_bytes]
            # avoid cutting a multibyte char
            text = raw.decode("utf-8", errors="ignore")
            truncated = True

        bytes_read = len(text.encode("utf-8"))
        return {
            "status": "ok",
            "path": path,
            "bytes_read": bytes_read,
            "total_bytes": total_bytes,
            "start_byte": 0,
            "end_byte": bytes_read,
            "start_line": first + 1 if total_lines else 0,
            "end_line": first + text.count("\n") + (0 if text.endswith("\n") else 1),
            "total_lines": total_lines,
            "truncated": truncated or (last < total_lines),
            "next_start": None if (last >= total_lines and not truncated) else None,
            "content": text,
        }

    # --- byte-based selection -------------------------------------------
    if start < 0:
        start = 0
    if start > total_bytes:
        start = total_bytes

    with open(full, "rb") as f:
        f.seek(start)
        data = f.read(max_bytes)

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        # trim trailing partial multibyte char
        text = data.decode("utf-8", errors="ignore")

    bytes_read = len(data)
    end_byte = start + bytes_read
    more = end_byte < total_bytes

    return {
        "status": "ok",
        "path": path,
        "bytes_read": bytes_read,
        "total_bytes": total_bytes,
        "start_byte": start,
        "end_byte": end_byte,
        "total_lines": None,
        "start_line": None,
        "end_line": None,
        "truncated": more,
        "next_start": end_byte if more else None,
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
   args: {
     "path":       "<string>",
     "max_bytes"?: <int>,     # cap on bytes returned for this slice (default 20000)
     "start"?:     <int>,     # byte offset to start from (default 0)
     "start_line"?: <int>,    # 1-based first line (inclusive); overrides start
     "end_line"?:   <int>,    # 1-based last line (inclusive); used with start_line
     "offset"?:    <int>      # 0-based number of lines to skip (alt. to start_line)
   }
   Reads a SELECTED PORTION of a UTF-8 text file to save token output.
   Prefer using start_line/end_line (or offset) when you only need part of
   a file. Response includes total_bytes, truncated, next_start and
   (for line-based reads) total_lines so you can page through large files.

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
- To save tokens, read only the portion of a file you actually need
  (use start_line/end_line or max_bytes), rather than whole large files.
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

    header = (
        chalk.bg_blue.white.bold(" plan ")
        + " "
        + chalk.blue(f"turn {turns}/{MAX_TURNS}")
        + chalk.gray(" · ")
        + chalk.yellow(f"attempt {attempts}/{MAX_ATTEMPTS}")
    )
    print("\n" + header)

    user_message = build_single_user_message(state)

    raw = None  # initialize before try
    try:
        response = client.chat.completions.create(
            model="deepseek-v3",
            messages=[{"role": "user", "content": user_message}],   # exactly one
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
        return {"attempts": attempts, "turns": turns,
                "raw_response": raw, "error": f"parse: {e}", "calls": None}

    # detect finish_task — it must be terminal
    if any(c.get("tool") == "finish_task" for c in calls):
        if calls[-1].get("tool") != "finish_task":
            print(chalk.red("[plan] finish_task must be the last call"))
            return {"attempts": attempts, "turns": turns,
                    "raw_response": raw, "calls": None,
                    "error": "finish_task must be the last call"}
        print(chalk.green("✔ finish_task called — finished."))
        return {"attempts": 0, "turns": turns, "calls": calls,
                "raw_response": raw, "error": None, "finished": True}

    print(chalk.gray(f"[plan] parsed {len(calls)} call(s)."))
    return {"attempts": 0, "turns": turns, "calls": calls,
            "raw_response": raw, "error": None, "finished": False}


def wait_node(state: AgentState) -> AgentState:
    print(chalk.gray(f"⏳ waiting {RETRY_DELAY_SECONDS}s…"))
    time.sleep(RETRY_DELAY_SECONDS)
    return state


def execute_node(state: AgentState) -> AgentState:
    calls = state.get("calls") or []
    print("\n" + chalk.bg_magenta.white.bold(f" execute · {len(calls)} call(s) "))
    results: list[dict] = []
    for i, call in enumerate(calls):
        name = call.get("tool")
        args = call.get("args", {}) or {}
        print(f"  {chalk.gray(f'[{i+1}/{len(calls)}]')} {_tool(name)} {_dim_json(args)}")

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

        # Trim huge content so the single-message payload stays bounded
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
    compiled = g.compile()
    # merm = compiled.get_graph().draw_mermaid()
    # with open("graph.mmd", "w") as f:
    #     f.write(merm)
    return compiled


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

    print("\n" + chalk.bg_black.white.bold(" agent ") +
          chalk.gray(f"  workdir = {WORKDIR}"))
    preview = user_request[:200] + ("…" if len(user_request) > 200 else "")
    print(chalk.blue("request: ") + chalk.white(preview))
    print()

    final_state = app.invoke({
        "user_request": user_request,
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
                + chalk.gray(res.get("message", "")[:120])
            )
    else:
        print(chalk.gray("(no tool results)"))

    finished = final_state.get("finished")
    print(
        chalk.blue("finished:   ")
        + (chalk.green("yes") if finished else chalk.red("no"))
    )
