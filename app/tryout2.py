from __future__ import annotations

import json

from dotenv import load_dotenv
from yachalk import chalk

from helpers import TOOL_EXECUTORS, WORKDIR
from agent import (
    build_graph, read_request, print_summary,
    AgentState, MAX_DEPTH,
)

load_dotenv()


# ------------------------------------------------------------------ spec
BASE_TOOLS_SPEC = """
Available tools:

1. list_dir
   args: { "path": "<string>" }         # default "."
   Lists a directory. Use this FIRST to discover the folder layout.

2. read_file
   args: { "path": "<string>" }
   Reads a UTF-8 text file and always returns the ENTIRE file content.
   Reads before writing/deciding.

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

8. comment
   args: { "text": "<string>" }
   Prints a free-form comment authored by you to the terminal in color.
   Use it to explain your reasoning or findings to the user.

9. finish_task
   args: { "summary"?: "<string>" }
   Call this WHEN AND ONLY WHEN the current (sub)task is fully satisfied.
   Inside a subtask this finishes ONLY that subtask and returns its summary
   to the parent — it does NOT end the whole agent.
   This is the ONLY way to end a task. Never emit an empty array [] as a
   "done" signal — always call finish_task instead.
"""

SUBTASK_TOOL_SPEC = """
10. spawn_task   <-- NEST A SUBTASK (isolated context, saves input tokens)
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
-Do not type any unneccesay non-computation friendly text. If neccessay just use the comment tool . 
- Do NOT reply with [] to finish. You MUST call finish_task to finish.
- finish_task must be the LAST call in the array; do not batch it with other
  mutating calls whose results you haven't seen yet.
"""


def build_tools_spec(depth: int) -> str:
    if depth < MAX_DEPTH:
        return (BASE_TOOLS_SPEC + SUBTASK_TOOL_SPEC).strip()
    return (
        BASE_TOOLS_SPEC
        + f"\n(Note: you are at the maximum nesting depth {MAX_DEPTH}; "
        "spawn_task is unavailable at this level.)\n"
    ).strip()


# ------------------------------------------------------------------ wiring
def build_message(state: AgentState) -> str:
    depth = state.get("depth", 0)
    parts = [
        "=== SYSTEM / RULES ===",
        build_tools_spec(depth),
        RULES_SPEC.strip(),
        "\n=== TASK ===",
        state["user_request"],
    ]
    if state.get("parent_context"):
        parts += ["\n=== CONTEXT (from parent task) ===", state["parent_context"]]
    if depth:
        parts.append(f"\n=== NESTING ===\nYou are a subtask at depth {depth}/{MAX_DEPTH}.")

    history = state.get("history", [])
    if history:
        parts.append(f"\n=== HISTORY ({len(history)} turn(s)) ===")
        for i, t in enumerate(history, 1):
            parts.append(f"\n--- Turn {i}: calls ---")
            parts.append(json.dumps(t.get("calls", []), indent=2))
            parts.append(f"--- Turn {i}: results ---")
            parts.append(json.dumps(t.get("results", []), indent=2))

    parts.append(
        "\n=== INSTRUCTION ===\n"
        "Respond with ONLY a JSON array of tool calls for the NEXT step. "
        "When the task is complete, respond with a JSON array containing a "
        "single finish_task call. No prose, no markdown fences."
    )
    return "\n".join(parts)


def run_subtask(task: str, context: str = "", parent_depth: int = 0) -> dict:
    child_depth = parent_depth + 1
    if child_depth > MAX_DEPTH:
        return {"status": "error", "message": f"max nesting depth {MAX_DEPTH} exceeded"}
    if not task.strip():
        return {"status": "error", "message": "spawn_task requires a non-empty `task`"}

    print("\n" + chalk.bg_yellow.black.bold(" subtask ")
          + chalk.gray(f" depth {child_depth}/{MAX_DEPTH}"))
    print(chalk.yellow("task: ") + chalk.white(task))

    final = build_graph(build_message, dispatch).invoke({
        "user_request": task,
        "parent_context": context,
        "depth": child_depth,
        "history": [],
        "attempts": 0,
        "turns": 0,
        "finished": False,
    })

    ok = bool(final.get("finished"))
    summary = final.get("summary") or (
        f"subtask ended without finish (error={final.get('error')})" if not ok else ""
    )
    print(chalk.bg_yellow.black.bold(" subtask done ") + " "
          + (chalk.green("ok") if ok else chalk.red("unfinished")))

    return {
        "status": "ok" if ok else "error",
        "summary": summary,
        "turns": final.get("turns", 0),
    }


def dispatch(name: str, args: dict, depth: int) -> dict:
    if name == "spawn_task":
        return run_subtask(
            str(args.get("task", "")),
            str(args.get("context", "") or ""),
            parent_depth=depth,
        )
    return TOOL_EXECUTORS[name](**args)


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    app = build_graph(build_message, dispatch, write_mermaid=True)
    req = read_request()

    print("\n" + chalk.bg_black.white.bold(" agent ") + chalk.gray(f"  workdir = {WORKDIR}"))
    print(chalk.blue("request: ") + chalk.white(req) + "\n")

    final = app.invoke({
        "user_request": req,
        "parent_context": "",
        "depth": 0,
        "history": [],
        "attempts": 0,
        "turns": 0,
        "finished": False,
    })
    print_summary(final)