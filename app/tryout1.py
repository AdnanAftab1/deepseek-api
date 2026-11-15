from __future__ import annotations

import json

from yachalk import chalk

from helpers import TOOL_EXECUTORS, WORKDIR
from agent import (
    build_graph, read_request, print_summary,
    AgentState,
)


# ------------------------------------------------------------------ spec
TOOLS_SPEC = """
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
   Call this WHEN AND ONLY WHEN the user's request has been fully satisfied.
   This is the ONLY way to end the task. Never emit an empty array [] as a
   "done" signal — always call finish_task instead.
"""

RULES_SPEC = """
RULES
- All paths are relative to the working directory and may not escape it.
- Prefer discovering state (list_dir / read_file) before mutating.
- To save output tokens, make targeted edits with edit_file instead of
  rewriting whole files. Keep `old`/`new` snippets as small as possible while
  still being unique in the file.
- Reply with ONLY a JSON array of { "tool": ..., "args": { ... } }.
- Do NOT reply with [] to finish. You MUST call finish_task to finish.
- finish_task must be the LAST call in the array; do not batch it with other
  mutating calls whose results you haven't seen yet.
"""


# ------------------------------------------------------------------ wiring
def build_message(state: AgentState) -> str:
    parts = [
        "=== SYSTEM / RULES ===",
        TOOLS_SPEC.strip(),
        RULES_SPEC.strip(),
        "\n=== USER REQUEST ===",
        state["user_request"],
    ]
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


def dispatch(name: str, args: dict, depth: int) -> dict:
    """tryout1 has no spawn_task — only the flat tool set."""
    return TOOL_EXECUTORS[name](**args)


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    app = build_graph(build_message, dispatch, write_mermaid=False)
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