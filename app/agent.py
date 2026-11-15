from __future__ import annotations

import json
import re
import sys
import time
from typing import Any, Callable, TypedDict

from yachalk import chalk
from langgraph.graph import StateGraph, START, END

from helpers import (
    client,
    _status, _tool, _dim_json,
    MAX_ATTEMPTS, RETRY_DELAY_SECONDS, MAX_TURNS, WORKDIR,
)

MAX_DEPTH = 4

# A dispatch callable: (name, args, depth) -> result dict.
Dispatch = Callable[[str, dict, int], dict]
# A message builder: (state) -> str.
BuildMessage = Callable[["AgentState"], str]


# ------------------------------------------------------------------ parsing
# Matches both ```json ... ``` and bare ``` ... ``` fences. Non-greedy so
# adjacent blocks are captured separately. DOTALL so the body can span lines.
_FENCE_RE = re.compile(
    r"```(?:[ \t]*json[ \t]*)?\r?\n(.*?)\r?\n[ \t]*```",
    re.DOTALL | re.IGNORECASE,
)


def _strip_fences(text: str) -> str:
    """Strip a single outer ```...``` block if the WHOLE response is one."""
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
    # Accept anything that *could* contain a JSON payload: it starts with [ / {
    # or it contains a fenced block we can dig into.
    if s.startswith("[") or s.startswith("{") or "```" in s:
        return False
    return True


def _normalize_calls(data: Any) -> list[dict]:
    """Validate/normalize a decoded JSON payload into a list of tool calls."""
    if isinstance(data, dict) and "calls" in data:
        data = data["calls"]
    if not isinstance(data, list):
        raise ValueError("Payload is not a JSON array of tool calls.")
    if data == []:
        raise ValueError(
            "Empty array is not a valid response. "
            "Call finish_task when done, otherwise emit tool calls."
        )
    for el in data:
        if not isinstance(el, dict) or "tool" not in el:
            raise ValueError(f"Malformed tool call element: {el!r}")
    return data


def _try_decode(text: str) -> tuple[list[dict] | None, str | None]:
    """Try to decode `text` as a tool-call payload.

    Returns (calls, None) on success, or (None, error_message) on failure.
    Never raises.
    """
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError) as e:
        return None, f"json: {e}"
    try:
        return _normalize_calls(data), None
    except ValueError as e:
        return None, str(e)


def _extract_fenced_json(raw: str) -> list[dict]:
    """Fallback: find ```json ... ``` (or bare ``` ... ```) blocks in `raw`,
    decode each, and require EXACTLY ONE to be a valid tool-call payload.

    Used when `raw` itself fails to JSONify (e.g. it is a single string that
    embeds a fenced JSON segment). Raises ValueError if zero or >1 valid
    payloads are found.
    """
    blocks = _FENCE_RE.findall(raw)
    if not blocks:
        raise ValueError("No fenced ```json block found to fall back to.")

    successes: list[list[dict]] = []
    failures: list[str] = []
    for idx, body in enumerate(blocks, 1):
        body_calls, body_err = _try_decode(body.strip())
        if body_calls is not None:
            successes.append(body_calls)
        else:
            failures.append(f"block {idx}: {body_err}")

    if len(successes) == 1:
        return successes[0]
    if len(successes) > 1:
        raise ValueError(
            f"Ambiguous response: found {len(successes)} valid JSON tool-call "
            f"payloads in fenced blocks; expected exactly 1. "
            f"Failures: {failures}"
        )
    raise ValueError(
        f"No valid JSON tool-call payload found in {len(blocks)} fenced block(s). "
        f"Failures: {failures}"
    )


def _parse_calls(raw: str) -> list[dict]:
    """Parse a model reply into a validated list of tool calls.

    Order of attempts:
      1. Whole reply, after stripping a single outer fence (fast path).
      2. If that fails to JSONify, scan every ```json / ``` fence found
         anywhere in the reply (e.g. raw is a single string embedding a
         fenced JSON segment). More than one valid payload is an error.

    Contract:
      - exactly one payload must parse and validate;
      - zero → ValueError describing all failures;
      - more than one → ValueError (ambiguous; we refuse to guess).
    """
    if not raw or not raw.strip():
        raise ValueError("Empty model response.")

    # --- Path 1: whole-reply fast path ----------------------------------
    whole = _strip_fences(raw)
    calls, err = _try_decode(whole)
    if calls is not None:
        return calls

    # --- Path 2: raw failed to JSONify — look for embedded fenced JSON --
    # If `raw` is a single string that embeds ```json ...``` segment(s), pull
    # them out. Exactly one valid payload is required; more than one is
    # ambiguous and rejected.
    try:
        return _extract_fenced_json(raw)
    except ValueError as fence_err:
        raise ValueError(
            f"{err or 'Could not parse response as JSON.'} | {fence_err}"
        )


# ------------------------------------------------------------------ state
class AgentState(TypedDict, total=False):
    user_request: str
    parent_context: str
    depth: int
    history: list[dict]
    attempts: int
    turns: int
    raw_response: str | None
    calls: list[dict] | None
    last_results: list[dict] | None
    error: str | None
    finished: bool
    summary: str


# ------------------------------------------------------------------ nodes
def make_plan_node(build_message: BuildMessage):
    def plan_node(state: AgentState) -> AgentState:
        attempts = state.get("attempts", 0) + 1
        turns    = state.get("turns", 0) + 1
        depth    = state.get("depth", 0)

        print("\n" + (
            chalk.bg_blue.white.bold(" plan ")
            + " " + chalk.blue(f"turn {turns}/{MAX_TURNS}")
            + chalk.gray(" · ") + chalk.yellow(f"attempt {attempts}/{MAX_ATTEMPTS}")
            + (chalk.gray(" · ") + chalk.magenta(f"depth {depth}/{MAX_DEPTH}") if depth else "")
        ))

        raw = None
        try:
            resp = client.chat.completions.create(
                model="deepseek-v3",
                messages=[{"role": "user", "content": build_message(state)}],
            )
            raw = resp.choices[0].message.content
        except Exception as e:
            print(chalk.red(f"[plan] API error: {e}") + chalk.gray(f" raw={raw!r}"))
            return {"attempts": attempts, "turns": turns, "error": f"api: {e}", "calls": None}

        if _looks_degenerate(raw):
            print(chalk.yellow("[plan] degenerate response"))
            return {"attempts": attempts, "turns": turns,
                    "raw_response": raw, "error": "degenerate", "calls": None}

        try:
            calls = _parse_calls(raw)
        except (json.JSONDecodeError, ValueError) as e:
            print(chalk.red(f"[plan] parse failed: {e}"))
            print(chalk.gray(f"       raw: {(raw or '')[:200]!r}"))
            if (raw or "").strip() == "[消息发送过于频繁，请稍后重试]":
                time.sleep(RETRY_DELAY_SECONDS * 10)
            return {"attempts": attempts, "turns": turns,
                    "raw_response": raw, "error": f"parse: {e}", "calls": None}

        if any(c.get("tool") == "finish_task" for c in calls):
            if calls[-1].get("tool") != "finish_task":
                print(chalk.red("[plan] finish_task must be the last call"))
                return {"attempts": attempts, "turns": turns,
                        "raw_response": raw, "calls": None,
                        "error": "finish_task must be the last call"}
            print(chalk.green("✔ finish_task called — finished."))
            summary = (calls[-1].get("args") or {}).get("summary", "")
            return {"attempts": 0, "turns": turns, "calls": calls,
                    "raw_response": raw, "error": None, "finished": True,
                    "summary": summary}

        print(chalk.gray(f"[plan] parsed {len(calls)} call(s)."))
        return {"attempts": 0, "turns": turns, "calls": calls,
                "raw_response": raw, "error": None, "finished": False}
    return plan_node


def wait_node(state: AgentState) -> AgentState:
    print(chalk.gray(f"⏳ waiting {RETRY_DELAY_SECONDS}s…"))
    time.sleep(RETRY_DELAY_SECONDS)
    return state


def make_execute_node(dispatch: Dispatch):
    """dispatch(name, args, depth) -> result dict.

    `dispatch` is responsible for its own tool set. Unknown tool names must
    raise `KeyError`; any exception is reported back to the model as an error
    result and does NOT abort the loop.
    """
    def execute_node(state: AgentState) -> AgentState:
        calls = state.get("calls") or []
        depth = state.get("depth", 0)
        print("\n" + chalk.bg_magenta.white.bold(f" execute · {len(calls)} call(s) "))
        results: list[dict] = []
        for i, call in enumerate(calls):
            name = call.get("tool")
            args = call.get("args", {}) or {}
            print(f"  {chalk.gray(f'[{i+1}/{len(calls)}]')} {_tool(name)} {_dim_json(args)}")

            try:
                res = dispatch(name, args, depth)
            except KeyError:
                res = {"status": "error", "message": f"unknown tool: {name!r}"}
            except Exception as e:
                res = {"status": "error", "message": str(e)}

            results.append({"tool": name, "args": args, "result": res})
            mark  = "✔" if res.get("status") == "ok" else "✖"
            color = chalk.green if res.get("status") == "ok" else chalk.red
            print(f"      {color(mark)} {_status(res)}")

        history = list(state.get("history", []))
        history.append({"calls": state.get("calls") or [], "results": results})
        return {"last_results": results, "history": history, "calls": None,
                "finished": state.get("finished", False)}
    return execute_node


def done_node(state: AgentState) -> AgentState:
    print("\n" + chalk.green.bold("✔ done"))
    return state


def give_up_node(state: AgentState) -> AgentState:
    print("\n" + chalk.bg_red.white.bold(" give-up ") + " " +
          chalk.red(f"after {state.get('attempts')} attempt(s) on turn {state.get('turns')}")
          + chalk.gray(f" · last error: {state.get('error')}"))
    return state


# ------------------------------------------------------------------ routing
def route_after_plan(state: AgentState) -> str:
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
def build_graph(build_message: BuildMessage, dispatch: Dispatch,
                write_mermaid: bool = False) -> Any:
    g = StateGraph(AgentState)
    g.add_node("plan",    make_plan_node(build_message))
    g.add_node("wait",    wait_node)
    g.add_node("execute", make_execute_node(dispatch))
    g.add_node("done",    done_node)
    g.add_node("give_up", give_up_node)

    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", route_after_plan,
        {"execute": "execute", "wait": "wait", "give_up": "give_up"})
    g.add_edge("wait", "plan")
    g.add_conditional_edges("execute", route_after_execute,
        {"plan": "plan", "done": "done", "give_up": "give_up"})
    g.add_edge("done", END)
    g.add_edge("give_up", END)

    compiled = g.compile()
    if write_mermaid:
        with open("graph.mmd", "w") as f:
            f.write(compiled.get_graph().draw_mermaid())
    return compiled


# ------------------------------------------------------------------ I/O
DEFAULT_PROMPT = """
    Read the file prompt.txt to know the request.
"""


def read_request() -> str:
    if not sys.stdin.isatty():
        data = sys.stdin.read().strip()
        if not data:
            raise SystemExit("Empty request; aborting.")
        return data

    print(chalk.blue("Enter your request."))
    print(chalk.gray("Finish with a line containing only 'END' "
                     "(or Ctrl+Z+Enter on Windows / Ctrl+D on Unix)."))
    print(chalk.gray("Type '1' on its own line to use the prompt from prompt.txt.") + "\n")

    lines: list[str] = []
    while True:
        try:
            line = input(chalk.cyan("> ") if not lines else chalk.gray("… "))
        except EOFError:
            break
        if line.strip().upper() == "END":
            break
        if not lines and line.strip() == "1":
            try:
                with open("prompt.txt", "r", encoding="utf-8") as f:
                    content = f.read().strip()
            except OSError as e:
                raise SystemExit(f"Could not read prompt.txt: {e}")
            if not content:
                raise SystemExit("prompt.txt is empty; aborting.")
            print(chalk.magenta("[using prompt from prompt.txt]"))
            return content
        lines.append(line)

    data = "\n".join(lines).strip()
    if not data:
        raise SystemExit("Empty request; aborting.")
    return data


def print_summary(final_state: dict) -> None:
    print("\n" + chalk.bg_black.white.bold(" summary "))
    print(chalk.blue("turns used: ") + chalk.white(str(final_state.get("turns"))))
    rows = final_state.get("last_results") or []
    if rows:
        for r in rows:
            tool = r.get("tool", "?")
            res  = r.get("result", {})
            msg  = res.get("message", res.get("summary", ""))
            print(f"  {chalk.cyan(tool):<14} {_status(res)} " + chalk.gray(str(msg)[:120]))
    else:
        print(chalk.gray("(no tool results)"))
    finished = final_state.get("finished")
    print(chalk.blue("finished:   ") + (chalk.green("yes") if finished else chalk.red("no")))