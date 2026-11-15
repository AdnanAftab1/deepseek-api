from __future__ import annotations

import json
import os
import zipfile
from typing import Any

from yachalk import chalk
from openai import OpenAI

# ------------------------------------------------------------------ client
client = OpenAI(
    base_url="https://xqxmedozyom24kmy522qsbacqu0qtiaq.lambda-url.ap-south-1.on.aws/v1",
    api_key="sk-dummy-key",
)

MAX_ATTEMPTS        = 5
RETRY_DELAY_SECONDS = 1.0
MAX_TURNS           = 25
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


def read_file(path: str) -> dict:
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

    return {
        "status": "ok",
        "path": path,
        "bytes_read": total_bytes,
        "total_bytes": total_bytes,
        "total_lines": total_lines,
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