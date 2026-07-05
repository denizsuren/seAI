"""Tools the agent can call: file ops (sandboxed) + web search + run code."""
import os
import pathlib
import shlex
import subprocess
import sys

from config import WORKSPACE

os.makedirs(WORKSPACE, exist_ok=True)
_ROOT = pathlib.Path(WORKSPACE).resolve()

# How long a single command/script may run before it is killed.
RUN_TIMEOUT = int(os.getenv("RUN_TIMEOUT", "30"))
# Cap captured output so a runaway print loop can't flood the model's context.
_MAX_OUTPUT = 4000


def _safe_path(path: str) -> pathlib.Path:
    """Resolve `path` inside the workspace and refuse to escape the sandbox."""
    p = (_ROOT / path).resolve()
    if _ROOT != p and _ROOT not in p.parents:
        raise ValueError(f"Path '{path}' escapes the workspace sandbox.")
    return p


# --- Tool implementations ----------------------------------------------------
def read_file(path: str) -> str:
    """Return the UTF-8 text contents of a file in the workspace."""
    p = _safe_path(path)
    if not p.exists() or not p.is_file():
        return f"ERROR: file not found: {path}"
    return p.read_text(encoding="utf-8", errors="replace")


def write_file(path: str, content: str) -> str:
    """Create or overwrite a file in the workspace with `content`."""
    p = _safe_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"OK: wrote {len(content)} characters to {path}"


def list_files(directory: str = ".") -> str:
    """Recursively list files and folders under a workspace directory."""
    base = _safe_path(directory)
    if not base.exists():
        return f"ERROR: directory not found: {directory}"
    lines = []
    for entry in sorted(base.rglob("*")):
        tag = "DIR " if entry.is_dir() else "FILE"
        lines.append(f"{tag}  {entry.relative_to(_ROOT)}")
    return "\n".join(lines) if lines else "(empty)"


def web_search(query: str, max_results: int = 5) -> str:
    """Search the web (DuckDuckGo, no API key needed) and return top results."""
    try:
        from ddgs import DDGS
    except ImportError:
        return "ERROR: search dependency missing. Run: pip install ddgs"
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append(
                f"- {r.get('title')}\n"
                f"  {r.get('href')}\n"
                f"  {r.get('body')}"
            )
    return "\n\n".join(results) if results else "No results."


def _run(cmd: list[str]) -> str:
    """Run a subprocess in the workspace, capture output, enforce a timeout."""
    try:
        proc = subprocess.run(
            cmd,
            cwd=WORKSPACE,
            capture_output=True,
            text=True,
            timeout=RUN_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return f"ERROR: timed out after {RUN_TIMEOUT}s (possible infinite loop)."
    except FileNotFoundError:
        return f"ERROR: command not found: {cmd[0]}"

    out = (proc.stdout or "").strip()[:_MAX_OUTPUT]
    err = (proc.stderr or "").strip()[:_MAX_OUTPUT]
    parts = [f"exit_code: {proc.returncode}"]
    if out:
        parts.append(f"stdout:\n{out}")
    if err:
        parts.append(f"stderr:\n{err}")
    return "\n".join(parts)


def run_python(code: str = "", path: str = "") -> str:
    """Run Python code (inline `code` or an existing `path`) inside the workspace."""
    if path:
        target = _safe_path(path)
        if not target.exists():
            return f"ERROR: file not found: {path}"
        return _run([sys.executable, str(target)])
    if code:
        snippet = _safe_path("_snippet.py")
        snippet.write_text(code, encoding="utf-8")
        return _run([sys.executable, str(snippet)])
    return "ERROR: provide either 'code' or 'path'."


def run_command(command: str) -> str:
    """Run a shell-style command (e.g. 'pytest -q') in the workspace.

    Parsed with shlex, executed WITHOUT a shell, so pipes/redirects don't work
    (that's intentional — it keeps the surface small).
    """
    try:
        cmd = shlex.split(command)
    except ValueError as exc:
        return f"ERROR parsing command: {exc}"
    if not cmd:
        return "ERROR: empty command."
    return _run(cmd)


# --- Schemas (OpenAI function-calling format) --------------------------------
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file from the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the workspace root."}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a text file in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the workspace root."},
                    "content": {"type": "string", "description": "Full file contents to write."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "Recursively list files and folders in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {"type": "string", "description": "Directory to list. Defaults to workspace root."}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information, docs, or examples.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "max_results": {"type": "integer", "description": "How many results (default 5)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Run Python and get exit code, stdout and stderr. "
                           "Pass inline 'code' for a quick check, or 'path' to run a file you wrote.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Inline Python source to execute."},
                    "path": {"type": "string", "description": "Path of a workspace .py file to run."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a command in the workspace, e.g. 'pytest -q' or 'python solution.py'. "
                           "Returns exit code, stdout and stderr. No shell pipes/redirects.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Command line to run."},
                },
                "required": ["command"],
            },
        },
    },
]

# Map tool name -> the function that runs it.
DISPATCH = {
    "read_file": read_file,
    "write_file": write_file,
    "list_files": list_files,
    "web_search": web_search,
    "run_python": run_python,
    "run_command": run_command,
}
