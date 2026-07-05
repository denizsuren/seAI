"""The agent loop: think -> call tool(s) -> observe -> repeat -> answer."""
import json
from types import SimpleNamespace

from openai import OpenAI

from config import API_KEY, BASE_URL, MAX_STEPS, MODEL_ID, TEMPERATURE
from router import pick_model
from tools import DISPATCH, TOOL_SCHEMAS

SYSTEM_PROMPT = """You are a coding assistant agent.

You can:
- read, write and list files in the user's workspace,
- search the web for docs, examples or current information,
- run Python and shell commands and read their output.

Guidelines:
- Work step by step. Call a tool whenever it moves the task forward; do not guess
  file contents you can read, and do not invent facts you can look up.
- IMPORTANT: when the user asks you to create, update, fix or write code, you MUST
  save it to a file with the write_file tool. Do NOT just print the code in your
  reply — printing it without calling write_file does not count as done.
- After writing code, VERIFY it: run it with run_python (or write a small test and
  run `pytest`). If it fails, read the error, fix the file, and run again.
- Only after the file is written and runs cleanly, give a short final answer with
  NO tool call.
- Prefer small, verifiable steps over one giant action.
"""


def _to_tool_calls(tool_calls):
    """Normalise tool_calls into OpenAI-style dicts for the next turn."""
    calls = []
    for tc in tool_calls:
        args = tc.function.arguments
        args_str = args if isinstance(args, str) else json.dumps(args)
        calls.append({
            "id": tc.id,
            "type": "function",
            "function": {"name": tc.function.name, "arguments": args_str},
        })
    return calls


def _parse_text_tool_calls(content: str):
    """Fallback for local models (e.g. some Ollama models) that print tool calls
    as JSON text in the content instead of the structured tool_calls field.

    Scans the text for {"name": ..., "arguments": ...} objects and turns them
    into tool-call objects the normal loop can execute.
    """
    if not content:
        return []
    decoder = json.JSONDecoder()
    calls, idx, n = [], 0, len(content)
    while idx < n:
        brace = content.find("{", idx)
        if brace == -1:
            break
        try:
            obj, end = decoder.raw_decode(content, brace)
            idx = end
        except json.JSONDecodeError:
            idx = brace + 1
            continue
        if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
            args = obj["arguments"]
            args_str = args if isinstance(args, str) else json.dumps(args)
            calls.append(SimpleNamespace(
                id=f"call_{len(calls)}",
                function=SimpleNamespace(name=obj["name"], arguments=args_str),
            ))
    return calls


def _clean_text(s):
    """Drop lone surrogates (e.g. '\\udcc4' from a non-UTF-8 terminal byte) so the
    JSON encoder in the OpenAI client doesn't crash with UnicodeEncodeError."""
    if not isinstance(s, str):
        return s
    return s.encode("utf-8", "replace").decode("utf-8")


def _sanitize(obj):
    """Recursively clean all strings in a messages payload before sending."""
    if isinstance(obj, str):
        return _clean_text(obj)
    if isinstance(obj, list):
        return [_sanitize(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    return obj


def _parse_args(tc):
    """Return the tool arguments as a dict, regardless of string/dict form."""
    args = tc.function.arguments
    if isinstance(args, str):
        return json.loads(args) if args.strip() else {}
    return args or {}


class CodeAgent:
    def __init__(self, verbose: bool = True, client=None, use_router: bool = False,
                 router_llm: bool = False):
        # Inject a client for testing; otherwise use any OpenAI-compatible endpoint.
        self.client = client or OpenAI(base_url=BASE_URL, api_key=API_KEY)
        self.verbose = verbose
        self.use_router = use_router          # pick a model per task when True
        self.router_llm = router_llm          # classify difficulty with a model vs heuristic
        self.model = MODEL_ID
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def _log(self, text: str):
        if self.verbose:
            print(text)

    def step(self, user_input: str) -> str:
        """Run one user turn to completion (may involve several tool calls)."""
        user_input = _clean_text(user_input)
        if self.use_router:
            level, self.model = pick_model(user_input, client=self.client, use_llm=self.router_llm)
            self._log(f"  [router] difficulty={level} -> {self.model}")
        self.messages.append({"role": "user", "content": user_input})

        for i in range(1, MAX_STEPS + 1):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=_sanitize(self.messages),
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                temperature=TEMPERATURE,
            )
            msg = response.choices[0].message

            # Prefer structured tool_calls; fall back to JSON-in-text for local models.
            tool_calls = list(msg.tool_calls) if msg.tool_calls else _parse_text_tool_calls(msg.content)

            # No tool call -> this is the final answer.
            if not tool_calls:
                self.messages.append({"role": "assistant", "content": msg.content or ""})
                return msg.content or ""

            # Record the assistant's tool-call turn.
            self.messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": _to_tool_calls(tool_calls),
            })

            # Execute each requested tool and feed the result back.
            for tc in tool_calls:
                name = tc.function.name
                try:
                    args = _parse_args(tc)
                    fn = DISPATCH.get(name)
                    result = fn(**args) if fn else f"ERROR: unknown tool '{name}'"
                except Exception as exc:  # noqa: BLE001 - surface any tool error to the model
                    result = f"ERROR while running {name}: {exc}"

                self._log(f"  [tool] {name}({args})  ->  {str(result)[:120].strip()}...")
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(result),
                })

        return "Reached the maximum number of steps without finishing."
