"""Tests for seAI.

Run with:  pytest -q
No model or API key is needed: the model is replaced by a scripted FakeClient,
so the agent loop, tools and router are all tested offline and for free.
"""
import json
import pathlib
from types import SimpleNamespace

import pytest

import tools
from agent import CodeAgent
from config import WORKSPACE


# --------------------------------------------------------------------------- #
# 1) Tools — the actions the agent can take
# --------------------------------------------------------------------------- #
def test_write_then_read():
    tools.write_file("t/x.txt", "hello")
    assert tools.read_file("t/x.txt") == "hello"


def test_sandbox_blocks_escape():
    # File tools must never reach outside the workspace.
    with pytest.raises(ValueError):
        tools.read_file("../../../etc/passwd")


def test_run_python_ok():
    out = tools.run_python(code="print(6 * 7)")
    assert "exit_code: 0" in out and "42" in out


def test_run_python_surfaces_errors():
    out = tools.run_python(code="raise ValueError('boom')")
    assert "exit_code: 1" in out and "boom" in out


def test_run_python_timeout(monkeypatch):
    monkeypatch.setattr(tools, "RUN_TIMEOUT", 2)  # keep the test fast
    out = tools.run_python(code="while True: pass")
    assert "timed out" in out.lower()


def test_run_command_runs_a_file():
    tools.write_file("hey.py", "print('from file')")
    out = tools.run_command("python hey.py")
    assert "from file" in out and "exit_code: 0" in out


# --------------------------------------------------------------------------- #
# 2) Agent loop — driven by a fake, scripted model (no API calls)
# --------------------------------------------------------------------------- #
def _msg(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _tool_call(call_id, name, arguments):
    return SimpleNamespace(id=call_id,
                           function=SimpleNamespace(name=name, arguments=arguments))


def _response(message):
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeClient:
    """Plays back a fixed list of model responses, one per create() call."""

    def __init__(self, scripted):
        self._scripted = list(scripted)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **_kwargs):
        return _response(self._scripted.pop(0))


def test_agent_writes_runs_and_reports():
    """Model decides to: write a file, run it, then give a final answer."""
    script = [
        _msg(tool_calls=[_tool_call("c1", "write_file", json.dumps({
            "path": "add.py",
            "content": "def add(a, b):\n    return a + b\n\nprint(add(2, 3))\n",
        }))]),
        _msg(tool_calls=[_tool_call("c2", "run_python", json.dumps({"path": "add.py"}))]),
        _msg(content="Done — add.py prints 5."),
    ]
    agent = CodeAgent(verbose=False, client=FakeClient(script))
    answer = agent.step("Create add.py and verify it works.")
    assert "5" in answer
    assert (pathlib.Path(WORKSPACE) / "add.py").exists()
    tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
    assert any("exit_code: 0" in m["content"] for m in tool_msgs)


def test_agent_handles_bad_tool_args_gracefully():
    """A failing tool must not crash the loop; the error is fed back to the model."""
    script = [
        _msg(tool_calls=[_tool_call("c1", "read_file", '{"path": "does_not_exist"}')]),
        _msg(content="File was missing, as reported."),
    ]
    agent = CodeAgent(verbose=False, client=FakeClient(script))
    answer = agent.step("Read a file that isn't there.")
    assert "missing" in answer.lower()


def test_agent_handles_text_format_tool_calls():
    """Some local models print tool calls as JSON text instead of the structured
    tool_calls field. The agent must still detect and run them."""
    text_call = ('{\n  "name": "write_file",\n  "arguments": {\n'
                 '    "path": "greet.py",\n    "content": "print(\'hi\')"\n  }\n}')
    script = [
        _msg(content=text_call, tool_calls=None),
        _msg(content="Wrote greet.py."),
    ]
    agent = CodeAgent(verbose=False, client=FakeClient(script))
    answer = agent.step("create greet.py")
    assert (pathlib.Path(WORKSPACE) / "greet.py").exists()
    assert "greet.py" in answer


def test_sanitize_removes_surrogates():
    """Non-UTF-8 bytes from the terminal must not crash the JSON request."""
    from agent import _clean_text, _sanitize
    bad = "merhaba \udcc4 dunya"
    _clean_text(bad).encode("utf-8")                       # must not raise
    _sanitize([{"role": "user", "content": bad}])[0]["content"].encode("utf-8")


# --------------------------------------------------------------------------- #
# 3) Router — difficulty classification -> model choice
# --------------------------------------------------------------------------- #
def test_router_easy_for_short_simple():
    from router import TIERS, pick_model
    level, model = pick_model("merhaba nasilsin")
    assert level == "easy" and model == TIERS["easy"]


def test_router_hard_for_complex_task():
    from router import TIERS, pick_model
    prompt = ("Refactor this recursion into an async algorithm, then optimize its "
              "complexity and add a benchmark. First profile it, after that fix the bug.")
    level, model = pick_model(prompt)
    assert level == "hard" and model == TIERS["hard"]


def test_router_llm_path_with_fake_client():
    from router import TIERS, pick_model
    fake = FakeClient([_msg(content="hard")])   # model classifies as 'hard'
    level, model = pick_model("anything", client=fake, use_llm=True)
    assert level == "hard" and model == TIERS["hard"]


def teardown_module(_module):
    """Remove files created during the tests."""
    for name in ("t", "hey.py", "add.py", "greet.py", "_snippet.py"):
        p = pathlib.Path(WORKSPACE) / name
        if p.is_dir():
            for f in p.rglob("*"):
                f.unlink(missing_ok=True)
            p.rmdir()
        elif p.exists():
            p.unlink()
