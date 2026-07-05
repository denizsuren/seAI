# seAI

### your beginner coding guide

seAI is a small **AI coding agent** that runs entirely on your own machine. You
describe a task in plain language; seAI writes the code, **saves it to a real
`.py` file**, runs it to check that it works, fixes it if it fails, and only then
reports back. It can also search the web when it needs current information.

It's built to be easy to read and easy to run — a friendly starting point for
seeing how an agent actually works under the hood.

---

## What seAI can do

- **Writes code to real files.** Every solution is saved as a `.py` file in your
  workspace — not just printed in the chat. You end up with files you can open,
  run, and keep.
- **Runs and verifies its own code.** It executes what it writes, reads the
  output, and fixes errors before telling you it's done.
- **Implements simple algorithms with ease.** Ask for a factorial, a sorting
  routine, string manipulation, a small parser — it writes it, tests it, done.
- **Searches the web.** When a task needs docs or current info, it can look it
  up (DuckDuckGo, no API key needed).
- **Runs locally and free.** Powered by [Ollama](https://ollama.com); no cloud,
  no account, no usage limits.

---

## How it works

seAI is a loop, not a single question-and-answer. The model looks at your goal,
decides whether to use a tool, sees the result, and repeats until the job is
done.

```
        your request
             |
             v
   +--------------------+   wants a tool?
   |       model        |------- yes ------>  run the tool
   |   (local, Ollama)  |                     (write / run / search ...)
   |                    |<--------------------  feed result back
   +--------------------+
             | no tool needed
             v
       final answer
             |
             v
     files saved in  workspace/
```

The loop has a safety cap (`MAX_STEPS`) so it can never spin forever.

### Tools the agent can use

| Tool           | What it does                                             |
|----------------|----------------------------------------------------------|
| `write_file`   | Save code to a `.py` (or any) file in the workspace      |
| `read_file`    | Read an existing file                                    |
| `list_files`   | See what's in the workspace                              |
| `run_python`   | Run a script or snippet, capture output and errors       |
| `run_command`  | Run a command like `pytest -q`                           |
| `web_search`   | Look something up on the web                             |

All file actions are locked to the `workspace/` folder — the agent can't touch
anything outside it.

---

## Routing - matching the model to the job

This is the heart of seAI. Instead of always calling one big model, seAI keeps a
few models in **tiers** and picks the right one for each request. Simple messages
go to a small, fast model; real work goes to a stronger one.

```
   your prompt
        |
        v
 +--------------+     easy  -->  llama3.2:3b        (small, fast)
 |  classify    |     medium -->  qwen2.5-coder:7b  (the workhorse)
 |  difficulty  |     hard  -->  qwen2.5-coder:7b   (same as medium)
 +--------------+
```

**How the difficulty is decided.** There are two ways (see `router.py`):

- **Heuristic (default, free):** quick rules - prompt length, "hard" keywords
  (like *algorithm*, *async*, *refactor*, *optimize*), whether code is involved,
  and whether it's a multi-step request. These add up to a score, and the score
  maps to a tier. No model call, instant.
- **Model-based:** ask the small model to label the task `easy` / `medium` /
  `hard` itself. More flexible, but costs one extra call.

**Why medium and hard share a model.** A laptop can't hold three large models in
memory at once. So seAI loads only two: a tiny model for easy chit-chat, and one
solid coder (`qwen2.5-coder:7b`) for everything that needs real thinking. The
"hard" tier points at the same coder as "medium" - the routing structure stays
in place, ready to grow if you run it on a bigger machine.

> **Why local at all?** seAI first ran on Hugging Face's hosted inference, but the
> free tier only includes a tiny monthly quota - it ran out after just a handful
> of requests. Running the models locally through Ollama removes that limit
> entirely: free, offline, and as many requests as you like.

---

## Setup

### 1. Install Ollama and pull the two models

Download Ollama from **https://ollama.com** (it runs in the background). Then, in
your terminal:

```bash
ollama pull llama3.2:3b        # ~2 GB   - the "easy" tier
ollama pull qwen2.5-coder:7b   # ~4.7 GB - the "medium/hard" coder
```

Check they're installed:

```bash
ollama list
```

### 2. Set up the project

```bash
cd seAI
python3 -m venv .venv
source .venv/bin/activate       # run this in every new terminal
pip install -r requirements.txt
cp .env.example .env            # defaults already point at local Ollama
```

### 3. Run the tests (no model needed)

```bash
pytest -q
```

The tests replace the model with a fake, so they verify the agent loop, tools and
router **offline and for free**. You should see `13 passed`.

### 4. Talk to the agent

```bash
python main.py
```

Then just type:

```
you> write a factorial function without any libraries, save it and run it
```

seAI will write `factorial.py`, run it, and confirm it works - and the file will
be sitting in `workspace/`.

---

## Usage

```bash
python main.py                  # chat mode (remembers the conversation)
python main.py --router         # chat mode + difficulty-based routing
python main.py "one-off task"   # run a single task and exit
```

Type `exit` (or Ctrl-C) to quit. In `--router` mode you'll see a line like
`[router] difficulty=easy -> llama3.2:3b` before each answer, showing which model
was chosen.

---

## Project structure

```
seAI/
├── main.py          # command-line chat entry point
├── agent.py         # the think -> act -> observe loop
├── tools.py         # the actions: write/read/list files, run code, web search
├── router.py        # difficulty-based model routing
├── config.py        # settings (model, endpoint, workspace, limits)
├── test_agent.py    # offline tests (no model / no API key)
├── requirements.txt
├── .env.example
└── workspace/       # where the agent's files land (sandboxed)
```

---

## Configuration

Everything is set in `.env` (copied from `.env.example`). The defaults target
local Ollama, so usually you don't need to change anything. Handy knobs:

- `MODEL_ID` - the model used when routing is off (default `qwen2.5-coder:7b`)
- `MAX_STEPS` - max tool steps per turn (default `12`)
- `TEMPERATURE` - lower is more precise, better for code (default `0.2`)

To use a hosted provider instead of Ollama (e.g. Groq's free tier), just change
`BASE_URL`, `API_KEY` and `MODEL_ID` in `.env` - no code changes needed.Also some examples of seAI's work has been added to workspace directory.
