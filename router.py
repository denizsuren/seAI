"""Difficulty-based model routing (a "cascading" router).

The idea in one line: match the model to the job.
  1. Keep models in tiers, from small/fast (easy) to strong (hard).
  2. Look at the incoming prompt and decide how hard it is.
  3. Send it to the model for that tier.

Simple tasks go to a small, fast model; hard tasks go to a stronger one. This
saves time and memory instead of always running the biggest model.
"""
import re

# Models per difficulty tier. These are local Ollama model names.
# Pull them first:  ollama pull llama3.2:3b   and   ollama pull qwen2.5-coder:7b
#
# Note: medium and hard use the SAME model on purpose. A laptop can't keep three
# large models in memory, so we load only two: a tiny one for easy chatter, and
# one solid coder (qwen2.5-coder:7b) for everything that actually needs thinking.
TIERS = {
    "easy":   "llama3.2:3b",       # short, simple messages
    "medium": "qwen2.5-coder:7b",  # real work
    "hard":   "qwen2.5-coder:7b",  # same as medium (memory limit)
}
ORDER = ["easy", "medium", "hard"]

# Words that hint a task is non-trivial (English + Turkish).
_HARD_WORDS = (
    "algorithm", "algoritma", "optimize", "optimiz", "complexity", "karmaşıklık",
    "prove", "kanıtla", "derive", "design", "tasarla", "architecture", "mimari",
    "refactor", "concurrency", "async", "recursion", "özyineleme", "benchmark",
    "debug", "stack trace", "exception", "regex", "multithread", "distributed",
)
# Words that suggest code is involved.
_CODE_SIGNALS = ("```", "def ", "class ", "import ", "function", "fonksiyon", "kod", "code")


def classify_heuristic(prompt: str) -> str:
    """Guess difficulty with cheap, deterministic rules: 'easy'|'medium'|'hard'."""
    t = prompt.lower()
    score = 0

    # 1) Longer prompts tend to be more involved.
    if len(t) > 400:
        score += 2
    elif len(t) > 140:
        score += 1

    # 2) The more distinct "hard" keywords, the harder (capped at +3).
    hard_hits = sum(1 for w in _HARD_WORDS if w in t)
    score += min(hard_hits, 3)

    # 3) Code present or requested.
    if any(s in t for s in _CODE_SIGNALS):
        score += 1

    # 4) Multi-step or multi-question requests.
    steps = len(re.findall(r"\b(then|after|next|sonra|önce|adım|step)\b", t))
    if steps >= 2 or t.count("?") >= 2:
        score += 1

    # Turn the score into a tier.
    if score >= 4:
        return "hard"
    if score >= 2:
        return "medium"
    return "easy"


def classify_llm(prompt: str, client) -> str:
    """Alternative: let the small model rate the difficulty itself."""
    instruction = (
        "Rate the difficulty of the task below for an AI assistant. "
        "Answer with exactly one word: easy, medium, or hard.\n\nTask: " + prompt
    )
    resp = client.chat.completions.create(
        model=TIERS["easy"],
        messages=[{"role": "user", "content": instruction}],
        temperature=0,
        max_tokens=3,
    )
    label = (resp.choices[0].message.content or "").strip().lower()
    return label if label in TIERS else "medium"


def pick_model(prompt: str, client=None, use_llm: bool = False) -> tuple[str, str]:
    """Return (difficulty_level, model_id) for a prompt.

    use_llm=True classifies with the small model (needs `client`);
    otherwise the free heuristic is used.
    """
    level = classify_llm(prompt, client) if (use_llm and client) else classify_heuristic(prompt)
    return level, TIERS[level]
