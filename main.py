"""seAI - interactive command-line chat. Run: python main.py"""
import sys

from agent import CodeAgent
from config import API_KEY, BASE_URL, MODEL_ID, WORKSPACE


def main():
    # A key is only needed for hosted providers, not local Ollama.
    needs_key = ("huggingface" in BASE_URL or "groq" in BASE_URL)
    if needs_key and API_KEY in (None, "ollama", "not-needed"):
        print("This endpoint needs an API key. Set API_KEY in your .env, then retry.")
        print("Or run fully free & local: install Ollama, keep BASE_URL=http://localhost:11434/v1")
        sys.exit(1)

    print(f"Endpoint:  {BASE_URL}")
    print(f"Model:     {MODEL_ID}")
    print(f"Workspace: {WORKSPACE}")

    # Flags: --router (heuristic), --router-llm (model-based classification)
    args = [a for a in sys.argv[1:]]
    use_router = "--router" in args or "--router-llm" in args
    router_llm = "--router-llm" in args
    args = [a for a in args if not a.startswith("--")]
    if use_router:
        print(f"Router:    on ({'llm' if router_llm else 'heuristic'})")
    print("Type your request. 'exit' or Ctrl-C to quit.\n")

    agent = CodeAgent(use_router=use_router, router_llm=router_llm)

    # Allow a one-shot goal passed on the command line: python main.py "do X"
    if args:
        goal = " ".join(args)
        print(f"you> {goal}")
        print(f"\nagent> {agent.step(goal)}\n")
        return

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        if user_input.lower() in {"exit", "quit"}:
            print("bye")
            break
        if not user_input:
            continue
        answer = agent.step(user_input)
        print(f"\nagent> {answer}\n")


if __name__ == "__main__":
    main()
