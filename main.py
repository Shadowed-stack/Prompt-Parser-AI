"""
PRANAG-AI  —  Prompt Parser CLI

Usage:
    python main.py
    python main.py "wheat for Jodhpur at 48°C, drought resistant"
"""
import sys
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)

from search_engine.vector_store import populate
from orchestrator.workflow import run_pipeline


def main():
    # Seed vector store on first run (no-op if already populated)
    print("🌱  Initialising vector store …")
    populate()

    # Accept prompt from CLI arg or interactive input
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    else:
        prompt = input("\nEnter your PRANAG prompt: ").strip()
        if not prompt:
            print("No prompt entered.  Exiting.")
            return

    print(f"\n🚀  Running pipeline for: '{prompt}'\n")
    result = run_pipeline(prompt)

    print("─" * 60)
    print("PIPELINE OUTPUT")
    print("─" * 60)
    print(json.dumps(result, indent=2))

    if result.get("error") and not result.get("spec"):
        print("\n⚠️  Pipeline could not produce a valid spec.")
        print(f"   Error: {result['error']}")
        sys.exit(1)
    else:
        print("\n✅  spec.json produced successfully.")


if __name__ == "__main__":
    main()
