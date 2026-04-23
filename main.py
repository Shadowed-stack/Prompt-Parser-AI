from orchestrator.workflow import run_pipeline
from search_engine.vector_store import populate

if __name__ == "__main__":
    populate()

    prompt = input("Enter prompt: ")
    result = run_pipeline(prompt)

    print("\nFINAL OUTPUT:\n")
    print(result)
