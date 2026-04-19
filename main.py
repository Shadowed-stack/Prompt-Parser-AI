from orchestrator.workflow import run_pipeline

if __name__ == "__main__":
    prompt = input("Enter prompt: ")
    result = run_pipeline(prompt)

    print("\nFINAL OUTPUT:\n")
    print(result["spec"])
