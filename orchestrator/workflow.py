import uuid
from shared.config import settings

from orchestrator.prompt_parser import parse_prompt
from search_engine.similarity_search import search_traits
from orchestrator.research_fetcher import fetch_research
from orchestrator.spec_builder import build_spec
from orchestrator.output_validator import validate_spec


def run_pipeline(prompt):
    retries = 0

    while retries < settings.max_retries:
        parsed = parse_prompt(prompt)
        traits = search_traits(prompt)
        research = fetch_research(prompt)

        spec = build_spec(parsed, traits, research)
        valid = validate_spec(spec)

        if valid:
            return {
                "pipeline_id": str(uuid.uuid4()),
                "spec": valid
            }

        retries += 1

    return {"spec": None}
