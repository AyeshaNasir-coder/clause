from pathlib import Path

PROMPTS_DIR = Path(__file__).parent #api/app/prompts


def get_prompt(name:str, version: str = "v1") ->str:
    return (PROMPTS_DIR / name / f"{version}.md").read_text(encoding="utf-8").strip()


# name: agent_name/prompt_type
# name: support/system/

# path: api/app/prompts/support/system/v3.md