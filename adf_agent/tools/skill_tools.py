"""
Skill 工具

提供 load_skill 工具，用于按需加载 Skill 详细指令（Level 2）。
"""

from langchain.tools import tool, ToolRuntime

from ..context import ADFAgentContext


@tool
def load_skill(skill_name: str, runtime: ToolRuntime[ADFAgentContext]) -> str:
    """
    Load a skill's detailed instructions.

    This tool reads the SKILL.md file for the specified skill and returns
    its complete instructions. Use this when the user's request matches
    a skill's description from the available skills list.

    Args:
        skill_name: Name of the skill to load (e.g., 'news-extractor')
    """
    loader = runtime.context.skill_loader
    if loader is None:
        return "[FAILED] Skills system is not initialized."

    skill_content = loader.load_skill(skill_name)

    if not skill_content:
        skills = loader.scan_skills()
        if skills:
            available = [s.name for s in skills]
            return f"Skill '{skill_name}' not found. Available skills: {', '.join(available)}"
        else:
            return f"Skill '{skill_name}' not found. No skills are currently available."

    skill_path = skill_content.metadata.skill_path

    return f"""# Skill: {skill_name}

{skill_content.instructions}

## Skill Directory: `{skill_path}`"""


SKILL_TOOLS = [load_skill]
