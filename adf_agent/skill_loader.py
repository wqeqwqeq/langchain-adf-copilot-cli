"""
Skills 发现和加载器

实现 Skills 两层加载机制：
- Level 1: scan_skills() - 扫描并加载所有 Skills 元数据到 system prompt
- Level 2: load_skill(skill_name) - 按需加载指定 Skill 的详细指令

Skills 目录结构：
    my-skill/
    ├── SKILL.md          # 必需：YAML frontmatter + 指令
    ├── scripts/          # 可选：可执行脚本
    ├── references/       # 可选：参考文档
    └── assets/           # 可选：模板和资源

SKILL.md 格式：
    ---
    name: skill-name
    description: 何时使用此 skill 的描述
    ---
    # Skill Title
    详细指令内容...
"""

import re
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import yaml


# 默认 Skills 搜索路径（项目级优先，用户级兜底）
DEFAULT_SKILL_PATHS = [
    Path.cwd() / ".claude" / "skills",   # 项目级 Skills - 优先
    Path.home() / ".claude" / "skills",   # 用户级 Skills - 兜底
]


@dataclass
class SkillMetadata:
    """
    Skill 元数据（Level 1）

    启动时从 YAML frontmatter 解析，用于注入 system prompt。
    """
    name: str               # skill 唯一名称
    description: str        # 何时使用此 skill 的描述
    skill_path: Path        # skill 目录路径

    def to_prompt_line(self) -> str:
        """生成 system prompt 中的单行描述"""
        return f"- **{self.name}**: {self.description}"


@dataclass
class SkillContent:
    """
    Skill 完整内容（Level 2）

    用户请求匹配时加载，包含 SKILL.md 的完整指令。
    """
    metadata: SkillMetadata
    instructions: str  # SKILL.md body 内容


class SkillLoader:
    """
    Skills 加载器

    核心职责：
    1. scan_skills(): 发现文件系统中的 Skills，解析元数据
    2. load_skill(): 按需加载 Skill 详细内容
    """

    def __init__(self, skill_paths: list[Path] | None = None):
        """
        初始化加载器

        Args:
            skill_paths: 自定义 Skills 搜索路径，默认为:
                - .claude/skills/ (项目级，优先)
                - ~/.claude/skills/ (用户级，兜底)
        """
        self.skill_paths = skill_paths or DEFAULT_SKILL_PATHS
        self._metadata_cache: dict[str, SkillMetadata] = {}

    def scan_skills(self) -> list[SkillMetadata]:
        """
        Level 1: 扫描所有 Skills 元数据

        遍历 skill_paths，查找包含 SKILL.md 的目录，
        解析 YAML frontmatter 提取 name 和 description。

        Returns:
            所有发现的 Skills 元数据列表
        """
        skills = []
        seen_names = set()

        for base_path in self.skill_paths:
            if not base_path.exists():
                continue

            for skill_dir in base_path.iterdir():
                if not skill_dir.is_dir():
                    continue

                skill_md = skill_dir / "SKILL.md"
                if not skill_md.exists():
                    continue

                metadata = self._parse_skill_metadata(skill_md)
                if metadata and metadata.name not in seen_names:
                    skills.append(metadata)
                    seen_names.add(metadata.name)
                    self._metadata_cache[metadata.name] = metadata

        return skills

    def _parse_skill_metadata(self, skill_md_path: Path) -> Optional[SkillMetadata]:
        """
        解析 SKILL.md 的 YAML frontmatter

        Args:
            skill_md_path: SKILL.md 文件路径

        Returns:
            解析后的元数据，解析失败返回 None
        """
        try:
            content = skill_md_path.read_text(encoding="utf-8")
        except Exception:
            return None

        frontmatter_match = re.match(
            r'^---\s*\n(.*?)\n---\s*\n',
            content,
            re.DOTALL
        )

        if not frontmatter_match:
            return None

        try:
            frontmatter = yaml.safe_load(frontmatter_match.group(1))

            name = frontmatter.get("name", "")
            description = frontmatter.get("description", "")

            if not name:
                return None

            return SkillMetadata(
                name=name,
                description=description,
                skill_path=skill_md_path.parent,
            )
        except yaml.YAMLError:
            return None

    def load_skill(self, skill_name: str) -> Optional[SkillContent]:
        """
        Level 2: 加载 Skill 完整内容

        读取 SKILL.md 的完整指令。

        Args:
            skill_name: Skill 名称（如 "news-extractor"）

        Returns:
            Skill 完整内容，未找到返回 None
        """
        metadata = self._metadata_cache.get(skill_name)
        if not metadata:
            self.scan_skills()
            metadata = self._metadata_cache.get(skill_name)

        if not metadata:
            return None

        skill_md = metadata.skill_path / "SKILL.md"
        try:
            content = skill_md.read_text(encoding="utf-8")
        except Exception:
            return None

        body_match = re.match(
            r'^---\s*\n.*?\n---\s*\n(.*)$',
            content,
            re.DOTALL
        )
        instructions = body_match.group(1).strip() if body_match else content

        return SkillContent(
            metadata=metadata,
            instructions=instructions,
        )
