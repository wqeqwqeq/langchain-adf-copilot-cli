"""
ADF Agent Context

Provides configuration and state management needed at Agent runtime.
"""

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from azure.identity import DefaultAzureCredential

from .skill_loader import SkillLoader


def _use_workspace() -> bool:
    """Check whether to use workspace directory (otherwise use temp files)"""
    val = os.getenv("USE_WORKSPACE", "true").lower()
    return val in ("true", "1", "yes")


@dataclass
class ADFConfig:
    """
    ADF Configuration - may be empty, requires validation

    Loaded from environment variables; if not configured, the user will be prompted.
    """
    resource_group: Optional[str] = None
    factory_name: Optional[str] = None
    subscription_id: Optional[str] = None  # Optional, SDK auto-detects

    def is_configured(self) -> bool:
        """Check if required configuration is complete"""
        return bool(self.resource_group and self.factory_name)

    def missing_fields(self) -> list[str]:
        """Return list of missing field names"""
        missing = []
        if not self.resource_group:
            missing.append("resource_group")
        if not self.factory_name:
            missing.append("factory_name")
        return missing


TargetMap = dict[str, dict[str, ADFConfig]]

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "adf_config.json"


def _load_targets() -> TargetMap:
    """Load ADF targets from adf_config.json at project root."""
    if not _CONFIG_PATH.exists():
        return {}
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    targets: TargetMap = {}
    for domain, envs in raw.items():
        targets[domain] = {}
        for env, cfg in envs.items():
            targets[domain][env] = ADFConfig(
                resource_group=cfg["resource_group"],
                factory_name=cfg["resource_name"],
                subscription_id=cfg.get("subscription_id"),
            )
    return targets


ADF_TARGETS: TargetMap = _load_targets()


@dataclass
class ADFAgentContext:
    """
    ADF Agent Runtime Context

    Accessed in tools via ToolRuntime[ADFAgentContext].

    Storage location controlled by the USE_WORKSPACE environment variable:
    - USE_WORKSPACE=true (default): Uses ./workspace/sessions/{timestamp}/
    - USE_WORKSPACE=false: Uses system temp directory /tmp/adf_agent/{timestamp}/
    """
    working_directory: Path = field(default_factory=Path.cwd)
    adf_config: ADFConfig = field(default_factory=ADFConfig)
    _credential: Optional[DefaultAzureCredential] = field(default=None, repr=False)
    _session_id: Optional[str] = field(default=None, repr=False)
    _script_counter: int = field(default=0, repr=False)
    _temp_dir: Optional[Path] = field(default=None, repr=False)
    _cache: dict = field(default_factory=dict, repr=False)
    skill_loader: Optional[SkillLoader] = field(default=None, repr=False)

    @property
    def credential(self) -> DefaultAzureCredential:
        """Lazy-load DefaultAzureCredential"""
        if self._credential is None:
            self._credential = DefaultAzureCredential()
        return self._credential

    @property
    def use_workspace(self) -> bool:
        """Whether to use workspace directory (otherwise use temp files)"""
        return _use_workspace()

    @property
    def workspace(self) -> Path:
        """workspace/ directory for tool output files

        Tools write data to this directory to avoid placing large JSON data in context.
        The Agent can use exec_python to analyze these files.

        If USE_WORKSPACE=false, returns a temp directory.
        """
        if self.use_workspace:
            ws = self.working_directory / "workspace"
        else:
            # Use system temp directory
            ws = Path(tempfile.gettempdir()) / "adf_agent"
        ws.mkdir(parents=True, exist_ok=True)
        return ws

    @property
    def session_id(self) -> str:
        """Get the current session ID (timestamp format)"""
        if self._session_id is None:
            self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self._session_id

    @property
    def session_dir(self) -> Path:
        """Get the current session directory

        Format depends on the USE_WORKSPACE environment variable:
        - true:  ./workspace/sessions/{timestamp}/
        - false: /tmp/adf_agent/sessions/{timestamp}/

        Used to save ADF tool output JSON and exec_python scripts.
        """
        session_path = self.workspace / "sessions" / self.session_id
        session_path.mkdir(parents=True, exist_ok=True)
        return session_path

    def next_script_number(self) -> int:
        """Get the next script number"""
        self._script_counter += 1
        return self._script_counter

    def save_script(self, code: str, output: str, success: bool) -> Path:
        """Save an executed Python script and its output to the session directory

        Args:
            code: Python code
            output: Execution output
            success: Whether execution succeeded

        Returns:
            Path to the saved script file
        """
        script_num = self.next_script_number()
        status = "ok" if success else "failed"

        # Save Python script
        script_file = self.session_dir / f"{script_num:03d}_{status}.py"
        script_file.write_text(code, encoding="utf-8")

        # Save output to a file with the same name but .out extension
        output_file = self.session_dir / f"{script_num:03d}_{status}.out"
        output_file.write_text(output, encoding="utf-8")

        return script_file
