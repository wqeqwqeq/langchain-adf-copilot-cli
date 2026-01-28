"""
ADF Agent 上下文

提供 Agent 运行时所需的配置和状态管理。
"""

import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from azure.identity import DefaultAzureCredential


def _use_workspace() -> bool:
    """检查是否使用 workspace 目录（否则用 temp file）"""
    val = os.getenv("USE_WORKSPACE", "true").lower()
    return val in ("true", "1", "yes")


@dataclass
class ADFConfig:
    """
    ADF 配置 - 可能为空，需要验证

    从环境变量加载，如果未配置则需要向用户询问。
    """
    resource_group: Optional[str] = None
    factory_name: Optional[str] = None
    subscription_id: Optional[str] = None  # 可选，SDK 会自动获取

    def is_configured(self) -> bool:
        """检查必要配置是否完整"""
        return bool(self.resource_group and self.factory_name)

    def missing_fields(self) -> list[str]:
        """返回缺失的字段名"""
        missing = []
        if not self.resource_group:
            missing.append("resource_group")
        if not self.factory_name:
            missing.append("factory_name")
        return missing


@dataclass
class ADFAgentContext:
    """
    ADF Agent 运行时上下文

    通过 ToolRuntime[ADFAgentContext] 在 tool 中访问。

    存储位置由环境变量 USE_WORKSPACE 控制：
    - USE_WORKSPACE=true (默认): 使用 ./workspace/sessions/{timestamp}/
    - USE_WORKSPACE=false: 使用系统临时目录 /tmp/adf_agent/{timestamp}/
    """
    working_directory: Path = field(default_factory=Path.cwd)
    adf_config: ADFConfig = field(default_factory=ADFConfig)
    _credential: Optional[DefaultAzureCredential] = field(default=None, repr=False)
    _session_id: Optional[str] = field(default=None, repr=False)
    _script_counter: int = field(default=0, repr=False)
    _temp_dir: Optional[Path] = field(default=None, repr=False)

    @property
    def credential(self) -> DefaultAzureCredential:
        """Lazy-load DefaultAzureCredential"""
        if self._credential is None:
            self._credential = DefaultAzureCredential()
        return self._credential

    @property
    def use_workspace(self) -> bool:
        """是否使用 workspace 目录（否则用 temp file）"""
        return _use_workspace()

    @property
    def workspace(self) -> Path:
        """workspace/ directory for tool output files

        工具将数据写入此目录，避免将大量 JSON 数据放入上下文。
        Agent 可以使用 exec_python 分析这些文件。

        如果 USE_WORKSPACE=false，返回临时目录。
        """
        if self.use_workspace:
            ws = self.working_directory / "workspace"
        else:
            # 使用系统临时目录
            ws = Path(tempfile.gettempdir()) / "adf_agent"
        ws.mkdir(parents=True, exist_ok=True)
        return ws

    @property
    def session_id(self) -> str:
        """获取当前 session ID (timestamp 格式)"""
        if self._session_id is None:
            self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self._session_id

    @property
    def session_dir(self) -> Path:
        """获取当前 session 的目录

        格式取决于 USE_WORKSPACE 环境变量：
        - true:  ./workspace/sessions/{timestamp}/
        - false: /tmp/adf_agent/sessions/{timestamp}/

        用于保存 ADF 工具输出的 JSON 和 exec_python 执行的脚本。
        """
        session_path = self.workspace / "sessions" / self.session_id
        session_path.mkdir(parents=True, exist_ok=True)
        return session_path

    def next_script_number(self) -> int:
        """获取下一个脚本编号"""
        self._script_counter += 1
        return self._script_counter

    def save_script(self, code: str, output: str, success: bool) -> Path:
        """保存执行的 Python 脚本和输出到 session 目录

        Args:
            code: Python 代码
            output: 执行输出
            success: 是否执行成功

        Returns:
            保存的脚本文件路径
        """
        script_num = self.next_script_number()
        status = "ok" if success else "failed"

        # 保存 Python 脚本
        script_file = self.session_dir / f"{script_num:03d}_{status}.py"
        script_file.write_text(code, encoding="utf-8")

        # 保存输出到同名的 .out 文件
        output_file = self.session_dir / f"{script_num:03d}_{status}.out"
        output_file.write_text(output, encoding="utf-8")

        return script_file
