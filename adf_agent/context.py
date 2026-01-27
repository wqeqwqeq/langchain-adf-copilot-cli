"""
ADF Agent 上下文

提供 Agent 运行时所需的配置和状态管理。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from azure.identity import DefaultAzureCredential


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
    """
    working_directory: Path = field(default_factory=Path.cwd)
    adf_config: ADFConfig = field(default_factory=ADFConfig)
    _credential: Optional[DefaultAzureCredential] = field(default=None, repr=False)

    @property
    def credential(self) -> DefaultAzureCredential:
        """Lazy-load DefaultAzureCredential"""
        if self._credential is None:
            self._credential = DefaultAzureCredential()
        return self._credential

    @property
    def workspace(self) -> Path:
        """workspace/ directory for tool output files

        工具将数据写入此目录，避免将大量 JSON 数据放入上下文。
        Agent 可以使用 exec_python 分析这些文件。
        """
        ws = self.working_directory / "workspace"
        ws.mkdir(parents=True, exist_ok=True)
        return ws
