"""
ToolResultFormatter - 工具结果格式化器

基于内容特征智能格式化工具输出。
"""

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, List

from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from rich.markdown import Markdown

from .utils import SUCCESS_PREFIX, FAILURE_PREFIX, is_success as _is_success, truncate


class ContentType(Enum):
    """内容类型"""
    SUCCESS = "success"
    ERROR = "error"
    JSON = "json"
    MARKDOWN = "markdown"
    TEXT = "text"


@dataclass
class FormattedResult:
    """格式化结果"""
    content_type: ContentType
    elements: List[Any]  # Rich 可渲染元素
    success: bool = True  # 是否成功


class ToolResultFormatter:
    """工具结果格式化器

    使用示例：
        formatter = ToolResultFormatter()
        result = formatter.format("bash", output, max_length=800)
        for elem in result.elements:
            console.print(elem)
    """

    def detect_type(self, content: str) -> ContentType:
        """检测内容类型"""
        content = content.strip()

        # 1. 基于状态标记判断（最高优先级）
        if content.startswith(SUCCESS_PREFIX):
            # 检查是否有 JSON 输出
            body = self._extract_body(content)
            if self._is_json(body):
                return ContentType.JSON
            return ContentType.SUCCESS

        if content.startswith(FAILURE_PREFIX):
            return ContentType.ERROR

        # 2. JSON 检测
        if self._is_json(content):
            return ContentType.JSON

        # 3. 真正的错误检测
        if self._is_error(content):
            return ContentType.ERROR

        # 4. Markdown 检测
        if self._is_markdown(content):
            return ContentType.MARKDOWN

        return ContentType.TEXT

    def is_success(self, content: str) -> bool:
        """判断内容是否表示成功执行"""
        return _is_success(content)

    def format(self, name: str, content: str, max_length: int = 800) -> FormattedResult:
        """格式化工具结果"""
        content_type = self.detect_type(content)
        success = self.is_success(content)

        # 分派到具体格式化方法
        formatter_map = {
            ContentType.SUCCESS: self._format_success,
            ContentType.ERROR: self._format_error,
            ContentType.JSON: self._format_json,
            ContentType.MARKDOWN: self._format_markdown,
            ContentType.TEXT: self._format_text,
        }

        formatter = formatter_map.get(content_type, self._format_text)
        elements = formatter(name, content, max_length)

        return FormattedResult(content_type=content_type, elements=elements, success=success)

    # === 私有方法：类型检测 ===

    def _extract_body(self, content: str) -> str:
        """提取状态标记后的内容体"""
        lines = content.split("\n", 2)
        return lines[2].strip() if len(lines) > 2 else ""

    def _is_json(self, content: str) -> bool:
        """检查是否是 JSON"""
        content = content.strip()
        if not content:
            return False
        if (content.startswith('{') and content.endswith('}')) or \
           (content.startswith('[') and content.endswith(']')):
            try:
                json.loads(content)
                return True
            except (json.JSONDecodeError, ValueError):
                pass
        return False

    def _is_error(self, content: str) -> bool:
        """检查是否是错误内容"""
        error_patterns = [
            'Traceback (most recent call last)',
            'Exception:',
            'Error:',
        ]
        return any(pattern in content for pattern in error_patterns)

    def _is_markdown(self, content: str) -> bool:
        """检查是否是 Markdown"""
        md_patterns = ['```', '**', '##', '- **']
        return content.startswith('#') or any(p in content for p in md_patterns)

    # === 私有方法：格式化 ===

    def _format_success(self, name: str, content: str, max_length: int) -> List[Any]:
        """格式化成功输出"""
        display = self._truncate(content, max_length)
        return [Panel(
            Text(display, style="green"),
            title=f"{name}",
            border_style="green",
        )]

    def _format_error(self, name: str, content: str, max_length: int) -> List[Any]:
        """格式化错误输出"""
        display = self._truncate(content, max_length)
        return [Panel(
            Text(display, style="red"),
            title=f"{name}",
            border_style="red",
        )]

    def _format_json(self, name: str, content: str, max_length: int) -> List[Any]:
        """格式化 JSON 输出"""
        # 提取 JSON 内容
        json_content = content
        if content.startswith(SUCCESS_PREFIX):
            json_content = self._extract_body(content)

        try:
            data = json.loads(json_content)
            formatted = json.dumps(data, indent=2, ensure_ascii=False)
            formatted = self._truncate(formatted, max_length)
            return [
                Text(f"{name}", style="cyan bold"),
                Syntax(formatted, "json", theme="monokai", line_numbers=False),
            ]
        except (json.JSONDecodeError, ValueError):
            return self._format_text(name, content, max_length)

    def _format_markdown(self, name: str, content: str, max_length: int) -> List[Any]:
        """格式化 Markdown 输出"""
        display = self._truncate(content, max_length)
        return [Panel(
            Markdown(display),
            title=f"{name}",
            border_style="cyan dim",
        )]

    def _format_text(self, name: str, content: str, max_length: int) -> List[Any]:
        """格式化普通文本输出"""
        display = self._truncate(content, max_length)
        return [
            Text(f"{name}:", style="cyan bold"),
            Text(f"   {display}", style="dim"),
        ]

    def _truncate(self, content: str, max_length: int) -> str:
        """截断内容"""
        return truncate(content, max_length)
