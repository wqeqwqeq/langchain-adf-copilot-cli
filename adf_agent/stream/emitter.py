"""
StreamEventEmitter - 统一事件格式

所有事件都包含 type 和相关数据。
"""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class StreamEvent:
    """统一的流式事件"""
    type: str
    data: Dict[str, Any]


class StreamEventEmitter:
    """流式事件发射器"""

    @staticmethod
    def thinking(content: str, thinking_id: int = 0) -> StreamEvent:
        """思考内容事件"""
        return StreamEvent("thinking", {"type": "thinking", "content": content, "id": thinking_id})

    @staticmethod
    def text(content: str) -> StreamEvent:
        """文本内容事件"""
        return StreamEvent("text", {"type": "text", "content": content})

    @staticmethod
    def tool_call(name: str, args: Dict[str, Any], tool_id: str = "") -> StreamEvent:
        """工具调用事件"""
        return StreamEvent("tool_call", {"type": "tool_call", "name": name, "args": args, "id": tool_id})

    @staticmethod
    def tool_result(name: str, content: str, success: bool = True) -> StreamEvent:
        """工具结果事件"""
        return StreamEvent("tool_result", {
            "type": "tool_result",
            "name": name,
            "content": content,
            "success": success,
        })

    @staticmethod
    def done(response: str = "") -> StreamEvent:
        """完成事件"""
        return StreamEvent("done", {"type": "done", "response": response})

    @staticmethod
    def error(message: str) -> StreamEvent:
        """错误事件"""
        return StreamEvent("error", {"type": "error", "message": message})

    @staticmethod
    def token_usage(
        input_tokens: int,
        output_tokens: int,
        total_tokens: int = 0,
        cache_creation_input_tokens: int = 0,
        cache_read_input_tokens: int = 0,
        is_total: bool = False,
        parallel_count: int = 1,
    ) -> StreamEvent:
        """Token 使用量事件

        Args:
            is_total: True 表示这是所有 turn 的汇总，False 表示单次 API 调用的用量
            parallel_count: 该 API 调用中并行执行的 tool 数量（>1 表示 parallel tool use）
        """
        return StreamEvent("token_usage", {
            "type": "token_usage",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens or (input_tokens + output_tokens),
            "cache_creation_input_tokens": cache_creation_input_tokens,
            "cache_read_input_tokens": cache_read_input_tokens,
            "is_total": is_total,
            "parallel_count": parallel_count,
        })
