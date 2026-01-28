"""
TokenTracker - Token 使用量追踪

从 AIMessage/AIMessageChunk 中提取 usage_metadata，
并在多次 LLM 调用（如工具使用场景）中累积统计。
"""

from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, AIMessageChunk


@dataclass
class TokenUsageInfo:
    """Token 使用量信息"""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: "TokenUsageInfo") -> "TokenUsageInfo":
        """支持 + 运算符累加"""
        return TokenUsageInfo(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )

    def is_empty(self) -> bool:
        """检查是否为空（没有任何 token 统计）"""
        return self.total_tokens == 0


@dataclass
class TokenTracker:
    """
    Token 使用量追踪器

    用于在流式输出中累积 token 统计：
    - 从每个 AIMessageChunk 的 usage_metadata 提取数据
    - 支持多次 LLM 调用（工具使用场景）的累积
    - 提供最终的汇总统计
    """
    # 当前 turn 的累积
    _current_turn: TokenUsageInfo = field(default_factory=TokenUsageInfo)
    # 所有 turn 的总计
    _total: TokenUsageInfo = field(default_factory=TokenUsageInfo)
    # 是否已接收到当前 turn 的 usage 数据
    _has_current_usage: bool = False

    def update(self, chunk: AIMessage | AIMessageChunk) -> None:
        """
        从 chunk 更新 token 统计

        Args:
            chunk: AIMessage 或 AIMessageChunk，可能包含 usage_metadata
        """
        usage = getattr(chunk, "usage_metadata", None)
        if not usage:
            return

        input_tokens = 0
        output_tokens = 0

        if isinstance(usage, dict):
            input_tokens = usage.get("input_tokens", 0) or 0
            output_tokens = usage.get("output_tokens", 0) or 0
        else:
            input_tokens = getattr(usage, "input_tokens", 0) or 0
            output_tokens = getattr(usage, "output_tokens", 0) or 0

        # 只在有实际数据时更新
        if input_tokens > 0 or output_tokens > 0:
            # 流式输出中，usage 通常只在最后一个 chunk 出现
            # 所以我们直接替换当前 turn 的统计（而不是累加）
            self._current_turn = TokenUsageInfo(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            )
            self._has_current_usage = True

    def finalize_turn(self) -> TokenUsageInfo | None:
        """
        结束当前 turn，返回该 turn 的使用量并累加到总计

        在工具结果返回后调用，表示一轮 LLM 调用结束。

        Returns:
            该 turn 的 TokenUsageInfo，如果没有使用量则返回 None
        """
        if self._has_current_usage:
            current = self._current_turn
            self._total = self._total + current
            self._current_turn = TokenUsageInfo()
            self._has_current_usage = False
            return current
        return None

    def get_usage(self) -> TokenUsageInfo:
        """
        获取当前统计（包括未结束的 turn）

        Returns:
            TokenUsageInfo 包含所有累积的 token 统计
        """
        if self._has_current_usage:
            return self._total + self._current_turn
        return self._total

    def reset(self) -> None:
        """重置所有统计"""
        self._current_turn = TokenUsageInfo()
        self._total = TokenUsageInfo()
        self._has_current_usage = False
