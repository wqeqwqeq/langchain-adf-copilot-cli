"""
TokenTracker - Token 使用量追踪

从 AIMessage/AIMessageChunk 中提取 usage_metadata，
并在多次 LLM 调用（如工具使用场景）中累积统计。

API 返回的 usage_metadata 是每次 API 调用的独立值（非跨 turn 累积）：
- input_tokens: 该次调用的输入 token（因 context 增长，每次自然递增）
- output_tokens: 该次调用的输出 token（独立值）
"""

from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, AIMessageChunk


@dataclass
class TokenUsageInfo:
    """Token 使用量信息"""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    def __add__(self, other: "TokenUsageInfo") -> "TokenUsageInfo":
        """支持 + 运算符累加"""
        return TokenUsageInfo(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cache_creation_input_tokens=self.cache_creation_input_tokens + other.cache_creation_input_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens + other.cache_read_input_tokens,
        )

    def is_empty(self) -> bool:
        """检查是否为空（没有任何 token 统计）"""
        return self.total_tokens == 0


@dataclass
class TokenTracker:
    """
    Token 使用量追踪器

    每次 LLM API 调用返回的 usage_metadata 是该次调用的独立值。
    TokenTracker 直接使用 raw 值作为 per-turn 统计，
    并通过 SUM 所有 turn 得到最终总计。
    """
    # 当前 turn 的 usage（直接来自 API raw 值）
    _current_turn: TokenUsageInfo = field(default_factory=TokenUsageInfo)
    # 所有已 finalize 的 turn 的总计
    _total: TokenUsageInfo = field(default_factory=TokenUsageInfo)
    # 是否已接收到当前 turn 的 usage 数据
    _has_current_usage: bool = False

    def update(self, chunk: AIMessage | AIMessageChunk) -> None:
        """
        从 chunk 提取 token 统计

        API 每次调用返回独立的 usage，直接使用 raw 值。
        流式输出中 usage 通常只出现在最后一个 chunk，
        所以同一 turn 内直接替换（后到的更完整）。
        """
        usage = getattr(chunk, "usage_metadata", None)
        if not usage:
            return

        if isinstance(usage, dict):
            input_tokens = usage.get("input_tokens", 0) or 0
            output_tokens = usage.get("output_tokens", 0) or 0
            cache_creation = usage.get("cache_creation_input_tokens", 0) or 0
            cache_read = usage.get("cache_read_input_tokens", 0) or 0
        else:
            input_tokens = getattr(usage, "input_tokens", 0) or 0
            output_tokens = getattr(usage, "output_tokens", 0) or 0
            cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
            cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0

        if input_tokens > 0 or output_tokens > 0:
            self._current_turn = TokenUsageInfo(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                cache_creation_input_tokens=cache_creation,
                cache_read_input_tokens=cache_read,
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
        获取总计（包括未 finalize 的 turn）

        Returns:
            TokenUsageInfo: 所有 turn 的 SUM
        """
        if self._has_current_usage:
            return self._total + self._current_turn
        return self._total

    def reset(self) -> None:
        """重置所有统计"""
        self._current_turn = TokenUsageInfo()
        self._total = TokenUsageInfo()
        self._has_current_usage = False
