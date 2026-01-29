"""
TokenTracker - Token 使用量追踪

从 AIMessage/AIMessageChunk 中提取 usage_metadata，
并在多次 LLM 调用（如工具使用场景）中累积统计。

API 返回的 usage_metadata 是每次 API 调用的独立值（非跨 turn 累积）：
- input_tokens: 该次调用的总输入 token（已包含 cache tokens）
- output_tokens: 该次调用的输出 token（独立值）
- input_token_details.cache_creation: 首次写入缓存的 token 数（cache init）
- input_token_details.cache_read: 从缓存命中的 token 数（cached）

LangChain 的 input_tokens = raw_input + cache_creation + cache_read，
即 cache tokens 是 input_tokens 的子集，不是额外的。
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

        使用 merge 策略：取各字段的 max 值，确保 usage 分散在多个
        chunk 时（如 input 和 cache 在前、output 在后），不会丢失数据。

        LangChain input_tokens 已包含 cache tokens：
        input_tokens = raw_input + cache_read + cache_creation
        """
        usage = getattr(chunk, "usage_metadata", None)
        if not usage:
            return

        input_tokens, output_tokens, cache_creation, cache_read = \
            self._extract_usage(usage)

        if input_tokens > 0 or output_tokens > 0:
            # Merge: 取 max 保留各 chunk 的非零值
            cur = self._current_turn
            merged_input = max(cur.input_tokens, input_tokens)
            merged_output = max(cur.output_tokens, output_tokens)
            self._current_turn = TokenUsageInfo(
                input_tokens=merged_input,
                output_tokens=merged_output,
                total_tokens=merged_input + merged_output,
                cache_creation_input_tokens=max(cur.cache_creation_input_tokens, cache_creation),
                cache_read_input_tokens=max(cur.cache_read_input_tokens, cache_read),
            )
            self._has_current_usage = True

    @staticmethod
    def _extract_usage(usage) -> tuple[int, int, int, int]:
        """从 usage_metadata 提取 (input, output, cache_creation, cache_read)"""
        if isinstance(usage, dict):
            input_tokens = usage.get("input_tokens", 0) or 0
            output_tokens = usage.get("output_tokens", 0) or 0
            details = usage.get("input_token_details", {}) or {}
            cache_creation = details.get("cache_creation", 0) or 0
            cache_read = details.get("cache_read", 0) or 0
        else:
            input_tokens = getattr(usage, "input_tokens", 0) or 0
            output_tokens = getattr(usage, "output_tokens", 0) or 0
            details = getattr(usage, "input_token_details", None)
            if details and isinstance(details, dict):
                cache_creation = details.get("cache_creation", 0) or 0
                cache_read = details.get("cache_read", 0) or 0
            else:
                cache_creation = getattr(details, "cache_creation", 0) or 0 if details else 0
                cache_read = getattr(details, "cache_read", 0) or 0 if details else 0
        return input_tokens, output_tokens, cache_creation, cache_read

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
