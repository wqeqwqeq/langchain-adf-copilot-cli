# Token Tracking & Cache Display

Token 使用量追踪系统的设计、实现和 Anthropic Prompt Caching 行为说明。

---

## 架构概览

```
Anthropic API Response
    ↓ (usage: input_tokens, cache_creation_input_tokens, cache_read_input_tokens)
LangChain ChatAnthropic (_create_usage_metadata)
    ↓ (usage_metadata dict, input_tokens 已包含 cache)
Agent stream_events() → TokenTracker.update(chunk)
    ↓ (_extract_usage → merge 到 _current_turn)
TokenTracker.finalize_turn()
    ↓ (累加到 _total，返回当前 turn 的 TokenUsageInfo)
StreamEventEmitter.token_usage()
    ↓ (event dict)
CLI StreamState → format_turn_token_usage() / display_token_usage()
    ↓
终端显示
```

### 关键组件

| 文件 | 职责 |
|------|------|
| `stream/token_tracker.py` | 从 chunk 提取 usage，merge 累积，finalize per-turn |
| `stream/emitter.py` | 将 TokenUsageInfo 转为 event dict |
| `agent.py` (`stream_events`) | 编排 token 追踪流程，处理 parallel tool 批次 |
| `cli.py` | 将 event dict 格式化为终端显示 |

---

## Anthropic Prompt Caching 机制

### 什么是 Prompt Caching

Anthropic 的 prompt caching 允许将 system prompt 等固定内容缓存在服务端。后续 API 调用可以直接读取缓存，避免重复处理相同的 tokens。

### 启用方式

在 `prompts.py` 中，system prompt 的 content blocks 带有 `cache_control`：

```python
blocks = [
    {"type": "text", "text": prompt_text, "cache_control": {"type": "ephemeral"}},
]
if skills:
    blocks.append({"type": "text", "text": skills_text, "cache_control": {"type": "ephemeral"}})
```

`"type": "ephemeral"` 表示使用 5 分钟 TTL 的缓存。

### API 返回的 Token 字段

Anthropic API 的 `usage` 对象包含：

| 字段 | 说明 |
|------|------|
| `input_tokens` | **非缓存**的输入 tokens（不包含 cache 部分） |
| `cache_creation_input_tokens` | 本次调用**写入缓存**的 tokens（首次创建） |
| `cache_read_input_tokens` | 本次调用**从缓存读取**的 tokens |
| `output_tokens` | 模型生成的输出 tokens |

**重要**：Anthropic 原始的 `input_tokens` **不包含** cache tokens。

### LangChain 的转换

LangChain 在 `_create_usage_metadata` 中将三者合并：

```python
# langchain_anthropic/chat_models.py
input_tokens = (
    (getattr(anthropic_usage, "input_tokens", 0) or 0)        # 非缓存 input
    + (input_token_details["cache_read"] or 0)                 # 缓存读取
    + (input_token_details["cache_creation"] or 0)             # 缓存写入
)
```

所以 **LangChain 的 `input_tokens` 已包含 cache tokens**。最终的 `usage_metadata` 结构为：

```python
{
    "input_tokens": 3625,       # = 356 (new) + 3269 (cache)
    "output_tokens": 155,
    "total_tokens": 3780,
    "input_token_details": {
        "cache_creation": 3269,  # 写入缓存的 tokens（首次）
        "cache_read": 0,         # 从缓存读取的 tokens
    }
}
```

### Cache 生命周期

```
第一次 API 调用（冷缓存）:
    input_tokens = 356 (new) + 3269 (cache_creation) = 3625
    → system prompt 被写入缓存

后续调用（5 分钟内，热缓存）:
    input_tokens = 1431 (new) + 3269 (cache_read) = 4700
    → system prompt 从缓存读取

5 分钟后（缓存过期）:
    input_tokens = 356 (new) + 3269 (cache_creation) = 3625
    → 缓存重新创建
```

### 计费影响

| Token 类型 | 价格倍率 |
|-----------|---------|
| 普通 input | 1x |
| cache_creation (write) | 1.25x（贵 25%） |
| cache_read (read) | 0.1x（便宜 90%） |

一次 5-turn 会话的 system prompt 成本对比：

- **无缓存**: 3,269 × 5 = 16,345 tokens @ 1x
- **有缓存**: 3,269 × 1.25 + 3,269 × 0.1 × 4 = 4,086 + 1,308 = 5,394 tokens equivalent
- **节省约 67%**

---

## Token 追踪实现

### TokenUsageInfo

```python
@dataclass
class TokenUsageInfo:
    input_tokens: int = 0                    # LangChain 合并后的 input（含 cache）
    output_tokens: int = 0
    total_tokens: int = 0                    # input + output
    cache_creation_input_tokens: int = 0     # 写入缓存的 tokens
    cache_read_input_tokens: int = 0         # 从缓存读取的 tokens
```

支持 `+` 运算符用于跨 turn 累加。

### TokenTracker.update() — Merge 策略

从 `AIMessageChunk.usage_metadata` 提取 token 统计。使用 **merge（取 max）** 而非 replace：

```python
# 取 max 保留各 chunk 的非零值
cur = self._current_turn
self._current_turn = TokenUsageInfo(
    input_tokens=max(cur.input_tokens, input_tokens),
    output_tokens=max(cur.output_tokens, output_tokens),
    total_tokens=...,
    cache_creation_input_tokens=max(cur.cache_creation_input_tokens, cache_creation),
    cache_read_input_tokens=max(cur.cache_read_input_tokens, cache_read),
)
```

**为什么用 merge 而非 replace？**

在流式输出中，usage 可能分散在多个 chunk 中：

| Chunk | 包含 |
|-------|------|
| message_start | input_tokens + cache 信息 |
| message_delta | output_tokens |

如果用 replace，第二个 chunk 会覆盖第一个 chunk 的 cache 数据（因为它只有 output_tokens，cache 字段为 0）。

用 `max()` merge 确保两个 chunk 的数据都被保留。

> **注**: 当前 LangChain (v1.3.1) 实际上在 `message_delta` 中发送完整的聚合 usage，所以每个 turn 只收到一个 usage chunk。但 merge 策略更健壮，能应对 LangChain 未来版本的行为变化。

### Parallel Tool 处理

一次 API 调用可能返回多个 tool_call（并行执行）。Token usage 只在该批次的最后一个 tool_result 后发送：

```python
# agent.py stream_events()
if turn_usage and not turn_usage.is_empty():
    # 第一个 tool result（有 usage 数据）
    pending_turn_usage = turn_usage
    parallel_count = 1
elif pending_turn_usage is not None:
    # 同批次后续 tool（finalize 返回 None）
    parallel_count += 1
```

流程：

```
API Call → tool_call_1, tool_call_2
         → tool_result_1: finalize_turn() 返回 usage, parallel_count=1
         → tool_result_2: finalize_turn() 返回 None, parallel_count=2
新的 AI chunk → _emit_pending() 发送 usage (parallel_count=2)
```

---

## 终端显示格式

### Per-turn 显示

格式：`↳ {new} + {cached} cache {type} / {output} out`

```
↳ 356 + 3,269 cache write / 162 out          # 首次调用，缓存写入
↳ 1,437 + 3,269 cache read / 63 out          # 后续调用，缓存读取
↳ 1,583 + 3,269 cache read / 133 out (2 tools)  # 并行工具
```

其中：
- `356` = new input tokens（非缓存部分）= `input_tokens - cache_read - cache_creation`
- `3,269` = 缓存 tokens（system prompt 大小）
- `cache write` / `cache read` = 缓存操作类型
- `(2 tools)` = 该 API 调用中并行执行的工具数量

### Total 显示

格式取决于缓存状态：

```
# 只有 cache read（热缓存）:
Tokens: 8,525 + 16,345 cache read = 24,870 in / 692 out

# 同时有 read 和 write（冷启动）:
Tokens: 8,497 + 16,345 cache (13,076 read, 3,269 write) = 24,842 in / 679 out

# 无缓存:
Tokens: 5,000 in / 200 out
```

### 如何解读

以这个输出为例：

```
Tokens: 8,537 + 16,345 cache (13,076 read, 3,269 write) = 24,882 in / 727 out
```

| 部分 | 值 | 含义 |
|------|-----|------|
| `8,537` | new input | 对话内容、tool results 等非缓存 tokens |
| `16,345` | cached | system prompt 的缓存 tokens (5 次调用) |
| `13,076 read` | cache read | 4 次调用 × 3,269 = 13,076（从缓存读取） |
| `3,269 write` | cache write | 1 次调用 × 3,269（首次写入缓存） |
| `24,882 in` | total input | 8,537 + 16,345 = 24,882 |
| `727 out` | output | 模型生成的所有输出 tokens |

展开每个 turn：

| Turn | 工具 | New | Cache | 类型 | Output |
|------|------|-----|-------|------|--------|
| 1 | load_skill | 356 | 3,269 | write | 162 |
| 2 | list | 1,437 | 3,269 | read | 63 |
| 3 | get ×2 | 1,583 | 3,269 | read | 133 |
| 4 | test ×2 | 2,437 | 3,269 | read | 156 |
| 5 | 最终回复 | 2,724 | 3,269 | read | 213 |
| **Total** | | **8,537** | **16,345** | | **727** |

---

## 常见问题

### Q: 为什么每次运行都显示 "cache write"？

Anthropic 的 prompt cache TTL 是 **5 分钟**。如果两次运行间隔超过 5 分钟，缓存过期，第一次 API 调用需要重新创建缓存。

立即再次运行会看到全部变成 `cache read`（无 write）。

### Q: cache write 是不是浪费？

不是。cache write 比普通 input 贵 25%，但后续的 cache read 便宜 90%。只要在 5 分钟内有 2 次以上调用（单次会话通常有 3-6 次 API 调用），缓存就已经划算了。

### Q: 为什么 new input 每个 turn 都在增长？

因为每次 API 调用都要发送完整的对话历史（之前的 messages + tool calls + tool results）。随着对话进行，历史越来越长，所以 new input 从 356 → 1,437 → 1,583 → 2,437 递增。

### Q: input_tokens 和 cache tokens 是什么关系？

Cache tokens 是 input_tokens 的**子集**，不是额外的：

```
input_tokens (LangChain) = new_input + cache_read + cache_creation
         3,625           =    356    +   3,269   +      0
```

### Q: 为什么 per-turn 没有显示最终回复的 token usage？

最终回复（Turn 5）没有 tool_result 触发显示。它的 usage 包含在 total 汇总中，但不会单独显示为 `↳` 行。

---

## 代码版本

- `langchain-anthropic`: 1.3.1
- `langchain-core`: 1.2.7
- `anthropic`: 0.76.0

---

*文档创建时间: 2025-01*
