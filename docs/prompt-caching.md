# Progressive Prompt Caching

How prompt caching works in this project, why skill content was not being cached, and how progressive (incremental) caching fixes it.

Reference: [Anthropic Prompt Caching Documentation](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)

---

## Cache Hierarchy

Anthropic's cache operates as a **prefix match** across three layers, in this order:

```
tools → system → messages
```

Each layer builds on the previous. A cache hit means the entire prefix up to a breakpoint matches a previous request exactly.

### Cache Breakpoints

- Up to **4 breakpoints** per request (via `cache_control: {"type": "ephemeral"}`)
- Each breakpoint marks the end of a cacheable prefix segment
- Breakpoints themselves are free — you only pay for cache writes (1.25x) and reads (0.1x)

### Minimum Cacheable Length

| Model | Minimum Tokens |
|-------|---------------|
| Claude Sonnet 4.5 / 4 | 1,024 |
| Claude Opus 4.5 | 4,096 |
| Claude Haiku 4.5 | 4,096 |

Below the threshold, `cache_control` is silently ignored — no error, no extra cost.

---

## What Was Cached Before (System-Only)

Previously, only the system prompt had `cache_control`:

```python
# prompts.py
blocks = [
    {"type": "text", "text": prompt_text, "cache_control": {"type": "ephemeral"}},  # breakpoint 1
]
if skills:
    skills_text = build_skills_section(skills)
    blocks.append({"type": "text", "text": skills_text, "cache_control": {"type": "ephemeral"}})  # breakpoint 2
```

This cached ~1,700 tokens (system prompt) + ~100 tokens (skill catalog summary). The full skill instructions loaded via `load_skill` were **not cached**.

### Why Skill Content Was Not Cached

`load_skill` returns a plain string:

```python
# skill_tools.py
return f"""# Skill: {skill_name}
{skill_content.instructions}
## Skill Directory: `{skill_path}`"""
```

This string flows through:

```
load_skill() returns str
    → LangChain wraps as ToolMessage(content="...")
        → ChatAnthropic converts to API tool_result block
            → Sent to Anthropic API
```

The resulting API payload:

```json
{
  "role": "user",
  "content": [{
    "type": "tool_result",
    "tool_use_id": "toolu_01HVb...",
    "content": "# Skill: find-pipelines-by-service\n\n..."
  }]
}
```

No `cache_control` on the `tool_result` block → not cached.

### The Missing Link: LangChain's Built-in Mechanism

`ChatAnthropic._get_request_payload()` has this logic:

```python
cache_control = kwargs.pop("cache_control", None)
if cache_control and formatted_messages:
    # Automatically adds cache_control to the last content block of the last message
    for formatted_message in reversed(formatted_messages):
        for block in reversed(content):
            block["cache_control"] = cache_control
            break
        break
```

But LangGraph's `create_agent` calls `model.invoke(messages)` **without passing `cache_control` as a kwarg**, so `kwargs.pop("cache_control", None)` always returns `None`.

---

## Progressive Caching: How It Works

Progressive (incremental) caching places a `cache_control` breakpoint on the **last message block of every API call**. Each turn's breakpoint advances forward, so all previous content becomes cached.

### Single-Turn Example (Tool Loop)

```
API Call 1 (agent calls load_skill):
  tools:   [tool definitions]
  system:  [prompt] [skills_summary]              ← cache_control (breakpoint 1-2)
  messages: [user_msg, ai_tool_use,
             tool_result_skill]                    ← cache_control auto-added (breakpoint 3)

API Call 2 (agent calls adf_pipeline_list):
  tools:   [tool definitions]
  system:  [prompt] [skills_summary]              ← cache_read ✓
  messages: [user_msg, ai_tool_use,
             tool_result_skill,                    ← cache_read ✓ (matched prefix)
             ai_tool_use_2,
             tool_result_pipelines]                ← cache_control auto-added (breakpoint 3)

API Call 3 (agent calls exec_python):
  tools:   [tool definitions]
  system:  [prompt] [skills_summary]              ← cache_read ✓
  messages: [user_msg, ...,
             tool_result_pipelines,                ← cache_read ✓ (matched prefix)
             ai_tool_use_3,
             tool_result_exec]                     ← cache_control auto-added (breakpoint 3)
```

Each call only pays `cache_creation` for the **new** messages since the last breakpoint. Everything before is `cache_read` at 0.1x.

### Cost Breakdown (Single Turn, 5 API Calls)

| Call | tools+system | Old messages | New messages | New input |
|------|-------------|-------------|-------------|-----------|
| 1 | cache_creation | — | cache_creation | input_tokens |
| 2 | cache_read | cache_read | cache_creation | input_tokens |
| 3 | cache_read | cache_read | cache_creation | input_tokens |
| 4 | cache_read | cache_read | cache_creation | input_tokens |
| 5 | cache_read | cache_read | cache_creation | input_tokens |

---

## Multi-Turn Behavior: Extended Thinking Cache Invalidation

### Why Messages Cache Breaks Between User Turns

When Extended Thinking is enabled, the conversation history contains thinking blocks:

```
messages in Call 3 (end of Turn 1):
  user: "Find Snowflake pipelines"
  assistant: [thinking_1] + [tool_use: load_skill]
  user: [tool_result: skill content]
  assistant: [thinking_2] + [tool_use: pipeline_list]
  user: [tool_result: pipelines]
  assistant: [thinking_3] + [text: "Here are the results..."]
```

When the user sends a **new message** (non-tool-result), Anthropic strips all previous thinking blocks:

```
messages in Call 4 (start of Turn 2):
  user: "Find Snowflake pipelines"
  assistant: [tool_use: load_skill]              ← thinking_1 stripped
  user: [tool_result: skill content]
  assistant: [tool_use: pipeline_list]           ← thinking_2 stripped
  user: [tool_result: pipelines]
  assistant: [text: "Here are the results..."]   ← thinking_3 stripped
  user: "What about AzureBlobStorage?"           ← new user message (triggers strip)
```

The message sequence changed (thinking blocks removed) → prefix no longer matches → **messages layer cache miss**.

### What Gets Invalidated vs What Survives

| Layer | Turn 2 Status | Why |
|-------|--------------|-----|
| tools | cache_read ✓ | Unchanged |
| system | cache_read ✓ | Unchanged |
| messages | cache_creation ✘ | Thinking blocks stripped → prefix mismatch |

**Only the messages layer needs to be re-cached.** tools and system remain cached.

### Full Multi-Turn Timeline

```
Turn 1, Call 1:
  tools:    cache_creation (1.25x)
  system:   cache_creation (1.25x)
  messages: cache_creation (1.25x)

Turn 1, Call 2-5 (tool loop):
  tools:    cache_read (0.1x) ✓
  system:   cache_read (0.1x) ✓
  messages: cache_read (0.1x) + cache_creation for new content (1.25x)

Turn 2, Call 1 (user sends new message):
  tools:    cache_read (0.1x) ✓
  system:   cache_read (0.1x) ✓
  messages: cache_creation (1.25x) ← full re-cache (thinking stripped)

Turn 2, Call 2-5 (tool loop):
  tools:    cache_read (0.1x) ✓
  system:   cache_read (0.1x) ✓
  messages: cache_read (0.1x) + cache_creation for new content (1.25x)

Turn 3, Call 1: same pattern as Turn 2 Call 1
  ...
```

Each turn's first API call re-caches the messages layer (because thinking blocks were stripped). Within a turn's tool loop, progressive caching works normally.

### Cache Invalidation Summary

| Event | tools | system | messages |
|-------|-------|--------|----------|
| Same turn, tool loop | ✓ read | ✓ read | ✓ read + creation for new |
| New user message (thinking stripped) | ✓ read | ✓ read | ✘ re-creation |
| Tool definitions changed | ✘ | ✘ | ✘ |
| System prompt changed | ✓ | ✘ | ✘ |
| Thinking budget changed | ✓ | ✓ | ✘ |

---

## Implementation

### The Fix: Override `_get_request_payload`

Both provider classes override `_get_request_payload` to inject `cache_control` into every API call:

```python
def _get_request_payload(self, input_, *, stop=None, **kwargs):
    kwargs.setdefault("cache_control", {"type": "ephemeral"})
    return super()._get_request_payload(input_, stop=stop, **kwargs)
```

`setdefault` ensures it doesn't override an explicitly passed value. The parent class then automatically applies it to the last content block of the last message.

### Files Modified

| File | Change |
|------|--------|
| `adf_agent/azure_claude.py` | Added `_get_request_payload` override to `ChatAzureFoundryClaude` |
| `adf_agent/agent.py` | Added `CachedChatAnthropic` subclass with same override for direct Anthropic provider |

### Breakpoint Usage (3 of 4 max)

| # | Location | Content |
|---|----------|---------|
| 1 | system block 1 | Main system prompt (~1,700 tokens) |
| 2 | system block 2 | Skill catalog summary (~100 tokens) |
| 3 | last message block (auto) | Entire conversation history up to current point |

One breakpoint remains available for future use (e.g., caching tool definitions separately).

---

## Verification

After implementing, run a multi-turn conversation and check the token usage output:

### Expected: Turn 1, Call 1 (Cold Cache)

```
↳ 50 + 3,800 cache write / 200 out
```

All content (system + messages) written to cache.

### Expected: Turn 1, Call 2+ (Tool Loop, Warm Cache)

```
↳ 200 + 5,500 cache (3,800 read, 1,700 write) / 150 out
```

Previous content read from cache, only new tool results written.

### Expected: Turn 2, Call 1 (New User Message)

```
↳ 50 + 6,000 cache (1,800 read, 4,200 write) / 180 out
```

- `1,800 read` = tools + system (still cached)
- `4,200 write` = messages layer re-cached (thinking stripped)

### Red Flags

- `cache_read_input_tokens: 0` on Turn 1, Call 2+ → fix not working
- `cache_creation` equals total input on every call → nothing is being cached

---

*Document created: 2025-01*
*Reference: [Anthropic Prompt Caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)*
