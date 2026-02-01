# ADF Agent

A **highly scalable** Azure Data Factory agent that looks up and cross-references information across **Linked Services, Datasets, Pipelines, and Integration Runtimes**. Built on LangChain's `create_agent` API with Claude, it handles real-world ADF instances with hundreds of pipelines while keeping token usage minimal.

**Model hosting**: support claude model hosted in **Anthorpic** or **Microsoft Foundry**

**Why a CLI?** This is designed to facilitate local development and learning. A CLI lets you see exactly what's happening — the raw JSON returned by Azure, the Python code the LLM generates, the tool call sequence, token usage per turn — all in your terminal. Combined with local MLflow tracking, you get full transparency into every agent decision without deploying anything.

The core idea: **atomic tools + composable skills** save tokens while enabling arbitrarily complex operations. Five design principles make it scale:

1. **Atomic Tools** — tools only do list and get, nothing more
2. **Skills as Business Logic** — multi-step workflows live in Markdown skills, not code
3. **Reasoning-Action Loop** — think → plan → call tools → observe → replan → repeat until done
4. **Progressive Cache Strategy** — ~40% average token savings per LLM call
5. **`exec_python`** — inspired by Claude Code: explore files on disk, generate code to process data, avoid context window explosion

```
$ uv run adf_agent "Which pipelines in sales dev use Snowflake?"

  💭 Thinking...
  🔧 resolve_adf_target("sales", "dev")         → OK
  🔧 adf_pipeline_list()                        → 242 pipelines saved
  🔧 adf_linked_service_list()                  → 18 linked services
  🔧 adf_dataset_list()                         → 65 datasets saved
  🔧 exec_python(cross_reference_script)         → 20 pipelines matched

  Found 20 pipelines using Snowflake linked services:
  | Pipeline         | Linked Service      |
  |------------------|---------------------|
  | daily_load       | snowflake_prod_ls   |
  | hourly_sync      | snowflake_v2_ls     |
  ...
```

## Design Principles

### 1. Atomic Tools — Only List and Get

Tools are intentionally minimal. Each tool does exactly one thing: **list** all resources of a type, or **get** a single resource by name. No business logic, no cross-referencing, no filtering inside tools. This makes them reusable, composable, and cheap to call.

| Tool | Operation |
|------|-----------|
| `adf_pipeline_list` | List all pipelines; save each as JSON to session dir |
| `adf_pipeline_get` | Get one pipeline definition |
| `adf_linked_service_list` | List all linked services (name + type) |
| `adf_linked_service_get` | Get one linked service definition |
| `adf_linked_service_test` | Test a linked service connection |
| `adf_dataset_list` | List all datasets with linked service mappings |
| `adf_integration_runtime_list` | List all Integration Runtimes |
| `adf_integration_runtime_get` | Get IR status |
| `adf_integration_runtime_enable` | Enable interactive authoring on a Managed IR |
| `resolve_adf_target` | Set the active ADF instance (domain + environment) |

The agent also has file-system tools inspired by Claude Code — `read_file`, `write_file`, `glob`, `grep`, `list_dir` — so it can explore JSON files saved by ADF tools, understand their schema, and then write targeted analysis code (see [Principle 5](#5-exec_python--avoid-context-window-explosion)).

### 2. Business Logic Defined in Skills

Complex multi-step workflows don't live in tools — they live in **Skills**. Skills are Markdown files in `.claude/skills/` with step-by-step instructions. At startup, only skill names and one-line descriptions are injected into the system prompt. When a user request matches a skill, the agent calls `load_skill()` to load the full instructions on-demand.

This two-tier design means:
- **Small system prompt** — skill catalog is just a summary table, saving tokens
- **On-demand detail** — full instructions loaded only when needed
- **Easy to extend** — drop a new `.md` file to add a new capability, no code changes

Current skills:

| Skill | Description |
|-------|-------------|
| `find-pipelines-by-service` | Cross-reference pipelines, datasets, and linked services to find all pipelines using a given service type (e.g. Snowflake). 7-step workflow: resolve target → list resources in parallel → identify matching services → read sample JSON to learn schema → write and run cross-reference script via `exec_python` → debug/retry → present results. |
| `test-linked-service` | Test linked service connections with automatic IR detection and managed IR activation. Handles single service, by type, or all services. |

### 3. Reasoning-Action Loop

The agent follows a **ReAct (Reasoning + Acting)** loop powered by LangChain's `create_agent`:

1. **Think** — Claude reads the question and reasons about what to do (Extended Thinking)
2. **Plan** — Decides which tools to call and in what order
3. **Act** — Calls one or more tools (supports parallel tool calls)
4. **Observe** — Reads tool outputs, evaluates progress
5. **Replan** — If the job isn't done, loops back to step 1 with updated context

The agent keeps iterating until the question is fully answered. It is not a one-shot tool call — it is a loop that can recover from errors, adjust strategy based on intermediate results, and chain together multiple steps autonomously.

### 4. Progressive Cache Strategy — ~40% Average Savings per LLM Call

In a ReAct agent loop, each API call re-sends the full conversation history. Without caching, every call pays full price for all accumulated context — tool results, skill instructions, previous reasoning.

Both provider classes (`CachedChatAnthropic` and `ChatAzureFoundryClaude`) override `_get_request_payload` to inject `cache_control` breakpoints automatically:

```python
def _get_request_payload(self, input_, *, stop=None, **kwargs):
    kwargs.setdefault("cache_control", {"type": "ephemeral"})
    return super()._get_request_payload(input_, stop=stop, **kwargs)
```

This places a cache breakpoint on the **last message block** of every API call. Each turn's breakpoint advances forward, so all previous content becomes cached prefix at 0.1× cost:

```
Call 1:  [system ✍] [user_msg, tool_result_1 ✍]                    ← all cache_creation
Call 2:  [system ✓] [user_msg, tool_result_1 ✓] [tool_result_2 ✍]  ← prefix cached, only new part created
Call 3:  [system ✓] [..., tool_result_2 ✓] [tool_result_3 ✍]       ← prefix cached, only new part created
```

Result: ~40% average token cost savings per LLM call across multi-step workflows. See [Progressive Prompt Caching](#progressive-prompt-caching) for full technical details including cache breakpoint layout and Extended Thinking invalidation behavior.

### 5. `exec_python` — Avoid Context Window Explosion

This is the most critical design choice for scalability, inspired by how Claude Code works.

**The problem:** A real-world ADF instance easily has 200–500+ pipelines. A query like *"which pipelines use Snowflake linked services?"* requires cross-referencing every pipeline's activities and dataset references against linked service types. If you dump all pipeline JSON into the LLM context, that's 200K–500K+ tokens — **the context window simply cannot hold it**. Even if it could, the cost would be prohibitive. Without `exec_python`, this class of queries is impossible to complete on any non-trivial ADF instance. It does not scale.

**The solution:** Mimic Claude Code's approach — give the agent tools to **explore** and **understand** data on disk first, then **generate and execute** code to process it:

1. **List & Save** — `adf_pipeline_list()` fetches all 242 pipelines and saves each as a JSON file to the session workspace. Only a summary (`"242 pipelines saved to pipelines/"`) enters the LLM context.
2. **Explore schema (`read_file`)** — The agent uses `read_file` to read 2–3 sample files (e.g. `datasets.json`, two pipeline JSONs) into context to understand the exact JSON structure and field names. This **does** consume tokens — but reading 2–3 samples is fundamentally different from reading all 242. This step is critical: the agent needs to see real data to write a correct script on the first try.
3. **Generate & Execute** — Based on the schema learned in step 2, the agent writes a Python script and runs it via `exec_python` in a subprocess. The script reads **all 242 pipeline files** from disk, cross-references against datasets and linked services, and prints a concise result.
4. **Observe & Iterate** — Only the script's printed output enters the LLM context. If the script has errors (e.g. wrong field names), the agent reads different sample files to diagnose, fixes the script, and retries.

```
Without exec_python:                    With exec_python:
─────────────────────                   ─────────────────
adf_pipeline_list()                     adf_pipeline_list()
  → 242 pipelines as JSON                → "242 pipelines saved to pipelines/"
  → ~500K tokens in context               → ~50 tokens in context

LLM reads all 500K tokens              read_file(sample1.json, sample2.json)
to cross-reference pipelines              → 2–3 samples to learn schema
  → context window exceeded               → ~3K tokens in context
  → impossible at scale
                                        exec_python(analysis_script)
                                          → script reads ALL 242 files from disk
                                          → prints "20 pipelines matched"
                                          → ~100 tokens in context

Total: ~500K tokens (impossible)        Total: ~3K tokens (scalable)
```

For **pipeline-related linked service queries**, this is not an optimization — it is a **hard requirement**. An ADF with hundreds of pipelines generates hundreds of JSON files, each containing nested activities, dataset references, and parameters. Cross-referencing this against datasets and linked services is a data processing task, not a language task. `exec_python` moves the heavy lifting out of the LLM and into a Python subprocess where it belongs.

#### Pre-loaded Runtime

To keep `exec_python` scripts concise, a helper module (`_exec_runtime.py`) is deployed to the session directory once and auto-imported in every execution:

```python
# Available without importing:
json, re, sys, Path, Counter, defaultdict

# Helper functions:
load_json("datasets.json")       # Load from session dir
save_json("results.json", data)  # Save to session dir
pretty_print(data)               # Pretty-print with truncation
session_dir                      # Path to current session directory
```

Example — cross-referencing 242 pipelines × 65 datasets × 18 linked services entirely on disk:

```python
exec_python("""
pipelines_dir = session_dir / "pipelines"
datasets = load_json("datasets.json")
linked_services = load_json("linked_services.json")

# Build lookup: dataset name -> linked service name
ds_to_ls = {d["name"]: d["linked_service"] for d in datasets}

# Find Snowflake linked services
snowflake_ls = {ls["name"] for ls in linked_services if "Snowflake" in ls["type"]}

# Cross-reference: for each pipeline, check if any activity references a Snowflake dataset
results = []
for f in sorted(pipelines_dir.glob("*.json")):
    pipeline = json.loads(f.read_text())
    for activity in pipeline.get("properties", {}).get("activities", []):
        for ds_ref in activity.get("inputs", []) + activity.get("outputs", []):
            ds_name = ds_ref.get("referenceName", "")
            if ds_to_ls.get(ds_name) in snowflake_ls:
                results.append((pipeline["name"], ds_name, ds_to_ls[ds_name]))

print(f"Found {len(results)} pipeline-dataset-service matches")
for pipe, ds, ls in results:
    print(f"  {pipe} -> {ds} -> {ls}")
""")
```

The LLM reads 2–3 sample files to learn the schema (~3K tokens), then `exec_python` processes all 242 pipeline files on disk. The bulk data never enters the context.

## Architecture

```
User prompt
  │
  ▼
┌─────────────────────────────────────────────┐
│  Claude (Sonnet / Opus)                     │
│  Extended Thinking → Reasoning → Tool Calls │
│                                             │
│  System Prompt                              │
│  ├─ ADF domain knowledge                   │
│  ├─ Tool descriptions                      │
│  └─ Skills catalog (loaded at startup)      │
└───────┬─────────────────────────────────────┘
        │  tool calls
        ▼
┌─────────────────────────────────────────────┐
│  LangChain Agent Loop (create_agent)         │
│                                             │
│  Tools                    Skills            │
│  ├─ adf_pipeline_list     ├─ find-pipe...   │
│  ├─ adf_linked_service_*  └─ test-linked..  │
│  ├─ adf_dataset_list                        │
│  ├─ adf_integration_runtime_*               │
│  ├─ exec_python  ◄── token saver            │
│  ├─ read_file / write_file                  │
│  ├─ glob / grep / list_dir                  │
│  └─ resolve_adf_target                      │
│                                             │
│  Context: ADFConfig, session_dir, cache     │
└───────┬─────────────────────────────────────┘
        │  Azure SDK calls
        ▼
┌─────────────────────────────────────────────┐
│  Azure Data Factory REST API                │
│  (via azure-mgmt-datafactory SDK)           │
└─────────────────────────────────────────────┘
```

## Observability with MLflow (Local)

All tracing is **local by default** — no remote server needed. The agent uses `mlflow.langchain.autolog()` for zero-config tracing, saving everything to `./mlruns/` on disk. This fits the CLI-first, local development philosophy: run the agent, then open MLflow UI to inspect exactly what happened.

```python
# adf_agent/observability/mlflow_setup.py
mlflow.set_experiment("ADF-Agent")
mlflow.langchain.autolog()
```

Every agent invocation is logged as an MLflow run under the `ADF-Agent` experiment, capturing:
- Input/output messages
- Tool calls and results
- Token usage
- Latency

```bash
# View local traces — no setup required
mlflow ui  # http://localhost:5000
```

Optionally point to a remote server:

```bash
export MLFLOW_TRACKING_URI=http://your-mlflow-server:5000
uv run adf_agent --interactive
```

## Model Support

The agent supports models hosted by **Anthropic** directly or via **Azure AI Foundry**.

### Anthropic (default)

```env
CLAUDE_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
CLAUDE_MODEL=claude-sonnet-4-5-20250929   # optional, this is the default
```

### Azure AI Foundry

```env
CLAUDE_PROVIDER=azure_foundry
ANTHROPIC_FOUNDRY_API_KEY=your-key
ANTHROPIC_FOUNDRY_BASE_URL=https://<resource>.services.ai.azure.com/anthropic
CLAUDE_MODEL=claude-sonnet-4-5-20250929   # optional
```

The Azure Foundry integration uses a custom `ChatAzureFoundryClaude` class that extends `ChatAnthropic` and swaps the HTTP client to `AnthropicFoundry`, so all LangChain features (streaming, tool calling, Extended Thinking) work identically on both providers.

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Azure credentials configured (`az login` or service principal)
- Anthropic API key or Azure Foundry endpoint

### Install

```bash
uv sync
```

### Configure

Create a `.env` file (or run `uv run adf_agent` for guided onboarding):

```env
# Model provider
CLAUDE_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Optional overrides
CLAUDE_MODEL=claude-sonnet-4-5-20250929
MAX_TOKENS=16000
```

Azure credentials are resolved via `DefaultAzureCredential` (Azure CLI, managed identity, environment variables, etc.).

### Run

```bash
# Interactive mode (default)
uv run adf_agent

# Explicit interactive flag
uv run adf_agent --interactive

# Single request
uv run adf_agent "list all pipelines in sales prod"

# Disable Extended Thinking
uv run adf_agent --no-thinking "list linked services"
```

## Token Tracking

The CLI displays per-turn and total token usage with Anthropic Prompt Caching breakdown:

```
─── Token Usage (turn) ───
 Input: 3,625  (cache_create: 3,269 · cache_read: 0)
 Output: 155
 Total: 3,780

─── Token Usage (total) ───
 Input: 7,250  (cache_create: 3,269 · cache_read: 3,269)
 Output: 410
 Total: 7,660
```

System prompt and skills catalog are marked with `cache_control: ephemeral` (5-min TTL), so multi-turn conversations benefit from cache hits at 0.1x the cost of fresh input tokens.

## Progressive Prompt Caching

Anthropic's prompt cache is a **prefix match** across three layers: `tools → system → messages`. A cache hit means the entire prefix up to a breakpoint matches a previous request exactly. We use progressive (incremental) caching to maximize cache hits during tool loops.

### The Problem

In a ReAct agent loop, each API call re-sends the full conversation history. Without caching on the messages layer, every call pays full price for all previous content — tool results, skill instructions, etc.

LangChain's `ChatAnthropic` has built-in support for injecting `cache_control` into messages, but LangGraph's `create_agent` never passes the `cache_control` kwarg, so it was unused.

### The Fix

Both provider classes (`CachedChatAnthropic` and `ChatAzureFoundryClaude`) override `_get_request_payload` to inject `cache_control` automatically:

```python
def _get_request_payload(self, input_, *, stop=None, **kwargs):
    kwargs.setdefault("cache_control", {"type": "ephemeral"})
    return super()._get_request_payload(input_, stop=stop, **kwargs)
```

This places a cache breakpoint on the **last message block of every API call**. Each turn's breakpoint advances forward, so all previous content becomes cached prefix.

### How It Works

```
Call 1:  [system ✍] [user_msg, tool_result_1 ✍]           ← all cache_creation
Call 2:  [system ✓] [user_msg, tool_result_1 ✓] [tool_result_2 ✍]  ← prefix read, new creation
Call 3:  [system ✓] [..., tool_result_2 ✓] [tool_result_3 ✍]       ← prefix read, new creation
```

Each call only pays `cache_creation` (1.25x) for **new** messages. Everything before is `cache_read` at 0.1x.

### Cache Breakpoints (3 of 4 max)

| # | Location | Content |
|---|----------|---------|
| 1 | system block 1 | Main system prompt (~1,700 tokens) |
| 2 | system block 2 | Skill catalog summary |
| 3 | last message block (auto) | Conversation history up to current point |

### Multi-Turn: Extended Thinking Cache Invalidation

When Extended Thinking is enabled, thinking blocks are stripped from history when the user sends a new message. This changes the message sequence, causing a **messages layer cache miss** on the first call of each new turn. The `tools` and `system` layers remain cached.

| Event | tools | system | messages |
|-------|-------|--------|----------|
| Same turn, tool loop | cache_read | cache_read | cache_read + creation for new |
| New user message (thinking stripped) | cache_read | cache_read | re-creation |

For full details, see [docs/prompt-caching.md](docs/prompt-caching.md).

## Project Structure

```
ADFAgent/
├── adf_agent/
│   ├── agent.py              # Agent core: model init, ReAct loop, streaming
│   ├── cli.py                # Interactive CLI with Rich live display
│   ├── context.py            # Runtime context: ADF config, session dir, cache
│   ├── prompts.py            # System prompt builder with skills injection
│   ├── skill_loader.py       # Two-tier skill discovery and loading
│   ├── azure_claude.py       # Azure Foundry ChatAnthropic adapter
│   ├── tools/
│   │   ├── adf_tools.py      # ADF pipeline/dataset/linked service/IR tools
│   │   ├── general_tools.py  # File ops, exec_python, target resolution
│   │   ├── skill_tools.py    # load_skill tool
│   │   ├── azure_adf_client.py  # Azure SDK wrapper
│   │   └── _exec_runtime.py  # Pre-loaded helpers for exec_python
│   ├── stream/               # Streaming event system + token tracking
│   └── observability/        # MLflow autolog setup
├── azure_tools/              # Reusable Azure SDK wrappers (ADF, KeyVault, Storage, Batch)
├── .claude/skills/           # Skill definitions (Markdown with YAML frontmatter)
└── workspace/sessions/       # Per-session output (pipeline JSON, scripts, results)
```
