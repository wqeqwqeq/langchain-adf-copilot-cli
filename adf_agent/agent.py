"""
ADF Agent Core

ADF Agent implementation using LangChain 1.0 create_agent API, supporting:
- Extended Thinking to display model reasoning process
- Event-level streaming output (thinking / text / tool_call / tool_result)
- Azure Data Factory operations
"""

import os
from pathlib import Path
from typing import Optional, Iterator

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, AIMessageChunk
from langgraph.checkpoint.memory import InMemorySaver

from .context import ADFAgentContext, ADFConfig
from .skill_loader import SkillLoader
from .tools import ALL_TOOLS
from .prompts import build_system_prompt
from .stream import StreamEventEmitter, ToolCallTracker, TokenTracker, is_success, DisplayLimits


# Load environment variables (override=True ensures .env overrides system env vars)
load_dotenv(override=True)


# Default configuration
DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_MAX_TOKENS = 16000
DEFAULT_TEMPERATURE = 1.0  # Extended Thinking requires temperature = 1.0
DEFAULT_THINKING_BUDGET = 10000


def get_claude_config() -> dict:
    """
    Get Claude configuration, supporting multiple providers

    Supported providers:
    - anthropic (default): Direct Anthropic API
    - azure_foundry: Azure AI Foundry

    Environment variables:
    - CLAUDE_PROVIDER: Select provider (anthropic or azure_foundry)

    Anthropic (default):
    - ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN
    - ANTHROPIC_BASE_URL (optional)

    Azure AI Foundry:
    - ANTHROPIC_FOUNDRY_API_KEY
    - ANTHROPIC_FOUNDRY_BASE_URL

    Returns:
        Dict containing model_class and init_kwargs
    """
    provider = os.getenv("CLAUDE_PROVIDER", "anthropic").lower()

    if provider == "azure_foundry":
        from .azure_claude import ChatAzureFoundryClaude
        return {
            "model_class": ChatAzureFoundryClaude,
            "init_kwargs": {
                "api_key": os.getenv("ANTHROPIC_FOUNDRY_API_KEY"),
                "base_url": os.getenv("ANTHROPIC_FOUNDRY_BASE_URL"),
            },
        }
    else:
        from langchain_anthropic import ChatAnthropic

        class CachedChatAnthropic(ChatAnthropic):
            """ChatAnthropic with progressive prompt caching on every API call."""

            def _get_request_payload(self, input_, *, stop=None, **kwargs):
                kwargs.setdefault("cache_control", {"type": "ephemeral"})
                return super()._get_request_payload(input_, stop=stop, **kwargs)

        return {
            "model_class": CachedChatAnthropic,
            "init_kwargs": {
                "api_key": os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"),
                "base_url": os.getenv("ANTHROPIC_BASE_URL"),
            },
        }


def get_anthropic_credentials() -> tuple[str | None, str | None]:
    """
    Get Anthropic API credentials

    Supports multiple authentication methods:
    1. ANTHROPIC_API_KEY - Standard API Key
    2. ANTHROPIC_AUTH_TOKEN - Third-party proxy auth token

    Returns:
        (api_key, base_url) tuple
    """
    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")
    base_url = os.getenv("ANTHROPIC_BASE_URL")
    return api_key, base_url


def check_api_credentials() -> bool:
    """Check if API credentials are configured"""
    config = get_claude_config()
    api_key = config["init_kwargs"].get("api_key")
    return api_key is not None


def load_adf_config() -> ADFConfig:
    """
    Load ADF configuration from environment variables

    Configuration may be incomplete; the Agent will ask the user when needed.
    """
    return ADFConfig(
        resource_group=os.getenv("ADF_RESOURCE_GROUP"),
        factory_name=os.getenv("ADF_FACTORY_NAME"),
        subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID"),
    )


class ADFAgent:
    """
    ADF Agent - Azure Data Factory Assistant

    Usage:
        agent = ADFAgent()

        # View system prompt
        print(agent.get_system_prompt())

        # Run agent
        for event in agent.stream_events("List all pipelines"):
            print(event)
    """

    def __init__(
        self,
        model: Optional[str] = None,
        working_directory: Optional[Path] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        enable_thinking: bool = True,
        thinking_budget: int = DEFAULT_THINKING_BUDGET,
        adf_config: Optional[ADFConfig] = None,
        skill_paths: Optional[list[Path]] = None,
    ):
        """
        Initialize the Agent

        Args:
            model: Model name, defaults to claude-sonnet-4-5-20250929
            working_directory: Working directory
            max_tokens: Maximum tokens
            temperature: Temperature parameter (forced to 1.0 when thinking is enabled)
            enable_thinking: Whether to enable Extended Thinking
            thinking_budget: Token budget for thinking
            adf_config: ADF configuration; loaded from env vars if not provided
            skill_paths: Skills search path list, defaults to .claude/skills/ and ~/.claude/skills/
        """
        # Thinking configuration
        self.enable_thinking = enable_thinking
        self.thinking_budget = thinking_budget

        # Configuration (temperature must be 1.0 when thinking is enabled)
        self.model_name = model or os.getenv("CLAUDE_MODEL", DEFAULT_MODEL)
        self.max_tokens = max_tokens or int(os.getenv("MAX_TOKENS", str(DEFAULT_MAX_TOKENS)))
        if enable_thinking:
            self.temperature = 1.0  # Anthropic requires temperature = 1.0 when thinking is enabled
        else:
            self.temperature = temperature or float(os.getenv("MODEL_TEMPERATURE", str(DEFAULT_TEMPERATURE)))
        self.working_directory = working_directory or Path.cwd()

        # Load ADF configuration
        self.adf_config = adf_config or ADFConfig()

        # Initialize Skills loader
        self.skill_loader = SkillLoader(skill_paths)
        skills = self.skill_loader.scan_skills()

        # Build system prompt
        self.system_prompt = build_system_prompt(skills=skills)

        # Create context (used by tools)
        self.context = ADFAgentContext(
            working_directory=self.working_directory,
            adf_config=self.adf_config,
            skill_loader=self.skill_loader,
        )

        # Create LangChain Agent
        self.agent = self._create_agent()

    def _create_agent(self):
        """
        Create LangChain Agent

        Uses LangChain 1.0 create_agent API:
        - model: Can be a string ID or model instance
        - tools: List of tools
        - system_prompt: System prompt
        - context_schema: Context type (used by ToolRuntime)
        - checkpointer: Session memory

        Supports multiple providers:
        - anthropic (default): Uses init_chat_model
        - azure_foundry: Uses ChatAzureFoundryClaude directly
        """
        # Get provider configuration
        config = get_claude_config()
        model_class = config["model_class"]
        provider_kwargs = config["init_kwargs"]

        # Build initialization parameters
        init_kwargs = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        # Add authentication parameters
        if provider_kwargs.get("api_key"):
            init_kwargs["api_key"] = provider_kwargs["api_key"]
        if provider_kwargs.get("base_url"):
            init_kwargs["base_url"] = provider_kwargs["base_url"]

        # Extended Thinking configuration
        if self.enable_thinking:
            init_kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget,
            }

        # Initialize model
        # For azure_foundry, instantiate directly with model_class
        # For anthropic, also instantiate directly with ChatAnthropic (for consistency)
        model = model_class(**init_kwargs)

        # Create Agent
        agent = create_agent(
            model=model,
            tools=ALL_TOOLS,
            system_prompt=self.system_prompt,
            context_schema=ADFAgentContext,
            checkpointer=InMemorySaver(),
        )

        return agent

    def get_system_prompt(self) -> str:
        """Get the current system prompt text"""
        return self.system_prompt.content[0]["text"]

    def get_adf_config(self) -> ADFConfig:
        """Get the current ADF configuration"""
        return self.adf_config

    def invoke(self, message: str, thread_id: str = "default") -> dict:
        """
        Synchronous Agent invocation

        Args:
            message: User message
            thread_id: Session ID (for multi-turn conversations)

        Returns:
            Agent response
        """
        config = {"configurable": {"thread_id": thread_id}}

        result = self.agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config=config,
            context=self.context,
        )

        return result

    def stream(self, message: str, thread_id: str = "default") -> Iterator[dict]:
        """
        Streaming Agent invocation (state level)

        Args:
            message: User message
            thread_id: Session ID

        Yields:
            Streaming response chunks (full state updates)
        """
        config = {"configurable": {"thread_id": thread_id}}

        for chunk in self.agent.stream(
            {"messages": [{"role": "user", "content": message}]},
            config=config,
            context=self.context,
            stream_mode="values",
        ):
            yield chunk

    def stream_events(self, message: str, thread_id: str = "default") -> Iterator[dict]:
        """
        Event-level streaming output, supporting thinking and token-level streaming

        Args:
            message: User message
            thread_id: Session ID

        Yields:
            Event dicts in the following format:
            - {"type": "thinking", "content": "..."} - Thinking content fragment
            - {"type": "text", "content": "..."} - Response text fragment
            - {"type": "tool_call", "name": "...", "args": {...}} - Tool call
            - {"type": "tool_result", "name": "...", "content": "...", "success": bool} - Tool result
            - {"type": "done", "response": "..."} - Completion marker with full response
        """
        config = {"configurable": {"thread_id": thread_id}}
        emitter = StreamEventEmitter()
        tracker = ToolCallTracker()
        token_tracker = TokenTracker()

        full_response = ""
        debug = os.getenv("ADF_DEBUG", "").lower() in ("1", "true", "yes")

        # Parallel tool call detection:
        # A single API call may return multiple tool_calls (parallel execution),
        # but usage_metadata only appears once. We defer sending token_usage
        # until after the last tool_result in the batch.
        from .stream.token_tracker import TokenUsageInfo
        pending_turn_usage: TokenUsageInfo | None = None
        parallel_count = 0

        def _emit_pending():
            """Send buffered per-turn token_usage (from previous batch)"""
            nonlocal pending_turn_usage, parallel_count
            if pending_turn_usage is not None and not pending_turn_usage.is_empty():
                ev = emitter.token_usage(
                    input_tokens=pending_turn_usage.input_tokens,
                    output_tokens=pending_turn_usage.output_tokens,
                    total_tokens=pending_turn_usage.total_tokens,
                    cache_creation_input_tokens=pending_turn_usage.cache_creation_input_tokens,
                    cache_read_input_tokens=pending_turn_usage.cache_read_input_tokens,
                    is_total=False,
                    parallel_count=parallel_count,
                ).data
                pending_turn_usage = None
                parallel_count = 0
                return ev
            pending_turn_usage = None
            parallel_count = 0
            return None

        # Use messages mode for token-level streaming
        try:
            for event in self.agent.stream(
                {"messages": [{"role": "user", "content": message}]},
                config=config,
                context=self.context,
                stream_mode="messages",
            ):
                # event may be tuple(message, metadata) or a direct message
                if isinstance(event, tuple) and len(event) >= 2:
                    chunk = event[0]
                else:
                    chunk = event

                if debug:
                    chunk_type = type(chunk).__name__
                    print(f"[DEBUG] Event: {chunk_type}")

                # Handle AIMessageChunk / AIMessage
                if isinstance(chunk, (AIMessageChunk, AIMessage)):
                    # New API call started -> send buffered token_usage from previous batch
                    pending_ev = _emit_pending()
                    if pending_ev:
                        yield pending_ev

                    # Update token statistics
                    token_tracker.update(chunk)

                    # Process content
                    for ev in self._process_chunk_content(chunk, emitter, tracker):
                        if ev.type == "text":
                            full_response += ev.data.get("content", "")
                        if debug:
                            print(f"[DEBUG] Yielding: {ev.type}")
                        yield ev.data

                    # Handle tool_calls (sometimes in chunk.tool_calls)
                    if hasattr(chunk, "tool_calls") and chunk.tool_calls:
                        for ev in self._process_tool_calls(chunk.tool_calls, emitter, tracker):
                            if debug:
                                print(f"[DEBUG] Yielding from tool_calls: {ev.type}")
                            yield ev.data

                # Handle ToolMessage (tool execution result)
                elif hasattr(chunk, "type") and chunk.type == "tool":
                    turn_usage = token_tracker.finalize_turn()

                    if debug:
                        tool_name = getattr(chunk, "name", "unknown")
                        print(f"[DEBUG] Processing tool result: {tool_name}")

                    # Process tool result
                    for ev in self._process_tool_result(chunk, emitter, tracker):
                        if debug:
                            print(f"[DEBUG] Yielding: {ev.type}")
                        yield ev.data

                    # Buffer token_usage, send when batch ends
                    if turn_usage and not turn_usage.is_empty():
                        # First tool in batch (has usage data)
                        pending_turn_usage = turn_usage
                        parallel_count = 1
                    elif pending_turn_usage is not None:
                        # Subsequent parallel tool in same batch (finalize returns None)
                        parallel_count += 1

            if debug:
                print("[DEBUG] Stream completed normally")

        except Exception as e:
            if debug:
                import traceback
                print(f"[DEBUG] Stream error: {e}")
                traceback.print_exc()
            # Send error event to notify the user
            yield emitter.error(str(e)).data
            raise

        # Send buffered per-turn token_usage from previous batch
        pending_ev = _emit_pending()
        if pending_ev:
            yield pending_ev

        # Last turn (final reply, no tool_result to trigger finalize)
        last_turn = token_tracker.finalize_turn()
        if last_turn and not last_turn.is_empty():
            yield emitter.token_usage(
                input_tokens=last_turn.input_tokens,
                output_tokens=last_turn.output_tokens,
                total_tokens=last_turn.total_tokens,
                cache_creation_input_tokens=last_turn.cache_creation_input_tokens,
                cache_read_input_tokens=last_turn.cache_read_input_tokens,
                is_total=False,
                parallel_count=1,
            ).data

        # Send aggregate token usage (SUM of all turns)
        usage = token_tracker.get_usage()
        if not usage.is_empty():
            yield emitter.token_usage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
                cache_creation_input_tokens=usage.cache_creation_input_tokens,
                cache_read_input_tokens=usage.cache_read_input_tokens,
                is_total=True,
            ).data

        # Send completion event
        yield emitter.done(full_response).data

    def _process_chunk_content(self, chunk, emitter: StreamEventEmitter, tracker: ToolCallTracker):
        """Process chunk content"""
        content = chunk.content

        if isinstance(content, str):
            if content:
                yield emitter.text(content)
                return

        blocks = None
        if hasattr(chunk, "content_blocks"):
            try:
                blocks = chunk.content_blocks
            except Exception:
                blocks = None

        if blocks is None:
            if isinstance(content, dict):
                blocks = [content]
            elif isinstance(content, list):
                blocks = content
            else:
                return

        for raw_block in blocks:
            block = raw_block
            if not isinstance(block, dict):
                if hasattr(block, "model_dump"):
                    block = block.model_dump()
                elif hasattr(block, "dict"):
                    block = block.dict()
                else:
                    continue

            block_type = block.get("type")

            if block_type in ("thinking", "reasoning"):
                thinking_text = block.get("thinking") or block.get("reasoning") or ""
                if thinking_text:
                    yield emitter.thinking(thinking_text)

            elif block_type == "text":
                text = block.get("text") or block.get("content") or ""
                if text:
                    yield emitter.text(text)

            elif block_type in ("tool_use", "tool_call"):
                tool_id = block.get("id", "")
                name = block.get("name", "")
                args = block.get("input") if block_type == "tool_use" else block.get("args")
                args_payload = args if isinstance(args, dict) else {}

                if tool_id:
                    tracker.update(tool_id, name=name, args=args_payload)
                    # Send immediately (show "running" status), args may be incomplete
                    if tracker.is_ready(tool_id):
                        tracker.mark_emitted(tool_id)
                        yield emitter.tool_call(name, args_payload, tool_id)

            elif block_type == "input_json_delta":
                # Accumulate JSON fragments (args arrive in batches)
                partial_json = block.get("partial_json", "")
                if partial_json:
                    tracker.append_json_delta(partial_json, block.get("index", 0))

            elif block_type == "tool_call_chunk":
                tool_id = block.get("id", "")
                name = block.get("name", "")
                if tool_id:
                    tracker.update(tool_id, name=name)
                partial_args = block.get("args", "")
                if isinstance(partial_args, str) and partial_args:
                    tracker.append_json_delta(partial_args, block.get("index", 0))

    def _process_tool_calls(self, tool_calls: list, emitter: StreamEventEmitter, tracker: ToolCallTracker):
        """Process chunk.tool_calls - send tool_call events immediately"""
        for tc in tool_calls:
            tool_id = tc.get("id", "")
            if tool_id:
                name = tc.get("name", "")
                args = tc.get("args", {})
                args_payload = args if isinstance(args, dict) else {}

                tracker.update(tool_id, name=name, args=args_payload)
                if tracker.is_ready(tool_id):
                    tracker.mark_emitted(tool_id)
                    yield emitter.tool_call(name, args_payload, tool_id)

    def _process_tool_result(self, chunk, emitter: StreamEventEmitter, tracker: ToolCallTracker):
        """Process tool result"""
        # Finalize: parse accumulated JSON fragments into args
        tracker.finalize_all()

        # Send updates for all tool calls (args are now complete)
        for info in tracker.get_all():
            yield emitter.tool_call(info.name, info.args, info.id)

        # Send result
        name = getattr(chunk, "name", "unknown")
        raw_content = str(getattr(chunk, "content", ""))
        content = raw_content[:DisplayLimits.TOOL_RESULT_MAX]
        if len(raw_content) > DisplayLimits.TOOL_RESULT_MAX:
            content += "\n... (truncated)"

        # Determine success based on content
        success = is_success(content)

        yield emitter.tool_result(name, content, success)

    def get_last_response(self, result: dict) -> str:
        """
        Extract the last AI response text from the result

        Args:
            result: Result from invoke or stream

        Returns:
            AI response text
        """
        messages = result.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                if isinstance(msg.content, str):
                    return msg.content
                elif isinstance(msg.content, list):
                    # Handle multi-part content
                    text_parts = []
                    for part in msg.content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        elif isinstance(part, str):
                            text_parts.append(part)
                    return "\n".join(text_parts)
        return ""


def create_adf_agent(
    model: Optional[str] = None,
    working_directory: Optional[Path] = None,
    enable_thinking: bool = True,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
    skill_paths: Optional[list[Path]] = None,
) -> ADFAgent:
    """
    Convenience function: Create an ADF Agent

    Args:
        model: Model name
        working_directory: Working directory
        enable_thinking: Whether to enable Extended Thinking
        thinking_budget: Token budget for thinking
        skill_paths: Skills search path list

    Returns:
        Configured ADFAgent instance
    """
    return ADFAgent(
        model=model,
        working_directory=working_directory,
        enable_thinking=enable_thinking,
        thinking_budget=thinking_budget,
        skill_paths=skill_paths,
    )
