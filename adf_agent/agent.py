"""
ADF Agent 主体

使用 LangChain 1.0 的 create_agent API 实现 ADF Agent，支持：
- Extended Thinking 显示模型思考过程
- 事件级流式输出 (thinking / text / tool_call / tool_result)
- Azure Data Factory 操作
"""

import os
from pathlib import Path
from typing import Optional, Iterator

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, AIMessageChunk
from langgraph.checkpoint.memory import InMemorySaver

from .context import ADFAgentContext, ADFConfig
from .tools import ALL_TOOLS
from .prompts import build_system_prompt
from .stream import StreamEventEmitter, ToolCallTracker, TokenTracker, is_success, DisplayLimits


# 加载环境变量（override=True 确保 .env 文件覆盖系统环境变量）
load_dotenv(override=True)


# 默认配置
DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_MAX_TOKENS = 16000
DEFAULT_TEMPERATURE = 1.0  # Extended Thinking 要求温度为 1.0
DEFAULT_THINKING_BUDGET = 10000


def get_claude_config() -> dict:
    """
    获取 Claude 配置，支持多种 provider

    支持的 provider：
    - anthropic (默认): 直接使用 Anthropic API
    - azure_foundry: 使用 Azure AI Foundry

    环境变量：
    - CLAUDE_PROVIDER: 选择 provider (anthropic 或 azure_foundry)

    Anthropic (默认):
    - ANTHROPIC_API_KEY 或 ANTHROPIC_AUTH_TOKEN
    - ANTHROPIC_BASE_URL (可选)

    Azure AI Foundry:
    - ANTHROPIC_FOUNDRY_API_KEY
    - ANTHROPIC_FOUNDRY_BASE_URL

    Returns:
        包含 model_class 和 init_kwargs 的配置字典
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
        return {
            "model_class": ChatAnthropic,
            "init_kwargs": {
                "api_key": os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"),
                "base_url": os.getenv("ANTHROPIC_BASE_URL"),
            },
        }


def get_anthropic_credentials() -> tuple[str | None, str | None]:
    """
    获取 Anthropic API 认证信息

    支持多种认证方式：
    1. ANTHROPIC_API_KEY - 标准 API Key
    2. ANTHROPIC_AUTH_TOKEN - 第三方代理认证 Token

    Returns:
        (api_key, base_url) 元组
    """
    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")
    base_url = os.getenv("ANTHROPIC_BASE_URL")
    return api_key, base_url


def check_api_credentials() -> bool:
    """检查是否配置了 API 认证"""
    config = get_claude_config()
    api_key = config["init_kwargs"].get("api_key")
    return api_key is not None


def load_adf_config() -> ADFConfig:
    """
    从环境变量加载 ADF 配置

    配置可能不完整，Agent 会在需要时询问用户。
    """
    return ADFConfig(
        resource_group=os.getenv("ADF_RESOURCE_GROUP"),
        factory_name=os.getenv("ADF_FACTORY_NAME"),
        subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID"),
    )


class ADFAgent:
    """
    ADF Agent - Azure Data Factory 助手

    使用示例：
        agent = ADFAgent()

        # 查看 system prompt
        print(agent.get_system_prompt())

        # 运行 agent
        for event in agent.stream_events("列出所有 pipeline"):
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
    ):
        """
        初始化 Agent

        Args:
            model: 模型名称，默认 claude-sonnet-4-5-20250929
            working_directory: 工作目录
            max_tokens: 最大 tokens
            temperature: 温度参数 (启用 thinking 时强制为 1.0)
            enable_thinking: 是否启用 Extended Thinking
            thinking_budget: thinking 的 token 预算
            adf_config: ADF 配置，如果未提供则从环境变量加载
        """
        # thinking 配置
        self.enable_thinking = enable_thinking
        self.thinking_budget = thinking_budget

        # 配置 (启用 thinking 时温度必须为 1.0)
        self.model_name = model or os.getenv("CLAUDE_MODEL", DEFAULT_MODEL)
        self.max_tokens = max_tokens or int(os.getenv("MAX_TOKENS", str(DEFAULT_MAX_TOKENS)))
        if enable_thinking:
            self.temperature = 1.0  # Anthropic 要求启用 thinking 时温度为 1.0
        else:
            self.temperature = temperature or float(os.getenv("MODEL_TEMPERATURE", str(DEFAULT_TEMPERATURE)))
        self.working_directory = working_directory or Path.cwd()

        # 加载 ADF 配置
        self.adf_config = adf_config or load_adf_config()

        # 构建 system prompt
        self.system_prompt = build_system_prompt(self.adf_config)

        # 创建上下文（供 tools 使用）
        self.context = ADFAgentContext(
            working_directory=self.working_directory,
            adf_config=self.adf_config,
        )

        # 创建 LangChain Agent
        self.agent = self._create_agent()

    def _create_agent(self):
        """
        创建 LangChain Agent

        使用 LangChain 1.0 的 create_agent API:
        - model: 可以是字符串 ID 或 model 实例
        - tools: 工具列表
        - system_prompt: 系统提示
        - context_schema: 上下文类型（供 ToolRuntime 使用）
        - checkpointer: 会话记忆

        支持多种 provider:
        - anthropic (默认): 使用 init_chat_model
        - azure_foundry: 使用 ChatAzureFoundryClaude 直接实例化
        """
        # 获取 provider 配置
        config = get_claude_config()
        model_class = config["model_class"]
        provider_kwargs = config["init_kwargs"]

        # 构建初始化参数
        init_kwargs = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        # 添加认证参数
        if provider_kwargs.get("api_key"):
            init_kwargs["api_key"] = provider_kwargs["api_key"]
        if provider_kwargs.get("base_url"):
            init_kwargs["base_url"] = provider_kwargs["base_url"]

        # Extended Thinking 配置
        if self.enable_thinking:
            init_kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget,
            }

        # 初始化模型
        # 对于 azure_foundry，直接使用 model_class 实例化
        # 对于 anthropic，也直接使用 ChatAnthropic 实例化（保持一致性）
        model = model_class(**init_kwargs)

        # 创建 Agent
        agent = create_agent(
            model=model,
            tools=ALL_TOOLS,
            system_prompt=self.system_prompt,
            context_schema=ADFAgentContext,
            checkpointer=InMemorySaver(),
        )

        return agent

    def get_system_prompt(self) -> str:
        """获取当前 system prompt"""
        return self.system_prompt

    def get_adf_config(self) -> ADFConfig:
        """获取当前 ADF 配置"""
        return self.adf_config

    def invoke(self, message: str, thread_id: str = "default") -> dict:
        """
        同步调用 Agent

        Args:
            message: 用户消息
            thread_id: 会话 ID（用于多轮对话）

        Returns:
            Agent 响应
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
        流式调用 Agent (state 级别)

        Args:
            message: 用户消息
            thread_id: 会话 ID

        Yields:
            流式响应块 (完整状态更新)
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
        事件级流式输出，支持 thinking 和 token 级流式

        Args:
            message: 用户消息
            thread_id: 会话 ID

        Yields:
            事件字典，格式如下:
            - {"type": "thinking", "content": "..."} - 思考内容片段
            - {"type": "text", "content": "..."} - 响应文本片段
            - {"type": "tool_call", "name": "...", "args": {...}} - 工具调用
            - {"type": "tool_result", "name": "...", "content": "...", "success": bool} - 工具结果
            - {"type": "done", "response": "..."} - 完成标记，包含完整响应
        """
        config = {"configurable": {"thread_id": thread_id}}
        emitter = StreamEventEmitter()
        tracker = ToolCallTracker()
        token_tracker = TokenTracker()

        full_response = ""
        debug = os.getenv("ADF_DEBUG", "").lower() in ("1", "true", "yes")

        # Parallel tool call 检测：
        # 一次 API 调用可能返回多个 tool_call（并行执行），
        # 但 usage_metadata 只出现一次。我们把 token_usage 延迟到
        # 该批次的最后一个 tool_result 之后再发送。
        from .stream.token_tracker import TokenUsageInfo
        pending_turn_usage: TokenUsageInfo | None = None
        parallel_count = 0

        def _emit_pending():
            """发送缓冲的 per-turn token_usage（上一批次的）"""
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

        # 使用 messages 模式获取 token 级流式
        try:
            for event in self.agent.stream(
                {"messages": [{"role": "user", "content": message}]},
                config=config,
                context=self.context,
                stream_mode="messages",
            ):
                # event 可能是 tuple(message, metadata) 或直接 message
                if isinstance(event, tuple) and len(event) >= 2:
                    chunk = event[0]
                else:
                    chunk = event

                if debug:
                    chunk_type = type(chunk).__name__
                    print(f"[DEBUG] Event: {chunk_type}")

                # 处理 AIMessageChunk / AIMessage
                if isinstance(chunk, (AIMessageChunk, AIMessage)):
                    # 新的 API 调用开始 → 发送上一批次缓冲的 token_usage
                    pending_ev = _emit_pending()
                    if pending_ev:
                        yield pending_ev

                    # 更新 token 统计
                    token_tracker.update(chunk)

                    # 处理 content
                    for ev in self._process_chunk_content(chunk, emitter, tracker):
                        if ev.type == "text":
                            full_response += ev.data.get("content", "")
                        if debug:
                            print(f"[DEBUG] Yielding: {ev.type}")
                        yield ev.data

                    # 处理 tool_calls (有些情况下在 chunk.tool_calls 中)
                    if hasattr(chunk, "tool_calls") and chunk.tool_calls:
                        for ev in self._process_tool_calls(chunk.tool_calls, emitter, tracker):
                            if debug:
                                print(f"[DEBUG] Yielding from tool_calls: {ev.type}")
                            yield ev.data

                # 处理 ToolMessage (工具执行结果)
                elif hasattr(chunk, "type") and chunk.type == "tool":
                    turn_usage = token_tracker.finalize_turn()

                    if debug:
                        tool_name = getattr(chunk, "name", "unknown")
                        print(f"[DEBUG] Processing tool result: {tool_name}")

                    # 处理工具结果
                    for ev in self._process_tool_result(chunk, emitter, tracker):
                        if debug:
                            print(f"[DEBUG] Yielding: {ev.type}")
                        yield ev.data

                    # 缓冲 token_usage，等批次结束再发送
                    if turn_usage and not turn_usage.is_empty():
                        # 该批次第一个 tool（有 usage 数据）
                        pending_turn_usage = turn_usage
                        parallel_count = 1
                    elif pending_turn_usage is not None:
                        # 同一批次的后续 parallel tool（finalize 返回 None）
                        parallel_count += 1

            if debug:
                print("[DEBUG] Stream completed normally")

        except Exception as e:
            if debug:
                import traceback
                print(f"[DEBUG] Stream error: {e}")
                traceback.print_exc()
            # 发送错误事件让用户知道发生了什么
            yield emitter.error(str(e)).data
            raise

        # 发送上一批次缓冲的 per-turn token_usage
        pending_ev = _emit_pending()
        if pending_ev:
            yield pending_ev

        # 最后一个 turn（最终回复，没有 tool_result 触发 finalize）
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

        # 发送汇总 token 使用量（所有 turn 的 SUM）
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

        # 发送完成事件
        yield emitter.done(full_response).data

    def _process_chunk_content(self, chunk, emitter: StreamEventEmitter, tracker: ToolCallTracker):
        """处理 chunk 的 content"""
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
                    # 立即发送（显示"执行中"状态），参数可能尚不完整
                    if tracker.is_ready(tool_id):
                        tracker.mark_emitted(tool_id)
                        yield emitter.tool_call(name, args_payload, tool_id)

            elif block_type == "input_json_delta":
                # 累积 JSON 片段（args 分批到达）
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
        """处理 chunk.tool_calls - 立即发送 tool_call 事件"""
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
        """处理工具结果"""
        # 最终化：解析累积的 JSON 片段为 args
        tracker.finalize_all()

        # 发送所有工具调用的更新（参数现在是完整的）
        for info in tracker.get_all():
            yield emitter.tool_call(info.name, info.args, info.id)

        # 发送结果
        name = getattr(chunk, "name", "unknown")
        raw_content = str(getattr(chunk, "content", ""))
        content = raw_content[:DisplayLimits.TOOL_RESULT_MAX]
        if len(raw_content) > DisplayLimits.TOOL_RESULT_MAX:
            content += "\n... (truncated)"

        # 基于内容判断是否成功
        success = is_success(content)

        yield emitter.tool_result(name, content, success)

    def get_last_response(self, result: dict) -> str:
        """
        从结果中提取最后的 AI 响应文本

        Args:
            result: invoke 或 stream 的结果

        Returns:
            AI 响应文本
        """
        messages = result.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                if isinstance(msg.content, str):
                    return msg.content
                elif isinstance(msg.content, list):
                    # 处理多部分内容
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
) -> ADFAgent:
    """
    便捷函数：创建 ADF Agent

    Args:
        model: 模型名称
        working_directory: 工作目录
        enable_thinking: 是否启用 Extended Thinking
        thinking_budget: thinking 的 token 预算

    Returns:
        配置好的 ADFAgent 实例
    """
    return ADFAgent(
        model=model,
        working_directory=working_directory,
        enable_thinking=enable_thinking,
        thinking_budget=thinking_budget,
    )
