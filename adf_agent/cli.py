"""
ADF Agent CLI

命令行入口，提供交互式对话功能：
- 流式输出支持 Extended Thinking
- ADF 配置状态显示
- 工具调用可视化
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.formatted_text import HTML
from rich.console import Console, Group
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.live import Live
from rich.text import Text
from rich.spinner import Spinner

from .agent import ADFAgent, check_api_credentials, load_adf_config
from .stream import (
    ToolResultFormatter,
    has_args,
    DisplayLimits,
    ToolStatus,
    format_tool_compact,
    is_success,
)


# 加载环境变量
load_dotenv(override=True)

# Rich Console 配置
console = Console(
    legacy_windows=(sys.platform == 'win32'),
    no_color=os.getenv('NO_COLOR') is not None,
)

# 全局工具结果格式化器
formatter = ToolResultFormatter()


# === 终端高度计算 ===

def get_display_heights(con: Console) -> dict:
    """根据终端高度计算各区域的显示行数限制"""
    terminal_height = con.height or 25
    available = terminal_height - 5  # 预留空间

    return {
        "thinking": max(3, available // 5),
        "tools": max(5, available * 3 // 10),
        "response": max(5, available // 2),
    }


def truncate_to_lines(text: str, max_lines: int) -> str:
    """截断文本到指定行数，保留最新内容"""
    lines = text.split('\n')
    if len(lines) <= max_lines:
        return text
    return "...\n" + '\n'.join(lines[-max_lines + 1:])


# === 流式处理状态 ===

class StreamState:
    """流式处理状态容器"""

    def __init__(self):
        self.thinking_text = ""
        self.response_text = ""
        self.tool_calls = []
        self.tool_results = []
        self.is_thinking = False
        self.is_responding = False
        self.is_processing = False

    def handle_event(self, event: dict) -> str:
        """处理单个流式事件"""
        event_type = event.get("type")

        if event_type == "thinking":
            self.is_thinking = True
            self.is_responding = False
            self.is_processing = False
            self.thinking_text += event.get("content", "")

        elif event_type == "text":
            self.is_thinking = False
            self.is_responding = True
            self.is_processing = False
            self.response_text += event.get("content", "")

        elif event_type == "tool_call":
            self.is_thinking = False
            self.is_responding = False
            self.is_processing = False

            tool_id = event.get("id", "")
            tc_data = {
                "id": tool_id,
                "name": event.get("name", "unknown"),
                "args": event.get("args", {}),
            }

            # 用 tool_id 去重和更新
            if tool_id:
                updated = False
                for i, tc in enumerate(self.tool_calls):
                    if tc.get("id") == tool_id:
                        self.tool_calls[i] = tc_data
                        updated = True
                        break
                if not updated:
                    self.tool_calls.append(tc_data)
            else:
                self.tool_calls.append(tc_data)

        elif event_type == "tool_result":
            self.is_processing = True
            self.tool_results.append({
                "name": event.get("name", "unknown"),
                "content": event.get("content", ""),
            })

        elif event_type == "done":
            self.is_processing = False
            if not self.response_text:
                self.response_text = event.get("response", "")

        elif event_type == "error":
            self.is_processing = False
            self.is_thinking = False
            self.is_responding = False
            error_msg = event.get("message", "Unknown error")
            self.response_text += f"\n\n[Error] {error_msg}"

        return event_type

    def get_display_args(self) -> dict:
        """获取用于 create_streaming_display 的参数"""
        return {
            "thinking_text": self.thinking_text,
            "response_text": self.response_text,
            "tool_calls": self.tool_calls,
            "tool_results": self.tool_results,
            "is_thinking": self.is_thinking,
            "is_responding": self.is_responding,
            "is_processing": self.is_processing,
        }


def format_tool_result_compact(name: str, content: str, max_lines: int = 5) -> list:
    """使用树形格式显示工具结果"""
    elements = []

    if not content.strip():
        elements.append(Text("  └ (empty)", style="dim"))
        return elements

    lines = content.strip().split("\n")
    total_lines = len(lines)
    display_lines = lines[:max_lines]

    for i, line in enumerate(display_lines):
        prefix = "└" if i == 0 else " "
        if len(line) > 80:
            line = line[:77] + "..."
        style = "dim" if is_success(content) else "red dim"
        elements.append(Text(f"  {prefix} {line}", style=style))

    remaining = total_lines - max_lines
    if remaining > 0:
        elements.append(Text(f"    ... +{remaining} lines", style="dim italic"))

    return elements


def display_final_results(
    state: StreamState,
    thinking_max_length: int = DisplayLimits.THINKING_FINAL,
    tool_result_max_length: int = DisplayLimits.TOOL_RESULT_FINAL,
    show_thinking: bool = True,
    show_tools: bool = True,
    show_response_panel: bool = True,
):
    """显示最终结果"""
    # 显示 thinking
    if show_thinking and state.thinking_text:
        display_thinking = state.thinking_text
        if len(display_thinking) > thinking_max_length:
            half = thinking_max_length // 2
            display_thinking = display_thinking[:half] + "\n\n... (truncated) ...\n\n" + display_thinking[-half:]
        console.print(Panel(
            Text(display_thinking, style="dim"),
            title="Thinking",
            border_style="blue",
        ))

    # 显示工具调用和结果
    if show_tools and state.tool_calls:
        for i, tc in enumerate(state.tool_calls):
            has_result = i < len(state.tool_results)
            tr = state.tool_results[i] if has_result else None
            content = tr.get('content', '') if tr else ''

            if has_result and is_success(content):
                status = ToolStatus.SUCCESS
                style = "bold green"
            elif has_result:
                status = ToolStatus.ERROR
                style = "bold red"
            else:
                status = ToolStatus.PENDING
                style = "dim"

            tool_compact = format_tool_compact(tc['name'], tc.get('args'))
            tool_text = Text()
            tool_text.append(f"{status.value} ", style=style)
            tool_text.append(tool_compact, style=style)
            console.print(tool_text)

            if has_result:
                result_elements = format_tool_result_compact(
                    tr['name'],
                    content,
                    max_lines=10,
                )
                for elem in result_elements:
                    console.print(elem)
        console.print()

    # 显示最终响应
    if state.response_text:
        if show_response_panel:
            console.print(Panel(
                Markdown(state.response_text),
                title="Response",
                border_style="green",
            ))
        else:
            console.print(f"\n[bold blue]Assistant:[/bold blue]")
            console.print(Markdown(state.response_text))
            console.print()


def create_streaming_display(
    thinking_text: str = "",
    response_text: str = "",
    tool_calls: list = None,
    tool_results: list = None,
    is_thinking: bool = False,
    is_responding: bool = False,
    is_waiting: bool = False,
    is_processing: bool = False,
    max_heights: dict = None,
) -> Group:
    """创建流式显示的布局"""
    elements = []
    tool_calls = tool_calls or []
    tool_results = tool_results or []

    if max_heights is None:
        max_heights = {"thinking": 10, "tools": 10, "response": 15}

    # 初始等待状态
    if is_waiting and not thinking_text and not response_text and not tool_calls:
        spinner = Spinner("dots", text=" AI 正在思考中...", style="cyan")
        elements.append(spinner)
        return Group(*elements)

    # Thinking 面板
    if thinking_text:
        thinking_title = "Thinking"
        if is_thinking:
            thinking_title += " ..."
        display_thinking = truncate_to_lines(thinking_text, max_heights["thinking"])
        thinking_height = min(len(display_thinking.split('\n')), max_heights["thinking"]) + 2
        elements.append(Panel(
            Text(display_thinking, style="dim"),
            title=thinking_title,
            border_style="blue",
            padding=(0, 1),
            height=thinking_height,
        ))

    # Tool Calls 显示
    if tool_calls:
        tools_max_lines = max_heights["tools"]
        lines_per_tool = max(2, tools_max_lines // max(1, len(tool_calls)))

        for i, tc in enumerate(tool_calls):
            has_result = i < len(tool_results)
            tr = tool_results[i] if has_result else None

            if has_result:
                content = tr.get('content', '') if tr else ''
                if is_success(content):
                    status = ToolStatus.SUCCESS
                    style = "bold green"
                else:
                    status = ToolStatus.ERROR
                    style = "bold red"
            else:
                status = ToolStatus.RUNNING
                style = "bold yellow"

            tool_compact = format_tool_compact(tc['name'], tc.get('args'))
            tool_text = Text()
            tool_text.append(f"{status.value} ", style=style)
            tool_text.append(tool_compact, style=style)
            elements.append(tool_text)

            if has_result:
                result_elements = format_tool_result_compact(
                    tr['name'],
                    tr.get('content', ''),
                    max_lines=lines_per_tool,
                )
                elements.extend(result_elements[:lines_per_tool])
            else:
                spinner = Spinner("dots", text=" 执行中...", style="yellow")
                elements.append(spinner)

    # 工具执行后等待
    if is_processing and not is_thinking and not is_responding and not response_text:
        spinner = Spinner("dots", text=" AI 正在分析结果...", style="cyan")
        elements.append(spinner)

    # Response 面板
    if response_text:
        response_title = "Response"
        if is_responding:
            response_title += " ..."
        display_response = truncate_to_lines(response_text, max_heights["response"])
        response_height = min(len(display_response.split('\n')), max_heights["response"]) + 2
        elements.append(Panel(
            Markdown(display_response),
            title=response_title,
            border_style="green",
            padding=(0, 1),
            height=response_height,
        ))
    elif is_responding and not thinking_text:
        elements.append(Text("⏳ Generating response...", style="dim"))

    return Group(*elements) if elements else Text("⏳ Processing...", style="dim")


def print_banner():
    """打印欢迎横幅"""
    banner = """
[bold cyan]ADF Agent[/bold cyan]
[dim]Azure Data Factory 助手[/dim]

帮助你探索和管理 Azure Data Factory 资源：
- 列出和分析 Pipelines、Linked Services、Integration Runtimes
- 测试连接、启用 Interactive Authoring
- 使用 Python 分析 JSON 数据
"""
    console.print(Panel(banner, title="ADF Agent", border_style="cyan"))


def show_config_status():
    """显示配置状态"""
    config = load_adf_config()

    if config.is_configured():
        console.print(f"[green]✓[/green] ADF: {config.factory_name} (RG: {config.resource_group})")
    else:
        missing = config.missing_fields()
        console.print(f"[yellow]![/yellow] ADF config incomplete - missing: {', '.join(missing)}")
        console.print("[dim]  Agent will ask when ADF operations are needed[/dim]")


def cmd_run(prompt: str, enable_thinking: bool = True):
    """执行单次请求"""
    console.print(Panel(f"[bold cyan]User Request:[/bold cyan]\n{prompt}"))
    console.print()

    if not check_api_credentials():
        console.print("[red]Error: API credentials not set[/red]")
        console.print("Please set ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN in .env file")
        sys.exit(1)

    agent = ADFAgent(enable_thinking=enable_thinking)

    console.print("[dim]Running agent...[/dim]\n")

    try:
        state = StreamState()

        with Live(console=console, refresh_per_second=10, transient=True) as live:
            live.update(create_streaming_display(is_waiting=True))

            for event in agent.stream_events(prompt):
                event_type = state.handle_event(event)
                heights = get_display_heights(console)
                live.update(create_streaming_display(
                    **state.get_display_args(),
                    max_heights=heights,
                ))

                if event_type in ("tool_call", "tool_result"):
                    live.refresh()

        console.print()
        display_final_results(
            state,
            tool_result_max_length=1000,
            show_response_panel=True,
        )

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise


def cmd_interactive(enable_thinking: bool = True):
    """交互式对话模式"""
    print_banner()

    if not check_api_credentials():
        console.print("[red]Error: API credentials not set[/red]")
        console.print("Please set ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN in .env file")
        sys.exit(1)

    # 显示配置状态
    show_config_status()
    console.print()

    agent = ADFAgent(enable_thinking=enable_thinking)

    thinking_status = "[green]enabled[/green]" if enable_thinking else "[dim]disabled[/dim]"
    console.print(f"[dim]Extended Thinking: {thinking_status}[/dim]")
    console.print("[dim]Commands: /exit to quit, /help for examples[/dim]\n")

    thread_id = "interactive"

    # 初始化 prompt_toolkit session
    history_file = str(Path.home() / ".adf_agent_history")
    session = PromptSession(
        history=FileHistory(history_file),
        auto_suggest=AutoSuggestFromHistory(),
        enable_history_search=True,
    )

    while True:
        try:
            user_input = session.prompt(
                HTML('<ansigreen><b>You:</b></ansigreen> ')
            ).strip()

            if not user_input:
                continue

            # 特殊命令
            if user_input.lower() in ("/exit", "/quit", "/q"):
                console.print("[dim]Goodbye![/dim]")
                break

            if user_input.lower() == "/help":
                show_help()
                continue

            if user_input.lower() == "/config":
                show_config_status()
                continue

            # 运行 agent
            console.print()

            state = StreamState()

            with Live(console=console, refresh_per_second=10, transient=True) as live:
                live.update(create_streaming_display(is_waiting=True))

                for event in agent.stream_events(user_input, thread_id=thread_id):
                    event_type = state.handle_event(event)
                    heights = get_display_heights(console)
                    live.update(create_streaming_display(
                        **state.get_display_args(),
                        max_heights=heights,
                    ))

                    if event_type in ("tool_call", "tool_result"):
                        live.refresh()

            # 显示最终结果（交互模式简化显示）
            display_final_results(
                state,
                thinking_max_length=500,
                tool_result_max_length=DisplayLimits.TOOL_RESULT_FINAL,
                show_thinking=False,
                show_tools=False,
                show_response_panel=False,
            )

        except KeyboardInterrupt:
            console.print("\n[dim]Goodbye![/dim]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


def show_help():
    """显示帮助信息"""
    help_text = """
## Example Queries

**List resources:**
- 列出所有 pipeline
- 列出所有 linked service
- 列出 Snowflake 类型的 linked service

**Find relationships:**
- 哪些 pipeline 使用了 Snowflake linked service?
- 分析 linked service 类型分布

**Test connections:**
- 测试 linked service "my-snowflake" 的连接
- 启用 Integration Runtime "ir-managed" 的 interactive authoring

**Analyze data:**
- 分析 workspace/pipelines.json 中的数据
- 统计每个 pipeline 的 activity 数量

## Commands

- `/exit` - Exit the agent
- `/help` - Show this help
- `/config` - Show ADF configuration status
"""
    console.print(Markdown(help_text))


def main():
    """CLI 主入口"""
    parser = argparse.ArgumentParser(
        description="ADF Agent - Azure Data Factory 助手",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 交互式模式
  %(prog)s --interactive

  # 执行单次请求
  %(prog)s "列出所有 pipeline"

  # 禁用 thinking
  %(prog)s --no-thinking "列出所有 linked service"
""",
    )

    parser.add_argument(
        "prompt",
        nargs="?",
        help="要执行的请求",
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="进入交互式对话模式",
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="禁用 Extended Thinking",
    )
    parser.add_argument(
        "--cwd",
        type=str,
        help="设置工作目录",
    )

    args = parser.parse_args()

    # 设置工作目录
    if args.cwd:
        os.chdir(args.cwd)

    # thinking 开关
    enable_thinking = not args.no_thinking

    # 执行命令
    if args.interactive:
        cmd_interactive(enable_thinking=enable_thinking)
    elif args.prompt:
        cmd_run(args.prompt, enable_thinking=enable_thinking)
    else:
        # 默认进入交互模式
        cmd_interactive(enable_thinking=enable_thinking)


if __name__ == "__main__":
    main()
