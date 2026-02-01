"""
General Tool Definitions

Provides basic tools for file operations, code execution, etc.
"""

import subprocess
import sys
import re
from pathlib import Path

from langchain.tools import tool, ToolRuntime

from ..context import ADFAgentContext
from ..stream import resolve_path


@tool
def read_file(file_path: str, runtime: ToolRuntime[ADFAgentContext]) -> str:
    """
    Read the contents of a file.

    Use this to:
    - Read data files from workspace
    - View configuration files
    - Inspect any text file

    Args:
        file_path: Path to the file (absolute or relative to working directory)
    """
    path = resolve_path(file_path, runtime.context.working_directory)

    # Fallback: try session_dir for relative paths (ADF tools save files there)
    if not path.exists() and not Path(file_path).is_absolute():
        session_path = resolve_path(file_path, runtime.context.session_dir)
        if session_path.exists():
            path = session_path

    if not path.exists():
        return f"[FAILED] File not found: {file_path}"

    if not path.is_file():
        return f"[FAILED] Not a file: {file_path}"

    try:
        content = path.read_text(encoding="utf-8")
        lines = content.split("\n")

        # Add line numbers
        numbered_lines = []
        for i, line in enumerate(lines[:2000], 1):  # Limit line count
            numbered_lines.append(f"{i:4d}| {line}")

        if len(lines) > 2000:
            numbered_lines.append(f"... ({len(lines) - 2000} more lines)")

        return "[OK]\n\n" + "\n".join(numbered_lines)

    except UnicodeDecodeError:
        return f"[FAILED] Cannot read file (binary or unknown encoding): {file_path}"
    except Exception as e:
        return f"[FAILED] {str(e)}"


@tool
def write_file(file_path: str, content: str, runtime: ToolRuntime[ADFAgentContext]) -> str:
    """
    Write content to a file.

    Use this to:
    - Save analysis results
    - Create new files
    - Modify existing files

    Args:
        file_path: Path to the file (absolute or relative to working directory)
        content: Content to write to the file
    """
    path = resolve_path(file_path, runtime.context.working_directory)

    try:
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text(content, encoding="utf-8")
        return f"[OK]\n\nFile written: {path}"

    except Exception as e:
        return f"[FAILED] {str(e)}"


@tool
def glob(pattern: str, runtime: ToolRuntime[ADFAgentContext]) -> str:
    """
    Find files matching a glob pattern.

    Use this to:
    - Find files by name pattern (e.g., "**/*.json" for all JSON files)
    - List files in workspace
    - Discover available data files

    Args:
        pattern: Glob pattern (e.g., "**/*.json", "workspace/*.json", "*.md")
    """
    cwd = runtime.context.working_directory

    try:
        # Use Path.glob for matching
        matches = sorted(cwd.glob(pattern))

        if not matches:
            return f"[OK]\n\nNo files matching pattern: {pattern}"

        # 限制返回数量
        max_results = 100
        result_lines = []

        for path in matches[:max_results]:
            try:
                rel_path = path.relative_to(cwd)
                result_lines.append(str(rel_path))
            except ValueError:
                result_lines.append(str(path))

        result = "\n".join(result_lines)

        if len(matches) > max_results:
            result += f"\n... and {len(matches) - max_results} more files"

        return f"[OK]\n\n{result}"

    except Exception as e:
        return f"[FAILED] {str(e)}"


@tool
def grep(pattern: str, path: str, runtime: ToolRuntime[ADFAgentContext]) -> str:
    """
    Search for a pattern in files.

    Use this to:
    - Find text containing specific patterns
    - Search for function/class definitions
    - Locate usages of variables or imports

    Args:
        pattern: Regular expression pattern to search for
        path: File or directory path to search in (use "." for current directory)
    """
    cwd = runtime.context.working_directory
    search_path = resolve_path(path, cwd)

    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"[FAILED] Invalid regex pattern: {e}"

    results = []
    max_results = 50
    files_searched = 0

    try:
        if search_path.is_file():
            files = [search_path]
        else:
            # Search all text files, excluding common binary/hidden directories
            files = []
            for p in search_path.rglob("*"):
                if p.is_file():
                    # Exclude hidden files and common non-code directories
                    parts = p.parts
                    if any(part.startswith(".") or part in ("node_modules", "__pycache__", ".git", "venv", ".venv") for part in parts):
                        continue
                    files.append(p)

        for file_path in files:
            if len(results) >= max_results:
                break

            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                lines = content.split("\n")
                files_searched += 1

                for line_num, line in enumerate(lines, 1):
                    if regex.search(line):
                        try:
                            rel_path = file_path.relative_to(cwd)
                        except ValueError:
                            rel_path = file_path
                        results.append(f"{rel_path}:{line_num}: {line.strip()[:100]}")

                        if len(results) >= max_results:
                            break

            except (UnicodeDecodeError, PermissionError, IsADirectoryError):
                continue

        if not results:
            return f"[OK]\n\nNo matches found for pattern: {pattern} (searched {files_searched} files)"

        output = "\n".join(results)
        if len(results) >= max_results:
            output += f"\n... (truncated, showing first {max_results} matches)"

        return f"[OK]\n\n{output}"

    except Exception as e:
        return f"[FAILED] {str(e)}"


@tool
def list_dir(path: str, runtime: ToolRuntime[ADFAgentContext]) -> str:
    """
    List contents of a directory.

    Use this to:
    - Explore directory structure
    - See what files exist in workspace
    - Check if files/folders exist

    Args:
        path: Directory path (use "." for current directory, "workspace" for workspace)
    """
    dir_path = resolve_path(path, runtime.context.working_directory)

    if not dir_path.exists():
        return f"[FAILED] Directory not found: {path}"

    if not dir_path.is_dir():
        return f"[FAILED] Not a directory: {path}"

    try:
        entries = sorted(dir_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))

        result_lines = []
        for entry in entries[:100]:  # Limit count
            if entry.is_dir():
                result_lines.append(f"[DIR]  {entry.name}/")
            else:
                # Show file size
                size = entry.stat().st_size
                if size < 1024:
                    size_str = f"{size}B"
                elif size < 1024 * 1024:
                    size_str = f"{size // 1024}KB"
                else:
                    size_str = f"{size // (1024 * 1024)}MB"
                result_lines.append(f"[FILE] {entry.name} ({size_str})")

        if len(entries) > 100:
            result_lines.append(f"... and {len(entries) - 100} more entries")

        return f"[OK]\n\n{chr(10).join(result_lines)}"

    except PermissionError:
        return f"[FAILED] Permission denied: {path}"
    except Exception as e:
        return f"[FAILED] {str(e)}"


_EXEC_RUNTIME_SRC = Path(__file__).with_name("_exec_runtime.py")


def _ensure_runtime(session_dir: Path) -> None:
    """Deploy _exec_runtime.py to session_dir on first call, skip on subsequent calls."""
    dest = session_dir / "_exec_runtime.py"
    if not dest.exists():
        dest.write_text(_EXEC_RUNTIME_SRC.read_text(encoding="utf-8"), encoding="utf-8")


@tool
def exec_python(code: str, runtime: ToolRuntime[ADFAgentContext]) -> str:
    """
    Execute Python code for data analysis.

    IMPORTANT FOR ERROR HANDLING:
    - If execution fails, the error message and traceback will be returned
    - You should analyze the error, fix your code, and try again
    - Maximum 3 retry attempts for the same task

    Pre-loaded helpers:
    - load_json(filename): Load JSON from workspace
    - save_json(filename, data): Save JSON to workspace
    - workspace: Path to workspace directory

    Common patterns:
    - Load data: `data = load_json("pipelines.json")`
    - Filter: `results = [x for x in data if condition]`
    - Search: `matches = [x for x in data if "keyword" in json.dumps(x)]`
    - Save results: `save_json("results.json", results)`
    - Print for output: `print(json.dumps(results, indent=2))`

    Args:
        code: Python code to execute (use print() for output)
    """
    session_dir = runtime.context.session_dir

    # Write shared helpers to session_dir on first call
    _ensure_runtime(session_dir)

    setup_code = (
        f"from _exec_runtime import *\n"
        f"_init({str(session_dir)!r})\n"
    )

    full_code = setup_code + "\n" + code

    try:
        result = subprocess.run(
            [sys.executable, "-c", full_code],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(session_dir),
        )

        # Return detailed error info to help the agent fix issues
        if result.returncode != 0:
            output = f"""[FAILED] Exit code: {result.returncode}

--- stdout ---
{result.stdout.rstrip() or '(empty)'}

--- stderr (traceback) ---
{result.stderr.rstrip()}

Hint: Analyze the error above, fix your code, and try exec_python again.
Common fixes:
- KeyError: Use read_file to check JSON structure first
- FileNotFoundError: Use list_dir() to see available files
- SyntaxError: Double-check Python syntax
"""
            # Save full script to session directory (including helper functions)
            runtime.context.save_script(full_code, output, success=False)
            return output

        output = result.stdout.rstrip()
        if not output:
            output_msg = "[OK]\n\n(no output - use print() to display results)"
        else:
            output_msg = f"[OK]\n\n{output}"

        # 保存完整脚本到 session 目录（包含 helper 函数）
        runtime.context.save_script(full_code, output_msg, success=True)
        return output_msg

    except subprocess.TimeoutExpired:
        error_msg = "[FAILED] Execution timed out after 60 seconds."
        runtime.context.save_script(full_code, error_msg, success=False)
        return error_msg
    except Exception as e:
        error_msg = f"[FAILED] {str(e)}"
        runtime.context.save_script(full_code, error_msg, success=False)
        return error_msg


@tool
def resolve_adf_target(domain: str, environment: str,
                       runtime: ToolRuntime[ADFAgentContext]) -> str:
    """
    Switch the active ADF target to a specific domain and environment.

    Call this BEFORE any ADF operation when:
    - The user first specifies which ADF instance to work with
    - The user wants to switch to a different ADF target

    Both parameters are required. If the user does not provide both,
    ask for clarification instead of calling this tool.

    Args:
        domain: Domain name (e.g. "sales", "hr", "personal")
        environment: Environment name (e.g. "dev", "qa", "prod")
    """
    from ..context import ADF_TARGETS

    domain_targets = ADF_TARGETS.get(domain)
    if domain_targets is None:
        available = "\n".join(
            f"  - {d}: {', '.join(envs.keys())}"
            for d, envs in ADF_TARGETS.items()
        )
        return (
            f"[FAILED] Unknown domain: {domain}\n\n"
            f"Available targets (domain → environments):\n{available}"
        )

    config = domain_targets.get(environment)
    if config is None:
        return (
            f"[FAILED] Unknown environment '{environment}' for domain '{domain}'. "
            f"Available: {', '.join(domain_targets.keys())}"
        )

    old = runtime.context.adf_config
    old_label = f"{old.factory_name}" if old.is_configured() else None
    runtime.context.adf_config = config

    new_label = f"{domain}/{environment} ({config.factory_name})"
    if old_label:
        return f"[OK] Switched ADF target: {old_label} → {new_label}"
    return f"[OK] ADF target set: {new_label}"


# Export all general tools
GENERAL_TOOLS = [
    read_file,
    write_file,
    glob,
    grep,
    list_dir,
    exec_python,
    resolve_adf_target,
]
