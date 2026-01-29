"""
exec_python 运行时 helpers。

被 exec_python 通过 subprocess 导入，提供常用的数据处理工具函数。
用户代码可直接使用这些函数，无需重复定义。
"""

import json  # noqa: F401
import re  # noqa: F401
import sys  # noqa: F401
from collections import Counter, defaultdict  # noqa: F401
from pathlib import Path

__all__ = [
    # 常用标准库（用户代码直接可用）
    "json", "re", "sys", "Path", "Counter", "defaultdict",
    # 运行时变量
    "session_dir",
    # helper 函数
    "_init", "load_json", "save_json", "pretty_print",
]

# 由 _init() 设置
session_dir: Path = Path(".")


def _init(sd: str) -> None:
    """初始化 session_dir（由 exec_python 自动调用，cwd 已由 subprocess 设置）"""
    global session_dir
    session_dir = Path(sd)


def load_json(filename: str):
    """Load JSON file from session directory"""
    filepath = session_dir / filename
    if not filepath.exists():
        raise FileNotFoundError(
            f"File not found: {filename}. Use list_dir() to see available files."
        )
    return json.loads(filepath.read_text(encoding="utf-8"))


def save_json(filename: str, data) -> None:
    """Save data as JSON to session directory"""
    filepath = session_dir / filename
    filepath.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved to {filename}")


def pretty_print(data, max_items: int = 10) -> None:
    """Pretty print JSON data with truncation"""
    if isinstance(data, list) and len(data) > max_items:
        print(f"Showing first {max_items} of {len(data)} items:")
        print(json.dumps(data[:max_items], indent=2, ensure_ascii=False))
    else:
        print(json.dumps(data, indent=2, ensure_ascii=False))
