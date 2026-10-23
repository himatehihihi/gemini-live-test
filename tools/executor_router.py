from __future__ import annotations

import importlib
from typing import Any, Dict

from .tool_registry import get_tool
from .common import not_implemented, run_logged


_DOMAIN_MODULES = {
    "system": "system",
    "windows": "windows",
    "filesystem": "filesystem",
    "browser": "browser",
    "productivity": "productivity",
    "developer": "developer",
    "ai": "ai",
    "media": "media",
    "vision": "vision",
    "communication": "communication",
    "diagnostics": "diagnostics",
}


def _get_domain_executor(domain: str):
    module_name = _DOMAIN_MODULES.get(domain)

    if not module_name:
        return None

    try:
        module = importlib.import_module(
            f".{module_name}",
            package=__package__,
        )
        return getattr(module, "execute", None)
    except Exception as exc:
        print(f"[EXECUTOR] failed to load domain={domain}: {exc}")
        return None


def execute_tool(tool: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    spec = get_tool(tool)

    if not spec:
        return not_implemented(tool, "Tool not registered")

    domain = spec.get("domain")
    handler = _get_domain_executor(domain)

    if not handler:
        return not_implemented(
            tool,
            f"No domain executor for '{domain}'",
        )

    return run_logged(
        tool,
        lambda: handler(tool, parameters or {}),
    )