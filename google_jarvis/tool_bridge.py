"""
tool_bridge.py

Smallest possible adapter between:
  - Gemini Live's function-calling mechanism (already present in jarvis.py)
  - the existing tools/ package (tool_registry, tool_validator, executor_router)

Drop this file next to jarvis.py in google_jarvis/, and make sure the
`tools` and `hands` packages (from your J.A.R.V.I.S. project) are importable
-- e.g. by adding their parent folder to sys.path, or installing them as a
local package. See the two lines you need to add to jarvis.py at the bottom
of this file's docstring.

This file does NOT talk to Gemini directly. It only:
  1. Converts your TOOL_REGISTRY into Gemini `function_declarations` dicts.
  2. Validates + executes a tool call by name, using your existing
     tool_validator + executor_router.
  3. Returns a plain dict Gemini can use as a function response.

----------------------------------------------------------------------------
Two-line integration in jarvis.py (AI_Core.__init__):

    from tool_bridge import GEMINI_TOOL_DECLARATIONS, dispatch_tool_call, KNOWN_TOOL_NAMES

    tools = [
        {'google_search': {}},
        {'code_execution': {}},
        {"function_declarations": GEMINI_TOOL_DECLARATIONS},
    ]

And inside receive_text(), in the `for fc in chunk.tool_call.function_calls:`
loop, add one branch:

    elif fc.name in KNOWN_TOOL_NAMES:
        result = dispatch_tool_call(fc.name, dict(fc.args or {}))

(Full patch shown separately -- this file is standalone and importable on
its own so you can sanity-check it first with `python tool_bridge.py`.)
----------------------------------------------------------------------------
"""

from __future__ import annotations

import sys
import os
from typing import Any, Dict, List

# --- Make sure the sibling `tools` / `hands` packages are importable ---
# Adjust this if your folder layout differs. Given:
#   gemini_live_test/
#     google_jarvis/   <- jarvis.py, tool_bridge.py (this file)
#     hands/
#     tools/
# we add gemini_live_test/ (the parent of google_jarvis/) to sys.path.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from tools.tool_registry import list_tools  # noqa: E402
from tools.tool_validator import validate_tool_call  # noqa: E402
from tools.executor_router import execute_tool  # noqa: E402


# Tools that are unsafe to expose to a live voice model without a confirmation
# step we haven't built yet (shutdown, delete, git commit, installs, tests).
# Keep the first demo boring and safe: block CONFIRM_REQUIRED / BLOCKED_UNLESS_ENABLED.
_ALLOWED_SAFETY_LEVELS = {"SAFE"}


def _json_type(param_type: str) -> str:
    """Map our internal param types to Gemini's OpenAPI-style types."""
    return {
        "string": "STRING",
        "integer": "INTEGER",
        "number": "NUMBER",
        "boolean": "BOOLEAN",
        "object": "OBJECT",
        "array": "ARRAY",
    }.get(param_type, "STRING")


def _build_declaration(spec: Dict[str, Any]) -> Dict[str, Any]:
    properties: Dict[str, Any] = {}
    required: List[str] = []
    for key, schema in spec.get("parameters", {}).items():
        properties[key] = {
            "type": _json_type(schema.get("type", "string")),
            "description": schema.get("description", ""),
        }
        if schema.get("required"):
            required.append(key)

    decl: Dict[str, Any] = {
        "name": spec["tool"],
        "description": spec.get("description", spec["tool"]),
    }
    if properties:
        decl["parameters"] = {
            "type": "OBJECT",
            "properties": properties,
        }
        if required:
            decl["parameters"]["required"] = required
    return decl


def _build_all_declarations() -> List[Dict[str, Any]]:
    declarations = []
    for spec in list_tools():
        if spec.get("safety_level", "SAFE") not in _ALLOWED_SAFETY_LEVELS:
            continue  # skip dangerous tools for the first demo
        declarations.append(_build_declaration(spec))
    return declarations


# Built once at import time. Gemini reads this list of declarations.
GEMINI_TOOL_DECLARATIONS: List[Dict[str, Any]] = _build_all_declarations()
KNOWN_TOOL_NAMES = {d["name"] for d in GEMINI_TOOL_DECLARATIONS}


def dispatch_tool_call(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate + execute a tool call coming from Gemini's fc.name / fc.args.
    Always returns a plain JSON-serializable dict -- never raises -- so it's
    safe to hand straight to session.send_tool_response().
    """
    try:
        validation = validate_tool_call(name, parameters=args, confirmed=False, allow_blocked=False)
        if not validation.ok:
            return {
                "status": "error",
                "message": "; ".join(validation.errors) or "Validation failed",
            }

        result = execute_tool(validation.tool_name, validation.normalized_parameters)

        if isinstance(result, dict):
            ok = bool(result.get("success"))
            return {
                "status": "success" if ok else "error",
                "message": result.get("message") if ok else result.get("error", "Tool failed"),
            }

        return {"status": "success", "message": str(result)}

    except Exception as exc:  # noqa: BLE001 - must never crash the Live session
        return {"status": "error", "message": f"Tool bridge exception: {exc}"}


if __name__ == "__main__":
    # Quick sanity check you can run locally:
    #   cd google_jarvis && python tool_bridge.py
    print(f"Loaded {len(GEMINI_TOOL_DECLARATIONS)} SAFE tool declarations:")
    for d in GEMINI_TOOL_DECLARATIONS:
        print(f"  - {d['name']}: {d['description']}")
