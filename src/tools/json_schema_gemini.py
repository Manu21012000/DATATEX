"""Normalize JSON Schema for Google Gemini tool calling.

Gemini rejects function declarations when a ``type: array`` omits ``items``
(OpenAI is more lenient). MCP servers often emit incomplete array schemas.
"""

from __future__ import annotations

from typing import Any


def sanitize_json_schema_for_gemini(schema: Any) -> Any:
    """Recursively ensure ``array`` types define ``items`` (Gemini requirement)."""
    if schema is None:
        return schema
    if isinstance(schema, list):
        return [sanitize_json_schema_for_gemini(s) for s in schema]
    if not isinstance(schema, dict):
        return schema

    out = dict(schema)
    t = out.get("type")

    if t == "array" and "items" not in out:
        out["items"] = {"type": "string"}

    if isinstance(out.get("properties"), dict):
        out["properties"] = {
            k: sanitize_json_schema_for_gemini(v)
            for k, v in out["properties"].items()
        }

    for key in ("items", "additionalProperties", "not", "propertyNames"):
        if key in out and isinstance(out[key], dict):
            out[key] = sanitize_json_schema_for_gemini(out[key])

    if "anyOf" in out and isinstance(out["anyOf"], list):
        out["anyOf"] = [sanitize_json_schema_for_gemini(x) for x in out["anyOf"]]
    if "oneOf" in out and isinstance(out["oneOf"], list):
        out["oneOf"] = [sanitize_json_schema_for_gemini(x) for x in out["oneOf"]]

    return out
