"""Tests for the hermes-tools-as-MCP server module surface.

We don't run a live MCP session in unit tests — that requires the codex
subprocess + client + an event loop. These tests pin the static
contract: the module imports, the EXPOSED_TOOLS list is sane, and the
build helper assembles a server when the SDK is present.
"""

from __future__ import annotations

import base64
import inspect
import json
import sys

from agent.transports.hermes_tools_mcp_server import (
    _project_tool_result,
    _signature_from_schema,
)


class TestSignatureFromSchema:
    """Test the JSON Schema -> Python signature conversion."""

    def test_simple_required_string_param(self):
        """A required string param becomes str with no default."""
        schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
        sig, annots = _signature_from_schema(schema)

        assert len(sig.parameters) == 1
        param = sig.parameters["query"]
        assert param.name == "query"
        assert param.kind == inspect.Parameter.KEYWORD_ONLY
        assert annots["query"] == str
        assert param.default is inspect.Parameter.empty


    def test_skip_private_params(self):
        """Params starting with '_' are excluded from the signature."""
        schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "_internal": {"type": "string"},
            },
            "required": ["query", "_internal"],
        }
        sig, annots = _signature_from_schema(schema)

        assert "_internal" not in sig.parameters
        assert "_internal" not in annots
        assert "query" in sig.parameters

    def test_all_json_types(self):
        """All JSON schema types map to correct Python types."""
        schema = {
            "type": "object",
            "properties": {
                "s": {"type": "string"},
                "i": {"type": "integer"},
                "n": {"type": "number"},
                "b": {"type": "boolean"},
                "a": {"type": "array"},
                "o": {"type": "object"},
            },
            "required": ["s", "i", "n", "b", "a", "o"],
        }
        sig, annots = _signature_from_schema(schema)

        assert annots["s"] == str
        assert annots["i"] == int
        assert annots["n"] == float
        assert annots["b"] == bool
        assert annots["a"] == list
        assert annots["o"] == dict


class TestModuleSurface:

    def test_exposed_tools_are_safe_subset(self):
        """We MUST NOT expose tools codex already has, because codex'
        own builtins are better-integrated with its sandbox + approvals.
        Specifically: no terminal/shell, no read_file/write_file, no
        patch — those are codex's built-in tools."""
        from agent.transports.hermes_tools_mcp_server import EXPOSED_TOOLS
        forbidden = {
            "terminal", "shell", "read_file", "write_file", "patch",
            "search_files", "process",
        }
        leaked = forbidden & set(EXPOSED_TOOLS)
        assert not leaked, (
            f"these tools must NOT be exposed via the codex callback "
            f"because codex has built-in equivalents: {leaked}"
        )


class TestMain:
    def test_main_returns_2_when_mcp_unavailable(self, monkeypatch):
        """When the mcp package isn't installed, main() should exit
        cleanly with code 2 and an install hint, not crash."""
        import agent.transports.hermes_tools_mcp_server as m

        def boom_build(*a, **kw):
            raise ImportError("mcp not installed")

        monkeypatch.setattr(m, "_build_server", boom_build)
        rc = m.main(["--verbose"])
        assert rc == 2

    def test_main_handles_keyboard_interrupt(self, monkeypatch):
        import agent.transports.hermes_tools_mcp_server as m

        class FakeServer:
            def run(self):
                raise KeyboardInterrupt()

        monkeypatch.setattr(m, "_build_server", lambda: FakeServer())
        rc = m.main([])
        assert rc == 0

    def test_main_returns_1_on_runtime_error(self, monkeypatch):
        import agent.transports.hermes_tools_mcp_server as m

        class CrashingServer:
            def run(self):
                raise RuntimeError("boom")

        monkeypatch.setattr(m, "_build_server", lambda: CrashingServer())
        rc = m.main([])
        assert rc == 1


def test_multimodal_result_becomes_text_plus_image(tmp_path):
    """Screenshot-producing tools return a ``_multimodal`` envelope; MCP needs text plus an
    image block instead of the dict, which failed string validation and dropped the call."""
    shot = tmp_path / "shot.png"
    shot.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
    result = _project_tool_result("browser_vision", {
        "_multimodal": True, "text_summary": "page", "meta": {"screenshot_path": str(shot)},
        "content": [{"type": "text", "text": "page"}]})
    assert isinstance(result, list) and result[0].startswith("page")
    assert type(result[1]).__name__ == "Image"
    assert _project_tool_result("t", "plain") == "plain"
    assert _project_tool_result("t", {"a": 1}) == '{"a": 1}'


def test_vision_analyze_data_url_becomes_image_block():
    """vision_analyze carries no path: its image is only the data URL in ``content`` (``meta.image_url``
    is truncated), so the image block must hold exactly the bytes and type of that data URL."""
    from tools.vision_tools import _build_native_vision_tool_result

    raw = b"\xff\xd8\xff\xe0" + bytes(range(256)) * 4
    envelope = _build_native_vision_tool_result(
        image_url="https://example.com/cat.jpg", question="what is it?",
        image_data_url="data:image/jpeg;base64," + base64.b64encode(raw).decode(), image_size_bytes=len(raw))
    text, image = _project_tool_result("vision_analyze", envelope)
    assert text == envelope["text_summary"]
    block = image.to_image_content()
    assert base64.b64decode(block.data) == raw
    assert block.mime_type == "image/jpeg"


def test_only_the_canonical_envelope_takes_the_image_branch():
    """``_multimodal`` must be ``True`` with a ``content`` list (tool_dispatch_helpers' shape);
    a stray flag is ordinary data and is JSON-serialized, not rendered as an image-less summary."""
    for stray in ({"_multimodal": True, "text_summary": "x"}, {"_multimodal": "yes", "content": []}):
        assert json.loads(_project_tool_result("t", stray)) == stray


def test_images_degrade_to_text_without_an_inline_source_or_sdk_helper(monkeypatch):
    """A remote image URL is named in the text rather than fetched, and an SDK without an
    ``Image`` helper still returns the text instead of failing the call."""
    def envelope(url):
        return {"_multimodal": True, "text_summary": "seen",
                "content": [{"type": "image_url", "image_url": {"url": url}}]}

    remote = _project_tool_result("t", envelope("https://example.com/a.png"))
    assert isinstance(remote, str) and "https://example.com/a.png" in remote

    inline = envelope("data:image/png;base64," + base64.b64encode(b"\x89PNG\r\n\x1a\n").decode())
    assert isinstance(_project_tool_result("t", inline), list)
    for sdk_module in ("mcp.server.mcpserver.utilities.types", "mcp.server.fastmcp.utilities.types"):
        monkeypatch.setitem(sys.modules, sdk_module, None)  # None in sys.modules -> ImportError
    assert _project_tool_result("t", inline) == "seen"
