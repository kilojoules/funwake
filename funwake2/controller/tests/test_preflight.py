"""Agent-SDK preflight unit test — RUNS WITHOUT INVOKING CLAUDE.

Verifies the hard requirement: the Claude engine's preflight RAISES when
ANTHROPIC_API_KEY is present (so a raw key cannot shadow the subscription OAuth
token and silently bill the metered API), and passes only via the OAuth token.
No claude_agent_sdk / anthropic import, no network, no mutation is performed.
"""
import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from funwake2.controller.engines.claude_sdk import (
    ClaudeSDKEngine, AnthropicApiKeyPresentError)


def _clear(*keys):
    for k in keys:
        os.environ.pop(k, None)


def test_preflight_raises_when_api_key_present():
    saved = dict(os.environ)
    try:
        eng = ClaudeSDKEngine()
        # 1) key present (even truthy) -> must raise the specific error
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-should-not-be-used"
        os.environ["CLAUDE_CODE_OAUTH_TOKEN"] = "oauth-token"
        raised = False
        try:
            eng.preflight()
        except AnthropicApiKeyPresentError:
            raised = True
        assert raised, "preflight must raise when ANTHROPIC_API_KEY is set"

        # 2) EMPTY key still shadows the OAuth profile -> must raise
        os.environ["ANTHROPIC_API_KEY"] = ""
        raised = False
        try:
            eng.preflight()
        except AnthropicApiKeyPresentError:
            raised = True
        assert raised, "empty ANTHROPIC_API_KEY must also raise"

        # 3) key unset + OAuth token present -> passes (no raise)
        _clear("ANTHROPIC_API_KEY")
        os.environ["CLAUDE_CODE_OAUTH_TOKEN"] = "oauth-token"
        eng.preflight()  # should not raise

        # 4) key unset + OAuth token missing -> raises RuntimeError (not the
        #    api-key error) so a real run fails closed
        _clear("ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN")
        raised_missing = False
        try:
            eng.preflight()
        except AnthropicApiKeyPresentError:
            raise AssertionError("wrong error type for missing OAuth token")
        except RuntimeError:
            raised_missing = True
        assert raised_missing, "preflight must raise when OAuth token missing"
    finally:
        os.environ.clear()
        os.environ.update(saved)
    return True


if __name__ == "__main__":
    test_preflight_raises_when_api_key_present()
    print("PASS test_preflight (no Claude invoked)")
