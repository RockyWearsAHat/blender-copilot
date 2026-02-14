"""
OAuth helpers for the AI House Generator addon.

- **GitHub Copilot**: Full OAuth Device Flow (RFC 8628).
  The user sees a one-time code, opens a browser, authorises, and the
  addon receives a token automatically.

- **OpenAI / Anthropic**: These providers do not offer OAuth for API
  access, so we open the user's browser to their API-key management
  page and let them paste the key back.
"""

import json
import urllib.request
import urllib.parse
import urllib.error
import ssl
import webbrowser

# ═══════════════════════════════════════════════════════════════════════════
# GitHub Device Flow  (RFC 8628)
# ═══════════════════════════════════════════════════════════════════════════

GITHUB_DEVICE_CODE_URL = "https://github.com/login/device/code"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"

# Default GitHub OAuth App client ID.
# Replace this with your own if you register a GitHub OAuth App at
# https://github.com/settings/applications/new
# (Set "Device flow" to Enabled, no callback URL required.)
DEFAULT_GITHUB_CLIENT_ID = ""


def _post_form(url: str, params: dict, extra_headers: dict | None = None) -> dict:
    """POST an application/x-www-form-urlencoded request, return JSON."""
    data = urllib.parse.urlencode(params).encode("utf-8")
    headers = {"Accept": "application/json"}
    if extra_headers:
        headers.update(extra_headers)
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def github_request_device_code(client_id: str) -> dict:
    """Step 1: Ask GitHub for a device + user code.

    Returns dict with keys:
        device_code, user_code, verification_uri, expires_in, interval
    """
    if not client_id:
        raise RuntimeError(
            "GitHub OAuth Client ID is not set.\n"
            "Register a free GitHub OAuth App at:\n"
            "  https://github.com/settings/applications/new\n"
            "(Enable 'Device flow', no callback URL needed.)\n"
            "Then paste the Client ID into the addon preferences."
        )
    return _post_form(GITHUB_DEVICE_CODE_URL, {
        "client_id": client_id,
        "scope": "copilot",
    })


def github_poll_for_token(client_id: str, device_code: str) -> dict:
    """Step 3: Poll GitHub for the access token.

    Returns dict — check for:
      "access_token"  → success
      "error"         → "authorization_pending" | "slow_down" | "expired_token" | …
    """
    return _post_form(GITHUB_TOKEN_URL, {
        "client_id": client_id,
        "device_code": device_code,
        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
    })


# ═══════════════════════════════════════════════════════════════════════════
# Browser helpers for OpenAI / Anthropic
# ═══════════════════════════════════════════════════════════════════════════

OPENAI_KEYS_URL = "https://platform.openai.com/api-keys"
ANTHROPIC_KEYS_URL = "https://console.anthropic.com/settings/keys"


def open_openai_key_page():
    """Open the user's browser to the OpenAI API keys page."""
    webbrowser.open(OPENAI_KEYS_URL)


def open_anthropic_key_page():
    """Open the user's browser to the Anthropic API keys page."""
    webbrowser.open(ANTHROPIC_KEYS_URL)


def open_github_apps_page():
    """Open the user's browser to create a new GitHub OAuth App."""
    webbrowser.open("https://github.com/settings/applications/new")
