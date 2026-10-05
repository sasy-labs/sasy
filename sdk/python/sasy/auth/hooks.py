"""Authentication hooks for SDK requests.

Implement AuthHook to supply credentials from another authentication provider.
Configure a hook once before enabling instrumentation:

    import os
    import sasy
    from sasy.auth.hooks import OIDCAuthHook

    sasy.configure(
        sasy_url=os.environ["SASY_URL"],
        auth_hook=OIDCAuthHook(
            token_url="https://identity.example.com/realms/agents/protocol/openid-connect/token",
            client_id="agent-client",
            client_secret=os.environ["OIDC_CLIENT_SECRET"],
        ),
    )
    sasy.instrument()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from sasy._headers import API_KEY, AUTHORIZATION

if TYPE_CHECKING:
    import httpx


class AuthHook(ABC):
    """
    Base class for authentication hooks.

    Auth hooks provide credentials to be sent with requests to the reference monitor.
    Implement this class to integrate with your authentication provider.
    """

    @abstractmethod
    def get_metadata(self) -> list[tuple[str, str]]:
        """
        Get gRPC metadata (headers) to include with the reference monitor request.

        Returns:
            List of (key, value) tuples to add as gRPC metadata.
            Common keys: "authorization" for Bearer tokens, "x-api-key" for API keys.
        """
        pass


class NoAuthHook(AuthHook):
    """No authentication - returns empty metadata."""

    def get_metadata(self) -> list[tuple[str, str]]:
        return []


class StaticTokenAuthHook(AuthHook):
    """
    Static token authentication.

    Provides a fixed Bearer token for all requests. Useful for testing
    or when tokens are managed externally.
    """

    def __init__(self, token: str, header_key: str = AUTHORIZATION):
        """
        Args:
            token: The bearer token (without "Bearer " prefix)
            header_key: Header key to use (default: "authorization")
        """
        self.token = token
        self.header_key = header_key

    def get_metadata(self) -> list[tuple[str, str]]:
        return [(self.header_key, f"Bearer {self.token}")]


class APIKeyAuthHook(AuthHook):
    """
    API key authentication.

    Provides a fixed API key for all requests.
    """

    def __init__(self, api_key: str, header_key: str = API_KEY):
        """
        Args:
            api_key: The API key
            header_key: Header key to use (default: "x-api-key")
        """
        self.api_key = api_key
        self.header_key = header_key

    def get_metadata(self) -> list[tuple[str, str]]:
        return [(self.header_key, self.api_key)]


class CallableAuthHook(AuthHook):
    """
    Callable-based authentication hook.

    Wraps a callable that returns metadata, allowing for custom
    authentication logic or dynamic token refresh.
    """

    def __init__(self, get_metadata_fn: Callable[[], list[tuple[str, str]]]):
        """
        Args:
            get_metadata_fn: Callable that returns list of (key, value) metadata tuples
        """
        self._get_metadata = get_metadata_fn

    def get_metadata(self) -> list[tuple[str, str]]:
        return self._get_metadata()


class EnvironmentAuthHook(AuthHook):
    """
    Environment variable-based authentication.

    Reads token or API key from environment variables on each request.
    """

    def __init__(
        self,
        env_var: str = "REFERENCE_MONITOR_TOKEN",
        header_key: str = AUTHORIZATION,
        token_prefix: str = "Bearer ",
    ):
        """
        Args:
            env_var: Environment variable containing the token/key
            header_key: Header key to use (e.g., "authorization", "x-api-key")
            token_prefix: Prefix to add before the token (e.g., "Bearer ")
        """
        self.env_var = env_var
        self.header_key = header_key
        self.token_prefix = token_prefix

    def get_metadata(self) -> list[tuple[str, str]]:
        import os

        token = os.environ.get(self.env_var)
        if token:
            return [(self.header_key, f"{self.token_prefix}{token}")]
        return []


class OIDCAuthHook(AuthHook):
    """
    OpenID Connect / OAuth2 authentication hook.

    Automatically obtains and refreshes tokens from any OIDC-compliant provider
    (Keycloak, Okta, Auth0, Azure AD, Google, etc.).

    Supports:
    - Client Credentials Grant (service-to-service, no user)
    - Resource Owner Password Grant (with username/password)
    """

    def __init__(
        self,
        token_url: str,
        client_id: str,
        client_secret: str | None = None,
        username: str | None = None,
        password: str | None = None,
        scope: str | None = None,
        extra_params: dict[str, str] | None = None,
        ca_cert: str | None = None,
    ):
        """
        Initialize OIDC auth hook.

        For service accounts (no user), provide client_id and client_secret only.
        For user authentication, also provide username and password.

        Args:
            token_url: OAuth2 token endpoint URL
                - Keycloak: http://host/realms/{realm}/protocol/openid-connect/token
                - Okta: https://{domain}/oauth2/default/v1/token
                - Auth0: https://{domain}/oauth/token
                - Azure AD: https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token
            client_id: OAuth2 client ID
            client_secret: OAuth2 client secret (required for confidential clients)
            username: Username for password grant (optional)
            password: Password for password grant (optional)
            scope: OAuth2 scopes (space-separated, e.g., "openid profile email")
            extra_params: Additional parameters to include in token request
            ca_cert: Path to CA certificate file for TLS verification (optional)
        """
        self.token_url = token_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.username = username
        self.password = password
        self.scope = scope
        self.extra_params = extra_params or {}
        self.ca_cert = ca_cert
        self._token: str | None = None
        self._token_expires_at: float = 0

    def _get_http_client(self) -> httpx.Client:
        """Create an HTTP client with optional CA certificate."""
        import httpx

        if self.ca_cert:
            return httpx.Client(verify=self.ca_cert)
        return httpx.Client()

    def _refresh_token(self) -> None:
        """Obtain or refresh the access token."""
        import time

        if self.username and self.password:
            # Resource Owner Password Grant
            data = {
                "grant_type": "password",
                "client_id": self.client_id,
                "username": self.username,
                "password": self.password,
            }
        else:
            # Client Credentials Grant
            data = {
                "grant_type": "client_credentials",
                "client_id": self.client_id,
            }

        if self.client_secret:
            data["client_secret"] = self.client_secret

        if self.scope:
            data["scope"] = self.scope

        # Add any extra parameters
        data.update(self.extra_params)

        with self._get_http_client() as client:
            response = client.post(self.token_url, data=data)
            response.raise_for_status()

        token_data = response.json()
        self._token = token_data["access_token"]
        # Refresh 30 seconds before expiry
        self._token_expires_at = time.time() + token_data.get("expires_in", 300) - 30

    def get_metadata(self) -> list[tuple[str, str]]:
        import time

        # Refresh token if expired or not yet obtained
        if self._token is None or time.time() >= self._token_expires_at:
            self._refresh_token()

        return [(AUTHORIZATION, f"Bearer {self._token}")]


# Keycloak-specific name for the generic OIDC hook.
KeycloakAuthHook = OIDCAuthHook


def keycloak_auth_hook(
    server_url: str = "http://localhost:8080",
    realm: str = "sasy",
    client_id: str = "sasy-agents",
    client_secret: str | None = None,
    username: str | None = None,
    password: str | None = None,
    ca_cert: str | None = None,
) -> OIDCAuthHook:
    """
    Create an OIDC auth hook configured for Keycloak.

    This is a convenience factory that constructs the correct token URL
    from Keycloak server URL and realm name.

    The ``realm`` and ``client_id`` defaults are placeholder names, not a
    realm or client that exists anywhere: pass the realm and client your
    Keycloak is configured with.

    Args:
        server_url: Keycloak server URL (e.g., "https://localhost:8443")
        realm: Keycloak realm name
        client_id: OAuth2 client ID
        client_secret: OAuth2 client secret
        username: Username for password grant (optional)
        password: Password for password grant (optional)
        ca_cert: Path to CA certificate file for TLS verification (optional)

    Returns:
        Configured OIDCAuthHook
    """
    token_url = f"{server_url.rstrip('/')}/realms/{realm}/protocol/openid-connect/token"
    return OIDCAuthHook(
        token_url=token_url,
        client_id=client_id,
        client_secret=client_secret,
        username=username,
        password=password,
        ca_cert=ca_cert,
    )


def okta_auth_hook(
    domain: str,
    client_id: str,
    client_secret: str | None = None,
    username: str | None = None,
    password: str | None = None,
    authorization_server: str = "default",
    scope: str = "openid",
    ca_cert: str | None = None,
) -> OIDCAuthHook:
    """
    Create an OIDC auth hook configured for Okta.

    Args:
        domain: Okta domain (e.g., "dev-123456.okta.com")
        client_id: OAuth2 client ID
        client_secret: OAuth2 client secret
        username: Username for password grant (optional)
        password: Password for password grant (optional)
        authorization_server: Okta authorization server ID (default: "default")
        scope: OAuth2 scopes (default: "openid")
        ca_cert: Path to CA certificate file for TLS verification (optional)

    Returns:
        Configured OIDCAuthHook
    """
    token_url = f"https://{domain}/oauth2/{authorization_server}/v1/token"
    return OIDCAuthHook(
        token_url=token_url,
        client_id=client_id,
        client_secret=client_secret,
        username=username,
        password=password,
        scope=scope,
        ca_cert=ca_cert,
    )


def azure_ad_auth_hook(
    tenant_id: str,
    client_id: str,
    client_secret: str | None = None,
    username: str | None = None,
    password: str | None = None,
    scope: str = "openid profile email",
    ca_cert: str | None = None,
) -> OIDCAuthHook:
    """
    Create an OIDC auth hook configured for Azure AD / Entra ID.

    Args:
        tenant_id: Azure AD tenant ID
        client_id: OAuth2 client ID (Application ID)
        client_secret: OAuth2 client secret
        username: Username for password grant (optional)
        password: Password for password grant (optional)
        scope: OAuth2 scopes (default: "openid profile email")
        ca_cert: Path to CA certificate file for TLS verification (optional)

    Returns:
        Configured OIDCAuthHook
    """
    token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
    return OIDCAuthHook(
        token_url=token_url,
        client_id=client_id,
        client_secret=client_secret,
        username=username,
        password=password,
        scope=scope,
        ca_cert=ca_cert,
    )


class DeviceAuthHook(AuthHook):
    """
    OAuth2 Device Authorization Grant (RFC 8628).

    Best for CLI tools and headless environments. The user authenticates
    on a separate device (phone, laptop) by visiting a URL and entering a code.

    Flow:
    1. App requests device code from authorization server
    2. App displays URL and code to user
    3. User visits URL on any device and enters the code
    4. User authenticates in browser
    5. App polls token endpoint until authentication completes
    """

    def __init__(
        self,
        device_authorization_url: str,
        token_url: str,
        client_id: str,
        client_secret: str | None = None,
        scope: str | None = None,
        poll_interval: int = 5,
        timeout: int = 300,
        ca_cert: str | None = None,
    ):
        """
        Initialize device auth hook.

        Args:
            device_authorization_url: Device authorization endpoint URL
                - Keycloak: http://host/realms/{realm}/protocol/openid-connect/auth/device
                - Okta: https://{domain}/oauth2/default/v1/device/authorize
                - Azure AD: https://login.microsoftonline.com/{tenant}/oauth2/v2.0/devicecode
                - Google: https://oauth2.googleapis.com/device/code
            token_url: Token endpoint URL
            client_id: OAuth2 client ID
            client_secret: OAuth2 client secret (optional for public clients)
            scope: OAuth2 scopes (space-separated)
            poll_interval: Seconds between token polling attempts (default: 5)
            timeout: Max seconds to wait for user authentication (default: 300)
            ca_cert: Path to CA certificate file for TLS verification (optional)
        """
        self.device_authorization_url = device_authorization_url
        self.token_url = token_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.scope = scope
        self.poll_interval = poll_interval
        self.timeout = timeout
        self.ca_cert = ca_cert
        self._token: str | None = None
        self._refresh_token: str | None = None
        self._token_expires_at: float = 0

    def _get_http_client(self) -> httpx.Client:
        """Create an HTTP client with optional CA certificate."""
        import httpx

        if self.ca_cert:
            return httpx.Client(verify=self.ca_cert)
        return httpx.Client()

    def _request_device_code(self) -> dict:
        """Request a device code from the authorization server."""
        data = {"client_id": self.client_id}
        if self.scope:
            data["scope"] = self.scope

        with self._get_http_client() as client:
            response = client.post(self.device_authorization_url, data=data)
            if response.status_code != 200:
                # Log the actual error from the authorization server
                try:
                    error_body = response.json()
                    error_msg = error_body.get("error_description", error_body.get("error", response.text))
                except Exception:
                    error_msg = response.text
                raise RuntimeError(
                    f"Device authorization failed ({response.status_code}): {error_msg}\n"
                    f"Ensure the client '{self.client_id}' has OAuth 2.0 Device Authorization Grant enabled."
                )
            return response.json()

    def _poll_for_token(self, device_code: str, interval: int) -> dict:
        """Poll the token endpoint until the user completes authentication."""
        import time

        start_time = time.time()

        with self._get_http_client() as client:
            while time.time() - start_time < self.timeout:
                data = {
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    "device_code": device_code,
                    "client_id": self.client_id,
                }
                if self.client_secret:
                    data["client_secret"] = self.client_secret

                response = client.post(self.token_url, data=data)

                if response.status_code == 200:
                    return response.json()

                error_data = response.json()
                error = error_data.get("error")

                if error == "authorization_pending":
                    # User hasn't completed auth yet, keep polling
                    time.sleep(interval)
                elif error == "slow_down":
                    # Server wants us to slow down
                    interval += 5
                    time.sleep(interval)
                elif error == "expired_token":
                    raise TimeoutError("Device code expired. Please try again.")
                elif error == "access_denied":
                    raise PermissionError("User denied the authorization request.")
                else:
                    response.raise_for_status()

        raise TimeoutError(f"Authentication timed out after {self.timeout} seconds.")

    def _do_device_flow(self) -> None:
        """Execute the device authorization flow."""
        import time

        # Request device code
        device_response = self._request_device_code()

        device_code = device_response["device_code"]
        user_code = device_response["user_code"]
        verification_uri = device_response.get(
            "verification_uri", device_response.get("verification_url")
        )
        verification_uri_complete = device_response.get("verification_uri_complete")
        interval = device_response.get("interval", self.poll_interval)

        # Display instructions to user
        print("\n" + "=" * 60)
        print("AUTHENTICATION REQUIRED")
        print("=" * 60)
        if verification_uri_complete:
            print(f"Visit: {verification_uri_complete}")
        else:
            print(f"Visit: {verification_uri}")
            print(f"Enter code: {user_code}")
        print("=" * 60 + "\n")

        # Poll for token
        token_response = self._poll_for_token(device_code, interval)

        self._token = token_response["access_token"]
        self._refresh_token = token_response.get("refresh_token")
        self._token_expires_at = time.time() + token_response.get("expires_in", 300) - 30

    def _do_refresh(self) -> bool:
        """Attempt to refresh the access token using refresh token."""
        import time

        if not self._refresh_token:
            return False

        data = {
            "grant_type": "refresh_token",
            "refresh_token": self._refresh_token,
            "client_id": self.client_id,
        }
        if self.client_secret:
            data["client_secret"] = self.client_secret

        try:
            with self._get_http_client() as client:
                response = client.post(self.token_url, data=data)
                if response.status_code == 200:
                    token_response = response.json()
                    self._token = token_response["access_token"]
                    self._refresh_token = token_response.get("refresh_token", self._refresh_token)
                    self._token_expires_at = time.time() + token_response.get("expires_in", 300) - 30
                    return True
        except Exception:
            pass
        return False

    def get_metadata(self) -> list[tuple[str, str]]:
        import time

        # Check if token needs refresh
        if self._token is None or time.time() >= self._token_expires_at:
            # Try refresh first, fall back to full device flow
            if not self._do_refresh():
                self._do_device_flow()

        return [(AUTHORIZATION, f"Bearer {self._token}")]


class BrowserAuthHook(AuthHook):
    """
    OAuth2 Authorization Code Grant with PKCE via local callback server.

    Opens the system browser for authentication and receives the callback
    on a local HTTP server. Best for desktop applications where a browser
    is available.

    Flow:
    1. Generate PKCE code verifier and challenge
    2. Start local HTTP server for callback
    3. Open browser to authorization URL
    4. User authenticates in browser
    5. Authorization server redirects to local server with code
    6. Exchange code for tokens
    """

    def __init__(
        self,
        authorization_url: str,
        token_url: str,
        client_id: str,
        client_secret: str | None = None,
        scope: str | None = None,
        redirect_port: int = 8400,
        timeout: int = 120,
        ca_cert: str | None = None,
    ):
        """
        Initialize browser auth hook.

        Args:
            authorization_url: Authorization endpoint URL
                - Keycloak: http://host/realms/{realm}/protocol/openid-connect/auth
                - Okta: https://{domain}/oauth2/default/v1/authorize
                - Azure AD: https://login.microsoftonline.com/{tenant}/oauth2/v2.0/authorize
                - Google: https://accounts.google.com/o/oauth2/v2/auth
            token_url: Token endpoint URL
            client_id: OAuth2 client ID
            client_secret: OAuth2 client secret (optional for public clients with PKCE)
            scope: OAuth2 scopes (space-separated)
            redirect_port: Local port for callback server (default: 8400)
            timeout: Max seconds to wait for browser callback (default: 120)
            ca_cert: Path to CA certificate file for TLS verification (optional)
        """
        self.authorization_url = authorization_url
        self.token_url = token_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.scope = scope
        self.redirect_port = redirect_port
        self.timeout = timeout
        self.ca_cert = ca_cert
        self._token: str | None = None
        self._refresh_token: str | None = None
        self._token_expires_at: float = 0

    def _get_http_client(self) -> httpx.Client:
        """Create an HTTP client with optional CA certificate."""
        import httpx

        if self.ca_cert:
            return httpx.Client(verify=self.ca_cert)
        return httpx.Client()

    def _generate_pkce(self) -> tuple[str, str]:
        """Generate PKCE code verifier and challenge."""
        import base64
        import hashlib
        import secrets

        # Generate code verifier (43-128 characters)
        code_verifier = secrets.token_urlsafe(32)

        # Generate code challenge (S256)
        digest = hashlib.sha256(code_verifier.encode()).digest()
        code_challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()

        return code_verifier, code_challenge

    def _start_callback_server(self) -> tuple[Any, str]:
        """Start local HTTP server and return it with the redirect URI."""
        import http.server
        import threading

        class _CallbackHandler(http.server.BaseHTTPRequestHandler):
            authorization_code: str | None = None
            error: str | None = None
            state: str | None = None

            def do_GET(self):
                from urllib.parse import parse_qs, urlparse

                parsed = urlparse(self.path)
                params = parse_qs(parsed.query)

                if "code" in params:
                    _CallbackHandler.authorization_code = params["code"][0]
                    _CallbackHandler.state = params.get("state", [None])[0]
                    self.send_response(200)
                    self.send_header("Content-type", "text/html")
                    self.end_headers()
                    self.wfile.write(b"""
                        <html><body style="font-family: sans-serif; text-align: center; padding-top: 50px;">
                        <h1>Authentication Successful</h1>
                        <p>You can close this window and return to the application.</p>
                        </body></html>
                    """)
                elif "error" in params:
                    _CallbackHandler.error = params.get("error_description", params["error"])[0]
                    self.send_response(400)
                    self.send_header("Content-type", "text/html")
                    self.end_headers()
                    self.wfile.write(f"""
                        <html><body style="font-family: sans-serif; text-align: center; padding-top: 50px;">
                        <h1>Authentication Failed</h1>
                        <p>{_CallbackHandler.error}</p>
                        </body></html>
                    """.encode())
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format, *args):
                pass  # Suppress logging

        class _CallbackServer(http.server.HTTPServer):
            def __init__(self, port: int):
                super().__init__(("127.0.0.1", port), _CallbackHandler)
                self.handler_class = _CallbackHandler

        server = _CallbackServer(self.redirect_port)
        redirect_uri = f"http://127.0.0.1:{self.redirect_port}/callback"

        # Start server in background thread
        thread = threading.Thread(target=server.handle_request, daemon=True)
        thread.start()

        return server, redirect_uri

    def _build_authorization_url(
        self, redirect_uri: str, code_challenge: str, state: str
    ) -> str:
        """Build the full authorization URL."""
        from urllib.parse import urlencode

        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": redirect_uri,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "state": state,
        }
        if self.scope:
            params["scope"] = self.scope

        return f"{self.authorization_url}?{urlencode(params)}"

    def _exchange_code(self, code: str, redirect_uri: str, code_verifier: str) -> dict:
        """Exchange authorization code for tokens."""
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": self.client_id,
            "code_verifier": code_verifier,
        }
        if self.client_secret:
            data["client_secret"] = self.client_secret

        with self._get_http_client() as client:
            response = client.post(self.token_url, data=data)
            response.raise_for_status()
            return response.json()

    def _do_browser_flow(self) -> None:
        """Execute the browser-based authorization flow."""
        import secrets
        import time
        import webbrowser

        # Generate PKCE and state
        code_verifier, code_challenge = self._generate_pkce()
        state = secrets.token_urlsafe(16)

        # Start callback server
        server, redirect_uri = self._start_callback_server()

        # Build and open authorization URL
        auth_url = self._build_authorization_url(redirect_uri, code_challenge, state)

        print("\n" + "=" * 60)
        print("AUTHENTICATION REQUIRED")
        print("=" * 60)
        print("Opening browser for authentication...")
        print(f"If browser doesn't open, visit:\n{auth_url}")
        print("=" * 60 + "\n")

        webbrowser.open(auth_url)

        # Wait for callback
        start_time = time.time()
        handler = server.handler_class

        while time.time() - start_time < self.timeout:
            if handler.authorization_code is not None:
                break
            if handler.error is not None:
                raise PermissionError(f"Authentication failed: {handler.error}")
            time.sleep(0.1)
        else:
            raise TimeoutError(f"Authentication timed out after {self.timeout} seconds.")

        # Verify state
        if handler.state != state:
            raise ValueError("State mismatch - possible CSRF attack")

        # Exchange code for tokens
        token_response = self._exchange_code(
            handler.authorization_code, redirect_uri, code_verifier
        )

        self._token = token_response["access_token"]
        self._refresh_token = token_response.get("refresh_token")
        self._token_expires_at = time.time() + token_response.get("expires_in", 300) - 30

    def _do_refresh(self) -> bool:
        """Attempt to refresh the access token using refresh token."""
        import time

        if not self._refresh_token:
            return False

        data = {
            "grant_type": "refresh_token",
            "refresh_token": self._refresh_token,
            "client_id": self.client_id,
        }
        if self.client_secret:
            data["client_secret"] = self.client_secret

        try:
            with self._get_http_client() as client:
                response = client.post(self.token_url, data=data)
                if response.status_code == 200:
                    token_response = response.json()
                    self._token = token_response["access_token"]
                    self._refresh_token = token_response.get("refresh_token", self._refresh_token)
                    self._token_expires_at = time.time() + token_response.get("expires_in", 300) - 30
                    return True
        except Exception:
            pass
        return False

    def get_metadata(self) -> list[tuple[str, str]]:
        import time

        # Check if token needs refresh
        if self._token is None or time.time() >= self._token_expires_at:
            # Try refresh first, fall back to full browser flow
            if not self._do_refresh():
                self._do_browser_flow()

        return [(AUTHORIZATION, f"Bearer {self._token}")]


# Factory functions for interactive flows


def device_auth_hook_keycloak(
    server_url: str = "http://localhost:8080",
    realm: str = "sasy",
    client_id: str = "sasy-cli",
    scope: str = "openid",
    ca_cert: str | None = None,
) -> DeviceAuthHook:
    """Create a device auth hook configured for Keycloak.

    The ``realm`` and ``client_id`` defaults are placeholder names: pass the
    realm and client your Keycloak is configured with.
    """
    base = f"{server_url.rstrip('/')}/realms/{realm}/protocol/openid-connect"
    return DeviceAuthHook(
        device_authorization_url=f"{base}/auth/device",
        token_url=f"{base}/token",
        client_id=client_id,
        scope=scope,
        ca_cert=ca_cert,
    )


def device_auth_hook_azure(
    tenant_id: str,
    client_id: str,
    scope: str = "openid profile email",
    ca_cert: str | None = None,
) -> DeviceAuthHook:
    """Create a device auth hook configured for Azure AD / Entra ID."""
    return DeviceAuthHook(
        device_authorization_url=f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/devicecode",
        token_url=f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token",
        client_id=client_id,
        scope=scope,
        ca_cert=ca_cert,
    )


def browser_auth_hook_keycloak(
    server_url: str = "http://localhost:8080",
    realm: str = "sasy",
    client_id: str = "sasy-cli",
    scope: str = "openid",
    redirect_port: int = 8400,
    ca_cert: str | None = None,
) -> BrowserAuthHook:
    """Create a browser auth hook configured for Keycloak.

    The ``realm`` and ``client_id`` defaults are placeholder names: pass the
    realm and client your Keycloak is configured with.
    """
    base = f"{server_url.rstrip('/')}/realms/{realm}/protocol/openid-connect"
    return BrowserAuthHook(
        authorization_url=f"{base}/auth",
        token_url=f"{base}/token",
        client_id=client_id,
        scope=scope,
        redirect_port=redirect_port,
        ca_cert=ca_cert,
    )


def browser_auth_hook_azure(
    tenant_id: str,
    client_id: str,
    scope: str = "openid profile email",
    redirect_port: int = 8400,
    ca_cert: str | None = None,
) -> BrowserAuthHook:
    """Create a browser auth hook configured for Azure AD / Entra ID."""
    base = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0"
    return BrowserAuthHook(
        authorization_url=f"{base}/authorize",
        token_url=f"{base}/token",
        client_id=client_id,
        scope=scope,
        redirect_port=redirect_port,
        ca_cert=ca_cert,
    )


def browser_auth_hook_google(
    client_id: str,
    client_secret: str | None = None,
    scope: str = "openid email profile",
    redirect_port: int = 8400,
    ca_cert: str | None = None,
) -> BrowserAuthHook:
    """Create a browser auth hook configured for Google OAuth."""
    return BrowserAuthHook(
        authorization_url="https://accounts.google.com/o/oauth2/v2/auth",
        token_url="https://oauth2.googleapis.com/token",
        client_id=client_id,
        client_secret=client_secret,
        scope=scope,
        redirect_port=redirect_port,
        ca_cert=ca_cert,
    )
