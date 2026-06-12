"""OAuth 2.0 token manager with automatic refresh."""

import time
import logging
import requests
import threading
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class OAuthManager:
    """Manages OAuth token fetching and auto-refresh."""

    def __init__(
        self,
        token_endpoint: str,
        client_id: str,
        client_secret: str,
        scope: Optional[str] = None,
        refresh_buffer_minutes: int = 5,
        verify_ssl: bool = True
    ):
        self.token_endpoint = token_endpoint
        self.client_id = client_id
        self.client_secret = client_secret
        self.scope = scope
        self.refresh_buffer_seconds = max(refresh_buffer_minutes, 0) * 60
        # Never schedule immediate refresh loops; keep at least a small delay.
        self.min_refresh_delay_seconds = 1.0
        self.verify_ssl = verify_ssl

        self._access_token: Optional[str] = None
        self._expires_at: Optional[float] = None
        self._refresh_timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()

    def get_token(self) -> Optional[str]:
        """Get a valid access token (refreshes if needed)."""
        with self._lock:
            # Check if we need to refresh
            if not self._access_token or self._needs_refresh():
                self._fetch_token()

            return self._access_token

    def _needs_refresh(self) -> bool:
        """Check if token needs to be refreshed."""
        if not self._expires_at:
            return True

        # Refresh if we're within buffer time of expiry
        time_until_expiry = self._expires_at - time.time()
        return time_until_expiry <= self.refresh_buffer_seconds

    def _fetch_token(self) -> None:
        """Fetch a new OAuth token."""
        try:
            logger.info(f"Fetching OAuth token from {self.token_endpoint}")

            data = {
                'grant_type': 'client_credentials',
            }

            if self.scope:
                data['scope'] = self.scope

            # Try with Basic Auth first (some OAuth servers prefer this)
            from requests.auth import HTTPBasicAuth
            auth = HTTPBasicAuth(self.client_id, self.client_secret)

            logger.debug(f"OAuth request: grant_type=client_credentials, scope={self.scope}")

            response = requests.post(
                self.token_endpoint,
                data=data,
                auth=auth,
                headers={'Content-Type': 'application/x-www-form-urlencoded'},
                timeout=30,
                verify=self.verify_ssl
            )

            # If Basic Auth fails with 400, try with credentials in body
            if response.status_code == 400:
                logger.warning("Basic Auth failed, trying with credentials in request body...")
                data['client_id'] = self.client_id
                data['client_secret'] = self.client_secret

                response = requests.post(
                    self.token_endpoint,
                    data=data,
                    headers={'Content-Type': 'application/x-www-form-urlencoded'},
                    timeout=30,
                    verify=self.verify_ssl
                )

            if not response.ok:
                error_detail = ""
                try:
                    error_data = response.json()
                    error_detail = f": {error_data}"
                except:
                    error_detail = f": {response.text}"

                logger.error(f"OAuth token request failed with {response.status_code}{error_detail}")
                raise Exception(f"OAuth request failed: {response.status_code}{error_detail}")

            response.raise_for_status()
            token_data = response.json()

            self._access_token = token_data.get('access_token')
            expires_in = token_data.get('expires_in', 3600)
            self._expires_at = time.time() + expires_in

            logger.info(f"OAuth token obtained (expires in {expires_in}s)")

            # Schedule next refresh
            self._schedule_refresh()

        except Exception as e:
            logger.error(f"Failed to fetch OAuth token: {e}")
            self._access_token = None
            self._expires_at = None

    def _schedule_refresh(self) -> None:
        """Schedule the next token refresh."""
        if self._refresh_timer:
            self._refresh_timer.cancel()

        if not self._expires_at:
            return

        # Schedule refresh before expiry, but avoid zero-delay recursion.
        time_until_refresh = (self._expires_at - time.time()) - self.refresh_buffer_seconds
        if time_until_refresh < self.min_refresh_delay_seconds:
            time_until_refresh = self.min_refresh_delay_seconds

        self._refresh_timer = threading.Timer(time_until_refresh, self._refresh_token)
        self._refresh_timer.daemon = True
        self._refresh_timer.start()

    def _refresh_token(self) -> None:
        """Background token refresh."""
        logger.info("Refreshing OAuth token...")
        with self._lock:
            self._fetch_token()

    def get_token_info(self) -> Dict[str, Any]:
        """Get token information for debugging."""
        return {
            'has_token': bool(self._access_token),
            'expires_at': self._expires_at,
            'time_until_expiry': self._expires_at - time.time() if self._expires_at else None,
        }

    def destroy(self) -> None:
        """Clean up resources."""
        if self._refresh_timer:
            self._refresh_timer.cancel()
