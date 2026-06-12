"""Request handlers for kilo-launcher."""

from .dashboard_api import dashboard_bp
from .proxy_handler import proxy_bp

__all__ = ["dashboard_bp", "proxy_bp"]
