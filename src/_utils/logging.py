# =============================================================================
# logging.py - Rich Console Logging Utilities
# =============================================================================
#
# Purpose:
#   Provides consistent, visually appealing logging across all modules using
#   the Rich library. Includes custom SUCCESS log level, section headers,
#   and proper configuration for Ray distributed environments.

# Features:
#   - Custom SUCCESS log level (between INFO and WARNING)
#   - Section headers with horizontal rules
#   - Color-coded log levels with custom theme
#   - Proper handler configuration to avoid duplicates
#
# Usage:
#   from src._utils.logging import get_logger, log_section
#
#   logger = get_logger(__name__)
#   logger.info("Starting process...")
#   logger.success("Process completed!")  # Custom level
#
#   log_section("Data Loading", "📦")  # Section header
#
# Ray Compatibility:
#   According to Ray docs, logging should be configured AFTER ray.init()
#   but BEFORE creating workers. This module provides loggers that work
#   correctly in both driver and worker processes.
#
# =============================================================================
"""
Rich console logging utilities for consistent, beautiful output.

Provides:
- get_logger(): Logger with RichHandler for pretty formatting
- log_section(): Section headers for visual organization
- Custom SUCCESS level for milestone logging
"""

import logging
import sys

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# =============================================================================
# Custom Theme Configuration
# =============================================================================
CUSTOM_THEME = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "red bold",
        "critical": "red bold reverse",
        "success": "green bold",
        "debug": "blue",
    }
)

# Global console instance with full color support
console = Console(
    theme=CUSTOM_THEME,
    file=sys.stdout,
    force_terminal=True,  # Always use terminal formatting
    force_jupyter=False,  # Disable Jupyter detection
    force_interactive=False,  # Non-interactive mode for CI
    color_system="truecolor",  # Full 24-bit color support
    legacy_windows=False,  # Modern Windows terminal support
)

# =============================================================================
# Custom SUCCESS Log Level
# =============================================================================
SUCCESS_LEVEL = 25  # Between INFO (20) and WARNING (30)
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")


def success(self, message, *args, **kwargs):
    """
    Log a success message at custom SUCCESS level.

    Use for milestone completions, successful operations, or
    positive status updates that are more important than INFO.
    """
    if self.isEnabledFor(SUCCESS_LEVEL):
        self._log(SUCCESS_LEVEL, message, args, **kwargs)


# Monkey-patch success method onto Logger class
logging.Logger.success = success


# =============================================================================
# Logger Factory
# =============================================================================
def get_logger(name: str) -> logging.Logger:
    """
    Get a logger configured with RichHandler for pretty formatting.

    Creates a logger with:
    - Rich console output with colors and formatting
    - Timestamp display in logs
    - Rich tracebacks for exceptions
    - Markup support for [cyan]colored[/cyan] text

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logging.Logger instance

    Note:
        According to Ray docs, configure logging AFTER ray.init()
        but BEFORE creating workers. This function returns loggers
        that work in both driver and worker processes.
    """
    logger = logging.getLogger(name)

    # Only configure if no handlers exist (avoid duplicate handlers)
    if not logger.handlers:
        handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,  # Hide file paths for cleaner output
            rich_tracebacks=True,  # Beautiful exception formatting
            markup=True,  # Enable [color]text[/color] markup
            enable_link_path=False,  # Disable clickable file links
            log_time_format="[%Y-%m-%d %H:%M:%S]",
        )
        handler.setFormatter(logging.Formatter("%(message)s"))

        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False  # Don't propagate to root logger

    return logger


# =============================================================================
# Section Header Utility
# =============================================================================
def log_section(title: str, emoji: str = "📌") -> None:
    """
    Print a section header with horizontal rule.

    Creates a visually distinct section divider for organizing
    log output into logical groups. Uses print() instead of
    logging for Ray worker compatibility.

    Args:
        title: Section title text
        emoji: Emoji prefix for the section (default: 📌)

    Example:
        log_section("Loading Data", "📦")
        # Output: ─────────── 📦 Loading Data ───────────
    """
    console.rule(f"{emoji} {title}", style="bold cyan")
