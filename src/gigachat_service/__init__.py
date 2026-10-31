"""GigaChat service module for AgentDrone.

Provides AI inspector summary generation and offline fallback templates for AS 'Kalita'.
"""

from .service import GigaChatInspectorService
from .prompts import SYSTEM_INSPECTOR_PROMPT, build_inspector_prompt

__all__ = [
    "GigaChatInspectorService",
    "SYSTEM_INSPECTOR_PROMPT",
    "build_inspector_prompt",
]
