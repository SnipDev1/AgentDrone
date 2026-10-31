"""Report generation module for AgentDrone.

Provides the official Collateral Inspection Act (Приложение 1) template renderer.
"""

from .generator import ActGenerator
from .builder import build_act_report_data

__all__ = [
    "ActGenerator",
    "build_act_report_data",
]
