"""Credit Registry loader and query manager for Siberian Bank portfolio.

Loads and validates credit objects against Pydantic schemas from data/registry.json.
Provides lookup by cadastral number (both formatted and clean digits).
"""

from pathlib import Path
from typing import Dict, List, Optional
import json

from ..config import PROJECT_ROOT
from ..models import CreditObject

DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "data" / "registry.json"


class CreditRegistry:
    """Manages bank loan records associated with cadastral plots."""

    def __init__(self, registry_file: Optional[Path | str] = None):
        self.registry_file = Path(registry_file) if registry_file else DEFAULT_REGISTRY_PATH
        self._credits_by_clean_cadastral: Dict[str, CreditObject] = {}
        self._credits_by_id: Dict[str, CreditObject] = {}
        self.reload()

    def reload(self) -> None:
        """Reload and validate all credit records from the JSON registry."""
        if not self.registry_file.exists():
            raise FileNotFoundError(f"Реестр кредитов не найден: {self.registry_file}")

        with open(self.registry_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        self._credits_by_clean_cadastral.clear()
        self._credits_by_id.clear()

        for item in raw_data:
            credit = CreditObject.model_validate(item)
            self._credits_by_clean_cadastral[credit.clean_cadastral] = credit
            self._credits_by_id[credit.credit_id] = credit

    def list_all(self) -> List[CreditObject]:
        """Return all credit records in the portfolio."""
        return list(self._credits_by_clean_cadastral.values())

    def get_by_cadastral(self, cadastral_num: str) -> Optional[CreditObject]:
        """Look up credit record by raw or cleaned cadastral number."""
        clean = cadastral_num.replace(":", "").replace(" ", "").strip()
        return self._credits_by_clean_cadastral.get(clean)

    def get_by_credit_id(self, credit_id: str) -> Optional[CreditObject]:
        """Look up credit record by credit contract ID."""
        return self._credits_by_id.get(credit_id.strip())

    def get_overdue_loans(self) -> List[CreditObject]:
        """Return all loans with overdue debt (ПЗ > 0)."""
        return [c for c in self.list_all() if c.days_overdue > 0]

    def __len__(self) -> int:
        return len(self._credits_by_clean_cadastral)


# Global singleton helper
_registry_instance: Optional[CreditRegistry] = None


def get_default_registry() -> CreditRegistry:
    """Return cached singleton instance of CreditRegistry."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = CreditRegistry()
    return _registry_instance
