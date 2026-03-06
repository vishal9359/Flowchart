"""Disk cache for the Project Knowledge Base."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class PkbCache:
    """
    Caches PKB data keyed by a hash of functions.json content.
    Rebuilds only when functions.json changes.
    """

    def __init__(self, cache_dir: str) -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _content_hash(self, content: str) -> str:
        return hashlib.md5(content.encode("utf-8")).hexdigest()[:16]

    def _cache_path(self, content_hash: str) -> Path:
        return self._cache_dir / f"pkb_{content_hash}.json"

    def load(self, functions_json_content: str) -> Optional[Dict]:
        """Return cached PKB dict if functions.json hasn't changed, else None."""
        h = self._content_hash(functions_json_content)
        path = self._cache_path(h)
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.info("PKB loaded from cache: %s", path)
                return data
            except Exception as exc:
                logger.warning("Cache load failed: %s", exc)
        return None

    def save(self, functions_json_content: str, data: Dict) -> None:
        """Persist PKB dict to disk."""
        h = self._content_hash(functions_json_content)
        path = self._cache_path(h)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.info("PKB saved to cache: %s", path)
        except Exception as exc:
            logger.warning("Cache save failed: %s", exc)

    def invalidate_stale(self, functions_json_content: str) -> None:
        """Remove cache files that don't match the current functions.json hash."""
        current = f"pkb_{self._content_hash(functions_json_content)}.json"
        for f in self._cache_dir.glob("pkb_*.json"):
            if f.name != current:
                try:
                    f.unlink()
                    logger.debug("Removed stale cache: %s", f)
                except Exception:
                    pass
