"""
Output writer.

Writes one JSON file per source file into --out-dir.
Each JSON file contains an array of flowchart results.

Output filename: <source_file_basename>.json
  e.g. src/qos/qos_event_manager.cpp → <out-dir>/qos_event_manager.json

Output JSON schema (per file):
[
  {
    "functionKey": "src|qos_event_manager|pos::QosEventManager::_RateLimit|...",
    "qualifiedName": "pos::QosEventManager::_RateLimit",
    "flowchart": "flowchart TD\n    ..."
  },
  ...
]
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

from models import FileResult, FlowchartResult

logger = logging.getLogger(__name__)


class OutputWriter:
    """Writes per-source-file JSON output to out_dir."""

    def __init__(self, out_dir: str) -> None:
        self._out_dir = Path(out_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)

    def write_file_result(self, file_result: FileResult) -> Path:
        """
        Serialize a FileResult and write it to out_dir.
        Returns the path of the written file.
        """
        source_path = Path(file_result.source_file)
        # Use only the stem (filename without extension) + .json
        out_name = source_path.stem + ".json"
        out_path = self._out_dir / out_name

        payload = _serialize_file_result(file_result)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        logger.info("Written %d flowchart(s) → %s",
                    len(file_result.flowcharts), out_path)
        return out_path

    def write_all(self, file_results: List[FileResult]) -> List[Path]:
        """Write all file results and return paths."""
        written = []
        for fr in file_results:
            if fr.flowcharts:
                written.append(self.write_file_result(fr))
        return written

    def write_summary(self, file_results: List[FileResult],
                      total_functions: int,
                      total_errors: int) -> Path:
        """Write a summary JSON listing all output files and statistics."""
        summary = {
            "totalFunctions": total_functions,
            "totalErrors": total_errors,
            "files": [
                {
                    "sourceFile": fr.source_file,
                    "flowchartsGenerated": sum(
                        1 for fc in fr.flowcharts if not fc.error
                    ),
                    "errors": sum(
                        1 for fc in fr.flowcharts if fc.error
                    ),
                }
                for fr in file_results
            ],
        }
        out_path = self._out_dir / "_summary.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info("Summary written → %s", out_path)
        return out_path


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _serialize_file_result(file_result: FileResult) -> List[Dict]:
    """Convert FileResult → JSON-serializable list."""
    result = []
    for fc in file_result.flowcharts:
        entry: Dict = {
            "functionKey": fc.function_key,
            "qualifiedName": fc.qualified_name,
            "flowchart": fc.mermaid_script,
        }
        if fc.error:
            entry["error"] = fc.error
        result.append(entry)
    return result
