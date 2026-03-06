"""Engine configuration dataclass."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EngineConfig:
    functions_json_path: str
    metadata_json_path: str
    out_dir: str

    std: str = "c++17"
    clang_args: List[str] = field(default_factory=list)

    llm_url: str = "http://localhost:11434/api/generate"
    llm_model: str = "llama3"

    # Optional: generate flowchart only for this function key
    function_key: Optional[str] = None

    use_cache: bool = True
    cache_dir: str = ".flowchart_cache"

    # Statement segment thresholds per ACTION node
    max_stmts_per_segment: int = 5
    max_lines_per_segment: int = 10

    # LLM call settings
    llm_timeout: int = 120
    llm_max_retries: int = 2
    llm_temperature: float = 0.1
