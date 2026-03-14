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
    llm_model: str = "gpt-oss"

    # Optional: generate flowchart only for this function key
    function_key: Optional[str] = None

    use_cache: bool = True
    cache_dir: str = ".flowchart_cache"

    # Statement segment thresholds per ACTION node.
    # Reduced to 3 statements so that important function calls are unlikely
    # to be buried in a large segment where the LLM may omit them from the label.
    max_stmts_per_segment: int = 3
    max_lines_per_segment: int = 10

    # Optional: path to project_knowledge.json built by project_scanner.py
    knowledge_json_path: Optional[str] = None

    # LLM call settings
    llm_timeout: int = 120
    llm_max_retries: int = 2
    llm_temperature: float = 0.1
    # Nodes per LLM call.  Reduce for small-context-window models (e.g. 4k).
    # Each batch sends only the source lines of its nodes, not the full source.
    llm_batch_size: int = 8
    # Ollama num_ctx: explicitly sets the model's context window for each call.
    # Ollama defaults to 2048 for many models; prompts >2048 tokens return empty.
    # Set to 8192 to safely handle all prompt sizes up to ~2500 tokens.
    llm_num_ctx: int = 8192
