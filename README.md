# Flowchart Engine

Generates Mermaid flowcharts from C++ source code using libclang for static analysis and a local LLM (Ollama) for human-readable labels.

Given a C++ function, it produces a flowchart like this:

```
int classify(int x) {        flowchart TD
    if (x > 0)                   START([Start: classify])
        return 1;                START --> DECISION
    else                         DECISION{Is x positive?}
        return -1;               DECISION -->|Yes| RET1
}                                DECISION -->|No|  RET2
                                 RET1[Return positive result]
                                 RET2[Return negative result]
                                 RET1 --> END
                                 RET2 --> END
                                 END([End])
```

---

## How It Works

```
functions.json          metadata.json         project_knowledge.json
      |                      |                        |  (optional)
      v                      v                        v
 ┌─────────────────────────────────────────────────────────────┐
 │                    flowchart_engine.py                       │
 │                                                             │
 │  1. PKB Build                                               │
 │     Load all function entries (qualified name, file, line)  │
 │     Build caller/callee index for context injection         │
 │                          |                                  │
 │  2. CFG Extraction   (ast_engine/)                          │
 │     libclang parses the .cpp file                           │
 │     CFGBuilder walks the AST and creates:                   │
 │       Nodes: START, END, ACTION, DECISION,                  │
 │              LOOP_HEAD, SWITCH_HEAD, RETURN,                │
 │              CASE, BREAK, CONTINUE, TRY_HEAD, CATCH         │
 │       Edges: control-flow arrows with Yes/No labels         │
 │                          |                                  │
 │  3. Enrichment       (enrichment/)                          │
 │     Each node is enriched with extra context:               │
 │       - Function calls within the node                      │
 │       - Inline source comments                              │
 │       - Enum / macro / typedef / struct member info         │
 │                          |                                  │
 │  4. LLM Labeling     (llm/)                                 │
 │     Nodes are sorted topologically and split into batches   │
 │     Each batch is sent to the LLM with a context packet:    │
 │       - File and function purpose                           │
 │       - Caller context (who calls this function)            │
 │       - Callee context (what this function calls)           │
 │       - Phase hints (if project_scanner was run)            │
 │       - Neighbor node code (preceding / following)          │
 │       - Data-flow shared variables across the batch         │
 │     A coherence pass normalises labels across all batches   │
 │                          |                                  │
 │  5. Mermaid Render   (mermaid/)                             │
 │     Labeled CFG → Mermaid flowchart TD script               │
 │     Node shapes: oval=START/END  diamond=DECISION           │
 │                  rectangle=ACTION  subroutine=CATCH         │
 │                          |                                  │
 │  6. Output           (output/)                              │
 │     One JSON file per source file written to --out-dir      │
 └─────────────────────────────────────────────────────────────┘
                            |
                     output/myfile.json
```

---

## Project Structure

```
flowchart_engine.py     Main entry point and orchestration
models.py               Data models: CfgNode, CfgEdge, FunctionEntry, etc.
config.py               EngineConfig dataclass

ast_engine/
  cfg_builder.py        Builds ControlFlowGraph from a libclang AST cursor
  parser.py             SourceExtractor and TranslationUnitParser (libclang)
  resolver.py           Finds the libclang cursor for a given function

enrichment/
  enricher.py           Enriches CFG nodes with PKB/project-knowledge context

llm/
  generator.py          Batch LLM label generation + coherence pass
  prompts.py            System and user prompt templates
  client.py             Ollama HTTP client with auto-retry and auto-split

mermaid/
  builder.py            Converts a labeled CFG to a Mermaid TD script
  validator.py          Validates Mermaid script structure
  normalizer.py         Normalises edge labels (Yes/No, case values)

pkb/
  builder.py            ProjectKnowledgeBase — caller/callee index + context packets
  knowledge.py          FunctionKnowledge dataclass and JSON serialisation
  cache.py              Disk cache for the PKB

output/
  writer.py             Writes per-source-file JSON to --out-dir

project_scanner.py      Standalone tool — builds project_knowledge.json
                        (file summaries, function purposes, phase breakdowns)

tests/
  test_cfg_topo.py      Layer-1 & Layer-2 test runner (see Testing section)
```

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python      | 3.9+    |
| libclang    | 16+     |
| Ollama      | any     |
| A code-capable LLM model | e.g. `qwen2.5-coder:7b` |

```bash
pip install -r requirements.txt
```

Ollama must be running locally:
```bash
ollama serve
ollama pull qwen2.5-coder:7b
```

---

## Input Files

### functions.json
Generated by your C++ analyser. Each key is a unique function identifier.

```json
{
  "src|myfile|MyClass::myMethod|int,bool": {
    "qualifiedName": "MyClass::myMethod",
    "location": {
      "file": "src/myfile.cpp",
      "line": 42,
      "endLine": 85
    },
    "parameters": [
      { "type": "int",  "name": "x" },
      { "type": "bool", "name": "flag" }
    ],
    "callsIds":    ["src|util|helper|void"],
    "calledByIds": ["src|main|main|int"],
    "description": "Processes x with the given flag."
  }
}
```

### metadata.json
```json
{
  "basePath":    "/absolute/path/to/cpp/project",
  "projectName": "MyProject"
}
```

### project_knowledge.json (optional)
Built by `project_scanner.py`. Provides richer semantic context for LLM labels.

---

## Usage

### Basic — generate flowcharts

```bash
python flowchart_engine.py \
    --interface-json functions.json \
    --metaData-json  metadata.json  \
    --out-dir        output/        \
    --llm-url        http://localhost:11434 \
    --llm-model      qwen2.5-coder:7b
```

### With project knowledge (better labels)

```bash
# Step 1: build project knowledge (run once, or when code changes significantly)
python project_scanner.py \
    --interface-json functions.json \
    --metaData-json  metadata.json  \
    --llm-url        http://localhost:11434 \
    --llm-model      qwen2.5-coder:7b \
    --llm-summarize \
    --out             project_knowledge.json

# Step 2: generate flowcharts using that knowledge
python flowchart_engine.py \
    --interface-json  functions.json \
    --metaData-json   metadata.json  \
    --knowledge-json  project_knowledge.json \
    --out-dir         output/        \
    --llm-url         http://localhost:11434 \
    --llm-model       qwen2.5-coder:7b
```

### Single function only

```bash
python flowchart_engine.py \
    --interface-json functions.json \
    --metaData-json  metadata.json  \
    --out-dir        output/        \
    --function-key   "src|myfile|MyClass::myMethod|int,bool"
```

### With missing headers

If your project has headers in non-standard locations, pass them to libclang:

```bash
python flowchart_engine.py \
    --interface-json functions.json \
    --metaData-json  metadata.json  \
    --out-dir        output/        \
    --clang-arg      -I/path/to/include \
    --clang-arg      -I/another/include
```

---

## Output Format

Each source file produces one JSON file in `--out-dir`:

```
output/
  myfile.json
  another_module.json
  _summary.json
```

Each JSON file is an array:

```json
[
  {
    "functionKey":   "src|myfile|MyClass::myMethod|int,bool",
    "name":          "MyClass::myMethod",
    "flowchart":     "flowchart TD\n    N0([Start: myMethod])\n    ..."
  },
  {
    "functionKey":   "src|myfile|MyClass::otherMethod|void",
    "name":          "MyClass::otherMethod",
    "flowchart":     "flowchart TD\n    ...",
    "error":         null
  }
]
```

Paste the `flowchart` value into [mermaid.live](https://mermaid.live) to visualise.

---

## Example Walkthrough

### C++ function

```cpp
// src/classifier.cpp
int classify(int x) {
    if (x > 0) {
        return 1;
    } else if (x < 0) {
        return -1;
    } else {
        return 0;
    }
}
```

### What the engine does step-by-step

**Step 1 — CFG Extraction**

libclang parses `classify` and produces:

```
Nodes:
  N0  START     ""
  N1  DECISION  "x > 0"
  N2  RETURN    "return 1"
  N3  DECISION  "x < 0"
  N4  RETURN    "return -1"
  N5  RETURN    "return 0"
  N6  END       ""

Edges:
  N0 → N1         (entry)
  N1 → N2  [Yes]  (x > 0 true branch)
  N1 → N3  [No]   (x > 0 false branch)
  N3 → N4  [Yes]  (x < 0 true branch)
  N3 → N5  [No]   (else branch)
  N2 → N6         (return to end)
  N4 → N6
  N5 → N6
```

**Step 2 — Topological Sort**

Nodes sorted in execution order: `N0 → N1 → N2 → N3 → N4 → N5 → N6`
No back-edges (no loops in this function).

**Step 3 — Batching**

Split at branch points:
```
Batch 1: [N1]       ← first DECISION — flush immediately
Batch 2: [N2, N3]   ← return + second DECISION
Batch 3: [N4, N5]   ← two return branches
```

**Step 4 — LLM Labeling**

LLM receives each batch with context and responds:
```
N1 → "Is x greater than zero?"
N2 → "Return positive (1)"
N3 → "Is x less than zero?"
N4 → "Return negative (-1)"
N5 → "Return zero"
```

**Step 5 — Mermaid Output**

```
flowchart TD
    N0([Start: classify])
    N1{Is x greater than zero?}
    N2[Return positive 1]
    N3{Is x less than zero?}
    N4[Return negative -1]
    N5[Return zero]
    N6([End])

    N0 --> N1
    N1 -->|Yes| N2
    N1 -->|No| N3
    N3 -->|Yes| N4
    N3 -->|No| N5
    N2 --> N6
    N4 --> N6
    N5 --> N6
```

---

## Debugging

### See exactly what the LLM receives

```bash
FLOWCHART_TRACE=1 python flowchart_engine.py ...
```

Prints the full system prompt and user prompt for every batch to stdout.

### LLM context overflow

If labels are falling back to raw code, the LLM context window may be too small. Increase it:

```bash
python flowchart_engine.py ... --llm-num-ctx 16384
```

---

## Testing

The test runner validates the deterministic parts of the pipeline (no LLM required).

### Layer 1 — CFG and Topological Sort invariants

```bash
python tests/test_cfg_topo.py \
    --interface-json functions.json \
    --metadata-json  metadata.json
```

What it checks for each function:

```
CFG invariants:
  - At least one node exists
  - Entry node is set and points to a real node
  - Every edge's source/target is a valid node ID
  - Exactly one START node
  - At least one END node

Topological sort invariants:
  - Every CFG node appears in the output exactly once
  - Entry node is first
  - For every forward edge A→B: A appears before B in the order
  - Every back-edge (loop) points from a later node to an earlier one
```

### Layer 2 — CFG node-type counts vs Mermaid shape counts

Requires previously-generated output from `flowchart_engine.py`:

```bash
python tests/test_cfg_topo.py \
    --interface-json functions.json \
    --metadata-json  metadata.json  \
    --out-dir        output/
```

Additional checks:

```
  - Count of DECISION+LOOP_HEAD+SWITCH_HEAD in CFG
      == count of diamond {..} shapes in Mermaid
  - Count of START+END in CFG
      == count of oval ([..]) shapes in Mermaid
  - Count of ACTION+RETURN+BREAK+... in CFG
      == count of rectangle [..] shapes in Mermaid
  - Count of CATCH in CFG
      == count of subroutine [[..]] shapes in Mermaid
```

### Test a single function

```bash
python tests/test_cfg_topo.py \
    --interface-json functions.json \
    --metadata-json  metadata.json  \
    --function-key   "src|myfile|MyClass::myMethod|int,bool"
```

### With extra include paths

```bash
python tests/test_cfg_topo.py \
    --interface-json functions.json \
    --metadata-json  metadata.json  \
    --clang-arg      -I/path/to/headers
```

### Sample output

```
Testing 42 function(s) from functions.json
Layer-2 (CFG vs Mermaid) enabled — reading from: output/
base_path=/home/user/myproject  std=c++17
============================================================

[PASS] MyClass::processRequest
       key: src|myfile|MyClass::processRequest|int
     OK  cfg.cursor_resolved
     OK  cfg.nodes_not_empty: 12 node(s)
     OK  cfg.entry_node_exists: entry_node_id='N0'
     OK  cfg.edges_reference_valid_nodes: 14 edge(s) all valid
     OK  cfg.exactly_one_start_node: START count=1
     OK  cfg.has_end_node: END count=1
     OK  topo.all_nodes_present_no_duplicates: 12 node(s) in order
     OK  topo.entry_node_is_first: first='N0' entry='N0'
     OK  topo.forward_edges_respect_order: all forward edges ordered correctly
     OK  topo.back_edges_are_backward: 0 back-edge(s) all valid
     OK  mermaid.oval_count_matches_cfg: CFG=2  Mermaid=2  OK
     OK  mermaid.diamond_count_matches_cfg: CFG=3  Mermaid=3  OK
     OK  mermaid.rectangle_count_matches_cfg: CFG=7  Mermaid=7  OK
     OK  mermaid.subroutine_count_matches_cfg: CFG=0  Mermaid=0  OK

[FAIL] MyClass::handleError
       key: src|myfile|MyClass::handleError|void
   FAIL  mermaid.diamond_count_matches_cfg
           → CFG=2  Mermaid=1  MISMATCH

============================================================
FAILED  1/42 functions  (1 check(s) failed)

Failed functions:
  MyClass::handleError
    - mermaid.diamond_count_matches_cfg: CFG=2  Mermaid=1  MISMATCH
```
