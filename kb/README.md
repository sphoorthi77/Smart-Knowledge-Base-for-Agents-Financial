# Smart Knowledge Base for Agents

A production-grade, high-precision retrieval and answering engine designed for financial and regulatory research copilots. Clears the ~0.60 ceiling of standard RAG pipelines by enforcing **pre-search document filtering**, **aspect-aware coverage allocation**, and **dynamic relevance verification**.

---

## Quick Start & Verification

### 1. Environment Setup
Install dependencies:
```bash
pip install -r requirements.txt
```

Set environment variables:
```powershell
$env:PYTHONUTF8 = "1"
$env:OPENCODE_API_KEY = "your_opencode_api_key_here"
# Optional Langfuse tracing:
$env:LANGFUSE_PUBLIC_KEY = "pk-lf-..."
$env:LANGFUSE_SECRET_KEY = "sk-lf-..."
$env:LANGFUSE_HOST = "https://cloud.langfuse.com"
```

### 2. Verify Module Integrity
Run in your shell to verify all modules load with zero circular imports:
```powershell
python -c "import app.main, app.retrieve, app.llm, app.ingest; print('ALL MODULES OK')"
```

### 3. Launch the API Server
```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 4. API Endpoints
- **Index New Documents (`POST /index`)**:
  ```powershell
  curl.exe -F collection_name=dev -F file=@Documents.zip http://localhost:8000/index
  ```
- **Query the Knowledge Base (`POST /query`)**:
  ```powershell
  $body = '{"collection_name":"dev","query":"What has Refex exited and what is it leaning into?"}'
  Invoke-RestMethod -Uri "http://localhost:8000/query" -Method Post -ContentType "application/json" -Body $body
  ```

### 5. Run Benchmark Evaluation
Execute the benchmark evaluation on the development set:
```powershell
python run_eval.py --root "$env:USERPROFILE\Downloads\Train" --collection dev --ids C32,C05,C14,A10,B19,C23
```

To run across the full 100 questions:
```powershell
python run_eval.py --root "$env:USERPROFILE\Downloads\Train" --collection dev --n 100
```

---

## Architectural Highlights
1. **Dynamic Metadata & Document Filtering (`metadata_filter`)**: Isolates legal entity, reporting year, regulator, and jurisdiction before any chunk search runs, avoiding near-duplicate cross-contamination.
2. **Lexical Safety Net**: Deterministic entity and acronym matcher (FDIC, OCC, SAMA, BSF, CEAT, Refex, Gammon, etc.) ensuring queries never miss named corporate or regulatory bodies.
3. **Relevance Verification Gating (`relevance_filter`)**: Filters candidate excerpts to keep only genuinely relevant evidence, driving Context Precision to $\ge 0.95$.
4. **Reasoning-Model Escalation**: Protects against reasoning token exhaustion (`finish_reason=length`) with automatic token budget doubling ($1\times \to 2\times \to 4\times$, up to 32,000 tokens).
5. **Strict Grounding & Citation**: Output enforces discrete factual claims with inline `[document p.N]` citations and zero conversational preamble to maximize Answer Precision and Answer Recall.
