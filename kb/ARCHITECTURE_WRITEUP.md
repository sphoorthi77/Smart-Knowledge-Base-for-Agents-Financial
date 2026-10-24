# Smart Knowledge Base for Agents: Architectural Design & Implementation

## 1. Executive Summary & Core Challenges Addressed
Standard retrieval systems (VectorDB + similarity search + rerankers) routinely plateau at ~0.60 precision and recall in high-stakes enterprise applications, failing for two fundamental reasons:
1. **Near-Duplicate / Multi-Period Confusion**: Financial and regulatory corpora contain dozens of documents with identical headings, terminology, and topics differing only by reporting period (e.g., FY2024-25 vs. FY2025-26), legal entity, or jurisdiction. Embedding vectors cluster these tightly, causing similarity search to retrieve wrong-year or wrong-entity excerpts.
2. **The Chunk-Size / Coverage Dilemma**: Complex financial questions require synthesizing facts distributed across tens of pages. Fixed top-$k$ retrieval either starves broad synthesis queries (yielding incomplete answers) or floods narrow queries with irrelevant chunks (destroying context precision).

To achieve **$\ge$0.95 Answer Precision, Answer Recall, Context Precision, and Context Recall**, our solution implements an advanced **Hybrid Graph & Vector Architecture** with pre-search metadata isolation:
- **Knowledge Graph Layer (NetworkX)**: Models the explicit relational topology connecting documents, legal entities, subsidiaries, reporting periods, and regulatory frameworks.
- **Vector Database Layer (FAISS)**: Implements indexed high-dimensional vector similarity (`IndexFlatIP`) across 28,000+ chunks, scoped dynamically to the graph-filtered candidate pool.
- **Mandatory Pre-Search Document Filtering**: An LLM planner filters the corpus to only the authorized document subset before chunk retrieval begins, reinforced by a deterministic lexical safety net.
- **Relevance Verification Gating**: An LLM-level relevance filter discards marginal chunks, maximizing Context Precision.
- **Grounded, Preamble-Free Generation**: Strictly bounded generation enforcing discrete factual claims, explicit disclosure of unstated metrics, and mandatory inline `[document p.N]` citations.

---

## 2. End-to-End System Architecture

```
User Query / API Request
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. Metadata Routing & Filtering (metadata_filter)           │
│    • LLM reads auto-generated Document Cards catalogue     │
│    • Mandatorily isolates target entity, period & scope     │
│    • Deterministic Lexical Safety Net restores named entities│
└──────────────────────────────┬──────────────────────────────┘
                               │ Filtered Document IDs
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Knowledge Graph Relationship Expansion                   │
│    (graph_relation_expansion)                               │
│    • NetworkX DiGraph traversal (92 nodes, 168 edges)       │
│    • Resolves subsidiary hierarchies, demerger entities,    │
│      interagency regulations, and cross-framework updates   │
└──────────────────────────────┬──────────────────────────────┘
                               │ Graph-Expanded Document IDs
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Scoped FAISS Vector DB Similarity Search                 │
│    (faiss_vector_similarity)                                │
│    • FAISS IndexFlatIP (28,233 vectors) + aspect max-fusion │
│    • Scoped strictly inside the graph-filtered pool         │
│    • Per-document quota allocation with coverage floor      │
└──────────────────────────────┬──────────────────────────────┘
                               │ Candidate Chunks
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Relevance Verification Gating (relevance_filter)         │
│    • Evaluates excerpts for direct factual utility          │
│    • Drops marginal/tangential excerpts to boost Context P  │
└──────────────────────────────┬──────────────────────────────┘
                               │ Verified Relevant Excerpts
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Grounded Answer Synthesis (llm_answer_generation)        │
│    • Zero conversational preamble (no hallucinated points)  │
│    • Point-by-point factual answers backed by [doc p.N]     │
│    • Explicit statements for unforecasted/missing metrics   │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
Output Response: { answer, chunks, trace, usage }
```

---

## 3. Detailed Component Breakdown

### 3.1 Knowledge Graph Architecture (`graph_relation_expansion`)
- **Graph Topology**: Built on `networkx.DiGraph`, with typed nodes:
  - `Document` nodes (`doc:N`)
  - `Entity` nodes (`entity:<name>`, e.g., Refex Industries, Venwind, FDIC, OCC, SAMA)
  - `Period` nodes (`period:<year>`, e.g., 2026, FY2025-26)
  - `Jurisdiction` nodes (`jurisdiction:<country>`, e.g., US, India, Saudi Arabia, Basel/Global)
  - `DocType` nodes (`type:<category>`, e.g., annual report, supervisory guidance)
- **Multi-Hop Traversal**:
  - Resolves corporate perimeters: When a query asks about *Refex*, the graph traverses `HAS_SUBSIDIARY` edges to automatically discover *Refex Green Mobility*, *Venwind Refex Power*, and *Refex Mobility*.
  - Resolves regulatory partnerships: Interagency ties link `FDIC` $\leftrightarrow$ `OCC` $\leftrightarrow$ `Federal Reserve` for joint CECL statements and Basel III proposals.

### 3.2 Vector Database Architecture (`faiss_vector_similarity`)
- **Index Implementation**: `faiss.IndexFlatIP` (normalized inner product / cosine similarity) indexing 28,233 chunk vectors.
- **Filtered-Pool Scoping**: Rather than running global vector search (which suffers from topical confusion across reporting years), vector similarity is strictly masked to the candidate pool determined by the Metadata Filter and Knowledge Graph.
- **Aspect-Aware Max-Fusion**: The query is expanded into 2–6 sub-aspects by the planner. Vector similarities across all aspects are fused via max-scaling with per-document coverage floors.

### 3.3 Dynamic Relevance Gating (`relevance_filter`)
- Eliminates the classic "top-$k$ clutter" that destroys Context Precision.
- An LLM evaluator grades candidate excerpts against the strict relevance criteria, retaining only those providing direct figures, definitions, illustrative examples, or explicit caveats.

### 3.4 Grounded Answer Synthesis (`llm_answer_generation`)
- Strictly penalizes conversational filler ("Based on the excerpts...", "In summary..."). Every single sentence is an atomic factual claim to prevent `extra_ids` penalties during evaluation.
- Explicitly declares missing data or unforecasted metrics (e.g. C32).
- Enforces inline citations: `[document p.N]`.

---

## 4. Evaluation Alignment & Metric Optimization

| Hackathon Metric | Target Bar | Architectural Mechanism Employed |
| :--- | :---: | :--- |
| **Answer Precision** | $\ge 0.95$ | Strict preamble suppression and atomic claim formatting eliminate extraneous, unsupported claims (`extra_ids`). |
| **Answer Recall** | $\ge 0.95$ | Multi-aspect query expansion and quota-enforced coverage across all filtered documents ensure no benchmark facts are missed. |
| **Context Precision** | $\ge 0.95$ | Pre-search document filtering drops near-duplicates; LLM relevance filter discards tangential chunks, keeping only verified useful excerpts. |
| **Context Recall** | $\ge 0.95$ | Graph relation expansion and breadth-adaptive candidate pooling ensure complete capture of all relevant pages. |
| **Trace Auditability** | 100% | 5-step trace array conforming strictly to `{step_name, description, input, output}`. |
