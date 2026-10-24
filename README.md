# Smart Knowledge Base for Agents — Financial(AGENT_KB)

A document-grounded knowledge retrieval system designed to help AI agents answer financial-domain questions using information from provided documents, with traceable evidence and explicit handling of unsupported queries.

## Overview

Financial documents often contain detailed policies, procedures, definitions, and domain-specific information spread across multiple PDF files. Retrieving the correct information requires more than a general-purpose chatbot: responses must be grounded in the source material, relevant passages must be retrieved accurately, and unsupported claims must be avoided.

**Smart Knowledge Base for Agents — Financial** aims to address this challenge through structured document ingestion, semantic retrieval, graph-based relationships, and evidence-grounded response generation.

The goal is to make financial knowledge more accessible to AI agents while preserving document provenance and reducing unsupported answers.

## Problem Statement

Traditional document-search systems can struggle with questions that require understanding context across multiple pages or connecting related information across documents. General-purpose language models can also generate plausible-sounding responses that are not supported by the available evidence.

In financial applications, these limitations can reduce reliability and make it difficult to verify the information behind an answer.

This project explores a knowledge retrieval architecture that combines vector similarity search with graph-based relationships and source-aware answer generation.

## Key Objectives

- **Document-grounded question answering:** Generate responses using retrieved evidence from the supplied PDF collection.
- **Semantic retrieval:** Identify relevant passages based on the meaning of a question rather than relying only on exact keyword matches.
- **Graph-based knowledge relationships:** Represent connections between documents, pages, chunks, and extracted entities.
- **Source traceability:** Preserve document names and original PDF page numbers so retrieved evidence can be verified.
- **Explicit no-data handling:** Return the exact response `No data available in the provided documents.` when sufficient supporting evidence is unavailable.
- **Query-level observability:** Capture retrieval and generation steps to make system behaviour easier to inspect and evaluate.
- **Extensible architecture:** Support the development of knowledge retrieval capabilities that can be integrated into AI-agent workflows.

## How It Works

1. **Document ingestion** — Load the supplied PDF documents and extract their text while retaining source-document and page-level information.
2. **Text chunking** — Divide extracted text into smaller passages while preserving document and page provenance.
3. **Embedding generation** — Convert text chunks into vector representations for semantic similarity search.
4. **Vector indexing** — Store and retrieve chunk embeddings through the selected VectorDB.
5. **Knowledge graph construction** — Represent document, page, chunk, and entity relationships in the selected GraphDB.
6. **Evidence retrieval** — Retrieve relevant passages and use graph relationships to provide additional structural context.
7. **Evidence validation** — Assess whether the retrieved material provides sufficient support for the question.
8. **Answer generation** — Generate a response grounded in the available evidence, including source references where supported.
9. **No-data response** — Return the required no-data message when the system determines that the documents do not contain sufficient supporting information.

## System Architecture

The intended architecture separates semantic retrieval, graph-based knowledge representation, and language-model generation.

| Component | Responsibility |
|---|---|
| PDF ingestion | Extract document text and preserve original page references |
| Chunking and metadata | Organize passages with source metadata |
| Embedding model | Generate semantic representations of passages |
| VectorDB | Index embeddings and retrieve semantically similar chunks |
| GraphDB | Store and query relationships between documents, pages, chunks, and entities |
| Retrieval and evidence validation | Assemble relevant evidence and assess its adequacy |
| Language model | Produce evidence-grounded answers |
| API layer | Expose indexing and query functionality |
| Observability and evaluation | Record query execution details and evaluate retrieval and answer quality |

**Important:** The VectorDB and GraphDB are separate components with different responsibilities. The vector store handles semantic similarity search; the graph database represents explicit relationships. The deployed implementation should use persistent database services if persistence is a requirement.

## API Design

The target API interface includes the following endpoints.

### `POST /index`

Indexes a supplied document collection.

Expected responsibilities:

- Accept documents or a document collection identifier.
- Extract text and retain page-level provenance.
- Generate and store chunk embeddings.
- Persist document and entity relationships in the graph database.
- Return an indexing status and relevant collection metadata.

### `POST /query`

Answers a question using an indexed collection.

Expected responsibilities:

- Accept a question and collection identifier.
- Retrieve relevant evidence from the indexed knowledge base.
- Apply evidence sufficiency checks.
- Return an answer with source citations and retrieved evidence.
- Return the required no-data response when adequate evidence is unavailable.
- Provide a structured execution trace for debugging and evaluation.

The precise request and response schemas should match the implemented API.

## Reliability and Evaluation

Reliability is a core design consideration for financial-domain question answering.

The system should be evaluated on:

- **Answer precision:** Whether generated answers contain supported, relevant information.
- **Answer recall:** Whether the answer covers the information required by the question.
- **Context precision:** Whether retrieved passages are relevant to the question.
- **Context recall:** Whether retrieval captures the evidence needed to answer the question.
- **Citation correctness:** Whether cited documents and page numbers support the associated claims.
- **Out-of-domain rejection:** Whether unrelated or unsupported questions receive the exact required no-data response.
- **Latency and cost:** The time and model cost associated with processing queries.

A benchmark dataset should be used to measure these metrics. Performance targets should be treated as goals until the evaluation has been executed and its results recorded.

## Expected Use Cases

- Searching financial reference documents using natural-language questions.
- Retrieving policy and procedure information from a supplied document collection.
- Tracing an answer to its supporting document and page.
- Connecting related information across documents through graph relationships.
- Providing a document-grounded retrieval layer for AI agents.
- Identifying questions that cannot be answered from the available knowledge base.

These are intended use cases; suitability for production financial workflows requires further validation.


### Prerequisites

Install the dependencies and services required by the actual implementation. Depending on the selected architecture, these may include:

- Python
- A PDF text-extraction library
- A sentence-embedding model
- A persistent VectorDB
- A persistent GraphDB
- An LLM provider and API credentials
- A web framework for API endpoints

## Future Improvements

- Improve retrieval accuracy across complex, multi-document questions.
- Calibrate evidence sufficiency thresholds against supported and unsupported queries.
- Expand graph relationships using validated entities and metadata.
- Strengthen citation validation and claim-to-evidence traceability.
- Add automated regression tests for indexing, retrieval, and no-data behaviour.
- Evaluate latency, scalability, and operating cost under realistic workloads.
- Integrate the retrieval layer with broader AI-agent workflows.
