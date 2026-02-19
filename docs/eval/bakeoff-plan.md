# Tooling Bake-Off Plan (Draft)

## Candidates

Vector store:
- LanceDB (current)
- Qdrant
- pgvector

Embedding runtimes:
- Ollama (local)
- TEI (local)
- OpenAI-compatible endpoint (cloud)

Document conversion (quality-first):
- PPTX conversion backends for `filesystem_documents`:
  - `soffice_pdf_docling` (default): `soffice` → PDF → `docling`
  - `pandoc`: direct `pandoc` PPTX → Markdown
  - `python_pptx`: `python-pptx` direct extraction
  - `markitdown`: `markitdown` library conversion

## Dataset

- Corpus: a representative sample of your canonical markdown (mix of short/long docs).
- Queries: 30–100 “realistic” questions mapped to expected source documents.
- Labels: at minimum, for each query, a set of “relevant doc_ids” or canonical markdown paths.

## Metrics

Retrieval quality:
- recall@k (k=5,10,20)
- MRR or nDCG@k

Operational:
- p50/p95 latency (query embedding, retrieval)
- indexing throughput (docs/min, chunks/min)
- disk usage (index size per doc/chunk)
- portability (copy/move `COLIBRI_HOME`)

## Promotion gate for a generation

Minimum:
- No serving-alignment issues (`colibri serve --check`)
- Safety routing enforced (restricted/confidential not routed to cloud)
- Quality metrics do not regress vs active generation on the evaluation set
