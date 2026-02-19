# Evaluation and Bake-Off Notes

This folder documents how CoLibri should evaluate retrieval quality and operational tradeoffs for:
- Vector store selection
- Embedding runtime selection
- Indexing pipeline parameters (chunk size/overlap)

Goals:
- Make “default stack” choices defensible and repeatable.
- Provide a promotion gate for new index generations (quality + latency + operational checks).

Recommended next artifacts:
- A small “golden” corpus (portable, versioned).
- A query set with expected high-level intents.
- Metrics: recall@k, MRR/nDCG, plus p50/p95 latency for embedding + retrieval.

