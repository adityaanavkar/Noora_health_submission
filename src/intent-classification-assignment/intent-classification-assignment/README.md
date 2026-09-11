# Intent Classification Assignment Artifacts

Candidate-facing data for the intent-classification take-home assignment.

## Alignment

For each split, row `i` in `*_queries.csv` corresponds to row `i` in every
matching `*_embeddings_?.npy` matrix. The LLM baseline uses `query_id` for
explicit test-row alignment.

## Files

- `train_queries.csv`: `query_id`, source `intent`, and binary `label`
- `test_queries.csv`: `query_id` only;
- `*_embeddings_a.npy`: normalized float32 vectors, 384 dimensions
- `*_embeddings_b.npy`: normalized float32 vectors, 768 dimensions
- `*_embeddings_c.npy`: normalized float32 vectors, 1536 dimensions
- `llm_baseline_train_predictions.json`: static training-set baseline predictions
- `llm_baseline_test_predictions.json`: static blind-test baseline predictions
- `label_mapping.json`: routing-label policy
- `manifest.json`: provenance, dimensions, and split sizes

Load a matrix with `numpy.load`. Test data must not be used for training,
model selection, calibration, or threshold selection.
