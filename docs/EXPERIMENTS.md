# Experiments

This document separates offline evaluation, which this repository runs, from online A/B testing, which it only prepares for. Offline results are not real user A/B tests.

## Offline evaluation (implemented)

**Data.** MovieLens 25M, positive ratings only (rating >= 4.0), users with at least 3 positives. Each user's last event is the test positive, the second-to-last is the validation positive, and the rest are training events. Test events are never used for fitting.

**Models compared**
- `random`: random unseen items
- `popularity`: most-played unseen items in training history
- `retrieval`: FAISS top unseen items from the two-tower model
- `ranker`: hybrid candidates (FAISS plus popular unseen) reranked by LightGBM

**Metrics**
- Recall@10 and NDCG@10, one positive per user
- Genre diversity: distinct primary genres divided by list length
- Coverage: share of the catalog that appears in at least one user's list
- 95% percentile bootstrap intervals over users (1,000 resamples)
- Paired bootstrap intervals for differences between models, on the same users

**Sampling.** Ranker training uses 20,000 validation users. Evaluation uses 20,000 test users, both drawn with a fixed seed. The same users are used for every model.

**Limits.** The offline setting measures how well a list matches a held-out event. It does not measure satisfaction, skips, or long-term engagement.

## Online A/B test (designed, not run)

`rtrec/experiments.py` implements the assignment step:

- `assign(experiment, user_id, treatment_share)` hashes `experiment:user_id` with SHA-256 and maps the result to one of 10,000 buckets. The first `treatment_share` of buckets are `treatment`, the rest are `control`.
- Assignment is deterministic and needs no stored state. A user keeps the same arm across requests and servers.
- Different experiments use different salts, so assignments are independent across experiments.

A real test would work like this:

1. **Hypothesis and metric.** For example: the hybrid ranker increases the rate of played tracks among recommended tracks, measured per user.
2. **Assignment.** Each request computes the arm with `assign`. The arm is logged with the pseudonymous user id and is not visible to users.
3. **Serving.** Control serves the current model. Treatment serves the candidate. Both use the same artifact format.
4. **Logging.** Log impressions and outcomes (plays, skips) keyed by pseudonym and arm.
5. **Analysis.** Compare arm means with a confidence interval over users, not over requests. Pre-register the metric, the minimum detectable effect, and the stopping rule.
6. **Guardrails.** Track latency, error rate, and cache hit rate per arm. Stop the test if a guardrail breaks.

Not implemented: the outcome logging pipeline, the per-arm serving switch in the API, and the statistical analysis of real impressions.
