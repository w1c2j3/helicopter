# BFCL-v4 Component Runs (2026-07-31)

These are local experiment records. They are intentionally untracked and are not uploaded to GitHub.

## Live component

Configuration: `live_parallel`, `live_parallel_multiple`, `live_relevance`; `format_sensitivity` was excluded because it is a non-scoring BFCL category and the installed third-party adapter raises `KeyError: 0` for its data shape.

| Model | Official direct accuracy | Acceptance report | BFCL LIVE | BFCL Overall |
| --- | ---: | ---: | ---: | ---: |
| G1h 7.2B | 20.00% (23/56) | `correct=23, model_error=33` | 20.00% | 2.00% |
| G1h 13.3B | 42.86% (24/56) | `correct=24, model_error=32` | 20.00% | 2.00% |

The component run has no Agentic, Multi-Turn, Non-Live, or Hallucination samples, so its weighted Overall is `0.1 * LIVE = 2.00%`. It is not a replacement for the primary BFCL-v4 direct-accuracy score.

## Agentic Web Search component

Configuration: `web_search_base` (100 samples), with `SERPAPI_API_KEY=null` and the same fixed 2048-token generation limit.

| Model | Official direct accuracy | Acceptance report | BFCL Agentic | BFCL Overall |
| --- | ---: | ---: | ---: | ---: |
| G1h 7.2B | 0/100 = 0.00% | `format_invalid=100` | 0.00% | 0.00% |
| G1h 13.3B | 0/100 = 0.00% | `format_invalid=100` | 0.00% | 0.00% |

The acceptance report's first synthesized raw response for both models was `[[[]]]`, and it was rejected by the strict BFCL-v4 parser because each item must be an object containing exactly one function entry. The lower-level `trace_report.json` is more diagnostic: the first native response had ordinary text content and `tool_calls=[]`. This classifies the zero as a model/protocol failure to emit a tool call, not an extractor repair opportunity. No tool name, argument, or answer was fabricated.

## Multi-Turn component

Configuration: `multi_turn_base`, `multi_turn_miss_func`, `multi_turn_miss_param`; 600 samples per model. Both runs are still active and have no score yet. Partial raw records are retained remotely under:

- `/home/rwkv/chase/EvalScope/results/evalscope/bfcl-v4-7p2b-multiturn-nolong`
- `/home/rwkv/chase/EvalScope/results/evalscope/bfcl-v4-13p3b-multiturn-nolong`

No partial Multi-Turn result will be reported as a score. Any context overflow is recorded as `context_truncated` and excluded from valid scoring.

Memory categories were not launched after a data-size audit: each has 192 entries and serialized entry sizes of approximately 3,578--21,891 characters (`memory_kv` median 11,013; `memory_vector` median 9,092; `memory_rec_sum` median 3,925), with 37 prerequisite IDs per category. With the fixed 10,240-token service window and 2,048-token output cap, this is a confirmed context-risk case; it is recorded as skipped rather than converted into a fabricated score.

The first observed Multi-Turn sample did not hit the model context limit: each upstream request was about 633 prompt tokens. However, the parallel-candidate router fanned the sample out to 28 candidate shards, taking about 120 seconds and accumulating about 10,241 prompt tokens across requests. This is a routing fan-out performance limitation, not a context-overflow score adjustment.

## Native tool-call cross-check

For the first Web Search request, a direct request to `http://127.0.0.1:29572/v1/chat/completions` returned `finish_reason=tool_calls` and a valid `search_engine_query` call. The same request through parallel-candidate returned ordinary text with `tool_calls=[]` and was synthesized as `[[[]]]`. This confirms a routing-layer incompatibility; the native full reruns are kept separate from the router diagnostic score.

Native full reruns were stopped after prolonged external-tool execution stalls. At stop time they had completed 2/100 (7.2B) and 3/100 (13.3B), with no context errors and no official reports. These partial native runs are not scores; the direct single-request tool-call probe remains the authoritative protocol evidence.

The full Multi-Turn runs were stopped after the ETA grew to approximately 15 hours for 7.2B and 35 hours for 13.3B. At stop time, 7.2B had reached 3/600 benchmark items and 13.3B had reached 1/600. No official report was generated, so these partial runs have no score. The model endpoints remained healthy after stopping and were not restarted or modified.

## Official scorer rerun after native fallback (2026-07-31)

These are the authoritative EvalScope 1.9.1 BFCL-v4 reports for the fixed short configuration. The official `bfcl_v4.json` top-level `score` is the direct accuracy over 1648 samples. The local acceptance report is diagnostic only and does not replace the official score.

| Model | Official score | Official aggregate `OVERALL` | Direct accuracy |
| --- | ---: | ---: | ---: |
| G1h 7.2B | 51.21% | 11.29% | 844/1648 |
| G1h 13.3B | 62.26% | 14.00% | 1026/1648 |

Official subsets:

- 7.2B: `irrelevance=2.08%`, `live_simple=57.75%`, `multiple=80.00%`, `parallel=34.50%`, `parallel_multiple=46.00%`, `simple_java=35.00%`, `simple_javascript=42.00%`, `simple_python=78.25%`; aggregates `NON_LIVE=53.06%`, `LIVE=57.75%`, `HALLUCINATION=2.08%`.
- 13.3B: `irrelevance=6.25%`, `live_simple=63.18%`, `multiple=81.00%`, `parallel=80.50%`, `parallel_multiple=66.00%`, `simple_java=35.00%`, `simple_javascript=46.00%`, `simple_python=83.75%`; aggregates `NON_LIVE=70.61%`, `LIVE=63.18%`, `HALLUCINATION=6.25%`.

The aggregate `OVERALL` values are not the requested full BFCL-v4 formula: Agentic and Multi-Turn were not included in this short scored run, so those components are absent. They must not be presented as a full BFCL-v4 overall score. The official reports are remote under `/home/rwkv/chase/EvalScope/results/evalscope/bfcl-v4-*-native-fallback/reports/.../bfcl_v4.json`; experiment artifacts remain untracked.
