import type {
  LeaderboardResponse,
  MatrixCell,
  MatrixColumn,
  MatrixDomain,
  MatrixRow,
  TuningBenchmark,
} from "./dtos/api/leaderboard";
import type { MetaResponse } from "./dtos/api/meta";

const DOMAIN_SPECS = [
  {
    key: "knowledge",
    label: "Knowledge",
    title: "知识与常识",
    offset: 0,
    benchmarks: [
      "mmlu_pro_test",
      "mmlu_test",
      "mmlu_redux_test",
      "gpqa_diamond",
      "ceval_test",
      "cmmlu_test",
      "arc_challenge_test",
      "hellaswag_validation",
      "truthfulqa_mc1",
      "simpleqa_verified",
      "openbookqa_test",
      "winogrande_validation",
    ],
  },
  {
    key: "math",
    label: "Math",
    title: "数学推理",
    offset: -9,
    benchmarks: [
      "aime_2025",
      "aime_2024",
      "math_500",
      "gsm8k_test",
      "gsm_hard",
      "olympiadbench",
      "amc23",
      "minerva_math",
      "gaokao_math",
      "svamp",
    ],
  },
  {
    key: "coding",
    label: "Coding",
    title: "代码能力",
    offset: -6,
    benchmarks: [
      "humaneval",
      "humaneval_plus",
      "mbpp",
      "mbpp_plus",
      "livecodebench",
      "bigcodebench",
      "cruxeval",
      "apps",
    ],
  },
  {
    key: "agent",
    label: "Agent",
    title: "Agent 与工具使用",
    offset: -12,
    benchmarks: [
      "agentbench",
      "tau_bench",
      "toolbench",
      "browsecomp",
      "mcp_bench",
      "skillsbench",
    ],
  },
  {
    key: "instruction_following",
    label: "Instruction Following",
    title: "指令遵循",
    offset: 2,
    benchmarks: [
      "ifeval",
      "ifbench",
      "followbench",
      "multi_if",
      "sysbench",
      "wildifeval",
    ],
  },
  {
    key: "function_call",
    label: "Function Call",
    title: "函数调用",
    offset: -4,
    benchmarks: [
      "bfcl_v3",
      "apibank",
      "toolalpaca",
      "nexus",
      "function_calling",
    ],
  },
] as const;

const PARAM_SPECS = [
  {
    param: "13.3b",
    base: 64,
    current: "rwkv7-g1h-13.3b-20260710-ctx10240",
    previous: "rwkv7-g1g-13.3b-20260523-ctx8192",
  },
  {
    param: "7.2b",
    base: 57,
    current: "rwkv7-g1h-7.2b-20260710-ctx10240",
    previous: "rwkv7-g1g-7.2b-20260523-ctx8192",
  },
  {
    param: "2.9b",
    base: 49,
    current: "rwkv7-g1h-2.9b-20260710-ctx10240",
    previous: "rwkv7-g1g-2.9b-20260526-ctx8192",
  },
  {
    param: "1.5b",
    base: 42,
    current: "rwkv7-g1h-1.5b-20260710-ctx10240",
    previous: "rwkv7-g1g-1.5b-20260526-ctx8192",
  },
] as const;

const clamp = (value: number) => Math.max(1, Math.min(98, value));
const rounded = (value: number) => Math.round(value * 10) / 10;

function cell(percent: number, metric: string, samples: number): MatrixCell {
  return {
    percent: rounded(percent),
    meta: null,
    metric,
    num_samples: samples,
    created_at: "2026-07-23T00:00:00",
  };
}

function buildDomain(
  domain: (typeof DOMAIN_SPECS)[number],
  domainIndex: number,
): MatrixDomain {
  const columns: MatrixColumn[] = domain.benchmarks.map((benchmark, index) => ({
    key: `${domain.key}:${benchmark}`,
    label: benchmark,
    metric: index % 4 === 0 ? "avg@16" : "avg@1",
    eval_method: index % 3 === 0 ? "NoCoT" : "CoT",
    num_samples: 120 + index * 137,
  }));
  const rows: MatrixRow[] = [];

  PARAM_SPECS.forEach((spec, paramIndex) => {
    const previousScores = columns.map((column, benchmarkIndex) => {
      const spread = ((benchmarkIndex * 7 + paramIndex * 5 + domainIndex * 3) % 19) - 9;
      return clamp(spec.base + domain.offset + spread);
    });
    const currentScores = previousScores.map((score, benchmarkIndex) => {
      const uplift = benchmarkIndex % 7 === 0
        ? -1.2
        : 2.4 + ((benchmarkIndex + paramIndex) % 5) * 0.9;
      return clamp(score + uplift);
    });
    [
      { model: spec.current, scores: currentScores, createdAt: "2026-07-10T00:00:00" },
      { model: spec.previous, scores: previousScores, createdAt: "2026-05-23T00:00:00" },
    ].forEach(({ model, scores, createdAt }, generationIndex) => {
      rows.push({
        rank: generationIndex + 1,
        model,
        param: spec.param,
        average: rounded(scores.reduce((sum, score) => sum + score, 0) / scores.length),
        coverage: scores.length,
        cells: scores.map((score, index) => ({
          ...cell(score, columns[index].metric ?? "score", columns[index].num_samples ?? 0),
          created_at: createdAt,
        })),
      });
    });
  });

  return {
    key: domain.key,
    label: domain.label,
    title: domain.title,
    columns,
    rows,
  };
}

function buildTuningBenchmark(): TuningBenchmark {
  const columns = [
    { key: "normal-cot-t02", label: "normal · CoT · T0.2", prompt_profile: "normal", cot_mode: "CoT", sampling_config: { temperature: 0.2 } },
    { key: "normal-cot-t06", label: "normal · CoT · T0.6", prompt_profile: "normal", cot_mode: "CoT", sampling_config: { temperature: 0.6 } },
    { key: "normal-nocot-t02", label: "normal · NoCoT · T0.2", prompt_profile: "normal", cot_mode: "NoCoT", sampling_config: { temperature: 0.2 } },
    { key: "normal-nocot-t08", label: "normal · NoCoT · T0.8", prompt_profile: "normal", cot_mode: "NoCoT", sampling_config: { temperature: 0.8 } },
  ];
  const rows = PARAM_SPECS.map((spec, index) => {
    const values = columns.map((_, configIndex) => clamp(spec.base + 2 + configIndex * 1.7 - index));
    return {
      rank: index + 1,
      model: spec.current,
      param: spec.param,
      best: Math.max(...values),
      average: rounded(values.reduce((sum, value) => sum + value, 0) / values.length),
      coverage: values.length,
      cells: values.map((value) => cell(value, "avg@1", 500)),
    };
  });
  return {
    key: "mmlu_pro_test:avg@1",
    label: "mmlu_pro_test",
    metric: "avg@1",
    num_samples: 500,
    columns,
    rows,
  };
}

export function createMockMeta(): MetaResponse {
  const models = PARAM_SPECS.flatMap((spec) => [spec.current, spec.previous]);
  return {
    auto_label: "模拟数据",
    default_view: "benchmark_detail_latest",
    table_views: [{ key: "benchmark_detail_latest", label: "当前代 vs 上一代" }],
    domain_groups: DOMAIN_SPECS.map(({ key, label, title }) => ({ key, label, title })),
    models,
    model_choices: models,
    entry_count: models.length,
    errors: [],
  };
}

export function createMockLeaderboard(): LeaderboardResponse {
  const domains = DOMAIN_SPECS.map(buildDomain);
  const modelCount = PARAM_SPECS.length * 2;
  return {
    view: "benchmark_detail_latest",
    view_label: "当前代 vs 上一代",
    is_delta: false,
    is_field_avg: false,
    param_columns: [],
    interaction_meta: {},
    domains: [],
    overview: null,
    selection: {
      dropdown_value: "模拟数据",
      selected_label: "模拟数据",
      auto_selected: true,
      model_sequence: PARAM_SPECS.flatMap((spec) => [spec.current, spec.previous]),
      skipped_small_params: 0,
      auto_label: "模拟数据",
    },
    charts: {
      knowledge: null,
      math: null,
      instruction_following: null,
      coding: null,
      agent: null,
    },
    matrix: {
      model_count: modelCount,
      benchmark_count: domains.reduce((sum, domain) => sum + domain.columns.length, 0),
      domains,
    },
    tuning_matrix: {
      benchmark_count: 1,
      benchmarks: [buildTuningBenchmark()],
    },
    errors: [],
  };
}
