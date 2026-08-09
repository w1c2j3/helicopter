export type BenchmarkDomainKey = "knowledge" | "math" | "coding" | "instruction_following";

export type BenchmarkCatalogItem = {
  key: string;
  label: string;
  domain: BenchmarkDomainKey;
  aliases: string[];
  samples: number | null;
  deferred?: string;
};

export const BENCHMARK_DOMAIN_LABELS: Record<BenchmarkDomainKey, string> = {
  knowledge: "Knowledge",
  math: "Math / Reasoning",
  coding: "Coding",
  instruction_following: "Instruction Following",
};

export const BENCHMARK_CATALOG: BenchmarkCatalogItem[] = [
  { key: "mmlu", label: "MMLU", domain: "knowledge", aliases: ["mmlu", "mmlu_test"], samples: 14042 },
  { key: "mmlu_pro", label: "MMLU-Pro", domain: "knowledge", aliases: ["mmlu_pro", "mmlu_pro_test"], samples: 12032 },
  { key: "mmlu_redux", label: "MMLU-Redux", domain: "knowledge", aliases: ["mmlu_redux", "mmlu_redux_test"], samples: 5431 },
  {
    key: "mmlu_sr_question_and_answer",
    label: "MMLU-SR Question+Answer",
    domain: "knowledge",
    aliases: ["mmlu_sr_question_and_answer", "mmlu_sr_question_and_answer_test"],
    samples: 14042,
  },
  { key: "gpqa_diamond", label: "GPQA-Diamond", domain: "knowledge", aliases: ["gpqa_diamond"], samples: 198 },
  { key: "gpqa_main", label: "GPQA-Main", domain: "knowledge", aliases: ["gpqa_main"], samples: 448 },
  {
    key: "gpqa_extended",
    label: "GPQA-Extended",
    domain: "knowledge",
    aliases: ["gpqa_extended"],
    samples: 546,
    deferred: "当前跳过",
  },
  { key: "arc_challenge", label: "ARC-Challenge", domain: "knowledge", aliases: ["arc_challenge", "arc_challenge_test"], samples: 1172 },
  { key: "arc_easy", label: "ARC-Easy", domain: "knowledge", aliases: ["arc_easy", "arc_easy_test"], samples: 2376 },
  { key: "hellaswag", label: "HellaSwag", domain: "knowledge", aliases: ["hellaswag", "hellaswag_validation"], samples: 10042 },
  { key: "bbh", label: "BBH", domain: "knowledge", aliases: ["bbh", "bbh_mcq", "bbh_mcq_test"], samples: 4070 },
  { key: "agieval", label: "AGIEval", domain: "knowledge", aliases: ["agieval", "agieval_mcq", "agieval_mcq_test"], samples: 5940 },
  { key: "truthfulqa_mc1", label: "TruthfulQA-MC1", domain: "knowledge", aliases: ["truthfulqa_mc1", "truthfulqa_mc1_validation"], samples: 817 },
  { key: "winogrande", label: "WinoGrande", domain: "knowledge", aliases: ["winogrande", "winogrande_validation"], samples: 1267 },
  { key: "openbookqa", label: "OpenBookQA", domain: "knowledge", aliases: ["openbookqa", "openbookqa_test"], samples: 500 },
  { key: "commonsense_qa", label: "CommonsenseQA", domain: "knowledge", aliases: ["commonsense_qa", "commonsense_qa_validation", "commonsenseqa"], samples: 1221 },
  { key: "ceval", label: "C-Eval", domain: "knowledge", aliases: ["ceval", "ceval_test"], samples: 12342 },
  { key: "cmmlu", label: "CMMLU", domain: "knowledge", aliases: ["cmmlu", "cmmlu_test"], samples: 11582 },
  { key: "kmmlu", label: "KMMLU", domain: "knowledge", aliases: ["kmmlu", "kmmlu_test"], samples: 35030 },
  { key: "medqa", label: "MedQA", domain: "knowledge", aliases: ["medqa", "medqa_test"], samples: 1273 },
  { key: "medmcqa", label: "MedMCQA", domain: "knowledge", aliases: ["medmcqa", "medmcqa_validation"], samples: 4183 },

  { key: "gsm8k", label: "GSM8K", domain: "math", aliases: ["gsm8k", "gsm8k_test"], samples: 1319 },
  { key: "math_500", label: "Math-500", domain: "math", aliases: ["math_500", "math_500_test"], samples: 500 },
  { key: "aime24", label: "AIME24", domain: "math", aliases: ["aime24", "aime24_test", "aime_2024"], samples: 30 },
  { key: "aime25", label: "AIME25", domain: "math", aliases: ["aime25", "aime25_test", "aime_2025"], samples: 30 },
  { key: "amc23", label: "AMC23", domain: "math", aliases: ["amc23", "amc23_test"], samples: 40 },
  { key: "olympiadbench", label: "OlympiadBench", domain: "math", aliases: ["olympiadbench", "olympiadbench_test"], samples: 675 },
  { key: "minerva_math", label: "Minerva Math", domain: "math", aliases: ["minerva_math", "minerva_math_test"], samples: 272 },
  { key: "svamp", label: "SVAMP", domain: "math", aliases: ["svamp", "svamp_test"], samples: 1000 },
  { key: "beyond_aime", label: "Beyond-AIME", domain: "math", aliases: ["beyond_aime", "beyond_aime_test"], samples: 100 },
  { key: "brumo25", label: "BRUMO25", domain: "math", aliases: ["brumo25", "brumo25_test"], samples: 30 },
  { key: "hmmt_feb25", label: "HMMT Feb 2025", domain: "math", aliases: ["hmmt_feb25", "hmmt_feb25_test"], samples: 30 },
  { key: "math_odyssey", label: "Math Odyssey", domain: "math", aliases: ["math_odyssey", "math_odyssey_test"], samples: 387 },
  { key: "comp_math_24_25", label: "COMP-MATH-24/25", domain: "math", aliases: ["comp_math_24_25", "comp_math_24_25_test"], samples: 256 },
  { key: "gaokao2023en", label: "Gaokao 2023 English", domain: "math", aliases: ["gaokao2023en", "gaokao2023en_test"], samples: 385 },
  { key: "answer_judge", label: "Answer Judge", domain: "math", aliases: ["answer_judge", "answer_judge_test"], samples: 200 },
  { key: "simpleqa_verified", label: "SimpleQA Verified", domain: "math", aliases: ["simpleqa_verified"], samples: 1000 },

  { key: "human_eval", label: "HumanEval", domain: "coding", aliases: ["human_eval", "human_eval_test", "humaneval"], samples: 164 },
  { key: "human_eval_cn", label: "HumanEval-CN", domain: "coding", aliases: ["human_eval_cn", "human_eval_cn_test"], samples: 164 },
  { key: "human_eval_fix", label: "HumanEval-Fix", domain: "coding", aliases: ["human_eval_fix", "human_eval_fix_test"], samples: 164 },
  { key: "human_eval_plus", label: "HumanEval-Plus", domain: "coding", aliases: ["human_eval_plus", "human_eval_plus_test", "humaneval_plus"], samples: 164 },
  { key: "mbpp", label: "MBPP", domain: "coding", aliases: ["mbpp", "mbpp_test"], samples: 427 },
  { key: "mbpp_plus", label: "MBPP-Plus", domain: "coding", aliases: ["mbpp_plus", "mbpp_plus_test"], samples: 378 },
  { key: "livecodebench", label: "LiveCodeBench", domain: "coding", aliases: ["livecodebench", "livecodebench_test"], samples: 1055 },
  {
    key: "swe_bench_verified",
    label: "SWE-bench Verified",
    domain: "coding",
    aliases: ["swe_bench_verified", "swe_bench_verified_test", "swebench_verified"],
    samples: 500,
    deferred: "之后单测",
  },
  {
    key: "swe_bench_multilingual",
    label: "SWE-bench Multilingual",
    domain: "coding",
    aliases: ["swe_bench_multilingual", "swe_bench_multilingual_test", "swebench_multilingual"],
    samples: 300,
    deferred: "之后单测",
  },
  {
    key: "swe_bench_pro",
    label: "SWE-bench Pro",
    domain: "coding",
    aliases: ["swe_bench_pro", "swe_bench_pro_test", "swebench_pro"],
    samples: null,
    deferred: "之后单测",
  },

  { key: "ifeval", label: "IFEval", domain: "instruction_following", aliases: ["ifeval", "ifeval_test"], samples: 541 },
  { key: "ifbench", label: "IFBench", domain: "instruction_following", aliases: ["ifbench", "ifbench_test"], samples: 300 },
];

/**
 * Compact, externally comparable dashboard scope.
 *
 * Keep BENCHMARK_CATALOG as the complete registry used by admin and mock-data
 * tooling.  The public research matrix deliberately uses this smaller list so
 * reducing the visible dashboard never deletes benchmark support or results.
 */
export const CORE_BENCHMARK_KEYS = [
  "mmlu",
  "mmlu_pro",
  "gpqa_diamond",
  "arc_challenge",
  "hellaswag",
  "bbh",
  "truthfulqa_mc1",
  "ceval",
  "gsm8k",
  "math_500",
  "aime24",
  "aime25",
  "amc23",
  "olympiadbench",
  "human_eval",
  "human_eval_plus",
  "mbpp_plus",
  "livecodebench",
  "ifeval",
  "ifbench",
] as const;

const benchmarkCatalogByKey = new Map(
  BENCHMARK_CATALOG.map((benchmark) => [benchmark.key, benchmark]),
);

export const CORE_BENCHMARK_CATALOG: BenchmarkCatalogItem[] = CORE_BENCHMARK_KEYS.map((key) => {
  const benchmark = benchmarkCatalogByKey.get(key);
  if (!benchmark) throw new Error(`Core benchmark is not registered: ${key}`);
  return benchmark;
});

export const BENCHMARK_DOMAIN_ORDER: BenchmarkDomainKey[] = [
  "knowledge",
  "math",
  "coding",
  "instruction_following",
];

export function normalizeBenchmarkName(value: string): string {
  return value.trim().toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "");
}

export function benchmarkMatches(item: BenchmarkCatalogItem, value: string): boolean {
  const normalized = normalizeBenchmarkName(value);
  return [item.key, ...item.aliases].some((alias) => normalizeBenchmarkName(alias) === normalized);
}
