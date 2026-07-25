export interface AdminEvalOptionsResponse {
  jobs: { name: string; domain: string }[];
  domains: string[];
  model_select: string[];
  worker_profile: string[];
  protocol: string[];
  run_mode: string[];
  configs: string[];
  model_options?: {
    name: string;
    weight_path: string | null;
    served_model_name: string | null;
    configs: string[];
    runtime: "local-vllm-rwkv";
  }[];
  gpu_options?: {
    id: string;
    name: string;
    memory_total_mib: number;
    memory_used_mib: number;
  }[];
}
