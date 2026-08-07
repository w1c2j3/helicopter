export interface EvalRecord {
  sample_index: number;
  repeat_index: number;
  pass_index: number;
  has_eval_record: boolean;
  is_passed: boolean | null;
  answer: string | null;
  ref_answer: string | null;
  fail_reason: string | null;
  context_preview?: string;
  final_stop_reason: string | null;
  final_stop_telemetry_observed: boolean;
  is_truncated: boolean;
}

export interface EvalRecordDiagnostics {
  blank_count: number;
  blank_rate: number | null;
  missing_prediction_count: number;
  missing_prediction_rate: number | null;
  truncated_count: number;
  truncation_rate: number | null;
  final_stop_telemetry_count: number;
  conditional_truncation_rate: number | null;
  missing_eval_count: number;
}

export interface EvalRecordsResponse {
  task_id: number;
  records: EvalRecord[];
  offset: number;
  limit: number | null;
  next_offset: number;
  has_more: boolean;
  total: number;
  eval_total: number;
  filtered_total: number;
  completion_total: number;
  missing_eval_count: number;
  diagnostics: EvalRecordDiagnostics;
}
