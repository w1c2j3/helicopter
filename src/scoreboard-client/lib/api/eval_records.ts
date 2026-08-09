import { getJson } from "../http";
import type { EvalRecordOutcome, EvalRecordsResponse } from "../dtos/api/eval_records";

export function evalRecords(
  taskId: number,
  onlyWrong: boolean,
  limit?: number,
  offset = 0,
  outcome: EvalRecordOutcome = "all",
): Promise<EvalRecordsResponse> {
  const params = new URLSearchParams({
    task_id: String(taskId),
    only_wrong: String(onlyWrong),
    offset: String(offset),
    outcome,
  });
  if (limit !== undefined) params.set("limit", String(limit));
  return getJson<EvalRecordsResponse>(`/api/eval-records?${params.toString()}`);
}
