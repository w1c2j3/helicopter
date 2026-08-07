import { getJson } from "../http";
import type { EvalRecordsResponse } from "../dtos/api/eval_records";

export function evalRecords(
  taskId: number,
  onlyWrong: boolean,
  limit?: number,
  offset = 0,
): Promise<EvalRecordsResponse> {
  const params = new URLSearchParams({
    task_id: String(taskId),
    only_wrong: String(onlyWrong),
    offset: String(offset),
  });
  if (limit !== undefined) params.set("limit", String(limit));
  return getJson<EvalRecordsResponse>(`/api/eval-records?${params.toString()}`);
}
