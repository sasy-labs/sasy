import { buildMetadata, getClient, resetClient, unaryCall } from "./channel.js";
import {
  RMProxyClient,
  type ToolCallResponse,
} from "./generated/reference_monitor.js";
import type { ToolCallRequest } from "./types.js";

const ensureClient = () => getClient(RMProxyClient);

/** Reset the cached stub (call after reconfiguring). */
export function resetReferenceMonitor(): void {
  resetClient(RMProxyClient);
}

/**
 * Check authorization for a tool call before execution.
 *
 * @param request - Tool call details (function name, args, graph context)
 * @returns Authorization decision with optional denial trace and transforms
 */
export async function checkToolCall(
  request: ToolCallRequest,
): Promise<ToolCallResponse> {
  const c = ensureClient();
  const req = {
    ...request,
    inputNodeIds: request.inputNodeIds ?? [],
    metadata: request.metadata ?? [],
  };
  return unaryCall<ToolCallResponse>((cb) => c.checkToolCall(req, buildMetadata(), cb));
}
