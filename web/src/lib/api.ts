/** TALM MCP Server API client.
 *
 * Calls the MCP server via JSON-RPC over HTTP (Streamable HTTP transport).
 * The Vite dev server proxies /mcp -> http://localhost:8000/mcp.
 */

const MCP_URL = "/mcp";

let _reqId = 0;

interface MCPResponse {
  jsonrpc: string;
  id: number;
  result?: {
    content: Array<{ type: string; text: string }>;
  };
  error?: { code: number; message: string };
}

async function callTool(name: string, args: Record<string, unknown> = {}): Promise<string> {
  const id = ++_reqId;
  const res = await fetch(MCP_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json, text/event-stream",
    },
    body: JSON.stringify({
      jsonrpc: "2.0",
      id,
      method: "tools/call",
      params: { name, arguments: args },
    }),
  });

  if (!res.ok) {
    throw new Error(`MCP server error: ${res.status} ${res.statusText}`);
  }

  const contentType = res.headers.get("content-type") || "";

  // Handle SSE (Streamable HTTP) — collect until we get a result
  if (contentType.includes("text/event-stream")) {
    const text = await res.text();
    const lines = text.split("\n");
    for (const line of lines) {
      if (line.startsWith("data: ")) {
        try {
          const data: MCPResponse = JSON.parse(line.slice(6));
          if (data.result?.content) {
            return data.result.content.map((c) => c.text).join("\n");
          }
          if (data.error) {
            throw new Error(data.error.message);
          }
        } catch {
          // skip non-JSON lines
        }
      }
    }
    throw new Error("No result in SSE stream");
  }

  // Handle plain JSON-RPC response
  const data: MCPResponse = await res.json();
  if (data.error) throw new Error(data.error.message);
  if (data.result?.content) {
    return data.result.content.map((c) => c.text).join("\n");
  }
  throw new Error("Unexpected response format");
}

// --- Public API ---

export interface GenerateResult {
  status: string;
  code: string;
  reasoning: string;
  error_log: string;
  tree_depth?: number;
}

export interface MemoryStats {
  record_count: number;
  config: Record<string, unknown>;
}

export interface TALMConfig {
  tree: { max_depth: number; initial_branching: number; decay_rate: number };
  validation: { max_retries: number };
  memory: { similarity_threshold: number; top_k: number; merge_threshold: number };
  llm: { provider: string; model: string; temperature: number };
  embedding: { provider: string; model: string };
  vector_db: { provider: string; persist_dir: string };
  sandbox: { timeout: number; enabled: boolean };
}

export async function generateCode(taskDescription: string): Promise<GenerateResult> {
  const raw = await callTool("generate_code", { task_description: taskDescription });
  return JSON.parse(raw);
}

export async function generateCodeSimple(taskDescription: string): Promise<GenerateResult> {
  const raw = await callTool("generate_code_simple", { task_description: taskDescription });
  return JSON.parse(raw);
}

export async function getConfig(): Promise<TALMConfig> {
  const raw = await callTool("get_config");
  return JSON.parse(raw);
}

export async function getMemoryStats(): Promise<MemoryStats> {
  const raw = await callTool("get_memory_stats");
  return JSON.parse(raw);
}

export async function clearMemory(): Promise<void> {
  await callTool("clear_memory");
}

export async function updateConfig(params: Record<string, unknown>): Promise<void> {
  await callTool("update_config", params);
}
