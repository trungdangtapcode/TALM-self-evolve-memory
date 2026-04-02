/** TALM REST API client.
 *
 * Calls the TALM server's REST endpoints (not MCP protocol).
 * Vite dev server proxies /api -> http://localhost:8000/api.
 */

const BASE = "/api";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

async function post<T>(path: string, body: Record<string, unknown>): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error || res.statusText);
  }
  return res.json();
}

async function put<T>(path: string, body: Record<string, unknown>): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

// --- Types ---

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

// --- Public API ---

export async function generateCode(taskDescription: string): Promise<GenerateResult> {
  return post("/generate", { task_description: taskDescription, mode: "full" });
}

export async function generateCodeSimple(taskDescription: string): Promise<GenerateResult> {
  return post("/generate", { task_description: taskDescription, mode: "simple" });
}

export async function getConfig(): Promise<TALMConfig> {
  return get("/config");
}

export async function getMemoryStats(): Promise<MemoryStats> {
  return get("/memory/stats");
}

export async function clearMemory(): Promise<void> {
  await post("/memory/clear", {});
}

export async function updateConfig(params: Record<string, unknown>): Promise<void> {
  await put("/config", params);
}

export async function healthCheck(): Promise<boolean> {
  try {
    await get("/health");
    return true;
  } catch {
    return false;
  }
}
