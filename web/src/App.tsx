import { useState } from "react";
import TaskInput from "@/components/TaskInput";
import CodeOutput from "@/components/CodeOutput";
import MemoryExplorer from "@/components/MemoryExplorer";
import ConfigPanel from "@/components/ConfigPanel";
import TreeVisualizer from "@/components/TreeVisualizer";
import { Badge } from "@/components/ui/badge";
import { generateCode, generateCodeSimple, type GenerateResult } from "@/lib/api";
import { TreePine, Clock } from "lucide-react";

export default function App() {
  const [result, setResult] = useState<GenerateResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState<"full" | "simple">("full");
  const [memoryRefresh, setMemoryRefresh] = useState(0);
  const [elapsed, setElapsed] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (task: string, submitMode: "full" | "simple") => {
    setLoading(true);
    setMode(submitMode);
    setResult(null);
    setError(null);
    setElapsed(null);
    const start = Date.now();

    try {
      const fn = submitMode === "full" ? generateCode : generateCodeSimple;
      const res = await fn(task);
      setResult(res);
      setElapsed(Date.now() - start);
      // Refresh memory stats after generation
      setMemoryRefresh((n) => n + 1);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="border-b border-[var(--border)] bg-[var(--card)]">
        <div className="max-w-7xl mx-auto px-4 h-12 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <TreePine className="h-5 w-5 text-[var(--primary)]" />
            <span className="font-bold text-sm">TALM Demo</span>
            <Badge variant="outline" className="text-[10px]">
              v0.1.0
            </Badge>
          </div>
          <div className="flex items-center gap-3 text-xs text-[var(--muted-foreground)]">
            {elapsed !== null && (
              <span className="flex items-center gap-1">
                <Clock className="h-3 w-3" />
                {(elapsed / 1000).toFixed(1)}s
              </span>
            )}
            <Badge variant="secondary">MCP Server</Badge>
          </div>
        </div>
      </header>

      {/* Main */}
      <main className="max-w-7xl mx-auto p-4 grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Left column: input + output */}
        <div className="lg:col-span-2 flex flex-col gap-4">
          <TaskInput onSubmit={handleSubmit} loading={loading} />

          {error && (
            <div className="p-4 rounded-[var(--radius)] border border-[var(--destructive)]/30 bg-[var(--destructive)]/10 text-sm">
              <p className="font-medium text-[var(--destructive)]">Connection Error</p>
              <p className="text-xs text-[var(--muted-foreground)] mt-1">{error}</p>
              <p className="text-xs text-[var(--muted-foreground)] mt-2">
                Make sure the TALM MCP server is running:
              </p>
              <pre className="mt-1 text-xs bg-[var(--secondary)] p-2 rounded">
                source .venv/bin/activate{"\n"}export GOOGLE_API_KEY="..."{"\n"}python -m talm.server
              </pre>
            </div>
          )}

          <CodeOutput result={result} />
        </div>

        {/* Right column: tree + memory + config */}
        <div className="flex flex-col gap-4">
          <TreeVisualizer result={result} mode={mode} loading={loading} />
          <MemoryExplorer refreshTrigger={memoryRefresh} />
          <ConfigPanel />
        </div>
      </main>
    </div>
  );
}
