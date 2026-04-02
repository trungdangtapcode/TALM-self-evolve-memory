import { useEffect, useState, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Database, Trash2, RefreshCw, HardDrive, Layers, Search, GitMerge } from "lucide-react";
import { getMemoryStats, clearMemory, type MemoryStats } from "@/lib/api";

interface Props {
  refreshTrigger: number; // increment to trigger refresh
}

export default function MemoryExplorer({ refreshTrigger }: Props) {
  const [stats, setStats] = useState<MemoryStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchStats = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getMemoryStats();
      setStats(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to connect");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchStats();
  }, [fetchStats, refreshTrigger]);

  const handleClear = async () => {
    if (!confirm("Clear all long-term memory records?")) return;
    try {
      await clearMemory();
      await fetchStats();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed");
    }
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Database className="h-4 w-4" />
              Long-Term Memory
            </CardTitle>
            <CardDescription>Vector database records stored by TALM agents</CardDescription>
          </div>
          <div className="flex gap-1.5">
            <Button variant="ghost" size="icon" onClick={fetchStats} disabled={loading}>
              <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
            </Button>
            <Button variant="ghost" size="icon" onClick={handleClear}>
              <Trash2 className="h-4 w-4 text-[var(--destructive)]" />
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {error ? (
          <div className="text-sm text-[var(--muted-foreground)] p-4 text-center border border-dashed border-[var(--border)] rounded-[var(--radius)]">
            <p className="text-[var(--destructive)] mb-1">Not connected</p>
            <p className="text-xs">Start the TALM server: <code className="bg-[var(--secondary)] px-1.5 py-0.5 rounded">python -m talm.server</code></p>
          </div>
        ) : stats ? (
          <div className="space-y-4">
            {/* Record count */}
            <div className="flex items-center gap-3 p-3 rounded-[var(--radius)] bg-[var(--secondary)]">
              <div className="h-10 w-10 rounded-full bg-[var(--primary)]/15 flex items-center justify-center">
                <HardDrive className="h-5 w-5 text-[var(--primary)]" />
              </div>
              <div>
                <div className="text-2xl font-bold">{stats.record_count}</div>
                <div className="text-xs text-[var(--muted-foreground)]">Memory Records</div>
              </div>
            </div>

            {/* Config details */}
            <div className="grid grid-cols-2 gap-2">
              <ConfigItem
                icon={<Search className="h-3.5 w-3.5" />}
                label="Retrieval"
                value={`top-${stats.config.top_k}, sim >= ${stats.config.similarity_threshold}`}
              />
              <ConfigItem
                icon={<GitMerge className="h-3.5 w-3.5" />}
                label="Merge"
                value={`sim >= ${stats.config.merge_threshold}`}
              />
              <ConfigItem
                icon={<Layers className="h-3.5 w-3.5" />}
                label="Embedding"
                value={String(stats.config.embedding_model)}
              />
              <ConfigItem
                icon={<Database className="h-3.5 w-3.5" />}
                label="Vector DB"
                value={String(stats.config.vector_db_provider)}
              />
            </div>

            {/* How it works */}
            <div className="space-y-2">
              <div className="text-xs font-medium text-[var(--muted-foreground)]">Memory Pipeline</div>
              <div className="flex items-center gap-1.5 flex-wrap">
                <PipelineStep label="Embed Task" badge="MiniLM" />
                <Arrow />
                <PipelineStep label="K-NN Search" badge={`top-${stats.config.top_k}`} />
                <Arrow />
                <PipelineStep label="Depth Filter" badge="same level" />
                <Arrow />
                <PipelineStep label="Return" badge={`>= ${stats.config.similarity_threshold}`} />
              </div>
              <div className="flex items-center gap-1.5 flex-wrap mt-2">
                <PipelineStep label="Task Success" badge="update" variant="success" />
                <Arrow />
                <PipelineStep label="Check Duplicates" badge={`>= ${stats.config.merge_threshold}`} variant="warning" />
                <Arrow />
                <PipelineStep label="LLM Merge or Insert" badge="consolidate" variant="success" />
              </div>
            </div>
          </div>
        ) : (
          <div className="h-32 flex items-center justify-center text-[var(--muted-foreground)] text-sm">
            Loading...
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function ConfigItem({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <div className="flex items-center gap-2 p-2 rounded-[var(--radius)] bg-[var(--secondary)]">
      <span className="text-[var(--muted-foreground)]">{icon}</span>
      <div className="min-w-0">
        <div className="text-xs text-[var(--muted-foreground)]">{label}</div>
        <div className="text-xs font-medium truncate">{value}</div>
      </div>
    </div>
  );
}

function PipelineStep({
  label,
  badge,
  variant = "outline",
}: {
  label: string;
  badge: string;
  variant?: "outline" | "success" | "warning";
}) {
  return (
    <div className="flex items-center gap-1 px-2 py-1 rounded border border-[var(--border)] bg-[var(--secondary)]">
      <span className="text-xs">{label}</span>
      <Badge variant={variant === "outline" ? "outline" : variant} className="text-[10px] px-1 py-0">
        {badge}
      </Badge>
    </div>
  );
}

function Arrow() {
  return <span className="text-[var(--muted-foreground)] text-xs">&rarr;</span>;
}
