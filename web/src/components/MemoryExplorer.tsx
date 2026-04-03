import { useEffect, useState, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Database, Trash2, RefreshCw, HardDrive, ChevronDown, ChevronRight,
  FileText, Code, Brain, Layers,
} from "lucide-react";
import {
  getMemoryStats, getMemoryRecords, clearMemory,
  type MemoryStats, type MemoryRecord,
} from "@/lib/api";

interface Props {
  refreshTrigger: number;
}

export default function MemoryExplorer({ refreshTrigger }: Props) {
  const [stats, setStats] = useState<MemoryStats | null>(null);
  const [records, setRecords] = useState<MemoryRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [s, r] = await Promise.all([getMemoryStats(), getMemoryRecords()]);
      setStats(s);
      setRecords(r);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to connect");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData, refreshTrigger]);

  const handleClear = async () => {
    if (!confirm("Clear all long-term memory records?")) return;
    try {
      await clearMemory();
      await fetchData();
      setExpandedId(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed");
    }
  };

  const toggle = (id: string) => setExpandedId(expandedId === id ? null : id);

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Database className="h-4 w-4" />
              Long-Term Memory
            </CardTitle>
            <CardDescription>Browse all records stored in vector DB</CardDescription>
          </div>
          <div className="flex gap-1.5">
            <Button variant="ghost" size="icon" onClick={fetchData} disabled={loading}>
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
            <p className="text-xs">
              Start the server:{" "}
              <code className="bg-[var(--secondary)] px-1.5 py-0.5 rounded">python -m talm.server</code>
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            {/* Stats bar */}
            {stats && (
              <div className="flex items-center gap-3 p-2.5 rounded-[var(--radius)] bg-[var(--secondary)]">
                <HardDrive className="h-4 w-4 text-[var(--primary)]" />
                <span className="text-sm font-bold">{stats.record_count}</span>
                <span className="text-xs text-[var(--muted-foreground)]">records</span>
                <span className="text-xs text-[var(--muted-foreground)] ml-auto">
                  {String(stats.config.vector_db_provider)} / {String(stats.config.embedding_model)}
                </span>
              </div>
            )}

            {/* Records list */}
            {records.length === 0 ? (
              <div className="py-8 text-center text-sm text-[var(--muted-foreground)] border border-dashed border-[var(--border)] rounded-[var(--radius)]">
                No memory records yet. Generate code to populate.
              </div>
            ) : (
              <div className="space-y-1.5 max-h-[600px] overflow-y-auto">
                {records.map((rec) => (
                  <RecordItem
                    key={rec.id}
                    record={rec}
                    expanded={expandedId === rec.id}
                    onToggle={() => toggle(rec.id)}
                  />
                ))}
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function RecordItem({
  record,
  expanded,
  onToggle,
}: {
  record: MemoryRecord;
  expanded: boolean;
  onToggle: () => void;
}) {
  return (
    <div className="rounded-[var(--radius)] border border-[var(--border)] overflow-hidden">
      {/* Header row — always visible */}
      <button
        onClick={onToggle}
        className="w-full flex items-center gap-2 p-2.5 text-left hover:bg-[var(--secondary)] transition-colors cursor-pointer"
      >
        {expanded ? (
          <ChevronDown className="h-3.5 w-3.5 text-[var(--muted-foreground)] shrink-0" />
        ) : (
          <ChevronRight className="h-3.5 w-3.5 text-[var(--muted-foreground)] shrink-0" />
        )}
        <span className="text-xs truncate flex-1">{record.task_description}</span>
        <Badge variant="outline" className="text-[10px] px-1.5 py-0 shrink-0">
          depth={record.tree_depth}
        </Badge>
      </button>

      {/* Expanded detail */}
      {expanded && (
        <div className="border-t border-[var(--border)] bg-[var(--secondary)] p-3 space-y-3">
          {/* ID */}
          <div className="flex items-center gap-2">
            <span className="text-[10px] text-[var(--muted-foreground)] font-mono">{record.id}</span>
          </div>

          {/* Task */}
          <Section icon={<FileText className="h-3 w-3" />} label="Task Description">
            <p className="text-xs whitespace-pre-wrap">{record.task_description}</p>
          </Section>

          {/* Reasoning */}
          <Section icon={<Brain className="h-3 w-3" />} label="Reasoning Trace">
            <pre className="text-xs whitespace-pre-wrap text-[var(--muted-foreground)] max-h-40 overflow-y-auto">
              {record.reasoning_trace}
            </pre>
          </Section>

          {/* Code */}
          <Section icon={<Code className="h-3 w-3" />} label="Generated Code">
            <pre className="text-xs font-mono bg-[#0d1117] text-[#c9d1d9] p-3 rounded max-h-60 overflow-auto">
              {record.generated_code}
            </pre>
          </Section>

          {/* Metadata */}
          <div className="flex items-center gap-2">
            <Layers className="h-3 w-3 text-[var(--muted-foreground)]" />
            <span className="text-[10px] text-[var(--muted-foreground)]">tree_depth={record.tree_depth}</span>
          </div>
        </div>
      )}
    </div>
  );
}

function Section({
  icon,
  label,
  children,
}: {
  icon: React.ReactNode;
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <div className="flex items-center gap-1.5 mb-1 text-[10px] font-medium text-[var(--muted-foreground)] uppercase tracking-wide">
        {icon} {label}
      </div>
      {children}
    </div>
  );
}
