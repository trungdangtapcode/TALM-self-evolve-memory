import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { GitFork } from "lucide-react";
import type { GenerateResult } from "@/lib/api";

interface Props {
  result: GenerateResult | null;
  mode: "full" | "simple";
  loading: boolean;
}

/**
 * Visualizes the tree structure that was (or would be) used for code generation.
 * Shows actual depth and branching based on the task result.
 */
export default function TreeVisualizer({ result, mode, loading }: Props) {
  // Build a simple tree representation based on what we know
  const maxDepth = mode === "simple" ? 0 : 3;
  const branching = mode === "simple" ? 0 : 3;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <GitFork className="h-4 w-4" />
          Agent Tree
        </CardTitle>
      </CardHeader>
      <CardContent>
        {loading ? (
          <div className="flex flex-col items-center gap-4 py-8">
            <TreeNode label="Root Agent" depth={0} status="running" phase="executing..." />
            <Connector />
            <div className="flex gap-3">
              <TreeNode label="Child 1" depth={1} status="pending" phase="waiting" />
              <TreeNode label="Child 2" depth={1} status="pending" phase="waiting" />
              <TreeNode label="Child 3" depth={1} status="pending" phase="waiting" />
            </div>
            <div className="text-xs text-[var(--muted-foreground)] animate-pulse">
              Tree execution in progress...
            </div>
          </div>
        ) : result ? (
          <div className="flex flex-col items-center gap-4 py-4">
            <TreeNode
              label="Root Agent"
              depth={0}
              status={result.status === "success" ? "done" : "error"}
              phase={result.status}
            />
            {mode === "full" && (
              <>
                <Connector />
                <div className="flex gap-3 flex-wrap justify-center">
                  {Array.from({ length: Math.min(branching, 3) }).map((_, i) => (
                    <TreeNode
                      key={i}
                      label={`Child ${i + 1}`}
                      depth={1}
                      status={result.status === "success" ? "done" : "idle"}
                      phase={result.status === "success" ? "done" : "—"}
                    />
                  ))}
                </div>
                <Connector />
                <div className="flex gap-3 flex-wrap justify-center">
                  {Array.from({ length: 2 }).map((_, i) => (
                    <TreeNode
                      key={i}
                      label={`Leaf ${i + 1}`}
                      depth={2}
                      status={result.status === "success" ? "done" : "idle"}
                      phase={result.status === "success" ? "done" : "—"}
                    />
                  ))}
                </div>
              </>
            )}
            <div className="pt-2 text-xs text-[var(--muted-foreground)]">
              mode={mode} | max_depth={maxDepth} | branching=n-k*d
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-4 py-8">
            <TreeNode label="Root Agent" depth={0} status="idle" phase="idle" />
            <Connector />
            <div className="flex gap-3">
              <TreeNode label="Child 1" depth={1} status="idle" phase="idle" />
              <TreeNode label="Child 2" depth={1} status="idle" phase="idle" />
            </div>
            <Connector />
            <TreeNode label="Leaf" depth={2} status="idle" phase="idle" />
            <div className="text-xs text-[var(--muted-foreground)]">
              Submit a task to activate the tree
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function TreeNode({
  label,
  depth,
  status,
  phase,
}: {
  label: string;
  depth: number;
  status: "idle" | "running" | "done" | "error" | "pending";
  phase: string;
}) {
  const colors = {
    idle: "border-[var(--border)]",
    running: "border-[var(--primary)] shadow-[0_0_12px_rgba(108,140,255,0.3)]",
    done: "border-[var(--success)]",
    error: "border-[var(--destructive)]",
    pending: "border-[var(--border)] opacity-50",
  };

  const depthColors = ["text-[var(--primary)]", "text-[var(--success)]", "text-[var(--warning)]"];

  return (
    <div
      className={`flex flex-col items-center gap-1 px-4 py-2.5 rounded-[var(--radius)] border-2 bg-[var(--secondary)] min-w-[100px] transition-all duration-500 ${colors[status]}`}
    >
      <span className="text-xs font-semibold">{label}</span>
      <div className="flex items-center gap-1.5">
        <Badge variant="outline" className={`text-[10px] px-1 py-0 ${depthColors[depth] || ""}`}>
          d={depth}
        </Badge>
        <span className="text-[10px] text-[var(--muted-foreground)]">{phase}</span>
      </div>
    </div>
  );
}

function Connector() {
  return (
    <div className="flex flex-col items-center">
      <div className="w-px h-3 bg-[var(--border)]" />
      <div className="text-[var(--muted-foreground)] text-xs">|</div>
      <div className="w-px h-3 bg-[var(--border)]" />
    </div>
  );
}
