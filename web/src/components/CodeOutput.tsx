import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Code, FileText, AlertTriangle } from "lucide-react";
import type { GenerateResult } from "@/lib/api";

interface Props {
  result: GenerateResult | null;
}

export default function CodeOutput({ result }: Props) {
  if (!result) {
    return (
      <Card className="border-dashed">
        <CardContent className="flex items-center justify-center h-48 text-[var(--muted-foreground)] text-sm">
          Run a task to see generated code here
        </CardContent>
      </Card>
    );
  }

  const isSuccess = result.status === "success";

  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between">
        <CardTitle className="flex items-center gap-2">
          <Code className="h-4 w-4" />
          Generated Code
        </CardTitle>
        <Badge variant={isSuccess ? "success" : "destructive"}>
          {result.status}
        </Badge>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        {/* Reasoning */}
        {result.reasoning && (
          <details className="group">
            <summary className="flex items-center gap-2 cursor-pointer text-xs font-medium text-[var(--muted-foreground)] hover:text-[var(--foreground)]">
              <FileText className="h-3.5 w-3.5" />
              Plan / Reasoning
            </summary>
            <pre className="mt-2 p-3 rounded-[var(--radius)] bg-[var(--secondary)] text-xs overflow-auto max-h-60 whitespace-pre-wrap">
              {result.reasoning}
            </pre>
          </details>
        )}

        {/* Code */}
        {result.code && (
          <div className="relative">
            <pre className="p-4 rounded-[var(--radius)] bg-[#0d1117] text-sm overflow-auto max-h-[500px] font-mono leading-relaxed text-[#c9d1d9]">
              {result.code}
            </pre>
          </div>
        )}

        {/* Errors */}
        {result.error_log && (
          <div className="flex items-start gap-2 p-3 rounded-[var(--radius)] bg-[var(--destructive)]/10 border border-[var(--destructive)]/20">
            <AlertTriangle className="h-4 w-4 text-[var(--destructive)] shrink-0 mt-0.5" />
            <pre className="text-xs text-[var(--destructive)] overflow-auto whitespace-pre-wrap">
              {result.error_log}
            </pre>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
