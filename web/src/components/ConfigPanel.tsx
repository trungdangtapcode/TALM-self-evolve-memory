import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Settings, TreePine, Brain, Shield, Cpu } from "lucide-react";
import { getConfig, type TALMConfig } from "@/lib/api";

export default function ConfigPanel() {
  const [config, setConfig] = useState<TALMConfig | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    getConfig().then(setConfig).catch(() => setError(true));
  }, []);

  if (error || !config) {
    return (
      <Card className="border-dashed">
        <CardContent className="flex items-center justify-center h-32 text-sm text-[var(--muted-foreground)]">
          {error ? "Server not connected" : "Loading config..."}
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Settings className="h-4 w-4" />
          System Config
        </CardTitle>
        <CardDescription>Live hyperparameters from the TALM server</CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        {/* Tree */}
        <Section icon={<TreePine className="h-3.5 w-3.5" />} title="Tree Structure">
          <Param label="max_depth (m)" value={config.tree.max_depth} />
          <Param label="branching (n)" value={config.tree.initial_branching} />
          <Param label="decay (k)" value={config.tree.decay_rate} />
          <Param label="retries (r)" value={config.validation.max_retries} />
        </Section>

        {/* LLM */}
        <Section icon={<Cpu className="h-3.5 w-3.5" />} title="LLM">
          <div className="col-span-2 flex items-center gap-2">
            <Badge variant="outline">{config.llm.provider}</Badge>
            <span className="text-xs font-mono">{config.llm.model}</span>
            <Badge variant="secondary">temp={config.llm.temperature}</Badge>
          </div>
        </Section>

        {/* Embedding */}
        <Section icon={<Brain className="h-3.5 w-3.5" />} title="Embedding">
          <div className="col-span-2 flex items-center gap-2">
            <Badge variant="outline">{config.embedding.provider}</Badge>
            <span className="text-xs font-mono">{config.embedding.model}</span>
          </div>
        </Section>

        {/* Sandbox */}
        <Section icon={<Shield className="h-3.5 w-3.5" />} title="Sandbox">
          <div className="col-span-2 flex items-center gap-2">
            <Badge variant={config.sandbox.enabled ? "success" : "destructive"}>
              {config.sandbox.enabled ? "enabled" : "disabled"}
            </Badge>
            <span className="text-xs text-[var(--muted-foreground)]">
              timeout={config.sandbox.timeout}s
            </span>
          </div>
        </Section>
      </CardContent>
    </Card>
  );
}

function Section({
  icon,
  title,
  children,
}: {
  icon: React.ReactNode;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="p-2.5 rounded-[var(--radius)] bg-[var(--secondary)]">
      <div className="flex items-center gap-1.5 mb-2 text-xs font-medium text-[var(--muted-foreground)]">
        {icon} {title}
      </div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-1">{children}</div>
    </div>
  );
}

function Param({ label, value }: { label: string; value: number }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-xs text-[var(--muted-foreground)]">{label}</span>
      <span className="text-sm font-bold text-[var(--primary)]">{value}</span>
    </div>
  );
}
