import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Play, Zap, Loader2 } from "lucide-react";

interface Props {
  onSubmit: (task: string, mode: "full" | "simple") => void;
  loading: boolean;
}

const EXAMPLES = [
  "Write a Python function `is_palindrome(s: str) -> bool` that checks if a string is a palindrome, ignoring case and non-alphanumeric characters.",
  "Write a Python class `LRUCache` that implements a Least Recently Used cache with `get(key)` and `put(key, value)` methods. Support a configurable max capacity.",
  "Write a Python module with `merge_sort(arr)` and `binary_search(arr, target)` functions. Include a main block that demonstrates both.",
];

export default function TaskInput({ onSubmit, loading }: Props) {
  const [task, setTask] = useState(EXAMPLES[0]);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Task Input</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <textarea
          className="w-full min-h-[100px] rounded-[var(--radius)] border border-[var(--border)] bg-[var(--secondary)] p-3 text-sm text-[var(--foreground)] placeholder:text-[var(--muted-foreground)] focus:outline-none focus:ring-1 focus:ring-[var(--ring)] resize-y"
          value={task}
          onChange={(e) => setTask(e.target.value)}
          placeholder="Describe a coding task..."
        />
        <div className="flex gap-2 flex-wrap">
          {EXAMPLES.map((ex, i) => (
            <button
              key={i}
              onClick={() => setTask(ex)}
              className="text-xs px-2 py-1 rounded-full border border-[var(--border)] text-[var(--muted-foreground)] hover:text-[var(--foreground)] hover:border-[var(--primary)] transition-colors cursor-pointer"
            >
              Example {i + 1}
            </button>
          ))}
        </div>
        <div className="flex gap-2">
          <Button onClick={() => onSubmit(task, "full")} disabled={loading || !task.trim()}>
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
            Generate (Full Tree)
          </Button>
          <Button
            variant="secondary"
            onClick={() => onSubmit(task, "simple")}
            disabled={loading || !task.trim()}
          >
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Zap className="h-4 w-4" />}
            Simple Mode
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
