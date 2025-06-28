import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Clock, Copy, Download } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { useToast } from "@/hooks/use-toast";
import { formatDistanceToNow } from "date-fns";

interface Generation {
  id: number;
  prompt: string;
  output: string;
  tokensGenerated: number;
  generationTime: number;
  createdAt: string;
  config: any;
}

export function GenerationHistory() {
  const { toast } = useToast();

  const { data: generations = [], isLoading } = useQuery({
    queryKey: ['/api/generations', { limit: 20 }],
  });

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    toast({
      title: "Copied",
      description: "Text copied to clipboard"
    });
  };

  const downloadGeneration = (generation: Generation) => {
    const content = `Prompt: ${generation.prompt}\n\nGenerated Text:\n${generation.output}`;
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `generation_${generation.id}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const truncateText = (text: string, maxLength: number = 100) => {
    return text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
  };

  return (
    <Card className="overflow-hidden">
      <CardHeader className="bg-muted/50">
        <CardTitle className="flex items-center">
          <Clock className="w-5 h-5 mr-2 text-blue-500" />
          Generation History
        </CardTitle>
        <p className="text-sm text-muted-foreground">Recent text generations</p>
      </CardHeader>
      <CardContent className="p-6">
        {isLoading ? (
          <div className="space-y-4">
            {[1, 2, 3].map((i) => (
              <div key={i} className="animate-pulse">
                <div className="h-20 bg-muted rounded-lg"></div>
              </div>
            ))}
          </div>
        ) : generations.length > 0 ? (
          <div className="space-y-4 max-h-96 overflow-y-auto">
            {generations.map((generation: Generation) => (
              <div key={generation.id} className="generation-item">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-foreground">
                    Generation #{generation.id}
                  </span>
                  <span className="text-xs text-muted-foreground">
                    {formatDistanceToNow(new Date(generation.createdAt), { addSuffix: true })}
                  </span>
                </div>
                
                <div className="space-y-2">
                  <div>
                    <p className="text-sm text-muted-foreground">
                      <strong>Prompt:</strong> {truncateText(generation.prompt)}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-foreground">
                      {truncateText(generation.output, 150)}
                    </p>
                  </div>
                </div>
                
                <div className="flex items-center justify-between mt-3">
                  <div className="flex items-center space-x-4 text-xs text-muted-foreground">
                    <span>Tokens: {generation.tokensGenerated}</span>
                    <span>Time: {generation.generationTime?.toFixed(1)}s</span>
                    <span>Temp: {generation.config?.temperature || 'N/A'}</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => copyToClipboard(generation.output)}
                    >
                      <Copy className="w-3 h-3" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => downloadGeneration(generation)}
                    >
                      <Download className="w-3 h-3" />
                    </Button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8 text-muted-foreground">
            <Clock className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <p>No generations yet</p>
            <p className="text-sm">Generate some text to see your history here</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
