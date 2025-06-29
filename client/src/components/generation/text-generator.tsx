import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Copy, Zap, Download, Clock } from "lucide-react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import { GenerationConfigSchema, Model } from "@shared/schema";
import { useToast } from "@/hooks/use-toast";

interface GenerationResult {
  id: number;
  output: string;
  tokensGenerated: number;
  generationTime: number;
}

export function TextGenerator() {
  const [prompt, setPrompt] = useState("");
  const [output, setOutput] = useState("");
  const [selectedModelId, setSelectedModelId] = useState<number | null>(null);
  const [generationStats, setGenerationStats] = useState<{
    tokens: number;
    time: number;
    speed: number;
  } | null>(null);
  
  const { toast } = useToast();

  // Get available models with direct API call
  const { data: models = [], isLoading: modelsLoading, error: modelsError } = useQuery<Model[]>({
    queryKey: ['/api/models'],
    queryFn: async () => {
      console.log('🔄 Text Generator - Fetching models from API...');
      const response = await fetch('/api/models');
      if (!response.ok) {
        throw new Error(`Failed to fetch models: ${response.status}`);
      }
      const data = await response.json();
      console.log('📦 Text Generator - Fetched models:', data);
      return data;
    },
    refetchInterval: 5000,
    staleTime: 0, // Always refetch
  });

  console.log('🔍 Text Generator - Models:', models, 'Loading:', modelsLoading, 'Error:', modelsError);

  // Generation mutation
  const generateMutation = useMutation({
    mutationFn: async ({ prompt, modelId, config }: { prompt: string; modelId: number; config: any }) => {
      const response = await apiRequest('POST', '/api/generate', {
        prompt,
        modelId,
        config
      });
      return response.json() as Promise<GenerationResult>;
    },
    onSuccess: (result) => {
      setOutput(result.output);
      setGenerationStats({
        tokens: result.tokensGenerated,
        time: result.generationTime,
        speed: result.tokensGenerated / result.generationTime
      });
      toast({
        title: "Text Generated",
        description: `Generated ${result.tokensGenerated} tokens in ${result.generationTime.toFixed(2)}s`
      });
    },
    onError: (error) => {
      toast({
        title: "Generation Failed",
        description: error.message,
        variant: "destructive"
      });
    }
  });

  const handleGenerate = () => {
    if (!prompt.trim() || !selectedModelId) return;

    const config = GenerationConfigSchema.parse({
      temperature: 0.7,
      top_p: 0.9,
      top_k: 50,
      max_length: 500,
      repetition_penalty: 1.1,
      length_penalty: 1.0,
      no_repeat_ngram_size: 3,
      do_sample: true,
      early_stopping: false,
      num_beams: 1,
      use_multi_token_prediction: false
    });

    generateMutation.mutate({
      prompt,
      modelId: selectedModelId,
      config
    });
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    toast({
      title: "Copied",
      description: "Text copied to clipboard"
    });
  };

  const downloadOutput = () => {
    if (!output) return;
    
    const blob = new Blob([output], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `generation_${Date.now()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const promptTokenCount = Math.ceil(prompt.length / 4); // Rough estimate
  const promptCharCount = prompt.length;

  return (
    <Card className="overflow-hidden">
      <CardHeader className="bg-muted/50">
        <CardTitle className="flex items-center">
          <Zap className="w-5 h-5 mr-2 text-green-500" />
          Text Generation
        </CardTitle>
        <p className="text-sm text-muted-foreground">Generate text using your trained model</p>
      </CardHeader>
      <CardContent className="p-6 space-y-6">
        {/* Model Selection */}
        <div className="space-y-2">
          <label className="text-sm font-medium text-foreground">Select Model</label>
          <select
            className="w-full px-3 py-2 bg-background border border-border rounded-md"
            value={selectedModelId || ''}
            onChange={(e) => setSelectedModelId(e.target.value ? Number(e.target.value) : null)}
          >
            <option value="">
              {modelsLoading ? 'Loading models...' : 'Select a model...'}
            </option>
            {models.map((model) => (
              <option key={model.id} value={model.id}>
                {model.name} ({model.status})
              </option>
            ))}
            {!modelsLoading && models.length === 0 && (
              <option value="" disabled>
                No models found - Create one in Model Configuration
              </option>
            )}
            {modelsError && (
              <option value="" disabled>
                Error loading models - Check console
              </option>
            )}
          </select>
        </div>

        {/* Input Area */}
        <div className="space-y-4">
          <label className="block text-sm font-medium text-foreground">Input Prompt</label>
          <Textarea 
            placeholder="Enter your prompt here..." 
            className="min-h-32 resize-none"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
          />
          
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4 text-sm text-muted-foreground">
              <span>Tokens: <span className="text-foreground font-medium">{promptTokenCount}</span></span>
              <span>Characters: <span className="text-foreground font-medium">{promptCharCount}</span></span>
            </div>
            <Button 
              onClick={handleGenerate}
              disabled={!prompt.trim() || !selectedModelId || generateMutation.isPending}
              className="bg-blue-600 hover:bg-blue-700"
            >
              <Zap className="w-4 h-4 mr-2" />
              {generateMutation.isPending ? 'Generating...' : 'Generate'}
            </Button>
          </div>
        </div>
        
        {/* Output Area */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <label className="block text-sm font-medium text-foreground">Generated Output</label>
            <div className="flex items-center space-x-2">
              {generationStats && (
                <div className="flex items-center space-x-4 text-xs text-muted-foreground">
                  <span className="flex items-center">
                    <Clock className="w-3 h-3 mr-1" />
                    {generationStats.time.toFixed(2)}s
                  </span>
                  <span>{generationStats.tokens} tokens</span>
                  <span>{generationStats.speed.toFixed(1)} tok/s</span>
                </div>
              )}
              {output && (
                <div className="flex items-center space-x-1">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => copyToClipboard(output)}
                  >
                    <Copy className="w-4 h-4" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={downloadOutput}
                  >
                    <Download className="w-4 h-4" />
                  </Button>
                </div>
              )}
            </div>
          </div>
          
          <div className="min-h-48 p-4 bg-muted border rounded-lg overflow-y-auto">
            {generateMutation.isPending ? (
              <div className="flex items-center justify-center h-full">
                <div className="animate-pulse text-muted-foreground">Generating text...</div>
              </div>
            ) : output ? (
              <div className="whitespace-pre-wrap text-foreground">{output}</div>
            ) : (
              <div className="text-muted-foreground">Generated text will appear here...</div>
            )}
          </div>
        </div>

        {/* Generation Requirements */}
        {!selectedModelId && (
          <div className="p-3 bg-amber-500/10 border border-amber-500/20 rounded-lg">
            <p className="text-sm text-amber-400">Please select a ready model to start generating text</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
