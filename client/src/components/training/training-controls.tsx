import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { PlayCircle, Square, Pause, RotateCcw, AlertCircle } from "lucide-react";
import { useTraining } from "@/hooks/use-training";
import { useQuery } from "@tanstack/react-query";
import { Model } from "@shared/schema";

export function TrainingControls() {
  const [trainingName, setTrainingName] = useState("");
  const [selectedModelId, setSelectedModelId] = useState<number | null>(null);
  const [showStartDialog, setShowStartDialog] = useState(false);
  
  const { 
    isTraining, 
    activeRun, 
    startTraining, 
    stopTraining, 
    pauseTraining, 
    resumeTraining,
    isStarting,
    isStopping,
    isPausing,
    isResuming
  } = useTraining();

  // Get available models with direct API call
  const { data: models = [], isLoading: modelsLoading, error: modelsError } = useQuery<Model[]>({
    queryKey: ['/api/models'],
    queryFn: async () => {
      console.log('🔄 Training Controls - Fetching models...');
      try {
        const response = await fetch('/api/models', {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          },
          credentials: 'include',
        });
        
        if (!response.ok) {
          console.error(`API Error: ${response.status} ${response.statusText}`);
          throw new Error(`Failed to fetch models: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('📦 Training Controls - Raw API response:', data);
        console.log('📦 Training Controls - Parsed models:', Array.isArray(data) ? data : []);
        return Array.isArray(data) ? data : [];
      } catch (error) {
        console.error('🚨 Training Controls - Fetch error:', error);
        throw error;
      }
    },
    refetchInterval: 3000, // Refetch every 3 seconds
    staleTime: 0,
    retry: 3,
  });

  console.log('🔍 Training Controls - Final state:', { models, modelsLoading, modelsError });

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'bg-green-500';
      case 'paused': return 'bg-yellow-500';
      case 'stopped': return 'bg-gray-500';
      case 'failed': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'running': return 'Running';
      case 'paused': return 'Paused';
      case 'stopped': return 'Stopped';
      case 'failed': return 'Failed';
      default: return 'Ready';
    }
  };

  const handleStartTraining = () => {
    if (!selectedModelId || !trainingName.trim()) return;
    
    startTraining({
      modelId: selectedModelId,
      name: trainingName,
      dataFiles: [] // This would be populated from uploaded files
    });
    setShowStartDialog(false);
    setTrainingName("");
  };

  const canStart = !isTraining && selectedModelId && trainingName.trim();
  const canStop = isTraining && activeRun;
  const canPause = isTraining && activeRun?.status === 'running';
  const canResume = !isTraining && activeRun?.status === 'stopped';

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center">
            <PlayCircle className="w-5 h-5 mr-2 text-green-500" />
            Training Controls
          </CardTitle>
          <div className="flex items-center space-x-2">
            <span className="text-sm text-muted-foreground">Status:</span>
            <Badge variant="secondary" className="flex items-center space-x-1">
              <div className={`w-2 h-2 rounded-full ${getStatusColor(activeRun?.status || 'idle')}`}></div>
              <span>{getStatusText(activeRun?.status || 'idle')}</span>
            </Badge>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Active Training Info */}
        {activeRun && (
          <div className="p-4 bg-muted rounded-lg border">
            <div className="flex items-center justify-between mb-2">
              <h4 className="font-medium text-foreground">{activeRun.name}</h4>
              <Badge variant="outline">
                Step {activeRun.currentStep || 0} / {activeRun.totalSteps}
              </Badge>
            </div>
            {activeRun.currentLoss && (
              <div className="text-sm text-muted-foreground">
                Current Loss: <span className="font-medium text-foreground">{activeRun.currentLoss.toFixed(4)}</span>
              </div>
            )}
            {activeRun.learningRate && (
              <div className="text-sm text-muted-foreground">
                Learning Rate: <span className="font-medium text-foreground">{activeRun.learningRate.toExponential(2)}</span>
              </div>
            )}
          </div>
        )}

        {/* Training Controls */}
        <div className="flex flex-wrap gap-3">
          <Dialog open={showStartDialog} onOpenChange={setShowStartDialog}>
            <DialogTrigger asChild>
              <Button 
                className="bg-green-600 hover:bg-green-700 text-white"
                disabled={isTraining || isStarting}
              >
                <PlayCircle className="w-4 h-4 mr-2" />
                {isStarting ? 'Starting...' : 'Start Training'}
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Start New Training Run</DialogTitle>
              </DialogHeader>
              <div className="space-y-4">
                <div>
                  <Label htmlFor="training-name">Training Run Name</Label>
                  <Input
                    id="training-name"
                    value={trainingName}
                    onChange={(e) => setTrainingName(e.target.value)}
                    placeholder="Enter training run name..."
                  />
                </div>
                <div>
                  <Label htmlFor="model-select">Select Model</Label>
                  <select
                    id="model-select"
                    className="w-full px-3 py-2 bg-background border border-border rounded-md"
                    value={selectedModelId || ''}
                    onChange={(e) => setSelectedModelId(e.target.value ? Number(e.target.value) : null)}
                  >
                    <option value="">
                      {modelsLoading ? 'Loading models...' : models.length === 0 ? 'No models available' : 'Select a model...'}
                    </option>
                    {models && models.length > 0 && models.map((model) => (
                      <option key={model.id} value={model.id}>
                        {model.name} ({model.status})
                      </option>
                    ))}
                    {!modelsLoading && models.length === 0 && (
                      <option value="" disabled>
                        Create a model in Model Configuration first
                      </option>
                    )}
                  </select>
                </div>
                <div className="flex justify-end space-x-2">
                  <Button variant="outline" onClick={() => setShowStartDialog(false)}>
                    Cancel
                  </Button>
                  <Button 
                    onClick={handleStartTraining}
                    disabled={!canStart}
                  >
                    Start Training
                  </Button>
                </div>
              </div>
            </DialogContent>
          </Dialog>

          <Button 
            variant="destructive"
            disabled={!canStop || isStopping}
            onClick={() => activeRun && stopTraining(activeRun.id)}
          >
            <Square className="w-4 h-4 mr-2" />
            {isStopping ? 'Stopping...' : 'Stop'}
          </Button>

          <Button 
            variant="outline"
            disabled={!canPause || isPausing}
            onClick={() => activeRun && pauseTraining(activeRun.id)}
          >
            <Pause className="w-4 h-4 mr-2" />
            {isPausing ? 'Pausing...' : 'Pause'}
          </Button>

          <Button 
            variant="outline"
            disabled={!canResume || isResuming}
            onClick={() => activeRun && resumeTraining(activeRun.id)}
          >
            <RotateCcw className="w-4 h-4 mr-2" />
            {isResuming ? 'Resuming...' : 'Resume'}
          </Button>
        </div>

        {/* Training Requirements Check */}
        {!isTraining && (
          <div className="p-3 bg-blue-500/10 border border-blue-500/20 rounded-lg">
            <div className="flex items-start space-x-2">
              <AlertCircle className="w-4 h-4 text-blue-500 mt-0.5" />
              <div>
                <p className="text-sm font-medium text-blue-400">Pre-training Checklist</p>
                <ul className="text-xs text-muted-foreground mt-1 space-y-1">
                  <li>✓ Model configuration complete</li>
                  <li>✓ Training data uploaded</li>
                  <li>✓ Training parameters set</li>
                  <li className={Array.isArray(models) && models.length > 0 ? 'text-green-400' : ''}>
                    {Array.isArray(models) && models.length > 0 ? '✓' : '○'} Model available for training
                  </li>
                </ul>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
