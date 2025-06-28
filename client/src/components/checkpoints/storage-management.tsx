import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { ParameterControl } from "@/components/ui/parameter-control";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { HardDrive, Settings } from "lucide-react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";

interface CheckpointStats {
  totalCheckpoints: number;
  totalSize: number;
  averageSize: number;
  bestCheckpoint?: {
    id: number;
    step: number;
    loss: number;
  };
}

interface CheckpointSettings {
  saveFrequency: number;
  maxCheckpoints: number;
  saveOptimizerState: boolean;
  saveSchedulerState: boolean;
  autoCompress: boolean;
  verifyIntegrity: boolean;
}

export function StorageManagement() {
  const [settings, setSettings] = useState<CheckpointSettings>({
    saveFrequency: 1000,
    maxCheckpoints: 10,
    saveOptimizerState: true,
    saveSchedulerState: true,
    autoCompress: false,
    verifyIntegrity: true
  });

  const { toast } = useToast();
  const queryClient = useQueryClient();

  // Get checkpoint statistics
  const { data: stats } = useQuery({
    queryKey: ['/api/checkpoints/stats'],
  });

  // Cleanup old checkpoints mutation
  const cleanupMutation = useMutation({
    mutationFn: async ({ modelId, maxCheckpoints }: { modelId?: number; maxCheckpoints: number }) => {
      const response = await apiRequest('POST', '/api/checkpoints/cleanup', {
        modelId,
        maxCheckpoints
      });
      return response.json();
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['/api/checkpoints'] });
      queryClient.invalidateQueries({ queryKey: ['/api/checkpoints/stats'] });
      toast({
        title: "Cleanup Complete",
        description: `Deleted ${data.deletedCount} old checkpoints`
      });
    },
    onError: (error) => {
      toast({
        title: "Cleanup Failed",
        description: error.message,
        variant: "destructive"
      });
    }
  });

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const updateSettings = (updates: Partial<CheckpointSettings>) => {
    setSettings(prev => ({ ...prev, ...updates }));
  };

  const storageUsed = stats?.totalSize || 0;
  const storageTotal = 50 * 1024 * 1024 * 1024; // 50GB limit
  const storagePercentage = (storageUsed / storageTotal) * 100;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Storage Management */}
      <Card className="overflow-hidden">
        <CardHeader className="bg-muted/50">
          <CardTitle className="flex items-center">
            <HardDrive className="w-5 h-5 mr-2 text-blue-500" />
            Storage Management
          </CardTitle>
        </CardHeader>
        <CardContent className="p-6 space-y-4">
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-muted-foreground">Storage Used</span>
              <span className="text-sm font-medium text-foreground">
                {formatFileSize(storageUsed)} / {formatFileSize(storageTotal)}
              </span>
            </div>
            <Progress value={storagePercentage} className="w-full" />
          </div>
          
          <div className="grid grid-cols-2 gap-4">
            <div className="text-center p-3 bg-muted rounded-lg">
              <p className="text-sm text-muted-foreground">Total Checkpoints</p>
              <p className="text-xl font-bold text-foreground">{stats?.totalCheckpoints || 0}</p>
            </div>
            <div className="text-center p-3 bg-muted rounded-lg">
              <p className="text-sm text-muted-foreground">Average Size</p>
              <p className="text-xl font-bold text-foreground">
                {formatFileSize(stats?.averageSize || 0)}
              </p>
            </div>
          </div>

          {stats?.bestCheckpoint && (
            <div className="p-3 bg-emerald-500/10 border border-emerald-500/20 rounded-lg">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-emerald-400">Best Checkpoint</span>
                <span className="text-xs text-muted-foreground">
                  Step {stats.bestCheckpoint.step.toLocaleString()}
                </span>
              </div>
              <p className="text-sm text-muted-foreground">
                Loss: {stats.bestCheckpoint.loss.toFixed(4)}
              </p>
            </div>
          )}
          
          <div className="space-y-2">
            <Button 
              variant="outline" 
              className="w-full"
              onClick={() => cleanupMutation.mutate({ maxCheckpoints: settings.maxCheckpoints })}
              disabled={cleanupMutation.isPending}
            >
              {cleanupMutation.isPending ? 'Cleaning...' : 'Clean Old Checkpoints'}
            </Button>
            <Button variant="outline" className="w-full">
              Compress Checkpoints
            </Button>
          </div>
        </CardContent>
      </Card>
      
      {/* Checkpoint Settings */}
      <Card className="overflow-hidden">
        <CardHeader className="bg-muted/50">
          <CardTitle className="flex items-center">
            <Settings className="w-5 h-5 mr-2 text-purple-500" />
            Checkpoint Settings
          </CardTitle>
        </CardHeader>
        <CardContent className="p-6 space-y-4">
          <ParameterControl
            config={{
              label: 'Save Frequency',
              description: 'Save checkpoint every N steps',
              type: 'select',
              options: [
                { value: 500, label: 'Every 500 steps' },
                { value: 1000, label: 'Every 1000 steps' },
                { value: 2000, label: 'Every 2000 steps' },
                { value: 5000, label: 'Every 5000 steps' }
              ]
            }}
            value={settings.saveFrequency}
            onChange={(value) => updateSettings({ saveFrequency: value })}
          />
          
          <ParameterControl
            config={{
              label: 'Max Checkpoints',
              description: 'Maximum number of checkpoints to keep (1-20)',
              type: 'range',
              range: { min: 1, max: 20, step: 1, default: 10 }
            }}
            value={settings.maxCheckpoints}
            onChange={(value) => updateSettings({ maxCheckpoints: value })}
          />
          
          <div className="space-y-3">
            <div className="flex items-center space-x-3">
              <Switch
                checked={settings.saveOptimizerState}
                onCheckedChange={(checked) => updateSettings({ saveOptimizerState: checked })}
              />
              <div>
                <Label className="text-sm font-medium">Save Optimizer State</Label>
                <p className="text-xs text-muted-foreground">Include optimizer state in checkpoints</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-3">
              <Switch
                checked={settings.saveSchedulerState}
                onCheckedChange={(checked) => updateSettings({ saveSchedulerState: checked })}
              />
              <div>
                <Label className="text-sm font-medium">Save Scheduler State</Label>
                <p className="text-xs text-muted-foreground">Include learning rate scheduler state</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-3">
              <Switch
                checked={settings.autoCompress}
                onCheckedChange={(checked) => updateSettings({ autoCompress: checked })}
              />
              <div>
                <Label className="text-sm font-medium">Auto-compress Old Checkpoints</Label>
                <p className="text-xs text-muted-foreground">Automatically compress checkpoints older than 7 days</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-3">
              <Switch
                checked={settings.verifyIntegrity}
                onCheckedChange={(checked) => updateSettings({ verifyIntegrity: checked })}
              />
              <div>
                <Label className="text-sm font-medium">Verify Checkpoint Integrity</Label>
                <p className="text-xs text-muted-foreground">Verify checksums when loading checkpoints</p>
              </div>
            </div>
          </div>
          
          <Button className="w-full bg-blue-600 hover:bg-blue-700">
            Apply Settings
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}
