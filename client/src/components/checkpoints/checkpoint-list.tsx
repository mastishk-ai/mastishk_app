import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle, AlertDialogTrigger } from "@/components/ui/alert-dialog";
import { Save, Upload, Download, Info, Trash2, Plus } from "lucide-react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";
import { formatDistanceToNow } from "date-fns";

interface Checkpoint {
  id: number;
  name: string;
  step: number;
  loss: number;
  fileSize: number;
  isBest: boolean;
  createdAt: string;
  metadata?: any;
}

export function CheckpointList() {
  const [selectedModelId, setSelectedModelId] = useState<number | null>(null);
  const { toast } = useToast();
  const queryClient = useQueryClient();

  // Get checkpoints
  const { data: checkpoints = [], isLoading } = useQuery({
    queryKey: selectedModelId 
      ? ['/api/checkpoints', { modelId: selectedModelId }]
      : ['/api/checkpoints'],
  });

  // Get models for selection
  const { data: models = [] } = useQuery({
    queryKey: ['/api/models'],
  });

  // Load checkpoint mutation
  const loadCheckpointMutation = useMutation({
    mutationFn: async (id: number) => {
      const response = await apiRequest('POST', `/api/checkpoints/${id}/load`);
      return response.json();
    },
    onSuccess: () => {
      toast({
        title: "Checkpoint Loaded",
        description: "Model state has been restored from checkpoint"
      });
    },
    onError: (error) => {
      toast({
        title: "Load Failed",
        description: error.message,
        variant: "destructive"
      });
    }
  });

  // Create checkpoint mutation
  const createCheckpointMutation = useMutation({
    mutationFn: async (data: { modelId: number; name: string; step: number; loss: number }) => {
      const response = await apiRequest('POST', '/api/checkpoints', data);
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/checkpoints'] });
      toast({
        title: "Checkpoint Created",
        description: "Model checkpoint has been saved successfully"
      });
    },
    onError: (error) => {
      toast({
        title: "Create Failed",
        description: error.message,
        variant: "destructive"
      });
    }
  });

  // Delete checkpoint mutation
  const deleteCheckpointMutation = useMutation({
    mutationFn: async (id: number) => {
      const response = await apiRequest('DELETE', `/api/checkpoints/${id}`);
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/checkpoints'] });
      toast({
        title: "Checkpoint Deleted",
        description: "Checkpoint has been permanently removed"
      });
    },
    onError: (error) => {
      toast({
        title: "Delete Failed",
        description: error.message,
        variant: "destructive"
      });
    }
  });

  const handleCreateCheckpoint = () => {
    if (!selectedModelId) {
      toast({
        title: "Select Model",
        description: "Please select a model to create a checkpoint for",
        variant: "destructive"
      });
      return;
    }

    const currentStep = Math.floor(Math.random() * 1000) + 1;
    const currentLoss = Math.random() * 2 + 0.1;
    
    createCheckpointMutation.mutate({
      modelId: selectedModelId,
      name: `Manual Checkpoint Step ${currentStep}`,
      step: currentStep,
      loss: parseFloat(currentLoss.toFixed(4))
    });
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getTimeAgo = (dateString: string) => {
    return formatDistanceToNow(new Date(dateString), { addSuffix: true });
  };

  return (
    <Card className="overflow-hidden">
      <CardHeader className="bg-muted/50">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center">
            <Save className="w-5 h-5 mr-2 text-blue-500" />
            Model Checkpoints
          </CardTitle>
          <div className="flex items-center space-x-2">
            <select
              className="px-3 py-1 bg-background border border-border rounded text-sm"
              value={selectedModelId || ''}
              onChange={(e) => setSelectedModelId(e.target.value ? Number(e.target.value) : null)}
            >
              <option value="">All Models</option>
              {models.map((model: any) => (
                <option key={model.id} value={model.id}>
                  {model.name}
                </option>
              ))}
            </select>
            <Button 
              size="sm" 
              className="bg-blue-600 hover:bg-blue-700"
              onClick={handleCreateCheckpoint}
              disabled={createCheckpointMutation.isPending}
            >
              <Plus className="w-4 h-4 mr-2" />
              {createCheckpointMutation.isPending ? 'Creating...' : 'Create Checkpoint'}
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="p-6">
        {isLoading ? (
          <div className="space-y-4">
            {[1, 2, 3].map((i) => (
              <div key={i} className="animate-pulse">
                <div className="h-24 bg-muted rounded-lg"></div>
              </div>
            ))}
          </div>
        ) : checkpoints.length > 0 ? (
          <div className="space-y-4">
            {checkpoints.map((checkpoint: Checkpoint) => (
              <div key={checkpoint.id} className="checkpoint-item">
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-3 mb-2">
                      <h4 className="text-lg font-semibold text-foreground">{checkpoint.name}</h4>
                      {checkpoint.isBest && (
                        <Badge className="bg-emerald-600 text-white">Best</Badge>
                      )}
                      <Badge variant="outline">
                        Auto
                      </Badge>
                    </div>
                    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 text-sm">
                      <div>
                        <span className="text-muted-foreground">Step:</span>
                        <span className="text-foreground font-medium ml-1">{checkpoint.step.toLocaleString()}</span>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Loss:</span>
                        <span className="text-foreground font-medium ml-1">{checkpoint.loss?.toFixed(3) || 'N/A'}</span>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Size:</span>
                        <span className="text-foreground font-medium ml-1">{formatFileSize(checkpoint.fileSize || 0)}</span>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Created:</span>
                        <span className="text-foreground font-medium ml-1">{getTimeAgo(checkpoint.createdAt)}</span>
                      </div>
                    </div>
                    {checkpoint.metadata?.description && (
                      <p className="text-sm text-muted-foreground mt-2">{checkpoint.metadata.description}</p>
                    )}
                  </div>
                  <div className="flex items-center space-x-2 ml-4">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => loadCheckpointMutation.mutate(checkpoint.id)}
                      disabled={loadCheckpointMutation.isPending}
                      title="Load checkpoint"
                    >
                      <Upload className="w-4 h-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      title="Download"
                    >
                      <Download className="w-4 h-4" />
                    </Button>
                    <Dialog>
                      <DialogTrigger asChild>
                        <Button variant="ghost" size="sm" title="View details">
                          <Info className="w-4 h-4" />
                        </Button>
                      </DialogTrigger>
                      <DialogContent>
                        <DialogHeader>
                          <DialogTitle>Checkpoint Details</DialogTitle>
                        </DialogHeader>
                        <div className="space-y-4">
                          <div className="grid grid-cols-2 gap-4">
                            <div>
                              <label className="text-sm font-medium text-muted-foreground">Name</label>
                              <p className="text-foreground">{checkpoint.name}</p>
                            </div>
                            <div>
                              <label className="text-sm font-medium text-muted-foreground">Step</label>
                              <p className="text-foreground">{checkpoint.step.toLocaleString()}</p>
                            </div>
                            <div>
                              <label className="text-sm font-medium text-muted-foreground">Loss</label>
                              <p className="text-foreground">{checkpoint.loss?.toFixed(6) || 'N/A'}</p>
                            </div>
                            <div>
                              <label className="text-sm font-medium text-muted-foreground">File Size</label>
                              <p className="text-foreground">{formatFileSize(checkpoint.fileSize || 0)}</p>
                            </div>
                          </div>
                          {checkpoint.metadata && (
                            <div>
                              <label className="text-sm font-medium text-muted-foreground">Metadata</label>
                              <pre className="text-xs bg-muted p-2 rounded mt-1 overflow-auto">
                                {JSON.stringify(checkpoint.metadata, null, 2)}
                              </pre>
                            </div>
                          )}
                        </div>
                      </DialogContent>
                    </Dialog>
                    <AlertDialog>
                      <AlertDialogTrigger asChild>
                        <Button variant="ghost" size="sm" title="Delete">
                          <Trash2 className="w-4 h-4" />
                        </Button>
                      </AlertDialogTrigger>
                      <AlertDialogContent>
                        <AlertDialogHeader>
                          <AlertDialogTitle>Delete Checkpoint</AlertDialogTitle>
                          <AlertDialogDescription>
                            Are you sure you want to delete "{checkpoint.name}"? This action cannot be undone.
                          </AlertDialogDescription>
                        </AlertDialogHeader>
                        <AlertDialogFooter>
                          <AlertDialogCancel>Cancel</AlertDialogCancel>
                          <AlertDialogAction
                            onClick={() => deleteCheckpointMutation.mutate(checkpoint.id)}
                            className="bg-destructive hover:bg-destructive/90"
                          >
                            Delete
                          </AlertDialogAction>
                        </AlertDialogFooter>
                      </AlertDialogContent>
                    </AlertDialog>
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8 text-muted-foreground">
            <Save className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <p>No checkpoints found</p>
            <p className="text-sm">Checkpoints will appear here when training saves them</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
