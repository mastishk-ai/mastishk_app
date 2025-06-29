import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle, AlertDialogTrigger } from "@/components/ui/alert-dialog";
import { useToast } from "@/hooks/use-toast";
import { Trash2, Settings, Zap } from "lucide-react";

interface Model {
  id: number;
  name: string;
  status: string;
  createdAt: string;
  config: any;
}

export function ModelList() {
  const queryClient = useQueryClient();
  const { toast } = useToast();

  // Fetch models
  const { data: models, isLoading } = useQuery<Model[]>({
    queryKey: ['/api/models'],
    queryFn: () => fetch('/api/models').then(res => res.json())
  });

  // Delete model mutation
  const deleteModelMutation = useMutation({
    mutationFn: async (modelId: number) => {
      const response = await fetch(`/api/models/${modelId}`, {
        method: 'DELETE'
      });
      
      if (!response.ok) {
        throw new Error('Failed to delete model');
      }
      
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/models'] });
      toast({
        title: "Model Deleted",
        description: "Model deleted successfully"
      });
    },
    onError: () => {
      toast({
        title: "Delete Failed",
        description: "Failed to delete model. Please try again.",
        variant: "destructive"
      });
    }
  });

  if (isLoading) {
    return (
      <Card className="premium-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Settings className="w-5 h-5 text-primary" />
            Existing Models
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-4 text-muted-foreground">
            Loading models...
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="premium-card">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Settings className="w-5 h-5 text-primary" />
          Existing Models
        </CardTitle>
      </CardHeader>
      <CardContent>
        {models && models.length > 0 ? (
          <div className="space-y-3">
            {models.map((model) => (
              <div
                key={model.id}
                className="flex items-center justify-between p-4 border rounded-lg hover:bg-muted/50 transition-colors"
              >
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-3 mb-2">
                    <h4 className="font-medium text-foreground truncate">
                      {model.name}
                    </h4>
                    <Badge 
                      variant={model.status === 'ready' ? 'default' : model.status === 'training' ? 'secondary' : 'outline'}
                      className="flex-shrink-0"
                    >
                      {model.status === 'ready' && <Zap className="w-3 h-3 mr-1" />}
                      {model.status}
                    </Badge>
                  </div>
                  <div className="text-sm text-muted-foreground">
                    <div>ID: {model.id}</div>
                    <div>Created: {new Date(model.createdAt).toLocaleDateString()}</div>
                    <div>
                      Config: {model.config?.hidden_size || 768}d, {model.config?.num_hidden_layers || 12} layers
                    </div>
                  </div>
                </div>
                
                <div className="flex items-center gap-2 ml-4">
                  <AlertDialog>
                    <AlertDialogTrigger asChild>
                      <Button
                        variant="outline"
                        size="sm"
                        className="text-destructive hover:text-destructive hover:bg-destructive/10"
                        disabled={deleteModelMutation.isPending}
                      >
                        <Trash2 className="w-4 h-4" />
                      </Button>
                    </AlertDialogTrigger>
                    <AlertDialogContent>
                      <AlertDialogHeader>
                        <AlertDialogTitle>Delete Model</AlertDialogTitle>
                        <AlertDialogDescription>
                          Are you sure you want to delete "{model.name}"? This action cannot be undone and will also delete all associated training runs and checkpoints.
                        </AlertDialogDescription>
                      </AlertDialogHeader>
                      <AlertDialogFooter>
                        <AlertDialogCancel>Cancel</AlertDialogCancel>
                        <AlertDialogAction
                          onClick={() => deleteModelMutation.mutate(model.id)}
                          className="bg-destructive hover:bg-destructive/90 text-destructive-foreground"
                        >
                          Delete
                        </AlertDialogAction>
                      </AlertDialogFooter>
                    </AlertDialogContent>
                  </AlertDialog>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8 text-muted-foreground">
            <Settings className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <p>No models found</p>
            <p className="text-sm">Create your first model using the configuration above</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}