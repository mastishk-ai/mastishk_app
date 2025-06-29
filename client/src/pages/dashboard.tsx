import { useState } from "react";
import { useLocation } from "wouter";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Sidebar } from "@/components/layout/sidebar";
import { Header } from "@/components/layout/header";
import { useWebSocket } from "@/hooks/use-websocket";
import { useModelConfig } from "@/hooks/use-model-config";
import { useTraining } from "@/hooks/use-training";
import { apiRequest } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";
import { ModelConfig } from "@shared/schema";

// Import page components
import { ModelConfigPage } from "@/components/model-config/model-config-page";
import { TrainingPage } from "@/components/training/training-page";
import { GenerationPage } from "@/components/generation/generation-page";
import { MonitoringPage } from "@/components/monitoring/monitoring-page";
import { CheckpointsPage } from "@/components/checkpoints/checkpoints-page";
import { AnalyticsPageContent } from "@/components/analytics/analytics-page";
import { TestingPage } from "@/components/testing/testing-page";
import { DocumentationPage } from "@/components/documentation/documentation-page";

export default function Dashboard() {
  const [location] = useLocation();
  const { isConnected } = useWebSocket();
  const { trainingStatus } = useTraining();
  const queryClient = useQueryClient();
  const { toast } = useToast();

  // State for model configuration
  const [modelConfig, setModelConfig] = useState<ModelConfig>({
    hidden_size: 768,
    num_hidden_layers: 12,
    num_attention_heads: 12,
    intermediate_size: 3072,
    vocab_size: 50257,
    max_position_embeddings: 2048,
    hidden_act: 'swish',
    num_key_value_heads: 12,
    use_flash_attention: false,
    use_differential_attention: false,
    differential_lambda_init: 0.5,
    use_minimax: false,
    minimax_layer_frequency: 4,
    minimax_adversarial_epsilon: 0.1,
    minimax_iterations: 3,
    lolcats_enabled: false,
    lolcats_compression_dim: 512,
    use_multi_token_prediction: false,
    n_predict_tokens: 4,
    rms_norm_eps: 1e-5,
    initializer_range: 0.02,
    use_moe: false,
    use_mod: false
  });

  // Model creation mutation
  const createModelMutation = useMutation({
    mutationFn: async (modelData: { name: string; config: any }) => {
      console.log('🚀 Starting model creation for:', modelData.name);
      const response = await fetch('/api/models', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(modelData)
      });
      
      if (!response.ok) {
        throw new Error('Failed to create model');
      }
      
      const data = await response.json();
      console.log('🎉 Model created successfully:', data);
      return data;
    },
    onSuccess: (data) => {
      console.log('📊 Current models in cache before invalidation:', queryClient.getQueryData(['/api/models']));
      
      // Invalidate and refetch models
      queryClient.invalidateQueries({ queryKey: ['/api/models'] });
      
      toast({
        title: "Model Created",
        description: `Model "${data.name}" created successfully`
      });
      
      console.log('✅ Cache invalidated, model creation complete');
    },
    onError: (error) => {
      console.error('❌ Model creation failed:', error);
      toast({
        title: "Creation Failed",
        description: "Failed to create model. Please try again.",
        variant: "destructive"
      });
    }
  });

  // Get page info based on current route
  const getPageInfo = (path: string) => {
    switch (path) {
      case "/training":
        return {
          title: "Training Pipeline",
          subtitle: "Upload data and manage training sessions"
        };
      case "/generation":
        return {
          title: "Text Generation",
          subtitle: "Generate text using your trained model"
        };
      case "/monitoring":
        return {
          title: "Training Monitor",
          subtitle: "Real-time training metrics and system status"
        };
      case "/checkpoints":
        return {
          title: "Checkpoint Manager",
          subtitle: "Manage model checkpoints and saves"
        };
      case "/analytics":
        return {
          title: "Model Analytics",
          subtitle: "Comprehensive performance analysis"
        };
      case "/testing":
        return {
          title: "Testing Suite",
          subtitle: "Test and verify component functionality"
        };
      case "/docs":
        return {
          title: "Documentation",
          subtitle: "Comprehensive guide and reference"
        };
      default:
        return {
          title: "Mastishk© Configuration",
          subtitle: "Configure your transformer architecture and parameters"
        };
    }
  };

  const { title, subtitle } = getPageInfo(location);

  const renderPage = () => {
    switch (location) {
      case "/training":
        return <TrainingPage />;
      case "/generation":
        return <GenerationPage />;
      case "/monitoring":
        return <MonitoringPage />;
      case "/checkpoints":
        return <CheckpointsPage />;
      case "/analytics":
        return <AnalyticsPageContent />;
      case "/testing":
        return <TestingPage />;
      case "/docs":
        return <DocumentationPage />;
      default:
        return <ModelConfigPage 
          config={modelConfig}
          onUpdate={(updates) => {
            console.log('Dashboard: Model config updated with:', updates);
            setModelConfig(prev => ({ ...prev, ...updates }));
          }}
          onUpdateMoe={(moeUpdates) => {
            console.log('Dashboard: MoE config updated with:', moeUpdates);
            setModelConfig(prev => ({ ...prev, ...moeUpdates }));
          }}
          onUpdateMod={(modUpdates) => {
            console.log('Dashboard: MoD config updated with:', modUpdates);
            setModelConfig(prev => ({ ...prev, ...modUpdates }));
          }}
          onCreateModel={(modelData) => {
            console.log('Dashboard: Creating model with data:', modelData);
            createModelMutation.mutate({
              name: modelData.name,
              config: modelConfig
            });
          }}
          isCreating={createModelMutation.isPending}
          validateConfig={() => []}
        />;
    }
  };

  return (
    <div className="flex h-screen overflow-hidden" style={{backgroundColor: 'hsl(var(--background))', color: 'hsl(var(--foreground))'}}>
      <Sidebar 
        modelStatus={{
          status: (trainingStatus as any)?.isTraining ? 'training' : 'ready',
          lastTrained: '2 hours ago'
        }}
      />
      
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header 
          title={title}
          subtitle={subtitle}
          trainingStatus={trainingStatus as any}
        />
        
        <main className="flex-1 overflow-auto">
          {renderPage()}
        </main>
      </div>
    </div>
  );
}

// Page wrapper components
function ModelConfigurationPage() {
  const { config, updateConfig, updateMoeConfig, updateModConfig, createModel, isCreating, validateConfig } = useModelConfig();
  
  return (
    <div className="p-6 space-y-6">
      <ModelConfigPage
        config={config}
        onUpdate={updateConfig}
        onUpdateMoe={updateMoeConfig}
        onUpdateMod={updateModConfig}
        onCreateModel={createModel}
        isCreating={isCreating}
        validateConfig={validateConfig}
      />
    </div>
  );
}

function TrainingPipelinePage() {
  return (
    <div className="p-6 space-y-6">
      <TrainingPage />
    </div>
  );
}

function TextGenerationPage() {
  return (
    <div className="p-6 space-y-6">
      <GenerationPage />
    </div>
  );
}

function TrainingMonitorPage() {
  return (
    <div className="p-6 space-y-6">
      <MonitoringPage />
    </div>
  );
}

function CheckpointManagerPage() {
  return (
    <div className="p-6 space-y-6">
      <CheckpointsPage />
    </div>
  );
}

function AnalyticsPage() {
  return (
    <div className="p-6 space-y-6">
      <AnalyticsPageContent />
    </div>
  );
}


