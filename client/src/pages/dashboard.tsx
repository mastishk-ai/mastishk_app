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
import { formatDistanceToNow } from "date-fns";

// Import page components
import { ModelConfigPage } from "@/components/model-config/model-config-page";
import { TrainingPage } from "@/components/training/training-page";
import { GenerationPage } from "@/components/generation/generation-page";
import { MonitoringPage } from "@/components/monitoring/monitoring-page";
import { CheckpointsPage } from "@/components/checkpoints/checkpoints-page";
import { AnalyticsPageContent } from "@/components/analytics/analytics-page";
import { TestingPage } from "@/components/testing/testing-page";
import { DocumentationPage } from "@/components/documentation/documentation-page";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Mail, MapPin, Phone, Globe, Shield, FileText, Heart, Users } from "lucide-react";

export default function Dashboard() {
  const [location] = useLocation();
  const { isConnected } = useWebSocket();
  const { trainingStatus } = useTraining();
  const queryClient = useQueryClient();
  const { toast } = useToast();

  // Fetch models to get real model status
  const { data: models } = useQuery({
    queryKey: ['/api/models'],
    queryFn: () => fetch('/api/models').then(res => res.json())
  });

  // Fetch training runs to get last training time
  const { data: trainingRuns } = useQuery({
    queryKey: ['/api/training-runs'],
    queryFn: () => fetch('/api/training-runs').then(res => res.json())
  });

  // Calculate real model status
  const getModelStatus = () => {
    if ((trainingStatus as any)?.isTraining) {
      return { status: 'training', lastTrained: 'Currently training...' };
    }

    if (!models || models.length === 0) {
      return { status: 'idle', lastTrained: 'No models created yet' };
    }

    if (!trainingRuns || trainingRuns.length === 0) {
      return { status: 'ready', lastTrained: 'Never trained' };
    }

    // Find the most recent completed training run
    const completedRuns = trainingRuns.filter((run: any) => run.completedAt);
    if (completedRuns.length === 0) {
      return { status: 'ready', lastTrained: 'Never completed training' };
    }

    const mostRecentRun = completedRuns.reduce((latest: any, current: any) => {
      return new Date(current.completedAt) > new Date(latest.completedAt) ? current : latest;
    });

    const lastTrainedTime = formatDistanceToNow(new Date(mostRecentRun.completedAt), { addSuffix: true });
    return { status: 'ready', lastTrained: lastTrainedTime };
  };

  // Legal Page Components (defined inside Dashboard to access proper scope)
  const AboutPageContent = () => (
    <div className="space-y-8">
      <Card className="premium-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Heart className="w-5 h-5 text-primary" />
            About Mastishk Transformer Studio
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-4">
            <p className="text-muted-foreground leading-relaxed">
              Mastishk Transformer Studio is an advanced platform for transformer experimentation, 
              research, and development. Built with cutting-edge technology, it provides researchers 
              and developers with powerful tools for creating, training, and analyzing sophisticated 
              transformer models.
            </p>
            
            <div className="grid md:grid-cols-2 gap-6">
              <div className="space-y-3">
                <h3 className="font-semibold text-lg">Key Features</h3>
                <ul className="space-y-2 text-sm text-muted-foreground">
                  <li>• Advanced transformer architecture configuration</li>
                  <li>• Real-time training monitoring and analytics</li>
                  <li>• Interactive 3D model visualization</li>
                  <li>• Comprehensive checkpoint management</li>
                  <li>• MoE, MoD, and Flash Attention support</li>
                </ul>
              </div>
              
              <div className="space-y-3">
                <h3 className="font-semibold text-lg">Technology Stack</h3>
                <ul className="space-y-2 text-sm text-muted-foreground">
                  <li>• React with TypeScript frontend</li>
                  <li>• Node.js/Express backend</li>
                  <li>• Python ML integration</li>
                  <li>• PostgreSQL database</li>
                  <li>• Modern UI with shadcn/ui</li>
                </ul>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );

  const ContactPageContent = () => (
    <div className="space-y-8">
      <Card className="premium-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Mail className="w-5 h-5 text-primary" />
            Contact Information
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <div className="flex items-center gap-3">
                <Users className="w-5 h-5 text-primary" />
                <div>
                  <h3 className="font-semibold">Developer</h3>
                  <p className="text-sm text-muted-foreground">Aman Sharma</p>
                </div>
              </div>
              
              <div className="flex items-center gap-3">
                <Globe className="w-5 h-5 text-primary" />
                <div>
                  <h3 className="font-semibold">Platform</h3>
                  <p className="text-sm text-muted-foreground">Mastishk Transformer Studio</p>
                </div>
              </div>
            </div>
            
            <div className="space-y-4">
              <div className="flex items-center gap-3">
                <FileText className="w-5 h-5 text-primary" />
                <div>
                  <h3 className="font-semibold">Support</h3>
                  <p className="text-sm text-muted-foreground">Documentation and guides available</p>
                </div>
              </div>
              
              <div className="flex items-center gap-3">
                <Shield className="w-5 h-5 text-primary" />
                <div>
                  <h3 className="font-semibold">Security</h3>
                  <p className="text-sm text-muted-foreground">Privacy-focused development</p>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );

  const PrivacyPageContent = () => (
    <div className="space-y-8">
      <Card className="premium-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Shield className="w-5 h-5 text-primary" />
            Privacy Policy
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-4">
            <div>
              <h3 className="font-semibold mb-2">Data Collection</h3>
              <p className="text-sm text-muted-foreground">
                Mastishk Transformer Studio operates as a local development platform. 
                Your model configurations, training data, and generated content remain 
                on your local system and are not transmitted to external servers.
              </p>
            </div>
            
            <div>
              <h3 className="font-semibold mb-2">Local Storage</h3>
              <p className="text-sm text-muted-foreground">
                All data including model checkpoints, training progress, and configurations 
                are stored locally on your machine. No personal or model data is shared 
                with third parties.
              </p>
            </div>
            
            <div>
              <h3 className="font-semibold mb-2">Security</h3>
              <p className="text-sm text-muted-foreground">
                The platform is designed with privacy-first principles. Your research 
                and development work remains confidential and under your complete control.
              </p>
            </div>
            
            <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
              Privacy-First Design
            </Badge>
          </div>
        </CardContent>
      </Card>
    </div>
  );

  const TermsPageContent = () => (
    <div className="space-y-8">
      <Card className="premium-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <FileText className="w-5 h-5 text-primary" />
            Terms of Service
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-4">
            <div>
              <h3 className="font-semibold mb-2">Usage License</h3>
              <p className="text-sm text-muted-foreground">
                Mastishk Transformer Studio is provided for research and development purposes. 
                Users are granted permission to use the platform for transformer model 
                experimentation and analysis.
              </p>
            </div>
            
            <div>
              <h3 className="font-semibold mb-2">Intellectual Property</h3>
              <p className="text-sm text-muted-foreground">
                Users retain full ownership of their model architectures, training data, 
                and generated content. The platform source code and design remain 
                the property of the developer.
              </p>
            </div>
            
            <div>
              <h3 className="font-semibold mb-2">Responsibility</h3>
              <p className="text-sm text-muted-foreground">
                Users are responsible for their use of the platform and any models 
                created. The platform is provided "as-is" for educational and research purposes.
              </p>
            </div>
            
            <div>
              <h3 className="font-semibold mb-2">Updates</h3>
              <p className="text-sm text-muted-foreground">
                These terms may be updated to reflect platform improvements and changes. 
                Continued use constitutes acceptance of updated terms.
              </p>
            </div>
            
            <div className="pt-4 border-t">
              <p className="text-xs text-muted-foreground">
                Last updated: June 29, 2025
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );

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
    learning_rate: 5e-4,
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

  // Model deletion mutation
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
    onError: (error) => {
      toast({
        title: "Delete Failed", 
        description: "Failed to delete model. Please try again.",
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
      case "/about":
        return {
          title: "About Mastishk",
          subtitle: "Advanced transformer experimentation platform"
        };
      case "/contact":
        return {
          title: "Contact Us",
          subtitle: "Get in touch with our team"
        };
      case "/privacy":
        return {
          title: "Privacy Policy",
          subtitle: "How we protect your data and privacy"
        };
      case "/terms":
        return {
          title: "Terms of Service",
          subtitle: "Terms and conditions for using Mastishk"
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
      case "/about":
        return <AboutPageContent />;
      case "/contact":
        return <ContactPageContent />;
      case "/privacy":
        return <PrivacyPageContent />;
      case "/terms":
        return <TermsPageContent />;
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
        modelStatus={getModelStatus()}
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


