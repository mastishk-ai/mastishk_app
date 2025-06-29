import { useLocation } from "wouter";
import { useQuery } from "@tanstack/react-query";
import { Sidebar } from "@/components/layout/sidebar";
import { Header } from "@/components/layout/header";
import { useWebSocket } from "@/hooks/use-websocket";
import { useModelConfig } from "@/hooks/use-model-config";
import { useTraining } from "@/hooks/use-training";

// Import page components
import { ModelConfigPage } from "@/components/model-config/model-config-page";
import { TrainingPage } from "@/components/training/training-page";
import { GenerationPage } from "@/components/generation/generation-page";
import { MonitoringPage } from "@/components/monitoring/monitoring-page";
import { CheckpointsPage } from "@/components/checkpoints/checkpoints-page";
import { AnalyticsPageContent } from "@/components/analytics/analytics-page";

export default function Dashboard() {
  const [location] = useLocation();
  const { isConnected } = useWebSocket();
  const { trainingStatus } = useTraining();

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
      default:
        return {
          title: "Model Configuration",
          subtitle: "Configure your Mastishk Transformer parameters"
        };
    }
  };

  const { title, subtitle } = getPageInfo(location);

  const renderPage = () => {
    switch (location) {
      case "/training":
        return <TrainingPipelinePage />;
      case "/generation":
        return <TextGenerationPage />;
      case "/monitoring":
        return <TrainingMonitorPage />;
      case "/checkpoints":
        return <CheckpointManagerPage />;
      case "/analytics":
        return <AnalyticsPage />;
      default:
        return <ModelConfigurationPage />;
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


