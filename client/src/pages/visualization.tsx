import { useState } from 'react';
import { Sidebar } from '../components/layout/sidebar';
import { Header } from '../components/layout/header';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs';
import { ThreeDVisualizer } from '../components/3d-visualizer-simple';
import { TrainingMonitor } from '../components/training/training-monitor';
import { GenerativeArchitecture3D } from '../components/enhanced-generative-3d';

export function VisualizationPage() {
  const [activeTab, setActiveTab] = useState('architecture');
  return (
    <div className="flex h-screen overflow-hidden bg-background">
      <Sidebar 
        modelStatus={{
          status: 'ready',
          lastTrained: '2 hours ago'
        }}
      />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header 
          title="Model Visualization" 
          subtitle="Explore your model's architecture and training dynamics"
        />
        <main className="flex-1 overflow-y-auto p-8 bg-gradient-to-br from-background to-muted/20">
          <div className="max-w-7xl mx-auto space-y-8">
            <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
              <TabsList className="grid w-full grid-cols-3 glass-effect p-2 rounded-2xl h-14">
                <TabsTrigger 
                  value="architecture" 
                  className="rounded-xl font-semibold transition-all duration-300 data-[state=active]:premium-button data-[state=active]:text-white"
                >
                  3D Architecture
                </TabsTrigger>
                <TabsTrigger 
                  value="training" 
                  className="rounded-xl font-semibold transition-all duration-300 data-[state=active]:premium-button data-[state=active]:text-white"
                >
                  Training Dynamics
                </TabsTrigger>
                <TabsTrigger 
                  value="generative" 
                  className="rounded-xl font-semibold transition-all duration-300 data-[state=active]:premium-button data-[state=active]:text-white"
                >
                  Generative Architecture
                </TabsTrigger>
              </TabsList>

              <TabsContent value="architecture" className="space-y-8 animate-fade-in-up">
                <div className="premium-card rounded-3xl p-8">
                  <div className="mb-6">
                    <h3 className="text-2xl font-bold bg-gradient-to-r from-primary to-primary/80 bg-clip-text text-transparent mb-2">
                      3D Model Architecture
                    </h3>
                    <p className="text-muted-foreground">Interactive exploration of your transformer model structure</p>
                  </div>
                  <div className="h-[650px] rounded-2xl overflow-hidden bg-gradient-to-br from-background to-muted/10">
                    <ThreeDVisualizer />
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="training" className="space-y-8 animate-fade-in-up">
                <div className="premium-card rounded-3xl p-8">
                  <div className="mb-6">
                    <h3 className="text-2xl font-bold bg-gradient-to-r from-primary to-primary/80 bg-clip-text text-transparent mb-2">
                      Training Dynamics
                    </h3>
                    <p className="text-muted-foreground">Real-time visualization of training progress and metrics</p>
                  </div>
                  <div className="h-[650px] rounded-2xl overflow-hidden bg-gradient-to-br from-background to-muted/10">
                    <TrainingMonitor />
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="generative" className="space-y-8 animate-fade-in-up">
                <div className="premium-card rounded-3xl p-8">
                  <div className="mb-6">
                    <h3 className="text-2xl font-bold bg-gradient-to-r from-primary to-primary/80 bg-clip-text text-transparent mb-2">
                      Generative Architecture
                    </h3>
                    <p className="text-muted-foreground">Advanced 3D visualization of generative transformer components</p>
                  </div>
                  <div className="h-[750px] rounded-2xl overflow-hidden bg-gradient-to-br from-background to-muted/10">
                    <GenerativeArchitecture3D />
                  </div>
                </div>
              </TabsContent>
            </Tabs>
          </div>
        </main>
      </div>
    </div>
  );
}