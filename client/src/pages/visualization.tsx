import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { ThreeDVisualizer } from '@/components/3d-visualizer';
import { AdvancedThreeDViewer } from '@/components/advanced-3d-viewer';
import { Box, Zap, Sparkles } from 'lucide-react';

export default function VisualizationPage() {
  return (
    <div className="container mx-auto p-6">
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold tracking-tight">3D Visualizations</h1>
            <p className="text-muted-foreground">
              Interactive 3D insights and advanced neural network visualization
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="secondary" className="gap-1">
              <Sparkles className="w-3 h-3" />
              WebGL Accelerated
            </Badge>
            <Badge variant="outline">
              Real-time Rendering
            </Badge>
          </div>
        </div>

        <Tabs defaultValue="plotly" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="plotly" className="flex items-center gap-2">
              <Box className="w-4 h-4" />
              3D Model Visualizations
            </TabsTrigger>
            <TabsTrigger value="threejs" className="flex items-center gap-2">
              <Zap className="w-4 h-4" />
              Advanced 3D Neural Network
            </TabsTrigger>
          </TabsList>

          <TabsContent value="plotly" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Box className="w-5 h-5" />
                  Interactive Plotly 3D Visualizations
                </CardTitle>
                <p className="text-sm text-muted-foreground">
                  Six different 3D visualization types: Model Architecture, Attention Patterns, 
                  Training Landscape, Feature Activations, Model Comparison, and Checkpoint Evolution
                </p>
              </CardHeader>
              <CardContent>
                <ThreeDVisualizer />
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="threejs" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Zap className="w-5 h-5" />
                  Advanced Three.js Neural Network
                </CardTitle>
                <p className="text-sm text-muted-foreground">
                  Real-time 3D transformer architecture with interactive controls, 
                  particle systems, and dynamic animations
                </p>
              </CardHeader>
              <CardContent>
                <AdvancedThreeDViewer />
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}