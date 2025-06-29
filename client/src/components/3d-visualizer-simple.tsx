import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Box, Brain, Mountain } from 'lucide-react';

export function ThreeDVisualizer() {
  const [visualType, setVisualType] = useState('architecture');
  const [colorScheme, setColorScheme] = useState('mastishk');

  return (
    <div className="space-y-6 p-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold bg-gradient-to-r from-primary to-primary/80 bg-clip-text text-transparent">
            3D Model Visualization
          </h2>
          <p className="text-muted-foreground">Interactive exploration of transformer architecture</p>
        </div>
        
        <div className="flex items-center gap-4">
          <div className="space-y-2">
            <label className="text-sm font-medium">Visualization Type</label>
            <Select value={visualType} onValueChange={setVisualType}>
              <SelectTrigger className="w-[180px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="architecture">Model Architecture</SelectItem>
                <SelectItem value="training">Training Landscape</SelectItem>
                <SelectItem value="attention">Attention Patterns</SelectItem>
              </SelectContent>
            </Select>
          </div>
          
          <div className="space-y-2">
            <label className="text-sm font-medium">Color Scheme</label>
            <Select value={colorScheme} onValueChange={setColorScheme}>
              <SelectTrigger className="w-[140px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="mastishk">Mastishk</SelectItem>
                <SelectItem value="neural">Neural</SelectItem>
                <SelectItem value="energy">Energy</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
      </div>

      <Card className="premium-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            {visualType === 'architecture' && <Box className="w-5 h-5 text-primary" />}
            {visualType === 'training' && <Mountain className="w-5 h-5 text-primary" />}
            {visualType === 'attention' && <Brain className="w-5 h-5 text-primary" />}
            {visualType === 'architecture' && '3D Model Architecture'}
            {visualType === 'training' && '3D Training Landscape'}
            {visualType === 'attention' && '3D Attention Patterns'}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-[500px] flex items-center justify-center rounded-xl bg-gradient-to-br from-primary/5 to-primary/10">
            <div className="text-center">
              <div className="w-20 h-20 rounded-full bg-primary/20 flex items-center justify-center mx-auto mb-4">
                {visualType === 'architecture' && <Box className="w-10 h-10 text-primary" />}
                {visualType === 'training' && <Mountain className="w-10 h-10 text-primary" />}
                {visualType === 'attention' && <Brain className="w-10 h-10 text-primary" />}
              </div>
              <h3 className="text-xl font-semibold text-foreground mb-2">
                {visualType === 'architecture' && 'Model Architecture Visualization'}
                {visualType === 'training' && 'Training Progress Visualization'}
                {visualType === 'attention' && 'Attention Pattern Visualization'}
              </h3>
              <p className="text-muted-foreground mb-4">
                Interactive 3D visualization will appear here when model data is available
              </p>
              <Badge variant="secondary" className="bg-primary/10 text-primary">
                {colorScheme.charAt(0).toUpperCase() + colorScheme.slice(1)} Theme
              </Badge>
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="premium-card">
          <CardHeader className="pb-3">
            <CardTitle className="text-lg">Model Stats</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-sm text-muted-foreground">Layers:</span>
                <span className="text-sm font-medium">12</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-muted-foreground">Hidden Size:</span>
                <span className="text-sm font-medium">768</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-muted-foreground">Attention Heads:</span>
                <span className="text-sm font-medium">12</span>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="premium-card">
          <CardHeader className="pb-3">
            <CardTitle className="text-lg">Performance</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-sm text-muted-foreground">Training Loss:</span>
                <span className="text-sm font-medium">2.84</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-muted-foreground">Validation Loss:</span>
                <span className="text-sm font-medium">2.91</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-muted-foreground">Perplexity:</span>
                <span className="text-sm font-medium">17.1</span>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="premium-card">
          <CardHeader className="pb-3">
            <CardTitle className="text-lg">Controls</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <Button variant="outline" size="sm" className="w-full">
                Reset View
              </Button>
              <Button variant="outline" size="sm" className="w-full">
                Export Image
              </Button>
              <Button variant="outline" size="sm" className="w-full">
                Toggle Animation
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}