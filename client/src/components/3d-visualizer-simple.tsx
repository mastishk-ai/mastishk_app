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
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
        <div className="content-container">
          <h2 className="text-xl lg:text-2xl font-bold bg-gradient-to-r from-primary to-primary/80 bg-clip-text text-transparent mb-2">
            3D Model Visualization
          </h2>
          <p className="text-muted-foreground text-sm lg:text-base">Interactive exploration of transformer architecture</p>
        </div>
        
        <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4">
          <div className="space-y-2 min-w-0">
            <label className="text-sm font-medium block">Visualization Type</label>
            <Select value={visualType} onValueChange={setVisualType}>
              <SelectTrigger className="w-full sm:w-[180px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="architecture">Model Architecture</SelectItem>
                <SelectItem value="training">Training Landscape</SelectItem>
                <SelectItem value="attention">Attention Patterns</SelectItem>
              </SelectContent>
            </Select>
          </div>
          
          <div className="space-y-2 min-w-0">
            <label className="text-sm font-medium block">Color Scheme</label>
            <Select value={colorScheme} onValueChange={setColorScheme}>
              <SelectTrigger className="w-full sm:w-[140px]">
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
        <CardContent className="card-content-spacing">
          <div className="h-[400px] lg:h-[500px] flex items-center justify-center rounded-xl bg-gradient-to-br from-primary/5 to-primary/10 overflow-hidden">
            <div className="text-center p-6 content-container">
              <div className="w-16 h-16 lg:w-20 lg:h-20 rounded-full bg-primary/20 flex items-center justify-center mx-auto mb-4">
                {visualType === 'architecture' && <Box className="w-8 h-8 lg:w-10 lg:h-10 text-primary" />}
                {visualType === 'training' && <Mountain className="w-8 h-8 lg:w-10 lg:h-10 text-primary" />}
                {visualType === 'attention' && <Brain className="w-8 h-8 lg:w-10 lg:h-10 text-primary" />}
              </div>
              <h3 className="text-lg lg:text-xl font-semibold text-foreground mb-2 break-words">
                {visualType === 'architecture' && 'Model Architecture Visualization'}
                {visualType === 'training' && 'Training Progress Visualization'}
                {visualType === 'attention' && 'Attention Pattern Visualization'}
              </h3>
              <p className="text-muted-foreground mb-4 text-sm lg:text-base break-words">
                Interactive 3D visualization will appear here when model data is available
              </p>
              <Badge variant="secondary" className="bg-primary/10 text-primary text-xs lg:text-sm">
                {colorScheme.charAt(0).toUpperCase() + colorScheme.slice(1)} Theme
              </Badge>
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="premium-card">
          <CardHeader className="pb-3">
            <CardTitle className="text-lg text-truncate">Model Stats</CardTitle>
          </CardHeader>
          <CardContent className="card-content-spacing">
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground text-truncate">Layers:</span>
                <span className="text-sm font-medium flex-shrink-0">12</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground text-truncate">Hidden Size:</span>
                <span className="text-sm font-medium flex-shrink-0">768</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground text-truncate">Attention Heads:</span>
                <span className="text-sm font-medium flex-shrink-0">12</span>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="premium-card">
          <CardHeader className="pb-3">
            <CardTitle className="text-lg text-truncate">Performance</CardTitle>
          </CardHeader>
          <CardContent className="card-content-spacing">
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground text-truncate">Training Loss:</span>
                <span className="text-sm font-medium flex-shrink-0">2.84</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground text-truncate">Validation Loss:</span>
                <span className="text-sm font-medium flex-shrink-0">2.91</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground text-truncate">Perplexity:</span>
                <span className="text-sm font-medium flex-shrink-0">17.1</span>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="premium-card">
          <CardHeader className="pb-3">
            <CardTitle className="text-lg text-truncate">Controls</CardTitle>
          </CardHeader>
          <CardContent className="card-content-spacing">
            <div className="space-y-2">
              <Button variant="outline" size="sm" className="w-full text-sm">
                Reset View
              </Button>
              <Button variant="outline" size="sm" className="w-full text-sm">
                Export Image
              </Button>
              <Button variant="outline" size="sm" className="w-full text-sm">
                Toggle Animation
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}