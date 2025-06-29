import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { AlertCircle, Download, RotateCcw, Camera, Box, Brain, Mountain, Zap, BarChart, TrendingUp } from 'lucide-react';
import Plot from 'react-plotly.js';

interface ModelConfig {
  num_hidden_layers: number;
  hidden_size: number;
  num_attention_heads: number;
  vocab_size: number;
}

interface TrainingHistory {
  metrics: {
    total_steps: number;
    loss: number[];
    learning_rate: number[];
  };
}

interface CheckpointData {
  training_step: number;
  best_loss: number;
  creation_time: string;
}

interface ModelInfo {
  name: string;
  total_parameters: number;
  best_loss: number;
  tokens_per_second: number;
}

const getColorSchemes = (isDark: boolean) => ({
  mastishk: isDark 
    ? ['#4ECDC4', '#FF6B6B', '#45B7D1', '#96CEB4', '#FECA57']
    : ['#2dd4bf', '#ef4444', '#3b82f6', '#22c55e', '#f59e0b'],
  neural: isDark
    ? ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe'] 
    : ['#5b21b6', '#581c87', '#c084fc', '#e11d48', '#2563eb'],
  energy: isDark
    ? ['#fa709a', '#fee140', '#a8edea', '#fed6e3', '#d299c2']
    : ['#ec4899', '#eab308', '#06b6d4', '#f472b6', '#a855f7']
});

export function ThreeDVisualizer() {
  const [vizType, setVizType] = useState('architecture');
  const [colorScheme, setColorScheme] = useState('mastishk');
  const [showAnnotations, setShowAnnotations] = useState(true);
  const [interactiveMode, setInteractiveMode] = useState(true);
  const [isLoading, setIsLoading] = useState(false);

  // Mock data - in real app this would come from props or API
  const mockConfig: ModelConfig = {
    num_hidden_layers: 32,
    hidden_size: 4096,
    num_attention_heads: 32,
    vocab_size: 50257
  };

  const mockTrainingHistory: TrainingHistory = {
    metrics: {
      total_steps: 1000,
      loss: Array.from({length: 100}, (_, i) => 4.0 - (i * 0.03) + Math.random() * 0.2),
      learning_rate: Array.from({length: 100}, (_, i) => 0.0001 * Math.exp(-i * 0.01))
    }
  };

  const mockCheckpoints: CheckpointData[] = [
    { training_step: 100, best_loss: 3.8, creation_time: '2025-06-28T10:00:00Z' },
    { training_step: 200, best_loss: 3.5, creation_time: '2025-06-28T11:00:00Z' },
    { training_step: 500, best_loss: 3.2, creation_time: '2025-06-28T12:00:00Z' },
    { training_step: 1000, best_loss: 2.9, creation_time: '2025-06-28T13:00:00Z' }
  ];

  const mockModels: ModelInfo[] = [
    { name: 'Current Model', total_parameters: 7_200_000_000, best_loss: 2.9, tokens_per_second: 100 },
    { name: '1B Baseline', total_parameters: 1_000_000_000, best_loss: 3.8, tokens_per_second: 150 },
    { name: '7B Large', total_parameters: 7_000_000_000, best_loss: 2.4, tokens_per_second: 80 },
    { name: '13B XL', total_parameters: 13_000_000_000, best_loss: 2.1, tokens_per_second: 50 }
  ];

  const createModelArchitecture3D = () => {
    const colors = colorSchemes[colorScheme];
    const { num_hidden_layers, hidden_size, num_attention_heads, vocab_size } = mockConfig;
    
    const data = [];
    const layerHeight = 2.0;
    const layerSpacing = 0.5;

    // Create transformer layers
    for (let i = 0; i < num_hidden_layers; i++) {
      const zPos = i * (layerHeight + layerSpacing);
      
      // Attention layer (sphere)
      const attentionSize = Math.sqrt(num_attention_heads) * 0.3;
      data.push({
        x: [0], y: [1], z: [zPos],
        mode: 'markers',
        type: 'scatter3d',
        marker: {
          size: attentionSize * 20,
          color: colors[0],
          opacity: 0.8,
          symbol: 'circle'
        },
        name: `Attention Layer ${i+1}`,
        text: `Layer ${i+1}<br>Heads: ${num_attention_heads}<br>Hidden: ${hidden_size}`,
        hovertemplate: '%{text}<extra></extra>'
      });

      // MLP layer (cube)
      const mlpSize = Math.log(hidden_size) * 0.1;
      data.push({
        x: [0], y: [-1], z: [zPos],
        mode: 'markers',
        type: 'scatter3d',
        marker: {
          size: mlpSize * 25,
          color: colors[1],
          opacity: 0.8,
          symbol: 'square'
        },
        name: `MLP Layer ${i+1}`,
        text: `MLP ${i+1}<br>Size: ${hidden_size * 4}<br>Activation: SiLU`,
        hovertemplate: '%{text}<extra></extra>'
      });

      // Layer connections
      if (i > 0) {
        data.push({
          x: [0, 0], y: [1, 1], z: [zPos - (layerHeight + layerSpacing), zPos],
          mode: 'lines',
          type: 'scatter3d',
          line: { color: colors[2], width: 3, dash: 'dash' },
          showlegend: false,
          hoverinfo: 'skip'
        });
      }
    }

    // Input embedding
    data.push({
      x: [0], y: [0], z: [-1],
      mode: 'markers',
      type: 'scatter3d',
      marker: {
        size: Math.log(vocab_size) * 3,
        color: colors[3],
        opacity: 0.9,
        symbol: 'diamond'
      },
      name: 'Input Embedding',
      text: `Embedding<br>Vocab: ${vocab_size}<br>Dim: ${hidden_size}`,
      hovertemplate: '%{text}<extra></extra>'
    });

    // Output head
    const finalZ = num_hidden_layers * (layerHeight + layerSpacing);
    data.push({
      x: [0], y: [0], z: [finalZ + 1],
      mode: 'markers',
      type: 'scatter3d',
      marker: {
        size: Math.log(vocab_size) * 3,
        color: colors[4],
        opacity: 0.9,
        symbol: 'diamond'
      },
      name: 'Output Head',
      text: `LM Head<br>Vocab: ${vocab_size}<br>Dim: ${hidden_size}`,
      hovertemplate: '%{text}<extra></extra>'
    });

    return {
      data,
      layout: {
        title: {
          text: `Mastishk Transformer Architecture 3D<br><sub>${num_hidden_layers} Layers • ${num_attention_heads} Heads • ${hidden_size} Hidden</sub>`,
          x: 0.5,
          font: { size: 18 }
        },
        scene: {
          xaxis: { title: { text: '' }, showgrid: false, zeroline: false, showticklabels: false },
          yaxis: { 
            title: { text: 'Component Type' }, 
            showgrid: true, 
            tickvals: [-1, 0, 1], 
            ticktext: ['MLP', 'Embedding/Output', 'Attention']
          },
          zaxis: { title: { text: 'Layer Depth' }, showgrid: true },
          camera: { eye: { x: 1.5, y: 1.5, z: 1.5 } },
          bgcolor: 'rgba(0,0,0,0)',
          aspectmode: 'manual' as const,
          aspectratio: { x: 1, y: 1, z: 2 }
        },
        font: { family: "Arial Black", size: 12 },
        showlegend: true,
        legend: { x: 0.02, y: 0.98 },
        height: 600
      }
    };
  };

  const createTrainingLandscape3D = () => {
    const colors = colorSchemes[colorScheme];
    const steps = Array.from({length: 20}, (_, i) => i * 50);
    const learningRates = Array.from({length: 20}, (_, i) => 0.0001 * Math.exp(i * 0.1));
    
    const z = steps.map(step => 
      learningRates.map(lr => {
        const baseLoss = 4.0 - (step * 0.002);
        const lrPenalty = Math.abs(Math.log(lr / 0.001));
        return Math.max(0.5, baseLoss + lrPenalty + Math.random() * 0.3);
      })
    );

    const actualSteps = mockTrainingHistory.metrics.loss.map((_, i) => i * 10);
    const actualLoss = mockTrainingHistory.metrics.loss;
    const actualLR = mockTrainingHistory.metrics.learning_rate;

    return {
      data: [
        {
          x: steps,
          y: learningRates,
          z: z,
          type: 'surface',
          colorscale: [
            [0, colors[0]],
            [0.5, colors[1]],
            [1, colors[2]]
          ],
          name: 'Loss Landscape',
          opacity: 0.8
        },
        {
          x: actualSteps,
          y: actualLR,
          z: actualLoss,
          mode: 'lines+markers',
          type: 'scatter3d',
          line: { color: 'red', width: 8 },
          marker: { size: 4, color: 'red' },
          name: 'Training Trajectory'
        }
      ],
      layout: {
        title: 'Training Landscape 3D',
        scene: {
          xaxis: { title: 'Training Steps' },
          yaxis: { title: 'Learning Rate', type: 'log' },
          zaxis: { title: 'Loss Value' },
          camera: { eye: { x: 1.5, y: 1.5, z: 1.5 } }
        },
        height: 600
      }
    };
  };

  const createModelComparison3D = () => {
    const colors = colorSchemes[colorScheme];
    
    return {
      data: [{
        x: mockModels.map(m => Math.log10(m.total_parameters)),
        y: mockModels.map(m => 1 / m.best_loss), // Higher is better
        z: mockModels.map(m => m.tokens_per_second),
        mode: 'markers+text',
        type: 'scatter3d',
        marker: {
          size: 15,
          color: colors[0],
          opacity: 0.8
        },
        text: mockModels.map(m => m.name),
        textposition: 'top center',
        name: 'Models'
      }],
      layout: {
        title: 'Model Performance Comparison 3D',
        scene: {
          xaxis: { title: 'Model Size (log scale)' },
          yaxis: { title: 'Performance (1/loss)' },
          zaxis: { title: 'Speed (tokens/sec)' },
          camera: { eye: { x: 1.5, y: 1.5, z: 1.5 } }
        },
        height: 600
      }
    };
  };

  const createCheckpointEvolution3D = () => {
    const colors = colorSchemes[colorScheme];
    
    return {
      data: [{
        x: mockCheckpoints.map(c => c.training_step),
        y: mockCheckpoints.map(c => c.best_loss),
        z: mockCheckpoints.map((c, i) => i),
        mode: 'lines+markers',
        type: 'scatter3d',
        line: { color: colors[0], width: 6 },
        marker: {
          size: 10,
          color: colors[1],
          symbol: 'circle'
        },
        name: 'Checkpoint Evolution'
      }],
      layout: {
        title: 'Checkpoint Evolution 3D',
        scene: {
          xaxis: { title: 'Training Steps' },
          yaxis: { title: 'Best Loss' },
          zaxis: { title: 'Checkpoint Number' },
          camera: { eye: { x: 1.5, y: 1.5, z: 1.5 } }
        },
        height: 600
      }
    };
  };

  const createAttentionPatterns3D = () => {
    const colors = colorSchemes[colorScheme];
    const seqLen = 16;
    const numHeads = 4;
    
    const data = [];
    
    for (let head = 0; head < numHeads; head++) {
      const x = [];
      const y = [];
      const z = [];
      
      for (let i = 0; i < seqLen; i++) {
        for (let j = 0; j < seqLen; j++) {
          x.push(i);
          y.push(j);
          // Simulate attention pattern - higher attention for nearby tokens
          const attention = Math.exp(-Math.abs(i - j) * 0.5) + Math.random() * 0.3;
          z.push(attention + head * 0.5);
        }
      }
      
      data.push({
        x: x,
        y: y,
        z: z,
        type: 'mesh3d',
        opacity: 0.7,
        color: colors[head % colors.length],
        name: `Head ${head + 1}`
      });
    }
    
    return {
      data,
      layout: {
        title: {
          text: 'Attention Patterns 3D',
          font: { color: isDark ? '#ffffff' : '#000000' }
        },
        scene: {
          xaxis: { 
            title: { text: 'Query Position', font: { color: isDark ? '#ffffff' : '#000000' } },
            color: isDark ? '#ffffff' : '#000000',
            gridcolor: isDark ? '#444444' : '#cccccc'
          },
          yaxis: { 
            title: { text: 'Key Position', font: { color: isDark ? '#ffffff' : '#000000' } },
            color: isDark ? '#ffffff' : '#000000',
            gridcolor: isDark ? '#444444' : '#cccccc'
          },
          zaxis: { 
            title: { text: 'Attention Weight', font: { color: isDark ? '#ffffff' : '#000000' } },
            color: isDark ? '#ffffff' : '#000000',
            gridcolor: isDark ? '#444444' : '#cccccc'
          },
          camera: { eye: { x: 1.5, y: 1.5, z: 1.5 } },
          bgcolor: isDark ? '#1a1a1a' : '#ffffff'
        },
        height: 600,
        paper_bgcolor: isDark ? '#1a1a1a' : '#ffffff',
        plot_bgcolor: isDark ? '#1a1a1a' : '#ffffff',
        font: { color: isDark ? '#ffffff' : '#000000' }
      }
    };
  };

  const createFeatureActivations3D = () => {
    const colors = getColorSchemes(isDark)[colorScheme];
    const numLayers = 12;
    const activations = [];
    
    for (let layer = 0; layer < numLayers; layer++) {
      const x = Math.sin(layer * 0.5) * (layer + 1);
      const y = Math.cos(layer * 0.5) * (layer + 1);
      const z = layer * 2;
      activations.push([x, y, z]);
    }
    
    return {
      data: [
        {
          x: activations.map(a => a[0]),
          y: activations.map(a => a[1]),
          z: activations.map(a => a[2]),
          mode: 'markers+lines',
          type: 'scatter3d',
          marker: {
            size: 8,
            color: colors[0],
            opacity: 0.8
          },
          line: {
            color: colors[1],
            width: 4
          },
          name: 'Feature Flow'
        }
      ],
      layout: {
        ...getPlotlyLayout(isDark, 'Feature Activations 3D'),
        scene: {
          ...getPlotlyLayout(isDark).scene,
          xaxis: { 
            ...getPlotlyLayout(isDark).scene?.xaxis,
            title: { text: 'Feature Space X', font: { color: isDark ? '#f8fafc' : '#0f172a' } }
          },
          yaxis: { 
            ...getPlotlyLayout(isDark).scene?.yaxis,
            title: { text: 'Feature Space Y', font: { color: isDark ? '#f8fafc' : '#0f172a' } }
          },
          zaxis: { 
            ...getPlotlyLayout(isDark).scene?.zaxis,
            title: { text: 'Layer Depth', font: { color: isDark ? '#f8fafc' : '#0f172a' } }
          },
          camera: { eye: { x: 1.5, y: 1.5, z: 1.5 } }
        },
        height: 600
      }
    };
  };

  const getVisualizationData = () => {
    switch (vizType) {
      case 'architecture': return createModelArchitecture3D();
      case 'attention': return createAttentionPatterns3D();
      case 'landscape': return createTrainingLandscape3D();
      case 'activations': return createFeatureActivations3D();
      case 'comparison': return createModelComparison3D();
      case 'evolution': return createCheckpointEvolution3D();
      default: return createModelArchitecture3D();
    }
  };

  const getVizIcon = (type: string) => {
    const icons: Record<string, any> = {
      architecture: Box,
      attention: Brain,
      landscape: Mountain,
      activations: Zap,
      comparison: BarChart,
      evolution: TrendingUp
    };
    const IconComponent = icons[type] || Box;
    return <IconComponent className="w-4 h-4" />;
  };

  const getAnnotationText = () => {
    const annotations: Record<string, string> = {
      architecture: `Architecture Overview:
• Layers: ${mockConfig.num_hidden_layers} transformer layers
• Hidden Size: ${mockConfig.hidden_size} dimensions  
• Attention Heads: ${mockConfig.num_attention_heads} heads per layer
• Parameters: ~${(7.2).toFixed(1)}B

🔵 Blue spheres: Attention layers
🔴 Red squares: MLP/Feed-forward layers
🔶 Orange diamonds: Embedding & output layers`,
      
      attention: `Attention Pattern Analysis:
• Each surface represents one attention head
• X-axis: Query positions in sequence
• Y-axis: Key positions in sequence
• Z-axis: Attention weight strength
• Brighter colors = stronger attention`,
      
      landscape: `Training Landscape Analysis:
• Surface: Loss values across step/learning rate space
• Red line: Actual training trajectory
• Valleys: Lower loss regions (better performance)
• Latest session: ${mockTrainingHistory.metrics.total_steps} steps`,
      
      activations: `Feature Activation Flow:
• Each point represents a layer's activation
• Lines show information flow between layers
• Position indicates activation pattern
• Colors distinguish different layers`,
      
      comparison: `Model Performance Comparison:
• X-axis: Model size (logarithmic scale)
• Y-axis: Performance (inverse of loss)
• Z-axis: Training speed (tokens/second)
• Ideal models: High on all dimensions`,
      
      evolution: `Checkpoint Evolution Analysis:
• Total checkpoints: ${mockCheckpoints.length}
• X-axis: Training steps
• Y-axis: Loss values
• Z-axis: Checkpoint timeline
• Gold star: Best performing checkpoint`
    };
    
    return annotations[vizType] || '';
  };

  const vizData = getVisualizationData();

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Box className="w-5 h-5" />
            3D Model Visualizations
          </CardTitle>
          <p className="text-sm text-muted-foreground">
            Interactive 3D insights into your Mastishk Transformer
          </p>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
            {/* Controls */}
            <div className="space-y-4">
              <div className="space-y-2">
                <Label>Visualization Type</Label>
                <Select value={vizType} onValueChange={setVizType}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="architecture">
                      <div className="flex items-center gap-2">
                        {getVizIcon('architecture')} Model Architecture
                      </div>
                    </SelectItem>
                    <SelectItem value="attention">
                      <div className="flex items-center gap-2">
                        {getVizIcon('attention')} Attention Patterns
                      </div>
                    </SelectItem>
                    <SelectItem value="landscape">
                      <div className="flex items-center gap-2">
                        {getVizIcon('landscape')} Training Landscape
                      </div>
                    </SelectItem>
                    <SelectItem value="activations">
                      <div className="flex items-center gap-2">
                        {getVizIcon('activations')} Feature Activations
                      </div>
                    </SelectItem>
                    <SelectItem value="comparison">
                      <div className="flex items-center gap-2">
                        {getVizIcon('comparison')} Model Comparison
                      </div>
                    </SelectItem>
                    <SelectItem value="evolution">
                      <div className="flex items-center gap-2">
                        {getVizIcon('evolution')} Checkpoint Evolution
                      </div>
                    </SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label>Color Scheme</Label>
                <Select value={colorScheme} onValueChange={setColorScheme}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="mastishk">Mastishk</SelectItem>
                    <SelectItem value="neural">Neural</SelectItem>
                    <SelectItem value="energy">Energy</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-3">
                <div className="flex items-center space-x-2">
                  <Checkbox 
                    id="annotations" 
                    checked={showAnnotations}
                    onCheckedChange={(checked) => setShowAnnotations(checked === true)}
                  />
                  <Label htmlFor="annotations">Show Annotations</Label>
                </div>
                
                <div className="flex items-center space-x-2">
                  <Checkbox 
                    id="interactive" 
                    checked={interactiveMode}
                    onCheckedChange={(checked) => setInteractiveMode(checked === true)}
                  />
                  <Label htmlFor="interactive">Interactive Mode</Label>
                </div>
              </div>

              <div className="space-y-2 pt-4 border-t">
                <Label>3D Controls & Export</Label>
                <div className="space-y-2">
                  <Button variant="outline" size="sm" className="w-full">
                    <Camera className="w-4 h-4 mr-2" />
                    Save View
                  </Button>
                  <Button variant="outline" size="sm" className="w-full">
                    <RotateCcw className="w-4 h-4 mr-2" />
                    Reset Camera
                  </Button>
                  <Button variant="outline" size="sm" className="w-full">
                    <Download className="w-4 h-4 mr-2" />
                    Export PNG
                  </Button>
                </div>
              </div>
            </div>

            {/* Visualization */}
            <div className="lg:col-span-3 space-y-4">
              {showAnnotations && (
                <Card>
                  <CardContent className="pt-4">
                    <div className="flex items-start gap-2">
                      <AlertCircle className="w-4 h-4 mt-0.5 text-blue-500" />
                      <pre className="text-sm whitespace-pre-wrap font-mono">
                        {getAnnotationText()}
                      </pre>
                    </div>
                  </CardContent>
                </Card>
              )}

              <div className="bg-card rounded-lg p-4">
                <Plot
                  data={vizData.data as any}
                  layout={vizData.layout as any}
                  config={{
                    responsive: true,
                    displayModeBar: interactiveMode,
                    displaylogo: false,
                    modeBarButtonsToRemove: ['pan2d', 'select2d', 'lasso2d'] as any
                  }}
                  style={{ width: '100%', height: '600px' }}
                />
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}