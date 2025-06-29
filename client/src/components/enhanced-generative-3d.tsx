import { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Badge } from '@/components/ui/badge';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Play, Pause, RotateCcw, Eye, Settings } from 'lucide-react';
import { getPlotlyLayout, getModelArchitectureColors } from '@/lib/theme';

interface GenerativeArchitecture3DProps {
  modelConfig?: {
    hidden_size: number;
    num_hidden_layers: number;
    num_attention_heads: number;
    vocab_size: number;
    intermediate_size: number;
    use_moe?: boolean;
    use_mod?: boolean;
  };
}

export function GenerativeArchitecture3D({ modelConfig }: GenerativeArchitecture3DProps) {
  const [isDark, setIsDark] = useState(false);
  const [isAnimating, setIsAnimating] = useState(false);
  const [currentLayer, setCurrentLayer] = useState(0);
  const [showConnections, setShowConnections] = useState(true);
  const [showLabels, setShowLabels] = useState(true);
  const [viewMode, setViewMode] = useState<'overview' | 'layer' | 'flow'>('overview');
  const [animationSpeed, setAnimationSpeed] = useState(1000);

  useEffect(() => {
    const checkTheme = () => {
      const theme = localStorage.getItem('theme') || 'system';
      const isSystemDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      const shouldBeDark = theme === 'dark' || (theme === 'system' && isSystemDark);
      setIsDark(shouldBeDark);
    };

    checkTheme();
    
    // Listen for theme changes
    const handleThemeChange = () => checkTheme();
    window.addEventListener('themeChange', handleThemeChange);
    
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    mediaQuery.addEventListener('change', checkTheme);
    
    return () => {
      window.removeEventListener('themeChange', handleThemeChange);
      mediaQuery.removeEventListener('change', checkTheme);
    };
  }, []);

  // Model configuration with defaults
  const config = {
    hidden_size: 4096,
    num_hidden_layers: 32,
    num_attention_heads: 32,
    vocab_size: 32000,
    intermediate_size: 11008,
    use_moe: false,
    use_mod: false,
    ...modelConfig
  };

  const generateGenerativeArchitecture = () => {
    const traces = [];
    const animations = [];
    const colors = {
      embedding: '#FFD700',
      attention: '#4ECDC4', 
      mlp: '#FF6B6B',
      norm: '#96CEB4',
      output: '#FF69B4',
      connection: '#888888',
      flow: '#00FF00'
    };

    // Token flow path for generative process
    const tokenPath = [];
    
    for (let layer = 0; layer <= config.num_hidden_layers; layer++) {
      const z = layer * 4;
      
      if (layer === 0) {
        // Input Token Embedding
        traces.push({
          x: [0],
          y: [0],
          z: [z],
          mode: 'markers+text' as any,
          type: 'scatter3d' as any,
          marker: {
            size: 25,
            color: colors.embedding,
            symbol: 'circle',
            opacity: 0.9,
            line: { width: 3, color: '#FFA500' }
          },
          text: showLabels ? ['Token Embedding'] : [''],
          textposition: 'top center',
          name: 'Input Embedding',
          showlegend: false,
          hovertemplate: `Input Token Embedding<br>Vocab Size: ${config.vocab_size}<br>Hidden Size: ${config.hidden_size}<extra></extra>`
        });
        
        tokenPath.push([0, 0, z]);
      } else if (layer === config.num_hidden_layers) {
        // Output Generation Layer
        traces.push({
          x: [0],
          y: [0],
          z: [z],
          mode: 'markers+text' as any,
          type: 'scatter3d' as any,
          marker: {
            size: 25,
            color: colors.output,
            symbol: 'circle',
            opacity: 0.9,
            line: { width: 3, color: '#FF1493' }
          },
          text: showLabels ? ['Next Token'] : [''],
          textposition: 'top center',
          name: 'Output Generation',
          showlegend: false,
          hovertemplate: `Output Generation<br>Vocab Size: ${config.vocab_size}<br>Generates next token probability distribution<extra></extra>`
        });
        
        tokenPath.push([0, 0, z]);
      } else {
        // Transformer Layer Components
        const layerZ = z;
        const isCurrentLayer = currentLayer === layer - 1;
        const layerOpacity = viewMode === 'layer' ? (isCurrentLayer ? 1.0 : 0.3) : 0.8;
        
        // Multi-Head Attention Ring
        const attentionHeads = Math.min(config.num_attention_heads, 12); // Limit for clarity
        for (let head = 0; head < attentionHeads; head++) {
          const angle = (head / attentionHeads) * 2 * Math.PI;
          const radius = 2;
          const x = Math.cos(angle) * radius;
          const y = Math.sin(angle) * radius;
          
          traces.push({
            x: [x],
            y: [y],
            z: [layerZ],
            mode: 'markers' as any,
            type: 'scatter3d' as any,
            marker: {
              size: 12,
              color: colors.attention,
              opacity: layerOpacity,
              symbol: 'circle',
              line: { width: 1, color: '#333' }
            },
            name: `L${layer-1}H${head}`,
            showlegend: false,
            hovertemplate: `Layer ${layer-1}<br>Attention Head ${head}<br>Head Dim: ${Math.floor(config.hidden_size / config.num_attention_heads)}<extra></extra>`
          });
          
          // Attention connections to center
          if (showConnections) {
            traces.push({
              x: [x, 0],
              y: [y, 0],
              z: [layerZ, layerZ],
              mode: 'lines' as any,
              type: 'scatter3d' as any,
              line: {
                color: colors.attention,
                width: 2,
                opacity: layerOpacity * 0.5
              },
              showlegend: false,
              hoverinfo: 'skip'
            });
          }
        }
        
        // Central attention aggregation
        traces.push({
          x: [0],
          y: [0],
          z: [layerZ],
          mode: 'markers+text' as any,
          type: 'scatter3d' as any,
          marker: {
            size: 18,
            color: colors.attention,
            opacity: layerOpacity,
            symbol: 'diamond',
            line: { width: 2, color: '#333' }
          },
          text: showLabels ? [`L${layer-1} Attn`] : [''],
          textposition: 'middle center',
          name: `Layer ${layer-1} Attention`,
          showlegend: false,
          hovertemplate: `Layer ${layer-1} Multi-Head Attention<br>Heads: ${config.num_attention_heads}<br>Hidden Size: ${config.hidden_size}<extra></extra>`
        });
        
        // Feed-Forward Network
        const mlpComponents = [
          { name: 'Linear 1', pos: [-3, 0, layerZ + 1], size: 16 },
          { name: 'Activation', pos: [0, -3, layerZ + 1], size: 12 },
          { name: 'Linear 2', pos: [3, 0, layerZ + 1], size: 16 }
        ];
        
        mlpComponents.forEach((comp, idx) => {
          traces.push({
            x: [comp.pos[0]],
            y: [comp.pos[1]],
            z: [comp.pos[2]],
            mode: 'markers+text' as any,
            type: 'scatter3d' as any,
            marker: {
              size: comp.size,
              color: colors.mlp,
              opacity: layerOpacity,
              symbol: 'square',
              line: { width: 1, color: '#333' }
            },
            text: showLabels ? [comp.name] : [''],
            textposition: 'bottom center',
            name: `L${layer-1} ${comp.name}`,
            showlegend: false,
            hovertemplate: `Layer ${layer-1} ${comp.name}<br>Intermediate Size: ${config.intermediate_size}<extra></extra>`
          });
          
          // MLP connections
          if (showConnections && idx < mlpComponents.length - 1) {
            const nextComp = mlpComponents[idx + 1];
            traces.push({
              x: [comp.pos[0], nextComp.pos[0]],
              y: [comp.pos[1], nextComp.pos[1]],
              z: [comp.pos[2], nextComp.pos[2]],
              mode: 'lines' as any,
              type: 'scatter3d' as any,
              line: {
                color: colors.mlp,
                width: 3,
                opacity: layerOpacity * 0.6
              },
              showlegend: false,
              hoverinfo: 'skip'
            });
          }
        });
        
        // Layer Normalization
        traces.push({
          x: [0],
          y: [3],
          z: [layerZ + 0.5],
          mode: 'markers+text' as any,
          type: 'scatter3d' as any,
          marker: {
            size: 10,
            color: colors.norm,
            opacity: layerOpacity,
            symbol: 'diamond'
          },
          text: showLabels ? ['LayerNorm'] : [''],
          textposition: 'top center',
          name: `L${layer-1} Norm`,
          showlegend: false,
          hovertemplate: `Layer ${layer-1} Normalization<br>RMS Norm with learned scale<extra></extra>`
        });
        
        // Residual Connection
        if (layer > 1 && showConnections) {
          traces.push({
            x: [0, 0],
            y: [0, 0],
            z: [layerZ - 4, layerZ + 2],
            mode: 'lines' as any,
            type: 'scatter3d' as any,
            line: {
              color: colors.connection,
              width: 6,
              opacity: layerOpacity * 0.4,
              dash: 'dash'
            },
            name: `Residual ${layer-1}`,
            showlegend: false,
            hoverinfo: 'skip'
          });
        }
        
        tokenPath.push([0, 0, layerZ + 2]);
      }
    }
    
    // Token flow visualization
    if (viewMode === 'flow' && tokenPath.length > 1) {
      traces.push({
        x: tokenPath.map(p => p[0]),
        y: tokenPath.map(p => p[1]),
        z: tokenPath.map(p => p[2]),
        mode: 'lines+markers' as any,
        type: 'scatter3d' as any,
        line: {
          color: colors.flow,
          width: 8,
          opacity: 0.8
        },
        marker: {
          size: 6,
          color: colors.flow,
          opacity: 0.8
        },
        name: 'Token Flow',
        showlegend: false,
        hovertemplate: 'Generative Token Flow<br>Input → Layers → Next Token<extra></extra>'
      });
    }

    return traces;
  };

  const data = generateGenerativeArchitecture();
  
  const layout = {
    ...getPlotlyLayout(isDark, `Generative Transformer Architecture - ${config.num_hidden_layers} Layers • ${config.num_attention_heads} Heads • ${config.hidden_size} Hidden • ${config.vocab_size} Vocab`),
    scene: {
      ...getPlotlyLayout(isDark).scene,
      xaxis: { 
        ...getPlotlyLayout(isDark).scene?.xaxis,
        title: { text: 'Width', font: { color: isDark ? '#f8fafc' : '#0f172a' } }, 
        range: [-4, 4]
      },
      yaxis: { 
        ...getPlotlyLayout(isDark).scene?.yaxis,
        title: { text: 'Depth', font: { color: isDark ? '#f8fafc' : '#0f172a' } }, 
        range: [-4, 4]
      },
      zaxis: { 
        ...getPlotlyLayout(isDark).scene?.zaxis,
        title: { text: 'Layer Progression', font: { color: isDark ? '#f8fafc' : '#0f172a' } }, 
        range: [0, config.num_hidden_layers * 4]
      },
      camera: { 
        eye: { x: 1.8, y: 1.8, z: 1.2 },
        center: { x: 0, y: 0, z: config.num_hidden_layers * 2 }
      },
      aspectmode: 'manual' as const,
      aspectratio: { x: 1, y: 1, z: 2 }
    },
    height: 800,
    margin: { l: 0, r: 0, t: 80, b: 0 }
  };

  const config_plotly = {
    displayModeBar: true,
    modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'] as any,
    displaylogo: false,
    responsive: true
  };

  useEffect(() => {
    if (isAnimating) {
      const interval = setInterval(() => {
        setCurrentLayer(prev => (prev + 1) % config.num_hidden_layers);
      }, animationSpeed);
      return () => clearInterval(interval);
    }
  }, [isAnimating, animationSpeed, config.num_hidden_layers]);

  return (
    <div className="space-y-4">
      {/* Controls */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Settings className="w-5 h-5" />
            <span>Visualization Controls</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* View Mode */}
            <div className="space-y-3">
              <Label className="text-sm font-medium">View Mode</Label>
              <div className="flex flex-wrap gap-2">
                {['overview', 'layer', 'flow'].map((mode) => (
                  <Button
                    key={mode}
                    variant={viewMode === mode ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => setViewMode(mode as any)}
                    className="capitalize"
                  >
                    {mode}
                  </Button>
                ))}
              </div>
            </div>

            {/* Animation Controls */}
            <div className="space-y-3">
              <Label className="text-sm font-medium">Animation</Label>
              <div className="flex items-center space-x-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setIsAnimating(!isAnimating)}
                >
                  {isAnimating ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setCurrentLayer(0)}
                >
                  <RotateCcw className="w-4 h-4" />
                </Button>
                <Badge variant="secondary">Layer {currentLayer + 1}</Badge>
              </div>
            </div>

            {/* Display Options */}
            <div className="space-y-3">
              <Label className="text-sm font-medium">Display Options</Label>
              <div className="space-y-2">
                <div className="flex items-center space-x-2">
                  <Switch
                    id="connections"
                    checked={showConnections}
                    onCheckedChange={setShowConnections}
                  />
                  <Label htmlFor="connections" className="text-sm">Show Connections</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <Switch
                    id="labels"
                    checked={showLabels}
                    onCheckedChange={setShowLabels}
                  />
                  <Label htmlFor="labels" className="text-sm">Show Labels</Label>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* 3D Visualization */}
      <Card>
        <CardContent className="p-0">
          <Plot
            data={data as any}
            layout={layout as any}
            config={config_plotly}
            style={{ width: '100%', height: '800px' }}
          />
        </CardContent>
      </Card>

      {/* Architecture Info */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Eye className="w-5 h-5" />
            <span>Architecture Details</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">{config.num_hidden_layers}</div>
              <div className="text-sm text-muted-foreground">Transformer Layers</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">{config.num_attention_heads}</div>
              <div className="text-sm text-muted-foreground">Attention Heads</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">{config.hidden_size}</div>
              <div className="text-sm text-muted-foreground">Hidden Dimensions</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-orange-600">{config.vocab_size.toLocaleString()}</div>
              <div className="text-sm text-muted-foreground">Vocabulary Size</div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}