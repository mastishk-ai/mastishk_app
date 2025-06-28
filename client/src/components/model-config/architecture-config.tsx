import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ParameterControl } from "@/components/ui/parameter-control";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Layers } from "lucide-react";
import { ModelConfig } from "@shared/schema";

interface ArchitectureConfigProps {
  config: ModelConfig;
  onUpdate: (updates: Partial<ModelConfig>) => void;
}

export function ArchitectureConfig({ config, onUpdate }: ArchitectureConfigProps) {
  const coreParameters = [
    {
      key: 'hidden_size',
      config: {
        label: 'Hidden Size',
        description: 'Model dimension (512-16384)',
        type: 'range' as const,
        range: { min: 512, max: 16384, step: 128, default: 4096 }
      }
    },
    {
      key: 'num_hidden_layers',
      config: {
        label: 'Number of Layers',
        description: 'Transformer layers (6-96)',
        type: 'range' as const,
        range: { min: 6, max: 96, step: 2, default: 32 }
      }
    },
    {
      key: 'num_attention_heads',
      config: {
        label: 'Attention Heads',
        description: 'Multi-head attention (8-128)',
        type: 'range' as const,
        range: { min: 8, max: 128, step: 4, default: 32 }
      }
    },
    {
      key: 'num_key_value_heads',
      config: {
        label: 'Key-Value Heads',
        description: 'For grouped query attention (1-128)',
        type: 'range' as const,
        range: { min: 1, max: 128, step: 1, default: 8 }
      }
    },
    {
      key: 'intermediate_size',
      config: {
        label: 'Intermediate Size',
        description: 'FFN dimension (1024-65536)',
        type: 'range' as const,
        range: { min: 1024, max: 65536, step: 256, default: 11008 }
      }
    },
    {
      key: 'max_position_embeddings',
      config: {
        label: 'Max Position Embeddings',
        description: 'Maximum sequence length',
        type: 'range' as const,
        range: { min: 512, max: 262144, step: 512, default: 4096 }
      }
    }
  ];

  const advancedFeatures = [
    { key: 'use_flash_attention', label: 'Flash Attention', description: 'Memory-efficient attention' },
    { key: 'use_differential_attention', label: 'Differential Attention', description: 'Enhanced attention mechanism' },
    { key: 'use_minimax', label: 'MiniMax Optimization', description: 'Adversarial training' },
    { key: 'lolcats_enabled', label: 'LoLCATs Compression', description: 'Low-rank compression' },
    { key: 'use_multi_token_prediction', label: 'Multi-Token Prediction', description: 'Predict multiple tokens ahead' },
  ];

  return (
    <Card className="overflow-hidden">
      <CardHeader className="bg-muted/50">
        <CardTitle className="flex items-center">
          <Layers className="w-5 h-5 mr-2 text-blue-500" />
          Architecture Configuration
        </CardTitle>
        <p className="text-sm text-muted-foreground">Configure the core transformer architecture</p>
      </CardHeader>
      <CardContent className="p-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
          {/* Core Parameters */}
          <div className="space-y-4">
            <h4 className="text-sm font-semibold text-foreground uppercase tracking-wide">Core Parameters</h4>
            {coreParameters.map(({ key, config: paramConfig }) => (
              <ParameterControl
                key={key}
                config={paramConfig}
                value={(config as any)[key]}
                onChange={(value) => onUpdate({ [key]: value } as any)}
              />
            ))}
          </div>
          
          {/* Advanced Features */}
          <div className="space-y-4">
            <h4 className="text-sm font-semibold text-foreground uppercase tracking-wide">Advanced Features</h4>
            <div className="space-y-3">
              {advancedFeatures.map(({ key, label, description }) => (
                <div key={key} className="flex items-center space-x-3">
                  <Switch
                    checked={(config as any)[key]}
                    onCheckedChange={(checked) => onUpdate({ [key]: checked } as any)}
                  />
                  <div>
                    <Label className="text-sm font-medium">{label}</Label>
                    <p className="text-xs text-muted-foreground">{description}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
          
          {/* Additional Parameters */}
          <div className="space-y-4">
            <h4 className="text-sm font-semibold text-foreground uppercase tracking-wide">Additional Settings</h4>
            
            <ParameterControl
              config={{
                label: 'RMS Norm Epsilon',
                description: 'Normalization epsilon',
                type: 'number'
              }}
              value={config.rms_norm_eps}
              onChange={(value) => onUpdate({ rms_norm_eps: value })}
            />
            
            <ParameterControl
              config={{
                label: 'Initializer Range',
                description: 'Weight initialization range',
                type: 'number'
              }}
              value={config.initializer_range}
              onChange={(value) => onUpdate({ initializer_range: value })}
            />
            
            <ParameterControl
              config={{
                label: 'Hidden Activation',
                description: 'Activation function',
                type: 'select',
                options: [
                  { value: 'silu', label: 'SiLU' },
                  { value: 'relu', label: 'ReLU' },
                  { value: 'gelu', label: 'GELU' },
                  { value: 'swiglu', label: 'SwiGLU' },
                ]
              }}
              value={config.hidden_act}
              onChange={(value) => onUpdate({ hidden_act: value })}
            />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
