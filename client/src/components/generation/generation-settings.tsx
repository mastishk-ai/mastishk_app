import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ParameterControl } from "@/components/ui/parameter-control";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Settings, Sliders } from "lucide-react";

interface GenerationConfig {
  temperature: number;
  top_p: number;
  top_k: number;
  max_length: number;
  repetition_penalty: number;
  length_penalty: number;
  no_repeat_ngram_size: number;
  do_sample: boolean;
  early_stopping: boolean;
  num_beams: number;
  use_multi_token_prediction: boolean;
}

interface GenerationSettingsProps {
  config: GenerationConfig;
  onUpdate: (updates: Partial<GenerationConfig>) => void;
}

export function GenerationSettings({ config, onUpdate }: GenerationSettingsProps) {
  const basicParameters = [
    {
      key: 'max_length',
      config: {
        label: 'Max Length',
        description: 'Maximum number of tokens to generate (10-2048)',
        type: 'range' as const,
        range: { min: 10, max: 2048, step: 10, default: 500 }
      }
    },
    {
      key: 'temperature',
      config: {
        label: 'Temperature',
        description: 'Controls randomness (0.1 = conservative, 2.0 = creative)',
        type: 'range' as const,
        range: { min: 0.1, max: 2.0, step: 0.1, default: 0.7 }
      }
    },
    {
      key: 'top_p',
      config: {
        label: 'Top-p (Nucleus Sampling)',
        description: 'Cumulative probability cutoff (0.1-1.0)',
        type: 'range' as const,
        range: { min: 0.1, max: 1.0, step: 0.05, default: 0.9 }
      }
    },
    {
      key: 'top_k',
      config: {
        label: 'Top-k',
        description: 'Number of top tokens to consider (1-100)',
        type: 'range' as const,
        range: { min: 1, max: 100, step: 1, default: 50 }
      }
    }
  ];

  const advancedParameters = [
    {
      key: 'repetition_penalty',
      config: {
        label: 'Repetition Penalty',
        description: 'Penalty for repeating tokens (1.0-2.0)',
        type: 'range' as const,
        range: { min: 1.0, max: 2.0, step: 0.05, default: 1.1 }
      }
    },
    {
      key: 'length_penalty',
      config: {
        label: 'Length Penalty',
        description: 'Penalty for length (0.5-2.0)',
        type: 'range' as const,
        range: { min: 0.5, max: 2.0, step: 0.1, default: 1.0 }
      }
    },
    {
      key: 'no_repeat_ngram_size',
      config: {
        label: 'No Repeat N-gram Size',
        description: 'Prevent repeating n-grams (0-10)',
        type: 'range' as const,
        range: { min: 0, max: 10, step: 1, default: 3 }
      }
    },
    {
      key: 'num_beams',
      config: {
        label: 'Number of Beams',
        description: 'Beam search width (1-10)',
        type: 'range' as const,
        range: { min: 1, max: 10, step: 1, default: 1 }
      }
    }
  ];

  const booleanOptions = [
    { key: 'do_sample', label: 'Do Sample', description: 'Enable sampling-based generation' },
    { key: 'early_stopping', label: 'Early Stopping', description: 'Stop when EOS token is generated' },
    { key: 'use_multi_token_prediction', label: 'Multi-Token Prediction', description: 'Use multi-token prediction if available' }
  ];

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Basic Settings */}
      <Card className="overflow-hidden">
        <CardHeader className="bg-muted/50">
          <CardTitle className="flex items-center">
            <Settings className="w-5 h-5 mr-2 text-blue-500" />
            Generation Settings
          </CardTitle>
        </CardHeader>
        <CardContent className="p-6 space-y-4">
          {basicParameters.map(({ key, config: paramConfig }) => (
            <ParameterControl
              key={key}
              config={paramConfig}
              value={(config as any)[key]}
              onChange={(value) => onUpdate({ [key]: value } as any)}
            />
          ))}
        </CardContent>
      </Card>
      
      {/* Advanced Settings */}
      <Card className="overflow-hidden">
        <CardHeader className="bg-muted/50">
          <CardTitle className="flex items-center">
            <Sliders className="w-5 h-5 mr-2 text-purple-500" />
            Advanced Settings
          </CardTitle>
        </CardHeader>
        <CardContent className="p-6 space-y-4">
          {advancedParameters.map(({ key, config: paramConfig }) => (
            <ParameterControl
              key={key}
              config={paramConfig}
              value={(config as any)[key]}
              onChange={(value) => onUpdate({ [key]: value } as any)}
            />
          ))}
          
          {/* Boolean Options */}
          <div className="space-y-3 pt-4 border-t">
            <h4 className="text-sm font-semibold text-foreground">Options</h4>
            {booleanOptions.map(({ key, label, description }) => (
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
        </CardContent>
      </Card>
    </div>
  );
}
