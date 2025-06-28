import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ParameterControl } from "@/components/ui/parameter-control";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Settings } from "lucide-react";
import { useTraining } from "@/hooks/use-training";

export function TrainingConfig() {
  const { config, updateConfig } = useTraining();

  const optimizationParameters = [
    {
      key: 'learning_rate',
      config: {
        label: 'Learning Rate',
        description: 'Step size for parameter updates (1e-6 to 1e-2)',
        type: 'range' as const,
        range: { min: 0.00001, max: 0.01, step: 0.00001, default: 0.0005 }
      }
    },
    {
      key: 'weight_decay',
      config: {
        label: 'Weight Decay',
        description: 'L2 regularization strength (0.0 to 1.0)',
        type: 'range' as const,
        range: { min: 0.0, max: 1.0, step: 0.001, default: 0.01 }
      }
    },
    {
      key: 'max_grad_norm',
      config: {
        label: 'Max Gradient Norm',
        description: 'Gradient clipping threshold (0.0 to 10.0)',
        type: 'range' as const,
        range: { min: 0.0, max: 10.0, step: 0.1, default: 1.0 }
      }
    }
  ];

  const trainingStepsParameters = [
    {
      key: 'max_steps',
      config: {
        label: 'Max Steps',
        description: 'Maximum training steps',
        type: 'number' as const
      }
    },
    {
      key: 'eval_steps',
      config: {
        label: 'Eval Steps',
        description: 'Evaluation frequency',
        type: 'number' as const
      }
    },
    {
      key: 'save_steps',
      config: {
        label: 'Save Steps',
        description: 'Checkpoint save frequency',
        type: 'number' as const
      }
    },
    {
      key: 'warmup_steps',
      config: {
        label: 'Warmup Steps',
        description: 'Learning rate warmup period',
        type: 'number' as const
      }
    }
  ];

  const batchParameters = [
    {
      key: 'batch_size',
      config: {
        label: 'Batch Size',
        description: 'Number of samples per batch',
        type: 'select' as const,
        options: [
          { value: 1, label: '1' },
          { value: 2, label: '2' },
          { value: 4, label: '4' },
          { value: 8, label: '8' },
          { value: 16, label: '16' },
          { value: 32, label: '32' }
        ]
      }
    },
    {
      key: 'gradient_accumulation_steps',
      config: {
        label: 'Gradient Accumulation',
        description: 'Steps to accumulate gradients (1-64)',
        type: 'range' as const,
        range: { min: 1, max: 64, step: 1, default: 4 }
      }
    }
  ];

  const advancedOptions = [
    { key: 'mixed_precision', label: 'Mixed Precision', description: 'Use FP16 for faster training' },
    { key: 'gradient_checkpointing', label: 'Gradient Checkpointing', description: 'Trade computation for memory' },
    { key: 'early_stopping', label: 'Early Stopping', description: 'Stop when loss plateaus' },
    { key: 'save_optimizer_state', label: 'Save Optimizer State', description: 'Include optimizer in checkpoints' },
    { key: 'save_scheduler_state', label: 'Save Scheduler State', description: 'Include scheduler in checkpoints' },
    { key: 'verify_integrity', label: 'Verify Integrity', description: 'Verify checkpoint integrity' }
  ];

  return (
    <Card className="overflow-hidden">
      <CardHeader className="bg-muted/50">
        <CardTitle className="flex items-center">
          <Settings className="w-5 h-5 mr-2 text-purple-500" />
          Training Configuration
        </CardTitle>
        <p className="text-sm text-muted-foreground">Configure training hyperparameters and options</p>
      </CardHeader>
      <CardContent className="p-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Optimization Settings */}
          <div className="space-y-4">
            <h4 className="text-sm font-semibold text-foreground uppercase tracking-wide">Optimization</h4>
            
            {optimizationParameters.map(({ key, config: paramConfig }) => (
              <ParameterControl
                key={key}
                config={paramConfig}
                value={(config as any)[key]}
                onChange={(value) => updateConfig({ [key]: value } as any)}
              />
            ))}

            {batchParameters.map(({ key, config: paramConfig }) => (
              <ParameterControl
                key={key}
                config={paramConfig}
                value={(config as any)[key]}
                onChange={(value) => updateConfig({ [key]: value } as any)}
              />
            ))}
          </div>
          
          {/* Training Steps */}
          <div className="space-y-4">
            <h4 className="text-sm font-semibold text-foreground uppercase tracking-wide">Training Steps</h4>
            
            {trainingStepsParameters.map(({ key, config: paramConfig }) => (
              <ParameterControl
                key={key}
                config={paramConfig}
                value={(config as any)[key]}
                onChange={(value) => updateConfig({ [key]: value } as any)}
              />
            ))}

            {/* Early Stopping Configuration */}
            {config.early_stopping && (
              <>
                <ParameterControl
                  config={{
                    label: 'Early Stopping Patience',
                    description: 'Steps to wait before stopping',
                    type: 'number'
                  }}
                  value={config.early_stopping_patience}
                  onChange={(value) => updateConfig({ early_stopping_patience: value })}
                />
                <ParameterControl
                  config={{
                    label: 'Early Stopping Threshold',
                    description: 'Minimum improvement threshold',
                    type: 'number'
                  }}
                  value={config.early_stopping_threshold}
                  onChange={(value) => updateConfig({ early_stopping_threshold: value })}
                />
              </>
            )}
          </div>
          
          {/* Advanced Options */}
          <div className="space-y-4">
            <h4 className="text-sm font-semibold text-foreground uppercase tracking-wide">Advanced</h4>
            
            <div className="space-y-3">
              {advancedOptions.map(({ key, label, description }) => (
                <div key={key} className="flex items-center space-x-3">
                  <Switch
                    checked={(config as any)[key]}
                    onCheckedChange={(checked) => updateConfig({ [key]: checked } as any)}
                  />
                  <div>
                    <Label className="text-sm font-medium">{label}</Label>
                    <p className="text-xs text-muted-foreground">{description}</p>
                  </div>
                </div>
              ))}
            </div>

            <ParameterControl
              config={{
                label: 'Max Checkpoints',
                description: 'Maximum checkpoints to keep',
                type: 'number'
              }}
              value={config.max_checkpoints}
              onChange={(value) => updateConfig({ max_checkpoints: value })}
            />

            <ParameterControl
              config={{
                label: 'Random Seed',
                description: 'Seed for reproducibility',
                type: 'number'
              }}
              value={config.seed}
              onChange={(value) => updateConfig({ seed: value })}
            />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
