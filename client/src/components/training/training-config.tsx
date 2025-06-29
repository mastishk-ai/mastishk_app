import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ParameterControl } from "@/components/ui/parameter-control";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { Settings, Eye, Activity } from "lucide-react";
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
    { key: 'verify_integrity', label: 'Verify Integrity', description: 'Verify checkpoint integrity' },
    { key: 'enable_weight_logging', label: 'Weight Logging', description: 'Log weight snapshots during training' },
    { key: 'weight_verification', label: 'Weight Verification', description: 'Verify weight updates after optimizer steps' }
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

            {/* Weight Logging Section */}
            <div className="mt-6 p-4 bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-950/20 dark:to-indigo-950/20 rounded-lg border border-blue-200 dark:border-blue-800">
              <div className="flex items-center space-x-3 mb-3">
                <Eye className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                <h5 className="text-sm font-semibold text-blue-900 dark:text-blue-100 uppercase tracking-wide">Weight Monitoring</h5>
              </div>
              
              <div className="space-y-3">
                <div className="flex items-center space-x-3">
                  <Checkbox
                    id="enable-weight-logging"
                    checked={(config as any).enable_weight_logging || false}
                    onCheckedChange={(checked) => updateConfig({ enable_weight_logging: checked } as any)}
                    className="data-[state=checked]:bg-blue-600 data-[state=checked]:border-blue-600"
                  />
                  <div className="flex-1">
                    <Label htmlFor="enable-weight-logging" className="text-sm font-medium text-blue-900 dark:text-blue-100 cursor-pointer">
                      Enable Weight Snapshots
                    </Label>
                    <p className="text-xs text-blue-700 dark:text-blue-300">
                      {(config as any).enable_weight_logging 
                        ? "Weight snapshots will be logged every optimizer step" 
                        : "Click to enable detailed weight change tracking"}
                    </p>
                  </div>
                  <Activity className={`w-4 h-4 ${(config as any).enable_weight_logging ? 'text-green-500' : 'text-gray-400'}`} />
                </div>

                {(config as any).enable_weight_logging && (
                  <div className="ml-6 pl-3 border-l-2 border-blue-200 dark:border-blue-700 space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-medium text-blue-800 dark:text-blue-200">Snapshot Frequency</span>
                      <span className="text-xs text-blue-600 dark:text-blue-400">Every optimizer step</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-medium text-blue-800 dark:text-blue-200">Verification</span>
                      <span className="text-xs text-blue-600 dark:text-blue-400">Pre & post optimizer updates</span>
                    </div>
                    <p className="text-xs text-blue-600 dark:text-blue-400 italic">
                      This will track weight changes to verify training progress and detect optimization issues.
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
