import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ParameterControl } from "@/components/ui/parameter-control";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { GitBranch } from "lucide-react";
import { ModelConfig } from "@shared/schema";

interface MoeConfigProps {
  config: ModelConfig;
  onUpdate: (updates: Partial<ModelConfig>) => void;
  onUpdateMoe: (updates: Partial<NonNullable<ModelConfig['moe_config']>>) => void;
}

export function MoeConfig({ config, onUpdate, onUpdateMoe }: MoeConfigProps) {
  console.log('🔀 MoeConfig rendered with config:', config);
  
  // Provide default config if undefined
  const safeConfig = config || {
    use_moe: false,
    moe_config: {
      num_experts: 8,
      expert_capacity_factor: 1.0,
      top_k_experts: 2,
      load_balancing_loss_weight: 0.01,
      expert_dropout: 0.0
    }
  };
  const moeParameters = [
    {
      key: 'num_experts',
      config: {
        label: 'Number of Experts',
        description: 'Total number of expert networks (2-32)',
        type: 'range' as const,
        range: { min: 2, max: 32, step: 2, default: 8 }
      }
    },
    {
      key: 'num_experts_per_tok',
      config: {
        label: 'Experts per Token',
        description: 'Number of experts to activate per token (1-8)',
        type: 'range' as const,
        range: { min: 1, max: 8, step: 1, default: 2 }
      }
    },
    {
      key: 'expert_capacity_factor',
      config: {
        label: 'Expert Capacity Factor',
        description: 'Controls expert load balancing (0.5-2.0)',
        type: 'range' as const,
        range: { min: 0.5, max: 2.0, step: 0.25, default: 1.25 }
      }
    },
    {
      key: 'aux_loss_weight',
      config: {
        label: 'Auxiliary Loss Weight',
        description: 'Weight for load balancing loss (0.001-0.1)',
        type: 'range' as const,
        range: { min: 0.001, max: 0.1, step: 0.001, default: 0.01 }
      }
    },
    {
      key: 'router_dropout',
      config: {
        label: 'Router Dropout',
        description: 'Dropout rate for router (0.0-0.5)',
        type: 'range' as const,
        range: { min: 0.0, max: 0.5, step: 0.05, default: 0.1 }
      }
    },
    {
      key: 'moe_layer_frequency',
      config: {
        label: 'MoE Layer Frequency',
        description: 'Apply MoE every N layers (1-8)',
        type: 'range' as const,
        range: { min: 1, max: 8, step: 1, default: 2 }
      }
    }
  ];

  const routerTypes = [
    { value: 'top_k', label: 'Top-K Routing' },
    { value: 'expert_choice', label: 'Expert Choice' },
    { value: 'soft', label: 'Soft Routing' },
  ];

  const loadBalancingTypes = [
    { value: 'aux_loss', label: 'Auxiliary Loss' },
    { value: 'switch', label: 'Switch Transformer' },
    { value: 'sinkhorn', label: 'Sinkhorn' },
  ];

  return (
    <Card className="overflow-hidden">
      <CardHeader className="bg-muted/50">
        <CardTitle className="flex items-center">
          <GitBranch className="w-5 h-5 mr-2 text-purple-500" />
          Mixture of Experts (MoE)
        </CardTitle>
        <p className="text-sm text-muted-foreground">Configure sparse expert routing for efficient scaling</p>
      </CardHeader>
      <CardContent className="p-6">
        <div className="space-y-6">
          {/* Enable MoE Toggle */}
          <div className="flex items-center space-x-3">
            <Switch
              checked={config.use_moe}
              onCheckedChange={(checked) => onUpdate({ use_moe: checked })}
            />
            <Label className="text-sm font-medium">Enable Mixture of Experts</Label>
          </div>

          {config.use_moe && config.moe_config && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* MoE Parameters */}
              <div className="space-y-4">
                <h4 className="text-sm font-semibold text-foreground uppercase tracking-wide">Expert Configuration</h4>
                {moeParameters.slice(0, 3).map(({ key, config: paramConfig }) => (
                  <ParameterControl
                    key={key}
                    config={paramConfig}
                    value={(config.moe_config as any)[key]}
                    onChange={(value) => onUpdateMoe({ [key]: value } as any)}
                  />
                ))}
              </div>

              {/* Load Balancing */}
              <div className="space-y-4">
                <h4 className="text-sm font-semibold text-foreground uppercase tracking-wide">Load Balancing</h4>
                
                <ParameterControl
                  config={{
                    label: 'Router Type',
                    description: 'Expert selection strategy',
                    type: 'select',
                    options: routerTypes
                  }}
                  value={config.moe_config.router_type}
                  onChange={(value) => onUpdateMoe({ router_type: value })}
                />
                
                <ParameterControl
                  config={{
                    label: 'Load Balancing Type',
                    description: 'Load balancing strategy',
                    type: 'select',
                    options: loadBalancingTypes
                  }}
                  value={config.moe_config.load_balancing_type}
                  onChange={(value) => onUpdateMoe({ load_balancing_type: value })}
                />

                {moeParameters.slice(3).map(({ key, config: paramConfig }) => (
                  <ParameterControl
                    key={key}
                    config={paramConfig}
                    value={(config.moe_config as any)[key]}
                    onChange={(value) => onUpdateMoe({ [key]: value } as any)}
                  />
                ))}
              </div>
            </div>
          )}

          {!config.use_moe && (
            <div className="text-center py-8 text-muted-foreground">
              <GitBranch className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <p>Enable Mixture of Experts to configure expert routing parameters</p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
