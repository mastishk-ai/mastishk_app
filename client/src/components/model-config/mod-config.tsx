import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ParameterControl } from "@/components/ui/parameter-control";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { GitBranch } from "lucide-react";
import { ModelConfig } from "@shared/schema";

interface ModConfigProps {
  config: ModelConfig;
  onUpdate: (updates: Partial<ModelConfig>) => void;
  onUpdateMod: (updates: Partial<NonNullable<ModelConfig['mod_config']>>) => void;
}

export function ModConfig({ config, onUpdate, onUpdateMod }: ModConfigProps) {
  const modParameters = [
    {
      key: 'skip_probability',
      config: {
        label: 'Skip Probability',
        description: 'Base probability of skipping a layer (0.0-0.5)',
        type: 'range' as const,
        range: { min: 0.0, max: 0.5, step: 0.05, default: 0.2 }
      }
    },
    {
      key: 'min_layers_per_token',
      config: {
        label: 'Min Layers per Token',
        description: 'Minimum layers each token must pass through (4-32)',
        type: 'range' as const,
        range: { min: 4, max: 32, step: 2, default: 12 }
      }
    },
    {
      key: 'capacity_factor',
      config: {
        label: 'Capacity Factor',
        description: 'Fraction of tokens to process per layer (0.1-1.0)',
        type: 'range' as const,
        range: { min: 0.1, max: 1.0, step: 0.1, default: 0.8 }
      }
    },
    {
      key: 'router_aux_loss_weight',
      config: {
        label: 'Router Aux Loss Weight',
        description: 'Weight for router auxiliary loss (0.001-0.1)',
        type: 'range' as const,
        range: { min: 0.001, max: 0.1, step: 0.001, default: 0.01 }
      }
    },
    {
      key: 'router_hidden_dim',
      config: {
        label: 'Router Hidden Dimension',
        description: 'Hidden dimension for router network (64-1024)',
        type: 'range' as const,
        range: { min: 64, max: 1024, step: 64, default: 256 }
      }
    },
    {
      key: 'temperature',
      config: {
        label: 'Router Temperature',
        description: 'Temperature for router softmax (0.1-2.0)',
        type: 'range' as const,
        range: { min: 0.1, max: 2.0, step: 0.1, default: 1.0 }
      }
    }
  ];

  const routerTypes = [
    { value: 'learned', label: 'Learned Router' },
    { value: 'random', label: 'Random Router' },
    { value: 'periodic', label: 'Periodic Router' },
  ];

  const loadBalancingTypes = [
    { value: 'auxiliary', label: 'Auxiliary Loss' },
    { value: 'capacity', label: 'Capacity Based' },
  ];

  return (
    <Card className="overflow-hidden">
      <CardHeader className="bg-muted/50">
        <CardTitle className="flex items-center">
          <GitBranch className="w-5 h-5 mr-2 text-green-500" />
          Mixture of Depths (MoD)
        </CardTitle>
        <p className="text-sm text-muted-foreground">Dynamic depth control for efficient processing</p>
      </CardHeader>
      <CardContent className="p-6">
        <div className="space-y-6">
          {/* Enable MoD Toggle */}
          <div className="flex items-center space-x-3">
            <Switch
              checked={config.use_mod}
              onCheckedChange={(checked) => onUpdate({ use_mod: checked })}
            />
            <Label className="text-sm font-medium">Enable Mixture of Depths</Label>
          </div>

          {config.use_mod && config.mod_config && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Basic Configuration */}
              <div className="space-y-4">
                <h4 className="text-sm font-semibold text-foreground uppercase tracking-wide">Basic Configuration</h4>
                
                <ParameterControl
                  config={{
                    label: 'Router Type',
                    description: 'Type of routing strategy',
                    type: 'select',
                    options: routerTypes
                  }}
                  value={config.mod_config.router_type}
                  onChange={(value) => onUpdateMod({ router_type: value })}
                />

                {modParameters.slice(0, 3).map(({ key, config: paramConfig }) => (
                  <ParameterControl
                    key={key}
                    config={paramConfig}
                    value={(config.mod_config as any)[key]}
                    onChange={(value) => onUpdateMod({ [key]: value } as any)}
                  />
                ))}
              </div>

              {/* Advanced Configuration */}
              <div className="space-y-4">
                <h4 className="text-sm font-semibold text-foreground uppercase tracking-wide">Advanced Settings</h4>
                
                <ParameterControl
                  config={{
                    label: 'Load Balancing Type',
                    description: 'Load balancing strategy',
                    type: 'select',
                    options: loadBalancingTypes
                  }}
                  value={config.mod_config.load_balancing_type}
                  onChange={(value) => onUpdateMod({ load_balancing_type: value })}
                />

                {modParameters.slice(3).map(({ key, config: paramConfig }) => (
                  <ParameterControl
                    key={key}
                    config={paramConfig}
                    value={(config.mod_config as any)[key]}
                    onChange={(value) => onUpdateMod({ [key]: value } as any)}
                  />
                ))}

                {/* Additional Switches */}
                <div className="space-y-3">
                  <div className="flex items-center space-x-3">
                    <Switch
                      checked={config.mod_config.use_gumbel_softmax}
                      onCheckedChange={(checked) => onUpdateMod({ use_gumbel_softmax: checked })}
                    />
                    <div>
                      <Label className="text-sm font-medium">Use Gumbel Softmax</Label>
                      <p className="text-xs text-muted-foreground">Differentiable discrete sampling</p>
                    </div>
                  </div>

                  <div className="flex items-center space-x-3">
                    <Switch
                      checked={config.mod_config.straight_through}
                      onCheckedChange={(checked) => onUpdateMod({ straight_through: checked })}
                    />
                    <div>
                      <Label className="text-sm font-medium">Straight Through</Label>
                      <p className="text-xs text-muted-foreground">Use straight-through gradients</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {!config.use_mod && (
            <div className="text-center py-8 text-muted-foreground">
              <GitBranch className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <p>Enable Mixture of Depths to configure dynamic depth parameters</p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
