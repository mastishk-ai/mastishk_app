import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Zap } from "lucide-react";
import { ModelConfig } from "@shared/schema";

interface ModelPresetsProps {
  onApplyPreset: (preset: Partial<ModelConfig>) => void;
  currentConfig?: any;
}

export function ModelPresets({ onApplyPreset, currentConfig }: ModelPresetsProps) {
  
  const handlePresetClick = (preset: any) => {
    console.log('🚀 Preset button clicked:', preset.name);
    console.log('📋 Preset configuration:', preset.config);
    
    // Apply the preset
    onApplyPreset(preset.config);
    
    console.log('✅ Preset applied successfully');
  };
  const presets = [
    {
      name: "Small (1B)",
      description: "Fast training, good for experiments",
      config: {
        hidden_size: 2048,
        intermediate_size: 5504,
        num_hidden_layers: 24,
        num_attention_heads: 16,
        num_key_value_heads: 4,
        max_position_embeddings: 2048,
        use_moe: false,
        use_mod: false,
        use_flash_attention: true
      }
    },
    {
      name: "Medium (7B)",
      description: "Balanced performance",
      config: {
        hidden_size: 4096,
        intermediate_size: 11008,
        num_hidden_layers: 32,
        num_attention_heads: 32,
        num_key_value_heads: 8,
        max_position_embeddings: 4096,
        use_moe: true,
        use_mod: true,
        use_flash_attention: true,
        moe_config: {
          num_experts: 8,
          num_experts_per_tok: 2,
          expert_capacity_factor: 1.25,
          aux_loss_weight: 0.01,
          router_z_loss_weight: 0.001,
          router_dropout: 0.1,
          expert_dropout: 0.1,
          moe_layer_frequency: 2,
          load_balancing_type: "aux_loss",
          router_type: "top_k",
        }
      }
    },
    {
      name: "Large (13B)",
      description: "High performance, slower training",
      config: {
        hidden_size: 5120,
        intermediate_size: 13824,
        num_hidden_layers: 40,
        num_attention_heads: 40,
        num_key_value_heads: 8,
        max_position_embeddings: 8192,
        use_moe: true,
        use_mod: true,
        use_flash_attention: true,
        use_differential_attention: true,
        moe_config: {
          num_experts: 16,
          num_experts_per_tok: 2,
          expert_capacity_factor: 1.25,
          aux_loss_weight: 0.01,
          router_z_loss_weight: 0.001,
          router_dropout: 0.1,
          expert_dropout: 0.1,
          moe_layer_frequency: 2,
          load_balancing_type: "aux_loss",
          router_type: "top_k",
        },
        mod_config: {
          enabled: true,
          router_type: "learned",
          skip_probability: 0.25,
          min_layers_per_token: 16,
          capacity_factor: 0.8,
          router_aux_loss_weight: 0.01,
          router_z_loss_weight: 0.001,
          load_balancing_type: "auxiliary",
          router_hidden_dim: 512,
          router_dropout: 0.1,
          temperature: 1.0,
          use_gumbel_softmax: true,
          straight_through: true,
          block_size: 1,
        }
      }
    },
    {
      name: "Research",
      description: "All advanced features enabled",
      config: {
        hidden_size: 4096,
        intermediate_size: 11008,
        num_hidden_layers: 32,
        num_attention_heads: 32,
        num_key_value_heads: 8,
        max_position_embeddings: 16384,
        use_moe: true,
        use_mod: true,
        use_flash_attention: true,
        use_differential_attention: true,
        use_minimax: true,
        lolcats_enabled: true,
        use_multi_token_prediction: true,
        minimax_layer_frequency: 4,
        minimax_adversarial_epsilon: 0.1,
        minimax_iterations: 3,
        lolcats_compression_dim: 512,
        n_predict_tokens: 4,
        moe_config: {
          num_experts: 8,
          num_experts_per_tok: 2,
          expert_capacity_factor: 1.25,
          aux_loss_weight: 0.01,
          router_z_loss_weight: 0.001,
          router_dropout: 0.1,
          expert_dropout: 0.1,
          moe_layer_frequency: 2,
          load_balancing_type: "aux_loss",
          router_type: "top_k",
        },
        mod_config: {
          enabled: true,
          router_type: "learned",
          skip_probability: 0.2,
          min_layers_per_token: 12,
          capacity_factor: 0.8,
          router_aux_loss_weight: 0.01,
          router_z_loss_weight: 0.001,
          load_balancing_type: "auxiliary",
          router_hidden_dim: 256,
          router_dropout: 0.1,
          temperature: 1.0,
          use_gumbel_softmax: true,
          straight_through: true,
          block_size: 1,
        }
      }
    }
  ];

  return (
    <Card className="flex-1 premium-card">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-lg text-truncate">
          <Zap className="w-5 h-5 text-yellow-500 flex-shrink-0" />
          <span className="text-truncate">Quick Presets</span>
        </CardTitle>
      </CardHeader>
      <CardContent className="card-content-spacing">
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {presets.map((preset) => (
            <Button
              key={preset.name}
              variant="outline"
              className="p-4 h-auto text-left overflow-hidden bg-background text-foreground border-border hover:bg-primary/10 active:bg-primary/20 transition-all duration-200 cursor-pointer active:scale-95"
              onClick={(e) => {
                e.preventDefault();
                e.stopPropagation();
                console.log('🔥 PRESET BUTTON CLICKED:', preset.name);
                console.log('🔧 Config being applied:', preset.config);
                try {
                  onApplyPreset(preset.config);
                  console.log('✅ PRESET APPLIED SUCCESSFULLY');
                } catch (error) {
                  console.error('❌ PRESET APPLICATION FAILED:', error);
                }
              }}
            >
              <div className="w-full overflow-hidden">
                <div className="font-medium text-foreground text-truncate">{preset.name}</div>
                <div className="text-xs text-muted-foreground mt-1 break-words line-clamp-2">{preset.description}</div>
              </div>
            </Button>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
