import { useState, useCallback } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { apiRequest } from '@/lib/queryClient';
import { ModelConfig } from '@shared/schema';
import { useToast } from '@/hooks/use-toast';

export function useModelConfig() {
  const { toast } = useToast();
  const [config, setConfig] = useState<ModelConfig>({
    // Core architecture defaults
    vocab_size: 32000,
    hidden_size: 4096,
    intermediate_size: 11008,
    num_hidden_layers: 32,
    num_attention_heads: 32,
    num_key_value_heads: 8,
    hidden_act: "silu",
    max_position_embeddings: 4096,
    initializer_range: 0.02,
    rms_norm_eps: 1e-5,
    learning_rate: 5e-4,
    
    // Advanced features
    use_flash_attention: true,
    use_differential_attention: false,
    differential_lambda_init: 0.5,
    use_minimax: false,
    minimax_layer_frequency: 4,
    minimax_adversarial_epsilon: 0.1,
    minimax_iterations: 3,
    lolcats_enabled: false,
    lolcats_compression_dim: 512,
    use_multi_token_prediction: false,
    n_predict_tokens: 4,
    
    // MoE configuration
    use_moe: false,
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
    
    // MoD configuration
    use_mod: false,
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
    },
  });

  const queryClient = useQueryClient();

  const updateConfig = useCallback((updates: Partial<ModelConfig>) => {
    console.log('🎲 useModelConfig: Updating config with:', updates);
    setConfig(prev => {
      const newConfig = { ...prev, ...updates };
      console.log('🆕 New config state:', newConfig);
      return newConfig;
    });
  }, []);

  const updateMoeConfig = useCallback((updates: Partial<NonNullable<ModelConfig['moe_config']>>) => {
    setConfig(prev => ({
      ...prev,
      moe_config: { ...prev.moe_config!, ...updates }
    }));
  }, []);

  const updateModConfig = useCallback((updates: Partial<NonNullable<ModelConfig['mod_config']>>) => {
    setConfig(prev => ({
      ...prev,
      mod_config: { ...prev.mod_config!, ...updates }
    }));
  }, []);

  const applyPreset = useCallback((preset: Partial<ModelConfig>) => {
    setConfig(prev => ({ ...prev, ...preset }));
  }, []);

  const createModelMutation = useMutation({
    mutationFn: async ({ name }: { name: string }) => {
      console.log('🚀 Starting model creation for:', name);
      console.log('📋 Using config:', config);
      
      const response = await apiRequest('POST', '/api/models', {
        name,
        config,
      });
      
      const result = await response.json();
      console.log('✅ Model creation API response:', result);
      return result;
    },
    onSuccess: async (data) => {
      console.log('🎉 Model created successfully:', data);
      
      // Log current cache state before invalidation
      const currentModels = queryClient.getQueryData(['/api/models']);
      console.log('📊 Current models in cache before invalidation:', currentModels);
      
      // Force immediate cache invalidation and refetch
      console.log('🔄 Invalidating models cache...');
      await queryClient.invalidateQueries({ queryKey: ['/api/models'] });
      
      console.log('🔄 Refetching models...');
      await queryClient.refetchQueries({ queryKey: ['/api/models'] });
      
      // Check cache state after refetch
      const updatedModels = queryClient.getQueryData(['/api/models']);
      console.log('📊 Updated models in cache after refetch:', updatedModels);
      
      // Invalidate related queries
      queryClient.invalidateQueries({ queryKey: ['/api/training-runs'] });
      queryClient.invalidateQueries({ queryKey: ['/api/checkpoints'] });
      
      console.log('💾 Cache operations completed for model:', data.name);
      
      // Show success notification
      toast({
        title: "Model Created Successfully",
        description: `Model "${data.name}" is now available in dropdowns`,
      });
    },
    onError: (error) => {
      console.error('❌ Model creation failed:', error);
      toast({
        title: "Model Creation Failed",
        description: error instanceof Error ? error.message : "Failed to create model",
        variant: "destructive",
      });
    },
  });

  const validateConfig = useCallback((): string[] => {
    const errors: string[] = [];

    // Basic validation
    if (config.hidden_size % config.num_attention_heads !== 0) {
      errors.push('Hidden size must be divisible by number of attention heads');
    }

    if (config.head_dim && config.head_dim * config.num_attention_heads !== config.hidden_size) {
      errors.push('Head dimension * attention heads must equal hidden size');
    }

    if (config.use_moe && config.moe_config) {
      if (config.moe_config.num_experts_per_tok > config.moe_config.num_experts) {
        errors.push('Experts per token cannot exceed total number of experts');
      }
    }

    if (config.use_mod && config.mod_config) {
      if (config.mod_config.skip_probability > 0.5) {
        errors.push('Skip probability should not exceed 0.5');
      }
    }

    return errors;
  }, [config]);

  return {
    config,
    updateConfig,
    updateMoeConfig,
    updateModConfig,
    applyPreset,
    createModel: createModelMutation.mutate,
    isCreating: createModelMutation.isPending,
    validateConfig,
  };
}
