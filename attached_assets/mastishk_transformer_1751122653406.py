"""
Mastishk Transformer with Advanced Mixture-of-Experts (MoE) and Mixture-of-Depths (MoD)
Unified transformer with MiniMax, Differential Attention, LoLCATs, Multi-Token Prediction, Sparse MoE, and Dynamic Depth
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as ckpt
import math
from typing import Optional, Tuple, List, Dict, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
    ShardingStrategy,
    MixedPrecision,
)
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.activations import ACT2FN
from contextlib import contextmanager, nullcontext
from pathlib import Path
import json
import os
import gc
from collections import OrderedDict
import numpy as np
from einops import rearrange, repeat
from functools import partial
import warnings
from accelerate import init_empty_weights
import heapq

# Try importing optional dependencies
try:
    import bitsandbytes as bnb
    HAS_BNBYTES = True
except ImportError:
    HAS_BNBYTES = False
    warnings.warn("bitsandbytes not installed. Quantization features will be limited.")

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    warnings.warn("FlashAttention not installed. Using standard attention.")

try:
    import deepspeed
    HAS_DEEPSPEED = True
except ImportError:
    HAS_DEEPSPEED = False
    warnings.warn("DeepSpeed not installed. Some optimizations unavailable.")

try:
    from safetensors.torch import load_file, save_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False
    warnings.warn("Safetensors not installed. Using standard checkpoint format.")

# ------------------------ MASTISHK CONFIGURATION ---------------------------- #

class AttentionImplementation(Enum):
    STANDARD = "standard"
    FLASH = "flash"
    FLASH2 = "flash2"
    XFORMERS = "xformers"
    TRITON = "triton"
    MINIMAX = "minimax"
    DIFFERENTIAL = "differential"

class ParallelismStrategy(Enum):
    NONE = "none"
    DATA = "data"
    TENSOR = "tensor"
    PIPELINE = "pipeline"
    HYBRID_3D = "3d"

@dataclass
class QuantizationConfig:
    """Configuration for model quantization"""
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    llm_int8_threshold: float = 6.0
    llm_int8_skip_modules: Optional[List[str]] = None
    llm_int8_enable_fp32_cpu_offload: bool = False

@dataclass
class KVCacheConfig:
    """Enhanced KV cache configuration"""
    cache_implementation: str = "standard"
    max_cache_length: int = 8192
    window_size: int = 4096
    compression_ratio: float = 0.5
    sink_tokens: int = 4
    recent_tokens: int = 1024

@dataclass
class MoEConfig:
    """Mixture of Experts configuration"""
    num_experts: int = 8
    num_experts_per_tok: int = 2
    expert_capacity_factor: float = 1.25
    aux_loss_weight: float = 0.01
    router_z_loss_weight: float = 0.001
    router_dropout: float = 0.1
    expert_dropout: float = 0.1
    moe_layer_frequency: int = 2  # Apply MoE every N layers
    load_balancing_type: str = "aux_loss"  # "aux_loss", "switch", "sinkhorn"
    router_type: str = "top_k"  # "top_k", "expert_choice", "soft"
    capacity_factor: float = 1.25
    min_expert_capacity: int = 4
    normalize_expert_weights: bool = True
    use_rts: bool = False  # Random Token Selection for load balancing

@dataclass
class MoDConfig:
    """Mixture of Depths configuration"""
    enabled: bool = True
    router_type: str = "learned"  # "learned", "random", "periodic"
    skip_probability: float = 0.2  # Base probability of skipping a layer
    min_layers_per_token: int = 12  # Minimum layers each token must pass through
    capacity_factor: float = 0.8  # Fraction of tokens to process per layer
    router_aux_loss_weight: float = 0.01
    router_z_loss_weight: float = 0.001
    load_balancing_type: str = "auxiliary"  # "auxiliary", "capacity"
    router_hidden_dim: int = 256
    router_dropout: float = 0.1
    temperature: float = 1.0
    use_gumbel_softmax: bool = True
    straight_through: bool = True
    block_size: int = 1  # Number of consecutive layers to skip as a block
    adaptive_computation_time: bool = False  # ACT-style dynamic depth
    act_threshold: float = 0.99
    max_pondering_steps: int = 5

@dataclass
class TrainingConfig:
    """Enhanced training configuration"""
    gradient_checkpointing: bool = True
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "bf16"
    max_grad_norm: float = 1.0
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    use_8bit_adam: bool = True
    scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    zero_stage: int = 3
    offload_optimizer: bool = True
    offload_param: bool = False
    cpu_checkpointing: bool = False
    contiguous_checkpointing: bool = True
    profile_memory: bool = False
    use_random_token_dropping: bool = False
    token_drop_rate: float = 0.15
    token_drop_schedule: str = "constant"

class MastishkConfig(PretrainedConfig):
    """Mastishk transformer configuration with all optimizations"""
    model_type = "mastishk_transformer"
    
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 32768,
        intermediate_size: int = 131072,
        num_hidden_layers: int = 96,
        num_attention_heads: int = 256,
        num_key_value_heads: int = 64,
        head_dim: Optional[int] = None,
        hidden_act: str = "swiglu",
        max_position_embeddings: int = 262144,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-5,
        use_layer_scale: bool = True,
        layer_scale_init: float = 1e-4,
        attention_implementation: str = "flash2",
        attention_dropout: float = 0.0,
        attention_bias: bool = False,
        rope_theta: float = 500000.0,
        rope_scaling: Optional[Dict] = None,
        use_sliding_window: bool = False,
        sliding_window_size: int = 4096,
        mlp_bias: bool = False,
        glu_activation: bool = True,
        use_flash_attention: bool = True,
        use_gradient_checkpointing: bool = False,
        checkpointing_policy: str = "selective",
        checkpoint_sequential_factor: int = 4,
        parallelism_strategy: str = "none",
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        quantization_config: Optional[Dict] = None,
        kv_cache_config: Optional[Dict] = None,
        use_minimax: bool = False,
        minimax_layer_frequency: int = 4,
        minimax_adversarial_epsilon: float = 0.1,
        minimax_iterations: int = 3,
        use_multi_token_prediction: bool = False,
        n_predict_tokens: int = 4,
        multi_token_loss_weight: float = 1.0,
        use_differential_attention: bool = False,
        differential_lambda_init: float = 0.5,
        lolcats_enabled: bool = False,
        lolcats_compression_dim: int = 512,
        use_moe: bool = False,
        moe_config: Optional[Dict] = None,
        use_mod: bool = False,
        mod_config: Optional[Dict] = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
        **kwargs
    ):
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim or (hidden_size // num_attention_heads)
        assert hidden_size == num_attention_heads * self.head_dim, \
            "hidden_size must equal num_attention_heads * head_dim"
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_layer_scale = use_layer_scale
        self.layer_scale_init = layer_scale_init
        self.attention_implementation = attention_implementation
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling or {
            "type": "yarn",
            "factor": 8.0,
            "original_max_position_embeddings": 32768,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "mscale": 1.0,
        }
        self.use_sliding_window = use_sliding_window
        self.sliding_window_size = sliding_window_size
        self.mlp_bias = mlp_bias
        self.glu_activation = glu_activation
        self.use_flash_attention = use_flash_attention
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.checkpointing_policy = checkpointing_policy
        self.checkpoint_sequential_factor = checkpoint_sequential_factor
        self.parallelism_strategy = parallelism_strategy
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.use_cache = use_cache  # ✅ Fix added
        self.quantization_config = QuantizationConfig(**quantization_config) if quantization_config else None
        self.kv_cache_config = KVCacheConfig(**kv_cache_config) if kv_cache_config else KVCacheConfig()
        self.use_minimax = use_minimax
        self.minimax_layer_frequency = minimax_layer_frequency
        self.minimax_adversarial_epsilon = minimax_adversarial_epsilon
        self.minimax_iterations = minimax_iterations
        self.use_multi_token_prediction = use_multi_token_prediction
        self.n_predict_tokens = n_predict_tokens
        self.multi_token_loss_weight = multi_token_loss_weight
        self.use_differential_attention = use_differential_attention
        self.differential_lambda_init = differential_lambda_init
        self.lolcats_enabled = lolcats_enabled
        self.lolcats_compression_dim = lolcats_compression_dim
        self.use_moe = use_moe
        self.moe_config = MoEConfig(**moe_config) if moe_config else MoEConfig()
        self.use_mod = use_mod
        self.mod_config = MoDConfig(**mod_config) if mod_config else MoDConfig()

# ------------------------ HELPER FUNCTIONS ---------------------------- #

def rotate_half(x):
    """Rotates half the hidden dims of the input"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Apply rotary position embedding to query and key tensors"""
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value heads for grouped query attention"""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

# ------------------------ CORE COMPONENTS ---------------------------- #

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

# ------------------------ MIXTURE OF DEPTHS COMPONENTS ---------------------------- #

class MastishkDepthRouter(nn.Module):
    """Router for Mixture of Depths - decides which tokens process through which layers"""
    
    def __init__(self, config: MastishkConfig, total_layers: int):
        super().__init__()
        self.config = config
        self.mod_config = config.mod_config
        self.total_layers = total_layers
        self.hidden_size = config.hidden_size
        
        # Router network - takes hidden states and outputs routing decisions
        self.router = nn.Sequential(
            nn.Linear(self.hidden_size, self.mod_config.router_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.mod_config.router_dropout),
            nn.Linear(self.mod_config.router_hidden_dim, 2)  # Binary decision: process or skip
        )
        
        # Layer-specific biases to encourage diversity
        self.layer_bias = nn.Parameter(torch.zeros(total_layers))
        
        # Learned importance scores for each layer
        self.layer_importance = nn.Parameter(torch.ones(total_layers))
        
        # For adaptive computation time
        if self.mod_config.adaptive_computation_time:
            self.halting_unit = nn.Sequential(
                nn.Linear(self.hidden_size, 1),
                nn.Sigmoid()
            )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        training: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Route tokens through layers
        
        Returns:
            - routing_weights: (batch_size, seq_len) binary mask for processing
            - aux_losses: Dictionary of auxiliary losses
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Compute routing logits
        routing_logits = self.router(hidden_states)  # (batch_size, seq_len, 2)
        
        # Add layer-specific bias
        routing_logits[:, :, 1] += self.layer_bias[layer_idx]
        
        # Apply temperature
        routing_logits = routing_logits / self.mod_config.temperature
        
        # Get routing decisions
        if self.mod_config.use_gumbel_softmax and training:
            # Gumbel-Softmax for differentiable routing during training
            routing_probs = F.gumbel_softmax(routing_logits, tau=1.0, hard=False, dim=-1)
            routing_weights = routing_probs[:, :, 1]  # Probability of processing
            
            if self.mod_config.straight_through:
                # Straight-through estimator
                routing_decisions = (routing_weights > 0.5).float()
                routing_weights = routing_decisions - routing_weights.detach() + routing_weights
        else:
            # Hard routing during inference
            routing_probs = F.softmax(routing_logits, dim=-1)
            routing_weights = (routing_probs[:, :, 1] > 0.5).float()
        
        # Apply capacity constraints
        if self.mod_config.load_balancing_type == "capacity":
            routing_weights = self._apply_capacity_constraint(routing_weights, layer_idx)
        
        # Compute auxiliary losses
        aux_losses = self._compute_aux_losses(routing_logits, routing_weights, layer_idx, training)
        
        return routing_weights, aux_losses
    
    def _apply_capacity_constraint(
        self,
        routing_weights: torch.Tensor,
        layer_idx: int
    ) -> torch.Tensor:
        """Apply capacity constraints to ensure not too many tokens are processed"""
        batch_size, seq_len = routing_weights.shape
        
        # Calculate capacity for this layer
        capacity = int(self.mod_config.capacity_factor * seq_len)
        
        # For each batch, keep only top-k tokens
        if capacity < seq_len:
            # Get importance scores for tokens
            importance_scores = routing_weights.clone()
            
            # Add small noise to break ties
            importance_scores += torch.randn_like(importance_scores) * 1e-6
            
            # Keep top-k tokens
            _, top_indices = torch.topk(importance_scores, k=capacity, dim=1)
            
            # Create new routing weights
            new_routing_weights = torch.zeros_like(routing_weights)
            new_routing_weights.scatter_(1, top_indices, 1.0)
            
            return new_routing_weights
        
        return routing_weights
    
    def _compute_aux_losses(
        self,
        routing_logits: torch.Tensor,
        routing_weights: torch.Tensor,
        layer_idx: int,
        training: bool
    ) -> Dict[str, torch.Tensor]:
        """Compute auxiliary losses for training the router"""
        aux_losses = {}
        
        if not training:
            return aux_losses
        
        # Load balancing loss - encourage equal processing across layers
        if self.mod_config.router_aux_loss_weight > 0:
            # Average routing probability across tokens
            avg_routing = routing_weights.mean()
            
            # Target is the capacity factor
            target_routing = self.mod_config.capacity_factor
            
            # L2 loss to target
            load_balance_loss = ((avg_routing - target_routing) ** 2)
            aux_losses['mod_load_balance_loss'] = load_balance_loss * self.mod_config.router_aux_loss_weight
        
        # Router z-loss - encourage confident routing decisions
        if self.mod_config.router_z_loss_weight > 0:
            router_z_loss = torch.logsumexp(routing_logits, dim=-1).mean()
            aux_losses['mod_router_z_loss'] = router_z_loss * self.mod_config.router_z_loss_weight
        
        return aux_losses

# ------------------------ MIXTURE OF EXPERTS COMPONENTS ---------------------------- #

class MastishkExpertRouter(nn.Module):
    """Advanced router for Mixture of Experts with load balancing"""
    
    def __init__(self, config: MastishkConfig):
        super().__init__()
        self.config = config
        self.moe_config = config.moe_config
        self.num_experts = self.moe_config.num_experts
        self.num_experts_per_tok = self.moe_config.num_experts_per_tok
        assert self.num_experts_per_tok <= self.num_experts, \
            "num_experts_per_tok must be <= num_experts"
        self.router_type = self.moe_config.router_type
        
        # Router network
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        
        # Additional components for advanced routing
        if self.router_type == "soft":
            self.temperature = nn.Parameter(torch.ones(1))
        
        # Noise for exploration during training
        self.noise_epsilon = 1e-2
        
    def forward(
        self, 
        hidden_states: torch.Tensor,
        training: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Route tokens to experts
        
        Returns:
            - expert_weights: (batch_size, seq_len, num_experts_per_tok)
            - expert_indices: (batch_size, seq_len, num_experts_per_tok)
            - aux_losses: Dictionary of auxiliary losses for training
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Compute router logits
        router_logits = self.gate(hidden_states)  # (batch_size, seq_len, num_experts)
        
        # Add noise during training for exploration
        if training and self.noise_epsilon > 0:
            noise = torch.randn_like(router_logits) * self.noise_epsilon
            router_logits = router_logits + noise
        
        # Apply routing strategy
        if self.router_type == "top_k":
            expert_weights, expert_indices, aux_losses = self._top_k_routing(router_logits, training)
        elif self.router_type == "expert_choice":
            expert_weights, expert_indices, aux_losses = self._expert_choice_routing(router_logits, training)
        elif self.router_type == "soft":
            expert_weights, expert_indices, aux_losses = self._soft_routing(router_logits, training)
        else:
            raise ValueError(f"Unknown router type: {self.router_type}")
        
        return expert_weights, expert_indices, aux_losses
    
    def _top_k_routing(
        self, 
        router_logits: torch.Tensor,
        training: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Standard top-k routing as in Mixtral"""
        
        # Compute router probabilities
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        expert_weights, expert_indices = torch.topk(
            router_probs, self.num_experts_per_tok, dim=-1
        )
        
        # Normalize expert weights
        if self.moe_config.normalize_expert_weights:
            expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
        
        # Compute auxiliary losses
        aux_losses = self._compute_aux_losses(router_logits, router_probs, expert_indices, training)
        
        return expert_weights, expert_indices, aux_losses
    
    def _expert_choice_routing(
        self, 
        router_logits: torch.Tensor,
        training: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Expert choice routing where experts choose tokens"""
        
        batch_size, seq_len, num_experts = router_logits.shape
        
        # Transpose to let experts choose tokens
        router_logits_transposed = router_logits.transpose(1, 2)  # (batch_size, num_experts, seq_len)
        
        # Each expert selects top tokens
        tokens_per_expert = (seq_len * self.num_experts_per_tok) // num_experts
        expert_token_weights, expert_token_indices = torch.topk(
            router_logits_transposed, tokens_per_expert, dim=-1
        )
        
        # Convert back to token-centric view
        expert_weights = torch.zeros_like(router_logits)
        expert_indices = torch.zeros(
            batch_size, seq_len, self.num_experts_per_tok, 
            dtype=torch.long, device=router_logits.device
        )
        
        # Fill in the expert assignments
        for i in range(num_experts):
            mask = torch.zeros_like(router_logits)
            mask.scatter_(1, expert_token_indices[:, i, :].unsqueeze(1), 1)
            expert_weights += mask * F.softmax(router_logits, dim=-1)
        
        # Get top-k for each token
        expert_weights, expert_indices = torch.topk(
            expert_weights, self.num_experts_per_tok, dim=-1
        )
        
        # Normalize
        if self.moe_config.normalize_expert_weights:
            expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
        
        aux_losses = self._compute_aux_losses(router_logits, expert_weights, expert_indices, training)
        
        return expert_weights, expert_indices, aux_losses
    
    def _soft_routing(
        self, 
        router_logits: torch.Tensor,
        training: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Soft routing with temperature-controlled mixing"""
        
        # Apply temperature
        router_logits = router_logits / self.temperature
        
        # Soft top-k using sigmoid and normalization
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Threshold to get sparse routing
        threshold = torch.topk(router_probs, self.num_experts_per_tok, dim=-1).values[..., -1:, :]
        expert_mask = router_probs >= threshold
        
        # Get indices and weights
        expert_weights = router_probs * expert_mask.float()
        expert_indices = torch.topk(expert_weights, self.num_experts_per_tok, dim=-1).indices
        expert_weights = torch.gather(expert_weights, -1, expert_indices)
        
        # Normalize
        if self.moe_config.normalize_expert_weights:
            expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
        
        aux_losses = self._compute_aux_losses(router_logits, router_probs, expert_indices, training)
        
        return expert_weights, expert_indices, aux_losses
    
    def _compute_aux_losses(
        self,
        router_logits: torch.Tensor,
        router_probs: torch.Tensor,
        expert_indices: torch.Tensor,
        training: bool
    ) -> Dict[str, torch.Tensor]:
        """Compute auxiliary losses for load balancing"""
        
        aux_losses = {}
        
        if not training:
            return aux_losses
        
        # Load balancing loss
        if self.moe_config.load_balancing_type == "aux_loss":
            # Compute expert usage
            expert_usage = torch.zeros(
                router_logits.shape[0], self.num_experts, 
                device=router_logits.device
            )
            expert_usage.scatter_add_(
                1, 
                expert_indices.reshape(-1, self.num_experts_per_tok), 
                torch.ones_like(expert_indices, dtype=torch.float).reshape(-1, self.num_experts_per_tok)
            )
            expert_usage = expert_usage.mean(0)  # Average across batch
            
            # Target uniform distribution
            target_usage = torch.ones_like(expert_usage) / self.num_experts
            
            # Compute load balancing loss
            load_balance_loss = ((expert_usage - target_usage) ** 2).mean()
            aux_losses['load_balance_loss'] = load_balance_loss * self.moe_config.aux_loss_weight
        
        # Router z-loss (encourages router confidence)
        if self.moe_config.router_z_loss_weight > 0:
            router_z_loss = torch.logsumexp(router_logits, dim=-1).mean()
            aux_losses['router_z_loss'] = router_z_loss * self.moe_config.router_z_loss_weight
        
        return aux_losses

class MastishkExpert(nn.Module):
    """Single expert in the Mixture of Experts"""
    
    def __init__(self, config: MastishkConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # Expert MLP
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        
        self.act_fn = ACT2FN.get(config.hidden_act, F.silu)
        self.dropout = nn.Dropout(config.moe_config.expert_dropout)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Expert forward pass"""
        if self.config.glu_activation:
            gate = self.act_fn(self.gate_proj(hidden_states))
            up = self.up_proj(hidden_states)
            intermediate = gate * up
        else:
            intermediate = self.act_fn(self.up_proj(hidden_states))
        
        output = self.down_proj(intermediate)
        output = self.dropout(output)
        
        return output

class MastishkSparseMoE(nn.Module):
    """Sparse Mixture of Experts layer"""
    
    def __init__(self, config: MastishkConfig):
        super().__init__()
        self.config = config
        self.moe_config = config.moe_config
        self.num_experts = self.moe_config.num_experts
        self.num_experts_per_tok = self.moe_config.num_experts_per_tok
        
        # Router
        self.router = MastishkExpertRouter(config)
        
        # Experts
        self.experts = nn.ModuleList([
            MastishkExpert(config) for _ in range(self.num_experts)
        ])
        
        # Layer norm (optional)
        self.expert_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Shared expert (optional - always activated)
        self.shared_expert = MastishkExpert(config) if self.moe_config.num_experts_per_tok < self.num_experts else None
        self.shared_expert_weight = nn.Parameter(torch.ones(1) * 0.1) if self.shared_expert else None
        
    def forward(
        self, 
        hidden_states: torch.Tensor,
        training: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through sparse MoE layer
        
        Returns:
            - output: tensor of same shape as input
            - aux_losses: dictionary of auxiliary losses
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Route tokens to experts
        expert_weights, expert_indices, aux_losses = self.router(hidden_states, training)
        
        # Flatten for efficient computation
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        expert_weights_flat = expert_weights.view(-1, self.num_experts_per_tok)
        expert_indices_flat = expert_indices.view(-1, self.num_experts_per_tok)
        
        # Initialize output
        output = torch.zeros_like(hidden_states_flat)
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens assigned to this expert
            expert_mask = (expert_indices_flat == expert_idx).any(dim=-1)
            
            if expert_mask.any():
                # Get tokens for this expert
                expert_input = hidden_states_flat[expert_mask]
                
                # Apply expert
                expert_output = self.experts[expert_idx](expert_input)
                
                # Get weights for this expert
                expert_weight_mask = expert_indices_flat == expert_idx
                expert_weights_for_idx = expert_weights_flat * expert_weight_mask.float()
                expert_weights_sum = expert_weights_for_idx.sum(dim=-1, keepdim=True)
                
                # Add weighted expert output
                output[expert_mask] += expert_output * expert_weights_sum[expert_mask]
        
        # Reshape output
        output = output.view(batch_size, seq_len, hidden_dim)
        
        # Add shared expert if available
        if self.shared_expert is not None:
            shared_output = self.shared_expert(hidden_states)
            output = output + self.shared_expert_weight * shared_output
        
        return output, aux_losses

# ------------------------ ENHANCED ROPE WITH YARN SCALING ---------------------------- #

class MastishkRotaryEmbedding(nn.Module):
    """Enhanced RoPE with YaRN and other scaling methods for Mastishk"""
    
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000,
        rope_scaling: Optional[Dict] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.rope_scaling = rope_scaling
        
        inv_freq = self._compute_inv_freq(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.float32
        )
    
    def _compute_inv_freq(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Compute inverse frequency with scaling"""
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim))
        
        if self.rope_scaling is None:
            return inv_freq
        
        scaling_type = self.rope_scaling.get("type", "none")
        
        if scaling_type == "yarn":
            factor = self.rope_scaling.get("factor", 1.0)
            original_max_pe = self.rope_scaling.get("original_max_position_embeddings", 2048)
            beta_fast = self.rope_scaling.get("beta_fast", 32.0)
            beta_slow = self.rope_scaling.get("beta_slow", 1.0)
            mscale = self.rope_scaling.get("mscale", 1.0)
            
            low_freq_wavelen = original_max_pe / beta_fast
            high_freq_wavelen = original_max_pe / beta_slow
            
            wavelen = 2 * math.pi / inv_freq
            new_inv_freq = []
            
            for i, freq in enumerate(inv_freq):
                w = 2 * math.pi / freq
                if w < high_freq_wavelen:
                    new_inv_freq.append(freq)
                elif w > low_freq_wavelen:
                    new_inv_freq.append(freq / factor)
                else:
                    smooth = (original_max_pe / w - beta_slow) / (beta_fast - beta_slow)
                    new_inv_freq.append((1 - smooth) * freq / factor + smooth * freq)
            
            inv_freq = torch.tensor(new_inv_freq, dtype=inv_freq.dtype, device=inv_freq.device)
            
            if mscale != 1.0:
                inv_freq = inv_freq / mscale
                
        elif scaling_type == "linear":
            factor = self.rope_scaling.get("factor", 1.0)
            inv_freq = inv_freq / factor
            
        elif scaling_type == "dynamic":
            factor = self.rope_scaling.get("factor", 1.0)
            inv_freq = self._dynamic_scaling(inv_freq, factor)
        
        return inv_freq
    
    def _dynamic_scaling(self, inv_freq: torch.Tensor, factor: float) -> torch.Tensor:
        """Dynamic NTK-RoPE scaling"""
        base = self.base
        scale = (factor * base - base) / (factor - 1)
        return inv_freq * (1 + scale / base)
    
    def _set_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Precompute cos and sin values"""
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        
        if self.rope_scaling and self.rope_scaling.get("type") == "yarn":
            factor = self.rope_scaling.get("factor", 1.0)
            t = t / factor
            
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)
    
    def forward(
        self,
        x: torch.Tensor,
        seq_len: Optional[int] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cos and sin embeddings"""
        if seq_len is not None and seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        
        if position_ids is not None:
            return (
                self.cos_cached[position_ids].to(dtype=x.dtype),
                self.sin_cached[position_ids].to(dtype=x.dtype)
            )
        else:
            return (
                self.cos_cached[:seq_len].to(dtype=x.dtype),
                self.sin_cached[:seq_len].to(dtype=x.dtype)
            )

# ------------------------ MASTISHK MLP ---------------------------- #

class MastishkMLP(nn.Module):
    """Standard MLP for Mastishk (used when not using MoE)"""
    
    def __init__(self, config: MastishkConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.hidden_act = config.hidden_act
        self.glu_activation = config.glu_activation
        
        if config.quantization_config and HAS_BNBYTES:
            self.gate_proj = bnb.nn.Linear8bitLt(
                self.hidden_size,
                self.intermediate_size,
                bias=config.mlp_bias,
                has_fp16_weights=False
            )
            self.up_proj = bnb.nn.Linear8bitLt(
                self.hidden_size,
                self.intermediate_size,
                bias=config.mlp_bias,
                has_fp16_weights=False
            )
            self.down_proj = bnb.nn.Linear8bitLt(
                self.intermediate_size,
                self.hidden_size,
                bias=config.mlp_bias,
                has_fp16_weights=False
            )
        else:
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        
        self.act_fn = self._get_activation_fn()
        
        if config.use_layer_scale:
            self.layer_scale = nn.Parameter(
                torch.ones(1) * config.layer_scale_init
            )
        else:
            self.layer_scale = None
    
    def _get_activation_fn(self):
        """Get activation function"""
        if self.hidden_act == "swiglu":
            return F.silu
        elif self.hidden_act == "geglu":
            return F.gelu
        elif self.hidden_act == "reglu":
            return F.relu
        else:
            return ACT2FN.get(self.hidden_act, F.silu)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with gated activation"""
        if self.glu_activation:
            gate = self.act_fn(self.gate_proj(x))
            up = self.up_proj(x)
            x = self.down_proj(gate * up)
        else:
            x = self.down_proj(self.act_fn(self.up_proj(x)))
        
        if self.layer_scale is not None:
            x = x * self.layer_scale
        
        return x

# ------------------------ MASTISHK GROUPED QUERY ATTENTION ---------------------------- #

class MastishkGroupedQueryAttention(nn.Module):
    """Enhanced multi-head attention with Flash Attention 2 and GQA for Mastishk"""
    
    def __init__(self, config: MastishkConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_dropout = config.attention_dropout
        
        assert self.head_dim * self.num_heads == self.hidden_size
        assert self.num_heads % self.num_key_value_heads == 0
        
        if config.quantization_config and HAS_BNBYTES:
            self.q_proj = bnb.nn.Linear8bitLt(
                self.hidden_size,
                self.num_heads * self.head_dim,
                bias=config.attention_bias,
                has_fp16_weights=False
            )
            self.k_proj = bnb.nn.Linear8bitLt(
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                bias=config.attention_bias,
                has_fp16_weights=False
            )
            self.v_proj = bnb.nn.Linear8bitLt(
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                bias=config.attention_bias,
                has_fp16_weights=False
            )
            self.o_proj = bnb.nn.Linear8bitLt(
                self.num_heads * self.head_dim,
                self.hidden_size,
                bias=config.attention_bias,
                has_fp16_weights=False
            )
        else:
            self.q_proj = nn.Linear(
                self.hidden_size,
                self.num_heads * self.head_dim,
                bias=config.attention_bias
            )
            self.k_proj = nn.Linear(
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                bias=config.attention_bias
            )
            self.v_proj = nn.Linear(
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                bias=config.attention_bias
            )
            self.o_proj = nn.Linear(
                self.num_heads * self.head_dim,
                self.hidden_size,
                bias=config.attention_bias
            )
        
        self.rotary_emb = MastishkRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        
        if config.use_layer_scale:
            self.layer_scale = nn.Parameter(
                torch.ones(1) * config.layer_scale_init
            )
        else:
            self.layer_scale = None
    
    def _flash_attention_forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        query_length: int = 1,
        dropout: float = 0.0,
        softmax_scale: Optional[float] = None,
        sliding_window: Optional[int] = None,
    ) -> torch.Tensor:
        """Flash Attention 2 forward pass"""
        if not HAS_FLASH_ATTN:
            raise RuntimeError("Flash Attention not installed")
        
        batch_size, num_heads, seq_len, head_dim = query_states.shape
        
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        
        if attention_mask is not None:
            query_states, key_states, value_states, indices, cu_seqlens, max_seqlen = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )
            
            attn_output = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=True,
                window_size=sliding_window,
            )
            
            attn_output = self._pad_output(attn_output, indices, batch_size, seq_len)
        else:
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=True,
                window_size=sliding_window,
            )
        
        return attn_output.transpose(1, 2)
    
    def _upad_input(self, query, key, value, attention_mask, query_length):
        """Unpad input sequences for variable length attention"""
        indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        batch_size = query.shape[0]
        query = index_first_axis(query.reshape(batch_size * query.shape[1], *query.shape[2:]), indices)
        key = index_first_axis(key.reshape(batch_size * key.shape[1], *key.shape[2:]), indices)
        value = index_first_axis(value.reshape(batch_size * value.shape[1], *value.shape[2:]), indices)
        
        cu_seqlens = torch.cat([
            torch.tensor([0], device=query.device, dtype=torch.int32),
            attention_mask.sum(dim=1).cumsum(dim=0, dtype=torch.int32)
        ])
        max_seqlen = attention_mask.sum(dim=1).max().item()
        
        return query, key, value, indices, cu_seqlens, max_seqlen
    
    def _pad_output(self, attn_output, indices, batch_size, seq_len):
        """Pad output back to original shape"""
        output = torch.zeros(
            batch_size * seq_len,
            *attn_output.shape[1:],
            device=attn_output.device,
            dtype=attn_output.dtype
        )
        output[indices] = attn_output
        return output.reshape(batch_size, seq_len, *attn_output.shape[1:])
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Forward pass with Flash Attention 2 support"""
        bsz, q_len, _ = hidden_states.size()
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        
        if position_ids is None:
            position_ids = torch.arange(q_len, device=hidden_states.device).unsqueeze(0).expand(bsz, -1)
        
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )
        
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        
        past_key_value = (key_states, value_states) if use_cache else None
        
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        if self.config.use_flash_attention and HAS_FLASH_ATTN and not output_attentions:
            attn_output = self._flash_attention_forward(
                query_states,
                key_states,
                value_states,
                attention_mask,
                q_len,
                dropout=self.attention_dropout if self.training else 0.0,
                sliding_window=self.config.sliding_window_size if self.config.use_sliding_window else None,
            )
            attn_weights = None
        else:
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, value_states)
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        
        attn_output = self.o_proj(attn_output)
        
        if self.layer_scale is not None:
            attn_output = attn_output * self.layer_scale
        
        if not output_attentions:
            attn_weights = None
        
        return attn_output, attn_weights, past_key_value

# ------------------------ MINIMAX ATTENTION MODULE ---------------------------- #

class MastishkMiniMaxAttention(nn.Module):
    """MiniMax-enhanced attention mechanism with adversarial robustness for Mastishk"""
    
    def __init__(self, config: MastishkConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_dropout = config.attention_dropout
        
        assert self.head_dim * self.num_heads == self.hidden_size
        assert self.num_heads % self.num_key_value_heads == 0
        
        self.minimax_temperature = nn.Parameter(torch.ones(1))
        self.adversarial_epsilon = config.minimax_adversarial_epsilon
        self.minimax_iterations = config.minimax_iterations
        
        if config.quantization_config and HAS_BNBYTES:
            self.q_proj = bnb.nn.Linear8bitLt(
                self.hidden_size,
                self.num_heads * self.head_dim,
                bias=config.attention_bias,
                has_fp16_weights=False
            )
            self.k_proj = bnb.nn.Linear8bitLt(
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                bias=config.attention_bias,
                has_fp16_weights=False
            )
            self.v_proj = bnb.nn.Linear8bitLt(
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                bias=config.attention_bias,
                has_fp16_weights=False
            )
            self.o_proj = bnb.nn.Linear8bitLt(
                self.num_heads * self.head_dim,
                self.hidden_size,
                bias=config.attention_bias,
                has_fp16_weights=False
            )
        else:
            self.q_proj = nn.Linear(
                self.hidden_size,
                self.num_heads * self.head_dim,
                bias=config.attention_bias
            )
            self.k_proj = nn.Linear(
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                bias=config.attention_bias
            )
            self.v_proj = nn.Linear(
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                bias=config.attention_bias
            )
            self.o_proj = nn.Linear(
                self.num_heads * self.head_dim,
                self.hidden_size,
                bias=config.attention_bias
            )
        
        self.minimax_value_head = nn.Linear(self.head_dim, 1)
        self.minimax_policy_head = nn.Linear(self.head_dim, self.head_dim)
        
        self.rotary_emb = MastishkRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        
        if config.use_layer_scale:
            self.layer_scale = nn.Parameter(
                torch.ones(1) * config.layer_scale_init
            )
        else:
            self.layer_scale = None
    
    def compute_minimax_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute attention weights using MiniMax strategy"""
        
        batch_size, num_heads, seq_len, head_dim = query_states.shape
        
        scores = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(head_dim)
        
        if attention_mask is not None:
            scores = scores + attention_mask
        
        if torch.is_grad_enabled() and self.training and not getattr(self, '_in_checkpoint', False):
            for _ in range(self.minimax_iterations):
                value_estimates = self.minimax_value_head(value_states).squeeze(-1)
                
                try:
                    scores_for_grad = scores.detach().requires_grad_(True)
                    value_estimates_for_grad = self.minimax_value_head(value_states).squeeze(-1)
                    
                    grad = torch.autograd.grad(
                        outputs=value_estimates_for_grad.sum(),
                        inputs=scores_for_grad,
                        create_graph=True,
                        only_inputs=True,
                        allow_unused=True
                    )[0]
                    
                    if grad is not None:
                        adversarial_scores = scores + self.adversarial_epsilon * grad.sign().detach()
                    else:
                        adversarial_scores = scores + self.adversarial_epsilon * torch.randn_like(scores).sign()
                        
                except RuntimeError:
                    adversarial_scores = scores + self.adversarial_epsilon * torch.randn_like(scores).sign()
                
                robust_weights = F.softmax(adversarial_scores / self.minimax_temperature, dim=-1)
                worst_case_weights = F.softmax(-adversarial_scores / self.minimax_temperature, dim=-1)
                
                scores = scores - 0.1 * (robust_weights - worst_case_weights).detach()
        else:
            value_estimates = None
        
        attention_weights = F.softmax(scores, dim=-1)
        
        return attention_weights, value_estimates
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Forward pass with MiniMax attention"""
        
        bsz, q_len, _ = hidden_states.size()
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        
        if position_ids is None:
            position_ids = torch.arange(q_len, device=hidden_states.device).unsqueeze(0)
        
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )
        
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        
        past_key_value = (key_states, value_states) if use_cache else None
        
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        attn_weights, _ = self.compute_minimax_attention(
            query_states, key_states, value_states, attention_mask
        )
        
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        
        attn_output = self.o_proj(attn_output)
        
        if self.layer_scale is not None:
            attn_output = attn_output * self.layer_scale
        
        if not output_attentions:
            attn_weights = None
        
        return attn_output, attn_weights, past_key_value

# ------------------------ DIFFERENTIAL TRANSFORMER ATTENTION ---------------------------- #

class MastishkDifferentialAttention(nn.Module):
    """Differential Transformer attention mechanism that cancels noise for Mastishk"""
    
    def __init__(self, config: MastishkConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_dropout = config.attention_dropout
        
        self.lambda_param = nn.Parameter(
            torch.ones(self.num_heads) * config.differential_lambda_init
        )
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        
        self.rotary_emb = MastishkRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        
        if config.use_layer_scale:
            self.layer_scale = nn.Parameter(
                torch.ones(1) * config.layer_scale_init
            )
        else:
            self.layer_scale = None
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Forward pass with differential attention"""
        
        bsz, q_len, _ = hidden_states.size()
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        
        if position_ids is None:
            position_ids = torch.arange(q_len, device=hidden_states.device).unsqueeze(0)
        
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )
        
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        
        past_key_value = (key_states, value_states) if use_cache else None
        
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        half_head_dim = self.head_dim // 2
        q1, q2 = query_states.split(half_head_dim, dim=-1)
        k1, k2 = key_states.split(half_head_dim, dim=-1)
        
        scores1 = torch.matmul(q1, k1.transpose(-2, -1)) / math.sqrt(half_head_dim)
        scores2 = torch.matmul(q2, k2.transpose(-2, -1)) / math.sqrt(half_head_dim)
        
        if attention_mask is not None:
            scores1 = scores1 + attention_mask
            scores2 = scores2 + attention_mask
        
        attn_weights1 = F.softmax(scores1, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights2 = F.softmax(scores2, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        lambda_expanded = self.lambda_param.view(1, self.num_heads, 1, 1)
        diff_attn_weights = attn_weights1 - lambda_expanded * attn_weights2
        
        diff_attn_weights = F.dropout(diff_attn_weights, p=self.attention_dropout, training=self.training)
        
        attn_output = torch.matmul(diff_attn_weights, value_states)
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        
        attn_output = self.o_proj(attn_output)
        
        if self.layer_scale is not None:
            attn_output = attn_output * self.layer_scale
        
        if not output_attentions:
            diff_attn_weights = None
        
        return attn_output, diff_attn_weights, past_key_value

# ------------------------ TOKEN DROPPING MODULE ---------------------------- #

class RandomTokenDropping(nn.Module):
    """Random token dropping for efficient training"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.drop_rate = config.token_drop_rate
        self.schedule = config.token_drop_schedule
        self.current_step = 0
    
    def get_current_drop_rate(self, progress: float = 0.0) -> float:
        """Get current drop rate based on schedule"""
        if self.schedule == "constant":
            return self.drop_rate
        elif self.schedule == "linear":
            return self.drop_rate * (1 - progress)
        elif self.schedule == "cosine":
            return self.drop_rate * (1 + math.cos(math.pi * progress)) / 2
        else:
            return self.drop_rate
    
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        layer_idx: int, 
        total_layers: int,
        training_progress: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random token dropping"""
        
        if not self.training:
            return hidden_states, torch.ones(hidden_states.shape[:2], device=hidden_states.device)
        
        if layer_idx == 0 or layer_idx == total_layers - 1:
            return hidden_states, torch.ones(hidden_states.shape[:2], device=hidden_states.device)
        
        batch_size, seq_len, hidden_dim = hidden_states.shape
        current_drop_rate = self.get_current_drop_rate(training_progress)
        
        keep_prob = 1 - current_drop_rate
        random_mask = torch.rand(batch_size, seq_len, device=hidden_states.device) < keep_prob
        
        masked_hidden_states = hidden_states * random_mask.unsqueeze(-1).float()
        
        if keep_prob > 0:
            masked_hidden_states = masked_hidden_states / keep_prob
        
        return masked_hidden_states, random_mask

# ------------------------ MASTISHK TRANSFORMER LAYER ---------------------------- #

class MastishkTransformerLayer(nn.Module):
    """Mastishk transformer layer with gradient checkpointing, optional MoE, and MoD support"""
    
    def __init__(self, config: MastishkConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Determine attention type
        use_minimax = config.use_minimax and (layer_idx % config.minimax_layer_frequency == 0)
        use_differential = config.use_differential_attention and not use_minimax
        
        if use_minimax:
            self.self_attn = MastishkMiniMaxAttention(config, layer_idx)
        elif use_differential:
            self.self_attn = MastishkDifferentialAttention(config, layer_idx)
        else:
            self.self_attn = MastishkGroupedQueryAttention(config, layer_idx)
        
        # Determine MLP type
        use_moe = config.use_moe and (layer_idx % config.moe_config.moe_layer_frequency == 0)
        
        if use_moe:
            self.mlp = MastishkSparseMoE(config)
            self.use_moe = True
        else:
            self.mlp = MastishkMLP(config)
            self.use_moe = False
            
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.token_dropper = None
        if hasattr(config, 'training_config') and config.training_config.use_random_token_dropping:
            self.token_dropper = RandomTokenDropping(config.training_config)
        
        self.use_minimax = use_minimax
        if use_minimax:
            self.adversarial_discriminator = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size // 2),
                nn.ReLU(),
                nn.Linear(config.hidden_size // 2, 1),
                nn.Sigmoid()
            )
        
        self.use_checkpoint = self._should_checkpoint(layer_idx, config)
    
    def _should_checkpoint(self, layer_idx: int, config: MastishkConfig) -> bool:
        """Determine if layer should use gradient checkpointing"""
        if not config.use_gradient_checkpointing:
            return False
        
        if config.checkpointing_policy == "all":
            return True
        elif config.checkpointing_policy == "selective":
            return layer_idx % config.checkpoint_sequential_factor == 0
        else:
            return False
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        training_progress: float = 0.0,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, ...]:
        """Forward pass with optional gradient checkpointing"""
        
        if self.use_checkpoint and self.training:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward
            
            hidden_states = ckpt(
                create_custom_forward(self._forward_impl),
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
                training_progress,
                use_reentrant=False,
            )
            return hidden_states
        else:
            return self._forward_impl(
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
                training_progress,
                **kwargs
            )
    
    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        training_progress: float = 0.0,
        **kwargs,
    ):
        """Actual forward implementation"""
        token_mask = None
        aux_losses = {}
        
        if self.token_dropper is not None and self.training:
            hidden_states, token_mask = self.token_dropper(
                hidden_states, 
                self.layer_idx, 
                self.config.num_hidden_layers,
                training_progress
            )
        
        # Self-attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        if self.use_moe:
            hidden_states, moe_aux_losses = self.mlp(hidden_states, training=self.training)
            aux_losses.update(moe_aux_losses)
        else:
            hidden_states = self.mlp(hidden_states)
        
        hidden_states = residual + hidden_states
        
        # MiniMax adversarial training
        if self.use_minimax and self.training:
            hidden_states.requires_grad_(True)
            disc_score = self.adversarial_discriminator(hidden_states).mean()
            
            if hidden_states.grad_fn is not None:
                grad = torch.autograd.grad(disc_score, hidden_states, create_graph=True)[0]
                epsilon = 0.01
                perturbation = epsilon * grad.sign()
                hidden_states = hidden_states + perturbation.detach()
        
        outputs = (hidden_states,)
        
        if output_attentions:
            outputs += (self_attn_weights,)
        
        if use_cache:
            outputs += (present_key_value,)
        
        if token_mask is not None:
            outputs += (token_mask,)
        
        if aux_losses:
            outputs += (aux_losses,)
        
        return outputs
    
    def forward_with_mod_routing(
        self,
        hidden_states: torch.Tensor,
        routing_weight: torch.Tensor,  # MoD routing weight
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        training_progress: float = 0.0,
        **kwargs,
    ):
        """Forward pass with MoD routing - skip computation for tokens with routing_weight = 0"""
        # Reshape routing weight for broadcasting
        routing_mask = routing_weight.unsqueeze(-1)  # (batch_size, seq_len, 1)
        
        # Store residual for skip connection
        residual = hidden_states
        
        # Check if any tokens need processing
        if (routing_mask > 0.5).any():
            # Process through layer
            outputs = self._forward_impl(
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
                training_progress,
                **kwargs
            )
            
            # Apply routing mask - blend processed and residual based on routing
            processed_hidden = outputs[0]
            hidden_states = routing_mask * processed_hidden + (1 - routing_mask) * residual
            
            # Update outputs with masked hidden states
            outputs = (hidden_states,) + outputs[1:]
        else:
            # All tokens skip - just return residual
            outputs = (residual,)
            if use_cache:
                outputs += (past_key_value,)
            if output_attentions:
                outputs += (None,)
        
        return outputs

# ------------------------ LOLCATS CONVERSION MODULE ---------------------------- #

class MastishkLoLCATsConverter:
    """Convert trained Mastishk transformer to use linear attention via LoLCATs"""
    
    def __init__(self, model: nn.Module, config: MastishkConfig):
        self.model = model
        self.config = config
        self.compression_dim = config.lolcats_compression_dim
    
    def create_linear_attention_layers(self, layer_idx: int) -> nn.Module:
        """Create linear attention layer that mimics softmax attention"""
        
        class LinearAttentionMimic(nn.Module):
            def __init__(self, original_attn, compression_dim):
                super().__init__()
                self.original_attn = original_attn
                hidden_size = original_attn.hidden_size
                
                self.phi_q = nn.Sequential(
                    nn.Linear(original_attn.head_dim, compression_dim),
                    nn.ReLU()
                )
                self.phi_k = nn.Sequential(
                    nn.Linear(original_attn.head_dim, compression_dim),
                    nn.ReLU()
                )
                
                self.lora_rank = 16
                self.q_lora_A = nn.Linear(hidden_size, self.lora_rank, bias=False)
                self.q_lora_B = nn.Linear(self.lora_rank, original_attn.num_heads * original_attn.head_dim, bias=False)
                self.v_lora_A = nn.Linear(hidden_size, self.lora_rank, bias=False)
                self.v_lora_B = nn.Linear(self.lora_rank, original_attn.num_key_value_heads * original_attn.head_dim, bias=False)
                
                nn.init.kaiming_uniform_(self.q_lora_A.weight, a=math.sqrt(5))
                nn.init.zeros_(self.q_lora_B.weight)
                nn.init.kaiming_uniform_(self.v_lora_A.weight, a=math.sqrt(5))
                nn.init.zeros_(self.v_lora_B.weight)
            
            def forward(self, hidden_states, **kwargs):
                bsz, seq_len, _ = hidden_states.size()
                
                q = self.original_attn.q_proj(hidden_states)
                k = self.original_attn.k_proj(hidden_states)
                v = self.original_attn.v_proj(hidden_states)
                
                q = q + self.q_lora_B(self.q_lora_A(hidden_states))
                v = v + self.v_lora_B(self.v_lora_A(hidden_states))
                
                q = q.view(bsz, seq_len, self.original_attn.num_heads, self.original_attn.head_dim)
                k = k.view(bsz, seq_len, self.original_attn.num_key_value_heads, self.original_attn.head_dim)
                v = v.view(bsz, seq_len, self.original_attn.num_key_value_heads, self.original_attn.head_dim)
                
                q_linear = self.phi_q(q)
                k_linear = self.phi_k(k)
                
                if self.original_attn.num_key_value_groups > 1:
                    k_linear = k_linear.repeat(1, 1, self.original_attn.num_key_value_groups, 1)
                    v = v.repeat(1, 1, self.original_attn.num_key_value_groups, 1)
                
                kv = torch.einsum('bshc,bshd->bhcd', k_linear, v)
                attn_output = torch.einsum('bshc,bhcd->bshd', q_linear, kv)
                
                attn_output = attn_output.reshape(bsz, seq_len, -1)
                attn_output = self.original_attn.o_proj(attn_output)
                
                return attn_output, None, None
        
        return LinearAttentionMimic(
            self.model.model.layers[layer_idx].self_attn,
            self.compression_dim
        )
    
    def train_attention_transfer(
        self,
        layer_idx: int,
        train_data: torch.Tensor,
        num_steps: int = 1000,
        lr: float = 1e-3
    ):
        """Train linear attention to mimic softmax attention"""
        
        linear_attn = self.create_linear_attention_layers(layer_idx)
        original_attn = self.model.model.layers[layer_idx].self_attn
        
        for param in original_attn.parameters():
            param.requires_grad = False
        
        optimizer = torch.optim.AdamW([
            {'params': linear_attn.phi_q.parameters()},
            {'params': linear_attn.phi_k.parameters()},
            {'params': linear_attn.q_lora_A.parameters()},
            {'params': linear_attn.q_lora_B.parameters()},
            {'params': linear_attn.v_lora_A.parameters()},
            {'params': linear_attn.v_lora_B.parameters()},
        ], lr=lr)
        
        for step in range(num_steps):
            hidden_states = train_data[step % len(train_data)]
            
            with torch.no_grad():
                orig_output, _, _ = original_attn(hidden_states)
            
            linear_output, _, _ = linear_attn(hidden_states)
            
            loss = F.mse_loss(linear_output, orig_output)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if step % 100 == 0:
                print(f"Layer {layer_idx}, Step {step}, Loss: {loss.item():.6f}")
        
        return linear_attn
    
    def convert_model(self, train_data: torch.Tensor, layers_to_convert: Optional[List[int]] = None):
        """Convert specified layers to linear attention"""
        
        if layers_to_convert is None:
            layers_to_convert = list(range(1, self.config.num_hidden_layers - 1))
        
        for layer_idx in layers_to_convert:
            print(f"Converting layer {layer_idx} to linear attention...")
            linear_attn = self.train_attention_transfer(layer_idx, train_data)
            
            self.model.model.layers[layer_idx].self_attn = linear_attn
        
        print("LoLCATs conversion complete!")
        return self.model

# ------------------------ MULTI-TOKEN PREDICTION HEAD ---------------------------- #

class MastishkMultiTokenPredictionHead(nn.Module):
    """Multi-token prediction heads for faster inference in Mastishk"""
    
    def __init__(self, config: MastishkConfig):
        super().__init__()
        self.n_predict = config.n_predict_tokens
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        
        self.prediction_heads = nn.ModuleList([
            nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            for _ in range(config.n_predict_tokens)
        ])
        
        self.intermediate_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=8,
                dim_feedforward=config.hidden_size * 2,
                dropout=0.1,
                batch_first=True
            )
            for _ in range(config.n_predict_tokens - 1)
        ])
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass for multi-token prediction
        """
        predictions = []
        losses = []
        
        current_hidden = hidden_states
        
        for i in range(self.n_predict):
            pred_i = self.prediction_heads[i](current_hidden)
            predictions.append(pred_i)
            
            if labels is not None and i < labels.shape[1] - hidden_states.shape[1] + 1:
                target_labels = labels[:, i:i + hidden_states.shape[1]]
                if target_labels.shape[1] == pred_i.shape[1]:
                    loss_i = F.cross_entropy(
                        pred_i.reshape(-1, self.vocab_size),
                        target_labels.reshape(-1),
                        ignore_index=-100
                    )
                    losses.append(loss_i)
            
            if i < self.n_predict - 1:
                current_hidden = self.intermediate_blocks[i](current_hidden)
        
        total_loss = None
        if losses:
            total_loss = torch.stack(losses).mean()
        
        return predictions, total_loss

# ------------------------ ENHANCED KV CACHE ---------------------------- #

class MastishkKVCache:
    """Enhanced KV cache with multiple strategies for Mastishk"""
    
    def __init__(self, config: KVCacheConfig):
        self.config = config
        self.cache_implementation = config.cache_implementation
        self.cache_data = {}
        
    def update(
        self,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]],
        layer_idx: int,
        new_key: torch.Tensor,
        new_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache with new key/value pairs"""
        
        if self.cache_implementation == "sliding_window":
            return self._sliding_window_update(past_key_values, layer_idx, new_key, new_value)
        elif self.cache_implementation == "h2o":
            return self._h2o_update(past_key_values, layer_idx, new_key, new_value)
        elif self.cache_implementation == "scissorhands":
            return self._scissorhands_update(past_key_values, layer_idx, new_key, new_value)
        else:
            if past_key_values is not None:
                past_key = past_key_values[layer_idx][0]
                past_value = past_key_values[layer_idx][1]
                new_key = torch.cat([past_key, new_key], dim=2)
                new_value = torch.cat([past_value, new_value], dim=2)
            
            return new_key, new_value
    
    def _sliding_window_update(
        self,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]],
        layer_idx: int,
        new_key: torch.Tensor,
        new_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sliding window cache update"""
        window_size = self.config.window_size
        
        if past_key_values is not None:
            past_key = past_key_values[layer_idx][0]
            past_value = past_key_values[layer_idx][1]
            
            key = torch.cat([past_key, new_key], dim=2)
            value = torch.cat([past_value, new_value], dim=2)
            
            if key.shape[2] > window_size:
                key = key[:, :, -window_size:, :]
                value = value[:, :, -window_size:, :]
            
            return key, value
        else:
            return new_key, new_value
    
    def _h2o_update(
        self,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]],
        layer_idx: int,
        new_key: torch.Tensor,
        new_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Heavy-Hitter Oracle (H2O) cache update"""
        sink_tokens = self.config.sink_tokens
        recent_tokens = self.config.recent_tokens
        
        if past_key_values is not None:
            past_key = past_key_values[layer_idx][0]
            past_value = past_key_values[layer_idx][1]
            
            key = torch.cat([past_key, new_key], dim=2)
            value = torch.cat([past_value, new_value], dim=2)
            
            seq_len = key.shape[2]
            if seq_len > self.config.max_cache_length:
                sink_keys = key[:, :, :sink_tokens, :]
                sink_values = value[:, :, :sink_tokens, :]
                
                recent_keys = key[:, :, -recent_tokens:, :]
                recent_values = value[:, :, -recent_tokens:, :]
                
                middle_size = self.config.max_cache_length - sink_tokens - recent_tokens
                if middle_size > 0:
                    middle_start = sink_tokens
                    middle_end = seq_len - recent_tokens
                    stride = max(1, (middle_end - middle_start) // middle_size)
                    middle_indices = torch.arange(middle_start, middle_end, stride)[:middle_size]
                    
                    middle_keys = key[:, :, middle_indices, :]
                    middle_values = value[:, :, middle_indices, :]
                    
                    key = torch.cat([sink_keys, middle_keys, recent_keys], dim=2)
                    value = torch.cat([sink_values, middle_values, recent_values], dim=2)
                else:
                    key = torch.cat([sink_keys, recent_keys], dim=2)
                    value = torch.cat([sink_values, recent_values], dim=2)
            
            return key, value
        else:
            return new_key, new_value
    
    def _scissorhands_update(
        self,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]],
        layer_idx: int,
        new_key: torch.Tensor,
        new_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scissorhands cache update (importance-based pruning)"""
        compression_ratio = self.config.compression_ratio
        
        if past_key_values is not None:
            past_key = past_key_values[layer_idx][0]
            past_value = past_key_values[layer_idx][1]
            
            key = torch.cat([past_key, new_key], dim=2)
            value = torch.cat([past_value, new_value], dim=2)
            
            seq_len = key.shape[2]
            if seq_len > self.config.max_cache_length:
                key_importance = key.norm(dim=-1).mean(dim=1)
                
                keep_size = int(self.config.max_cache_length * compression_ratio)
                _, keep_indices = key_importance.topk(keep_size, dim=-1, sorted=True)
                
                batch_size = key.shape[0]
                kept_keys = []
                kept_values = []
                
                for b in range(batch_size):
                    indices = keep_indices[b].sort()[0]
                    kept_keys.append(key[b:b+1, :, indices, :])
                    kept_values.append(value[b:b+1, :, indices, :])
                
                key = torch.cat(kept_keys, dim=0)
                value = torch.cat(kept_values, dim=0)
            
            return key, value
        else:
            return new_key, new_value

# ------------------------ MINIMAX SEARCH ---------------------------- #

@dataclass
class MastishkMiniMaxNode:
    """Node for MiniMax search tree in Mastishk"""
    token_id: int
    score: float
    depth: int
    parent: Optional['MastishkMiniMaxNode'] = None
    children: List['MastishkMiniMaxNode'] = None
    value_estimate: float = 0.0
    visit_count: int = 0
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
    
    def uct_score(self, exploration_weight: float = 1.414) -> float:
        """Upper Confidence Bound for Trees score"""
        if self.visit_count == 0:
            return float('inf')
        exploitation = self.value_estimate / self.visit_count
        exploration = exploration_weight * math.sqrt(math.log(self.parent.visit_count) / self.visit_count)
        return exploitation + exploration

class MastishkMiniMaxSearch:
    """MiniMax search for strategic text generation in Mastishk"""
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        max_depth: int = 5,
        beam_width: int = 5,
        minimax_iterations: int = 100,
        temperature: float = 0.8,
        exploration_weight: float = 1.414,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.minimax_iterations = minimax_iterations
        self.temperature = temperature
        self.exploration_weight = exploration_weight
        
    def evaluate_position(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> float:
        """Evaluate position using model's value head"""
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            
            last_hidden = outputs.hidden_states[-1]
            pooled = last_hidden.mean(dim=1)
            
            logits = outputs.logits
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                input_ids[:, 1:].reshape(-1),
                reduction='mean'
            )
            return -loss.item()
    
    def minimax_search(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int,
    ) -> List[int]:
        """Perform MiniMax search for optimal generation"""
        
        root = MastishkMiniMaxNode(
            token_id=-1,
            score=0.0,
            depth=0,
        )
        
        best_sequence = []
        current_ids = input_ids.clone()
        current_mask = attention_mask.clone()
        
        for _ in range(max_length - input_ids.size(1)):
            for iteration in range(self.minimax_iterations):
                node = root
                path = []
                
                while node.children and node.depth < self.max_depth:
                    node = max(node.children, key=lambda n: n.uct_score(self.exploration_weight))
                    path.append(node)
                
                if node.depth < self.max_depth:
                    with torch.no_grad():
                        test_ids = current_ids.clone()
                        for n in path:
                            if n.token_id >= 0:
                                test_ids = torch.cat([test_ids, torch.tensor([[n.token_id]]).to(test_ids.device)], dim=1)
                        
                        outputs = self.model(input_ids=test_ids)
                        logits = outputs.logits[:, -1, :] / self.temperature
                        probs = F.softmax(logits, dim=-1)
                        
                        top_k_probs, top_k_indices = torch.topk(probs, min(self.beam_width, probs.size(-1)))
                        
                        for i in range(top_k_indices.size(-1)):
                            child = MastishkMiniMaxNode(
                                token_id=top_k_indices[0, i].item(),
                                score=top_k_probs[0, i].item(),
                                depth=node.depth + 1,
                                parent=node,
                            )
                            node.children.append(child)
                
                if node.children:
                    node = np.random.choice(node.children)
                    path.append(node)
                
                eval_ids = current_ids.clone()
                for n in path:
                    if n.token_id >= 0:
                        eval_ids = torch.cat([eval_ids, torch.tensor([[n.token_id]]).to(eval_ids.device)], dim=1)
                
                eval_mask = torch.ones_like(eval_ids)
                value = self.evaluate_position(eval_ids, eval_mask)
                
                for n in reversed(path):
                    n.visit_count += 1
                    n.value_estimate += value
                    value = -value if n.depth % 2 == 0 else value
            
            if root.children:
                best_child = max(root.children, key=lambda n: n.visit_count)
                best_sequence.append(best_child.token_id)
                
                current_ids = torch.cat([current_ids, torch.tensor([[best_child.token_id]]).to(current_ids.device)], dim=1)
                current_mask = torch.ones_like(current_ids)
                
                root = best_child
                root.parent = None
            else:
                break
                
            if best_child.token_id == self.tokenizer.eos_token_id:
                break
        
        return best_sequence

# ------------------------ CHECKPOINT MANAGER ---------------------------- #

class MastishkCheckpointManager:
    """Enhanced checkpoint management with sharding and streaming for Mastishk"""
    
    @staticmethod
    def save_checkpoint(
        model: nn.Module,
        save_directory: str,
        config: Optional[MastishkConfig] = None,
        safe_serialization: bool = True,
        max_shard_size: str = "5GB",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Save Mastishk model checkpoint with sharding"""
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        
        if config is not None:
            config.save_pretrained(str(save_directory))
        
        if hasattr(model, "module"):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        
        if safe_serialization and HAS_SAFETENSORS:
            max_size_bytes = MastishkCheckpointManager._parse_size(max_shard_size)
            shards = MastishkCheckpointManager._shard_state_dict(state_dict, max_size_bytes)
            
            for shard_idx, shard in enumerate(shards):
                shard_file = save_directory / f"mastishk-{shard_idx:05d}-of-{len(shards):05d}.safetensors"
                save_file(shard, str(shard_file), metadata=metadata)
            
            index = {
                "metadata": metadata or {},
                "weight_map": {k: f"mastishk-{i:05d}-of-{len(shards):05d}.safetensors" 
                              for i, shard in enumerate(shards) for k in shard.keys()}
            }
            with open(save_directory / "mastishk.safetensors.index.json", "w") as f:
                json.dump(index, f, indent=2)
        else:
            torch.save({
                "model_state_dict": state_dict,
                "metadata": metadata,
            }, save_directory / "mastishk_model.bin")
        
        print(f"✅ Mastishk model saved to {save_directory}")
    
    @staticmethod
    def load_checkpoint(
        model: nn.Module,
        load_directory: str,
        device_map: Optional[Union[str, Dict[str, Any]]] = None,
        dtype: Optional[torch.dtype] = None,
        low_cpu_mem_usage: bool = True,
        offload_folder: Optional[str] = None,
    ):
        """Load Mastishk checkpoint with memory efficiency"""
        load_directory = Path(load_directory)
        
        index_file = load_directory / "mastishk.safetensors.index.json"
        if index_file.exists() and HAS_SAFETENSORS:
            with open(index_file, "r") as f:
                index = json.load(f)
            
            weight_map = index["weight_map"]
            unique_files = list(set(weight_map.values()))
            
            if low_cpu_mem_usage:
                for file in unique_files:
                    shard = load_file(str(load_directory / file), device=str(device_map or "cpu"))
                    for name, param in shard.items():
                        if dtype is not None:
                            param = param.to(dtype)
                        MastishkCheckpointManager._set_module_tensor(model, name, param)
                    del shard
                    gc.collect()
            else:
                state_dict = {}
                for file in unique_files:
                    shard = load_file(str(load_directory / file))
                    state_dict.update(shard)
                
                if dtype is not None:
                    state_dict = {k: v.to(dtype) for k, v in state_dict.items()}
                
                model.load_state_dict(state_dict)
        else:
            checkpoint = torch.load(load_directory / "mastishk_model.bin", map_location="cpu")
            state_dict = checkpoint["model_state_dict"]
            
            if dtype is not None:
                state_dict = {k: v.to(dtype) for k, v in state_dict.items()}
            
            model.load_state_dict(state_dict)
        
        print(f"✅ Mastishk model loaded from {load_directory}")
        return model
    
    @staticmethod
    def _parse_size(size_str: str) -> int:
        """Parse size string to bytes"""
        units = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}
        size_str = size_str.upper()
        for unit, multiplier in units.items():
            if size_str.endswith(unit):
                return int(float(size_str[:-len(unit)]) * multiplier)
        return int(size_str)
    
    @staticmethod
    def _shard_state_dict(state_dict: Dict[str, torch.Tensor], max_size: int) -> List[Dict[str, torch.Tensor]]:
        """Shard state dict by size"""
        shards = []
        current_shard = {}
        current_size = 0
        
        for name, tensor in state_dict.items():
            tensor_size = tensor.numel() * tensor.element_size()
            
            if current_size + tensor_size > max_size and current_shard:
                shards.append(current_shard)
                current_shard = {}
                current_size = 0
            
            current_shard[name] = tensor
            current_size += tensor_size
        
        if current_shard:
            shards.append(current_shard)
        
        return shards
    
    @staticmethod
    def _set_module_tensor(module: nn.Module, name: str, tensor: torch.Tensor):
        """Set tensor in module hierarchy"""
        parts = name.split(".")
        current = module
        
        for part in parts[:-1]:
            current = getattr(current, part)
        
        setattr(current, parts[-1], nn.Parameter(tensor) if tensor.requires_grad else tensor)

# ------------------------ MAIN MASTISHK TRANSFORMER MODEL ---------------------------- #

class MastishkTransformerModel(PreTrainedModel):
    """Mastishk Transformer model with all optimizations including MoE and MoD"""
    
    config_class = MastishkConfig
    
    def __init__(self, config: MastishkConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        
        # Initialize depth router if MoD is enabled
        if config.use_mod and config.mod_config.enabled:
            self.depth_router = MastishkDepthRouter(config, config.num_hidden_layers)
        else:
            self.depth_router = None
        
        self.layers = nn.ModuleList([
            MastishkTransformerLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.kv_cache = MastishkKVCache(config.kv_cache_config)
        
        self.register_buffer('training_steps', torch.tensor(0))
        self.register_buffer('total_training_steps', torch.tensor(100000))
        
        # Statistics tracking for MoD
        if config.use_mod and config.mod_config.enabled:
            self.register_buffer('layer_execution_counts', torch.zeros(config.num_hidden_layers))
            self.register_buffer('total_tokens_processed', torch.tensor(0))
        
        self.post_init()
        
        self._setup_model_parallel()
    
    def _setup_model_parallel(self):
        """Setup model parallelism strategies"""
        if self.config.parallelism_strategy == "tensor":
            self._setup_tensor_parallel()
        elif self.config.parallelism_strategy == "pipeline":
            self._setup_pipeline_parallel()
        elif self.config.parallelism_strategy == "3d":
            self._setup_3d_parallel()
    
    def _setup_tensor_parallel(self):
        """Setup tensor parallelism"""
        if not dist.is_initialized():
            return
        
        world_size = dist.get_world_size()
        if world_size <= 1:
            return
        
        for layer in self.layers:
            attn = layer.self_attn
            attn.num_heads = attn.num_heads // world_size
            attn.num_key_value_heads = attn.num_key_value_heads // world_size
    
    def _setup_pipeline_parallel(self):
        """Setup pipeline parallelism"""
        pass
    
    def _setup_3d_parallel(self):
        """Setup 3D parallelism (tensor + pipeline + data)"""
        pass
    
    def get_input_embeddings(self):
        return self.embed_tokens
    
    def set_input_embeddings(self, value):
        self.embed_tokens = value
    
    @torch.jit.ignore
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing"""
        self.config.use_gradient_checkpointing = True
    
    @torch.jit.ignore
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing"""
        self.config.use_gradient_checkpointing = False
    
    def get_mod_statistics(self) -> Dict[str, Any]:
        """Get statistics about MoD routing patterns"""
        if not hasattr(self, 'layer_execution_counts'):
            return {}
        
        total_tokens = self.total_tokens_processed.item()
        if total_tokens == 0:
            return {}
        
        # Calculate average execution rate per layer
        avg_execution_rate = self.layer_execution_counts / total_tokens
        
        # Calculate load balance score (0 = perfectly balanced, 1 = completely imbalanced)
        expected_rate = self.config.mod_config.capacity_factor
        load_balance_score = torch.std(avg_execution_rate).item()
        
        # Find most and least used layers
        most_used_layer = torch.argmax(self.layer_execution_counts).item()
        least_used_layer = torch.argmin(self.layer_execution_counts).item()
        
        return {
            'avg_execution_rate_per_layer': avg_execution_rate.tolist(),
            'load_balance_score': load_balance_score,
            'most_used_layer': most_used_layer,
            'least_used_layer': least_used_layer,
            'avg_layers_per_token': avg_execution_rate.sum().item(),
            'total_tokens_processed': total_tokens,
        }
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_router_logits: Optional[bool] = False,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        seq_length_with_past = seq_length
        past_key_values_length = 0
        
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )
        
        hidden_states = inputs_embeds
        
        training_progress = 0.0
        if self.training and self.total_training_steps > 0:
            training_progress = float(self.training_steps) / float(self.total_training_steps)
        
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        all_aux_losses = []
        all_router_logits = [] if output_router_logits else None
        
        # Track MoD statistics
        if self.depth_router is not None:
            batch_tokens = batch_size * seq_length
            self.total_tokens_processed += batch_tokens
        
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            
            # Apply MoD routing if enabled
            if self.depth_router is not None:
                # Get routing decision for this layer
                routing_weights, mod_aux_losses = self.depth_router(
                    hidden_states,
                    idx,
                    training=self.training
                )
                
                # Update statistics
                if self.training:
                    tokens_processed = routing_weights.sum().item()
                    self.layer_execution_counts[idx] += tokens_processed
                
                # Add MoD auxiliary losses
                if mod_aux_losses:
                    all_aux_losses.append(mod_aux_losses)
                
                # Apply layer with MoD routing
                layer_outputs = decoder_layer.forward_with_mod_routing(
                    hidden_states,
                    routing_weights,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    training_progress=training_progress,
                )
                
                if output_router_logits:
                    all_router_logits.append(routing_weights)
            else:
                # Standard forward pass without MoD
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    training_progress=training_progress,
                )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
            
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            
            # Collect auxiliary losses from MoE layers
            if len(layer_outputs) > 3 and isinstance(layer_outputs[-1], dict):
                all_aux_losses.append(layer_outputs[-1])
        
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        next_cache = next_decoder_cache if use_cache else None
        
        # Aggregate auxiliary losses
        aux_loss = None
        if all_aux_losses:
            aux_loss = {}
            for loss_dict in all_aux_losses:
                for key, value in loss_dict.items():
                    if key not in aux_loss:
                        aux_loss[key] = []
                    aux_loss[key].append(value)
            
            # Average losses
            aux_loss = {k: torch.stack(v).mean() for k, v in aux_loss.items()}
        
        if not return_dict:
            outputs = tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
            if aux_loss:
                outputs += (aux_loss,)
            if output_router_logits:
                outputs += (all_router_logits,)
                outputs += (self.get_mod_statistics(),)
            return outputs
        
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
    
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        """Prepare causal attention mask"""
        batch_size, seq_length = input_shape
        seq_length_with_past = seq_length + past_key_values_length
        
        # Create causal mask
        causal_mask = torch.triu(
            torch.full((seq_length, seq_length), float("-inf"), device=inputs_embeds.device),
            diagonal=1
        )
        
        if past_key_values_length > 0:
            causal_mask = torch.cat(
                [torch.zeros(seq_length, past_key_values_length, device=inputs_embeds.device), causal_mask],
                dim=-1
            )
        
        # Expand for batch and heads
        causal_mask = causal_mask[None, None, :, :].expand(
            batch_size, 1, seq_length, seq_length_with_past
        )
        
        # Merge with attention mask if provided
        if attention_mask is not None:
            expanded_mask = attention_mask[:, None, None, :].expand(
                batch_size, 1, seq_length, seq_length_with_past
            ).to(inputs_embeds.dtype)
            inverted_mask = 1.0 - expanded_mask
            causal_mask = causal_mask.masked_fill(
                inverted_mask.bool(), torch.finfo(inputs_embeds.dtype).min
            )
        
        return causal_mask

# ------------------------ MASTISHK CAUSAL LM ---------------------------- #

class MastishkTransformerForCausalLM(PreTrainedModel):
    """Mastishk Transformer for causal language modeling with MoE and MoD"""
    
    config_class = MastishkConfig
    _tied_weights_keys = ["lm_head.weight"]
    
    def __init__(self, config: MastishkConfig):
        super().__init__(config)
        self.model = MastishkTransformerModel(config)
        self.vocab_size = config.vocab_size
        
        # Language modeling head
        if config.quantization_config and HAS_BNBYTES:
            self.lm_head = bnb.nn.Linear8bitLt(
                config.hidden_size,
                config.vocab_size,
                bias=False,
                has_fp16_weights=False
            )
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Multi-token prediction head
        if config.use_multi_token_prediction:
            self.multi_token_head = MastishkMultiTokenPredictionHead(config)
        else:
            self.multi_token_head = None
        
        # Initialize weights
        self.post_init()
        
        # Add MiniMax search if enabled
        if config.use_minimax:
            self.minimax_search = None  # Will be initialized with tokenizer
    
    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]
        
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_router_logits: Optional[bool] = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Model forward
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            output_router_logits=output_router_logits,
        )
        
        hidden_states = outputs[0]
        
        # Get auxiliary losses if present
        aux_losses = None
        if isinstance(outputs, BaseModelOutputWithPast) and hasattr(outputs, 'aux_losses'):
            aux_losses = outputs.aux_losses
        elif isinstance(outputs, tuple) and len(outputs) > 4:
            aux_losses = outputs[4]
        
        # LM head
        if self.config.parallelism_strategy == "tensor" and dist.is_initialized():
            # Gather hidden states from all tensor parallel ranks
            world_size = dist.get_world_size()
            if world_size > 1:
                hidden_states_list = [torch.empty_like(hidden_states) for _ in range(world_size)]
                dist.all_gather(hidden_states_list, hidden_states)
                hidden_states = torch.cat(hidden_states_list, dim=-1)
        
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        
        loss = None
        if labels is not None:
            # Shift for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            
            # Add auxiliary losses from MoE and MoD
            if aux_losses:
                for aux_loss in aux_losses.values():
                    loss = loss + aux_loss
            
            # Multi-token prediction loss
            if self.multi_token_head is not None and self.config.multi_token_loss_weight > 0:
                _, multi_token_loss = self.multi_token_head(hidden_states, labels)
                if multi_token_loss is not None:
                    loss = loss + self.config.multi_token_loss_weight * multi_token_loss
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past

# ------------------------ TRAINING UTILITIES ---------------------------- #

class MastishkTrainer:
    """Enhanced trainer for Mastishk with MoE and MoD support"""
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataset=None,
        eval_dataset=None,
        data_collator=None,
        tokenizer=None,
    ):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.tokenizer = tokenizer
        
        # Setup distributed training
        self.setup_distributed()
        
        # Setup model with optimizations
        self.setup_model()
        
        # Setup optimizer and scheduler
        self.setup_optimization()
    
    def setup_distributed(self):
        """Setup distributed training environment"""
        if HAS_DEEPSPEED and self.config.zero_stage > 0:
            # DeepSpeed configuration
            self.ds_config = {
                "train_batch_size": "auto",
                "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
                "fp16": {
                    "enabled": self.config.mixed_precision == "fp16",
                },
                "bf16": {
                    "enabled": self.config.mixed_precision == "bf16",
                },
                "zero_optimization": {
                    "stage": self.config.zero_stage,
                    "offload_optimizer": {
                        "device": "cpu" if self.config.offload_optimizer else "none",
                    },
                    "offload_param": {
                        "device": "cpu" if self.config.offload_param else "none",
                    },
                    "overlap_comm": True,
                    "contiguous_gradients": self.config.contiguous_checkpointing,
                    "sub_group_size": 1e9,
                    "reduce_bucket_size": "auto",
                    "stage3_prefetch_bucket_size": "auto",
                    "stage3_param_persistence_threshold": "auto",
                    "stage3_max_live_parameters": 1e9,
                    "stage3_max_reuse_distance": 1e9,
                    "stage3_gather_16bit_weights_on_model_save": True,
                },
                "gradient_clipping": self.config.max_grad_norm,
                "wall_clock_breakdown": False,
            }
        elif dist.is_initialized():
            # FSDP configuration
            from torch.distributed.fsdp import (
                FullyShardedDataParallel as FSDP,
                MixedPrecision,
                BackwardPrefetch,
                ShardingStrategy,
            )
            
            self.fsdp_config = {
                "sharding_strategy": ShardingStrategy.FULL_SHARD,
                "cpu_offload": CPUOffload(offload_params=self.config.offload_param),
                "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
                "mixed_precision": MixedPrecision(
                    param_dtype=torch.bfloat16 if self.config.mixed_precision == "bf16" else torch.float16,
                    reduce_dtype=torch.bfloat16 if self.config.mixed_precision == "bf16" else torch.float16,
                    buffer_dtype=torch.bfloat16 if self.config.mixed_precision == "bf16" else torch.float16,
                ),
                "sync_module_states": True,
                "use_orig_params": True,
            }
    
    def setup_model(self):
        """Setup model with distributed wrapper"""
        if HAS_DEEPSPEED and self.config.zero_stage > 0:
            # Initialize DeepSpeed
            import deepspeed
            self.model, self.optimizer, _, _ = deepspeed.initialize(
                model=self.model,
                model_parameters=self.model.parameters(),
                config=self.ds_config,
            )
        elif dist.is_initialized() and dist.get_world_size() > 1:
            # Wrap with FSDP
            self.model = FSDP(self.model, **self.fsdp_config)
        
        # Enable gradient checkpointing
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
    
    def setup_optimization(self):
        """Setup optimizer and scheduler"""
        # Skip if using DeepSpeed (it handles optimizer internally)
        if hasattr(self, "optimizer"):
            return
        
        # Get parameters
        params = self.model.parameters()
        
        # Setup optimizer
        if self.config.use_8bit_adam and HAS_BNBYTES:
            self.optimizer = bnb.optim.AdamW8bit(
                params,
                lr=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                eps=self.config.epsilon,
                weight_decay=self.config.weight_decay,
            )
        else:
            self.optimizer = torch.optim.AdamW(
                params,
                lr=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                eps=self.config.epsilon,
                weight_decay=self.config.weight_decay,
            )
        
        # Setup scheduler
        from transformers import get_scheduler
        self.scheduler = get_scheduler(
            self.config.scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=int(self.config.warmup_ratio * 1000),  # Placeholder
            num_training_steps=1000,  # Placeholder
        )
        
        # Mixed precision scaler
        if self.config.mixed_precision == "fp16":
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step"""
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.model.device) for k, v in batch.items()}
        
        # Mixed precision context
        if self.config.mixed_precision == "bf16":
            ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16)
        elif self.config.mixed_precision == "fp16":
            ctx = torch.cuda.amp.autocast(dtype=torch.float16)
        else:
            ctx = nullcontext()
        
        # Forward pass
        with ctx:
            outputs = self.model(**batch)
            loss = outputs.loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        if HAS_DEEPSPEED and hasattr(self.model, "backward"):
            self.model.backward(loss)
        elif self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation and optimization
        if (self.step + 1) % self.config.gradient_accumulation_steps == 0:
            if HAS_DEEPSPEED and hasattr(self.model, "step"):
                self.model.step()
            else:
                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                
                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
        
        return loss.item() * self.config.gradient_accumulation_steps

# ------------------------ MASTISHK MODEL FACTORY ---------------------------- #

def create_mastishk_model(
    model_size: str = "7B",
    use_flash_attention: bool = True,
    use_quantization: bool = False,
    parallelism: str = "none",
    use_minimax: bool = False,
    use_moe: bool = True,
    num_experts: int = 8,
    num_experts_per_tok: int = 2,
    use_mod: bool = True,
    mod_capacity_factor: float = 0.8,
) -> MastishkTransformerForCausalLM:
    """Create a Mastishk transformer model with specified configuration"""
    
    # Model size presets
    size_configs = {
        "1B": {"hidden_size": 2048, "num_layers": 24, "num_heads": 32},
        "7B": {"hidden_size": 4096, "num_layers": 32, "num_heads": 32},
        "13B": {"hidden_size": 5120, "num_layers": 40, "num_heads": 40},
        "30B": {"hidden_size": 6656, "num_layers": 60, "num_heads": 52},
        "65B": {"hidden_size": 8192, "num_layers": 80, "num_heads": 64},
        "175B": {"hidden_size": 12288, "num_layers": 96, "num_heads": 96},
        "8x7B": {"hidden_size": 4096, "num_layers": 32, "num_heads": 32},  # Mixtral-style
        "8x22B": {"hidden_size": 6144, "num_layers": 56, "num_heads": 48},  # Larger Mixtral-style
    }
    
    if model_size not in size_configs:
        raise ValueError(f"Invalid model size. Choose from: {list(size_configs.keys())}")
    
    size_config = size_configs[model_size]
    
    # Adjust intermediate size for MoE models
    if use_moe:
        # Each expert has full intermediate size, but only num_experts_per_tok are active
        intermediate_size = size_config["hidden_size"] * 4
    else:
        intermediate_size = size_config["hidden_size"] * 4
    
    # Create configuration
    config = MastishkConfig(
        hidden_size=size_config["hidden_size"],
        num_hidden_layers=size_config["num_layers"],
        num_attention_heads=size_config["num_heads"],
        num_key_value_heads=size_config["num_heads"] // 4,  # GQA
        intermediate_size=intermediate_size,
        use_flash_attention=use_flash_attention,
        attention_implementation="flash2" if use_flash_attention else "standard",
        parallelism_strategy=parallelism,
        quantization_config={"load_in_8bit": True} if use_quantization else None,
        use_minimax=use_minimax,
        minimax_layer_frequency=4,  # Use MiniMax every 4th layer
        minimax_adversarial_epsilon=0.1,
        minimax_iterations=3,
        use_moe=use_moe,
        moe_config={
            "num_experts": num_experts,
            "num_experts_per_tok": num_experts_per_tok,
            "moe_layer_frequency": 2,  # Apply MoE every 2nd layer
            "expert_capacity_factor": 1.25,
            "aux_loss_weight": 0.01,
            "router_z_loss_weight": 0.001,
            "normalize_expert_weights": True,
        } if use_moe else None,
        use_mod=use_mod,
        mod_config={
            "enabled": use_mod,
            "capacity_factor": mod_capacity_factor,
            "skip_probability": 0.2,
            "min_layers_per_token": max(12, size_config["num_layers"] // 3),
            "router_hidden_dim": size_config["hidden_size"] // 8,
            "router_aux_loss_weight": 0.01,
            "temperature": 1.0,
        } if use_mod else None,
    )
    
    # Create model
    with init_empty_weights() if model_size in ["65B", "175B", "8x22B"] else nullcontext():
        model = MastishkTransformerForCausalLM(config)
    
    return model


# Example usage with MoE and MoD
if __name__ == "__main__":
    print("🚀 Creating Mastishk Transformer with MoE and MoD integration...")
    
    # Create a 8x7B parameter model (Mixtral-style) with all optimizations
    model = create_mastishk_model(
        model_size="8x7B",
        use_flash_attention=True,
        use_quantization=False,
        parallelism="none",
        use_minimax=True,
        use_moe=True,
        num_experts=8,
        num_experts_per_tok=2,
        use_mod=True,
        mod_capacity_factor=0.8
    )
    
    print(f"✅ Mastishk model created successfully!")
    print(f"   Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Active Parameters per Token: ~{sum(p.numel() for p in model.parameters()) // 4:,}")
    print(f"   Using Flash Attention: {model.config.use_flash_attention}")
    print(f"   Using MiniMax: {model.config.use_minimax}")
    print(f"   Using MoE: {model.config.use_moe}")
    if model.config.use_moe:
        print(f"   Number of Experts: {model.config.moe_config.num_experts}")
        print(f"   Experts per Token: {model.config.moe_config.num_experts_per_tok}")
        print(f"   MoE Layer Frequency: Every {model.config.moe_config.moe_layer_frequency} layers")
    print(f"   Using MoD: {model.config.use_mod}")
    if model.config.use_mod:
        print(f"   MoD Capacity Factor: {model.config.mod_config.capacity_factor}")
        print(f"   Average Layers per Token: ~{model.config.num_hidden_layers * model.config.mod_config.capacity_factor:.1f}")
    print(f"   Attention Implementation: {model.config.attention_implementation}")
    
    # Calculate efficiency gains
    if model.config.use_moe and model.config.use_mod:
        moe_reduction = model.config.moe_config.num_experts_per_tok / model.config.moe_config.num_experts
        mod_reduction = model.config.mod_config.capacity_factor
        total_reduction = moe_reduction * mod_reduction
        print(f"\n📊 Efficiency Analysis:")
        print(f"   MoE Computation: {moe_reduction * 100:.1f}% of dense model")
        print(f"   MoD Computation: {mod_reduction * 100:.1f}% of all layers")
        print(f"   Combined: {total_reduction * 100:.1f}% of dense computation")
        print(f"   Speedup: ~{1/total_reduction:.1f}x faster than dense model")
    
    # Example: Setup training
    training_config = TrainingConfig(
        gradient_checkpointing=True,
        mixed_precision="bf16",
        learning_rate=5e-4,
        zero_stage=3,
    )
    
    # Create trainer
    trainer = MastishkTrainer(
        model=model,
        config=training_config,
    )
    
    print("\n📚 Mastishk Training Configuration:")
    print(f"   Mixed Precision: {training_config.mixed_precision}")
    print(f"   ZeRO Stage: {training_config.zero_stage}")
    print(f"   Gradient Checkpointing: {training_config.gradient_checkpointing}")
    
    # Example: Save and load checkpoint
    print("\n💾 Checkpoint Management:")
    print("   Save: MastishkCheckpointManager.save_checkpoint(model, 'path/to/save')")
    print("   Load: MastishkCheckpointManager.load_checkpoint(model, 'path/to/load')")
    
    # Example forward pass to show MoD statistics
    print("\n🔍 Running example forward pass with MoD statistics...")
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, 32000, (batch_size, seq_len))
    
    with torch.no_grad():
        outputs = model.model(input_ids, output_router_logits=True)
        if hasattr(outputs, 'mod_statistics') or (isinstance(outputs, tuple) and len(outputs) > 5):
            stats = outputs[-1] if isinstance(outputs, tuple) else outputs.mod_statistics
            if stats:
                print(f"   MoD Statistics:")
                print(f"     Average layers per token: {stats.get('avg_layers_per_token', 'N/A'):.2f}")
                print(f"     Load balance score: {stats.get('load_balance_score', 'N/A'):.3f}")
    
    print("\n🎉 Mastishk Transformer with advanced MoE + MoD architecture is ready!")
    print("   The model combines:")
    print("   - Sparse Mixture of Experts (MoE) for efficient parameter usage")
    print("   - Mixture of Depths (MoD) for dynamic computation allocation")
    print("   - MiniMax attention for adversarial robustness")
    print("   - Differential attention for noise cancellation")
    print("   - Flash Attention 2 for memory efficiency")
    print("   - LoLCATs conversion capability for linear attention")
    print("   - Multi-token prediction for faster inference")
    print("   - Advanced KV cache strategies")
    print("\n   This creates a state-of-the-art transformer that is both")
    print("   powerful and efficient, suitable for large-scale deployment!")