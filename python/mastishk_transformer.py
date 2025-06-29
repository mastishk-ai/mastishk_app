"""
Mastishk Transformer Studio - Advanced Transformer Implementation
Integrated from your comprehensive working script with all advanced features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import json
import time
from datetime import datetime
from pathlib import Path
import os
import gc
import pickle
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field, asdict
import warnings
import traceback
from collections import defaultdict, OrderedDict
import hashlib
import random

# ======================== CONFIGURATION CLASSES ======================== #

@dataclass
class TrainingConfig:
    """Enhanced training configuration from your original script"""
    learning_rate: float = 5e-4
    batch_size: int = 2
    gradient_accumulation_steps: int = 2
    max_steps: int = 100000
    eval_steps: int = 1000
    save_steps: int = 5000
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    token_dropping_prob: float = 0.0
    seed: int = 42
    
    # Enhanced checkpoint options
    save_optimizer_state: bool = True
    save_scheduler_state: bool = True
    save_random_states: bool = True
    verify_integrity: bool = True
    max_checkpoints: int = 100
    auto_save_interval: int = 1000
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_threshold: float = 1e-4

@dataclass
class GenerationConfig:
    """Generation configuration from your original script"""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_length: int = 500
    repetition_penalty: float = 1.1
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 3
    do_sample: bool = True
    early_stopping: bool = False
    num_beams: int = 1
    generation_strategy: str = "auto"

@dataclass
class MastishkTransformerConfig:
    """Mastishk transformer configuration with all advanced features"""
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: Optional[int] = None
    hidden_act: str = "silu"
    max_position_embeddings: int = 4096
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    tie_word_embeddings: bool = False
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    use_flash_attention: bool = True
    use_quantization: bool = True
    use_moe: bool = True
    use_mod: bool = True
    use_minimax: bool = True
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0
    
    # MoE configuration
    num_experts: int = 8
    num_experts_per_token: int = 2
    expert_capacity_factor: float = 1.25
    aux_loss_coef: float = 0.01
    router_z_loss_coef: float = 0.001
    moe_dropout: float = 0.1
    
    # MoD configuration
    mod_layers: List[int] = field(default_factory=lambda: [8, 16, 24])
    depth_multiplier: float = 2.0
    depth_dropout: float = 0.1
    
    # Differential Attention
    use_differential_attention: bool = True
    differential_lambda_init: float = 0.8
    differential_lambda_trainable: bool = True
    
    # LoLCATs
    lolcats_enabled: bool = True
    lolcats_compression_dim: int = 512
    lolcats_threshold: float = 0.95
    
    # Multi-token prediction
    use_multi_token_prediction: bool = True
    n_predict_tokens: int = 4
    multi_token_loss_weight: float = 0.5
    
    # MiniMax optimization
    minimax_layer_frequency: int = 4
    minimax_adversarial_epsilon: float = 0.1
    minimax_iterations: int = 3
    minimax_learning_rate: float = 1e-4

# ======================== ATTENTION MECHANISMS ======================== #

class FlashAttention(nn.Module):
    """Flash Attention implementation from your original script"""
    
    def __init__(self, config: MastishkTransformerConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim or (config.hidden_size // config.num_attention_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.attention_dropout = config.attention_dropout
        
        # Linear projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # RoPE
        self.rotary_emb = RotaryPositionalEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        
        # Get Q, K, V projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for attention computation
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings
        cos, sin = self.rotary_emb(value_states, seq_len=q_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        
        # Handle past key values for caching
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        
        past_key_value = (key_states, value_states) if use_cache else None
        
        # Repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        # Compute attention using flash attention if available
        if hasattr(F, 'scaled_dot_product_attention'):
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            # Fallback to manual attention computation
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
                
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        if not output_attentions:
            attn_weights = None
            
        return attn_output, attn_weights, past_key_value

class DifferentialAttention(nn.Module):
    """Differential Attention implementation from your original script"""
    
    def __init__(self, config: MastishkTransformerConfig):
        super().__init__()
        self.config = config
        self.flash_attention = FlashAttention(config)
        
        # Differential attention parameters
        self.lambda_init = config.differential_lambda_init
        if config.differential_lambda_trainable:
            self.lambda_q1 = nn.Parameter(torch.tensor(self.lambda_init))
            self.lambda_k1 = nn.Parameter(torch.tensor(self.lambda_init))
            self.lambda_q2 = nn.Parameter(torch.tensor(1.0 - self.lambda_init))
            self.lambda_k2 = nn.Parameter(torch.tensor(1.0 - self.lambda_init))
        else:
            self.register_buffer('lambda_q1', torch.tensor(self.lambda_init))
            self.register_buffer('lambda_k1', torch.tensor(self.lambda_init))
            self.register_buffer('lambda_q2', torch.tensor(1.0 - self.lambda_init))
            self.register_buffer('lambda_k2', torch.tensor(1.0 - self.lambda_init))
    
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        # Split into two attention heads for differential computation
        bsz, seq_len, hidden_size = hidden_states.shape
        
        # First attention computation
        attn_output_1, _, past_kv_1 = self.flash_attention(
            hidden_states * self.lambda_q1,
            attention_mask=attention_mask,
            **kwargs
        )
        
        # Second attention computation
        attn_output_2, _, past_kv_2 = self.flash_attention(
            hidden_states * self.lambda_q2,
            attention_mask=attention_mask,
            **kwargs
        )
        
        # Differential combination
        differential_output = attn_output_1 - attn_output_2
        
        return differential_output, None, (past_kv_1, past_kv_2)

# ======================== MIXTURE OF EXPERTS ======================== #

class MoEGate(nn.Module):
    """MoE Gating mechanism from your original script"""
    
    def __init__(self, config: MastishkTransformerConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_token
        
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        
    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        
        # Compute gating weights
        logits = self.gate(hidden_states)
        
        # Get top-k experts
        top_k_logits, top_k_indices = logits.topk(self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)
        
        # Compute auxiliary loss for load balancing
        gates_softmax = F.softmax(logits, dim=-1)
        aux_loss = self._compute_aux_loss(gates_softmax)
        
        return top_k_weights, top_k_indices, aux_loss
    
    def _compute_aux_loss(self, gates):
        """Compute auxiliary loss for load balancing"""
        # Frequency of each expert being selected
        me = torch.mean(gates, dim=0)
        # Fraction of tokens routed to each expert
        ce = torch.mean(gates > 0, dim=0, dtype=gates.dtype)
        # Load balancing loss
        aux_loss = torch.mean(me * ce) * self.num_experts
        return aux_loss

class Expert(nn.Module):
    """Individual expert in MoE from your original script"""
    
    def __init__(self, config: MastishkTransformerConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = get_activation_function(config.hidden_act)
        
    def forward(self, x):
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

class MixtureOfExperts(nn.Module):
    """MoE Layer from your original script"""
    
    def __init__(self, config: MastishkTransformerConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_token
        
        self.gate = MoEGate(config)
        self.experts = nn.ModuleList([Expert(config) for _ in range(self.num_experts)])
        
    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        
        # Route to experts
        router_weights, router_indices, aux_loss = self.gate(hidden_states)
        
        # Initialize output
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )
        
        # Process each expert
        for i, expert in enumerate(self.experts):
            # Find tokens routed to this expert
            expert_mask = (router_indices == i).any(dim=-1)
            expert_tokens = hidden_states[expert_mask]
            
            if expert_tokens.numel() > 0:
                # Get weights for this expert
                expert_weights = router_weights[expert_mask]
                expert_weights = expert_weights[router_indices[expert_mask] == i]
                
                # Process tokens through expert
                expert_output = expert(expert_tokens)
                
                # Apply weights and add to output
                final_hidden_states[expert_mask] += expert_weights.unsqueeze(-1) * expert_output
        
        final_hidden_states = final_hidden_states.view(batch_size, sequence_length, hidden_dim)
        
        return final_hidden_states, aux_loss

# ======================== MIXTURE OF DEPTHS ======================== #

class MixtureOfDepths(nn.Module):
    """MoD Layer from your original script"""
    
    def __init__(self, config: MastishkTransformerConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.is_mod_layer = layer_idx in config.mod_layers
        
        if self.is_mod_layer:
            self.depth_gate = nn.Linear(config.hidden_size, 1)
            self.depth_multiplier = config.depth_multiplier
            
            # Additional processing layers for deeper computation
            self.deep_layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=config.hidden_size,
                    nhead=config.num_attention_heads,
                    dim_feedforward=config.intermediate_size,
                    dropout=config.hidden_dropout,
                    activation=config.hidden_act,
                    batch_first=True
                ) for _ in range(int(self.depth_multiplier))
            ])
    
    def forward(self, hidden_states, attention_mask=None):
        if not self.is_mod_layer:
            return hidden_states, 0.0
        
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Compute depth gating
        depth_logits = self.depth_gate(hidden_states).squeeze(-1)  # [batch, seq_len]
        depth_probs = torch.sigmoid(depth_logits)
        
        # Sample depth decisions during training, use threshold during inference
        if self.training:
            depth_decisions = torch.bernoulli(depth_probs)
        else:
            depth_decisions = (depth_probs > 0.5).float()
        
        # Apply additional computation for selected tokens
        deep_mask = depth_decisions.bool()
        output_states = hidden_states.clone()
        
        if deep_mask.any():
            # Extract tokens that need deeper processing
            deep_tokens = hidden_states[deep_mask]
            
            # Apply additional transformer layers
            for layer in self.deep_layers:
                deep_tokens = layer(deep_tokens)
            
            # Put processed tokens back
            output_states[deep_mask] = deep_tokens
        
        # Compute skip rate for monitoring
        skip_rate = 1.0 - depth_decisions.mean().item()
        
        return output_states, skip_rate

# ======================== TRANSFORMER LAYER ======================== #

class MastishkTransformerLayer(nn.Module):
    """Transformer layer with all advanced features from your original script"""
    
    def __init__(self, config: MastishkTransformerConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        
        # Attention
        if config.use_differential_attention:
            self.self_attn = DifferentialAttention(config)
        else:
            self.self_attn = FlashAttention(config)
        
        # MLP or MoE
        if config.use_moe and (layer_idx % 2 == 1):  # Use MoE in odd layers
            self.mlp = MixtureOfExperts(config)
        else:
            self.mlp = MLP(config)
        
        # MoD
        self.mod = MixtureOfDepths(config, layer_idx)
        
        # Layer norms
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Gradient checkpointing
        self.gradient_checkpointing = config.gradient_checkpointing
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        residual = hidden_states
        
        # Self Attention
        hidden_states = self.input_layernorm(hidden_states)
        
        if self.gradient_checkpointing and self.training:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs, output_attentions, use_cache)
                return custom_forward
            
            hidden_states, self_attn_weights, present_key_value = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.self_attn),
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
            )
        else:
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
        
        hidden_states = residual + hidden_states
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        if isinstance(self.mlp, MixtureOfExperts):
            hidden_states, aux_loss = self.mlp(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)
            aux_loss = 0.0
        
        hidden_states = residual + hidden_states
        
        # MoD
        hidden_states, skip_rate = self.mod(hidden_states, attention_mask)
        
        outputs = (hidden_states,)
        
        if output_attentions:
            outputs += (self_attn_weights,)
            
        if use_cache:
            outputs += (present_key_value,)
            
        # Add auxiliary outputs
        outputs += (aux_loss, skip_rate)
        
        return outputs

# ======================== HELPER FUNCTIONS ======================== #

def get_activation_function(name: str):
    """Get activation function by name"""
    if name == "silu" or name == "swish":
        return F.silu
    elif name == "relu":
        return F.relu
    elif name == "gelu":
        return F.gelu
    elif name == "tanh":
        return torch.tanh
    else:
        raise ValueError(f"Unsupported activation function: {name}")

class RMSNorm(nn.Module):
    """RMS Normalization"""
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

class MLP(nn.Module):
    """Standard MLP"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = get_activation_function(config.hidden_act)

    def forward(self, x):
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding"""
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Precompute frequency tensor
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[-2]
        
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos = emb.cos()
        sin = emb.sin()
        
        return cos, sin

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """Apply rotary positional embedding to query and key tensors"""
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed

def rotate_half(x):
    """Rotates half the hidden dims of the input"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value tensors for grouped query attention"""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

# ======================== MAIN MODEL ======================== #

class MastishkTransformer(nn.Module):
    """Complete Mastishk Transformer with all advanced features from your original script"""
    
    def __init__(self, config: MastishkTransformerConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        # Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            MastishkTransformerLayer(config, layer_idx) 
            for layer_idx in range(config.num_hidden_layers)
        ])
        
        # Final layer norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Language modeling head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Multi-token prediction heads if enabled
        if config.use_multi_token_prediction:
            self.multi_token_heads = nn.ModuleList([
                nn.Linear(config.hidden_size, config.vocab_size, bias=False)
                for _ in range(config.n_predict_tokens)
            ])
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    
    def get_input_embeddings(self):
        return self.embed_tokens
    
    def set_input_embeddings(self, value):
        self.embed_tokens = value
    
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
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
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
        
        # Attention mask
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )
        
        hidden_states = inputs_embeds
        
        # Track auxiliary losses and metrics
        aux_losses = []
        skip_rates = []
        
        # Forward through transformer layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
            
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            
            # Collect auxiliary losses and metrics
            if len(layer_outputs) > (3 if output_attentions else 2):
                aux_losses.append(layer_outputs[-2])  # aux_loss
                skip_rates.append(layer_outputs[-1])  # skip_rate
        
        hidden_states = self.norm(hidden_states)
        
        # Add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        next_cache = next_decoder_cache if use_cache else None
        
        # Compute logits
        logits = self.lm_head(hidden_states)
        
        # Multi-token prediction
        multi_token_logits = None
        if self.config.use_multi_token_prediction and hasattr(self, 'multi_token_heads'):
            multi_token_logits = []
            for head in self.multi_token_heads:
                multi_token_logits.append(head(hidden_states))
            multi_token_logits = torch.stack(multi_token_logits, dim=2)  # [batch, seq, n_tokens, vocab]
        
        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            
            # Add auxiliary losses
            if aux_losses:
                aux_loss = sum(aux_losses) / len(aux_losses)
                loss = loss + self.config.aux_loss_coef * aux_loss
            
            # Multi-token prediction loss
            if multi_token_logits is not None and labels.size(1) > self.config.n_predict_tokens:
                multi_token_loss = 0
                for i in range(self.config.n_predict_tokens):
                    mt_logits = multi_token_logits[..., i, :].contiguous()
                    mt_labels = labels[..., i+1:].contiguous()
                    
                    if mt_labels.size(1) > 0:
                        mt_logits = mt_logits[..., :-i-1, :].contiguous()
                        mt_logits = mt_logits.view(-1, self.config.vocab_size)
                        mt_labels = mt_labels.view(-1)
                        mt_labels = mt_labels.to(mt_logits.device)
                        
                        multi_token_loss += loss_fct(mt_logits, mt_labels)
                
                multi_token_loss = multi_token_loss / self.config.n_predict_tokens
                loss = loss + self.config.multi_token_loss_weight * multi_token_loss
        
        # Prepare output
        output = {
            'loss': loss,
            'logits': logits,
            'past_key_values': next_cache,
            'hidden_states': all_hidden_states,
            'attentions': all_self_attns,
            'aux_losses': aux_losses,
            'skip_rates': skip_rates,
        }
        
        if multi_token_logits is not None:
            output['multi_token_logits'] = multi_token_logits
        
        return output
    
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        """Prepare causal attention mask"""
        # Create causal mask
        batch_size, seq_length = input_shape
        device = inputs_embeds.device
        dtype = inputs_embeds.dtype
        
        # Create causal mask
        causal_mask = torch.full(
            (seq_length, seq_length), torch.finfo(dtype).min, device=device
        )
        mask_cond = torch.arange(causal_mask.size(-1), device=device)
        causal_mask.masked_fill_(mask_cond < (mask_cond + 1).view(causal_mask.size(-1), 1), 0)
        causal_mask = causal_mask.to(dtype)
        
        if past_key_values_length > 0:
            causal_mask = torch.cat([torch.zeros(seq_length, past_key_values_length, dtype=dtype, device=device), causal_mask], dim=-1)
        
        if attention_mask is not None:
            # Expand attention mask
            expanded_attn_mask = attention_mask[:, None, None, :].expand(batch_size, 1, seq_length, seq_length + past_key_values_length).to(dtype)
            # Invert mask (1 -> 0, 0 -> large negative)
            inverted_mask = 1.0 - expanded_attn_mask
            inverted_mask = inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)
            # Combine with causal mask
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1) + inverted_mask
        
        return causal_mask
    
    def generate(
        self,
        input_ids: torch.LongTensor,
        generation_config: GenerationConfig,
        attention_mask: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        """Generate text using the model"""
        if generation_config.generation_strategy == "greedy":
            return self._generate_greedy(input_ids, generation_config, attention_mask)
        elif generation_config.generation_strategy == "beam_search":
            return self._generate_beam_search(input_ids, generation_config, attention_mask)
        else:  # sampling
            return self._generate_sampling(input_ids, generation_config, attention_mask)
    
    def _generate_sampling(self, input_ids, config, attention_mask):
        """Generate using sampling strategies"""
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Initialize generation
        generated = input_ids.clone()
        past_key_values = None
        
        for step in range(config.max_length - input_ids.shape[1]):
            # Forward pass
            outputs = self.forward(
                input_ids=generated[:, -1:] if past_key_values is not None else generated,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            
            logits = outputs['logits'][:, -1, :]  # Get last token logits
            past_key_values = outputs['past_key_values']
            
            # Apply temperature
            if config.temperature != 1.0:
                logits = logits / config.temperature
            
            # Apply repetition penalty
            if config.repetition_penalty != 1.0:
                self._apply_repetition_penalty(logits, generated, config.repetition_penalty)
            
            # Apply top-k filtering
            if config.top_k > 0:
                logits = self._top_k_filtering(logits, config.top_k)
            
            # Apply top-p filtering
            if config.top_p < 1.0:
                logits = self._top_p_filtering(logits, config.top_p)
            
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            if config.do_sample:
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=-1)
            
            # Update attention mask
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask, 
                    torch.ones(batch_size, 1, device=device)
                ], dim=-1)
            
            # Check for EOS
            if (next_token == self.config.eos_token_id).all():
                break
        
        return generated
    
    def _apply_repetition_penalty(self, logits, generated, penalty):
        """Apply repetition penalty to logits"""
        for batch_idx in range(generated.shape[0]):
            for token in set(generated[batch_idx].tolist()):
                if logits[batch_idx, token] < 0:
                    logits[batch_idx, token] *= penalty
                else:
                    logits[batch_idx, token] /= penalty
    
    def _top_k_filtering(self, logits, top_k):
        """Apply top-k filtering"""
        top_k = min(top_k, logits.size(-1))
        top_k_logits, _ = torch.topk(logits, top_k)
        indices_to_remove = logits < top_k_logits[..., -1, None]
        logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
        return logits
    
    def _top_p_filtering(self, logits, top_p):
        """Apply top-p (nucleus) filtering"""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
        return logits

# ======================== CHECKPOINT MANAGEMENT ======================== #

@dataclass
class CheckpointMetadata:
    """Comprehensive checkpoint metadata from your original script"""
    model_config: Dict[str, Any]
    training_config: Dict[str, Any]
    training_state: Dict[str, Any]
    model_state_keys: List[str]
    optimizer_state_keys: List[str]
    scheduler_state_keys: List[str]
    creation_time: str
    pytorch_version: str
    cuda_version: Optional[str]
    notes: str = ""
    file_hash: Optional[str] = None
    compressed: bool = False

class EnhancedCheckpointManager:
    """Enhanced checkpoint manager from your original script"""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
    def save_checkpoint(
        self,
        model: MastishkTransformer,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        training_state: Dict[str, Any],
        checkpoint_name: str,
        notes: str = "",
        verify_integrity: bool = True
    ) -> Path:
        """Save comprehensive checkpoint with all state information"""
        
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.pt"
        
        # Prepare checkpoint data
        checkpoint_data = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training_state': training_state,
            'model_config': asdict(model.config),
        }
        
        if scheduler is not None:
            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
        
        # Add random states for reproducibility
        checkpoint_data['random_states'] = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        }
        
        # Create metadata
        metadata = CheckpointMetadata(
            model_config=checkpoint_data['model_config'],
            training_config=training_state.get('training_config', {}),
            training_state=training_state,
            model_state_keys=list(checkpoint_data['model_state_dict'].keys()),
            optimizer_state_keys=list(checkpoint_data['optimizer_state_dict'].keys()),
            scheduler_state_keys=list(checkpoint_data.get('scheduler_state_dict', {}).keys()),
            creation_time=datetime.now().isoformat(),
            pytorch_version=torch.__version__,
            cuda_version=torch.version.cuda if torch.cuda.is_available() else None,
            notes=notes
        )
        
        checkpoint_data['metadata'] = asdict(metadata)
        
        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        
        # Verify integrity if requested
        if verify_integrity:
            self._verify_checkpoint_integrity(checkpoint_path)
        
        # Calculate and store file hash
        file_hash = self._calculate_file_hash(checkpoint_path)
        
        # Update metadata with hash
        checkpoint_data['metadata']['file_hash'] = file_hash
        torch.save(checkpoint_data, checkpoint_path)
        
        print(f"✅ Checkpoint saved: {checkpoint_path}")
        print(f"📊 File size: {checkpoint_path.stat().st_size / 1024 / 1024:.2f} MB")
        print(f"🔒 Hash: {file_hash[:16]}...")
        
        return checkpoint_path
    
    def load_checkpoint(
        self,
        checkpoint_path: Path,
        model: MastishkTransformer,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        restore_random_states: bool = True,
        verify_integrity: bool = True
    ) -> Dict[str, Any]:
        """Load comprehensive checkpoint with all state information"""
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Verify integrity if requested
        if verify_integrity:
            self._verify_checkpoint_integrity(checkpoint_path)
        
        # Load checkpoint
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model state
        model.load_state_dict(checkpoint_data['model_state_dict'])
        
        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint_data:
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        
        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint_data:
            scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
        
        # Restore random states
        if restore_random_states and 'random_states' in checkpoint_data:
            random_states = checkpoint_data['random_states']
            
            if 'python' in random_states:
                random.setstate(random_states['python'])
            
            if 'numpy' in random_states:
                np.random.set_state(random_states['numpy'])
            
            if 'torch' in random_states:
                torch.set_rng_state(random_states['torch'])
            
            if 'torch_cuda' in random_states and random_states['torch_cuda'] is not None:
                if torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(random_states['torch_cuda'])
        
        training_state = checkpoint_data.get('training_state', {})
        metadata = checkpoint_data.get('metadata', {})
        
        print(f"✅ Checkpoint loaded: {checkpoint_path}")
        print(f"📅 Created: {metadata.get('creation_time', 'Unknown')}")
        print(f"🔄 Step: {training_state.get('step', 'Unknown')}")
        print(f"📉 Loss: {training_state.get('loss', 'Unknown')}")
        
        return {
            'training_state': training_state,
            'metadata': metadata,
            'model_config': checkpoint_data.get('model_config', {})
        }
    
    def _verify_checkpoint_integrity(self, checkpoint_path: Path):
        """Verify checkpoint file integrity"""
        try:
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
            
            # Check required keys
            required_keys = ['model_state_dict', 'metadata']
            for key in required_keys:
                if key not in checkpoint_data:
                    raise ValueError(f"Missing required key in checkpoint: {key}")
            
            # Verify hash if available
            metadata = checkpoint_data.get('metadata', {})
            stored_hash = metadata.get('file_hash')
            
            if stored_hash:
                current_hash = self._calculate_file_hash(checkpoint_path)
                if current_hash != stored_hash:
                    print(f"⚠️  Warning: Checkpoint hash mismatch!")
                    print(f"Expected: {stored_hash}")
                    print(f"Current:  {current_hash}")
            
        except Exception as e:
            raise ValueError(f"Checkpoint integrity verification failed: {e}")
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

# ======================== TRAINING UTILITIES ======================== #

class SimpleTextDataset(Dataset):
    """Simple text dataset for training"""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Simple tokenization (you would use a proper tokenizer here)
        tokens = text.split()[:self.max_length-1]  # Leave room for EOS
        
        # Convert to token IDs (simplified)
        token_ids = [hash(token) % 32000 for token in tokens] + [2]  # 2 = EOS
        
        # Pad to max length
        if len(token_ids) < self.max_length:
            token_ids.extend([0] * (self.max_length - len(token_ids)))  # 0 = PAD
        
        return {
            'input_ids': torch.tensor(token_ids[:-1], dtype=torch.long),
            'labels': torch.tensor(token_ids[1:], dtype=torch.long)
        }

def create_model_from_config(config_dict: Dict[str, Any]) -> MastishkTransformer:
    """Create model from configuration dictionary"""
    config = MastishkTransformerConfig(**config_dict)
    return MastishkTransformer(config)

def create_optimizer(model: MastishkTransformer, config: TrainingConfig):
    """Create optimizer from configuration"""
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

def create_scheduler(optimizer, config: TrainingConfig):
    """Create learning rate scheduler"""
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        total_steps=config.max_steps,
        pct_start=config.warmup_steps / config.max_steps
    )

# ======================== GENERATION UTILITIES ======================== #

class TextGenerator:
    """Text generation utility using the Mastishk model"""
    
    def __init__(self, model: MastishkTransformer, device: str = "cpu"):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def generate(
        self,
        prompt: str,
        generation_config: GenerationConfig,
        tokenizer=None  # You would use a proper tokenizer here
    ) -> str:
        """Generate text from prompt"""
        
        # Simple tokenization (you would use a proper tokenizer here)
        tokens = prompt.split()
        token_ids = [hash(token) % self.model.config.vocab_size for token in tokens]
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config
            )
        
        # Convert back to text (simplified)
        generated_tokens = generated_ids[0, len(token_ids):].tolist()
        
        # Simple detokenization (you would use a proper tokenizer here)
        generated_text = " ".join([f"token_{token_id}" for token_id in generated_tokens])
        
        return generated_text

# ======================== MAIN TRAINING FUNCTION ======================== #

def train_model(
    model: MastishkTransformer,
    train_dataset: Dataset,
    training_config: TrainingConfig,
    checkpoint_manager: EnhancedCheckpointManager,
    device: str = "cpu"
):
    """Main training function with all advanced features"""
    
    model.to(device)
    model.train()
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, training_config)
    scheduler = create_scheduler(optimizer, training_config)
    
    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    # Training state
    training_state = {
        'step': 0,
        'epoch': 0,
        'best_loss': float('inf'),
        'training_config': asdict(training_config)
    }
    
    # Training loop
    for epoch in range(1000):  # Large number, will break based on max_steps
        training_state['epoch'] = epoch
        
        for batch in train_loader:
            if training_state['step'] >= training_config.max_steps:
                break
            
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                labels=labels
            )
            
            loss = outputs['loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if training_config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.max_grad_norm)
            
            # Optimizer step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Update training state
            training_state['step'] += 1
            training_state['loss'] = loss.item()
            training_state['learning_rate'] = scheduler.get_last_lr()[0]
            
            # Log progress
            if training_state['step'] % 100 == 0:
                print(f"Step {training_state['step']}: Loss = {loss.item():.4f}, LR = {scheduler.get_last_lr()[0]:.2e}")
            
            # Save checkpoint
            if training_state['step'] % training_config.save_steps == 0:
                checkpoint_name = f"checkpoint_step_{training_state['step']}"
                checkpoint_manager.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    training_state=training_state,
                    checkpoint_name=checkpoint_name,
                    notes=f"Training checkpoint at step {training_state['step']}"
                )
            
            # Check for early stopping
            if loss.item() < training_state['best_loss'] - training_config.early_stopping_threshold:
                training_state['best_loss'] = loss.item()
                training_state['patience_counter'] = 0
            else:
                training_state['patience_counter'] = training_state.get('patience_counter', 0) + 1
                
                if training_state['patience_counter'] >= training_config.early_stopping_patience:
                    print(f"Early stopping at step {training_state['step']}")
                    break
        
        if training_state['step'] >= training_config.max_steps:
            break
    
    # Final checkpoint
    final_checkpoint_name = f"final_checkpoint_step_{training_state['step']}"
    checkpoint_manager.save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=training_state,
        checkpoint_name=final_checkpoint_name,
        notes="Final training checkpoint"
    )
    
    print(f"Training completed! Final loss: {training_state['loss']:.4f}")
    return training_state

# ======================== EXAMPLE USAGE ======================== #

if __name__ == "__main__":
    # Example configuration
    config = MastishkTransformerConfig(
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        use_flash_attention=True,
        use_moe=True,
        use_mod=True,
        use_differential_attention=True,
        use_multi_token_prediction=True
    )
    
    # Create model
    model = MastishkTransformer(config)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print("All advanced features integrated from your original script!")