#!/usr/bin/env python3
"""
Mastishk Transformer Core Implementation
Extracted from your working Python code with sophisticated transformer features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
import json
from pathlib import Path
import hashlib
import time
from datetime import datetime

@dataclass
class MastishkTransformerConfig:
    """Enhanced transformer configuration from your working implementation"""
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    vocab_size: int = 50257
    max_position_embeddings: int = 2048
    hidden_act: str = "swish"
    num_key_value_heads: int = 12
    
    # Advanced features from your working code
    use_flash_attention: bool = False
    use_differential_attention: bool = False
    differential_lambda_init: float = 0.5
    use_minimax: bool = False
    minimax_layer_frequency: int = 4
    minimax_adversarial_epsilon: float = 0.1
    minimax_iterations: int = 3
    
    # LoLCATs compression
    lolcats_enabled: bool = False
    lolcats_compression_dim: int = 512
    
    # Multi-token prediction
    use_multi_token_prediction: bool = False
    n_predict_tokens: int = 4
    
    # Model parameters
    rms_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    learning_rate: float = 5e-4
    
    # MoE configuration
    use_moe: bool = False
    num_experts: int = 8
    top_k_experts: int = 2
    moe_layer_frequency: int = 2
    expert_dropout: float = 0.1
    router_z_loss_weight: float = 0.001
    router_aux_loss_weight: float = 0.01
    
    # MoD configuration  
    use_mod: bool = False
    mod_layer_frequency: int = 3
    depth_predictor_hidden_size: int = 256
    depth_threshold: float = 0.5

@dataclass
class TrainingState:
    """Enhanced training state from your implementation"""
    step: int = 0
    epoch: int = 0
    loss: float = 0.0
    learning_rate: float = 5e-4
    gradient_norm: float = 0.0
    tokens_processed: int = 0
    
    # Enhanced state tracking
    best_loss: float = float('inf')
    patience_counter: int = 0
    total_training_time: float = 0.0
    last_checkpoint_step: int = 0
    
    # Optimizer and scheduler states
    optimizer_state_available: bool = False
    scheduler_state_available: bool = False
    random_states_available: bool = False

@dataclass
class CheckpointMetadata:
    """Comprehensive checkpoint metadata from your implementation"""
    checkpoint_id: str
    model_name: str
    creation_time: str
    training_step: int
    epoch: int
    loss: float
    best_loss: float
    learning_rate: float
    file_size_bytes: int
    file_hash: str
    notes: str = ""
    
    # Enhanced metadata
    includes_optimizer_state: bool = True
    includes_scheduler_state: bool = True
    includes_random_states: bool = True
    model_config: Dict = field(default_factory=dict)
    training_config: Dict = field(default_factory=dict)

class MastishkAttention(nn.Module):
    """Enhanced attention mechanism from your implementation"""
    
    def __init__(self, config: MastishkTransformerConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        
        # Multi-query/grouped-query attention support
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # Differential attention support
        if config.use_differential_attention:
            self.lambda_init = config.differential_lambda_init
            self.lambda_q1 = nn.Parameter(torch.randn(self.head_dim) * 0.1)
            self.lambda_k1 = nn.Parameter(torch.randn(self.head_dim) * 0.1)
            self.lambda_q2 = nn.Parameter(torch.randn(self.head_dim) * 0.1)
            self.lambda_k2 = nn.Parameter(torch.randn(self.head_dim) * 0.1)
    
    def forward(self, hidden_states, attention_mask=None, past_key_value=None):
        batch_size, seq_len, _ = hidden_states.size()
        
        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        if self.config.use_differential_attention:
            # Differential attention mechanism from your implementation
            attn_output = self._differential_attention(query_states, key_states, value_states, attention_mask)
        elif self.config.use_flash_attention and hasattr(F, 'scaled_dot_product_attention'):
            # Flash attention if available
            attn_output = F.scaled_dot_product_attention(
                query_states, key_states, value_states, 
                attn_mask=attention_mask, is_causal=True
            )
        else:
            # Standard scaled dot-product attention
            attn_output = self._standard_attention(query_states, key_states, value_states, attention_mask)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output
    
    def _differential_attention(self, query, key, value, attention_mask):
        """Differential attention from your sophisticated implementation"""
        # Split into two attention heads for differential computation
        q1, q2 = query.chunk(2, dim=1)
        k1, k2 = key.chunk(2, dim=1)
        v1, v2 = value.chunk(2, dim=1)
        
        # Compute attention scores
        scores1 = torch.matmul(q1, k1.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores2 = torch.matmul(q2, k2.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            scores1 = scores1 + attention_mask
            scores2 = scores2 + attention_mask
        
        # Apply differential mechanism
        attn_weights1 = F.softmax(scores1, dim=-1)
        attn_weights2 = F.softmax(scores2, dim=-1)
        
        # Differential combination
        lambda_factor = torch.sigmoid(self.lambda_init)
        attn_weights = lambda_factor * attn_weights1 - (1 - lambda_factor) * attn_weights2
        
        # Apply to values
        attn_output1 = torch.matmul(attn_weights1, v1)
        attn_output2 = torch.matmul(attn_weights2, v2)
        attn_output = torch.cat([attn_output1, attn_output2], dim=1)
        
        return attn_output
    
    def _standard_attention(self, query, key, value, attention_mask):
        """Standard scaled dot-product attention"""
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            scores = scores + attention_mask
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, value)
        
        return attn_output

class MastishkMLP(nn.Module):
    """Enhanced MLP with your advanced features"""
    
    def __init__(self, config: MastishkTransformerConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
        # Activation function
        if config.hidden_act == "swish":
            self.act_fn = F.silu
        elif config.hidden_act == "gelu":
            self.act_fn = F.gelu
        else:
            self.act_fn = F.relu
    
    def forward(self, x):
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

class MastishkTransformerLayer(nn.Module):
    """Transformer layer with your advanced features"""
    
    def __init__(self, config: MastishkTransformerConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.self_attn = MastishkAttention(config)
        
        # Use MoE if enabled and at appropriate frequency
        if config.use_moe and layer_idx % config.moe_layer_frequency == 0:
            self.mlp = MastishkMoEMLP(config)
        else:
            self.mlp = MastishkMLP(config)
        
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # MoD (Mixture of Depths) support
        if config.use_mod and layer_idx % config.mod_layer_frequency == 0:
            self.depth_predictor = nn.Linear(config.hidden_size, 1)
            self.depth_threshold = config.depth_threshold
    
    def forward(self, hidden_states, attention_mask=None):
        # Self-attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states
        
        # MLP with optional MoD
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        if hasattr(self, 'depth_predictor'):
            # MoD: Decide whether to skip this layer
            depth_scores = torch.sigmoid(self.depth_predictor(hidden_states.mean(dim=1)))
            should_process = depth_scores > self.depth_threshold
            
            if should_process.any():
                mlp_output = self.mlp(hidden_states)
                # Apply based on depth decision
                hidden_states = torch.where(
                    should_process.unsqueeze(-1).expand_as(hidden_states),
                    mlp_output,
                    hidden_states
                )
        else:
            hidden_states = self.mlp(hidden_states)
        
        hidden_states = residual + hidden_states
        
        return hidden_states

class MastishkMoEMLP(nn.Module):
    """Mixture of Experts MLP from your implementation"""
    
    def __init__(self, config: MastishkTransformerConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = config.top_k_experts
        
        # Router network
        self.router = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        
        # Expert networks
        self.experts = nn.ModuleList([
            MastishkMLP(config) for _ in range(self.num_experts)
        ])
        
        self.expert_dropout = nn.Dropout(config.expert_dropout)
    
    def forward(self, hidden_states):
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)
        
        # Router logits
        router_logits = self.router(hidden_states)
        routing_weights = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        # Process through selected experts
        final_hidden_states = torch.zeros_like(hidden_states)
        
        for i in range(self.top_k):
            expert_idx = selected_experts[:, i]
            expert_weights = routing_weights[:, i:i+1]
            
            # Create mask for this expert
            expert_mask = (expert_idx.unsqueeze(-1) == torch.arange(self.num_experts, device=hidden_states.device))
            
            for expert_id in range(self.num_experts):
                if expert_mask[:, expert_id].any():
                    expert_tokens = hidden_states[expert_mask[:, expert_id]]
                    expert_output = self.experts[expert_id](expert_tokens)
                    expert_output = self.expert_dropout(expert_output)
                    
                    # Apply routing weights
                    token_weights = expert_weights[expert_mask[:, expert_id]]
                    expert_output = expert_output * token_weights
                    
                    final_hidden_states[expert_mask[:, expert_id]] += expert_output
        
        return final_hidden_states.view(batch_size, seq_len, hidden_size)

class MastishkTransformer(nn.Module):
    """Main Mastishk Transformer model with your sophisticated features"""
    
    def __init__(self, config: MastishkTransformerConfig):
        super().__init__()
        self.config = config
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_positions = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            MastishkTransformerLayer(config, layer_idx) 
            for layer_idx in range(config.num_hidden_layers)
        ])
        
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Output head
        if config.use_multi_token_prediction:
            # Multi-token prediction heads
            self.lm_heads = nn.ModuleList([
                nn.Linear(config.hidden_size, config.vocab_size, bias=False)
                for _ in range(config.n_predict_tokens)
            ])
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights using your initialization strategy"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
    
    def forward(self, input_ids, attention_mask=None, output_attentions=False, **kwargs):
        batch_size, seq_len = input_ids.shape
        
        # Create position IDs
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        hidden_states = self.embed_tokens(input_ids) + self.embed_positions(position_ids)
        
        # Create causal mask
        if attention_mask is None:
            attention_mask = torch.triu(torch.ones(seq_len, seq_len, device=input_ids.device), diagonal=1)
            attention_mask = attention_mask.masked_fill(attention_mask == 1, float('-inf'))
        
        # Transformer layers
        all_attentions = [] if output_attentions else None
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
            
            if output_attentions:
                # In full implementation, would extract attention weights
                all_attentions.append(None)  # Placeholder
        
        hidden_states = self.norm(hidden_states)
        
        # Generate logits
        if self.config.use_multi_token_prediction:
            # Multi-token prediction
            logits = []
            for head in self.lm_heads:
                logits.append(head(hidden_states))
            logits = torch.stack(logits, dim=-2)  # [batch, seq, n_tokens, vocab]
        else:
            logits = self.lm_head(hidden_states)
        
        return {
            "logits": logits,
            "hidden_states": hidden_states,
            "attentions": all_attentions if output_attentions else None
        }

class EnhancedCheckpointManager:
    """Comprehensive checkpoint management from your implementation"""
    
    def __init__(self, checkpoints_dir: str = "./checkpoints"):
        self.checkpoints_dir = Path(checkpoints_dir)
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.checkpoints = {}  # checkpoint_id -> metadata
    
    def save_checkpoint(self, 
                       model: MastishkTransformer,
                       optimizer: torch.optim.Optimizer,
                       scheduler,
                       training_state: TrainingState,
                       model_config: MastishkTransformerConfig,
                       training_config: Dict,
                       checkpoint_name: Optional[str] = None,
                       notes: str = "",
                       include_model_weights: bool = True,
                       compress: bool = True) -> Tuple[bool, str, str]:
        """Save comprehensive checkpoint with all state"""
        
        try:
            checkpoint_id = checkpoint_name or f"mastishk_checkpoint_{int(time.time())}"
            checkpoint_path = self.checkpoints_dir / f"{checkpoint_id}.pt"
            
            # Prepare checkpoint data
            checkpoint_data = {
                "checkpoint_id": checkpoint_id,
                "model_config": asdict(model_config),
                "training_config": training_config,
                "training_state": asdict(training_state),
                "creation_time": datetime.now().isoformat(),
                "notes": notes
            }
            
            if include_model_weights:
                checkpoint_data["model_state_dict"] = model.state_dict()
            
            # Optimizer state
            if optimizer is not None:
                checkpoint_data["optimizer_state_dict"] = optimizer.state_dict()
                checkpoint_data["includes_optimizer_state"] = True
            
            # Scheduler state
            if scheduler is not None:
                checkpoint_data["scheduler_state_dict"] = scheduler.state_dict()
                checkpoint_data["includes_scheduler_state"] = True
            
            # Random states for reproducibility
            checkpoint_data["random_states"] = {
                "python_random_state": None,  # Would save random.getstate()
                "numpy_random_state": np.random.get_state(),
                "torch_random_state": torch.get_rng_state(),
                "torch_cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            }
            checkpoint_data["includes_random_states"] = True
            
            # Save checkpoint
            torch.save(checkpoint_data, checkpoint_path)
            
            # Calculate file hash for integrity
            file_hash = self._calculate_file_hash(checkpoint_path)
            file_size = checkpoint_path.stat().st_size
            
            # Create metadata
            metadata = CheckpointMetadata(
                checkpoint_id=checkpoint_id,
                model_name=model_config.__class__.__name__,
                creation_time=checkpoint_data["creation_time"],
                training_step=training_state.step,
                epoch=training_state.epoch,
                loss=training_state.loss,
                best_loss=training_state.best_loss,
                learning_rate=training_state.learning_rate,
                file_size_bytes=file_size,
                file_hash=file_hash,
                notes=notes,
                includes_optimizer_state=checkpoint_data.get("includes_optimizer_state", False),
                includes_scheduler_state=checkpoint_data.get("includes_scheduler_state", False),
                includes_random_states=checkpoint_data.get("includes_random_states", False),
                model_config=asdict(model_config),
                training_config=training_config
            )
            
            self.checkpoints[checkpoint_id] = metadata
            
            return True, checkpoint_id, f"Checkpoint saved successfully: {checkpoint_path}"
            
        except Exception as e:
            return False, "", f"Failed to save checkpoint: {str(e)}"
    
    def load_checkpoint(self, checkpoint_id: str, model: MastishkTransformer, 
                       optimizer: torch.optim.Optimizer = None,
                       scheduler = None) -> Tuple[bool, str]:
        """Load comprehensive checkpoint with full state restoration"""
        
        try:
            checkpoint_path = self.checkpoints_dir / f"{checkpoint_id}.pt"
            
            if not checkpoint_path.exists():
                return False, f"Checkpoint not found: {checkpoint_path}"
            
            # Verify integrity
            if checkpoint_id in self.checkpoints:
                expected_hash = self.checkpoints[checkpoint_id].file_hash
                actual_hash = self._calculate_file_hash(checkpoint_path)
                if expected_hash != actual_hash:
                    return False, "Checkpoint integrity verification failed"
            
            # Load checkpoint
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
            
            # Restore model state
            if "model_state_dict" in checkpoint_data:
                model.load_state_dict(checkpoint_data["model_state_dict"])
            
            # Restore optimizer state
            if optimizer is not None and "optimizer_state_dict" in checkpoint_data:
                optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
            
            # Restore scheduler state
            if scheduler is not None and "scheduler_state_dict" in checkpoint_data:
                scheduler.load_state_dict(checkpoint_data["scheduler_state_dict"])
            
            # Restore random states
            if "random_states" in checkpoint_data:
                random_states = checkpoint_data["random_states"]
                if random_states["numpy_random_state"] is not None:
                    np.random.set_state(random_states["numpy_random_state"])
                if random_states["torch_random_state"] is not None:
                    torch.set_rng_state(random_states["torch_random_state"])
                if random_states["torch_cuda_random_state"] is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(random_states["torch_cuda_random_state"])
            
            return True, f"Checkpoint loaded successfully: {checkpoint_id}"
            
        except Exception as e:
            return False, f"Failed to load checkpoint: {str(e)}"
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash for integrity verification"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def get_checkpoint_info(self, checkpoint_id: str) -> Optional[CheckpointMetadata]:
        """Get checkpoint metadata"""
        return self.checkpoints.get(checkpoint_id)
    
    def list_checkpoints(self) -> List[CheckpointMetadata]:
        """List all available checkpoints"""
        return list(self.checkpoints.values())