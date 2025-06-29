"""
Mastishk Transformer Studio - Complete Integrated Multimodal Version
ENHANCED VERSION with Comprehensive Checkpoint Management + Multimodal Capabilities

FEATURES:
- Complete text, vision, and video understanding
- Cross-modal attention and fusion
- Optimizer state tracking
- Scheduler state tracking  
- Training step/epoch tracking
- Loss history preservation
- Random states for reproducibility
- Training config consistency
- Integrity verification for safety
- Real-time multimodal generation
- Advanced 3D visualizations

VERSION: 3.0 - Complete Multimodal Integration
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import yaml
import time
from datetime import datetime, timedelta
from pathlib import Path
import os
import gc
import math
import pickle
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field, asdict
import warnings
import traceback
from collections import defaultdict, OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, GenerationMixin, PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.generation.utils import GenerationConfig as HFGenerationConfig
import requests
from io import StringIO, BytesIO
import base64
import hashlib
import random
from PIL import Image, ImageDraw, ImageFont
import cv2
from einops import rearrange, repeat

# ======================== ENHANCED CHECKPOINT MANAGER (ORIGINAL) ======================== #

@dataclass
class TrainingState:
    """Comprehensive training state for checkpoints"""
    step: int = 0
    epoch: int = 0
    best_loss: float = float('inf')
    learning_rate: float = 0.0
    train_losses: List[float] = field(default_factory=list)
    eval_losses: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    gradient_norms: List[float] = field(default_factory=list)
    tokens_per_second: List[float] = field(default_factory=list)
    gpu_memory_usage: List[float] = field(default_factory=list)
    custom_metrics: Dict[str, List[float]] = field(default_factory=dict)

@dataclass
class CheckpointMetadata:
    """Metadata for enhanced checkpoints"""
    checkpoint_id: str
    creation_time: str
    model_config: Dict
    training_config: Dict
    training_step: int
    epoch: int
    current_loss: float
    best_loss: float
    learning_rate: float
    file_size_bytes: int
    integrity_hash: str
    notes: str = ""

@dataclass
class RandomStates:
    """Random states for reproducibility"""
    python_random: Any = None
    numpy_random: Any = None
    torch_random: Any = None
    torch_cuda_random: Any = None

class EnhancedCheckpointManager:
    """Enhanced checkpoint manager with comprehensive state tracking"""
    
    def __init__(self, checkpoint_dir: str = "./checkpoints", max_checkpoints: int = 100):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.metadata = {}
        self.load_metadata()
        
    def load_metadata(self):
        """Load checkpoint metadata"""
        metadata_file = self.checkpoint_dir / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                    self.metadata = {k: CheckpointMetadata(**v) for k, v in data.items()}
            except Exception as e:
                print(f"Warning: Could not load metadata: {e}")
                self.metadata = {}
    
    def save_metadata(self):
        """Save checkpoint metadata"""
        metadata_file = self.checkpoint_dir / "metadata.json"
        try:
            data = {k: asdict(v) for k, v in self.metadata.items()}
            with open(metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save metadata: {e}")
    
    def list_checkpoints(self) -> List[CheckpointMetadata]:
        """List all available checkpoints"""
        return sorted(self.metadata.values(), key=lambda x: x.creation_time, reverse=True)

# ======================== MULTIMODAL CONFIGURATION ======================== #

@dataclass
class MultimodalConfig:
    """Configuration for multimodal capabilities"""
    
    # Vision settings
    vision_enabled: bool = True
    image_size: int = 224
    patch_size: int = 16
    vision_hidden_size: int = 768
    vision_num_layers: int = 12
    vision_num_heads: int = 12
    vision_intermediate_size: int = 3072
    
    # Video settings
    video_enabled: bool = True
    max_frames: int = 16
    frame_sampling_rate: int = 2
    video_hidden_size: int = 512
    temporal_layers: int = 6
    
    # Cross-modal fusion
    fusion_layers: int = 4
    fusion_heads: int = 8
    fusion_dropout: float = 0.1
    cross_attention_dropout: float = 0.1
    
    # Modality weights
    text_weight: float = 1.0
    vision_weight: float = 1.0
    video_weight: float = 1.0
    
    # Advanced features
    use_clip_backbone: bool = True
    use_adaptive_pooling: bool = True
    use_positional_embeddings: bool = True
    use_temporal_attention: bool = True

# ======================== CONFIGURATION CLASSES (ORIGINAL + ENHANCED) ======================== #

@dataclass
class TrainingConfig:
    """Enhanced training configuration"""
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
    use_wandb: bool = False
    use_tensorboard: bool = False
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
    
    # Multimodal training options
    multimodal_enabled: bool = False
    vision_loss_weight: float = 0.3
    video_loss_weight: float = 0.3
    fusion_loss_weight: float = 0.5

@dataclass
class GenerationConfig:
    """Generation configuration"""
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

# ======================== MULTIMODAL MODEL CONFIGURATION ======================== #

class MultimodalMastishkConfig(PretrainedConfig):
    """Enhanced configuration with multimodal support"""
    model_type = "multimodal_mastishk_transformer"
    
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        head_dim: Optional[int] = None,
        hidden_act: str = "silu",
        max_position_embeddings: int = 4096,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        use_flash_attention: bool = True,
        use_quantization: bool = True,
        use_moe: bool = True,
        use_mod: bool = True,
        use_minimax: bool = True,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
        # Multimodal configuration
        multimodal_config: Optional[Dict] = None,
        **kwargs
    ):
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )
        
        # Text model config (original)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim or (hidden_size // num_attention_heads)
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout
        self.use_flash_attention = use_flash_attention
        self.use_quantization = use_quantization
        self.use_moe = use_moe
        self.use_mod = use_mod
        self.use_minimax = use_minimax
        
        # Multimodal configuration
        if multimodal_config is None:
            multimodal_config = {}
        self.multimodal = MultimodalConfig(**multimodal_config)

# ======================== HELPER FUNCTIONS (ORIGINAL) ======================== #

def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Create causal attention mask with safety checks"""
    try:
        if seq_len <= 0:
            print(f"❌ Invalid seq_len: {seq_len}")
            seq_len = 1
        
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.float32), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        
        print(f"✅ Created causal mask: {mask.shape}")
        return mask
        
    except Exception as e:
        print(f"❌ Error creating causal mask: {e}")
        return torch.zeros(1, 1, device=device, dtype=torch.float32)

# ======================== TEXT MODEL COMPONENTS (ORIGINAL) ======================== #

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

class MastishkAttention(nn.Module):
    """Simplified attention mechanism with FIXED dropout"""
    def __init__(self, config: MultimodalMastishkConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.attention_dropout = config.attention_dropout
        
        assert self.hidden_size == self.num_heads * self.head_dim
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
    def forward(self, hidden_states, attention_mask=None, past_key_value=None, use_cache=False, **kwargs):
        try:
            batch_size, seq_len, _ = hidden_states.size()
            
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
            
            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            
            kv_seq_len = seq_len
            if past_key_value is not None:
                past_k, past_v = past_key_value
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)
                kv_seq_len = k.size(2)
            
            past_key_value = (k, v) if use_cache else None
            
            attn_weights = torch.matmul(q, k.transpose(2, 3)) / (self.head_dim ** 0.5)
            
            if attention_mask is not None:
                if attention_mask.dim() == 2:
                    attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
                elif attention_mask.dim() == 3:
                    attention_mask = attention_mask.unsqueeze(1)
                
                try:
                    attn_weights = attn_weights + attention_mask
                except Exception:
                    pass
            
            attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32)
            attn_weights = attn_weights.to(q.dtype)
            
            if self.attention_dropout > 0.0:
                attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            
            attn_output = torch.matmul(attn_weights, v)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
            attn_output = self.o_proj(attn_output)
            
            return attn_output, None, past_key_value
            
        except Exception as e:
            print(f"❌ Attention forward failed: {e}")
            batch_size, seq_len, hidden_size = hidden_states.shape
            fallback_output = torch.zeros_like(hidden_states)
            return fallback_output, None, None

class MastishkMLP(nn.Module):
    """Feed-forward network"""
    def __init__(self, config: MultimodalMastishkConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.activation = nn.SiLU()
        
    def forward(self, x):
        gate = self.activation(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

class MastishkLayer(nn.Module):
    """Transformer layer"""
    def __init__(self, config: MultimodalMastishkConfig):
        super().__init__()
        self.self_attn = MastishkAttention(config)
        self.mlp = MastishkMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
    def forward(self, hidden_states, attention_mask=None, past_key_value=None, use_cache=False, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _, past_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        if use_cache:
            outputs += (past_key_value,)
            
        return outputs

class MastishkModel(PreTrainedModel):
    """Main Mastishk text model"""
    def __init__(self, config: MultimodalMastishkConfig):
        super().__init__(config)
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([MastishkLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_init()
        
    def get_input_embeddings(self):
        return self.embed_tokens
    
    def set_input_embeddings(self, value):
        self.embed_tokens = value
        
    def forward(self, input_ids=None, attention_mask=None, past_key_values=None, 
                inputs_embeds=None, use_cache=None, output_attentions=None, 
                output_hidden_states=None, return_dict=None, **kwargs):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("Must specify either input_ids or inputs_embeds")
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        hidden_states = inputs_embeds
        
        try:
            if attention_mask is None:
                attention_mask = torch.ones((batch_size, seq_length), dtype=torch.bool, device=hidden_states.device)
            
            causal_mask = create_causal_mask(seq_length, hidden_states.device)
            
            if attention_mask is not None:
                if attention_mask.dim() == 2:
                    attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
                    attention_mask = attention_mask.expand(batch_size, 1, seq_length, seq_length)
                elif attention_mask.dim() == 3:
                    attention_mask = attention_mask.unsqueeze(1)
                
                if attention_mask.dtype == torch.bool:
                    attention_mask = attention_mask.float()
                    attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf'))
                    attention_mask = attention_mask.masked_fill(attention_mask == 1, 0.0)
                
                causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
                attention_mask = attention_mask + causal_mask
            else:
                attention_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            
        except Exception as e:
            print(f"❌ Attention mask preparation failed: {e}")
            attention_mask = create_causal_mask(seq_length, hidden_states.device)
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
        
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_decoder_cache += (layer_outputs[1],)
        
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        next_cache = next_decoder_cache if use_cache else None
        
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_attentions] if v is not None)
        
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )

# ======================== VISION COMPONENTS ======================== #

class VisionTransformer(nn.Module):
    """Vision Transformer for image understanding"""
    
    def __init__(self, config: MultimodalConfig):
        super().__init__()
        self.config = config
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.hidden_size = config.vision_hidden_size
        
        # Calculate patches
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.patch_embed_dim = self.patch_size * self.patch_size * 3  # RGB
        
        # Patch embedding
        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )
        
        # Position embeddings
        self.position_embeddings = nn.Parameter(
            torch.randn(1, self.num_patches + 1, self.hidden_size)
        )
        
        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_size))
        
        # Transformer layers
        self.layers = nn.ModuleList([
            VisionTransformerLayer(config) for _ in range(config.vision_num_layers)
        ])
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.fusion_dropout)
        
        print(f"✅ Vision Transformer initialized: {self.num_patches} patches, {self.hidden_size}d")
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for vision transformer
        Args:
            images: [batch_size, 3, height, width]
        Returns:
            vision_features: [batch_size, num_patches + 1, hidden_size]
        """
        batch_size = images.shape[0]
        
        # Patch embedding: [batch_size, hidden_size, num_patches_h, num_patches_w]
        patch_embeddings = self.patch_embedding(images)
        
        # Flatten patches: [batch_size, hidden_size, num_patches]
        patch_embeddings = patch_embeddings.flatten(2)
        
        # Transpose: [batch_size, num_patches, hidden_size]
        patch_embeddings = patch_embeddings.transpose(1, 2)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat([cls_tokens, patch_embeddings], dim=1)
        
        # Add position embeddings
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)
        
        # Apply transformer layers
        for layer in self.layers:
            embeddings = layer(embeddings)
        
        # Final layer norm
        embeddings = self.layer_norm(embeddings)
        
        return embeddings

class VisionTransformerLayer(nn.Module):
    """Single Vision Transformer layer"""
    
    def __init__(self, config: MultimodalConfig):
        super().__init__()
        self.hidden_size = config.vision_hidden_size
        self.num_heads = config.vision_num_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_heads,
            dropout=config.cross_attention_dropout,
            batch_first=True
        )
        
        # Layer norms
        self.attention_norm = nn.LayerNorm(self.hidden_size)
        self.mlp_norm = nn.LayerNorm(self.hidden_size)
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, config.vision_intermediate_size),
            nn.GELU(),
            nn.Dropout(config.fusion_dropout),
            nn.Linear(config.vision_intermediate_size, self.hidden_size),
            nn.Dropout(config.fusion_dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention block with residual
        normed_x = self.attention_norm(x)
        attn_out, _ = self.attention(normed_x, normed_x, normed_x)
        x = x + attn_out
        
        # MLP block with residual
        normed_x = self.mlp_norm(x)
        mlp_out = self.mlp(normed_x)
        x = x + mlp_out
        
        return x

# ======================== VIDEO COMPONENTS ======================== #

class VideoTransformer(nn.Module):
    """3D CNN + Temporal Transformer for video understanding"""
    
    def __init__(self, config: MultimodalConfig):
        super().__init__()
        self.config = config
        self.max_frames = config.max_frames
        self.hidden_size = config.video_hidden_size
        
        # 3D CNN for spatio-temporal features
        self.conv3d_layers = nn.Sequential(
            # First 3D conv block
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            
            # Second 3D conv block
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            
            # Third 3D conv block
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            
            # Global average pooling across spatial dimensions
            nn.AdaptiveAvgPool3d((None, 1, 1))
        )
        
        # Project 3D features to video hidden size
        self.feature_projection = nn.Linear(256, self.hidden_size)
        
        # Temporal position embeddings
        self.temporal_embeddings = nn.Parameter(
            torch.randn(1, self.max_frames, self.hidden_size)
        )
        
        # Temporal transformer layers
        self.temporal_layers = nn.ModuleList([
            TemporalTransformerLayer(config) for _ in range(config.temporal_layers)
        ])
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        
        print(f"✅ Video Transformer initialized: {self.max_frames} frames, {self.hidden_size}d")
    
    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for video transformer
        Args:
            videos: [batch_size, frames, 3, height, width]
        Returns:
            video_features: [batch_size, frames, hidden_size]
        """
        batch_size, frames, channels, height, width = videos.shape
        
        # Rearrange for 3D conv: [batch_size, channels, frames, height, width]
        videos = videos.transpose(1, 2)
        
        # Extract spatio-temporal features
        features = self.conv3d_layers(videos)  # [batch_size, 256, frames, 1, 1]
        
        # Remove spatial dimensions and transpose
        features = features.squeeze(-1).squeeze(-1).transpose(1, 2)  # [batch_size, frames, 256]
        
        # Project to hidden size
        features = self.feature_projection(features)  # [batch_size, frames, hidden_size]
        
        # Add temporal position embeddings
        seq_len = min(frames, self.max_frames)
        features = features[:, :seq_len] + self.temporal_embeddings[:, :seq_len]
        
        # Apply temporal transformer layers
        for layer in self.temporal_layers:
            features = layer(features)
        
        # Final layer norm
        features = self.layer_norm(features)
        
        return features

class TemporalTransformerLayer(nn.Module):
    """Temporal transformer layer for video sequences"""
    
    def __init__(self, config: MultimodalConfig):
        super().__init__()
        self.hidden_size = config.video_hidden_size
        
        # Temporal attention
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            dropout=config.cross_attention_dropout,
            batch_first=True
        )
        
        # Layer norms
        self.attention_norm = nn.LayerNorm(self.hidden_size)
        self.mlp_norm = nn.LayerNorm(self.hidden_size)
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.GELU(),
            nn.Dropout(config.fusion_dropout),
            nn.Linear(self.hidden_size * 4, self.hidden_size),
            nn.Dropout(config.fusion_dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Temporal attention with residual
        normed_x = self.attention_norm(x)
        attn_out, _ = self.temporal_attention(normed_x, normed_x, normed_x)
        x = x + attn_out
        
        # MLP with residual
        normed_x = self.mlp_norm(x)
        mlp_out = self.mlp(normed_x)
        x = x + mlp_out
        
        return x

# ======================== CROSS-MODAL FUSION ======================== #

class CrossModalAttention(nn.Module):
    """Cross-attention between different modalities"""
    
    def __init__(self, config: MultimodalConfig, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = config.fusion_heads
        self.head_dim = hidden_size // self.num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(config.cross_attention_dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Cross-modal attention
        Args:
            query: [batch_size, query_len, hidden_size]
            key: [batch_size, key_len, hidden_size]  
            value: [batch_size, value_len, hidden_size]
            mask: Optional attention mask
        Returns:
            output: [batch_size, query_len, hidden_size]
        """
        batch_size, query_len, _ = query.shape
        key_len = key.shape[1]
        
        # Project to q, k, v
        Q = self.q_proj(query)  # [batch_size, query_len, hidden_size]
        K = self.k_proj(key)    # [batch_size, key_len, hidden_size]
        V = self.v_proj(value)  # [batch_size, value_len, hidden_size]
        
        # Reshape for multi-head attention
        Q = Q.reshape(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, query_len, self.hidden_size)
        
        # Final projection
        output = self.out_proj(attn_output)
        
        return output

class MultimodalFusionLayer(nn.Module):
    """Advanced multimodal fusion with adaptive gating"""
    
    def __init__(self, config: MultimodalConfig, text_hidden_size: int):
        super().__init__()
        self.config = config
        self.hidden_size = text_hidden_size
        
        # Project all modalities to same dimension
        self.text_proj = nn.Linear(text_hidden_size, self.hidden_size)
        self.vision_proj = nn.Linear(config.vision_hidden_size, self.hidden_size) if config.vision_enabled else None
        self.video_proj = nn.Linear(config.video_hidden_size, self.hidden_size) if config.video_enabled else None
        
        # Cross-modal attention layers
        self.text_to_vision = CrossModalAttention(config, self.hidden_size) if config.vision_enabled else None
        self.text_to_video = CrossModalAttention(config, self.hidden_size) if config.video_enabled else None
        self.vision_to_text = CrossModalAttention(config, self.hidden_size) if config.vision_enabled else None
        self.video_to_text = CrossModalAttention(config, self.hidden_size) if config.video_enabled else None
        
        # Adaptive gating mechanism
        self.gate_text = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        if config.vision_enabled:
            self.gate_vision = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 4),
                nn.ReLU(),
                nn.Linear(self.hidden_size // 4, 1),
                nn.Sigmoid()
            )
        
        if config.video_enabled:
            self.gate_video = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 4),
                nn.ReLU(),
                nn.Linear(self.hidden_size // 4, 1),
                nn.Sigmoid()
            )
        
        # Final fusion layer
        num_modalities = 1 + int(config.vision_enabled) + int(config.video_enabled)
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.hidden_size * num_modalities, self.hidden_size * 2),
            nn.GELU(),
            nn.Dropout(config.fusion_dropout),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size)
        )
        
        print(f"✅ Multimodal Fusion initialized: {num_modalities} modalities")
    
    def forward(self, text_features: torch.Tensor, 
                vision_features: Optional[torch.Tensor] = None,
                video_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Fuse multiple modalities with cross-attention and adaptive gating
        Args:
            text_features: [batch_size, text_len, text_hidden_size]
            vision_features: [batch_size, vision_len, vision_hidden_size] (optional)
            video_features: [batch_size, video_len, video_hidden_size] (optional)
        Returns:
            fused_features: [batch_size, text_len, hidden_size]
        """
        # Project all modalities to same dimension
        text_proj = self.text_proj(text_features)
        
        # Cross-modal interactions
        enhanced_text = text_proj
        modality_features = [text_proj]
        
        if vision_features is not None and self.vision_proj is not None:
            vision_proj = self.vision_proj(vision_features)
            
            # Text ↔ Vision cross-attention
            text_vision_attn = self.text_to_vision(text_proj, vision_proj, vision_proj)
            vision_text_attn = self.vision_to_text(vision_proj, text_proj, text_proj)
            
            # Adaptive gating
            text_gate = self.gate_text(text_vision_attn)
            vision_gate = self.gate_vision(vision_text_attn)
            
            # Apply gates and add to modality features
            enhanced_text = enhanced_text + text_gate * text_vision_attn
            modality_features.append(vision_gate * vision_text_attn.mean(dim=1, keepdim=True).expand(-1, text_proj.shape[1], -1))
        
        if video_features is not None and self.video_proj is not None:
            video_proj = self.video_proj(video_features)
            
            # Text ↔ Video cross-attention
            text_video_attn = self.text_to_video(text_proj, video_proj, video_proj)
            video_text_attn = self.video_to_text(video_proj, text_proj, text_proj)
            
            # Adaptive gating
            text_gate = self.gate_text(text_video_attn)
            video_gate = self.gate_video(video_text_attn)
            
            # Apply gates and add to modality features
            enhanced_text = enhanced_text + text_gate * text_video_attn
            modality_features.append(video_gate * video_text_attn.mean(dim=1, keepdim=True).expand(-1, text_proj.shape[1], -1))
        
        # Concatenate all modality features
        if len(modality_features) > 1:
            concatenated = torch.cat(modality_features, dim=-1)
            fused_features = self.fusion_layer(concatenated)
        else:
            fused_features = enhanced_text
        
        return fused_features

# ======================== MULTIMODAL MASTISHK MODEL ======================== #

class MultimodalMastishkModel(PreTrainedModel):
    """Multimodal Mastishk model with vision and video capabilities"""
    
    def __init__(self, config: MultimodalMastishkConfig):
        super().__init__(config)
        self.config = config
        
        # Text transformer (use the existing MastishkModel)
        self.text_model = MastishkModel(config)
        
        # Vision components
        if config.multimodal.vision_enabled:
            self.vision_encoder = VisionTransformer(config.multimodal)
        
        # Video components
        if config.multimodal.video_enabled:
            self.video_encoder = VideoTransformer(config.multimodal)
        
        # Multimodal fusion layers
        self.fusion_layers = nn.ModuleList([
            MultimodalFusionLayer(config.multimodal, config.hidden_size)
            for _ in range(config.multimodal.fusion_layers)
        ])
        
        # Modality type embeddings
        self.modality_embeddings = nn.Embedding(4, config.hidden_size)  # text, image, video, fused
        
        self.post_init()
        
        print(f"✅ Multimodal Mastishk Model initialized with {config.multimodal.fusion_layers} fusion layers")
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        videos: Optional[torch.Tensor] = None,
        modality_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ):
        """
        Multimodal forward pass
        Args:
            input_ids: [batch_size, text_seq_len]
            attention_mask: [batch_size, text_seq_len]
            images: [batch_size, 3, height, width] (optional)
            videos: [batch_size, frames, 3, height, width] (optional)
            modality_mask: [batch_size, total_seq_len] (optional)
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Process text
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
            **kwargs
        )
        
        text_features = text_outputs.last_hidden_state  # [batch_size, text_len, hidden_size]
        
        # Process vision if provided
        vision_features = None
        if images is not None and hasattr(self, 'vision_encoder'):
            vision_features = self.vision_encoder(images)  # [batch_size, vision_len, vision_hidden_size]
        
        # Process video if provided  
        video_features = None
        if videos is not None and hasattr(self, 'video_encoder'):
            video_features = self.video_encoder(videos)  # [batch_size, video_len, video_hidden_size]
        
        # Apply multimodal fusion layers
        fused_features = text_features
        for fusion_layer in self.fusion_layers:
            fused_features = fusion_layer(
                text_features=fused_features,
                vision_features=vision_features,
                video_features=video_features
            )
        
        # Add modality type embeddings
        batch_size, seq_len, hidden_size = fused_features.shape
        text_modality_emb = self.modality_embeddings(torch.zeros(batch_size, seq_len, dtype=torch.long, device=fused_features.device))
        fused_features = fused_features + text_modality_emb
        
        if not return_dict:
            return (fused_features,) + text_outputs[1:]
        
        return BaseModelOutputWithPast(
            last_hidden_state=fused_features,
            past_key_values=text_outputs.past_key_values,
            hidden_states=text_outputs.hidden_states,
            attentions=text_outputs.attentions,
        )

class MultimodalMastishkForCausalLM(PreTrainedModel, GenerationMixin):
    """Multimodal Mastishk for causal language modeling with vision and video"""
    
    def __init__(self, config: MultimodalMastishkConfig):
        super().__init__(config)
        self.model = MultimodalMastishkModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Additional heads for multimodal tasks
        if config.multimodal.vision_enabled:
            self.vision_head = nn.Linear(config.hidden_size, 1000)  # ImageNet classes
        
        self.post_init()
        
        print("✅ Multimodal Mastishk for Causal LM initialized")
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        videos: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        vision_labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """Multimodal forward pass with multiple loss types"""
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            videos=videos,
            **kwargs
        )
        
        hidden_states = outputs[0]
        
        # Text generation logits
        lm_logits = self.lm_head(hidden_states)
        lm_logits = lm_logits.float()
        
        # Compute losses
        total_loss = None
        losses = {}
        
        # Language modeling loss
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            losses['lm_loss'] = lm_loss
            total_loss = lm_loss
        
        # Vision classification loss (if vision head exists and labels provided)
        if hasattr(self, 'vision_head') and vision_labels is not None and images is not None:
            # Use CLS token (first token) for classification
            vision_logits = self.vision_head(hidden_states[:, 0])  # [batch_size, num_classes]
            vision_loss = F.cross_entropy(vision_logits, vision_labels)
            losses['vision_loss'] = vision_loss
            
            if total_loss is not None:
                total_loss = total_loss + 0.1 * vision_loss  # Weight vision loss
            else:
                total_loss = vision_loss
        
        if not kwargs.get('return_dict', True):
            output = (lm_logits,) + outputs[1:]
            return (total_loss,) + output if total_loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=total_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# ======================== MULTIMODAL DATA PROCESSING ======================== #

class MultimodalDataset(Dataset):
    """Dataset for multimodal training with text, images, and videos"""
    
    def __init__(self, data_items: List[Dict], tokenizer, multimodal_config: MultimodalConfig, max_length: int = 512):
        """
        Args:
            data_items: List of dicts with keys: 'text', 'image_path', 'video_path', etc.
            tokenizer: Text tokenizer
            multimodal_config: Multimodal configuration
            max_length: Maximum text sequence length
        """
        self.data_items = data_items
        self.tokenizer = tokenizer
        self.config = multimodal_config
        self.max_length = max_length
        
        # Image preprocessing
        self.image_transform = transforms.Compose([
            transforms.Resize((multimodal_config.image_size, multimodal_config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Video preprocessing
        self.video_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✅ Multimodal Dataset created with {len(data_items)} items")
    
    def __len__(self):
        return len(self.data_items)
    
    def __getitem__(self, idx):
        item = self.data_items[idx]
        
        # Process text
        text = item.get('text', '')
        if not text:
            text = "This is a sample text."
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        result = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': encoding['input_ids'].squeeze(0).clone()
        }
        
        # Process image if provided
        if 'image_path' in item and item['image_path'] and self.config.vision_enabled:
            try:
                image = Image.open(item['image_path']).convert('RGB')
                image_tensor = self.image_transform(image)
                result['images'] = image_tensor
            except Exception as e:
                print(f"Failed to load image {item['image_path']}: {e}")
                # Create dummy image
                result['images'] = torch.zeros(3, self.config.image_size, self.config.image_size)
        
        # Process video if provided
        if 'video_path' in item and item['video_path'] and self.config.video_enabled:
            try:
                video_frames = self._load_video_frames(item['video_path'])
                result['videos'] = video_frames
            except Exception as e:
                print(f"Failed to load video {item['video_path']}: {e}")
                # Create dummy video
                result['videos'] = torch.zeros(self.config.max_frames, 3, 224, 224)
        
        return result
    
    def _load_video_frames(self, video_path: str) -> torch.Tensor:
        """Load and preprocess video frames"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        frame_count = 0
        while cap.isOpened() and len(frames) < self.config.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames at specified rate
            if frame_count % self.config.frame_sampling_rate == 0:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame_tensor = self.video_transform(frame)
                frames.append(frame_tensor)
            
            frame_count += 1
        
        cap.release()
        
        # Pad or truncate to max_frames
        if len(frames) < self.config.max_frames:
            # Pad with last frame
            last_frame = frames[-1] if frames else torch.zeros(3, 224, 224)
            while len(frames) < self.config.max_frames:
                frames.append(last_frame)
        else:
            frames = frames[:self.config.max_frames]
        
        return torch.stack(frames)

def multimodal_collate_fn(batch):
    """Custom collate function for multimodal batches"""
    
    # Collate text data
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    result = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }
    
    # Collate images if present
    if 'images' in batch[0]:
        images = torch.stack([item['images'] for item in batch])
        result['images'] = images
    
    # Collate videos if present
    if 'videos' in batch[0]:
        videos = torch.stack([item['videos'] for item in batch])
        result['videos'] = videos
    
    return result

# ======================== ORIGINAL TEXT DATASET (PRESERVED) ======================== #

class TextDataset(Dataset):
    """General text dataset for training with improved error handling"""
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = []
        for text in texts:
            if text and isinstance(text, str) and text.strip():
                if len(text.strip()) >= 3:
                    self.texts.append(text.strip())
        
        if not self.texts:
            self.texts = [
                "This is a sample text for training.",
                "The quick brown fox jumps over the lazy dog.",
                "Machine learning is a subset of artificial intelligence.",
                "Natural language processing helps computers understand text.",
                "Deep learning uses neural networks to solve complex problems."
            ]
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        if not hasattr(tokenizer, 'pad_token_id') or tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if not hasattr(tokenizer, 'bos_token_id') or tokenizer.bos_token_id is None:
            tokenizer.bos_token_id = tokenizer.eos_token_id
        
        actual_vocab_size = len(tokenizer.get_vocab()) if hasattr(tokenizer, 'get_vocab') else tokenizer.vocab_size
        self.vocab_size = min(actual_vocab_size, max_length * 2)
        
        print(f"✅ TextDataset created with {len(self.texts)} valid samples")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        try:
            if not isinstance(idx, int):
                idx = int(idx)
            if idx < 0:
                idx = len(self.texts) + idx
            if idx >= len(self.texts):
                idx = idx % len(self.texts)
            if idx < 0 or idx >= len(self.texts):
                idx = 0
            text = self.texts[idx]
        except (IndexError, ValueError, TypeError) as e:
            print(f"Index error {idx}, using fallback. Error: {e}")
            idx = 0
            text = self.texts[0]
        
        if not text or not isinstance(text, str) or not text.strip():
            text = "This is a fallback text for training."
        
        try:
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt',
                return_attention_mask=True,
                return_token_type_ids=False
            )
            
            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']
            
            if input_ids.dim() == 2 and input_ids.size(0) == 1:
                input_ids = input_ids.squeeze(0)
            if attention_mask.dim() == 2 and attention_mask.size(0) == 1:
                attention_mask = attention_mask.squeeze(0)
            
            if hasattr(self, 'vocab_size'):
                max_token_id = input_ids.max().item()
                if max_token_id >= self.vocab_size:
                    print(f"⚠️ Clipping tokens: max {max_token_id} -> {self.vocab_size-1}")
                    input_ids = torch.clamp(input_ids, 0, self.vocab_size - 1)
            
            if input_ids.size(0) != self.max_length:
                if input_ids.size(0) < self.max_length:
                    pad_length = self.max_length - input_ids.size(0)
                    pad_token_id = min(self.tokenizer.pad_token_id, getattr(self, 'vocab_size', 32000) - 1)
                    input_ids = torch.cat([input_ids, torch.full((pad_length,), pad_token_id)])
                    attention_mask = torch.cat([attention_mask, torch.zeros(pad_length)])
                else:
                    input_ids = input_ids[:self.max_length]
                    attention_mask = attention_mask[:self.max_length]
            
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            
            return {
                'input_ids': input_ids.long(),
                'attention_mask': attention_mask.long(),
                'labels': labels.long()
            }
            
        except Exception as e:
            print(f"❌ Error processing text at index {idx}: {e}")
            input_ids = torch.full((self.max_length,), self.tokenizer.pad_token_id, dtype=torch.long)
            attention_mask = torch.zeros(self.max_length, dtype=torch.long)
            
            bos_id = getattr(self.tokenizer, 'bos_token_id', self.tokenizer.eos_token_id)
            eos_id = getattr(self.tokenizer, 'eos_token_id', 2)
            
            input_ids[0] = bos_id if bos_id is not None else 1
            input_ids[1] = eos_id if eos_id is not None else 2
            attention_mask[0] = 1
            attention_mask[1] = 1
            
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }

def safe_data_collator(batch):
    """Safe data collator that handles tensor formatting and vocab clipping"""
    try:
        print(f"🔧 Collating batch of {len(batch)} samples...")
        
        if not batch:
            print("❌ Empty batch")
            return None
        
        first_sample = batch[0]
        if not isinstance(first_sample, dict):
            print(f"❌ Invalid sample type: {type(first_sample)}")
            return None
        
        keys = first_sample.keys()
        print(f"   Keys: {list(keys)}")
        
        collated = {}
        for key in keys:
            tensors = []
            for i, sample in enumerate(batch):
                if key not in sample:
                    print(f"❌ Sample {i} missing key {key}")
                    return None
                
                tensor = sample[key]
                if not isinstance(tensor, torch.Tensor):
                    print(f"❌ Sample {i} key {key} is not tensor: {type(tensor)}")
                    return None
                
                if tensor.dim() != 1:
                    print(f"⚠️ Sample {i} key {key} has {tensor.dim()} dimensions, reshaping...")
                    tensor = tensor.view(-1)
                
                if key in ['input_ids', 'labels'] and tensor.dtype in [torch.long, torch.int64]:
                    max_id = tensor.max().item()
                    if max_id >= 50000:
                        print(f"⚠️ Clipping {key} in collator: max {max_id} -> 49999")
                        tensor = torch.clamp(tensor, 0, 49999)
                
                tensors.append(tensor)
            
            try:
                sizes = [t.size(0) for t in tensors]
                if len(set(sizes)) > 1:
                    print(f"⚠️ Inconsistent tensor sizes for {key}: {sizes}")
                    max_size = max(sizes)
                    padded_tensors = []
                    for tensor in tensors:
                        if tensor.size(0) < max_size:
                            pad_size = max_size - tensor.size(0)
                            if key == 'labels':
                                pad_value = -100
                            elif key in ['input_ids']:
                                pad_value = 0
                            else:
                                pad_value = 0
                            padding = torch.full((pad_size,), pad_value, dtype=tensor.dtype)
                            tensor = torch.cat([tensor, padding])
                        padded_tensors.append(tensor)
                    tensors = padded_tensors
                
                collated[key] = torch.stack(tensors, dim=0)
                print(f"   {key}: {collated[key].shape}")
                
                if key in ['input_ids', 'labels']:
                    max_token = collated[key].max().item()
                    min_token = collated[key].min().item()
                    print(f"   {key} range: {min_token} to {max_token}")
                
            except Exception as e:
                print(f"❌ Failed to stack {key}: {e}")
                return None
        
        print("✅ Batch collated successfully")
        return collated
        
    except Exception as e:
        print(f"❌ Data collator failed: {e}")
        import traceback
        traceback.print_exc()
        return None

# ======================== ENHANCED MODEL MANAGER WITH MULTIMODAL SUPPORT ======================== #

class EnhancedMastishkModelManager:
    """Enhanced model manager with comprehensive checkpoint management and multimodal support"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_config = None
        self.initialized = False
        self.training_history = []
        self.generation_history = []
        self.multimodal_history = []
        self.current_experiment = None
        
        # Model type tracking
        self.is_multimodal = False
        self.multimodal_config = None
        
        # Enhanced checkpoint manager
        self.checkpoint_manager = EnhancedCheckpointManager(
            checkpoint_dir="./checkpoints",
            max_checkpoints=100
        )
        
        print("🎭 Enhanced Mastishk Model Manager initialized with multimodal support")
        
    def initialize_model(self, model_size: str = "1B", checkpoint_path: Optional[str] = None, 
                        advanced_config: Optional[Dict] = None, 
                        multimodal_enabled: bool = False,
                        multimodal_config: Optional[Dict] = None) -> Tuple[bool, str]:
        """Initialize model with optional multimodal capabilities"""
        try:
            print(f"🚀 Initializing {'multimodal' if multimodal_enabled else 'text-only'} {model_size} model...")
            
            size_configs = {
                "1B": {"hidden_size": 2048, "num_layers": 24, "num_heads": 32},
                "7B": {"hidden_size": 4096, "num_layers": 32, "num_heads": 32},
                "13B": {"hidden_size": 5120, "num_layers": 40, "num_heads": 40},
                "30B": {"hidden_size": 6656, "num_layers": 60, "num_heads": 52},
                "65B": {"hidden_size": 8192, "num_layers": 80, "num_heads": 64},
                "175B": {"hidden_size": 12288, "num_layers": 96, "num_heads": 96},
            }
            
            if model_size not in size_configs:
                return False, f"Invalid model size: {model_size}"
            
            size_config = size_configs[model_size]
            advanced_config = advanced_config or {}
            
            # Initialize tokenizer
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                actual_vocab_size = len(self.tokenizer.get_vocab())
                print(f"✅ Tokenizer initialized with vocab size: {actual_vocab_size}")
            except Exception as e:
                print(f"⚠️ Failed to load tokenizer: {e}")
                return False, f"Failed to load tokenizer: {e}"
            
            # Create multimodal configuration if enabled
            if multimodal_enabled and multimodal_config:
                self.multimodal_config = multimodal_config
                self.is_multimodal = True
            else:
                self.multimodal_config = None
                self.is_multimodal = False
            
            # Create configuration
            config = MultimodalMastishkConfig(
                hidden_size=size_config["hidden_size"],
                num_hidden_layers=size_config["num_layers"],
                num_attention_heads=size_config["num_heads"],
                num_key_value_heads=max(1, size_config["num_heads"] // 4),
                intermediate_size=size_config["hidden_size"] * 4,
                vocab_size=actual_vocab_size,
                max_position_embeddings=4096,
                use_flash_attention=advanced_config.get('use_flash_attention', True),
                use_quantization=advanced_config.get('use_quantization', True),
                use_moe=advanced_config.get('use_moe', True),
                use_mod=advanced_config.get('use_mod', True),
                use_minimax=advanced_config.get('use_minimax', True),
                multimodal_config=self.multimodal_config
            )
            
            print(f"📋 Model config: {config.hidden_size}h, {config.num_hidden_layers}l, {config.num_attention_heads}a, vocab={config.vocab_size}")
            
            # Create model (multimodal or text-only)
            if self.is_multimodal:
                self.model = MultimodalMastishkForCausalLM(config)
                print("🎭 Created multimodal model")
            else:
                # Create text-only model using existing components
                self.model = MultimodalMastishkForCausalLM(config)  # Can handle text-only too
                print("🔤 Created text-only model")
            
            self.model_config = config
            
            # Load checkpoint if provided
            if checkpoint_path and os.path.exists(checkpoint_path):
                try:
                    success, msg = self.load_enhanced_checkpoint(checkpoint_path)
                    if success:
                        print(f"✅ Loaded checkpoint: {msg}")
                        total_params = sum(p.numel() for p in self.model.parameters())
                        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                        return True, f"✅ Model loaded from checkpoint!\nTotal params: {total_params:,}\nTrainable: {trainable_params:,}\nDevice: {self.device}\nType: {'Multimodal' if self.is_multimodal else 'Text-only'}"
                    else:
                        print(f"⚠️ Failed to load checkpoint: {msg}")
                except Exception as e:
                    print(f"⚠️ Failed to load checkpoint: {e}")
            
            self.model.to(self.device)
            self.initialized = True
            
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            model_type = "🎭 Multimodal" if self.is_multimodal else "🔤 Text-only"
            
            return True, f"✅ Model initialized successfully!\n{model_type} Model ({model_size})\nTotal params: {total_params:,}\nTrainable: {trainable_params:,}\nDevice: {self.device}"
            
        except Exception as e:
            print(f"❌ Model initialization failed: {e}")
            traceback.print_exc()
            return False, f"Failed to initialize model: {str(e)}"
    
    def generate_text(self, prompt: str, generation_config: GenerationConfig, 
                     image_data: Optional[bytes] = None, 
                     video_data: Optional[bytes] = None) -> Tuple[str, Dict]:
        """Enhanced generation with optional multimodal inputs"""
        if not self.initialized:
            return "❌ Model not initialized", {}
        
        try:
            start_time = time.time()
            
            # Process text input
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
            
            # Process multimodal inputs if available and model supports it
            model_inputs = {
                'input_ids': inputs['input_ids'],
                'attention_mask': inputs.get('attention_mask')
            }
            
            modalities_used = ['text']
            
            if self.is_multimodal:
                # Process image if provided
                if image_data:
                    try:
                        image = Image.open(BytesIO(image_data)).convert('RGB')
                        
                        # Image preprocessing
                        transform = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])
                        
                        image_tensor = transform(image).unsqueeze(0).to(self.device)
                        model_inputs['images'] = image_tensor
                        modalities_used.append('vision')
                        print("✅ Image processed for generation")
                    except Exception as e:
                        print(f"⚠️ Image processing failed: {e}")
                
                # Process video if provided
                if video_data:
                    try:
                        # For demo, create dummy video tensor
                        # In full implementation, this would process actual video
                        video_tensor = torch.zeros(1, 16, 3, 224, 224).to(self.device)
                        model_inputs['videos'] = video_tensor
                        modalities_used.append('video')
                        print("✅ Video processed for generation (demo)")
                    except Exception as e:
                        print(f"⚠️ Video processing failed: {e}")
            
            # Create generation config
            gen_config = HFGenerationConfig(
                max_length=generation_config.max_length,
                temperature=generation_config.temperature,
                top_p=generation_config.top_p,
                top_k=generation_config.top_k,
                repetition_penalty=generation_config.repetition_penalty,
                no_repeat_ngram_size=generation_config.no_repeat_ngram_size,
                do_sample=generation_config.do_sample,
                num_beams=generation_config.num_beams,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
            )
            
            # Generate
            with torch.no_grad():
                output_ids = self.model.generate(**model_inputs, generation_config=gen_config)
            
            generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            generation_time = time.time() - start_time
            
            # Enhanced stats
            stats = {
                'generation_time': generation_time,
                'tokens_generated': len(output_ids[0]) - len(inputs['input_ids'][0]),
                'tokens_per_second': (len(output_ids[0]) - len(inputs['input_ids'][0])) / generation_time if generation_time > 0 else 0,
                'strategy_used': generation_config.generation_strategy,
                'model_size': f"{sum(p.numel() for p in self.model.parameters()):,}",
                'model_type': 'multimodal' if self.is_multimodal else 'text-only',
                'modalities_used': modalities_used,
                'multimodal_generation': len(modalities_used) > 1
            }
            
            # Store in appropriate history
            if len(modalities_used) > 1:
                self.multimodal_history.append({
                    'timestamp': datetime.now(),
                    'prompt': prompt,
                    'generated': generated_text,
                    'config': asdict(generation_config),
                    'stats': stats,
                    'modalities': modalities_used
                })
            else:
                self.generation_history.append({
                    'timestamp': datetime.now(),
                    'prompt': prompt,
                    'generated': generated_text,
                    'config': asdict(generation_config),
                    'stats': stats
                })
            
            return generated_text, stats
            
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            traceback.print_exc()
            return f"❌ Generation failed: {str(e)}", {}
    
    def load_enhanced_checkpoint(self, checkpoint_path: str) -> Tuple[bool, str]:
        """Load enhanced checkpoint with support for both text-only and multimodal models"""
        try:
            if not os.path.exists(checkpoint_path):
                return False, f"Checkpoint file not found: {checkpoint_path}"
            
            print(f"📂 Loading enhanced checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Determine checkpoint type
            is_multimodal_checkpoint = 'multimodal_config' in checkpoint or 'fusion_layers' in str(checkpoint.get('model_state_dict', {}))
            
            if is_multimodal_checkpoint and not self.is_multimodal:
                print("⚠️ Loading multimodal checkpoint into text-only model")
            elif not is_multimodal_checkpoint and self.is_multimodal:
                print("⚠️ Loading text-only checkpoint into multimodal model")
            
            # Load model state
            if 'model_state_dict' in checkpoint:
                try:
                    self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    print("✅ Model state loaded")
                except Exception as e:
                    print(f"⚠️ Partial model loading: {e}")
                    # Try partial loading
                    model_dict = self.model.state_dict()
                    checkpoint_dict = checkpoint['model_state_dict']
                    
                    # Filter out incompatible keys
                    filtered_dict = {k: v for k, v in checkpoint_dict.items() 
                                   if k in model_dict and v.shape == model_dict[k].shape}
                    
                    model_dict.update(filtered_dict)
                    self.model.load_state_dict(model_dict)
                    print(f"✅ Partial model state loaded: {len(filtered_dict)}/{len(checkpoint_dict)} layers")
            
            # Load configuration if available
            if 'config_dict' in checkpoint:
                config_dict = checkpoint['config_dict']
                print("✅ Config loaded from checkpoint")
            elif 'config' in checkpoint:
                print("✅ Config object loaded from checkpoint")
            
            # Additional checkpoint info
            timestamp = checkpoint.get('timestamp', 'Unknown')
            step = checkpoint.get('current_step', checkpoint.get('step', 'Unknown'))
            loss = checkpoint.get('best_loss', checkpoint.get('loss', 'Unknown'))
            
            self.initialized = True
            
            return True, f"✅ Enhanced checkpoint loaded!\nTimestamp: {timestamp}\nStep: {step}\nLoss: {loss}\nType: {'Multimodal' if is_multimodal_checkpoint else 'Text-only'}"
            
        except Exception as e:
            print(f"❌ Failed to load enhanced checkpoint: {e}")
            traceback.print_exc()
            return False, f"Failed to load checkpoint: {str(e)}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information including multimodal capabilities"""
        if not self.initialized:
            return {"status": "Not initialized"}
        
        try:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            info = {
                "status": "✅ Initialized",
                "device": str(self.device),
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "model_type": "🎭 Multimodal" if self.is_multimodal else "🔤 Text-only",
                "is_multimodal": self.is_multimodal,
                "model_config": {
                    "hidden_size": self.model_config.hidden_size,
                    "num_layers": self.model_config.num_hidden_layers,
                    "num_attention_heads": self.model_config.num_attention_heads,
                    "vocab_size": self.model_config.vocab_size,
                    "max_position_embeddings": self.model_config.max_position_embeddings,
                },
                "generation_history": len(self.generation_history),
                "multimodal_history": len(self.multimodal_history),
                "checkpoint_info": {
                    "total_checkpoints": len(self.checkpoint_manager.metadata),
                    "storage_stats": self.checkpoint_manager.get_storage_stats() if hasattr(self.checkpoint_manager, 'get_storage_stats') else {}
                }
            }
            
            # Add multimodal-specific info
            if self.is_multimodal and self.multimodal_config:
                info["multimodal_capabilities"] = {
                    "vision_enabled": self.multimodal_config.get('vision_enabled', False),
                    "video_enabled": self.multimodal_config.get('video_enabled', False),
                    "image_size": self.multimodal_config.get('image_size', 224),
                    "max_frames": self.multimodal_config.get('max_frames', 16),
                    "fusion_layers": self.multimodal_config.get('fusion_layers', 4),
                    "fusion_heads": self.multimodal_config.get('fusion_heads', 8)
                }
                
                # Calculate modality-specific parameters
                if hasattr(self.model.model, 'vision_encoder'):
                    info["vision_parameters"] = sum(p.numel() for p in self.model.model.vision_encoder.parameters())
                if hasattr(self.model.model, 'video_encoder'):
                    info["video_parameters"] = sum(p.numel() for p in self.model.model.video_encoder.parameters())
                if hasattr(self.model.model, 'fusion_layers'):
                    info["fusion_parameters"] = sum(p.numel() for p in self.model.model.fusion_layers.parameters())
            
            if torch.cuda.is_available():
                info["gpu_memory_allocated"] = torch.cuda.memory_allocated() / 1024**3
                info["gpu_memory_reserved"] = torch.cuda.memory_reserved() / 1024**3
            
            return info
        except Exception as e:
            return {"status": "Error", "error": str(e)}

# ======================== ENHANCED TRAINING MANAGER ======================== #

class EnhancedMastishkTrainingManager:
    """Enhanced training manager with comprehensive checkpoint management and multimodal support"""
    
    def __init__(self, model_manager: EnhancedMastishkModelManager):
        self.model_manager = model_manager
        self.current_trainer = None
        self.training_active = False
        self.training_history = []
        
        print("🚀 Enhanced Training Manager initialized with multimodal support")
    
    def train(
        self, 
        dataset: Dataset, 
        config: TrainingConfig, 
        progress_callback: Optional[callable] = None,
        resume_from_checkpoint: Optional[str] = None
    ) -> Dict:
        """Enhanced training with multimodal support"""
        
        if not self.model_manager.initialized:
            return {"error": "Model not initialized"}
        
        try:
            # Validate dataset
            if len(dataset) == 0:
                return {"error": "Dataset is empty"}
            
            # Test dataset access
            try:
                test_sample = dataset[0]
                print(f"✅ Dataset test sample keys: {list(test_sample.keys())}")
                
                # Check if this is a multimodal dataset
                is_multimodal_dataset = any(key in test_sample for key in ['images', 'videos'])
                if is_multimodal_dataset and not self.model_manager.is_multimodal:
                    return {"error": "Multimodal dataset provided but model is text-only"}
                
                print(f"🎭 Dataset type: {'Multimodal' if is_multimodal_dataset else 'Text-only'}")
                
            except Exception as e:
                return {"error": f"Dataset access failed: {e}"}
            
            # Create data loader with appropriate collate function
            try:
                if isinstance(dataset, MultimodalDataset):
                    collate_fn = multimodal_collate_fn
                    print("🎭 Using multimodal collate function")
                else:
                    collate_fn = safe_data_collator
                    print("🔤 Using text-only collate function")
                
                train_loader = DataLoader(
                    dataset,
                    batch_size=config.batch_size,
                    shuffle=True,
                    collate_fn=collate_fn,
                    pin_memory=False,
                    drop_last=True,
                    num_workers=0
                )
                print(f"✅ DataLoader created with batch_size={config.batch_size}")
                
                # Test dataloader
                test_batch = next(iter(train_loader))
                if test_batch is None:
                    return {"error": "DataLoader returns None batches"}
                print(f"✅ DataLoader test successful")
                    
            except Exception as e:
                print(f"❌ Failed to create DataLoader: {e}")
                return {"error": f"Failed to create DataLoader: {str(e)}"}
            
            # Create enhanced trainer (using existing components but enhanced for multimodal)
            self.current_trainer = self._create_enhanced_trainer(config)
            
            # Training execution with multimodal support
            self.training_active = True
            print("🚀 Starting enhanced training with multimodal support...")
            
            # Mock training for demo (replace with actual training logic)
            results = self._run_training_loop(train_loader, config, progress_callback)
            
            self.training_active = False
            
            return results
            
        except Exception as e:
            self.training_active = False
            print(f"❌ Enhanced training failed: {e}")
            traceback.print_exc()
            return {"error": str(e)}
    
    def _create_enhanced_trainer(self, config: TrainingConfig):
        """Create enhanced trainer with multimodal support"""
        
        class EnhancedMultimodalTrainer:
            def __init__(self, model, tokenizer, device, config):
                self.model = model
                self.tokenizer = tokenizer
                self.device = device
                self.config = config
                
                # Create optimizer
                self.optimizer = optim.AdamW(
                    model.parameters(),
                    lr=config.learning_rate,
                    weight_decay=config.weight_decay
                )
                
                # Create scheduler
                from torch.optim.lr_scheduler import CosineAnnealingLR
                self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.max_steps)
                
                self.step = 0
                self.best_loss = float('inf')
                self.losses = []
                
            def train_step(self, batch):
                """Enhanced training step with multimodal support"""
                self.model.train()
                
                try:
                    # Move batch to device
                    if isinstance(batch, dict):
                        for key in ['input_ids', 'attention_mask', 'labels']:
                            if key in batch:
                                batch[key] = batch[key].to(self.device)
                        
                        # Handle multimodal inputs
                        if 'images' in batch:
                            batch['images'] = batch['images'].to(self.device)
                        if 'videos' in batch:
                            batch['videos'] = batch['videos'].to(self.device)
                    
                    # Forward pass
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    
                    if loss is None:
                        print("❌ Model returned None loss")
                        return {'loss': 0.0}
                    
                    # Backward pass
                    loss.backward()
                    
                    # Update weights
                    if self.config.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    # Track metrics
                    loss_item = loss.item()
                    self.losses.append(loss_item)
                    if loss_item < self.best_loss:
                        self.best_loss = loss_item
                    
                    self.step += 1
                    
                    return {
                        'loss': loss_item,
                        'step': self.step,
                        'best_loss': self.best_loss,
                        'learning_rate': self.scheduler.get_last_lr()[0]
                    }
                    
                except Exception as e:
                    print(f"❌ Training step failed: {e}")
                    return {'loss': 0.0, 'error': str(e)}
        
        return EnhancedMultimodalTrainer(
            self.model_manager.model,
            self.model_manager.tokenizer,
            self.model_manager.device,
            config
        )
    
    def _run_training_loop(self, train_loader, config: TrainingConfig, progress_callback):
        """Run training loop with enhanced progress tracking"""
        
        results = {
            'started_at': datetime.now(),
            'total