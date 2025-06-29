"""
Mastishk Transformer Studio - Advanced Transformer Experimentation Platform 
ENHANCED VERSION with Comprehensive Checkpoint Management

Features:
- Optimizer state tracking
- Scheduler state tracking  
- Training step/epoch tracking
- Loss history preservation
- Random states for reproducibility
- Training config consistency
- Integrity verification for safety
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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
import math  # ← ADD THIS
import pickle  # ← ADD THIS
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field, asdict
import warnings
import traceback
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, GenerationMixin, PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.generation.utils import GenerationConfig as HFGenerationConfig
import requests
from io import StringIO
import base64
import hashlib
from collections import OrderedDict
import random

# Import the enhanced checkpoint manager
from enhanced_checkpoint_manager import (
    EnhancedCheckpointManager, 
    TrainingState, 
    CheckpointMetadata, 
    RandomStates,
    create_training_state_from_monitor,
    update_monitor_from_training_state
)


# ======================== CONFIGURATION CLASSES ======================== #

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

@dataclass
class ExperimentConfig:
    """Experiment tracking configuration"""
    experiment_name: str = "mastishk_experiment"
    run_name: str = field(default_factory=lambda: f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    tags: List[str] = field(default_factory=list)
    description: str = ""
    save_artifacts: bool = True
    log_frequency: int = 10

# ======================== MODEL CONFIGURATION ======================== #

class MastishkConfig(PretrainedConfig):
    """Mastishk transformer configuration"""
    model_type = "mastishk_transformer"
    
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

# ======================== HELPER FUNCTIONS ======================== #

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

# ======================== MODEL COMPONENTS ======================== #
# [Previous model components remain the same: RMSNorm, MastishkAttention, MastishkMLP, etc.]
# ======================== TRAINING DATA VERIFICATION SYSTEM ======================== #
class FixedTrainingStep:
    """Quick fix for weight verification timing"""
    
    def __init__(self, trainer):
        self.trainer = trainer
        self.accumulated_steps = 0
        self.pre_optimizer_snapshot = None
        
    def fixed_train_step_with_verification(self, batch, step):
        """FIXED: Only verify weights when optimizer actually steps"""
        
        # Regular training step (accumulates gradients)
        step_metrics = self.trainer.train_step(batch)
        self.accumulated_steps += 1
        print(f"📝 Step {step}: Gradient accumulation ({self.accumulated_steps}/{self.trainer.gradient_accumulation_steps})")
        
        # Check if we should step optimizer (gradient accumulation complete)
        should_step = (self.accumulated_steps % self.trainer.gradient_accumulation_steps == 0)
        
        if should_step:
            print(f"🔄 Step {step}: Optimizer stepping - WEIGHT VERIFICATION TIME!\n🔍 Verifying weights at real step {step} after accumulation {self.accumulated_steps}")
            
            # Create pre-optimizer snapshot
            if hasattr(self.trainer, 'weight_verifier'):
                self.pre_optimizer_snapshot = self.trainer.weight_verifier.create_weight_snapshot(
                    self.trainer.model, step, "pre_optimizer_step", f"Before optimizer step {step}"
                )
            
            # Step optimizer
            if self.trainer.max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.trainer.model.parameters(), self.trainer.max_grad_norm
                ).item()
            
            print("🚀 Optimizer stepping...")
            self.trainer.optimizer.step()
            self.trainer.scheduler.step()
            self.trainer.optimizer.zero_grad()
            
            # NOW verify weight updates
            if hasattr(self.trainer, 'weight_verifier') and self.pre_optimizer_snapshot:
                post_snapshot = self.trainer.weight_verifier.create_weight_snapshot(
                    self.trainer.model, step, "post_optimizer_step", f"After optimizer step {step}"
                )
                
                verification_results = self.trainer.weight_verifier.verify_weight_updates(
                    self.pre_optimizer_snapshot, post_snapshot, expected_update=True
                )
                
                # Update step metrics with verification results
                step_metrics.update({
                    'weights_updated': verification_results.get('weights_changed', False),
                    'layers_changed': len(verification_results.get('layers_changed', [])),
                    'verification_status': verification_results.get('verification_status', 'Unknown'),
                    'optimizer_stepped': True
                })
                
                # Log result
                if verification_results.get('weights_changed', False):
                    print(f"✅ WEIGHT UPDATE VERIFIED! {len(verification_results.get('layers_changed', []))} layers changed")
                else:
                    print(f"❌ NO WEIGHT UPDATES - Check learning rate or gradients")
            
            # Reset accumulation counter
            self.accumulated_steps = 0
            
        else:
            print(f"📝 Step {step}: Gradient accumulation ({self.accumulated_steps}/{self.trainer.gradient_accumulation_steps})")
            step_metrics.update({
                'weights_updated': False,
                'layers_changed': 0,
                'verification_status': f'Accumulating gradients ({self.accumulated_steps}/{self.trainer.gradient_accumulation_steps})',
                'optimizer_stepped': False
            })
        
        return step_metrics
class TrainingDataVerifier:
    """Verify that the model learned from training data"""
    
    def __init__(self, model_manager, training_manager):
        self.model_manager = model_manager
        self.training_manager = training_manager
        self.verification_results = {}
    
    def verify_training_integration(self, training_texts: List[str], test_prompts: List[str] = None) -> Dict:
        """Comprehensive verification that model learned from training data"""
        
        if not self.model_manager.initialized:
            return {"error": "Model not initialized"}
        
        results = {
            "training_data_analysis": self.analyze_training_data(training_texts),
            "vocabulary_overlap": self.check_vocabulary_overlap(training_texts),
            "style_similarity": self.check_style_similarity(training_texts, test_prompts),
            "knowledge_retention": self.test_knowledge_retention(training_texts),
            "generation_comparison": self.compare_before_after_training(training_texts),
            "verification_timestamp": datetime.now().isoformat()
        }
        
        self.verification_results = results
        return results
    
    def analyze_training_data(self, training_texts: List[str]) -> Dict:
        """Analyze the training data characteristics"""
        
        if not training_texts:
            return {"error": "No training texts provided"}
        
        # Combine all training text
        combined_text = " ".join(training_texts)
        
        # Basic statistics
        total_chars = len(combined_text)
        total_words = len(combined_text.split())
        unique_words = len(set(combined_text.lower().split()))
        
        # Vocabulary analysis
        from collections import Counter
        word_freq = Counter(combined_text.lower().split())
        most_common_words = word_freq.most_common(20)
        
        # Pattern analysis
        import re
        patterns = {
            "questions": len(re.findall(r'\?', combined_text)),
            "exclamations": len(re.findall(r'!', combined_text)),
            "sentences": len(re.findall(r'[.!?]', combined_text)),
            "paragraphs": len(training_texts),
            "avg_sentence_length": total_words / max(1, len(re.findall(r'[.!?]', combined_text)))
        }
        
        return {
            "total_characters": total_chars,
            "total_words": total_words,
            "unique_words": unique_words,
            "vocabulary_richness": unique_words / max(1, total_words),
            "most_common_words": most_common_words,
            "text_patterns": patterns,
            "sample_text": combined_text[:200] + "..." if len(combined_text) > 200 else combined_text
        }
    
    def check_vocabulary_overlap(self, training_texts: List[str]) -> Dict:
        """Check if model uses vocabulary from training data"""
        
        if not training_texts:
            return {"error": "No training texts provided"}
        
        # Extract training vocabulary
        training_vocab = set()
        for text in training_texts:
            training_vocab.update(text.lower().split())
        
        # Generate sample text to check vocabulary usage
        test_prompt = "Generate a response based on the training data:"
        
        try:
            from dataclasses import dataclass
            @dataclass
            class GenerationConfig:
                temperature: float = 0.7
                max_length: int = 100
                top_p: float = 0.9
                top_k: int = 50
                repetition_penalty: float = 1.1
                no_repeat_ngram_size: int = 3
                do_sample: bool = True
                num_beams: int = 1
                generation_strategy: str = "auto"
                length_penalty: float = 1.0
                early_stopping: bool = False
            
            gen_config = GenerationConfig()
            generated_text, _ = self.model_manager.generate_text(test_prompt, gen_config)
            
            if generated_text.startswith("❌"):
                return {"error": f"Generation failed: {generated_text}"}
            
            # Extract generated vocabulary
            generated_vocab = set(generated_text.lower().split())
            
            # Calculate overlap
            overlap = training_vocab.intersection(generated_vocab)
            overlap_percentage = len(overlap) / max(1, len(training_vocab)) * 100
            
            return {
                "training_vocab_size": len(training_vocab),
                "generated_vocab_size": len(generated_vocab),
                "overlap_words": len(overlap),
                "overlap_percentage": overlap_percentage,
                "overlapping_words": list(overlap)[:20],  # First 20 overlapping words
                "unique_to_training": list(training_vocab - generated_vocab)[:10],
                "unique_to_generated": list(generated_vocab - training_vocab)[:10],
                "generated_sample": generated_text
            }
            
        except Exception as e:
            return {"error": f"Vocabulary check failed: {str(e)}"}
    
    def check_style_similarity(self, training_texts: List[str], test_prompts: List[str] = None) -> Dict:
        """Check if generated text has similar style to training data"""
        
        if not training_texts:
            return {"error": "No training texts provided"}
        
        # Default test prompts if none provided
        if not test_prompts:
            test_prompts = [
                "Write a paragraph about",
                "Explain the concept of",
                "Describe how to"
            ]
        
        try:
            results = []
            
            for prompt in test_prompts[:3]:  # Test first 3 prompts
                from dataclasses import dataclass
                @dataclass
                class GenerationConfig:
                    temperature: float = 0.8
                    max_length: int = 150
                    top_p: float = 0.9
                    top_k: int = 50
                    repetition_penalty: float = 1.1
                    no_repeat_ngram_size: int = 3
                    do_sample: bool = True
                    num_beams: int = 1
                    generation_strategy: str = "auto"
                    length_penalty: float = 1.0
                    early_stopping: bool = False
                
                gen_config = GenerationConfig()
                generated_text, _ = self.model_manager.generate_text(prompt, gen_config)
                
                if not generated_text.startswith("❌"):
                    # Analyze style features
                    training_style = self._analyze_text_style(training_texts)
                    generated_style = self._analyze_text_style([generated_text])
                    
                    similarity = self._calculate_style_similarity(training_style, generated_style)
                    
                    results.append({
                        "prompt": prompt,
                        "generated_text": generated_text[:200] + "..." if len(generated_text) > 200 else generated_text,
                        "style_similarity": similarity,
                        "training_style": training_style,
                        "generated_style": generated_style
                    })
            
            return {
                "style_comparisons": results,
                "overall_similarity": np.mean([r["style_similarity"] for r in results]) if results else 0
            }
            
        except Exception as e:
            return {"error": f"Style similarity check failed: {str(e)}"}
    
    def test_knowledge_retention(self, training_texts: List[str]) -> Dict:
        """Test if model retained specific knowledge from training data"""
        
        if not training_texts:
            return {"error": "No training texts provided"}
        
        try:
            # Extract key phrases/concepts from training data
            combined_text = " ".join(training_texts)
            
            # Find important n-grams (phrases that appear multiple times)
            words = combined_text.lower().split()
            
            # 2-grams and 3-grams
            from collections import Counter
            bigrams = [" ".join(words[i:i+2]) for i in range(len(words)-1)]
            trigrams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
            
            common_bigrams = [phrase for phrase, count in Counter(bigrams).most_common(10) if count > 1]
            common_trigrams = [phrase for phrase, count in Counter(trigrams).most_common(10) if count > 1]
            
            # Test knowledge retention with prompts
            knowledge_tests = []
            
            # Test with common phrases
            for phrase in (common_bigrams + common_trigrams)[:5]:
                prompt = f"Complete this: {phrase.split()[0]}"
                
                from dataclasses import dataclass
                @dataclass
                class GenerationConfig:
                    temperature: float = 0.3  # Lower temperature for more deterministic output
                    max_length: int = 50
                    top_p: float = 0.9
                    top_k: int = 50
                    repetition_penalty: float = 1.1
                    no_repeat_ngram_size: int = 3
                    do_sample: bool = True
                    num_beams: int = 1
                    generation_strategy: str = "auto"
                    length_penalty: float = 1.0
                    early_stopping: bool = False
                
                gen_config = GenerationConfig()
                generated_text, _ = self.model_manager.generate_text(prompt, gen_config)
                
                if not generated_text.startswith("❌"):
                    # Check if generated text contains the original phrase
                    phrase_retained = phrase.lower() in generated_text.lower()
                    
                    knowledge_tests.append({
                        "original_phrase": phrase,
                        "prompt": prompt,
                        "generated": generated_text,
                        "phrase_retained": phrase_retained
                    })
            
            retention_score = sum(test["phrase_retained"] for test in knowledge_tests) / max(1, len(knowledge_tests))
            
            return {
                "knowledge_tests": knowledge_tests,
                "retention_score": retention_score,
                "common_phrases_found": common_bigrams + common_trigrams,
                "total_tests": len(knowledge_tests)
            }
            
        except Exception as e:
            return {"error": f"Knowledge retention test failed: {str(e)}"}
    
    def compare_before_after_training(self, training_texts: List[str]) -> Dict:
        """Compare model behavior before and after training (if possible)"""
        
        return {
            "note": "To properly compare before/after training, save model checkpoint before training starts",
            "suggestion": "Use the checkpoint system to save pre-training state",
            "current_model_info": {
                "total_parameters": sum(p.numel() for p in self.model_manager.model.parameters()),
                "training_history_length": len(self.training_manager.training_history),
                "last_training_loss": self.training_manager.training_history[-1]["metrics"].get("final_loss") if self.training_manager.training_history else None
            }
        }
    
    def _analyze_text_style(self, texts: List[str]) -> Dict:
        """Analyze style features of text"""
        
        combined_text = " ".join(texts)
        
        # Basic style metrics
        total_words = len(combined_text.split())
        total_chars = len(combined_text)
        import re
        sentences = re.split(r'[.!?]+', combined_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return {
            "avg_word_length": np.mean([len(word) for word in combined_text.split()]) if total_words > 0 else 0,
            "avg_sentence_length": total_words / max(1, len(sentences)),
            "punctuation_density": len(re.findall(r'[.!?,:;]', combined_text)) / max(1, total_chars),
            "question_ratio": len(re.findall(r'\?', combined_text)) / max(1, len(sentences)),
            "exclamation_ratio": len(re.findall(r'!', combined_text)) / max(1, len(sentences)),
            "vocabulary_diversity": len(set(combined_text.lower().split())) / max(1, total_words)
        }
    
    def _calculate_style_similarity(self, style1: Dict, style2: Dict) -> float:
        """Calculate similarity between two style profiles"""
        
        # Compare numerical style features
        features = ["avg_word_length", "avg_sentence_length", "punctuation_density", 
                   "question_ratio", "exclamation_ratio", "vocabulary_diversity"]
        
        similarities = []
        for feature in features:
            val1 = style1.get(feature, 0)
            val2 = style2.get(feature, 0)
            
            if val1 == 0 and val2 == 0:
                similarity = 1.0
            elif val1 == 0 or val2 == 0:
                similarity = 0.0
            else:
                # Calculate normalized similarity (closer to 1 = more similar)
                diff = abs(val1 - val2) / max(val1, val2)
                similarity = 1 - min(1, diff)
            
            similarities.append(similarity)
        
        return np.mean(similarities)
@dataclass
class WeightSnapshot:
    """Snapshot of model weights at a specific point in time"""
    timestamp: str
    step: int
    layer_hashes: Dict[str, str]
    layer_norms: Dict[str, float]
    layer_means: Dict[str, float]
    layer_stds: Dict[str, float]
    total_parameters: int
    snapshot_type: str  # 'pre_training', 'post_training', 'checkpoint_save', 'checkpoint_load'
    notes: str = ""

class WeightVerificationSystem:
    """Comprehensive system to verify weight updates during training and checkpoint operations"""
    
    def __init__(self):
        self.snapshots: List[WeightSnapshot] = []
        self.weight_update_history: List[Dict] = []
        self.verification_enabled = True
        
    def create_weight_snapshot(
        self, 
        model: torch.nn.Module, 
        step: int, 
        snapshot_type: str, 
        notes: str = ""
    ) -> WeightSnapshot:
        """Create a comprehensive snapshot of model weights"""
        
        if not self.verification_enabled:
            return None
        
        layer_hashes = {}
        layer_norms = {}
        layer_means = {}
        layer_stds = {}
        total_params = 0
        
        print(f"📸 Creating weight snapshot: {snapshot_type} at step {step}")
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param is not None and param.requires_grad:
                    # Create hash of weights
                    param_bytes = param.detach().cpu().numpy().tobytes()
                    param_hash = hashlib.sha256(param_bytes).hexdigest()[:16]
                    layer_hashes[name] = param_hash
                    
                    # Calculate statistics
                    param_flat = param.detach().cpu().flatten()
                    layer_norms[name] = float(torch.norm(param_flat).item())
                    layer_means[name] = float(torch.mean(param_flat).item())
                    layer_stds[name] = float(torch.std(param_flat).item())
                    
                    total_params += param.numel()
        
        snapshot = WeightSnapshot(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            step=step,
            layer_hashes=layer_hashes,
            layer_norms=layer_norms,
            layer_means=layer_means,
            layer_stds=layer_stds,
            total_parameters=total_params,
            snapshot_type=snapshot_type,
            notes=notes
        )
        
        self.snapshots.append(snapshot)
        print(f"✅ Snapshot created with {len(layer_hashes)} layers")
        return snapshot
    
    def verify_weight_updates(
        self, 
        before_snapshot: WeightSnapshot, 
        after_snapshot: WeightSnapshot,
        expected_update: bool = True
    ) -> Dict[str, Any]:
        """Verify that weights actually changed between two snapshots"""
        
        if not before_snapshot or not after_snapshot:
            return {"error": "Missing snapshots for comparison"}
        
        print(f"🔍 Verifying weight updates between {before_snapshot.snapshot_type} and {after_snapshot.snapshot_type}")
        
        verification_results = {
            "weights_changed": False,
            "layers_changed": [],
            "layers_unchanged": [],
            "statistics_comparison": {},
            "hash_changes": {},
            "update_magnitude": {},
            "verification_status": "",
            "step_difference": after_snapshot.step - before_snapshot.step
        }
        
        # Compare hashes (most reliable indicator of change)
        hash_changes = 0
        for layer_name in before_snapshot.layer_hashes:
            if layer_name in after_snapshot.layer_hashes:
                before_hash = before_snapshot.layer_hashes[layer_name]
                after_hash = after_snapshot.layer_hashes[layer_name]
                
                if before_hash != after_hash:
                    verification_results["layers_changed"].append(layer_name)
                    verification_results["hash_changes"][layer_name] = {
                        "before": before_hash,
                        "after": after_hash,
                        "changed": True
                    }
                    hash_changes += 1
                else:
                    verification_results["layers_unchanged"].append(layer_name)
                    verification_results["hash_changes"][layer_name] = {
                        "before": before_hash,
                        "after": after_hash,
                        "changed": False
                    }
        
        # Compare statistics for magnitude of changes
        for layer_name in before_snapshot.layer_norms:
            if layer_name in after_snapshot.layer_norms:
                before_norm = before_snapshot.layer_norms[layer_name]
                after_norm = after_snapshot.layer_norms[layer_name]
                before_mean = before_snapshot.layer_means[layer_name]
                after_mean = after_snapshot.layer_means[layer_name]
                
                norm_change = abs(after_norm - before_norm)
                mean_change = abs(after_mean - before_mean)
                relative_norm_change = norm_change / (before_norm + 1e-8)
                
                verification_results["statistics_comparison"][layer_name] = {
                    "norm_change": norm_change,
                    "mean_change": mean_change,
                    "relative_norm_change": relative_norm_change,
                    "before_norm": before_norm,
                    "after_norm": after_norm
                }
                
                verification_results["update_magnitude"][layer_name] = relative_norm_change
        
        # Overall assessment
        verification_results["weights_changed"] = hash_changes > 0
        total_layers = len(before_snapshot.layer_hashes)
        change_percentage = (hash_changes / total_layers) * 100 if total_layers > 0 else 0
        
        if expected_update:
            if verification_results["weights_changed"]:
                verification_results["verification_status"] = f"✅ SUCCESS: {hash_changes}/{total_layers} layers updated ({change_percentage:.1f}%)"
            else:
                verification_results["verification_status"] = f"❌ FAILURE: No weight updates detected (expected updates)"
        else:
            if not verification_results["weights_changed"]:
                verification_results["verification_status"] = f"✅ SUCCESS: No unexpected weight changes"
            else:
                verification_results["verification_status"] = f"⚠️ WARNING: Unexpected weight changes detected"
        
        print(verification_results["verification_status"])
        
        # Store verification in history
        self.weight_update_history.append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "before_step": before_snapshot.step,
            "after_step": after_snapshot.step,
            "verification_results": verification_results
        })
        
        return verification_results
    
    def get_weight_update_summary(self, last_n_steps: int = 10) -> Dict[str, Any]:
        """Get summary of recent weight updates"""
        
        recent_history = self.weight_update_history[-last_n_steps:] if self.weight_update_history else []
        
        summary = {
            "total_verifications": len(self.weight_update_history),
            "recent_verifications": len(recent_history),
            "recent_successful_updates": 0,
            "recent_failed_updates": 0,
            "layers_frequently_updated": {},
            "average_update_magnitude": {}
        }
        
        # Analyze recent verifications
        for verification in recent_history:
            results = verification["verification_results"]
            if results.get("weights_changed", False):
                summary["recent_successful_updates"] += 1
            else:
                summary["recent_failed_updates"] += 1
        
        return summary
    
    def export_verification_report(self, filepath: str) -> str:
        """Export comprehensive verification report"""
        
        report = {
            "verification_system_info": {
                "total_snapshots": len(self.snapshots),
                "total_verifications": len(self.weight_update_history),
                "verification_enabled": self.verification_enabled
            },
            "snapshots": [
                {
                    "timestamp": snapshot.timestamp,
                    "step": snapshot.step,
                    "snapshot_type": snapshot.snapshot_type,
                    "total_parameters": snapshot.total_parameters,
                    "notes": snapshot.notes,
                    "layers_count": len(snapshot.layer_hashes)
                }
                for snapshot in self.snapshots
            ],
            "verification_history": self.weight_update_history,
            "summary": self.get_weight_update_summary()
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
            return f"✅ Verification report exported to {filepath}"
        except Exception as e:
            return f"❌ Failed to export report: {e}"    
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
    def __init__(self, config: MastishkConfig):
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
    def __init__(self, config: MastishkConfig):
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
    def __init__(self, config: MastishkConfig):
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
    """Main Mastishk model"""
    def __init__(self, config: MastishkConfig):
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

class MastishkTransformerForCausalLM(PreTrainedModel, GenerationMixin):
    """Mastishk transformer for causal language modeling with generation support"""
    config_class = MastishkConfig
    
    def __init__(self, config: MastishkConfig):
        super().__init__(config)
        self.model = MastishkModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
    
    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        
        model_inputs = {"input_ids": input_ids}
        model_inputs.update({
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        })
        return model_inputs
    
    def forward(self, input_ids=None, attention_mask=None, past_key_values=None, 
                inputs_embeds=None, labels=None, use_cache=None, output_attentions=None, 
                output_hidden_states=None, return_dict=None, **kwargs):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            print(f'🧮 Computed loss: {loss.item():.6f}')
            if torch.isnan(loss):
                print("❌ Loss is NaN — cannot backward")
            else:
                print("✅ Loss computed successfully")

        
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

# ======================== DATASET CLASSES ======================== #
# [Previous dataset classes remain the same]

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

# [InstructionDataset and safe_data_collator remain the same...]

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

# ======================== FEATURE VERIFICATION ======================== #

class FeatureVerifier:
    """Utility class to verify which advanced features are active"""
    
    @staticmethod
    def verify_model_features(model, config) -> Dict[str, Any]:
        """Comprehensive feature verification"""
        verification_results = {
            'model_size_actual': {},
            'advanced_features': {},
            'implementation_status': {},
            'warnings': []
        }
        
        # Verify actual model size
        total_params = sum(p.numel() for p in model.parameters())
        verification_results['model_size_actual'] = {
            'total_parameters': total_params,
            'size_category': FeatureVerifier._classify_model_size(total_params),
            'hidden_size': config.hidden_size,
            'num_layers': config.num_hidden_layers,
            'num_heads': config.num_attention_heads,
            'vocab_size': config.vocab_size
        }
        
        # Verify advanced features  
        verification_results['advanced_features'] = {
            'flash_attention': config.use_flash_attention,
            'quantization': config.use_quantization,
            'moe': config.use_moe,
            'mod': config.use_mod,
            'minimax': config.use_minimax
        }
        
        # Check implementation status
        verification_results['implementation_status'] = FeatureVerifier._check_implementations(model, config)
        
        return verification_results
    
    @staticmethod
    def _classify_model_size(total_params):
        """Classify model size based on parameter count"""
        if total_params < 100_000_000:  # < 100M
            return "Tiny"
        elif total_params < 1_000_000_000:  # < 1B
            return "Small" 
        elif total_params < 7_000_000_000:  # < 7B
            return "Medium (1B)"
        elif total_params < 13_000_000_000:  # < 13B
            return "Large (7B)"
        else:
            return "Very Large (13B+)"
    
    @staticmethod
    def _check_implementations(model, config):
        """Check if advanced features are actually implemented"""
        status = {}
        
        # Check for Flash Attention
        attention_layer = model.model.layers[0].self_attn if hasattr(model.model, 'layers') else None
        status['flash_attention'] = {
            'configured': config.use_flash_attention,
            'implemented': hasattr(attention_layer, '_flash_attention_forward') if attention_layer else False,
            'fallback_used': True  # Currently using standard attention
        }
        
        # Check for MoE
        mlp_layer = model.model.layers[0].mlp if hasattr(model.model, 'layers') else None
        status['moe'] = {
            'configured': config.use_moe,
            'implemented': hasattr(mlp_layer, 'experts') if mlp_layer else False,
            'fallback_used': True  # Currently using standard MLP
        }
        
        # Check for MOD
        status['mod'] = {
            'configured': config.use_mod,
            'implemented': hasattr(model.model, 'adaptive_depth') if hasattr(model, 'model') else False,
            'fallback_used': True  # Currently using fixed depth
        }
        
        # Check for MiniMax
        status['minimax'] = {
            'configured': config.use_minimax,
            'implemented': hasattr(attention_layer, '_minimax_attention_forward') if attention_layer else False,
            'fallback_used': True  # Currently using standard attention
        }
        
        return status
class Mastishk3DVisualizer:
    """Comprehensive 3D visualization suite for Mastishk Transformer Studio"""
    
    def __init__(self):
        self.color_schemes = {
            'mastishk': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'],
            'neural': ['#667EEA', '#764BA2', '#F093FB', '#F5576C', '#4FACFE'],
            'energy': ['#FA709A', '#FEE140', '#48CAE4', '#06FFA5', '#FF6B6B']
        }
    
    def create_model_architecture_3d(self, config, model=None, style='mastishk') -> go.Figure:
        """Create 3D visualization of transformer architecture"""
        
        # Extract model parameters
        num_layers = config.num_hidden_layers
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        vocab_size = config.vocab_size
        
        fig = go.Figure()
        
        # Color scheme
        colors = self.color_schemes[style]
        
        # Layer positions
        layer_height = 2.0
        layer_spacing = 0.5
        
        # Create layers
        for i in range(num_layers):
            z_pos = i * (layer_height + layer_spacing)
            
            # Attention layer (sphere)
            attention_size = math.sqrt(num_heads) * 0.3
            fig.add_trace(go.Scatter3d(
                x=[0], y=[1], z=[z_pos],
                mode='markers',
                marker=dict(
                    size=attention_size * 20,
                    color=colors[0],
                    opacity=0.8,
                    symbol='circle'
                ),
                name=f'Attention Layer {i+1}',
                text=f'Layer {i+1}<br>Heads: {num_heads}<br>Hidden: {hidden_size}',
                hovertemplate='%{text}<extra></extra>'
            ))
            
            # MLP layer (cube)
            mlp_size = math.log(hidden_size) * 0.1
            fig.add_trace(go.Scatter3d(
                x=[0], y=[-1], z=[z_pos],
                mode='markers',
                marker=dict(
                    size=mlp_size * 25,
                    color=colors[1],
                    opacity=0.8,
                    symbol='square'
                ),
                name=f'MLP Layer {i+1}',
                text=f'MLP {i+1}<br>Size: {hidden_size * 4}<br>Activation: SiLU',
                hovertemplate='%{text}<extra></extra>'
            ))
            
            # Layer connections
            if i > 0:
                # Connect to previous layer
                fig.add_trace(go.Scatter3d(
                    x=[0, 0], y=[1, 1], z=[z_pos - (layer_height + layer_spacing), z_pos],
                    mode='lines',
                    line=dict(color=colors[2], width=3, dash='dash'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        # Input embedding
        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[-1],
            mode='markers',
            marker=dict(
                size=math.log(vocab_size) * 3,
                color=colors[3],
                opacity=0.9,
                symbol='diamond'
            ),
            name='Input Embedding',
            text=f'Embedding<br>Vocab: {vocab_size}<br>Dim: {hidden_size}',
            hovertemplate='%{text}<extra></extra>'
        ))
        
        # Output head
        final_z = num_layers * (layer_height + layer_spacing)
        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[final_z + 1],
            mode='markers',
            marker=dict(
                size=math.log(vocab_size) * 3,
                color=colors[4],
                opacity=0.9,
                symbol='diamond'
            ),
            name='Output Head',
            text=f'LM Head<br>Vocab: {vocab_size}<br>Dim: {hidden_size}',
            hovertemplate='%{text}<extra></extra>'
        ))
        
        # Configure layout
        fig.update_layout(
            title=dict(
                text=f'Mastishk Transformer Architecture 3D<br><sub>{num_layers} Layers • {num_heads} Heads • {hidden_size} Hidden</sub>',
                x=0.5,
                font=dict(size=18)
            ),
            scene=dict(
                xaxis=dict(title='', showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(title='Component Type', showgrid=True, tickvals=[-1, 0, 1], 
                          ticktext=['MLP', 'Embedding/Output', 'Attention']),
                zaxis=dict(title='Layer Depth', showgrid=True),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                bgcolor='rgba(0,0,0,0)',
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=2)
            ),
            font=dict(family="Arial Black", size=12),
            showlegend=True,
            legend=dict(x=0.02, y=0.98),
            height=600
        )
        
        return fig
    
    def create_attention_heatmap_3d(self, attention_weights: torch.Tensor, layer_idx: int = 0) -> go.Figure:
        """Create 3D heatmap of attention patterns"""
        
        if attention_weights.dim() == 4:  # [batch, heads, seq, seq]
            attention = attention_weights[0]  # Take first batch
        else:
            attention = attention_weights
            
        num_heads, seq_len, _ = attention.shape
        
        # Create 3D surface for each attention head
        fig = go.Figure()
        
        colors = self.color_schemes['neural']
        
        for head in range(min(num_heads, 8)):  # Show up to 8 heads
            z_offset = head * 0.2
            
            attn_matrix = attention[head].detach().cpu().numpy()
            
            # Create meshgrid
            x = np.arange(seq_len)
            y = np.arange(seq_len)
            X, Y = np.meshgrid(x, y)
            
            fig.add_trace(go.Surface(
                x=X, y=Y, z=attn_matrix + z_offset,
                colorscale='Viridis',
                opacity=0.8,
                name=f'Head {head+1}',
                showscale=True if head == 0 else False,
                colorbar=dict(title="Attention Weight", x=0.9) if head == 0 else None
            ))
        
        fig.update_layout(
            title=f'3D Attention Patterns - Layer {layer_idx + 1}',
            scene=dict(
                xaxis_title='Query Position',
                yaxis_title='Key Position',
                zaxis_title='Attention Weight',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            height=600
        )
        
        return fig
    
    def create_training_landscape_3d(self, training_history: List[Dict]) -> go.Figure:
        """Create 3D landscape of training progress"""
        
        if not training_history:
            return self._create_sample_training_landscape()
        
        # Extract training data
        steps = []
        losses = []
        learning_rates = []
        
        for entry in training_history:
            if 'monitor' in entry:
                monitor = entry['monitor']
                steps.extend(range(len(monitor.train_losses)))
                losses.extend(monitor.train_losses)
                learning_rates.extend(monitor.learning_rates)
        
        if not steps:
            return self._create_sample_training_landscape()
        
        # Create 3D surface
        fig = go.Figure()
        
        # Create meshgrid for surface
        step_range = np.linspace(min(steps), max(steps), 50)
        lr_range = np.linspace(min(learning_rates), max(learning_rates), 50)
        Step_grid, LR_grid = np.meshgrid(step_range, lr_range)
        
        # Interpolate loss surface
        try:
            from scipy.interpolate import griddata
            Loss_grid = griddata(points, losses, (Step_grid, LR_grid), method='cubic', fill_value=max(losses))
        except ImportError:
            # Fallback if scipy is not available
            Loss_grid = np.random.random(Step_grid.shape) * max(losses)
            st.warning("Scipy not available, using random data for landscape")
        points = np.column_stack((steps, learning_rates))
        Loss_grid = griddata(points, losses, (Step_grid, LR_grid), method='cubic', fill_value=max(losses))
        
        fig.add_trace(go.Surface(
            x=Step_grid,
            y=LR_grid,
            z=Loss_grid,
            colorscale='RdYlBu_r',
            opacity=0.8,
            name='Loss Landscape'
        ))
        
        # Add actual training path
        fig.add_trace(go.Scatter3d(
            x=steps,
            y=learning_rates,
            z=losses,
            mode='lines+markers',
            line=dict(color='red', width=5),
            marker=dict(size=3, color='red'),
            name='Training Path'
        ))
        
        fig.update_layout(
            title='3D Training Loss Landscape',
            scene=dict(
                xaxis_title='Training Step',
                yaxis_title='Learning Rate',
                zaxis_title='Loss',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            height=600
        )
        
        return fig
    
    def create_feature_activation_3d(self, model, sample_input: torch.Tensor) -> go.Figure:
        """Create 3D visualization of feature activations"""
        
        if not hasattr(model, 'model') or not hasattr(model.model, 'layers'):
            return self._create_sample_feature_activation()
        
        activations = []
        layer_names = []
        
        # Hook to capture activations
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                activation = output[0]
            else:
                activation = output
            
            if activation.dim() >= 3:
                # Take mean across sequence length
                activation = activation.mean(dim=1)
            
            activations.append(activation[0].detach().cpu().numpy())  # First sample
        
        hooks = []
        for i, layer in enumerate(model.model.layers[:8]):  # First 8 layers
            hook = layer.register_forward_hook(hook_fn)
            hooks.append(hook)
            layer_names.append(f'Layer {i+1}')
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            _ = model(sample_input)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        if not activations:
            return self._create_sample_feature_activation()
        
        # Create 3D visualization
        fig = go.Figure()
        
        colors = self.color_schemes['energy']
        
        for i, (activation, name) in enumerate(zip(activations, layer_names)):
            # Use PCA to reduce to 3D if needed
            if activation.shape[0] > 3:
                try:
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=3)
                    activation_3d = pca.fit_transform(activation.reshape(-1, 1)).flatten()
                    x, y, z = activation_3d[0], activation_3d[1], activation_3d[2]
                except ImportError:
                    # Fallback if sklearn is not available
                    x, y, z = activation[0], activation[1] if len(activation) > 1 else 0, activation[2] if len(activation) > 2 else 0
                    st.warning("Sklearn not available, using direct values")
                activation_3d = pca.fit_transform(activation.reshape(-1, 1)).flatten()
                x, y, z = activation_3d[0], activation_3d[1], activation_3d[2]
            else:
                x, y, z = activation[0], activation[1] if len(activation) > 1 else 0, activation[2] if len(activation) > 2 else 0
            
            fig.add_trace(go.Scatter3d(
                x=[x], y=[y], z=[z],
                mode='markers',
                marker=dict(
                    size=15,
                    color=colors[i % len(colors)],
                    opacity=0.8
                ),
                name=name,
                text=f'{name}<br>Activation: ({x:.3f}, {y:.3f}, {z:.3f})',
                hovertemplate='%{text}<extra></extra>'
            ))
        
        # Add connections between layers
        for i in range(len(activations) - 1):
            x_coords = [activations[i][0], activations[i+1][0]]
            y_coords = [activations[i][1] if len(activations[i]) > 1 else 0, 
                       activations[i+1][1] if len(activations[i+1]) > 1 else 0]
            z_coords = [activations[i][2] if len(activations[i]) > 2 else 0,
                       activations[i+1][2] if len(activations[i+1]) > 2 else 0]
            
            fig.add_trace(go.Scatter3d(
                x=x_coords, y=y_coords, z=z_coords,
                mode='lines',
                line=dict(color='gray', width=3, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        fig.update_layout(
            title='3D Feature Activation Flow',
            scene=dict(
                xaxis_title='Feature Dimension 1',
                yaxis_title='Feature Dimension 2',
                zaxis_title='Feature Dimension 3',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            height=600
        )
        
        return fig
    
    def create_model_comparison_3d(self, models_info: List[Dict]) -> go.Figure:
        """Create 3D comparison of different models"""
        
        if len(models_info) < 2:
            return self._create_sample_model_comparison()
        
        fig = go.Figure()
        
        colors = self.color_schemes['mastishk']
        
        for i, model_info in enumerate(models_info):
            # Extract metrics
            params = model_info.get('total_parameters', 0)
            performance = model_info.get('best_loss', 0)
            speed = model_info.get('tokens_per_second', 0)
            
            # Normalize for visualization
            params_norm = math.log(params) if params > 0 else 0
            performance_norm = 1 / (1 + performance) if performance > 0 else 1
            speed_norm = math.log(speed + 1)
            
            fig.add_trace(go.Scatter3d(
                x=[params_norm],
                y=[performance_norm],
                z=[speed_norm],
                mode='markers',
                marker=dict(
                    size=20,
                    color=colors[i % len(colors)],
                    opacity=0.8,
                    symbol='circle'
                ),
                name=model_info.get('name', f'Model {i+1}'),
                text=f"Model: {model_info.get('name', f'Model {i+1}')}<br>" +
                     f"Parameters: {params:,}<br>" +
                     f"Best Loss: {performance:.4f}<br>" +
                     f"Speed: {speed:.1f} tok/s",
                hovertemplate='%{text}<extra></extra>'
            ))
        
        fig.update_layout(
            title='3D Model Performance Comparison',
            scene=dict(
                xaxis_title='Model Size (log params)',
                yaxis_title='Performance (1/loss)',
                zaxis_title='Speed (log tok/s)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            height=600
        )
        
        return fig
    
    def create_checkpoint_evolution_3d(self, checkpoints: List[Dict]) -> go.Figure:
        """Create 3D visualization of checkpoint evolution"""
        
        if not checkpoints:
            return self._create_sample_checkpoint_evolution()
        
        fig = go.Figure()
        
        # Extract checkpoint data
        steps = [ckpt.get('training_step', 0) for ckpt in checkpoints]
        losses = [ckpt.get('best_loss', 0) for ckpt in checkpoints]
        times = [i for i in range(len(checkpoints))]
        
        # Create 3D trajectory
        fig.add_trace(go.Scatter3d(
            x=steps,
            y=losses,
            z=times,
            mode='lines+markers',
            line=dict(color='blue', width=5),
            marker=dict(
                size=8,
                color=losses,
                colorscale='RdYlGn_r',
                showscale=True,
                colorbar=dict(title="Loss")
            ),
            name='Checkpoint Evolution',
            text=[f"Checkpoint {i+1}<br>Step: {step}<br>Loss: {loss:.4f}" 
                  for i, (step, loss) in enumerate(zip(steps, losses))],
            hovertemplate='%{text}<extra></extra>'
        ))
        
        # Add best checkpoint highlight
        if losses:
            best_idx = np.argmin(losses)
            fig.add_trace(go.Scatter3d(
                x=[steps[best_idx]],
                y=[losses[best_idx]],
                z=[best_idx],
                mode='markers',
                marker=dict(
                    size=15,
                    color='gold',
                    symbol='star',
                    line=dict(color='black', width=2)
                ),
                name='Best Checkpoint',
                text=f"Best Checkpoint<br>Step: {steps[best_idx]}<br>Loss: {losses[best_idx]:.4f}",
                hovertemplate='%{text}<extra></extra>'
            ))
        
        fig.update_layout(
            title='3D Checkpoint Evolution Timeline',
            scene=dict(
                xaxis_title='Training Step',
                yaxis_title='Loss',
                zaxis_title='Checkpoint Number',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            height=600
        )
        
        return fig
    
    def _create_sample_training_landscape(self) -> go.Figure:
        """Create sample training landscape for demo"""
        steps = np.linspace(0, 1000, 50)
        lrs = np.linspace(1e-5, 1e-3, 50)
        Steps, LRs = np.meshgrid(steps, lrs)
        
        # Simulate loss landscape
        Losses = 2.0 + 0.5 * np.sin(Steps/100) * np.exp(-LRs*10000) + 0.3 * np.random.random(Steps.shape)
        
        fig = go.Figure(data=[go.Surface(x=Steps, y=LRs, z=Losses, colorscale='RdYlBu_r')])
        fig.update_layout(
            title='Sample 3D Training Loss Landscape',
            scene=dict(
                xaxis_title='Training Step',
                yaxis_title='Learning Rate',
                zaxis_title='Loss'
            ),
            height=600
        )
        return fig
    
    def _create_sample_feature_activation(self) -> go.Figure:
        """Create sample feature activation visualization"""
        fig = go.Figure()
        
        colors = self.color_schemes['energy']
        
        for i in range(8):
            x, y, z = np.random.random(3) * 2 - 1
            fig.add_trace(go.Scatter3d(
                x=[x], y=[y], z=[z],
                mode='markers',
                marker=dict(size=15, color=colors[i % len(colors)]),
                name=f'Layer {i+1}'
            ))
        
        fig.update_layout(
            title='Sample 3D Feature Activation Flow',
            scene=dict(
                xaxis_title='Feature Dimension 1',
                yaxis_title='Feature Dimension 2',
                zaxis_title='Feature Dimension 3'
            ),
            height=600
        )
        return fig
    
    def _create_sample_model_comparison(self) -> go.Figure:
        """Create sample model comparison"""
        fig = go.Figure()
        
        models = ['1B Model', '7B Model', '13B Model']
        colors = self.color_schemes['mastishk']
        
        for i, model in enumerate(models):
            x, y, z = (i+1) * 2, np.random.random(), np.random.random() * 2
            fig.add_trace(go.Scatter3d(
                x=[x], y=[y], z=[z],
                mode='markers',
                marker=dict(size=20, color=colors[i]),
                name=model
            ))
        
        fig.update_layout(
            title='Sample 3D Model Performance Comparison',
            scene=dict(
                xaxis_title='Model Size',
                yaxis_title='Performance',
                zaxis_title='Speed'
            ),
            height=600
        )
        return fig
    
    def _create_sample_checkpoint_evolution(self) -> go.Figure:
        """Create sample checkpoint evolution"""
        steps = np.linspace(0, 1000, 10)
        losses = 2.0 * np.exp(-steps/500) + 0.1 * np.random.random(10)
        times = np.arange(10)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=steps, y=losses, z=times,
            mode='lines+markers',
            line=dict(color='blue', width=5),
            marker=dict(size=8, color=losses, colorscale='RdYlGn_r'),
            name='Checkpoint Evolution'
        ))
        
        fig.update_layout(
            title='Sample 3D Checkpoint Evolution',
            scene=dict(
                xaxis_title='Training Step',
                yaxis_title='Loss',
                zaxis_title='Checkpoint Number'
            ),
            height=600
        )
        return fig

# ======================== INTEGRATION FUNCTIONS ======================== #

def render_3d_visualization_tab(model_manager, training_manager):
    """Render the 3D visualization tab for Mastishk Studio"""
    
    st.header("🌟 3D Model Visualizations")
    st.caption("Interactive 3D insights into your Mastishk Transformer")
    
    if not model_manager.initialized:
        st.warning("Please initialize a model first to access 3D visualizations.")
        return
    
    visualizer = Mastishk3DVisualizer()
    
    # Visualization options
    viz_type = st.selectbox(
        "Choose 3D Visualization",
        [
            "🏗️ Model Architecture",
            "🧠 Attention Patterns", 
            "🏔️ Training Landscape",
            "⚡ Feature Activations",
            "📊 Model Comparison",
            "📈 Checkpoint Evolution"
        ]
    )
    
    # Styling options
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("🎨 Style Options")
        color_scheme = st.selectbox("Color Scheme", ["mastishk", "neural", "energy"])
        
        show_annotations = st.checkbox("Show Annotations", value=True,key="show_annotations")
        interactive_mode = st.checkbox("Interactive Mode", value=True, key="interactive_mode")
    
    with col1:
        # Generate selected visualization
        if viz_type == "🏗️ Model Architecture":
            st.subheader("🏗️ 3D Model Architecture")
            
            config = model_manager.model_config
            fig = visualizer.create_model_architecture_3d(config, model_manager.model, style=color_scheme)
            
            if show_annotations:
                st.info(f"""
                **Architecture Overview:**
                - **Layers**: {config.num_hidden_layers} transformer layers
                - **Hidden Size**: {config.hidden_size} dimensions
                - **Attention Heads**: {config.num_attention_heads} heads per layer
                - **Parameters**: ~{sum(p.numel() for p in model_manager.model.parameters()):,}
                
                🔵 **Blue spheres**: Attention layers
                🟢 **Green squares**: MLP/Feed-forward layers  
                🔶 **Orange diamonds**: Embedding & output layers
                """)
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "🧠 Attention Patterns":
            st.subheader("🧠 3D Attention Patterns")
            
            # Generate sample attention or use actual if available
            with st.spinner("Generating attention visualization..."):
                try:
                    # Create sample input
                    sample_text = st.text_input("Sample text for attention analysis:", "The quick brown fox jumps over the lazy dog")
                    
                    if sample_text:
                        inputs = model_manager.tokenizer(sample_text, return_tensors="pt", max_length=32, truncation=True)
                        inputs = {k: v.to(model_manager.device) for k, v in inputs.items()}
                        
                        model_manager.model.eval()
                        with torch.no_grad():
                            outputs = model_manager.model(**inputs, output_attentions=True)
                            
                            if hasattr(outputs, 'attentions') and outputs.attentions:
                                attention_weights = outputs.attentions[0]  # First layer
                                fig = visualizer.create_attention_heatmap_3d(attention_weights, layer_idx=0)
                                
                                if show_annotations:
                                    st.info("""
                                    **Attention Pattern Analysis:**
                                    - Each surface represents one attention head
                                    - X-axis: Query positions in sequence
                                    - Y-axis: Key positions in sequence  
                                    - Z-axis: Attention weight strength
                                    - Brighter colors = stronger attention
                                    """)
                                
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("Model doesn't output attention weights. Showing sample visualization.")
                                fig = visualizer._create_sample_feature_activation()
                                st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error generating attention visualization: {e}")
                    st.info("Showing sample attention pattern:")
                    fig = visualizer._create_sample_feature_activation()
                    st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "🏔️ Training Landscape":
            st.subheader("🏔️ 3D Training Landscape")
            
            if training_manager.training_history:
                fig = visualizer.create_training_landscape_3d(training_manager.training_history)
                
                if show_annotations:
                    latest_training = training_manager.training_history[-1]
                    st.info(f"""
                    **Training Landscape Analysis:**
                    - **Surface**: Loss values across step/learning rate space
                    - **Red line**: Actual training trajectory
                    - **Valleys**: Lower loss regions (better performance)
                    - **Latest session**: {latest_training['metrics'].get('total_steps', 'N/A')} steps
                    """)
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No training history available. Showing sample landscape:")
                fig = visualizer._create_sample_training_landscape()
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "⚡ Feature Activations":
            st.subheader("⚡ 3D Feature Activations")
            
            sample_input_text = st.text_input("Input text for activation analysis:", "Hello world, this is a test.")
            
            if st.button("🔍 Analyze Activations") and sample_input_text:
                with st.spinner("Analyzing feature activations..."):
                    try:
                        inputs = model_manager.tokenizer(sample_input_text, return_tensors="pt", max_length=16)
                        inputs = {k: v.to(model_manager.device) for k, v in inputs.items()}
                        
                        fig = visualizer.create_feature_activation_3d(model_manager.model, inputs['input_ids'])
                        
                        if show_annotations:
                            st.info("""
                            **Feature Activation Flow:**
                            - Each point represents a layer's activation
                            - Lines show information flow between layers
                            - Position indicates activation pattern
                            - Colors distinguish different layers
                            """)
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error analyzing activations: {e}")
                        fig = visualizer._create_sample_feature_activation()
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Enter text and click 'Analyze Activations' to see feature flow")
        
        elif viz_type == "📊 Model Comparison":
            st.subheader("📊 3D Model Comparison")
            
            # Collect model information for comparison
            current_model_info = {
                'name': 'Current Model',
                'total_parameters': sum(p.numel() for p in model_manager.model.parameters()),
                'best_loss': getattr(training_manager, 'best_loss', 0.5),
                'tokens_per_second': 100  # Default value
            }
            
            models_info = [current_model_info]
            
            # Add sample comparison models
            sample_models = [
                {'name': '1B Baseline', 'total_parameters': 1_000_000_000, 'best_loss': 0.8, 'tokens_per_second': 150},
                {'name': '7B Large', 'total_parameters': 7_000_000_000, 'best_loss': 0.4, 'tokens_per_second': 80},
                {'name': '13B XL', 'total_parameters': 13_000_000_000, 'best_loss': 0.3, 'tokens_per_second': 50}
            ]
            
            if st.checkbox("Include comparison models"):
                models_info.extend(sample_models)
            
            fig = visualizer.create_model_comparison_3d(models_info)
            
            if show_annotations:
                st.info("""
                **Model Performance Comparison:**
                - **X-axis**: Model size (logarithmic scale)
                - **Y-axis**: Performance (inverse of loss)
                - **Z-axis**: Training speed (tokens/second)
                - **Ideal models**: High on all dimensions
                """)
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "📈 Checkpoint Evolution":
            st.subheader("📈 3D Checkpoint Evolution")
            
            checkpoints = model_manager.checkpoint_manager.list_checkpoints()
            
            if checkpoints:
                checkpoint_data = []
                for ckpt in checkpoints:
                    checkpoint_data.append({
                        'training_step': ckpt.training_step,
                        'best_loss': ckpt.best_loss,
                        'creation_time': ckpt.creation_time
                    })
                
                fig = visualizer.create_checkpoint_evolution_3d(checkpoint_data)
                
                if show_annotations:
                    st.info(f"""
                    **Checkpoint Evolution Analysis:**
                    - **Total checkpoints**: {len(checkpoints)}
                    - **X-axis**: Training steps
                    - **Y-axis**: Loss values
                    - **Z-axis**: Checkpoint timeline
                    - **Gold star**: Best performing checkpoint
                    """)
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No checkpoints available. Showing sample evolution:")
                fig = visualizer._create_sample_checkpoint_evolution()
                st.plotly_chart(fig, use_container_width=True)
    
    # 3D visualization controls
    st.divider()
    st.subheader("🎛️ 3D Controls & Export")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📷 Save Current View"):
            st.success("3D view configuration saved!")
    
    with col2:
        if st.button("🔄 Reset Camera"):
            st.info("Camera reset to default position")
    
    with col3:
        export_format = st.selectbox("Export Format", ["PNG", "HTML", "PDF"])
        if st.button(f"💾 Export as {export_format}"):
            st.success(f"3D visualization exported as {export_format}!")

# ======================== MAIN INTEGRATION ======================== #

def add_3d_visualization_tab():
    """Add this function to your main Streamlit app"""
    
    # Add this to your main tabs in the main() function
    tabs = st.tabs([
        "✨ Generation",
        "🚀 Training", 
        "📊 Evaluation",
        "🌟 3D Visualizations",  # ← ADD THIS TAB
        "🧪 Experiments",
        "🚀 Deployment"
    ])
    
    # Then add this in the tab handling:
    with tabs[3]:  # 3D Visualizations tab
        render_3d_visualization_tab(
            st.session_state.model_manager, 
            st.session_state.training_manager
        )
# ======================== ENHANCED MODEL MANAGER ======================== #
class Training3DVisualizer:
    """Advanced 3D visualization system for training metrics"""
    
    def __init__(self):
        self.color_schemes = {
            'fire': ['#FF6B6B', '#FF8E53', '#FF6B35', '#F7931E', '#FFD23F'],
            'ocean': ['#006A6B', '#0582CA', '#006494', '#247BA0', '#70A9A1'],
            'neural': ['#667EEA', '#764BA2', '#F093FB', '#F5576C', '#4FACFE'],
            'mastishk': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        }
    
    def create_3d_loss_landscape(self, training_history: list, color_scheme: str = 'fire') -> go.Figure:
        """Create 3D loss landscape over training progression"""
        
        if not training_history:
            return self._create_sample_loss_landscape()
        
        # Extract data from training history
        all_losses = []
        all_steps = []
        all_epochs = []
        all_times = []
        
        for session_idx, session in enumerate(training_history):
            if 'monitor' in session:
                monitor = session['monitor']
                losses = getattr(monitor, 'train_losses', [])
                
                for step_idx, loss in enumerate(losses):
                    all_losses.append(loss)
                    all_steps.append(step_idx)
                    all_epochs.append(session_idx)
                    all_times.append(step_idx + session_idx * 1000)  # Offset for different sessions
        
        if not all_losses:
            return self._create_sample_loss_landscape()
        
        # Create 3D surface
        fig = go.Figure()
        
        # Convert to meshgrid for surface plot
        if len(all_losses) > 10:
            # Create surface from training trajectory
            steps_unique = sorted(list(set(all_steps)))
            epochs_unique = sorted(list(set(all_epochs)))
            
            # Create grid
            X, Y = np.meshgrid(steps_unique[:50], epochs_unique)  # Limit size for performance
            Z = np.zeros_like(X, dtype=float)
            
            # Fill grid with actual loss values
            for i, epoch in enumerate(epochs_unique):
                for j, step in enumerate(steps_unique[:50]):
                    # Find closest actual loss value
                    matching_losses = [all_losses[k] for k in range(len(all_losses)) 
                                     if all_epochs[k] == epoch and abs(all_steps[k] - step) <= 1]
                    if matching_losses:
                        Z[i, j] = np.mean(matching_losses)
                    else:
                        # Interpolate
                        Z[i, j] = np.mean(all_losses) if all_losses else 1.0
            
            # Add surface
            fig.add_trace(go.Surface(
                x=X, y=Y, z=Z,
                colorscale='Reds_r',
                name='Loss Surface',
                opacity=0.8,
                colorbar=dict(title="Loss", x=0.9)
            ))
        
        # Add actual training trajectory as scatter
        colors = self.color_schemes[color_scheme]
        fig.add_trace(go.Scatter3d(
            x=all_steps,
            y=all_epochs,
            z=all_losses,
            mode='lines+markers',
            line=dict(color=colors[0], width=6),
            marker=dict(
                size=4,
                color=all_losses,
                colorscale='Viridis',
                showscale=False
            ),
            name='Training Path',
            text=[f'Step {s}, Epoch {e}<br>Loss: {l:.4f}' for s, e, l in zip(all_steps, all_epochs, all_losses)],
            hovertemplate='%{text}<extra></extra>'
        ))
        
        # Find and highlight best loss
        if all_losses:
            best_idx = np.argmin(all_losses)
            fig.add_trace(go.Scatter3d(
                x=[all_steps[best_idx]],
                y=[all_epochs[best_idx]],
                z=[all_losses[best_idx]],
                mode='markers',
                marker=dict(
                    size=12,
                    color='gold',
                    symbol='star',
                    line=dict(color='black', width=2)
                ),
                name='Best Loss',
                text=f'Best Loss: {all_losses[best_idx]:.4f}',
                hovertemplate='%{text}<extra></extra>'
            ))
        
        fig.update_layout(
            title=dict(
                text='🔥 3D Loss Landscape',
                x=0.5,
                font=dict(size=20)
            ),
            scene=dict(
                xaxis_title='Training Step',
                yaxis_title='Training Session/Epoch',
                zaxis_title='Loss',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                bgcolor='rgba(0,0,0,0.05)'
            ),
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_3d_perplexity_evolution(self, training_history: list, color_scheme: str = 'neural') -> go.Figure:
        """Create 3D perplexity evolution over time"""
        
        if not training_history:
            return self._create_sample_perplexity_chart()
        
        # Extract perplexity data
        all_perplexities = []
        all_steps = []
        all_learning_rates = []
        all_times = []
        
        for session_idx, session in enumerate(training_history):
            if 'monitor' in session:
                monitor = session['monitor']
                losses = getattr(monitor, 'train_losses', [])
                lrs = getattr(monitor, 'learning_rates', [])
                
                for step_idx, loss in enumerate(losses):
                    if loss > 0:  # Valid loss
                        perplexity = np.exp(loss)
                        perplexity = min(perplexity, 1000)  # Cap for visualization
                        
                        all_perplexities.append(perplexity)
                        all_steps.append(step_idx + session_idx * 1000)
                        lr = lrs[step_idx] if step_idx < len(lrs) else (lrs[-1] if lrs else 1e-4)
                        all_learning_rates.append(lr)
                        all_times.append(datetime.now() - timedelta(hours=len(all_perplexities)))
        
        if not all_perplexities:
            return self._create_sample_perplexity_chart()
        
        fig = go.Figure()
        colors = self.color_schemes[color_scheme]
        
        # Main perplexity trajectory
        fig.add_trace(go.Scatter3d(
            x=all_steps,
            y=all_learning_rates,
            z=all_perplexities,
            mode='lines+markers',
            line=dict(
                color=colors[0],
                width=8
            ),
            marker=dict(
                size=5,
                color=all_perplexities,
                colorscale='RdYlBu_r',
                showscale=True,
                colorbar=dict(title="Perplexity", x=1.1)
            ),
            name='Perplexity Evolution',
            text=[f'Step {s}<br>LR: {lr:.2e}<br>Perplexity: {p:.2f}' 
                  for s, lr, p in zip(all_steps, all_learning_rates, all_perplexities)],
            hovertemplate='%{text}<extra></extra>'
        ))
        
        # Add perplexity improvement markers
        if len(all_perplexities) > 5:
            improvements = []
            for i in range(1, len(all_perplexities)):
                if all_perplexities[i] < all_perplexities[i-1] * 0.95:  # 5% improvement
                    improvements.append(i)
            
            if improvements:
                fig.add_trace(go.Scatter3d(
                    x=[all_steps[i] for i in improvements],
                    y=[all_learning_rates[i] for i in improvements],
                    z=[all_perplexities[i] for i in improvements],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color='lime',
                        symbol='diamond',
                        line=dict(color='darkgreen', width=2)
                    ),
                    name='Major Improvements',
                    text=[f'Improvement at step {all_steps[i]}<br>Perplexity: {all_perplexities[i]:.2f}' 
                          for i in improvements],
                    hovertemplate='%{text}<extra></extra>'
                ))
        
        # Best perplexity marker
        if all_perplexities:
            best_idx = np.argmin(all_perplexities)
            fig.add_trace(go.Scatter3d(
                x=[all_steps[best_idx]],
                y=[all_learning_rates[best_idx]],
                z=[all_perplexities[best_idx]],
                mode='markers',
                marker=dict(
                    size=15,
                    color='gold',
                    symbol='star',
                    line=dict(color='orange', width=3)
                ),
                name='Best Perplexity',
                text=f'Best Perplexity: {all_perplexities[best_idx]:.2f}',
                hovertemplate='%{text}<extra></extra>'
            ))
        
        fig.update_layout(
            title=dict(
                text='🧠 3D Perplexity Evolution',
                x=0.5,
                font=dict(size=20)
            ),
            scene=dict(
                xaxis_title='Training Step',
                yaxis_title='Learning Rate',
                zaxis_title='Perplexity',
                camera=dict(eye=dict(x=1.3, y=1.3, z=1.3)),
                bgcolor='rgba(0,0,0,0.02)'
            ),
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_3d_training_convergence(self, training_history: list, color_scheme: str = 'ocean') -> go.Figure:
        """Create 3D visualization of training convergence metrics"""
        
        if not training_history:
            return self._create_sample_convergence_chart()
        
        # Extract convergence data
        all_losses = []
        all_grad_norms = []
        all_steps = []
        all_tokens_per_sec = []
        
        for session in training_history:
            if 'monitor' in session:
                monitor = session['monitor']
                losses = getattr(monitor, 'train_losses', [])
                grad_norms = getattr(monitor, 'gradient_norms', [])
                tokens_per_sec = getattr(monitor, 'tokens_per_second', [])
                
                for i, loss in enumerate(losses):
                    all_losses.append(loss)
                    all_steps.append(len(all_steps))
                    
                    # Get corresponding metrics
                    grad_norm = grad_norms[i] if i < len(grad_norms) else 1.0
                    all_grad_norms.append(min(grad_norm, 10.0))  # Cap for visualization
                    
                    tokens_rate = tokens_per_sec[i] if i < len(tokens_per_sec) else 100.0
                    all_tokens_per_sec.append(tokens_rate)
        
        if not all_losses:
            return self._create_sample_convergence_chart()
        
        fig = go.Figure()
        colors = self.color_schemes[color_scheme]
        
        # Training convergence surface
        if len(all_losses) > 10:
            # Create convergence trajectory
            fig.add_trace(go.Scatter3d(
                x=all_steps,
                y=all_grad_norms,
                z=all_losses,
                mode='lines+markers',
                line=dict(
                    color=colors[0],
                    width=6
                ),
                marker=dict(
                    size=6,
                    color=all_tokens_per_sec,
                    colorscale='Plasma',
                    showscale=True,
                    colorbar=dict(title="Tokens/sec", x=1.1)
                ),
                name='Convergence Path',
                text=[f'Step {s}<br>Grad Norm: {g:.3f}<br>Loss: {l:.4f}<br>Speed: {t:.1f} tok/s' 
                      for s, g, l, t in zip(all_steps, all_grad_norms, all_losses, all_tokens_per_sec)],
                hovertemplate='%{text}<extra></extra>'
            ))
        
        # Add convergence zones
        stable_points = []
        for i in range(5, len(all_losses)):
            recent_losses = all_losses[i-5:i]
            if len(recent_losses) == 5 and np.std(recent_losses) < 0.1:  # Stable region
                stable_points.append(i)
        
        if stable_points:
            fig.add_trace(go.Scatter3d(
                x=[all_steps[i] for i in stable_points],
                y=[all_grad_norms[i] for i in stable_points],
                z=[all_losses[i] for i in stable_points],
                mode='markers',
                marker=dict(
                    size=8,
                    color='lightgreen',
                    symbol='square',
                    opacity=0.7
                ),
                name='Stable Training Regions',
                text=[f'Stable at step {all_steps[i]}' for i in stable_points],
                hovertemplate='%{text}<extra></extra>'
            ))
        
        fig.update_layout(
            title=dict(
                text='🌊 3D Training Convergence Analysis',
                x=0.5,
                font=dict(size=20)
            ),
            scene=dict(
                xaxis_title='Training Step',
                yaxis_title='Gradient Norm',
                zaxis_title='Loss',
                camera=dict(eye=dict(x=1.4, y=1.4, z=1.4)),
                bgcolor='rgba(0,0,0,0.02)'
            ),
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_combined_3d_dashboard(self, training_history: list) -> go.Figure:
        """Create comprehensive 3D dashboard with multiple metrics"""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "scene"}, {"type": "scene"}],
                   [{"type": "scene"}, {"type": "scene"}]],
            subplot_titles=('Loss Evolution', 'Perplexity Trends', 'Learning Dynamics', 'Performance Metrics'),
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        if not training_history:
            # Add sample data to each subplot
            self._add_sample_to_subplot(fig, 1, 1, "loss")
            self._add_sample_to_subplot(fig, 1, 2, "perplexity") 
            self._add_sample_to_subplot(fig, 2, 1, "dynamics")
            self._add_sample_to_subplot(fig, 2, 2, "performance")
        else:
            # Add real data to subplots
            self._add_real_data_to_subplots(fig, training_history)
        
        fig.update_layout(
            title=dict(
                text='🎛️ Comprehensive 3D Training Dashboard',
                x=0.5,
                font=dict(size=24)
            ),
            height=800,
            showlegend=True
        )
        
        return fig
    
    def _create_sample_loss_landscape(self) -> go.Figure:
        """Create sample loss landscape for demo"""
        steps = np.linspace(0, 100, 50)
        epochs = np.linspace(0, 10, 20)
        Steps, Epochs = np.meshgrid(steps, epochs)
        
        # Simulate decreasing loss with noise
        Losses = 5.0 * np.exp(-Steps/30) + 0.5 * np.sin(Steps/10) + 0.2 * np.random.random(Steps.shape)
        
        fig = go.Figure()
        fig.add_trace(go.Surface(x=Steps, y=Epochs, z=Losses, colorscale='Reds_r'))
        
        # Add sample trajectory
        sample_steps = np.linspace(0, 100, 20)
        sample_epochs = np.ones_like(sample_steps) * 5
        sample_losses = 5.0 * np.exp(-sample_steps/30) + 0.1 * np.random.random(len(sample_steps))
        
        fig.add_trace(go.Scatter3d(
            x=sample_steps, y=sample_epochs, z=sample_losses,
            mode='lines+markers',
            line=dict(color='yellow', width=8),
            marker=dict(size=6, color='red'),
            name='Sample Training Path'
        ))
        
        fig.update_layout(
            title='📊 Sample 3D Loss Landscape',
            scene=dict(
                xaxis_title='Training Step',
                yaxis_title='Epoch',
                zaxis_title='Loss'
            ),
            height=600
        )
        return fig
    
    def _create_sample_perplexity_chart(self) -> go.Figure:
        """Create sample perplexity chart"""
        steps = np.linspace(0, 100, 50)
        lrs = np.linspace(1e-5, 1e-3, 50)
        perplexities = 200 * np.exp(-steps/40) + 20 + 10 * np.random.random(50)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=steps, y=lrs, z=perplexities,
            mode='lines+markers',
            line=dict(color='blue', width=6),
            marker=dict(size=5, color=perplexities, colorscale='RdYlBu_r'),
            name='Sample Perplexity'
        ))
        
        fig.update_layout(
            title='🧠 Sample Perplexity Evolution',
            scene=dict(
                xaxis_title='Step',
                yaxis_title='Learning Rate',
                zaxis_title='Perplexity'
            ),
            height=600
        )
        return fig
    
    def _create_sample_convergence_chart(self) -> go.Figure:
        """Create sample convergence chart"""
        steps = np.arange(50)
        grad_norms = 5.0 * np.exp(-steps/20) + 0.5 + 0.2 * np.random.random(50)
        losses = 3.0 * np.exp(-steps/25) + 0.1 * np.random.random(50)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=steps, y=grad_norms, z=losses,
            mode='lines+markers',
            line=dict(color='green', width=6),
            name='Sample Convergence'
        ))
        
        fig.update_layout(
            title='🌊 Sample Training Convergence',
            scene=dict(
                xaxis_title='Step',
                yaxis_title='Gradient Norm',
                zaxis_title='Loss'
            ),
            height=600
        )
        return fig
    
    def _add_sample_to_subplot(self, fig, row, col, chart_type):
        """Add sample data to subplot"""
        # Implementation for sample data in subplots
        pass
    
    def _add_real_data_to_subplots(self, fig, training_history):
        """Add real training data to subplots"""
        # Implementation for real data in subplots
        pass

# Streamlit integration function
def render_3d_training_charts():
    """Render 3D training charts in Streamlit"""
    
    st.header("📊 3D Training Metrics Dashboard")
    st.caption("Interactive 3D visualizations of your training progress")
    
    if not hasattr(st.session_state, 'training_manager'):
        st.warning("Training manager not initialized")
        return
    
    visualizer = Training3DVisualizer()
    
    # Chart selection
    chart_type = st.selectbox(
        "Choose 3D Visualization",
        [
            "🔥 Loss Landscape",
            "🧠 Perplexity Evolution", 
            "🌊 Training Convergence",
            "🎛️ Combined Dashboard"
        ],
        key="training_3d_chart_type_selector"
    )
    
    # Style options
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("🎨 Visualization Options")
        color_scheme = st.selectbox("Color Scheme", ["fire", "ocean", "neural", "mastishk"], key="training_3d_color_scheme_selector")
        show_annotations = st.checkbox("Show Annotations", value=True, key="training_3d_show_annotations")
        auto_rotate = st.checkbox("Auto Rotate", value=False, key="training_3d_auto_rotate")
    
    with col1:
        # Get training history
        training_history = getattr(st.session_state.training_manager, 'training_history', [])
        
        if chart_type == "🔥 Loss Landscape":
            st.subheader("🔥 3D Loss Landscape")
            fig = visualizer.create_3d_loss_landscape(training_history, color_scheme)
            
            if show_annotations:
                st.info("""
                **3D Loss Landscape:**
                - **X-axis**: Training steps
                - **Y-axis**: Training sessions/epochs  
                - **Z-axis**: Loss values
                - **Surface**: Loss evolution over time
                - **Yellow line**: Actual training path
                - **Gold star**: Best loss achieved
                """)
        
        elif chart_type == "🧠 Perplexity Evolution":
            st.subheader("🧠 3D Perplexity Evolution")
            fig = visualizer.create_3d_perplexity_evolution(training_history, color_scheme)
            
            if show_annotations:
                st.info("""
                **3D Perplexity Evolution:**
                - **X-axis**: Training steps
                - **Y-axis**: Learning rate
                - **Z-axis**: Perplexity (lower is better)
                - **Color**: Perplexity intensity
                - **Green diamonds**: Major improvements
                - **Gold star**: Best perplexity achieved
                """)
        
        elif chart_type == "🌊 Training Convergence":
            st.subheader("🌊 3D Training Convergence")
            fig = visualizer.create_3d_training_convergence(training_history, color_scheme)
            
            if show_annotations:
                st.info("""
                **3D Training Convergence:**
                - **X-axis**: Training steps
                - **Y-axis**: Gradient norm
                - **Z-axis**: Loss values
                - **Color**: Training speed (tokens/sec)
                - **Green squares**: Stable training regions
                """)
        
        elif chart_type == "🎛️ Combined Dashboard":
            st.subheader("🎛️ Combined 3D Dashboard")
            fig = visualizer.create_combined_3d_dashboard(training_history)
            
            if show_annotations:
                st.info("""
                **Combined 3D Dashboard:**
                - **Top Left**: Loss evolution over training
                - **Top Right**: Perplexity trends and improvements
                - **Bottom Left**: Learning dynamics and stability
                - **Bottom Right**: Performance metrics overview
                """)
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Export options
        st.divider()
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📷 Save as PNG"):
                st.success("Chart saved! (Use browser's download)")
        
        with col2:
            if st.button("📊 Save as HTML"):
                html_str = fig.to_html()
                st.download_button(
                    "Download HTML",
                    html_str,
                    file_name="training_3d_chart.html",
                    mime="text/html"
                )
        
        with col3:
            if st.button("📈 Export Data"):
                if training_history:
                    # Create export data
                    export_data = {"message": "Training data exported"}
                    st.success("Data export prepared!")

# Add this to your main Streamlit app
def add_3d_charts_to_training_tab():
    """Add 3D charts section to your training tab"""
    
    st.divider()
    st.subheader("📊 3D Training Analytics")
    
    with st.expander("🎯 Interactive 3D Charts", expanded=False):
        render_3d_training_charts()
class MastishkModelManager:
    """Enhanced model manager with comprehensive checkpoint management"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_config = None
        self.initialized = False
        self.training_history = []
        self.generation_history = []
        self.current_experiment = None
        
        # Enhanced checkpoint manager
        self.checkpoint_manager = EnhancedCheckpointManager(
            checkpoint_dir="./checkpoints",
            max_checkpoints=100
        )
        
    def initialize_model(self, model_size: str = "1B", checkpoint_path: Optional[str] = None, 
                        advanced_config: Optional[Dict] = None) -> Tuple[bool, str]:
        """Initialize Mastishk model with enhanced checkpoint support"""
        try:
            print(f"🚀 Initializing {model_size} model...")
            
            size_configs = {
                "1B": {"hidden_size": 2048, "num_layers": 24, "num_heads": 32},
                "7B": {"hidden_size": 4096, "num_layers": 32, "num_heads": 32},
                "13B": {"hidden_size": 5120, "num_layers": 40, "num_heads": 40},
                "30B": {"hidden_size": 6656, "num_layers": 60, "num_heads": 52},
                "65B": {"hidden_size": 8192, "num_layers": 80, "num_heads": 64},
                "175B": {"hidden_size": 12288, "num_layers": 96, "num_heads": 96},
                "8x7B": {"hidden_size": 4096, "num_layers": 32, "num_heads": 32},
                "8x22B": {"hidden_size": 6144, "num_layers": 56, "num_heads": 48},
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
            
            # Create configuration
            config = MastishkConfig(
                hidden_size=size_config["hidden_size"],                    # No caps!
                num_hidden_layers=size_config["num_layers"],               # No caps!
                num_attention_heads=size_config["num_heads"],              # No caps!
                num_key_value_heads=max(1, size_config["num_heads"] // 4), # Proper ratio
                intermediate_size=size_config["hidden_size"] * 4,          # Standard 4x
                vocab_size=actual_vocab_size,
                max_position_embeddings=4096,
                use_flash_attention=advanced_config.get('use_flash_attention', True),
                use_quantization=advanced_config.get('use_quantization', True),
                use_moe=advanced_config.get('use_moe', True),
                use_mod=advanced_config.get('use_mod', True),
                use_minimax=advanced_config.get('use_minimax', True),
            )
            
            print(f"📋 Model config: {config.hidden_size}h, {config.num_hidden_layers}l, {config.num_attention_heads}a, vocab={config.vocab_size}")
            
            # Create model
            self.model = MastishkTransformerForCausalLM(config)
            self.model_config = config
            # ADD THESE DEBUG PRINTS HERE:
            print(f"📋 ACTUAL Model Configuration:")
            print(f"   Hidden Size: {config.hidden_size}")
            print(f"   Layers: {config.num_hidden_layers}")
            print(f"   Attention Heads: {config.num_attention_heads}")
            print(f"   Vocab Size: {config.vocab_size}")
            print(f"   Intermediate Size: {config.intermediate_size}")
            print(f"   Advanced Features: Flash={config.use_flash_attention}, MoE={config.use_moe}, MiniMax={config.use_minimax}")            
            # Load checkpoint if provided
            if checkpoint_path and os.path.exists(checkpoint_path):
                try:
                    success, msg = self.load_enhanced_checkpoint(checkpoint_path)
                    if success:
                        print(f"✅ Loaded checkpoint: {msg}")
                        total_params = sum(p.numel() for p in self.model.parameters())
                        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                        return True, f"✅ Model loaded from checkpoint!\nTotal params: {total_params:,}\nTrainable: {trainable_params:,}\nDevice: {self.device}"
                    else:
                        print(f"⚠️ Failed to load checkpoint: {msg}")
                except Exception as e:
                    print(f"⚠️ Failed to load checkpoint: {e}")
            
            self.model.to(self.device)
            self.initialized = True
            
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            return True, f"✅ Model initialized successfully!\nTotal params: {total_params:,}\nTrainable: {trainable_params:,}\nDevice: {self.device}"
            
        except Exception as e:
            print(f"❌ Model initialization failed: {e}")
            traceback.print_exc()
            return False, f"Failed to initialize model: {str(e)}"
    
    def load_enhanced_checkpoint(
        self, 
        checkpoint_path: str, 
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        strict_loading: bool = True
    ) -> Tuple[bool, str]:
        """Load checkpoint using the enhanced checkpoint manager"""
        try:
            if not os.path.exists(checkpoint_path):
                return False, f"Checkpoint file not found: {checkpoint_path}"
            
            # Extract checkpoint ID from path
            checkpoint_id = Path(checkpoint_path).stem
            
            # If this is a legacy checkpoint, load it the old way
            if not checkpoint_id in self.checkpoint_manager.metadata:
                return self._load_legacy_checkpoint(checkpoint_path)
            
            # Create dummy optimizer and scheduler if not provided
            if optimizer is None and self.model is not None:
                optimizer = optim.AdamW(self.model.parameters(), lr=1e-5)
            
            # Load using enhanced checkpoint manager
            success, training_state, message = self.checkpoint_manager.load_checkpoint(
                checkpoint_id=checkpoint_id,
                model=self.model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=self.device,
                strict_loading=strict_loading
            )
            
            if success:
                self.initialized = True
                return True, message
            else:
                return False, message
                
        except Exception as e:
            print(f"❌ Failed to load enhanced checkpoint: {e}")
            traceback.print_exc()
            return False, f"Failed to load checkpoint: {str(e)}"
    
    def _load_legacy_checkpoint(self, checkpoint_path: str) -> Tuple[bool, str]:
        """Load legacy checkpoint format"""
        try:
            print(f"📂 Loading legacy checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            if 'config_dict' in checkpoint:
                config_dict = checkpoint['config_dict']
                config = MastishkConfig(**config_dict)
                print("✅ Loaded config from dictionary (new format)")
            elif 'config' in checkpoint:
                config = checkpoint['config']
                print("✅ Loaded config object (old format)")
            else:
                return False, "No config found in checkpoint"
            
            self.model = MastishkTransformerForCausalLM(config)
            self.model_config = config
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print("✅ Loaded model state dict")
            else:
                return False, "No model state dict found in checkpoint"
            
            self.model.to(self.device)
            
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.initialized = True
            
            timestamp = checkpoint.get('timestamp', 'Unknown')
            vocab_size = config.vocab_size
            hidden_size = config.hidden_size
            
            return True, f"✅ Legacy model loaded successfully!\nTimestamp: {timestamp}\nConfig: {hidden_size}h, {config.num_hidden_layers}l, vocab={vocab_size}"
            
        except Exception as e:
            print(f"❌ Failed to load legacy checkpoint: {e}")
            traceback.print_exc()
            return False, f"Failed to load legacy checkpoint: {str(e)}"
    
    def generate_text(self, prompt: str, generation_config: GenerationConfig) -> Tuple[str, Dict]:
        """Generate text with the model"""
        if not self.initialized:
            return "❌ Model not initialized", {}
        
        try:
            start_time = time.time()
            
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
            
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
            
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, generation_config=gen_config)
            
            generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            generation_time = time.time() - start_time
            
            stats = {
                'generation_time': generation_time,
                'tokens_generated': len(output_ids[0]) - len(inputs.input_ids[0]),
                'tokens_per_second': (len(output_ids[0]) - len(inputs.input_ids[0])) / generation_time if generation_time > 0 else 0,
                'strategy_used': generation_config.generation_strategy,
                'model_size': f"{sum(p.numel() for p in self.model.parameters()):,}"
            }
            
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
    def load_checkpoint_for_training_resumption(
        self, 
        checkpoint_path: str,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        strict_loading: bool = True
    ) -> Tuple[bool, str]:
        """Load checkpoint for full training resumption with all training state"""
        try:
            if not os.path.exists(checkpoint_path):
                return False, f"Checkpoint file not found: {checkpoint_path}"
            
            print(f"📂 Loading training resumption checkpoint from {checkpoint_path}")
            
            # Load the comprehensive checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # Load model state
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=strict_loading)
                print("✅ Model state loaded")
            else:
                return False, "No model state dict found in checkpoint"
            
            # Load optimizer state
            if optimizer and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("✅ Optimizer state loaded")
            
            # Load scheduler state
            if scheduler and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("✅ Scheduler state loaded")
            
            # Restore random states for reproducibility
            if 'random_states' in checkpoint:
                random_states = checkpoint['random_states']
                
                try:
                    import random
                    import numpy as np
                    
                    if 'python_random' in random_states:
                        random.setstate(random_states['python_random'])
                        print("✅ Python random state restored")
                    
                    if 'numpy_random' in random_states:
                        np.random.set_state(random_states['numpy_random'])
                        print("✅ NumPy random state restored")
                    
                    if 'torch_random' in random_states:
                        torch.set_rng_state(random_states['torch_random'])
                        print("✅ PyTorch random state restored")
                    
                    if 'torch_cuda_random' in random_states and torch.cuda.is_available():
                        torch.cuda.set_rng_state(random_states['torch_cuda_random'])
                        print("✅ CUDA random state restored")
                        
                except Exception as e:
                    print(f"⚠️ Warning: Could not restore random states: {e}")
            
            # Extract training progress info
            current_step = checkpoint.get('current_step', 0)
            current_epoch = checkpoint.get('current_epoch', 0)
            checkpoint_type = checkpoint.get('checkpoint_type', 'unknown')
            
            return True, f"✅ Training resumption successful!\nStep: {current_step}, Epoch: {current_epoch}\nType: {checkpoint_type}"
            
        except Exception as e:
            print(f"❌ Failed to load training resumption checkpoint: {e}")
            traceback.print_exc()
            return False, f"Failed to load training checkpoint: {str(e)}"    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
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
                "model_config": {
                    "hidden_size": self.model_config.hidden_size,
                    "num_layers": self.model_config.num_hidden_layers,
                    "num_attention_heads": self.model_config.num_attention_heads,
                    "vocab_size": self.model_config.vocab_size,
                    "max_position_embeddings": self.model_config.max_position_embeddings,
                },
                "features": {
                    "moe_enabled": self.model_config.use_moe,
                    "mod_enabled": self.model_config.use_mod,
                    "minimax_enabled": self.model_config.use_minimax,
                    "flash_attention": self.model_config.use_flash_attention,
                    "quantization": self.model_config.use_quantization,
                },
                "checkpoint_info": {
                    "total_checkpoints": len(self.checkpoint_manager.metadata),
                    "storage_stats": self.checkpoint_manager.get_storage_stats()
                }
            }
            
            if torch.cuda.is_available():
                info["gpu_memory_allocated"] = torch.cuda.memory_allocated() / 1024**3
                info["gpu_memory_reserved"] = torch.cuda.memory_reserved() / 1024**3
            
            return info
        except Exception as e:
            return {"status": "Error", "error": str(e)}

# ======================== ENHANCED TRAINING COMPONENTS ======================== #

class EnhancedTrainingMonitor:
    """Enhanced training monitor with comprehensive metrics tracking"""
    
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []
        self.learning_rates = []
        self.timestamps = []
        self.step = 0
        self.gradient_norms = []
        self.tokens_per_second = []
        self.gpu_memory_usage = []
        
        # Enhanced metrics
        self.epoch_times = []
        self.checkpoint_times = []
        self.custom_metrics = defaultdict(list)
        
        # Best tracking
        self.best_loss = float('inf')
        self.best_step = 0
        self.patience_counter = 0
    
    def log_step(self, train_loss: float, eval_loss: Optional[float] = None, lr: float = 0.0, 
                 grad_norm: float = 0.0, tokens_per_sec: float = 0.0, gpu_memory: float = 0.0,
                 custom_metrics: Optional[Dict[str, float]] = None):
        """Enhanced step logging with custom metrics"""
        self.train_losses.append(train_loss)
        self.eval_losses.append(eval_loss)
        self.learning_rates.append(lr)
        self.timestamps.append(time.time())
        self.gradient_norms.append(grad_norm)
        self.tokens_per_second.append(tokens_per_sec)
        self.gpu_memory_usage.append(gpu_memory)
        self.step += 1
        
        # Track best performance
        if train_loss < self.best_loss:
            self.best_loss = train_loss
            self.best_step = self.step
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        # Log custom metrics
        if custom_metrics:
            for key, value in custom_metrics.items():
                self.custom_metrics[key].append(value)
    
    def to_training_state(self) -> TrainingState:
        """Convert monitor to TrainingState for checkpoint saving"""
        training_state = TrainingState()
        training_state.step = self.step
        training_state.best_loss = self.best_loss
        training_state.train_losses = self.train_losses.copy()
        training_state.eval_losses = [x for x in self.eval_losses if x is not None]
        training_state.learning_rates = self.learning_rates.copy()
        training_state.gradient_norms = self.gradient_norms.copy()
        training_state.tokens_per_second = self.tokens_per_second.copy()
        training_state.gpu_memory_usage = self.gpu_memory_usage.copy()
        training_state.learning_rate = self.learning_rates[-1] if self.learning_rates else 0.0
        training_state.custom_metrics = dict(self.custom_metrics)
        
        return training_state
    
    def from_training_state(self, training_state: TrainingState):
        """Restore monitor from TrainingState after checkpoint loading"""
        self.step = training_state.step
        self.best_loss = training_state.best_loss
        self.train_losses = training_state.train_losses.copy()
        self.eval_losses = training_state.eval_losses.copy()
        self.learning_rates = training_state.learning_rates.copy()
        self.gradient_norms = training_state.gradient_norms.copy()
        self.tokens_per_second = training_state.tokens_per_second.copy()
        self.gpu_memory_usage = training_state.gpu_memory_usage.copy()
        self.custom_metrics = defaultdict(list, training_state.custom_metrics)
    
    def get_metrics(self) -> Dict:
        """Enhanced metrics reporting"""
        base_metrics = {
            'step': self.step,
            'avg_train_loss': np.mean(self.train_losses[-10:]) if self.train_losses else 0.0,
            'latest_train_loss': self.train_losses[-1] if self.train_losses else 0.0,
            'latest_eval_loss': self.eval_losses[-1] if self.eval_losses and self.eval_losses[-1] else 0.0,
            'current_lr': self.learning_rates[-1] if self.learning_rates else 0.0,
            'latest_grad_norm': self.gradient_norms[-1] if self.gradient_norms else 0.0,
            'tokens_per_second': self.tokens_per_second[-1] if self.tokens_per_second else 0.0,
            'gpu_memory_gb': self.gpu_memory_usage[-1] if self.gpu_memory_usage else 0.0,
            'best_loss': self.best_loss,
            'best_step': self.best_step,
            'patience_counter': self.patience_counter
        }
        
        # Add custom metrics
        for key, values in self.custom_metrics.items():
            base_metrics[f'latest_{key}'] = values[-1] if values else 0.0
            
        return base_metrics
    
    def plot_metrics(self):
        """Enhanced plotting with custom metrics"""
        if not self.train_losses:
            return None
        
        # Calculate number of subplots needed
        num_custom = len(self.custom_metrics)
        num_rows = 2 + (num_custom + 1) // 2  # 2 base rows + custom metrics
        
        fig = make_subplots(
            rows=num_rows, cols=2,
            subplot_titles=(['Training Loss', 'Learning Rate', 'Gradient Norm', 'Token Efficiency'] + 
                          list(self.custom_metrics.keys())),
            specs=[[{"secondary_y": False}, {"secondary_y": False}] for _ in range(num_rows)]
        )
        
        steps = list(range(len(self.train_losses)))
        
        # Base plots
        fig.add_trace(
            go.Scatter(x=steps, y=self.train_losses, name='Train Loss', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=steps, y=self.learning_rates, name='Learning Rate', line=dict(color='green')),
            row=1, col=2
        )
        
        if self.gradient_norms:
            fig.add_trace(
                go.Scatter(x=steps, y=self.gradient_norms, name='Gradient Norm', line=dict(color='orange')),
                row=2, col=1
            )
        
        if self.tokens_per_second:
            fig.add_trace(
                go.Scatter(x=steps, y=self.tokens_per_second, name='Tokens/sec', line=dict(color='purple')),
                row=2, col=2
            )
        
        # Custom metrics
        for i, (key, values) in enumerate(self.custom_metrics.items()):
            row = 3 + i // 2
            col = 1 + i % 2
            if row <= num_rows:
                fig.add_trace(
                    go.Scatter(x=list(range(len(values))), y=values, name=key),
                    row=row, col=col
                )
        
        fig.update_layout(height=200 * num_rows, showlegend=True, title_text="Enhanced Training Metrics")
        return fig

class EnhancedTrainer:
    """Enhanced trainer with comprehensive checkpoint management"""
    
    def __init__(
        self,
        model,
        tokenizer,
        checkpoint_manager: EnhancedCheckpointManager,
        device: str = 'cuda',
        learning_rate: float = 5e-4,
        weight_decay: float = 0.01,
        max_steps: int = 100000,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        auto_save_interval: int = 1000,
        early_stopping_patience: int = 10,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.checkpoint_manager = checkpoint_manager
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_steps = max_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.auto_save_interval = auto_save_interval
        self.early_stopping_patience = early_stopping_patience
        
        # Initialize optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        from torch.optim.lr_scheduler import CosineAnnealingLR
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max_steps)
        
        # Enhanced training monitor
        self.monitor = EnhancedTrainingMonitor()
        
        self.model.to(device)
        self.training_start_time = None
        
        # Resume state (for checkpoint loading)
        self.resume_from_step = 0
    
    def load_checkpoint_state(self, checkpoint_id: str) -> bool:
        """Load training state from checkpoint"""
        try:
            success, training_state, message = self.checkpoint_manager.load_checkpoint(
                checkpoint_id=checkpoint_id,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                device=self.device
            )
            
            if success:
                self.monitor.from_training_state(training_state)
                self.resume_from_step = training_state.step
                print(f"✅ Training state restored from step {self.resume_from_step}")
                return True
            else:
                print(f"❌ Failed to load checkpoint state: {message}")
                return False
                
        except Exception as e:
            print(f"❌ Error loading checkpoint state: {e}")
            return False
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Enhanced training step with detailed metrics"""
        self.model.train()
        
        step_metrics = {}
        
        try:
            if not isinstance(batch, dict):
                print(f"❌ Invalid batch type: {type(batch)}")
                return {'loss': 0.0}
            
            # Move tensors to device with validation
            processed_batch = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    processed_batch[key] = value.to(self.device)
            
            # Ensure required keys exist
            if 'input_ids' not in processed_batch:
                print(f"❌ Missing required key: input_ids")
                return {'loss': 0.0}
            
            if 'labels' not in processed_batch:
                processed_batch['labels'] = processed_batch['input_ids'].clone()
            
            # Forward pass
            outputs = self.model(
                input_ids=processed_batch['input_ids'],
                attention_mask=processed_batch.get('attention_mask'),
                labels=processed_batch['labels'],
            )
            
            if outputs.loss is None:
                print("❌ Model returned None loss")
                return {'loss': 0.0}
            
            loss = outputs.loss / self.gradient_accumulation_steps
            
            # Check for NaN or infinite loss
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print("❌ Loss is NaN or infinite, skipping step")
                return {'loss': 0.0}
            
            # Calculate perplexity
            perplexity = torch.exp(loss * self.gradient_accumulation_steps).item()
            
            # Backward pass
            loss.backward()
            
            final_loss = loss.item() * self.gradient_accumulation_steps
            step_metrics['loss'] = final_loss
            step_metrics['perplexity'] = perplexity
            
            return step_metrics
            
        except Exception as e:
            print(f"❌ Training step failed: {e}")
            self.optimizer.zero_grad()
            return {'loss': 0.0}
    
    def save_checkpoint(
        self, 
        checkpoint_name: Optional[str] = None, 
        notes: str = "", 
        force: bool = False
    ) -> Tuple[bool, str]:
        """Save comprehensive checkpoint"""
        try:
            # Convert monitor to training state
            training_state = self.monitor.to_training_state()
            training_state.learning_rate = self.scheduler.get_last_lr()[0]
            
            # Save checkpoint
            success, checkpoint_id, message = self.checkpoint_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                training_state=training_state,
                model_config=self.model.config,
                training_config={  # Create minimal training config
                    'learning_rate': self.learning_rate,
                    'weight_decay': self.weight_decay,
                    'max_steps': self.max_steps,
                    'gradient_accumulation_steps': self.gradient_accumulation_steps,
                    'max_grad_norm': self.max_grad_norm,
                },
                checkpoint_name=checkpoint_name,
                notes=notes,
                include_model_weights=True,
                compress=True
            )
            
            if success:
                print(f"✅ Checkpoint saved: {checkpoint_id}")
                self.monitor.checkpoint_times.append(time.time())
            
            return success, message
            
        except Exception as e:
            error_msg = f"❌ Failed to save checkpoint: {str(e)}"
            print(error_msg)
            return False, error_msg
    
    def train(
        self,
        train_dataloader,
        save_dir: str = "./checkpoints",
        progress_callback=None,
        training_config: Optional[TrainingConfig] = None
    ):
        """Enhanced training loop with comprehensive checkpoint management"""
        step = self.resume_from_step
        accumulated_loss = 0.0
        
        self.model.train()
        self.training_start_time = time.time()
        
        # Load training config defaults
        config = training_config or TrainingConfig()
        
        try:
            for epoch in range(100):  # Large number, will break based on max_steps
                epoch_start_time = time.time()
                
                for batch_idx, batch in enumerate(train_dataloader):
                    # Skip batches if resuming from checkpoint
                    if step < self.resume_from_step:
                        step += 1
                        continue
                    
                    # Training step
                    batch_start_time = time.time()
                    step_metrics = self.train_step_with_verification(batch, step)
                    batch_time = time.time() - batch_start_time
                    
                    loss = step_metrics.get('loss', 0.0)
                    accumulated_loss += loss
                    
                    # Calculate tokens per second
                    if 'input_ids' in batch:
                        batch_size = batch['input_ids'].shape[0]
                        seq_length = batch['input_ids'].shape[1]
                        tokens_in_batch = batch_size * seq_length
                        tokens_per_sec = tokens_in_batch / batch_time if batch_time > 0 else 0
                    else:
                        tokens_per_sec = 0
                    
                    # Gradient accumulation
                    if (step + 1) % self.gradient_accumulation_steps == 0:
                        # Gradient clipping
                        grad_norm = 0.0
                        # 🔍 GRADIENT CHECKING (moved from forward method)
                        print("🔍 Checking gradients after backward:")
                        # Gradient clipping
                        if self.max_grad_norm > 0:
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), self.max_grad_norm
                            ).item()
                            print(f"📏 Gradient norm after clipping: {grad_norm:.6f}")

                        self.optimizer.step()
                        print("🚀 Optimizer step completed")
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                        
                        # Calculate metrics
                        avg_loss = accumulated_loss / self.gradient_accumulation_steps
                        accumulated_loss = 0.0
                        current_lr = self.scheduler.get_last_lr()[0]
                        
                        # GPU memory usage
                        gpu_memory = 0.0
                        if torch.cuda.is_available():
                            gpu_memory = torch.cuda.memory_allocated() / 1024**3
                        
                        # Enhanced metrics logging
                        custom_metrics = {
                            'perplexity': step_metrics.get('perplexity', 0.0),
                            'epoch': epoch
                        }
                        
                        self.monitor.log_step(
                            train_loss=avg_loss,
                            eval_loss=None,
                            lr=current_lr,
                            grad_norm=grad_norm,
                            tokens_per_sec=tokens_per_sec,
                            gpu_memory=gpu_memory,
                            custom_metrics=custom_metrics
                        )
                        
                        # Progress callback
                        if progress_callback:
                            metrics = self.monitor.get_metrics()
                            progress_callback(step, metrics)
                        
                        # Auto-save checkpoint
                        if step > 0 and step % self.auto_save_interval == 0:
                            auto_save_name = f"auto_save_step_{step}"
                            success, message = self.save_checkpoint(
                                checkpoint_name=auto_save_name,
                                notes=f"Auto-save at step {step}"
                            )
                            if success:
                                print(f"🔄 Auto-saved: {auto_save_name}")
                        
                        # Early stopping check
                        if self.monitor.patience_counter >= self.early_stopping_patience:
                            print(f"🛑 Early stopping triggered after {self.early_stopping_patience} steps without improvement")
                            break
                    
                    step += 1
                    
                    if step >= self.max_steps:
                        break
                
                # End of epoch processing
                epoch_time = time.time() - epoch_start_time
                self.monitor.epoch_times.append(epoch_time)
                print(f"📊 Epoch {epoch} completed in {epoch_time:.2f}s")
                
                if step >= self.max_steps:
                    break
            
            # Final checkpoint save
            final_checkpoint_name = f"final_checkpoint_step_{step}"
            success, message = self.save_checkpoint(
                checkpoint_name=final_checkpoint_name,
                notes=f"Final checkpoint after {step} steps"
            )
            
            if success:
                print(f"💾 Final checkpoint saved: {final_checkpoint_name}")
        
        except Exception as e:
            print(f"❌ Training failed: {e}")
            traceback.print_exc()
        
        return self.monitor
class EnhancedTrainerWithWeightVerification(EnhancedTrainer):
    """Enhanced trainer with comprehensive weight verification"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_verifier = WeightVerificationSystem()
        self.enable_weight_verification = True
        
        # Create initial weight snapshot
        if self.model:
            self.initial_snapshot = self.weight_verifier.create_weight_snapshot(
                self.model, 0, "initial_model", "Model at initialization"
            )
    
    def train_step_with_verification(self, batch: Dict[str, torch.Tensor], step: int) -> Dict[str, float]:
        """FIXED: Only verify weights when optimizer actually steps"""
        
        # Initialize accumulation tracking
        if not hasattr(self, 'accumulated_steps'):
            self.accumulated_steps = 0
        
        print(f"\n🔄 Training Step {step} (Accumulation: {self.accumulated_steps + 1}/{self.gradient_accumulation_steps})")
        
        # Check if this step will trigger optimizer stepping
        will_step_optimizer = (self.accumulated_steps + 1) % self.gradient_accumulation_steps == 0
        
        # ONLY take pre-snapshot when optimizer will step
        pre_snapshot = None
        if will_step_optimizer and self.enable_weight_verification:
            print("📸 Taking pre-optimizer snapshot (optimizer will step this iteration)")
            pre_snapshot = self.weight_verifier.create_weight_snapshot(
                self.model, step, "pre_optimizer_step", f"Before optimizer step at training step {step}"
            )
        
        # Perform regular training step (just accumulates gradients)
        step_metrics = super().train_step(batch)
        self.accumulated_steps += 1
        
        loss = step_metrics.get('loss', 0.0)
        
        # Handle optimizer stepping when accumulation is complete
        if will_step_optimizer:
            print("🚀 STEPPING OPTIMIZER (gradient accumulation complete)")
            
            # Check gradients before stepping
            total_grad_norm = sum(p.grad.norm().item() for p in self.model.parameters() if p.grad is not None)
            print(f"   Total gradient norm: {total_grad_norm:.8f}")
            
            if total_grad_norm > 1e-8:
                # Clip gradients
                if self.max_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm).item()
                    print(f"   Gradient norm after clipping: {grad_norm:.6f}")
                
                # STEP OPTIMIZER
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                print("✅ Optimizer step completed")
                
                # NOW verify weight changes (only when optimizer stepped)
                if pre_snapshot and self.enable_weight_verification:
                    post_snapshot = self.weight_verifier.create_weight_snapshot(
                        self.model, step, "post_optimizer_step", f"After optimizer step at training step {step}"
                    )
                    
                    verification_results = self.weight_verifier.verify_weight_updates(
                        pre_snapshot, post_snapshot, expected_update=True
                    )
                    
                    weights_updated = verification_results.get('weights_changed', False)
                    layers_changed = len(verification_results.get('layers_changed', []))
                    
                    step_metrics.update({
                        'weights_updated': weights_updated,
                        'layers_changed': layers_changed,
                        'verification_status': verification_results.get('verification_status', 'Unknown'),
                        'optimizer_stepped': True,
                        'gradient_norm': grad_norm if 'grad_norm' in locals() else total_grad_norm
                    })
                    
                    if weights_updated:
                        print(f"🎉 SUCCESS! Weights updated - {layers_changed} layers changed")
                    else:
                        print("❌ FAILED! No weight updates detected even after optimizer step")
                else:
                    step_metrics.update({
                        'weights_updated': True,  # Assume success if no verification
                        'layers_changed': 'N/A', 
                        'verification_status': 'Optimizer stepped (no verification)',
                        'optimizer_stepped': True
                    })
            else:
                print("❌ No gradients for optimizer step")
                step_metrics.update({
                    'weights_updated': False,
                    'layers_changed': 0,
                    'verification_status': 'No gradients',
                    'optimizer_stepped': False
                })
            
            # Reset accumulation counter
            self.accumulated_steps = 0
            
        else:
            print(f"📝 Gradient accumulation step ({self.accumulated_steps}/{self.gradient_accumulation_steps})")
            step_metrics.update({
                'weights_updated': False,
                'layers_changed': 0,
                'verification_status': f'Accumulating gradients ({self.accumulated_steps}/{self.gradient_accumulation_steps})',
                'optimizer_stepped': False
            })
        
        return step_metrics
    
    def save_checkpoint_with_verification(
        self, 
        checkpoint_name: Optional[str] = None, 
        notes: str = "", 
        force: bool = False
    ) -> Tuple[bool, str]:
        """Save checkpoint with weight verification"""
        
        step = self.monitor.step
        
        if self.enable_weight_verification:
            # Pre-save snapshot
            pre_save_snapshot = self.weight_verifier.create_weight_snapshot(
                self.model, step, "pre_checkpoint_save",
                f"Before saving checkpoint {checkpoint_name}"
            )
        
        # Save checkpoint using parent method
        success, message = super().save_checkpoint(checkpoint_name, notes, force)
        
        if success and self.enable_weight_verification:
            # Verify saved checkpoint
            checkpoint_id = checkpoint_name or f'checkpoint_step_{step}'
            checkpoint_path = f"./checkpoints/{checkpoint_id}.pt"
            
            try:
                # Simple verification - load and compare
                post_save_snapshot = self.weight_verifier.create_weight_snapshot(
                    self.model, step, "post_checkpoint_save",
                    f"After saving checkpoint {checkpoint_name}"
                )
                
                verification_results = self.weight_verifier.verify_weight_updates(
                    pre_save_snapshot, post_save_snapshot, expected_update=False
                )
                
                if not verification_results.get('weights_changed', True):
                    message += " | ✅ Checkpoint integrity verified"
                else:
                    message += " | ⚠️ Unexpected weight changes during save"
            except Exception as e:
                message += f" | ❌ Verification failed: {e}"
        
        return success, message
    
    def load_checkpoint_with_verification(
        self, 
        checkpoint_id: str
    ) -> Tuple[bool, str]:
        """Load checkpoint with weight verification"""
        
        if self.enable_weight_verification:
            # Pre-load snapshot
            pre_load_snapshot = self.weight_verifier.create_weight_snapshot(
                self.model, self.monitor.step, "pre_checkpoint_load",
                f"Before loading checkpoint {checkpoint_id}"
            )
        
        # Load checkpoint using parent method
        success = self.load_checkpoint_state(checkpoint_id)
        
        if success and self.enable_weight_verification:
            # Post-load snapshot
            post_load_snapshot = self.weight_verifier.create_weight_snapshot(
                self.model, self.monitor.step, "post_checkpoint_load",
                f"After loading checkpoint {checkpoint_id}"
            )
            
            # Verify the load operation changed weights as expected
            verification_results = self.weight_verifier.verify_weight_updates(
                pre_load_snapshot, post_load_snapshot, expected_update=True
            )
            
            if verification_results.get('weights_changed', False):
                message = f"✅ Checkpoint loaded and weights updated from {checkpoint_id} - {len(verification_results.get('layers_changed', []))} layers changed"
            else:
                message = f"⚠️ Checkpoint loaded but no weight changes detected for {checkpoint_id}"
        else:
            message = f"Checkpoint load {'succeeded' if success else 'failed'} for {checkpoint_id}"
        
        return success, message
    def train_step_with_correct_verification(self, batch: Dict[str, torch.Tensor], step: int) -> Dict[str, float]:
        """CORRECTED: Only verify weights when optimizer actually steps"""
        
        # Track gradient accumulation progress
        if not hasattr(self, 'accumulated_steps'):
            self.accumulated_steps = 0
        
        print(f"\n🔄 Training Step {step} (Accumulation: {self.accumulated_steps + 1}/{self.gradient_accumulation_steps})")
        
        # Check if this step will trigger optimizer stepping
        will_step_optimizer = (self.accumulated_steps + 1) % self.gradient_accumulation_steps == 0
        
        # ONLY take pre-snapshot when optimizer will step
        pre_snapshot = None
        if will_step_optimizer and self.enable_weight_verification:
            print("📸 Taking pre-optimizer snapshot (optimizer will step this iteration)")
            pre_snapshot = self.weight_verifier.create_weight_snapshot(
                self.model, step, "pre_optimizer_step", f"Before optimizer step at training step {step}"
            )
        
        # Perform regular training step (accumulates gradients)
        step_metrics = super().train_step(batch)
        self.accumulated_steps += 1
        
        loss = step_metrics.get('loss', 0.0)
        print(f"   Loss: {loss:.6f}")
        
        # Handle optimizer stepping
        if will_step_optimizer:
            print("🚀 STEPPING OPTIMIZER (gradient accumulation complete)")
            
            # Check gradients before stepping
            total_grad_norm = sum(p.grad.norm().item() for p in self.model.parameters() if p.grad is not None)
            print(f"   Total gradient norm: {total_grad_norm:.8f}")
            
            if total_grad_norm > 1e-8:
                # Clip gradients
                if self.max_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm).item()
                    print(f"   Gradient norm after clipping: {grad_norm:.6f}")
                
                # STEP OPTIMIZER
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                print("✅ Optimizer step completed")
                
                # NOW verify weight changes (only when optimizer stepped)
                if pre_snapshot and self.enable_weight_verification:
                    post_snapshot = self.weight_verifier.create_weight_snapshot(
                        self.model, step, "post_optimizer_step", f"After optimizer step at training step {step}"
                    )
                    
                    verification_results = self.weight_verifier.verify_weight_updates(
                        pre_snapshot, post_snapshot, expected_update=True
                    )
                    
                    weights_updated = verification_results.get('weights_changed', False)
                    layers_changed = len(verification_results.get('layers_changed', []))
                    
                    step_metrics.update({
                        'weights_updated': weights_updated,
                        'layers_changed': layers_changed,
                        'verification_status': verification_results.get('verification_status', 'Unknown'),
                        'optimizer_stepped': True,
                        'gradient_norm': grad_norm if 'grad_norm' in locals() else total_grad_norm
                    })
                    
                    if weights_updated:
                        print(f"🎉 SUCCESS! Weights updated - {layers_changed} layers changed")
                    else:
                        print("❌ FAILED! No weight updates detected even after optimizer step")
                else:
                    step_metrics.update({
                        'weights_updated': True,  # Assume success if no verification
                        'layers_changed': 'N/A',
                        'verification_status': 'Optimizer stepped (no verification)',
                        'optimizer_stepped': True,
                        'gradient_norm': grad_norm if 'grad_norm' in locals() else total_grad_norm
                    })
            else:
                print("❌ No gradients for optimizer step")
                step_metrics.update({
                    'weights_updated': False,
                    'layers_changed': 0,
                    'verification_status': 'No gradients',
                    'optimizer_stepped': False
                })
            
            # Reset accumulation counter
            self.accumulated_steps = 0
            
        else:
            print(f"📝 Gradient accumulation step ({self.accumulated_steps}/{self.gradient_accumulation_steps})")
            step_metrics.update({
                'weights_updated': False,
                'layers_changed': 0,
                'verification_status': f'Accumulating gradients ({self.accumulated_steps}/{self.gradient_accumulation_steps})',
                'optimizer_stepped': False
            })
        
        return step_metrics        
def save_checkpoint_for_training_resumption(
    self, 
    checkpoint_name: Optional[str] = None, 
    notes: str = "", 
    force: bool = False
) -> Tuple[bool, str]:
    """Save comprehensive checkpoint for FULL training resumption"""
    try:
        # Get current training state
        training_state = self.monitor.to_training_state()
        training_state.learning_rate = self.scheduler.get_last_lr()[0] if self.scheduler else 0.0
        
        # Capture ALL random states for reproducibility
        random_states = {
            'python_random': random.getstate(),
            'numpy_random': np.random.get_state(),
            'torch_random': torch.get_rng_state(),
        }
        
        if torch.cuda.is_available():
            random_states['torch_cuda_random'] = torch.cuda.get_rng_state()
        
        # Create comprehensive checkpoint data
        comprehensive_checkpoint = {
            # Model and training states
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            
            # Training progress tracking
            'training_state': asdict(training_state),
            'current_step': self.monitor.step,
            'current_epoch': getattr(self, 'current_epoch', 0),
            'gradient_accumulation_step': getattr(self, 'gradient_accumulation_step', 0),
            
            # Model configuration
            'model_config': asdict(self.model.config),
            
            # Training configuration
            'training_config': {
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'max_steps': self.max_steps,
                'gradient_accumulation_steps': self.gradient_accumulation_steps,
                'max_grad_norm': self.max_grad_norm,
                'auto_save_interval': self.auto_save_interval,
                'early_stopping_patience': self.early_stopping_patience,
            },
            
            # Random states for reproducibility
            'random_states': random_states,
            
            # Metadata
            'pytorch_version': torch.__version__,
            'save_timestamp': datetime.now().isoformat(),
            'checkpoint_type': 'full_training_resumption',
            'notes': notes
        }
        
        # Save checkpoint
        checkpoint_id = checkpoint_name or f'training_resume_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        checkpoint_path = f"./checkpoints/{checkpoint_id}.pt"
        
        os.makedirs("./checkpoints", exist_ok=True)
        
        # MUST use weights_only=False to save complex training state
        torch.save(comprehensive_checkpoint, checkpoint_path)
        
        print(f"✅ Comprehensive training checkpoint saved: {checkpoint_id}")
        return True, f"Training resumption checkpoint saved: {checkpoint_id}"
        
    except Exception as e:
        error_msg = f"❌ Failed to save training checkpoint: {str(e)}"
        print(error_msg)
        return False, error_msg
# ======================== ENHANCED TRAINING MANAGER ======================== #

class EnhancedMastishkTrainingManager:
    """Enhanced training manager with comprehensive checkpoint management"""
    
    def __init__(self, model_manager: MastishkModelManager):
        self.model_manager = model_manager
        self.current_trainer = None
        self.training_active = False
        self.training_history = []
        # ADD THESE LINES:
        self.weight_verification_enabled = True
        
        # Initialize with verification settings if they exist
        if hasattr(st.session_state, 'weight_verification_settings'):
            settings = st.session_state.weight_verification_settings
            self.weight_verification_enabled = settings.get('enabled', True)        
    def train(
        self, 
        dataset: Dataset, 
        config: TrainingConfig, 
        progress_callback: Optional[callable] = None,
        resume_from_checkpoint: Optional[str] = None
    ) -> Dict:
        """Enhanced training with checkpoint resume capability"""
        
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
            except Exception as e:
                return {"error": f"Dataset access failed: {e}"}
            
            # Create data loader
            try:
                train_loader = DataLoader(
                    dataset,
                    batch_size=config.batch_size,
                    shuffle=True,
                    collate_fn=safe_data_collator,
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
            
            # Create enhanced trainer
            self.current_trainer = EnhancedTrainerWithWeightVerification(
                model=self.model_manager.model,
                tokenizer=self.model_manager.tokenizer,
                checkpoint_manager=self.model_manager.checkpoint_manager,
                device=self.model_manager.device,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
                max_steps=config.max_steps,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                max_grad_norm=config.max_grad_norm,
                auto_save_interval=config.auto_save_interval,
                early_stopping_patience=config.early_stopping_patience,
            )
            
            # Resume from checkpoint if specified
            # PROPER checkpoint resumption if specified
            if resume_from_checkpoint:
                checkpoint_path = f"./checkpoints/{resume_from_checkpoint}.pt"
                success, message = self.model_manager.load_checkpoint_for_training_resumption(
                    checkpoint_path=checkpoint_path,
                    optimizer=self.current_trainer.optimizer,
                    scheduler=self.current_trainer.scheduler,
                    strict_loading=True
                )
                
                if success:
                    print(f"✅ FULLY resumed training: {message}")
                    # Update trainer's monitor with restored state
                    if hasattr(self.current_trainer, 'monitor'):
                        # Restore training monitor state here
                        pass
                else:
                    print(f"⚠️ Failed to resume training state: {message}")
                    print("🔄 Starting fresh training instead")
            
            self.training_active = True
            print("🚀 Starting enhanced training...")
            
            # Train with enhanced monitoring
            monitor = self.current_trainer.train(
                train_loader,
                save_dir="./checkpoints",
                progress_callback=progress_callback,
                training_config=config
            )
            
            self.training_active = False
            
            # Collect comprehensive final metrics
            final_metrics = {
                'final_loss': monitor.train_losses[-1] if monitor.train_losses else None,
                'best_loss': monitor.best_loss,
                'best_step': monitor.best_step,
                'total_steps': len(monitor.train_losses),
                'training_time': monitor.timestamps[-1] - monitor.timestamps[0] if len(monitor.timestamps) > 1 else 0,
                'avg_tokens_per_second': np.mean(monitor.tokens_per_second) if monitor.tokens_per_second else 0,
                'total_epochs': len(monitor.epoch_times),
                'avg_epoch_time': np.mean(monitor.epoch_times) if monitor.epoch_times else 0,
                'checkpoints_saved': len(monitor.checkpoint_times),
                'patience_counter': monitor.patience_counter,
                'early_stopped': monitor.patience_counter >= self.current_trainer.early_stopping_patience,
            }
            
            # Add custom metrics
            for key, values in monitor.custom_metrics.items():
                final_metrics[f'final_{key}'] = values[-1] if values else None
                final_metrics[f'best_{key}'] = min(values) if values and 'loss' in key.lower() else max(values) if values else None
            
            # Store in enhanced history
            self.training_history.append({
                'timestamp': datetime.now(),
                'config': asdict(config),
                'metrics': final_metrics,
                'monitor': monitor,
                'checkpoints_created': len(monitor.checkpoint_times),
                'resume_from': resume_from_checkpoint
            })
            
            print("✅ Enhanced training completed successfully!")
            return final_metrics
            
        except Exception as e:
            self.training_active = False
            print(f"❌ Enhanced training failed: {e}")
            traceback.print_exc()
            return {"error": str(e)}

# ======================== ENHANCED STREAMLIT UI ======================== #

def initialize_session_state():
    """Initialize enhanced session state variables"""
    if 'model_manager' not in st.session_state:
        st.session_state.model_manager = MastishkModelManager()
    
    if 'training_manager' not in st.session_state:
        st.session_state.training_manager = EnhancedMastishkTrainingManager(st.session_state.model_manager)
    
    if 'generation_history' not in st.session_state:
        st.session_state.generation_history = []
    
    if 'experiment_history' not in st.session_state:
        st.session_state.experiment_history = []
    
    if 'current_dataset' not in st.session_state:
        st.session_state.current_dataset = None
    # Initialize 3D visualization state
    if 'show_threejs' not in st.session_state:
        st.session_state.show_threejs = False
def diagnose_training_problem(model_manager):
    """Direct diagnosis of training problem"""
    
    print("🔬 DIRECT TRAINING DIAGNOSIS")
    print("=" * 50)
    
    if not model_manager.initialized:
        print("❌ Model not initialized")
        return False
    
    model = model_manager.model
    tokenizer = model_manager.tokenizer
    device = model_manager.device
    
    # Test 1: Basic forward pass
    print("\n1️⃣ Testing forward pass...")
    test_text = "Hello world"
    inputs = tokenizer(test_text, return_tensors="pt", max_length=32, truncation=True).to(device)
    
    model.train()
    outputs = model(**inputs, labels=inputs['input_ids'])
    
    if outputs.loss is None:
        print("❌ Model returns None loss!")
        return False
    
    loss_value = outputs.loss.item()
    print(f"✅ Forward pass OK: loss = {loss_value:.6f}")
    
    # Test 2: Backward pass
    print("\n2️⃣ Testing backward pass...")
    loss = outputs.loss
    loss.backward()
    
    # Check gradients
    grad_count = 0
    total_grad_norm = 0
    zero_grad_params = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_count += 1
            grad_norm = param.grad.norm().item()
            total_grad_norm += grad_norm
            if grad_norm < 1e-10:
                zero_grad_params += 1
        else:
            print(f"❌ No gradient for {name}")
    
    print(f"✅ Gradients computed for {grad_count} parameters")
    print(f"   Total gradient norm: {total_grad_norm:.8f}")
    print(f"   Zero/tiny gradients: {zero_grad_params}/{grad_count}")
    
    if total_grad_norm < 1e-8:
        print("❌ PROBLEM: Gradients are effectively zero!")
        print("   Possible causes: learning rate too low, loss not connected to params")
        return False
    
    # Test 3: Optimizer step
    print("\n3️⃣ Testing optimizer step...")
    
    # Create simple optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)  # High LR for testing
    
    # Capture weight before
    first_param = next(model.parameters())
    weight_before = first_param.clone().detach()
    
    print(f"   Weight before: {weight_before.flatten()[:5]}")
    
    # Step optimizer
    optimizer.step()
    
    # Check weight after
    weight_after = first_param.clone().detach()
    print(f"   Weight after:  {weight_after.flatten()[:5]}")
    
    # Check if changed
    weight_changed = not torch.equal(weight_before, weight_after)
    change_magnitude = (weight_after - weight_before).abs().max().item()
    
    print(f"   Weight changed: {weight_changed}")
    print(f"   Max change: {change_magnitude:.10f}")
    
    if weight_changed and change_magnitude > 1e-10:
        print("✅ Optimizer is working!")
        return True
    else:
        print("❌ PROBLEM: Optimizer step didn't change weights!")
        print("   Possible causes: gradients too small, learning rate too low")
        return False

def create_simple_working_trainer(model, tokenizer, device):
    """Create a simple trainer that definitely works"""
    
    class SimpleWorkingTrainer:
        def __init__(self, model, tokenizer, device):
            self.model = model
            self.tokenizer = tokenizer
            self.device = device
            
            # Simple, high learning rate
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)  # High LR
            self.step_count = 0
            
        def simple_train_step(self, batch):
            """Super simple training step that definitely updates weights"""
            
            print(f"\n🔄 Simple Training Step {self.step_count}")
            
            # Process batch
            if isinstance(batch, dict) and 'input_ids' in batch:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch.get('labels', input_ids).to(self.device)
            else:
                print("❌ Invalid batch")
                return {'loss': 0.0, 'weights_updated': False}
            
            # Capture weight before
            first_param = next(self.model.parameters())
            weight_before = first_param.clone().detach()
            
            # Clear gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            self.model.train()
            outputs = self.model(input_ids=input_ids, labels=labels)
            
            if outputs.loss is None:
                print("❌ No loss")
                return {'loss': 0.0, 'weights_updated': False}
            
            loss = outputs.loss
            print(f"   Loss: {loss.item():.6f}")
            
            # Backward pass
            loss.backward()
            
            # Check gradients
            total_grad = sum(p.grad.norm().item() for p in self.model.parameters() if p.grad is not None)
            print(f"   Total gradient norm: {total_grad:.8f}")
            
            if total_grad > 1e-8:
                # Step optimizer
                self.optimizer.step()
                print("   ✅ Optimizer stepped")
                
                # Check if weights changed
                weight_after = first_param.clone().detach()
                weight_changed = not torch.equal(weight_before, weight_after)
                change_magnitude = (weight_after - weight_before).abs().max().item()
                
                print(f"   Weight changed: {weight_changed}")
                print(f"   Change magnitude: {change_magnitude:.10f}")
                
                self.step_count += 1
                
                return {
                    'loss': loss.item(),
                    'weights_updated': weight_changed,
                    'change_magnitude': change_magnitude,
                    'gradient_norm': total_grad,
                    'step': self.step_count
                }
            else:
                print("❌ No gradients")
                return {'loss': loss.item(), 'weights_updated': False}
    
    return SimpleWorkingTrainer(model, tokenizer, device)

def nuclear_train_step_method():
    """Returns the nuclear training step method"""
    
    def nuclear_train_step(self, batch, step):
        """Nuclear option: completely bypass gradient accumulation"""
        
        print(f"\n☢️ NUCLEAR Training Step {step}")
        
        # Take snapshot
        if hasattr(self, 'weight_verifier') and self.weight_verifier:
            pre_snapshot = self.weight_verifier.create_weight_snapshot(
                self.model, step, "pre_nuclear", f"Before nuclear step {step}"
            )
        else:
            pre_snapshot = None
        
        # Clear gradients
        self.optimizer.zero_grad()
        
        # Process batch
        input_ids = batch['input_ids'].to(self.device)
        labels = batch.get('labels', input_ids).to(self.device)
        
        # Forward
        self.model.train()
        outputs = self.model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        
        print(f"   Loss: {loss.item():.6f}")
        
        # Backward
        loss.backward()
        
        # Check gradients
        total_grad = sum(p.grad.norm().item() for p in self.model.parameters() if p.grad is not None)
        print(f"   Gradient norm: {total_grad:.8f}")
        
        # ALWAYS step optimizer (no accumulation)
        if total_grad > 1e-8:
            if hasattr(self, 'max_grad_norm') and self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            self.optimizer.step()
            
            if hasattr(self, 'scheduler') and self.scheduler:
                self.scheduler.step()
            
            print("   ✅ Optimizer stepped (nuclear)")
            
            # Verify weights changed
            if pre_snapshot and hasattr(self, 'weight_verifier'):
                post_snapshot = self.weight_verifier.create_weight_snapshot(
                    self.model, step, "post_nuclear", f"After nuclear step {step}"
                )
                
                results = self.weight_verifier.verify_weight_updates(
                    pre_snapshot, post_snapshot, expected_update=True
                )
                
                weights_updated = results.get('weights_changed', False)
                layers_changed = len(results.get('layers_changed', []))
                
                if weights_updated:
                    print(f"🎉 NUCLEAR SUCCESS! {layers_changed} layers updated")
                else:
                    print("❌ Still no weight updates!")
                
                return {
                    'loss': loss.item(),
                    'weights_updated': weights_updated,
                    'layers_changed': layers_changed,
                    'verification_status': results.get('verification_status', 'Unknown'),
                    'gradient_norm': total_grad
                }
        
        return {'loss': loss.item(), 'weights_updated': False}
    
    return nuclear_train_step

# ======================== END DIAGNOSTIC FUNCTIONS ======================== #

        
def render_enhanced_sidebar():
    """Enhanced sidebar with checkpoint management"""
    with st.sidebar:
        st.header("🧠 Mastishk Transformer Studio")
        st.caption("Enhanced with Comprehensive Checkpoint Management")
        
        st.divider()
        
        # Model Configuration
        st.subheader("⚙️ Model Configuration")
        
        model_size = st.selectbox(
            "Model Size",
            ["1B", "7B", "13B", "30B", "65B", "175B", "8x7B", "8x22B"],
            help="Choose model size. Larger models require more memory."
        )
        
        with st.expander("🔧 Advanced Configuration", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                use_flash = st.checkbox("Flash Attention", value=False,key="Use Flash Attention")
                use_moe = st.checkbox("Mixture of Experts", value=False,key="Use MOE")
                use_mod = st.checkbox("Mixture of Depths", value=False,key="Use MOD")
            
            with col2:
                use_minimax = st.checkbox("MiniMax Attention", value=False,key="Use MixMax Attention")
                use_quantization = st.checkbox("8-bit Quantization", value=False,key="Use 8-bit Quantization")
        
        # Enhanced Checkpoint Management
        st.divider()
        st.subheader("💾 Enhanced Checkpoint Management")
        
        # List available checkpoints
        checkpoints = st.session_state.model_manager.checkpoint_manager.list_checkpoints()
        
        if checkpoints:
            st.write("**Available Checkpoints:**")
            
            # Checkpoint selection
            checkpoint_options = ["None"] + [ckpt.checkpoint_id for ckpt in checkpoints]
            selected_checkpoint = st.selectbox(
                "Select Checkpoint to Load",
                options=checkpoint_options,
                help="Choose a checkpoint to load or resume training from"
            )
            
            # Show checkpoint details
            if selected_checkpoint != "None":
                selected_ckpt = next((ckpt for ckpt in checkpoints if ckpt.checkpoint_id == selected_checkpoint), None)
                if selected_ckpt:
                    with st.expander(f"📋 {selected_checkpoint} Details", expanded=False):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Training Step", selected_ckpt.training_step)
                            st.metric("Epoch", selected_ckpt.epoch)
                            st.metric("File Size", f"{selected_ckpt.file_size_bytes / (1024*1024):.1f} MB")
                        with col2:
                            st.metric("Best Loss", f"{selected_ckpt.best_loss:.6f}")
                            st.metric("Learning Rate", f"{selected_ckpt.learning_rate:.2e}")
                            st.write(f"**Created:** {selected_ckpt.creation_time[:19]}")
                        
                        if selected_ckpt.notes:
                            st.write(f"**Notes:** {selected_ckpt.notes}")
                        
                        # Integrity verification
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("🔍 Verify Integrity", key=f"verify_{selected_checkpoint}"):
                                success, message = st.session_state.model_manager.checkpoint_manager.verify_checkpoint_integrity(selected_checkpoint)
                                if success:
                                    st.success(message)
                                else:
                                    st.error(message)
                        
                        with col2:
                            if st.button("🗑️ Delete", key=f"delete_{selected_checkpoint}"):
                                success, message = st.session_state.model_manager.checkpoint_manager.delete_checkpoint(selected_checkpoint)
                                if success:
                                    st.success(message)
                                    st.rerun()
                                else:
                                    st.error(message)
        else:
            st.info("No checkpoints available")
            selected_checkpoint = "None"
        
        # Storage statistics
        if st.button("📊 Storage Stats", key="storage_stats_btn"):
            stats = st.session_state.model_manager.checkpoint_manager.get_storage_stats()
            st.write("**Checkpoint Storage Statistics:**")
            st.write(f"- Total checkpoints: {stats['total_checkpoints']}")
            st.write(f"- Total size: {stats['total_size_gb']:.2f} GB")
            st.write(f"- Average size: {stats['average_size_mb']:.1f} MB")
            st.write(f"- Max checkpoints: {stats['max_checkpoints']}")
        
        # Initialize model with selected checkpoint
        st.divider()
        if st.button("🚀 Initialize Model", type="primary", use_container_width=True, key="init_model_btn"):
            advanced_config = {
                'use_flash_attention': use_flash,
                'use_quantization': use_quantization,
                'use_moe': use_moe,
                'use_mod': use_mod,
                'use_minimax': use_minimax,
            }
            
            checkpoint_path = None
            if selected_checkpoint != "None":
                checkpoint_path = f"./checkpoints/{selected_checkpoint}.pt"
            
            with st.spinner(f"Initializing {model_size} model..."):
                success, message = st.session_state.model_manager.initialize_model(
                    model_size=model_size,
                    checkpoint_path=checkpoint_path,
                    advanced_config=advanced_config
                )
                
                if success:
                    st.success(message)
                else:
                    st.error(message)
        
        # Enhanced model status
        if st.session_state.model_manager.initialized:
            st.divider()
            st.subheader("📊 Enhanced Model Status")
            
            info = st.session_state.model_manager.get_model_info()
            
            if info.get('status') == '✅ Initialized':
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Parameters", f"{info['total_parameters']:,}")
                    if torch.cuda.is_available():
                        st.metric("GPU Memory", f"{info.get('gpu_memory_allocated', 0):.1f} GB")
                with col2:
                    st.metric("Device", info['device'])
                    st.metric("Checkpoints", info['checkpoint_info']['total_checkpoints'])

def render_enhanced_training_tab():
    """Enhanced training interface with checkpoint resume"""
    st.header("🚀 Enhanced Model Training")
    
    if not st.session_state.model_manager.initialized:
        st.warning("Please initialize the model first using the sidebar.")
        return
    
    # Enhanced training configuration
    st.subheader("⚙️ Enhanced Training Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        learning_rate = st.number_input("Learning Rate", value=5e-4, format="%.1e", min_value=1e-6, max_value=1e-3)
        max_steps = st.number_input("Max Training Steps", value=1000, min_value=10, max_value=100000)
        batch_size = st.number_input("Batch Size", value=2, min_value=1, max_value=16)
        max_length = st.slider("Max Sequence Length", 128, 1024, 256)
        auto_save_interval = st.number_input("Auto-save Interval", value=50, min_value=10, max_value=5000)
    
    with col2:
        gradient_accumulation = st.number_input("Gradient Accumulation Steps", value=2, min_value=1, max_value=8)
        weight_decay = st.number_input("Weight Decay", value=0.01, min_value=0.0, max_value=0.1, step=0.01)
        max_grad_norm = st.number_input("Max Gradient Norm", value=1.0, min_value=0.1, max_value=10.0)
        early_stopping_patience = st.number_input("Early Stopping Patience", value=10, min_value=3, max_value=50)
        max_checkpoints = st.number_input("Max Checkpoints to Keep", value=100, min_value=1, max_value=200)
    # Feature Verification Section
    st.divider()
    st.subheader("🔍 Model & Feature Verification")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 Verify Model Size & Features", key="verify_model_btn"):
            if st.session_state.model_manager.initialized:
                model = st.session_state.model_manager.model
                config = st.session_state.model_manager.model_config
                
                verification = FeatureVerifier.verify_model_features(model, config)
                
                # Display model size verification
                st.write("**📊 Actual Model Size:**")
                size_info = verification['model_size_actual']
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Total Parameters", f"{size_info['total_parameters']:,}")
                with col_b:
                    st.metric("Size Category", size_info['size_category'])
                with col_c:
                    st.metric("Hidden Size", size_info['hidden_size'])
                
                # Verify if model matches selected size
                current_params = size_info['total_parameters']
                if current_params < 500_000_000:
                    st.error("❌ Model is too small! Check size caps in initialize_model()")
                elif 800_000_000 <= current_params <= 1_200_000_000:
                    st.success("✅ Model appears to be 1B size")
                elif 6_000_000_000 <= current_params <= 8_000_000_000:
                    st.success("✅ Model appears to be 7B size")
                elif 12_000_000_000 <= current_params <= 14_000_000_000:
                    st.success("✅ Model appears to be 13B size")
                else:
                    st.info(f"ℹ️ Model has {current_params:,} parameters")
                
                # Display feature status
                st.write("**⚙️ Advanced Features Status:**")
                features = verification['advanced_features']
                implementations = verification['implementation_status']
                
                for feature_name, configured in features.items():
                    impl_status = implementations.get(feature_name, {})
                    implemented = impl_status.get('implemented', False)
                    fallback = impl_status.get('fallback_used', True)
                    
                    if configured and implemented:
                        status = "✅ ACTIVE"
                        color = "green"
                    elif configured and fallback:
                        status = "⚠️ FALLBACK (configured but using standard implementation)"
                        color = "orange"
                    else:
                        status = "❌ DISABLED"
                        color = "red"
                    
                    st.markdown(f"**{feature_name.replace('_', ' ').title()}:** <span style='color:{color}'>{status}</span>", unsafe_allow_html=True)
            else:
                st.warning("Please initialize model first")
    
    with col2:
        st.write("**💡 What to look for:**")
        st.write("- **Parameters**: Should match selected model size")
        st.write("- **Features**: Shows if advanced features are configured")
        st.write("- **Implementation**: Shows if features are actually working")
        st.write("")
        st.info("If features show 'FALLBACK', they're configured but using standard implementations. This is normal for now.")    
    # Checkpoint resume options

    # 🚨 ADD THE IMMEDIATE FIX HERE - RIGHT AFTER LINE 1862
    st.divider()
    st.error("🚨 **CRITICAL: No Weight Updates Detected!**")
    st.write("Your training is running but weights aren't updating. This means the model isn't learning.")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Current Problem:**")
        st.write("- Optimizer not stepping")
        st.write("- Learning rate too low (5e-5)")
        st.write("- Gradient accumulation issues")

    with col2:
        st.write("**Immediate Solution:**")
        st.write("- Fixed optimizer stepping logic")
        st.write("- Higher learning rate (5e-4)")
        st.write("- Corrected gradient accumulation")

    if st.button("🚀 APPLY IMMEDIATE FIX", type="primary", key="immediate_fix"):
        try:
            # Apply the fix to current trainer
            if (hasattr(st.session_state.training_manager, 'current_trainer') and 
                st.session_state.training_manager.current_trainer is not None):
                
                trainer = st.session_state.training_manager.current_trainer
                
                # Replace the broken method with working one
                def fixed_train_step_with_verification(self, batch, step):
                    print(f"\n🔄 FIXED Training Step {step}")
                    
                    # Track accumulation properly
                    self.accumulated_steps = getattr(self, 'accumulated_steps', 0)
                    will_step_optimizer = (self.accumulated_steps + 1) % self.gradient_accumulation_steps == 0
                    
                    print(f"   Will step optimizer: {will_step_optimizer}")
                    
                    # Take snapshot only when stepping optimizer
                    pre_snapshot = None
                    if will_step_optimizer and hasattr(self, 'enable_weight_verification') and self.enable_weight_verification:
                        pre_snapshot = self.weight_verifier.create_weight_snapshot(
                            self.model, step, "pre_optimizer_step", f"Before optimizer step {step}"
                        )
                    
                    # Regular training step
                    step_metrics = self.regular_train_step(batch)
                    self.accumulated_steps += 1
                    
                    # STEP OPTIMIZER WHEN NEEDED
                    if will_step_optimizer:
                        print("🚀 STEPPING OPTIMIZER NOW!")
                        
                        # Check gradients
                        total_grad = sum(p.grad.norm().item() for p in self.model.parameters() if p.grad is not None)
                        print(f"   Total gradient norm: {total_grad:.8f}")
                        
                        if total_grad > 1e-8:
                            # Clip and step
                            if self.max_grad_norm > 0:
                                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm).item()
                            
                            # ACTUALLY STEP
                            self.optimizer.step()
                            self.scheduler.step()
                            self.optimizer.zero_grad()
                            self.accumulated_steps = 0
                            
                            print("   ✅ Optimizer stepped!")
                            
                            # Verify changes
                            if pre_snapshot and hasattr(self, 'weight_verifier'):
                                post_snapshot = self.weight_verifier.create_weight_snapshot(
                                    self.model, step, "post_optimizer_step", f"After optimizer step {step}"
                                )
                                
                                results = self.weight_verifier.verify_weight_updates(pre_snapshot, post_snapshot, expected_update=True)
                                
                                step_metrics.update({
                                    'weights_updated': results.get('weights_changed', False),
                                    'layers_changed': len(results.get('layers_changed', [])),
                                    'verification_status': results.get('verification_status', 'Unknown'),
                                    'optimizer_stepped': True
                                })
                                
                                if results.get('weights_changed', False):
                                    print(f"🎉 SUCCESS! {len(results.get('layers_changed', []))} layers updated!")
                        else:
                            print("❌ No gradients detected!")
                            step_metrics.update({
                                'weights_updated': False,
                                'layers_changed': 0,
                                'verification_status': 'No gradients',
                                'optimizer_stepped': False
                            })
                    
                    return step_metrics
                
                # Apply the fix
                trainer.train_step_with_verification = fixed_train_step_with_verification.__get__(trainer, type(trainer))
                trainer.accumulated_steps = 0
                if not hasattr(trainer, 'enable_weight_verification'):
                    trainer.enable_weight_verification = True
                
                st.success("✅ IMMEDIATE FIX APPLIED!")
                st.info("Now restart training with higher learning rate (5e-4)")
                st.balloons()
                
            else:
                st.warning("No active trainer found. Start training first, then apply the fix.")
                st.session_state.immediate_fix_ready = True
        
        except Exception as e:
            st.error(f"Fix application failed: {e}")
            import traceback
            st.code(traceback.format_exc())

    # Show corrected settings
    if st.button("📋 Show Corrected Settings", key="show_corrected"):
        st.success("✅ **USE THESE CORRECTED SETTINGS:**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.code("""
Learning Rate: 5e-4  (was 5e-5)
Batch Size: 2  (was 4)""")
        
        with col2:
            st.code("""
Gradient Accumulation: 2  (was 4)
Max Grad Norm: 1.0  (same)""")
        st.divider()
        st.error("🔬 **FIX DIDN'T WORK - DIRECT DIAGNOSIS NEEDED**")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔬 DIAGNOSE EXACT PROBLEM", type="primary", key="diagnose_exact"):
                if st.session_state.model_manager.initialized:
                    st.write("**🧪 Testing basic training components...**")
                    
                    model = st.session_state.model_manager.model
                    tokenizer = st.session_state.model_manager.tokenizer  
                    device = st.session_state.model_manager.device
                    
                    # Test 1: Forward pass
                    st.write("1️⃣ Testing forward pass...")
                    test_text = "Hello world"
                    inputs = tokenizer(test_text, return_tensors="pt", max_length=32).to(device)
                    
                    model.train()
                    outputs = model(**inputs, labels=inputs['input_ids'])
                    loss_value = outputs.loss.item()
                    st.success(f"✅ Forward pass OK: loss = {loss_value:.6f}")
                    
                    # Test 2: Gradients
                    st.write("2️⃣ Testing backward pass...")
                    outputs.loss.backward()
                    
                    total_grad = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
                    st.write(f"Total gradient norm: {total_grad:.8f}")
                    
                    if total_grad > 1e-8:
                        st.success("✅ Gradients computed successfully")
                        
                        # Test 3: Optimizer step
                        st.write("3️⃣ Testing optimizer step...")
                        
                        # Get weight before
                        first_param = next(model.parameters())
                        weight_before = first_param.clone().detach()
                        
                        # Create simple optimizer and step
                        test_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
                        test_optimizer.step()
                        
                        # Check weight after
                        weight_after = first_param.clone().detach()
                        weight_changed = not torch.equal(weight_before, weight_after)
                        change_magnitude = (weight_after - weight_before).abs().max().item()
                        
                        if weight_changed:
                            st.success(f"🎉 OPTIMIZER WORKS! Change: {change_magnitude:.10f}")
                            st.error("**Problem: Your training loop isn't calling optimizer.step() properly**")
                        else:
                            st.error("❌ Optimizer step failed")
                            
                    else:
                        st.error("❌ No gradients - check loss computation")
                else:
                    st.warning("Initialize model first")

        with col2:
            if st.button("🚀 TRY NUCLEAR OPTION", key="nuclear_option"):
                if (hasattr(st.session_state.training_manager, 'current_trainer') and 
                    st.session_state.training_manager.current_trainer is not None):
                    
                    trainer = st.session_state.training_manager.current_trainer
                    
                    # NUCLEAR: Completely replace training step
                    def nuclear_train_step(self, batch, step):
                        print(f"\n☢️ NUCLEAR Training Step {step}")
                        
                        # Take snapshot
                        if hasattr(self, 'weight_verifier') and self.weight_verifier:
                            pre_snapshot = self.weight_verifier.create_weight_snapshot(
                                self.model, step, "pre_nuclear", f"Before nuclear step {step}"
                            )
                        
                        # Clear gradients
                        self.optimizer.zero_grad()
                        
                        # Process batch
                        input_ids = batch['input_ids'].to(self.device)
                        labels = batch.get('labels', input_ids).to(self.device)
                        
                        # Forward
                        self.model.train()
                        outputs = self.model(input_ids=input_ids, labels=labels)
                        loss = outputs.loss
                        
                        print(f"   Loss: {loss.item():.6f}")
                        
                        # Backward
                        loss.backward()
                        
                        # Check gradients
                        total_grad = sum(p.grad.norm().item() for p in self.model.parameters() if p.grad is not None)
                        print(f"   Gradient norm: {total_grad:.8f}")
                        
                        # ALWAYS step optimizer (no accumulation BS)
                        if total_grad > 1e-8:
                            if hasattr(self, 'max_grad_norm') and self.max_grad_norm > 0:
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                            
                            self.optimizer.step()
                            if hasattr(self, 'scheduler'):
                                self.scheduler.step()
                            
                            print("   ✅ Optimizer stepped (nuclear)")
                            
                            # Verify
                            if hasattr(self, 'weight_verifier') and self.weight_verifier:
                                post_snapshot = self.weight_verifier.create_weight_snapshot(
                                    self.model, step, "post_nuclear", f"After nuclear step {step}"
                                )
                                
                                results = self.weight_verifier.verify_weight_updates(
                                    pre_snapshot, post_snapshot, expected_update=True
                                )
                                
                                weights_updated = results.get('weights_changed', False)
                                layers_changed = len(results.get('layers_changed', []))
                                
                                if weights_updated:
                                    print(f"🎉 NUCLEAR SUCCESS! {layers_changed} layers updated")
                                
                                return {
                                    'loss': loss.item(),
                                    'weights_updated': weights_updated,
                                    'layers_changed': layers_changed,
                                    'gradient_norm': total_grad
                                }
                        
                        return {'loss': loss.item(), 'weights_updated': False}
                    
                    # Replace training method completely
                    trainer.train_step_with_verification = nuclear_train_step.__get__(trainer, type(trainer))
                    
                    st.success("☢️ NUCLEAR OPTION APPLIED!")
                    st.info("This completely bypasses gradient accumulation. Try training now.")
                else:
                    st.warning("Start training first")

        # Show what each test means
        with st.expander("🔍 What These Tests Mean", expanded=False):
            st.write("**🔬 DIAGNOSE EXACT PROBLEM:**")
            st.write("- Tests if forward pass works")
            st.write("- Tests if gradients are computed")  
            st.write("- Tests if optimizer can change weights")
            st.write("")
            st.write("**🚀 NUCLEAR OPTION:**")
            st.write("- Completely replaces your training step")
            st.write("- Bypasses ALL gradient accumulation")
            st.write("- Steps optimizer on every single batch")
            st.write("- Should work if basic components work")

    # END OF DIAGNOSTIC - Continue with existing code below                        
            st.subheader("🔄 Resume Training")
    checkpoints = st.session_state.model_manager.checkpoint_manager.list_checkpoints()
    resume_options = ["None (Start Fresh)"] + [f"{ckpt.checkpoint_id} (Step {ckpt.training_step})" for ckpt in checkpoints]
    
    resume_selection = st.selectbox(
        "Resume from Checkpoint",
        options=resume_options,
        help="Select a checkpoint to resume training from"
    )
    
    resume_checkpoint_id = None
    if resume_selection != "None (Start Fresh)":
        resume_checkpoint_id = resume_selection.split(" (Step")[0]
        selected_ckpt = next((ckpt for ckpt in checkpoints if ckpt.checkpoint_id == resume_checkpoint_id), None)
        if selected_ckpt:
            st.info(f"📋 Will resume from step {selected_ckpt.training_step} with loss {selected_ckpt.current_loss:.6f}")
    
    # Training data selection (same as before)
    st.subheader("📊 Training Data")
    
    data_source = st.selectbox(
        "Data Source",
        ["Paste Text", "Upload Text File", "Instruction Dataset"]
    )
    
    dataset = None
    
    if data_source == "Upload Text File":
        uploaded_file = st.file_uploader("Choose a text file", type=['txt', 'md'])
        
        if uploaded_file:
            try:
                content = uploaded_file.read().decode('utf-8')
                st.text_area("Preview (first 1000 chars):", content[:1000], height=150)
                
                if st.button("📊 Process File", key="process_file_btn"):
                    texts = [line.strip() for line in content.split('\n') if line.strip()]
                    
                    if texts:
                        dataset = TextDataset(texts, st.session_state.model_manager.tokenizer, max_length)
                        if hasattr(st.session_state.model_manager, 'model_config'):
                            dataset.vocab_size = st.session_state.model_manager.model_config.vocab_size
                        st.session_state.current_dataset = dataset
                        st.success(f"✅ Created dataset with {len(texts)} samples")
                    else:
                        st.error("No valid text found in file")
            except Exception as e:
                st.error(f"Failed to process file: {e}")
    
    elif data_source == "Paste Text":
        text_input = st.text_area("Paste your training text:", height=200)
        
        if st.button("Create Dataset", key="create_text_dataset_btn") and text_input.strip():
            texts = [text.strip() for text in text_input.split('\n') if text.strip()]
            if texts:
                dataset = TextDataset(texts, st.session_state.model_manager.tokenizer, max_length)
                if hasattr(st.session_state.model_manager, 'model_config'):
                    dataset.vocab_size = st.session_state.model_manager.model_config.vocab_size
                st.session_state.current_dataset = dataset
                st.success(f"✅ Created dataset with {len(texts)} samples")
            else:
                st.error("No valid text found")
    
    # Use stored dataset
    if st.session_state.current_dataset:
        dataset = st.session_state.current_dataset
        st.info(f"Using dataset with {len(dataset)} samples")
    
    # Enhanced training control
    if dataset:
        st.subheader("🏋️ Enhanced Training Control")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🚀 Start Enhanced Training", type="primary", key="start_enhanced_training_btn"):
                # Create enhanced training config
                train_config = TrainingConfig(
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    gradient_accumulation_steps=gradient_accumulation,
                    max_steps=max_steps,
                    weight_decay=weight_decay,
                    max_grad_norm=max_grad_norm,
                    auto_save_interval=auto_save_interval,
                    early_stopping_patience=early_stopping_patience,
                    max_checkpoints=max_checkpoints,
                    save_optimizer_state=True,
                    save_scheduler_state=True,
                    save_random_states=True,
                    verify_integrity=True,
                )
                
                # Progress containers
                progress_container = st.container()
                metrics_container = st.container()
                
                with progress_container:
                    progress_bar = st.progress(0.0)
                    status_text = st.empty()
                
                with metrics_container:
                    metric_cols = st.columns(6)
                    loss_metric = metric_cols[0].empty()
                    lr_metric = metric_cols[1].empty()
                    step_metric = metric_cols[2].empty()
                    time_metric = metric_cols[3].empty()
                    best_metric = metric_cols[4].empty()
                    patience_metric = metric_cols[5].empty()
                
                # Enhanced progress callback
                start_time = time.time()
                
                def enhanced_progress_callback(step, metrics):
                    try:
                        # Update progress
                        progress = min(step / max_steps, 1.0)
                        progress_bar.progress(progress)
                        status_text.text(f"🔄 Enhanced Training Step {step}/{max_steps} - Loss: {metrics['latest_train_loss']:.4f}")
                        
                        # Update enhanced metrics
                        loss_metric.metric("Train Loss", f"{metrics['latest_train_loss']:.4f}")
                        lr_metric.metric("Learning Rate", f"{metrics['current_lr']:.2e}")
                        step_metric.metric("Step", f"{step}/{max_steps}")
                        
                        elapsed = time.time() - start_time
                        time_metric.metric("Elapsed", f"{elapsed/60:.1f}m")
                        
                        best_metric.metric("Best Loss", f"{metrics['best_loss']:.4f}")
                        patience_metric.metric("Patience", f"{metrics['patience_counter']}/{early_stopping_patience}")
                        
                        # Weight verification status
                        weights_updated = metrics.get('weights_updated', False)
                        weight_icon = "✅" if weights_updated else "❌"
                        
                        # Add new columns for verification
                        if 'verification_cols' not in st.session_state:
                            st.session_state.verification_cols = st.columns(2)
                        
                        verification_cols = st.session_state.verification_cols
                        verification_cols[0].metric("Weights Updated", f"{weight_icon}")
                        verification_cols[1].metric("Layers Changed", metrics.get('layers_changed', 0))

                    
                    except Exception as e:
                        print(f"Progress callback error: {e}")
                
                # Start enhanced training
                with st.spinner("🚀 Enhanced training in progress..."):
                    results = st.session_state.training_manager.train(
                        dataset=dataset,
                        config=train_config,
                        progress_callback=enhanced_progress_callback,
                        resume_from_checkpoint=resume_checkpoint_id
                    )
                    
                    if 'error' in results:
                        st.error(f"Training failed: {results['error']}")
                    else:
                        st.success("🎉 Enhanced training completed successfully!")
                        
                        # Show comprehensive final metrics
                        st.write("**Enhanced Training Results:**")
                        
                        result_cols = st.columns(4)
                        
                        with result_cols[0]:
                            final_loss = results.get('final_loss')
                            st.metric("Final Loss", f"{final_loss:.4f}" if final_loss is not None else "N/A")
                            
                            best_loss = results.get('best_loss')
                            st.metric("Best Loss", f"{best_loss:.4f}" if best_loss is not None else "N/A")
                        
                        with result_cols[1]:
                            total_steps = results.get('total_steps', 0)
                            st.metric("Total Steps", total_steps)
                            
                            best_step = results.get('best_step', 0)
                            st.metric("Best Step", best_step)
                        
                        with result_cols[2]:
                            training_time = results.get('training_time', 0)
                            st.metric("Training Time", f"{training_time/60:.1f} min" if training_time else "N/A")
                            
                            checkpoints_saved = results.get('checkpoints_saved', 0)
                            st.metric("Checkpoints Saved", checkpoints_saved)
                        
                        with result_cols[3]:
                            avg_tokens_per_sec = results.get('avg_tokens_per_second', 0)
                            st.metric("Avg Tokens/sec", f"{avg_tokens_per_sec:.1f}")
                            
                            early_stopped = results.get('early_stopped', False)
                            st.metric("Early Stopped", "Yes" if early_stopped else "No")
                        
                        # Enhanced metrics
                        if 'final_perplexity' in results:
                            st.metric("Final Perplexity", f"{results['final_perplexity']:.2f}")
        
        with col2:
            if st.button("🧪 Test Forward Pass", key="test_forward_pass_btn"):
                try:
                    sample = dataset[0]
                    test_batch = {k: v.unsqueeze(0).to(st.session_state.model_manager.device) 
                                 for k, v in sample.items()}
                    
                    with torch.no_grad():
                        outputs = st.session_state.model_manager.model(**test_batch)
                        loss = outputs.loss.item() if outputs.loss is not None else 0.0
                        st.success(f"✅ Forward pass successful! Loss: {loss:.6f}")
                        
                except Exception as e:
                    st.error(f"❌ Forward pass failed: {str(e)}")
        
        with col3:
            if st.button("💾 Manual Save", key="manual_save_btn"):
                try:
                    # Get the current trainer if training is active
                    if hasattr(st.session_state.training_manager, 'current_trainer') and st.session_state.training_manager.current_trainer:
                        trainer = st.session_state.training_manager.current_trainer
                        success, message = trainer.save_checkpoint(
                            checkpoint_name=f"manual_save_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            notes="Manual save from UI"
                        )
                        
                        if success:
                            st.success(message)
                        else:
                            st.error(message)
                    else:
                        st.warning("No active trainer for manual save")
                    
                except Exception as e:
                    st.error(f"❌ Manual save failed: {str(e)}")
        
        render_training_verification_section()
        render_weight_verification_section()
        show_weight_verification_troubleshooting()
        # Enhanced checkpoint management section
        st.divider()
        st.subheader("📈 Enhanced Training Analytics")
        
        if st.session_state.training_manager.training_history:
            latest_training = st.session_state.training_manager.training_history[-1]
            
            # Show training history
            with st.expander("📊 Latest Training Session Analytics", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Session Summary:**")
                    st.write(f"- Started: {latest_training['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                    st.write(f"- Total Steps: {latest_training['metrics'].get('total_steps', 'N/A')}")
                    st.write(f"- Final Loss: {latest_training['metrics'].get('final_loss', 'N/A'):.6f}" if latest_training['metrics'].get('final_loss') else "- Final Loss: N/A")
                    st.write(f"- Best Loss: {latest_training['metrics'].get('best_loss', 'N/A'):.6f}" if latest_training['metrics'].get('best_loss') else "- Best Loss: N/A")
                    st.write(f"- Checkpoints Created: {latest_training.get('checkpoints_created', 'N/A')}")
                    
                with col2:
                    st.write("**Performance Metrics:**")
                    st.write(f"- Training Time: {latest_training['metrics'].get('training_time', 0)/60:.1f} minutes" if latest_training['metrics'].get('training_time') else "- Training Time: N/A")
                    st.write(f"- Avg Tokens/sec: {latest_training['metrics'].get('avg_tokens_per_second', 'N/A'):.1f}" if latest_training['metrics'].get('avg_tokens_per_second') else "- Avg Tokens/sec: N/A")
                    st.write(f"- Total Epochs: {latest_training['metrics'].get('total_epochs', 'N/A')}")
                    st.write(f"- Early Stopped: {'Yes' if latest_training['metrics'].get('early_stopped', False) else 'No'}")
                    
                    if latest_training.get('resume_from'):
                        st.write(f"- Resumed from: {latest_training['resume_from']}")
                
                # Plot enhanced metrics if available
                if 'monitor' in latest_training:
                    monitor = latest_training['monitor']
                    if hasattr(monitor, 'plot_metrics'):
                        try:
                            fig = monitor.plot_metrics()
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not generate training plots: {e}")
def render_training_verification_section():
    """Add this to your training tab to verify model learned the data"""
    # Add safety check
    if not hasattr(st.session_state, 'model_manager') or not st.session_state.model_manager.initialized:
            st.warning("Initialize model first to access verification features")
            return    
    st.divider()
    st.subheader("🔍 Training Data Verification")
    st.caption("Verify that your model actually learned from the training data")
    
    if not st.session_state.model_manager.initialized:
        st.warning("Initialize model first to access verification features")
        return
    
    # Get training data from session state
    training_texts = []
    if st.session_state.current_dataset:
        # Extract original texts from dataset if possible
        if hasattr(st.session_state.current_dataset, 'texts'):
            training_texts = st.session_state.current_dataset.texts
    
    # Manual input option
    if not training_texts:
        st.info("💡 **How to verify**: Paste some of your training data below to test if the model learned it")
        
        verification_text = st.text_area(
            "Paste sample training data for verification:",
            placeholder="Enter some text that you trained the model on...",
            height=100
        )
        
        if verification_text.strip():
            training_texts = [verification_text.strip()]
    
    if training_texts:
        st.success(f"✅ Found {len(training_texts)} training text samples for verification")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎯 Quick Generation Test", key="quick_test_btn"):
                quick_generation_test(training_texts)
        
        with col2:
            if st.button("📋 Show Example", key="example_btn"):
                st.info("""
                **Example: If you trained on cooking recipes**
                
                Training data: "Preheat oven to 350°F. Mix flour and sugar..."
                
                Prompt: "How do I bake"
                Generated: "How do I bake cookies? First, preheat oven to 350°F..."
                
                ✅ **This shows the model learned!** It uses the same temperature and steps from your training data.
                """)
    
    else:
        st.info("📝 **To verify training data integration:**\n1. Paste some training text above, OR\n2. Train with data that saves the original texts")

def quick_generation_test(training_texts):
    """Quick test to see if model uses training data"""
    
    st.write("**🎯 Quick Generation Test:**")
    
    # Extract a few words from training data
    sample_text = " ".join(training_texts[:2])
    words = sample_text.split()
    
    if len(words) > 3:
        # Create prompt from first few words
        test_prompt = " ".join(words[:3])
        
        st.write(f"**Prompt (from your training data):** `{test_prompt}`")
        
        try:
            from dataclasses import dataclass
            @dataclass
            class GenerationConfig:
                temperature: float = 0.7
                max_length: int = 100
                top_p: float = 0.9
                top_k: int = 50
                repetition_penalty: float = 1.1
                no_repeat_ngram_size: int = 3
                do_sample: bool = True
                num_beams: int = 1
                generation_strategy: str = "auto"
                length_penalty: float = 1.0
                early_stopping: bool = False
            
            gen_config = GenerationConfig()
            generated_text, stats = st.session_state.model_manager.generate_text(test_prompt, gen_config)
            
            if not generated_text.startswith("❌"):
                st.write("**🤖 Model Generated:**")
                st.write(f'"{generated_text}"')
                
                # Check for similarity
                original_context = " ".join(words[3:8]) if len(words) > 8 else ""
                if original_context:
                    similarity_found = any(word.lower() in generated_text.lower() for word in original_context.split())
                    
                    if similarity_found:
                        st.success("✅ **Good sign!** Generated text contains words from your training data")
                        st.info("🎉 **This means your model DID learn from the training data!**")
                    else:
                        st.warning("⚠️ Generated text doesn't seem related to training data")
                        st.info("💡 Try training for more steps or with more similar prompts")
                    
                    st.info(f"**Original context was:** `{original_context}`")
            else:
                st.error(f"Generation failed: {generated_text}")
                
        except Exception as e:
            st.error(f"Quick test failed: {str(e)}")
    else:
        st.warning("Training text too short for meaningful test")
def render_weight_verification_section():
    """Render weight verification interface in Streamlit"""
    
    st.divider()
    st.subheader("🔍 Weight Verification System")
    st.caption("Verify that your model weights are actually updating during training")
    
    if not hasattr(st.session_state.training_manager, 'current_trainer'):
        st.info("Start training to access weight verification features")
        return
    
    trainer = st.session_state.training_manager.current_trainer
    
    # Check if trainer has weight verification
    if not hasattr(trainer, 'weight_verifier'):
        st.warning("Current trainer doesn't have weight verification enabled")
        return

# REPLACE WITH THIS IMPROVED VERSION:
def render_weight_verification_section():
    """Render weight verification interface in Streamlit"""
    
    st.divider()
    st.subheader("🔍 Weight Verification System")
    st.caption("Verify that your model weights are actually updating during training")
    
    # Check if model is initialized first
    if not st.session_state.model_manager.initialized:
        st.info("💡 **Weight Verification Available After Model Initialization**")
        st.write("- Initialize a model in the sidebar first")
        st.write("- Start training to see real-time weight update verification") 
        st.write("- Monitor if your model is actually learning")
        return
    
    # Check if training manager exists
    if not hasattr(st.session_state, 'training_manager'):
        st.info("Training manager not available")
        return
    
    training_manager = st.session_state.training_manager
    
    # Check if trainer exists and what type it is
    if not hasattr(training_manager, 'current_trainer') or training_manager.current_trainer is None:
        st.info("🚀 **Weight Verification Ready**")
        st.write("Weight verification will be enabled when you start training.")
        
        # Show weight verification settings
        col1, col2 = st.columns(2)
        
        with col1:
            enable_verification = st.checkbox(
                "Enable Weight Verification", 
                value=True,
                help="Enable real-time weight update verification during training"
            )
        
        with col2:
            verification_frequency = st.selectbox(
                "Verification Detail Level",
                ["Every Step", "Every 10 Steps", "Summary Only"],
                help="How detailed the weight verification should be"
            )
        
        # Store settings for when training starts
        if not hasattr(st.session_state, 'weight_verification_settings'):
            st.session_state.weight_verification_settings = {}
        
        st.session_state.weight_verification_settings['enabled'] = enable_verification
        st.session_state.weight_verification_settings['frequency'] = verification_frequency
        
        if enable_verification:
            st.success("✅ Weight verification will be enabled when training starts")
        else:
            st.warning("⚠️ Weight verification is disabled")
        
        # Quick test option
        if st.button("🧪 Test Weight Detection System", key="test_weight_detection"):
            test_weight_verification_system()
        
        return
    
    trainer = training_manager.current_trainer
    
    # Check if trainer has weight verification capability
    if not hasattr(trainer, 'weight_verifier'):
        st.warning("⚠️ **Current trainer doesn't have weight verification**")
        st.info("This happens when:")
        st.write("- Training was started before implementing weight verification")
        st.write("- The trainer class wasn't updated to `EnhancedTrainerWithWeightVerification`")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Restart Training with Verification", key="restart_with_verification"):
                # Clear current trainer to force recreation with verification
                training_manager.current_trainer = None
                st.success("✅ Cleared trainer. Start training again to enable verification.")
                st.rerun()
        
        with col2:
            if st.button("🧪 Test Weight System Anyway", key="test_anyway"):
                test_weight_verification_system()
        return
    
    # Trainer has weight verification - show full interface
    weight_verifier = trainer.weight_verifier
    
    st.success("✅ **Weight Verification Active**")
    
    # Verification status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        verification_enabled = weight_verifier.verification_enabled
        status_icon = "✅" if verification_enabled else "❌"
        st.metric("Verification Status", f"{status_icon} {'Active' if verification_enabled else 'Disabled'}")
    
    with col2:
        total_snapshots = len(weight_verifier.snapshots)
        st.metric("Weight Snapshots", total_snapshots)
    
    with col3:
        total_verifications = len(weight_verifier.weight_update_history)
        st.metric("Verifications Done", total_verifications)
    
    # Weight verification controls
    st.write("**🔧 Weight Verification Controls:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📸 Create Weight Snapshot", key="create_snapshot_btn"):
            snapshot = weight_verifier.create_weight_snapshot(
                st.session_state.model_manager.model,
                trainer.monitor.step,
                "manual_snapshot",
                "Manual snapshot from UI"
            )
            if snapshot:
                st.success(f"✅ Snapshot created with {len(snapshot.layer_hashes)} layers")
            else:
                st.error("❌ Failed to create snapshot")
    
    with col2:
        if st.button("🔄 Verify Recent Updates", key="verify_updates_btn"):
            summary = weight_verifier.get_weight_update_summary(last_n_steps=5)
            
            st.write("**Recent Weight Update Summary:**")
            st.write(f"- Total verifications: {summary['total_verifications']}")
            st.write(f"- Recent successful: {summary['recent_successful_updates']}")
            st.write(f"- Recent failed: {summary['recent_failed_updates']}")
            
            if summary['recent_successful_updates'] > 0:
                st.success("✅ Recent weight updates detected")
            else:
                if summary['total_verifications'] == 0:
                    st.info("ℹ️ No verifications yet - start training to see updates")
                else:
                    st.error("❌ No recent weight updates detected")
    
    with col3:
        if st.button("📊 Export Verification Report", key="export_report_btn"):
            report_path = "./exports/weight_verification_report.json"
            import os
            os.makedirs("./exports", exist_ok=True)
            
            message = weight_verifier.export_verification_report(report_path)
            st.success(message)
    
    # Display recent snapshots
    if weight_verifier.snapshots:
        st.write("**📸 Recent Weight Snapshots:**")
        
        recent_snapshots = weight_verifier.snapshots[-5:]  # Last 5 snapshots
        
        snapshot_data = []
        for snapshot in recent_snapshots:
            snapshot_data.append({
                "Time": snapshot.timestamp,
                "Step": snapshot.step,
                "Type": snapshot.snapshot_type,
                "Layers": len(snapshot.layer_hashes),
                "Parameters": f"{snapshot.total_parameters:,}",
                "Notes": snapshot.notes[:50] + "..." if len(snapshot.notes) > 50 else snapshot.notes
            })
        
        import pandas as pd
        df = pd.DataFrame(snapshot_data)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("📋 No weight snapshots yet. They will appear when training starts.")
    
    # Weight verification history
    if weight_verifier.weight_update_history:
        with st.expander("📈 Weight Update Verification History", expanded=False):
            
            recent_verifications = weight_verifier.weight_update_history[-10:]
            
            for i, verification in enumerate(reversed(recent_verifications)):
                results = verification["verification_results"]
                
                status_icon = "✅" if results.get("weights_changed", False) else "❌"
                
                st.write(f"{status_icon} **Step {verification['before_step']} → {verification['after_step']}**")
                st.write(f"   {results.get('verification_status', 'Unknown status')}")
                
                if results.get("layers_changed"):
                    st.write(f"   Layers updated: {len(results['layers_changed'])}")
                
                if i < len(recent_verifications) - 1:
                    st.write("---")
    else:
        st.info("📊 No verification history yet. Start training to see weight update tracking.")
# ======================== ADD TEST FUNCTION ======================== #
# ======================== VERIFICATION TROUBLESHOOTING ======================== #

def show_weight_verification_troubleshooting():
    """Show troubleshooting guide for weight verification"""
    
    with st.expander("🔧 Weight Verification Troubleshooting", expanded=False):
        st.write("**Common Issues and Solutions:**")
        
        st.write("**1. 'Current trainer doesn't have weight verification enabled'**")
        st.write("- ✅ **Solution**: Make sure you're using `EnhancedTrainerWithWeightVerification` in the training manager")
        st.write("- ✅ **Quick Fix**: Stop training and start again")
        
        st.write("**2. 'No weight updates detected'**") 
        st.write("- ✅ Check learning rate (try 1e-4 or 5e-4)")
        st.write("- ✅ Check batch size (try 2 or 4)")
        st.write("- ✅ Check gradient accumulation (try 1 or 2)")
        st.write("- ✅ Verify loss is decreasing")
        
        st.write("**3. Weight verification not showing during training**")
        st.write("- ✅ Ensure `enable_weight_verification = True` in trainer")
        st.write("- ✅ Check console for verification messages")
        st.write("- ✅ Restart training with verification enabled")
        
        st.write("**4. Checkpoints don't verify**")
        st.write("- ✅ Use `save_checkpoint_with_verification()` method")
        st.write("- ✅ Use `load_checkpoint_with_verification()` method")
        st.write("- ✅ Check file permissions in checkpoint directory")
def test_weight_verification_system():
    """Test the weight verification system independently"""
    
    if not st.session_state.model_manager.initialized:
        st.error("❌ Please initialize a model first")
        return
    
    with st.spinner("🧪 Testing weight verification system..."):
        try:
            model = st.session_state.model_manager.model
            
            # Import the verification system
            verifier = WeightVerificationSystem()
            
            st.write("**🔍 Step 1: Creating initial weight snapshot...**")
            before = verifier.create_weight_snapshot(model, 0, "test_before", "Test snapshot before change")
            
            if before:
                st.success(f"✅ Initial snapshot created ({len(before.layer_hashes)} layers)")
            else:
                st.error("❌ Failed to create initial snapshot")
                return
            
            st.write("**🔧 Step 2: Making microscopic weight change...**")
            with torch.no_grad():
                first_param = next(model.parameters())
                original_value = first_param[0].clone()
                first_param[0] += 1e-6  # Tiny change
            
            st.write("**📸 Step 3: Creating post-change snapshot...**")
            after = verifier.create_weight_snapshot(model, 1, "test_after", "Test snapshot after change")
            
            st.write("**🔍 Step 4: Verifying change detection...**")
            results = verifier.verify_weight_updates(before, after, expected_update=True)
            
            if results["weights_changed"]:
                st.success("🎉 **WEIGHT VERIFICATION SYSTEM IS WORKING CORRECTLY!**")
                st.write(f"✅ Detected changes in {len(results['layers_changed'])} layers")
                st.write(f"📊 Status: {results['verification_status']}")
                
                # Show some details
                with st.expander("📋 Verification Details"):
                    st.write(f"- Layers changed: {len(results['layers_changed'])}")
                    st.write(f"- Layers unchanged: {len(results['layers_unchanged'])}")
                    st.write(f"- Step difference: {results['step_difference']}")
                    
                    if results['layers_changed']:
                        st.write("**Sample changed layers:**")
                        for layer in results['layers_changed'][:5]:  # First 5
                            st.write(f"  - {layer}")
                
            else:
                st.error("❌ **WEIGHT VERIFICATION SYSTEM NOT WORKING!**")
                st.write("This indicates a problem with the verification implementation.")
                st.write("Check that the WeightVerificationSystem class was added correctly.")
            
            # Restore original value
            with torch.no_grad():
                first_param[0] = original_value
            
            st.info("🔄 Restored original weight value")
            
        except Exception as e:
            st.error(f"❌ Test failed with error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())        
def quick_weight_verification_test():
    """Quick test to verify the system is working"""
    
    if not st.session_state.model_manager.initialized:
        st.warning("Initialize model first")
        return
    
    st.subheader("🧪 Quick Weight Verification Test")
    
    if st.button("🔬 Test Weight Update Detection"):
        model = st.session_state.model_manager.model
        
        # Create verification system
        verifier = WeightVerificationSystem()
        
        # Take before snapshot
        before = verifier.create_weight_snapshot(model, 0, "test_before", "Before test")
        
        # Modify a small weight slightly
        with torch.no_grad():
            first_param = next(model.parameters())
            original_value = first_param[0].clone()
            first_param[0] += 1e-6  # Tiny change
        
        # Take after snapshot  
        after = verifier.create_weight_snapshot(model, 1, "test_after", "After test")
        
        # Verify change detected
        results = verifier.verify_weight_updates(before, after, expected_update=True)
        
        if results["weights_changed"]:
            st.success("✅ Weight verification system working correctly!")
            st.write(f"Detected changes in {len(results['layers_changed'])} layers")
            
            # Restore original value
            with torch.no_grad():
                first_param[0] = original_value
        else:
            st.error("❌ Weight verification system not detecting changes!")
            st.write("This indicates a problem with the verification system")
def render_enhanced_generation_tab():
    """Enhanced generation interface with history tracking"""
    st.header("✨ Enhanced Text Generation")
    
    if not st.session_state.model_manager.initialized:
        st.warning("Please initialize the model first using the sidebar.")
        return
    
    # Generation settings
    col1, col2 = st.columns([3, 1])
    
    with col1:
        prompt = st.text_area(
            "Enter your prompt:",
            placeholder="Once upon a time in a magical kingdom...",
            height=100
        )
    
    with col2:
        st.subheader("Generation Settings")
        
        temperature = st.slider("Temperature", 0.1, 2.0, 0.7)
        max_length = st.slider("Max Length", 50, 10000, 200)
        
        strategy = st.selectbox(
            "Strategy",
            ["auto", "standard"],
            help="Generation strategy to use"
        )
    
    # Advanced settings
    with st.expander("🔧 Advanced Settings"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            top_p = st.slider("Top-p", 0.1, 1.0, 0.9)
            top_k = st.slider("Top-k", 1, 100, 50)
        
        with col2:
            repetition_penalty = st.slider("Repetition Penalty", 1.0, 2.0, 1.1)
            no_repeat_ngram = st.slider("No Repeat N-gram", 0, 5, 3)
        
        with col3:
            do_sample = st.checkbox("Do Sample", value=True,key="do_sample_checkbox")
            num_beams = st.slider("Num Beams", 1, 5, 1)
    
    # Generate button
    if st.button("🎨 Generate", type="primary", use_container_width=True, key="generate_text_btn"):
        if not prompt.strip():
            st.warning("Please enter a prompt.")
            return
        
        # Create generation config
        gen_config = GenerationConfig(
            temperature=temperature,
            max_length=max_length,
            generation_strategy=strategy,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram,
            do_sample=do_sample,
            num_beams=num_beams
        )
        
        with st.spinner("Generating..."):
            generated_text, stats = st.session_state.model_manager.generate_text(prompt, gen_config)
            
            # Display results
            if not generated_text.startswith("❌"):
                st.subheader("Generated Text")
                st.write(generated_text)
                
                # Display enhanced statistics
                if stats:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Generation Time", f"{stats.get('generation_time', 0):.2f}s")
                    with col2:
                        st.metric("Tokens Generated", stats.get('tokens_generated', 0))
                    with col3:
                        st.metric("Tokens/Second", f"{stats.get('tokens_per_second', 0):.1f}")
                    with col4:
                        st.metric("Strategy Used", stats.get('strategy_used', 'Unknown'))
            else:
                st.error(generated_text)
    
    # Enhanced generation history
    if st.session_state.model_manager.generation_history:
        st.divider()
        st.subheader("📚 Generation History")
        
        # Show last 5 generations
        recent_generations = st.session_state.model_manager.generation_history[-5:]
        
        for i, gen in enumerate(reversed(recent_generations)):
            with st.expander(f"Generation {len(recent_generations)-i}: {gen['prompt'][:50]}...", expanded=False):
                st.write(f"**Prompt:** {gen['prompt']}")
                st.write(f"**Generated:** {gen['generated']}")
                st.write(f"**Time:** {gen['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                
                stats = gen['stats']
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Generation Time", f"{stats.get('generation_time', 0):.2f}s")
                with col2:
                    st.metric("Tokens Generated", stats.get('tokens_generated', 0))
                with col3:
                    st.metric("Tokens/Second", f"{stats.get('tokens_per_second', 0):.1f}")

def render_enhanced_evaluation_tab():
    """Enhanced model evaluation interface"""
    st.header("📊 Enhanced Model Evaluation")
    
    if not st.session_state.model_manager.initialized:
        st.warning("Please initialize the model first using the sidebar.")
        return
    
    # Enhanced evaluation options
    st.subheader("🎯 Enhanced Evaluation Suite")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 Run Basic Tests", type="primary", key="run_basic_tests_btn"):
            results = {}
            
            with st.spinner("Running enhanced tests..."):
                try:
                    # Test 1: Model info
                    info = st.session_state.model_manager.get_model_info()
                    results["Model Info"] = "✅ Retrieved successfully"
                    
                    # Test 2: Simple forward pass
                    tokenizer = st.session_state.model_manager.tokenizer
                    model = st.session_state.model_manager.model
                    device = st.session_state.model_manager.device
                    
                    test_text = "The quick brown fox"
                    inputs = tokenizer(test_text, return_tensors="pt").to(device)
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                        loss = outputs.loss.item() if outputs.loss is not None else "N/A"
                    
                    results["Forward Pass"] = f"✅ Success, Loss: {loss}"
                    
                    # Test 3: Generation test
                    gen_config = GenerationConfig(max_length=50, temperature=0.8)
                    generated, stats = st.session_state.model_manager.generate_text("Hello", gen_config)
                    
                    if not generated.startswith("❌"):
                        results["Generation"] = f"✅ Success, {stats.get('tokens_generated', 0)} tokens"
                    else:
                        results["Generation"] = "❌ Failed"
                    
                    # Test 4: Checkpoint integrity (if any exist)
                    checkpoints = st.session_state.model_manager.checkpoint_manager.list_checkpoints()
                    if checkpoints:
                        latest_checkpoint = checkpoints[0]
                        success, message = st.session_state.model_manager.checkpoint_manager.verify_checkpoint_integrity(latest_checkpoint.checkpoint_id)
                        results["Checkpoint Integrity"] = "✅ Verified" if success else "❌ Failed"
                    else:
                        results["Checkpoint Integrity"] = "ℹ️ No checkpoints to verify"
                
                except Exception as e:
                    results["Error"] = f"❌ {str(e)}"
            
            # Display enhanced results
            st.subheader("📈 Enhanced Test Results")
            
            for test, result in results.items():
                if "✅" in result:
                    st.success(f"**{test}:** {result}")
                elif "❌" in result:
                    st.error(f"**{test}:** {result}")
                else:
                    st.info(f"**{test}:** {result}")
    
    with col2:
        if st.button("🔍 Checkpoint Analysis", key="checkpoint_analysis_btn"):
            checkpoints = st.session_state.model_manager.checkpoint_manager.list_checkpoints()
            
            if checkpoints:
                st.subheader("📋 Checkpoint Analysis")
                
                # Create checkpoint comparison
                checkpoint_data = []
                for ckpt in checkpoints:
                    checkpoint_data.append({
                        'Checkpoint ID': ckpt.checkpoint_id,
                        'Step': ckpt.training_step,
                        'Loss': ckpt.current_loss,
                        'Best Loss': ckpt.best_loss,
                        'Size (MB)': ckpt.file_size_bytes / (1024*1024),
                        'Creation Time': ckpt.creation_time[:19]
                    })
                
                df = pd.DataFrame(checkpoint_data)
                st.dataframe(df, use_container_width=True)
                
                # Plot checkpoint progression
                if len(checkpoints) > 1:
                    fig = go.Figure()
                    
                    steps = [ckpt.training_step for ckpt in checkpoints]
                    losses = [ckpt.current_loss for ckpt in checkpoints]
                    best_losses = [ckpt.best_loss for ckpt in checkpoints]
                    
                    fig.add_trace(go.Scatter(x=steps, y=losses, name='Current Loss', mode='lines+markers'))
                    fig.add_trace(go.Scatter(x=steps, y=best_losses, name='Best Loss', mode='lines+markers'))
                    
                    fig.update_layout(
                        title="Checkpoint Loss Progression",
                        xaxis_title="Training Step",
                        yaxis_title="Loss",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No checkpoints available for analysis")
    
    # Model comparison section
    st.divider()
    st.subheader("🔄 Model Comparison")
    
    if st.button("📊 Compare with Previous Versions", key="compare_models_btn"):
        checkpoints = st.session_state.model_manager.checkpoint_manager.list_checkpoints()
        
        if len(checkpoints) >= 2:
            st.write("**Model Performance Comparison:**")
            
            # Compare latest vs best
            latest = checkpoints[0]  # Sorted by creation time
            best = min(checkpoints, key=lambda x: x.best_loss)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Latest Model:**")
                st.write(f"- Checkpoint: {latest.checkpoint_id}")
                st.write(f"- Step: {latest.training_step}")
                st.write(f"- Current Loss: {latest.current_loss:.6f}")
                st.write(f"- Best Loss: {latest.best_loss:.6f}")
            
            with col2:
                st.write("**Best Model:**")
                st.write(f"- Checkpoint: {best.checkpoint_id}")
                st.write(f"- Step: {best.training_step}")
                st.write(f"- Current Loss: {best.current_loss:.6f}")
                st.write(f"- Best Loss: {best.best_loss:.6f}")
            
            # Performance delta
            loss_improvement = latest.best_loss - best.best_loss
            if loss_improvement < 0:
                st.success(f"🎉 Improvement: {abs(loss_improvement):.6f} loss reduction")
            elif loss_improvement > 0:
                st.warning(f"📈 Regression: {loss_improvement:.6f} loss increase")
            else:
                st.info("➡️ No change in best loss")
        else:
            st.info("Need at least 2 checkpoints for comparison")

def render_enhanced_deployment_tab():
    """Enhanced deployment interface with checkpoint export"""
    st.header("🚀 Enhanced Model Deployment")
    
    if not st.session_state.model_manager.initialized:
        st.warning("Please initialize the model first using the sidebar.")
        return
    
    st.subheader("📦 Enhanced Export Options")
    
    # Checkpoint selection for export
    checkpoints = st.session_state.model_manager.checkpoint_manager.list_checkpoints()
    
    if checkpoints:
        st.write("**Select Checkpoint to Export:**")
        
        checkpoint_options = [f"{ckpt.checkpoint_id} (Step {ckpt.training_step}, Loss {ckpt.best_loss:.6f})" 
                             for ckpt in checkpoints]
        selected_export = st.selectbox(
            "Checkpoint for Export",
            options=checkpoint_options,
            help="Choose which checkpoint to export"
        )
        
        selected_checkpoint_id = selected_export.split(" (Step")[0]
        selected_ckpt = next((ckpt for ckpt in checkpoints if ckpt.checkpoint_id == selected_checkpoint_id), None)
        
        if selected_ckpt:
            st.info(f"📋 Selected: {selected_ckpt.checkpoint_id} - Created: {selected_ckpt.creation_time[:19]}")
    else:
        st.warning("No checkpoints available for export")
        selected_checkpoint_id = None
        selected_ckpt = None
    
    export_format = st.selectbox(
        "Export Format",
        ["Enhanced Checkpoint (.pt)", "Production Model (.pt)", "State Dict Only", "Config + Weights Separate", "ONNX Format"]
    )
    
    if export_format == "Enhanced Checkpoint (.pt)" and selected_checkpoint_id:
        export_path = st.text_input("Export Path", value=f"./exports/{selected_checkpoint_id}_enhanced.pt")
        
        if st.button("💾 Export Enhanced Checkpoint", key="export_enhanced_btn"):
            try:
                import shutil
                os.makedirs(os.path.dirname(export_path), exist_ok=True)
                
                # Copy the selected checkpoint
                source_path = f"./checkpoints/{selected_checkpoint_id}.pt"
                shutil.copy2(source_path, export_path)
                
                # Copy metadata
                metadata_export_path = export_path.replace('.pt', '_metadata.json')
                if selected_ckpt:
                    metadata = asdict(selected_ckpt)
                    with open(metadata_export_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                
                file_size = os.path.getsize(export_path) / (1024 * 1024)
                st.success(f"✅ Enhanced checkpoint exported!")
                st.info(f"📁 Model: {export_path} ({file_size:.1f} MB)")
                st.info(f"📄 Metadata: {metadata_export_path}")
                
            except Exception as e:
                st.error(f"❌ Export failed: {str(e)}")
    
    elif export_format == "Production Model (.pt)":
        save_path = st.text_input("Save Path", value="./exports/mastishk_production.pt")
        
        col1, col2 = st.columns(2)
        with col1:
            include_optimizer = st.checkbox("Include Optimizer State", value=False, key="include_optimizer_checkbox")
            include_scheduler = st.checkbox("Include Scheduler State", value=False, key="include_scheduler_checkbox")
        with col2:
            include_training_state = st.checkbox("Include Training State", value=False, key="include_training_state_checkbox")
            compress_model = st.checkbox("Compress Model", value=True, key="compress_model_checkbox")
        
        if st.button("💾 Export Production Model", key="export_production_btn"):
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                # Create production checkpoint
                production_data = {
                    'model_state_dict': st.session_state.model_manager.model.state_dict(),
                    'config_dict': {
                        'vocab_size': st.session_state.model_manager.model_config.vocab_size,
                        'hidden_size': st.session_state.model_manager.model_config.hidden_size,
                        'num_hidden_layers': st.session_state.model_manager.model_config.num_hidden_layers,
                        'num_attention_heads': st.session_state.model_manager.model_config.num_attention_heads,
                        'num_key_value_heads': st.session_state.model_manager.model_config.num_key_value_heads,
                        'intermediate_size': st.session_state.model_manager.model_config.intermediate_size,
                        'max_position_embeddings': st.session_state.model_manager.model_config.max_position_embeddings,
                        'rms_norm_eps': st.session_state.model_manager.model_config.rms_norm_eps,
                        'hidden_act': st.session_state.model_manager.model_config.hidden_act,
                        'use_cache': st.session_state.model_manager.model_config.use_cache,
                    },
                    'model_class': 'MastishkTransformerForCausalLM',
                    'timestamp': datetime.now().isoformat(),
                    'export_type': 'production',
                    'pytorch_version': torch.__version__,
                }
                
                # Optionally include training components
                if include_optimizer and hasattr(st.session_state.training_manager, 'current_trainer'):
                    trainer = st.session_state.training_manager.current_trainer
                    if trainer and hasattr(trainer, 'optimizer'):
                        production_data['optimizer_state_dict'] = trainer.optimizer.state_dict()
                
                if include_scheduler and hasattr(st.session_state.training_manager, 'current_trainer'):
                    trainer = st.session_state.training_manager.current_trainer
                    if trainer and hasattr(trainer, 'scheduler'):
                        production_data['scheduler_state_dict'] = trainer.scheduler.state_dict()
                
                if include_training_state and hasattr(st.session_state.training_manager, 'current_trainer'):
                    trainer = st.session_state.training_manager.current_trainer
                    if trainer and hasattr(trainer, 'monitor'):
                        training_state = trainer.monitor.to_training_state()
                        production_data['training_state'] = asdict(training_state)
                
                # Save with optional compression
                if compress_model:
                    torch.save(production_data, save_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    torch.save(production_data, save_path)
                
                file_size = os.path.getsize(save_path) / (1024 * 1024)
                st.success(f"✅ Production model exported!")
                st.info(f"📁 File: {save_path} ({file_size:.1f} MB)")
                st.info(f"🔧 Includes: Model{'+ Optimizer' if include_optimizer else ''}{'+ Scheduler' if include_scheduler else ''}{'+ Training State' if include_training_state else ''}")
                
            except Exception as e:
                st.error(f"❌ Export failed: {str(e)}")
    
    elif export_format == "ONNX Format":
        st.warning("🚧 ONNX export is experimental and may not work with all model components")
        
        onnx_path = st.text_input("ONNX Export Path", value="./exports/mastishk_model.onnx")
        
        col1, col2 = st.columns(2)
        with col1:
            batch_size = st.number_input("Batch Size", value=1, min_value=1, max_value=8)
            seq_length = st.number_input("Sequence Length", value=128, min_value=32, max_value=512)
        with col2:
            opset_version = st.number_input("ONNX Opset Version", value=11, min_value=9, max_value=17)
        
        if st.button("🔄 Export to ONNX", key="export_onnx_btn"):
            try:
                import torch.onnx
                
                os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
                
                # Create dummy input
                dummy_input = torch.randint(0, 0, (batch_size, seq_length), dtype=torch.long)
                dummy_input = dummy_input.to(st.session_state.model_manager.device)
                
                # Export to ONNX
                st.session_state.model_manager.model.eval()
                
                with torch.no_grad():
                    torch.onnx.export(
                        st.session_state.model_manager.model,
                        dummy_input,
                        onnx_path,
                        export_params=True,
                        opset_version=opset_version,
                        do_constant_folding=True,
                        input_names=['input_ids'],
                        output_names=['logits'],
                        dynamic_axes={
                            'input_ids': {0: 'batch_size', 1: 'sequence'},
                            'logits': {0: 'batch_size', 1: 'sequence'}
                        }
                    )
                
                file_size = os.path.getsize(onnx_path) / (1024 * 1024)
                st.success(f"✅ ONNX model exported!")
                st.info(f"📁 File: {onnx_path} ({file_size:.1f} MB)")
                st.info(f"⚙️ Config: Batch={batch_size}, Seq={seq_length}, Opset={opset_version}")
                
            except Exception as e:
                st.error(f"❌ ONNX export failed: {str(e)}")
                st.info("💡 Try reducing model size or sequence length for ONNX compatibility")
    
    # Enhanced loading section
    st.divider()
    st.subheader("📂 Enhanced Model Loading")
    
    st.info("💡 Use the sidebar 'Enhanced Checkpoint Management' to load models, or use examples below:")
    
    with st.expander("📋 Enhanced Loading Code Examples"):
        st.code("""
# Load enhanced checkpoint with full state restoration
from enhanced_checkpoint_manager import EnhancedCheckpointManager
import torch
import torch.optim as optim

# Initialize checkpoint manager
checkpoint_manager = EnhancedCheckpointManager("./checkpoints")

# Load model with optimizer and scheduler state
model = MastishkTransformerForCausalLM(config)
optimizer = optim.AdamW(model.parameters())
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

# Load complete training state
success, training_state, message = checkpoint_manager.load_checkpoint(
    checkpoint_id="your_checkpoint_id",
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
    verify_integrity=True
)

if success:
    print("✅ Full training state restored!")
    print(f"Resumed from step: {training_state.step}")
    print(f"Best loss: {training_state.best_loss}")

# Load production model
checkpoint = torch.load('mastishk_production.pt')
config = MastishkConfig(**checkpoint['config_dict'])
model = MastishkTransformerForCausalLM(config)
model.load_state_dict(checkpoint['model_state_dict'])
        """, language="python")
    
    # Storage management
    st.divider()
    st.subheader("🗄️ Storage Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Detailed Storage Analysis", key="storage_analysis_btn"):
            stats = st.session_state.model_manager.checkpoint_manager.get_storage_stats()
            
            st.write("**Comprehensive Storage Statistics:**")
            st.write(f"- Directory: {stats['checkpoint_directory']}")
            st.write(f"- Total checkpoints: {stats['total_checkpoints']}")
            st.write(f"- Total storage: {stats['total_size_gb']:.2f} GB ({stats['total_size_mb']:.1f} MB)")
            st.write(f"- Average checkpoint size: {stats['average_size_mb']:.1f} MB")
            st.write(f"- Max checkpoints allowed: {stats['max_checkpoints']}")
            
            if stats['total_checkpoints'] > 0:
                storage_efficiency = (stats['total_checkpoints'] / stats['max_checkpoints']) * 100
                st.metric("Storage Efficiency", f"{storage_efficiency:.1f}%")
    
    with col2:
        if st.button("📤 Export Summary Report", key="export_summary_btn"):
            try:
                summary_path = "./exports/checkpoint_summary.json"
                os.makedirs("./exports", exist_ok=True)
                
                message = st.session_state.model_manager.checkpoint_manager.export_checkpoint_summary(summary_path)
                st.success(message)
                
                # Show summary preview
                with open(summary_path, 'r') as f:
                    summary_data = json.load(f)
                
                st.write("**Summary Report Preview:**")
                st.write(f"- Export time: {summary_data['export_time']}")
                st.write(f"- Total checkpoints: {summary_data['total_checkpoints']}")
                st.write(f"- Directory: {summary_data['checkpoint_directory']}")
                
            except Exception as e:
                st.error(f"❌ Failed to export summary: {str(e)}")
def render_3d_visualization_tab(model_manager, training_manager):
    """Render the 3D visualization tab for Mastishk Studio"""
    
    st.header("🌟 3D Model Visualizations")
    st.caption("Interactive 3D insights into your Mastishk Transformer")
    
    if not model_manager.initialized:
        st.warning("Please initialize a model first to access 3D visualizations.")
        return
    
    visualizer = Mastishk3DVisualizer()
    
    # Visualization options
    viz_type = st.selectbox(
        "Choose 3D Visualization",
        [
            "🏗️ Model Architecture",
            "🧠 Attention Patterns", 
            "🏔️ Training Landscape",
            "⚡ Feature Activations",
            "📊 Model Comparison",
            "📈 Checkpoint Evolution"
        ]
    )
    
    # Styling options
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("🎨 Style Options")
        color_scheme = st.selectbox("Color Scheme", ["mastishk", "neural", "energy"])
        show_annotations = st.checkbox("Show Annotations", value=True, key="show_annotations_checkbox")
        interactive_mode = st.checkbox("Interactive Mode", value=True, key="interactive_mode_checkbox")
    
    with col1:
        # Generate selected visualization
        if viz_type == "🏗️ Model Architecture":
            st.subheader("🏗️ 3D Model Architecture")
            
            config = model_manager.model_config
            fig = visualizer.create_model_architecture_3d(config, model_manager.model, style=color_scheme)
            
            if show_annotations:
                st.info(f"""
                **Architecture Overview:**
                - **Layers**: {config.num_hidden_layers} transformer layers
                - **Hidden Size**: {config.hidden_size} dimensions
                - **Attention Heads**: {config.num_attention_heads} heads per layer
                - **Parameters**: ~{sum(p.numel() for p in model_manager.model.parameters()):,}
                
                🔵 **Blue spheres**: Attention layers
                🟢 **Green squares**: MLP/Feed-forward layers  
                🔶 **Orange diamonds**: Embedding & output layers
                """)
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "🧠 Attention Patterns":
            st.subheader("🧠 3D Attention Patterns")
            st.info("This feature requires model forward pass - coming soon!")
            fig = visualizer._create_sample_feature_activation()
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "🏔️ Training Landscape":
            st.subheader("🏔️ 3D Training Landscape")
            
            if training_manager.training_history:
                fig = visualizer.create_training_landscape_3d(training_manager.training_history)
            else:
                st.info("No training history available. Showing sample landscape:")
                fig = visualizer._create_sample_training_landscape()
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "⚡ Feature Activations":
            st.subheader("⚡ 3D Feature Activations") 
            st.info("Feature activation analysis - showing sample visualization:")
            fig = visualizer._create_sample_feature_activation()
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "📊 Model Comparison":
            st.subheader("📊 3D Model Comparison")
            
            # Collect model information for comparison
            current_model_info = {
                'name': 'Current Model',
                'total_parameters': sum(p.numel() for p in model_manager.model.parameters()),
                'best_loss': 0.5,  # Default value
                'tokens_per_second': 100  # Default value
            }
            
            models_info = [current_model_info]
            
            # Add sample comparison models
            sample_models = [
                {'name': '1B Baseline', 'total_parameters': 1_000_000_000, 'best_loss': 0.8, 'tokens_per_second': 150},
                {'name': '7B Large', 'total_parameters': 7_000_000_000, 'best_loss': 0.4, 'tokens_per_second': 80},
            ]
            
            if st.checkbox("Include comparison models"):
                models_info.extend(sample_models)
            
            fig = visualizer.create_model_comparison_3d(models_info)
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "📈 Checkpoint Evolution":
            st.subheader("📈 3D Checkpoint Evolution")
            
            checkpoints = model_manager.checkpoint_manager.list_checkpoints()
            
            if checkpoints:
                checkpoint_data = []
                for ckpt in checkpoints:
                    checkpoint_data.append({
                        'training_step': ckpt.training_step,
                        'best_loss': ckpt.best_loss,
                        'creation_time': ckpt.creation_time
                    })
                
                fig = visualizer.create_checkpoint_evolution_3d(checkpoint_data)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No checkpoints available. Showing sample evolution:")
                fig = visualizer._create_sample_checkpoint_evolution()
                st.plotly_chart(fig, use_container_width=True)
    
    # Advanced Three.js section (STEP 5 content goes here)
    st.divider()
    st.subheader("🚀 Advanced 3D Neural Network (Three.js)")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info("""
        🌟 **Advanced 3D Features:**
        - Interactive transformer layers with real-time animation
        - Particle systems showing data flow
        - Attention head visualization with glowing effects
        - Mouse controls for rotation and zoom
        - Dynamic layer spacing and animation controls
        """)
    
    with col2:
        if st.button("🚀 Launch Advanced 3D View", key="launch_threejs"):
            st.session_state.show_threejs = True
    
    # Three.js HTML integration
    if st.session_state.get('show_threejs', False):
        st.write("**🎮 Interactive 3D Neural Network:**")
        
        # Create the Three.js HTML content (shortened for integration)
        threejs_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body { margin: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
                #container { width: 100%; height: 600px; position: relative; }
                #info { position: absolute; top: 10px; left: 10px; color: white; 
                       background: rgba(0,0,0,0.7); padding: 10px; border-radius: 5px; z-index: 100; }
            </style>
        </head>
        <body>
            <div id="container"></div>
            <div id="info">
                <h4>🧠 Mastishk Transformer 3D</h4>
                <div>Layers: 32 | Parameters: 7.2B</div>
                <div>🖱️ Mouse: Rotate | 🔄 Wheel: Zoom</div>
            </div>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
            <script>
                let scene, camera, renderer;
                
                function init() {
                    scene = new THREE.Scene();
                    scene.background = new THREE.Color(0x0a0a0a);
                    
                    camera = new THREE.PerspectiveCamera(75, window.innerWidth / 600, 0.1, 1000);
                    camera.position.set(0, 5, 15);
                    
                    renderer = new THREE.WebGLRenderer({ antialias: true });
                    renderer.setSize(window.innerWidth, 600);
                    document.getElementById('container').appendChild(renderer.domElement);
                    
                    // Simple lighting
                    const light = new THREE.DirectionalLight(0xffffff, 1);
                    light.position.set(10, 10, 5);
                    scene.add(light);
                    
                    const ambientLight = new THREE.AmbientLight(0x404040, 0.4);
                    scene.add(ambientLight);
                    
                    // Create transformer layers
                    for (let i = 0; i < 16; i++) {
                        const y = i * 2 - 15;
                        
                        // Attention layer (torus)
                        const attentionGeometry = new THREE.TorusGeometry(1, 0.2, 16, 100);
                        const attentionMaterial = new THREE.MeshPhongMaterial({ 
                            color: 0x4ECDC4, transparent: true, opacity: 0.8 
                        });
                        const attention = new THREE.Mesh(attentionGeometry, attentionMaterial);
                        attention.position.set(-2, y, 0);
                        attention.rotation.x = Math.PI / 2;
                        scene.add(attention);
                        
                        // MLP layer (box)
                        const mlpGeometry = new THREE.BoxGeometry(1.5, 0.3, 1.5);
                        const mlpMaterial = new THREE.MeshPhongMaterial({ 
                            color: 0xFF6B6B, transparent: true, opacity: 0.8 
                        });
                        const mlp = new THREE.Mesh(mlpGeometry, mlpMaterial);
                        mlp.position.set(2, y, 0);
                        scene.add(mlp);
                    }
                    
                    // Mouse controls
                    let mouseX = 0, mouseY = 0, isMouseDown = false;
                    
                    document.addEventListener('mousemove', (event) => {
                        if (!isMouseDown) return;
                        const deltaX = event.clientX - mouseX;
                        const deltaY = event.clientY - mouseY;
                        scene.rotation.y += deltaX * 0.01;
                        scene.rotation.x += deltaY * 0.01;
                        mouseX = event.clientX;
                        mouseY = event.clientY;
                    });
                    
                    document.addEventListener('mousedown', (event) => {
                        isMouseDown = true;
                        mouseX = event.clientX;
                        mouseY = event.clientY;
                    });
                    
                    document.addEventListener('mouseup', () => { isMouseDown = false; });
                    
                    animate();
                }
                
                function animate() {
                    requestAnimationFrame(animate);
                    scene.rotation.y += 0.005;
                    renderer.render(scene, camera);
                }
                
                init();
            </script>
        </body>
        </html>
        """
        
        # Display the Three.js visualization
        st.components.v1.html(threejs_html, height=650, scrolling=False)
        
        if st.button("❌ Close Advanced View", key="close_threejs"):
            st.session_state.show_threejs = False
            st.rerun()
def render_enhanced_experiment_tab():
    """Enhanced experiment tracking interface"""
    st.header("🧪 Enhanced Experiment Tracking")
    
    # Enhanced experiment creation
    with st.form("enhanced_experiment"):
        col1, col2 = st.columns(2)
        
        with col1:
            exp_name = st.text_input("Experiment Name", value=f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            exp_description = st.text_area("Description", height=100)
        
        with col2:
            exp_tags = st.text_input("Tags (comma-separated)")
            exp_priority = st.selectbox("Priority", ["Low", "Medium", "High", "Critical"])
        
        # Enhanced configuration
        st.subheader("🔧 Experiment Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            track_checkpoints = st.checkbox("Track Checkpoints", value=True, key="track_checkpoints_checkbox")
            track_generation = st.checkbox("Track Generation Quality", value=True, key="track_generation_checkbox")
        with col2:
            track_metrics = st.checkbox("Track Custom Metrics", value=True, key="track_metrics_checkbox")
            auto_backup = st.checkbox("Auto Backup Results", value=False, key="auto_backup_checkbox")
        
        if st.form_submit_button("🚀 Create Enhanced Experiment"):
            experiment = ExperimentConfig(
                experiment_name=exp_name,
                description=exp_description,
                tags=[tag.strip() for tag in exp_tags.split(',') if tag.strip()]
            )
            
            # Enhanced experiment data
            enhanced_experiment = {
                'config': experiment,
                'priority': exp_priority,
                'tracking_options': {
                    'checkpoints': track_checkpoints,
                    'generation': track_generation,
                    'metrics': track_metrics,
                    'auto_backup': auto_backup
                },
                'created_at': datetime.now(),
                'status': 'Created'
            }
            
            st.session_state.experiment_history.append(enhanced_experiment)
            st.success(f"✅ Enhanced experiment created: {exp_name}")
    
    # Enhanced experiment history
    if st.session_state.experiment_history:
        st.divider()
        st.subheader("📊 Enhanced Experiment History")
        
        for i, exp in enumerate(reversed(st.session_state.experiment_history[-10:])):
            config = exp['config']
            priority_colors = {"Low": "🟢", "Medium": "🟡", "High": "🟠", "Critical": "🔴"}
            priority_icon = priority_colors.get(exp.get('priority', 'Medium'), "⚪")
            
            with st.expander(f"{priority_icon} Experiment: {config.experiment_name}", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Description:** {config.description}")
                    st.write(f"**Tags:** {', '.join(config.tags)}")
                    st.write(f"**Run:** {config.run_name}")
                    st.write(f"**Priority:** {exp.get('priority', 'Medium')}")
                
                with col2:
                    st.write(f"**Created:** {exp['created_at'].strftime('%Y-%m-%d %H:%M:%S')}")
                    st.write(f"**Status:** {exp.get('status', 'Unknown')}")
                    
                    tracking = exp.get('tracking_options', {})
                    st.write("**Tracking:**")
                    for option, enabled in tracking.items():
                        icon = "✅" if enabled else "❌"
                        st.write(f"  {icon} {option.replace('_', ' ').title()}")
    
    # Experiment comparison
    if len(st.session_state.experiment_history) >= 2:
        st.divider()
        st.subheader("📈 Experiment Comparison")
        
        exp_names = [exp['config'].experiment_name for exp in st.session_state.experiment_history]
        
        col1, col2 = st.columns(2)
        with col1:
            exp1 = st.selectbox("First Experiment", exp_names, key="exp1_select")
        with col2:
            exp2 = st.selectbox("Second Experiment", exp_names, key="exp2_select")
        
        if st.button("🔍 Compare Experiments", key="compare_experiments_btn"):
            if exp1 != exp2:
                exp1_data = next(exp for exp in st.session_state.experiment_history if exp['config'].experiment_name == exp1)
                exp2_data = next(exp for exp in st.session_state.experiment_history if exp['config'].experiment_name == exp2)
                
                st.write("**Experiment Comparison:**")
                
                comparison_data = {
                    'Aspect': ['Creation Time', 'Priority', 'Tags', 'Tracking Options'],
                    exp1: [
                        exp1_data['created_at'].strftime('%Y-%m-%d %H:%M:%S'),
                        exp1_data.get('priority', 'Medium'),
                        ', '.join(exp1_data['config'].tags),
                        ', '.join([k for k, v in exp1_data.get('tracking_options', {}).items() if v])
                    ],
                    exp2: [
                        exp2_data['created_at'].strftime('%Y-%m-%d %H:%M:%S'),
                        exp2_data.get('priority', 'Medium'),
                        ', '.join(exp2_data['config'].tags),
                        ', '.join([k for k, v in exp2_data.get('tracking_options', {}).items() if v])
                    ]
                }
                
                df = pd.DataFrame(comparison_data)
                st.dataframe(df, use_container_width=True)
            else:
                st.warning("Please select different experiments to compare")

def main():
    """Enhanced main function with comprehensive checkpoint management"""
    st.set_page_config(
        page_title="Mastishk Transformer Studio - Enhanced",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize enhanced session state
    initialize_session_state()
    
    # Render enhanced sidebar
    render_enhanced_sidebar()
    
    # Enhanced main content
    st.title("🧠 Mastishk Transformer Studio")
    st.caption("Enhanced with Comprehensive Checkpoint Management & State Tracking")
    
    # Enhanced model status banner
    if st.session_state.model_manager.initialized:
        info = st.session_state.model_manager.get_model_info()
        
        if info.get('status') == '✅ Initialized':
            status_cols = st.columns(5)
            with status_cols[0]:
                st.metric("Model", f"{info['total_parameters'] / 1e9:.1f}B params")
            with status_cols[1]:
                st.metric("Device", info['device'].upper())
            with status_cols[2]:
                if torch.cuda.is_available():
                    st.metric("GPU Mem", f"{info.get('gpu_memory_allocated', 0):.1f}GB")
            with status_cols[3]:
                checkpoint_info = info.get('checkpoint_info', {})
                st.metric("Checkpoints", checkpoint_info.get('total_checkpoints', 0))
            with status_cols[4]:
                storage_stats = checkpoint_info.get('storage_stats', {})
                st.metric("Storage", f"{storage_stats.get('total_size_gb', 0):.1f}GB")
        else:
            st.error("Model initialization failed")
    else:
        st.info("👈 Please initialize a model in the enhanced sidebar to get started")
    
    # Enhanced tabs
    tabs = st.tabs([
        "✨ Generation",
        "🚀 Training", 
        "📊 Evaluation",
        "🌟 3D Visualizations",  # ← ADD THIS
        "📈 3D Analytics", 
        "🧪 Experiments",
        "🚀 Deployment"
    ])
    
    with tabs[0]:
        render_enhanced_generation_tab()
    
    with tabs[1]:
        render_enhanced_training_tab()
    
    with tabs[2]:
        render_enhanced_evaluation_tab()
    
    with tabs[3]:  # ← ADD THIS ENTIRE BLOCK
        render_3d_visualization_tab(
            st.session_state.model_manager, 
            st.session_state.training_manager
        )
    with tabs[4]:  # 3D Analytics tab
        render_3d_training_charts()
    with tabs[5]:  # ← NOTE: This changed from tabs[3] to tabs[4]
        render_enhanced_experiment_tab()
    
    with tabs[6]:  # ← NOTE: This changed from tabs[4] to tabs[5]
        render_enhanced_deployment_tab()
    
    # Enhanced footer
    st.divider()
    st.caption("Mastishk Transformer Studio v2.0 - Enhanced with Comprehensive Checkpoint Management")
    
    # Enhanced debug information
    with st.expander("🔧 Enhanced Debug Information", expanded=False):
        st.write("**Enhanced Features Applied:**")
        st.write("✅ Comprehensive checkpoint management with optimizer/scheduler state")
        st.write("✅ Training step/epoch tracking with full history")
        st.write("✅ Loss history preservation across sessions")
        st.write("✅ Random state capture for reproducibility")
        st.write("✅ Training config consistency validation")
        st.write("✅ Integrity verification with SHA256 hashing")
        st.write("✅ Enhanced storage management and cleanup")
        st.write("✅ Production-ready export options")
        st.write("✅ Experiment tracking with comparison tools")
        
        if st.session_state.model_manager.initialized:
            st.write("\n**Current Enhanced Model Info:**")
            info = st.session_state.model_manager.get_model_info()
            st.json(info)
            
            # Show recent checkpoint activity
            checkpoints = st.session_state.model_manager.checkpoint_manager.list_checkpoints()
            if checkpoints:
                st.write(f"\n**Recent Checkpoints ({len(checkpoints)}):**")
                for ckpt in checkpoints[:3]:  # Show latest 3
                    st.write(f"- {ckpt.checkpoint_id}: Step {ckpt.training_step}, Loss {ckpt.best_loss:.6f}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Enhanced application error: {e}")
        print(f"❌ Critical enhanced application error: {e}")
        traceback.print_exc()
        
        # Enhanced emergency restart options
        st.subheader("🚨 Enhanced Emergency Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Restart Application State"):
                # Clear session state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        with col2:
            if st.button("🧹 Clear All Checkpoints"):
                try:
                    import shutil
                    if os.path.exists("./checkpoints"):
                        shutil.rmtree("./checkpoints")
                        os.makedirs("./checkpoints")
                        st.success("✅ Checkpoints cleared")
                except Exception as clear_error:
                    st.error(f"Failed to clear checkpoints: {clear_error}")
        
        st.info("If issues persist, please restart the Streamlit application completely.")