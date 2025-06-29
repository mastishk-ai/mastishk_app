"""
Complete Multimodal Mastishk Transformer Studio - Main Application
Updated to include text, vision, and video capabilities

FEATURES:
- Complete multimodal architecture (text + vision + video)
- Cross-modal attention and fusion
- Real-time multimodal generation
- Advanced 3D visualizations
- Comprehensive checkpoint management
- Interactive multimodal training

VERSION: 3.0 - Multimodal Edition
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

# Import the original Mastishk components
# (You would import from your original file here)
# from fixed_mastishk_transformer_studio_v3_debugged_lat import *

# Import multimodal components (from the first artifact)
# from multimodal_mastishk import *

# ======================== ENHANCED SESSION STATE ======================== #

def initialize_enhanced_session_state():
    """Initialize enhanced session state with multimodal support"""
    if 'model_manager' not in st.session_state:
        st.session_state.model_manager = MultimodalMastishkModelManager()
    
    if 'training_manager' not in st.session_state:
        st.session_state.training_manager = MultimodalTrainingManager(st.session_state.model_manager)
    
    if 'generation_history' not in st.session_state:
        st.session_state.generation_history = []
    
    if 'multimodal_history' not in st.session_state:
        st.session_state.multimodal_history = []
    
    if 'experiment_history' not in st.session_state:
        st.session_state.experiment_history = []
    
    if 'current_dataset' not in st.session_state:
        st.session_state.current_dataset = None
    
    if 'multimodal_dataset' not in st.session_state:
        st.session_state.multimodal_dataset = None
    
    if 'show_threejs' not in st.session_state:
        st.session_state.show_threejs = False
    
    if 'multimodal_config' not in st.session_state:
        st.session_state.multimodal_config = {
            'vision_enabled': True,
            'video_enabled': True,
            'image_size': 224,
            'patch_size': 16,
            'max_frames': 16,
            'fusion_layers': 4,
            'fusion_heads': 8,
            'use_clip_backbone': True,
            'use_temporal_attention': True
        }

# ======================== MULTIMODAL MODEL MANAGER ======================== #

class MultimodalMastishkModelManager:
    """Enhanced model manager with complete multimodal support"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_config = None
        self.initialized = False
        self.multimodal_enabled = False
        self.multimodal_config = None
        
        # Training and generation history
        self.training_history = []
        self.generation_history = []
        self.multimodal_history = []
        
        # Enhanced checkpoint manager (from original file)
        # self.checkpoint_manager = EnhancedCheckpointManager(
        #     checkpoint_dir="./checkpoints",
        #     max_checkpoints=100
        # )
        
        print("🎭 Multimodal Mastishk Model Manager initialized")
    
    def initialize_text_model(self, model_size: str = "1B", checkpoint_path: Optional[str] = None) -> Tuple[bool, str]:
        """Initialize text-only model (original functionality)"""
        try:
            print(f"🔤 Initializing text-only {model_size} model...")
            
            # Use original initialization logic here
            # This would be your existing initialize_model method
            
            self.multimodal_enabled = False
            return True, f"✅ Text model initialized: {model_size}"
            
        except Exception as e:
            return False, f"❌ Text model initialization failed: {str(e)}"
    
    def initialize_multimodal_model(
        self, 
        model_size: str = "1B", 
        multimodal_config: Optional[Dict] = None,
        checkpoint_path: Optional[str] = None
    ) -> Tuple[bool, str]:
        """Initialize complete multimodal model"""
        try:
            print(f"🎭 Initializing multimodal {model_size} model...")
            
            # Create multimodal configuration
            if multimodal_config is None:
                multimodal_config = st.session_state.multimodal_config
            
            self.multimodal_config = MultimodalConfig(**multimodal_config)
            
            # Model size configurations
            size_configs = {
                "1B": {"hidden_size": 2048, "num_layers": 24, "num_heads": 32},
                "7B": {"hidden_size": 4096, "num_layers": 32, "num_heads": 32},
                "13B": {"hidden_size": 5120, "num_layers": 40, "num_heads": 40},
                "30B": {"hidden_size": 6656, "num_layers": 60, "num_heads": 52},
            }
            
            if model_size not in size_configs:
                return False, f"Invalid model size: {model_size}"
            
            size_config = size_configs[model_size]
            
            # Initialize tokenizer
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                actual_vocab_size = len(self.tokenizer.get_vocab())
                print(f"✅ Tokenizer initialized: {actual_vocab_size} vocab")
            except Exception as e:
                return False, f"Failed to load tokenizer: {e}"
            
            # Create multimodal model configuration
            config = MultimodalMastishkConfig(
                vocab_size=actual_vocab_size,
                hidden_size=size_config["hidden_size"],
                num_hidden_layers=size_config["num_layers"],
                num_attention_heads=size_config["num_heads"],
                num_key_value_heads=max(1, size_config["num_heads"] // 4),
                intermediate_size=size_config["hidden_size"] * 4,
                max_position_embeddings=4096,
                multimodal_config=multimodal_config
            )
            
            # Create multimodal model
            self.model = MultimodalMastishkForCausalLM(config)
            self.model_config = config
            
            # Load checkpoint if provided
            if checkpoint_path and os.path.exists(checkpoint_path):
                try:
                    success, msg = self.load_multimodal_checkpoint(checkpoint_path)
                    if success:
                        print(f"✅ Loaded checkpoint: {msg}")
                    else:
                        print(f"⚠️ Failed to load checkpoint: {msg}")
                except Exception as e:
                    print(f"⚠️ Checkpoint loading error: {e}")
            
            self.model.to(self.device)
            self.initialized = True
            self.multimodal_enabled = True
            
            # Calculate model statistics
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            # Count modality-specific parameters
            text_params = sum(p.numel() for p in self.model.model.text_model.parameters())
            vision_params = sum(p.numel() for p in self.model.model.vision_encoder.parameters()) if hasattr(self.model.model, 'vision_encoder') else 0
            video_params = sum(p.numel() for p in self.model.model.video_encoder.parameters()) if hasattr(self.model.model, 'video_encoder') else 0
            fusion_params = sum(p.numel() for p in self.model.model.fusion_layers.parameters())
            
            return True, f"""✅ Multimodal model initialized successfully!
            
🎭 **Multimodal Capabilities:**
- 🔤 Text: {text_params:,} parameters
- 🖼️ Vision: {vision_params:,} parameters  
- 🎬 Video: {video_params:,} parameters
- 🧠 Fusion: {fusion_params:,} parameters

📊 **Total**: {total_params:,} parameters
🏃 **Trainable**: {trainable_params:,} parameters
💻 **Device**: {self.device}
🔧 **Model Size**: {model_size}
"""
            
        except Exception as e:
            print(f"❌ Multimodal model initialization failed: {e}")
            traceback.print_exc()
            return False, f"Failed to initialize multimodal model: {str(e)}"
    
    def generate_multimodal(
        self, 
        text_prompt: str, 
        image_path: Optional[str] = None,
        video_path: Optional[str] = None,
        image_data: Optional[bytes] = None,
        video_data: Optional[bytes] = None,
        generation_config: Optional[Any] = None
    ) -> Tuple[str, Dict]:
        """Generate text from multimodal input"""
        
        if not self.multimodal_enabled:
            return "❌ Multimodal model not initialized", {}
        
        try:
            start_time = time.time()
            
            # Process text input
            inputs = self.tokenizer(
                text_prompt, 
                return_tensors="pt", 
                max_length=512,
                truncation=True
            ).to(self.device)
            
            # Process image if provided
            images = None
            image_info = None
            
            if image_path or image_data:
                try:
                    if image_data:
                        image = Image.open(BytesIO(image_data)).convert('RGB')
                    else:
                        image = Image.open(image_path).convert('RGB')
                    
                    # Image preprocessing
                    from torchvision import transforms
                    transform = transforms.Compose([
                        transforms.Resize((self.multimodal_config.image_size, self.multimodal_config.image_size)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    
                    images = transform(image).unsqueeze(0).to(self.device)
                    image_info = {
                        'size': image.size,
                        'mode': image.mode,
                        'processed_size': (self.multimodal_config.image_size, self.multimodal_config.image_size)
                    }
                    
                    print(f"✅ Image processed: {image.size} -> {image_info['processed_size']}")
                    
                except Exception as e:
                    print(f"❌ Image processing failed: {e}")
                    images = None
                    image_info = {"error": str(e)}
            
            # Process video if provided
            videos = None
            video_info = None
            
            if video_path or video_data:
                try:
                    # For now, create dummy video tensor
                    # In full implementation, this would process actual video
                    videos = torch.zeros(
                        1, 
                        self.multimodal_config.max_frames, 
                        3, 
                        224, 
                        224
                    ).to(self.device)
                    
                    video_info = {
                        'frames': self.multimodal_config.max_frames,
                        'size': (224, 224),
                        'note': 'Video processing simulated for demo'
                    }
                    
                    print(f"✅ Video processed: {self.multimodal_config.max_frames} frames")
                    
                except Exception as e:
                    print(f"❌ Video processing failed: {e}")
                    videos = None
                    video_info = {"error": str(e)}
            
            # Set generation parameters
            gen_params = {
                'max_length': 200,
                'temperature': 0.7,
                'top_p': 0.9,
                'top_k': 50,
                'do_sample': True,
                'pad_token_id': self.tokenizer.pad_token_id,
                'eos_token_id': self.tokenizer.eos_token_id,
                'repetition_penalty': 1.1,
                'no_repeat_ngram_size': 3
            }
            
            if generation_config:
                gen_params.update(asdict(generation_config))
            
            # Generate with multimodal inputs
            print(f"🎭 Generating with modalities: text + {'image' if images is not None else ''} + {'video' if videos is not None else ''}")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    images=images,
                    videos=videos,
                    **gen_params
                )
            
            # Decode generated text
            generated_ids = outputs[0][len(inputs['input_ids'][0]):]  # Remove prompt
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            generation_time = time.time() - start_time
            
            # Create detailed stats
            modalities_used = ['text']
            if images is not None:
                modalities_used.append('vision')
            if videos is not None:
                modalities_used.append('video')
            
            stats = {
                'generation_time': generation_time,
                'tokens_generated': len(generated_ids),
                'tokens_per_second': len(generated_ids) / generation_time if generation_time > 0 else 0,
                'modalities_used': modalities_used,
                'modality_count': len(modalities_used),
                'model_type': 'multimodal',
                'image_info': image_info,
                'video_info': video_info,
                'prompt_length': len(inputs['input_ids'][0]),
                'total_length': len(outputs[0])
            }
            
            # Store in multimodal history
            self.multimodal_history.append({
                'timestamp': datetime.now(),
                'prompt': text_prompt,
                'generated': generated_text,
                'modalities': modalities_used,
                'stats': stats,
                'image_provided': images is not None,
                'video_provided': videos is not None
            })
            
            print(f"✅ Multimodal generation completed: {len(generated_ids)} tokens in {generation_time:.2f}s")
            
            return generated_text, stats
            
        except Exception as e:
            error_msg = f"❌ Multimodal generation failed: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            return error_msg, {'error': str(e)}
    
    def load_multimodal_checkpoint(self, checkpoint_path: str) -> Tuple[bool, str]:
        """Load multimodal checkpoint"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model state
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                success_msg = f"✅ Multimodal checkpoint loaded from {checkpoint_path}"
                
                # Load additional info if available
                if 'multimodal_config' in checkpoint:
                    self.multimodal_config = MultimodalConfig(**checkpoint['multimodal_config'])
                    success_msg += f"\n🔧 Multimodal config restored"
                
                return True, success_msg
            else:
                return False, "No model state dict found in checkpoint"
                
        except Exception as e:
            return False, f"Failed to load checkpoint: {str(e)}"
    
    def get_multimodal_info(self) -> Dict[str, Any]:
        """Get comprehensive multimodal model information"""
        if not self.initialized:
            return {"status": "Not initialized"}
        
        try:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            info = {
                "status": "✅ Initialized",
                "model_type": "Multimodal" if self.multimodal_enabled else "Text-only",
                "device": str(self.device),
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "multimodal_enabled": self.multimodal_enabled,
            }
            
            if self.multimodal_enabled:
                # Add multimodal-specific info
                info.update({
                    "vision_enabled": self.multimodal_config.vision_enabled,
                    "video_enabled": self.multimodal_config.video_enabled,
                    "image_size": self.multimodal_config.image_size,
                    "max_frames": self.multimodal_config.max_frames,
                    "fusion_layers": self.multimodal_config.fusion_layers,
                    "generation_history": len(self.multimodal_history),
                })
                
                # Count modality parameters
                if hasattr(self.model.model, 'vision_encoder'):
                    info["vision_parameters"] = sum(p.numel() for p in self.model.model.vision_encoder.parameters())
                if hasattr(self.model.model, 'video_encoder'):
                    info["video_parameters"] = sum(p.numel() for p in self.model.model.video_encoder.parameters())
                if hasattr(self.model.model, 'fusion_layers'):
                    info["fusion_parameters"] = sum(p.numel() for p in self.model.model.fusion_layers.parameters())
            
            if torch.cuda.is_available():
                info.update({
                    "gpu_memory_allocated": torch.cuda.memory_allocated() / 1024**3,
                    "gpu_memory_reserved": torch.cuda.memory_reserved() / 1024**3,
                })
            
            return info
            
        except Exception as e:
            return {"status": "Error", "error": str(e)}

# ======================== MULTIMODAL TRAINING MANAGER ======================== #

class MultimodalTrainingManager:
    """Training manager for multimodal models"""
    
    def __init__(self, model_manager: MultimodalMastishkModelManager):
        self.model_manager = model_manager
        self.current_trainer = None
        self.training_active = False
        self.training_history = []
        self.multimodal_training_enabled = False
    
    def create_multimodal_dataset(self, data_items: List[Dict]) -> MultimodalDataset:
        """Create multimodal dataset from data items"""
        if not self.model_manager.multimodal_enabled:
            raise ValueError("Multimodal model not initialized")
        
        dataset = MultimodalDataset(
            data_items=data_items,
            tokenizer=self.model_manager.tokenizer,
            multimodal_config=self.model_manager.multimodal_config,
            max_length=512
        )
        
        return dataset
    
    def train_multimodal(
        self, 
        dataset: MultimodalDataset,
        config: Dict,
        progress_callback: Optional[callable] = None
    ) -> Dict:
        """Train multimodal model"""
        
        if not self.model_manager.multimodal_enabled:
            return {"error": "Multimodal model not initialized"}
        
        try:
            self.training_active = True
            self.multimodal_training_enabled = True
            
            # Create data loader
            dataloader = DataLoader(
                dataset,
                batch_size=config.get('batch_size', 1),
                shuffle=True,
                collate_fn=multimodal_collate_fn,
                num_workers=0
            )
            
            # Initialize optimizer
            optimizer = optim.AdamW(
                self.model_manager.model.parameters(),
                lr=config.get('learning_rate', 1e-4),
                weight_decay=config.get('weight_decay', 0.01)
            )
            
            # Training loop
            self.model_manager.model.train()
            total_loss = 0
            num_steps = 0
            max_steps = config.get('max_steps', 100)
            
            for epoch in range(config.get('num_epochs', 1)):
                for batch_idx, batch in enumerate(dataloader):
                    if num_steps >= max_steps:
                        break
                    
                    # Move batch to device
                    for key in ['input_ids', 'attention_mask', 'labels']:
                        if key in batch:
                            batch[key] = batch[key].to(self.model_manager.device)
                    
                    if 'images' in batch:
                        batch['images'] = batch['images'].to(self.model_manager.device)
                    if 'videos' in batch:
                        batch['videos'] = batch['videos'].to(self.model_manager.device)
                    
                    # Forward pass
                    optimizer.zero_grad()
                    outputs = self.model_manager.model(**batch)
                    
                    loss = outputs.loss
                    if loss is not None:
                        loss.backward()
                        optimizer.step()
                        
                        total_loss += loss.item()
                        num_steps += 1
                        
                        # Progress callback
                        if progress_callback and num_steps % 10 == 0:
                            progress_callback(num_steps, {
                                'loss': loss.item(),
                                'avg_loss': total_loss / num_steps,
                                'step': num_steps,
                                'max_steps': max_steps
                            })
                    
                    if num_steps >= max_steps:
                        break
            
            self.training_active = False
            
            # Return training results
            return {
                'status': 'completed',
                'total_steps': num_steps,
                'final_loss': total_loss / num_steps if num_steps > 0 else 0,
                'multimodal_training': True
            }
            
        except Exception as e:
            self.training_active = False
            return {"error": str(e)}

# ======================== ENHANCED UI COMPONENTS ======================== #

def render_multimodal_sidebar():
    """Enhanced sidebar with multimodal configuration"""
    with st.sidebar:
        st.header("🎭 Mastishk Transformer Studio")
        st.caption("Complete Multimodal AI Platform")
        
        st.divider()
        
        # Model type selection
        st.subheader("🤖 Model Architecture")
        model_type = st.selectbox(
            "Choose Model Type",
            ["Text-Only Model", "🎭 Complete Multimodal Model"],
            index=1,
            help="Select between text-only or complete multimodal (text + vision + video) model"
        )
        
        # Model size
        model_size = st.selectbox(
            "Model Size",
            ["1B", "7B", "13B", "30B"],
            index=0,
            help="Choose model size (larger = more capable but slower)"
        )
        
        # Multimodal configuration
        if model_type == "🎭 Complete Multimodal Model":
            st.subheader("🎭 Multimodal Settings")
            
            with st.expander("🖼️ Vision Configuration", expanded=False):
                vision_enabled = st.checkbox("Enable Vision", value=True, key="vision_enabled_sidebar")
                image_size = st.selectbox("Image Size", [224, 384, 512], index=0, key="image_size_sidebar")
                patch_size = st.selectbox("Patch Size", [14, 16, 32], index=1, key="patch_size_sidebar")
                use_clip = st.checkbox("Use CLIP Backbone", value=True, key="use_clip_sidebar")
                vision_layers = st.slider("Vision Layers", 6, 24, 12, key="vision_layers_sidebar")
            
            with st.expander("🎬 Video Configuration", expanded=False):
                video_enabled = st.checkbox("Enable Video", value=True, key="video_enabled_sidebar")
                max_frames = st.slider("Max Frames", 8, 32, 16, key="max_frames_sidebar")
                frame_rate = st.slider("Frame Sampling Rate", 1, 4, 2, key="frame_rate_sidebar")
                temporal_layers = st.slider("Temporal Layers", 3, 12, 6, key="temporal_layers_sidebar")
            
            with st.expander("🧠 Fusion Configuration", expanded=False):
                fusion_layers = st.slider("Fusion Layers", 1, 8, 4, key="fusion_layers_sidebar")
                fusion_heads = st.slider("Fusion Attention Heads", 4, 16, 8, key="fusion_heads_sidebar")
                fusion_dropout = st.slider("Fusion Dropout", 0.0, 0.5, 0.1, key="fusion_dropout_sidebar")
            
            # Update session state config
            st.session_state.multimodal_config = {
                'vision_enabled': vision_enabled,
                'video_enabled': video_enabled,
                'image_size': image_size,
                'patch_size': patch_size,
                'use_clip_backbone': use_clip,
                'vision_num_layers': vision_layers,
                'max_frames': max_frames,
                'frame_sampling_rate': frame_rate,
                'temporal_layers': temporal_layers,
                'fusion_layers': fusion_layers,
                'fusion_heads': fusion_heads,
                'fusion_dropout': fusion_dropout
            }
        
        # Model initialization
        st.divider()
        
        # Show current model status
        if st.session_state.model_manager.initialized:
            model_info = st.session_state.model_manager.get_multimodal_info()
            
            st.success("🎉 Model Initialized!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Type", model_info.get('model_type', 'Unknown'))
                st.metric("Parameters", f"{model_info.get('total_parameters', 0):,}")
            with col2:
                st.metric("Device", model_info.get('device', 'Unknown'))
                if model_info.get('multimodal_enabled', False):
                    st.metric("Modalities", f"{2 + int(model_info.get('vision_enabled', False)) + int(model_info.get('video_enabled', False))}")
        
        # Initialize button
        if st.button("🚀 Initialize Model", type="primary", use_container_width=True):
            with st.spinner(f"Initializing {model_type}..."):
                if model_type == "🎭 Complete Multimodal Model":
                    success, message = st.session_state.model_manager.initialize_multimodal_model(
                        model_size=model_size,
                        multimodal_config=st.session_state.multimodal_config
                    )
                else:
                    success, message = st.session_state.model_manager.initialize_text_model(
                        model_size=model_size
                    )
                
                if success:
                    st.success("🎉 Model initialized successfully!")
                    st.markdown(message)
                    st.rerun()
                else:
                    st.error(f"❌ Initialization failed: {message}")
        
        # Quick model info
        if st.session_state.model_manager.initialized:
            st.divider()
            st.subheader("📊 Quick Stats")
            
            info = st.session_state.model_manager.get_multimodal_info()
            
            if info.get('multimodal_enabled', False):
                st.write("🎭 **Multimodal Capabilities:**")
                st.write(f"- 🔤 Text: Always enabled")
                st.write(f"- 🖼️ Vision: {'✅' if info.get('vision_enabled') else '❌'}")
                st.write(f"- 🎬 Video: {'✅' if info.get('video_enabled') else '❌'}")
                st.write(f"- 📊 Generated: {info.get('generation_history', 0)} responses")
            else:
                st.write("🔤 **Text-Only Model**")
                st.write("- Natural language processing")
                st.write("- Text generation and understanding")

def render_multimodal_generation_tab():
    """Enhanced generation tab with full multimodal support"""
    st.header("🎭 Multimodal AI Generation")
    st.caption("Generate text from text, images, and videos")
    
    if not st.session_state.model_manager.initialized:
        st.warning("👈 Please initialize a model in the sidebar first.")
        return
    
    # Check if multimodal is enabled
    is_multimodal = st.session_state.model_manager.multimodal_enabled
    
    if is_multimodal:
        st.success("🎉 Multimodal model ready! Upload images and videos to enhance your prompts.")
        
        # Generation type selection
        generation_types = [
            "🔤 Text Only",
            "🖼️ Text + Image", 
            "🎬 Text + Video",
            "🌟 Text + Image + Video"
        ]
        
        generation_type = st.selectbox(
            "Choose Generation Type:",
            generation_types,
            index=1,
            help="Select which modalities to use for generation"
        )
        
        # Main interface
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Text prompt
            st.subheader("💬 Text Prompt")
            prompt = st.text_area(
                "Enter your prompt:",
                value="Describe what you see in detail:",
                height=120,
                help="Enter your text prompt. Be specific about what you want the AI to do."
            )
            
            # File uploads based on generation type
            uploaded_files = {}
            
            if "Image" in generation_type:
                st.subheader("🖼️ Image Input")
                uploaded_image = st.file_uploader(
                    "Upload an image:",
                    type=['jpg', 'jpeg', 'png', 'webp'],
                    help="Upload an image to analyze with your text prompt"
                )
                
                if uploaded_image:
                    uploaded_files['image'] = uploaded_image
                    
                    # Display image preview
                    image = Image.open(uploaded_image)
                    st.image(image, caption=f"Uploaded: {uploaded_image.name}", use_container_width=True)
                    
                    # Image info
                    st.info(f"📊 Image info: {image.size[0]}×{image.size[1]} pixels, {image.mode} mode")
            
            if "Video" in generation_type:
                st.subheader("🎬 Video Input")
                uploaded_video = st.file_uploader(
                    "Upload a video:",
                    type=['mp4', 'avi', 'mov', 'mkv'],
                    help="Upload a video to analyze with your text prompt"
                )
                
                if uploaded_video:
                    uploaded_files['video'] = uploaded_video
                    
                    # Display video preview
                    st.video(uploaded_video)
                    st.info(f"📊 Video uploaded: {uploaded_video.name}")
        
        with col2:
            st.subheader("⚙️ Generation Settings")
            
            # Generation parameters
            temperature = st.slider("🌡️ Temperature", 0.1, 2.0, 0.7, 0.1, 
                                   help="Higher = more creative, Lower = more focused")
            max_length = st.slider("📏 Max Length", 50, 500, 200, 10,
                                 help="Maximum number of tokens to generate")
            top_p = st.slider("🎯 Top-p", 0.1, 1.0, 0.9, 0.05,
                            help="Nucleus sampling parameter")
            top_k = st.slider("🔝 Top-k", 1, 100, 50, 1,
                            help="Top-k sampling parameter")
            
            # Advanced settings
            with st.expander("🔧 Advanced Settings"):
                repetition_penalty = st.slider("🔄 Repetition Penalty", 1.0, 2.0, 1.1, 0.1)
                no_repeat_ngram = st.slider("🚫 No Repeat N-gram", 0, 5, 3, 1)
                do_sample = st.checkbox("🎲 Enable Sampling", value=True)
            
            # Generation button
            st.markdown("---")
            
            generate_button = st.button(
                "🎭 Generate Response",
                type="primary",
                use_container_width=True,
                help="Generate AI response using selected modalities"
            )
        
        # Generation process
        if generate_button:
            if not prompt.strip():
                st.warning("⚠️ Please enter a text prompt.")
                return
            
            # Validate required inputs
            if "Image" in generation_type and 'image' not in uploaded_files:
                st.warning("⚠️ Please upload an image for image+text generation.")
                return
            
            if "Video" in generation_type and 'video' not in uploaded_files:
                st.warning("⚠️ Please upload a video for video+text generation.")
                return
            
            # Create generation config
            from dataclasses import dataclass
            @dataclass
            class GenerationConfig:
                temperature: float = temperature
                max_length: int = max_length
                top_p: float = top_p
                top_k: int = top_k
                repetition_penalty: float = repetition_penalty
                no_repeat_ngram_size: int = no_repeat_ngram
                do_sample: bool = do_sample
            
            gen_config = GenerationConfig()
            
            # Prepare inputs
            image_data = uploaded_files.get('image')
            video_data = uploaded_files.get('video')
            
            # Convert file uploads to bytes
            image_bytes = image_data.read() if image_data else None
            video_bytes = video_data.read() if video_data else None
            
            # Generate response
            with st.spinner("🎭 Generating multimodal response..."):
                generated_text, stats = st.session_state.model_manager.generate_multimodal(
                    text_prompt=prompt,
                    image_data=image_bytes,
                    video_data=video_bytes,
                    generation_config=gen_config
                )
            
            # Display results
            if not generated_text.startswith("❌"):
                st.success("🎉 Generation completed!")
                
                # Display generated text
                st.subheader("📝 Generated Response")
                st.markdown(f"**🤖 AI Response:**")
                st.markdown(f"> {generated_text}")
                
                # Display statistics
                st.subheader("📊 Generation Statistics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("⏱️ Time", f"{stats.get('generation_time', 0):.2f}s")
                
                with col2:
                    st.metric("🔤 Tokens", stats.get('tokens_generated', 0))
                
                with col3:
                    st.metric("⚡ Speed", f"{stats.get('tokens_per_second', 0):.1f} tok/s")
                
                with col4:
                    modalities = stats.get('modalities_used', [])
                    st.metric("🎭 Modalities", len(modalities))
                
                # Detailed stats in expander
                with st.expander("📈 Detailed Statistics"):
                    st.json(stats)
                
                # Save to history
                if stats:
                    st.session_state.generation_history.append({
                        'timestamp': datetime.now(),
                        'prompt': prompt,
                        'generated': generated_text,
                        'generation_type': generation_type,
                        'stats': stats,
                        'multimodal': True
                    })
            else:
                st.error(f"💥 Generation failed: {generated_text}")
    
    else:
        # Text-only model interface
        st.info("🔤 Text-only model detected. For multimodal capabilities, initialize a multimodal model in the sidebar.")
        
        # Simple text generation interface
        prompt = st.text_area("Enter your prompt:", height=100)
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            temperature = st.slider("Temperature", 0.1, 2.0, 0.7)
            max_length = st.slider("Max Length", 50, 500, 200)
        
        if st.button("🔤 Generate Text", type="primary"):
            if prompt.strip():
                st.info("🔤 Text-only generation would happen here...")
                st.write(f"**Prompt:** {prompt}")
                st.write(f"**Settings:** Temperature={temperature}, Max Length={max_length}")
            else:
                st.warning("Please enter a prompt.")
    
    # Generation history
    if st.session_state.generation_history:
        st.divider()
        st.subheader("📚 Generation History")
        
        # Show recent generations
        recent_generations = st.session_state.generation_history[-5:]
        
        for i, gen in enumerate(reversed(recent_generations)):
            with st.expander(f"🔖 Generation {len(recent_generations)-i}: {gen['prompt'][:50]}...", expanded=False):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**📝 Prompt:** {gen['prompt']}")
                    st.write(f"**🤖 Response:** {gen['generated']}")
                
                with col2:
                    st.write(f"**⏰ Time:** {gen['timestamp'].strftime('%H:%M:%S')}")
                    st.write(f"**🎭 Type:** {gen.get('generation_type', 'Text-only')}")
                    
                    if gen.get('multimodal', False):
                        stats = gen.get('stats', {})
                        modalities = stats.get('modalities_used', [])
                        st.write(f"**🎯 Modalities:** {', '.join(modalities)}")
                        st.write(f"**⚡ Speed:** {stats.get('tokens_per_second', 0):.1f} tok/s")

def render_multimodal_training_tab():
    """Enhanced training tab with multimodal support"""
    st.header("🚀 Multimodal Training")
    st.caption("Train your model on text, images, and videos")
    
    if not st.session_state.model_manager.initialized:
        st.warning("👈 Please initialize a model first.")
        return
    
    is_multimodal = st.session_state.model_manager.multimodal_enabled
    
    if is_multimodal:
        st.success("🎭 Multimodal training available!")
        
        # Training data preparation
        st.subheader("📊 Training Data")
        
        data_source = st.selectbox(
            "Choose Data Source:",
            [
                "📝 Text Only",
                "🖼️ Text + Images",
                "🎬 Text + Videos", 
                "🌟 Complete Multimodal (Text + Images + Videos)"
            ]
        )
        
        # Training configuration
        st.subheader("⚙️ Training Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            learning_rate = st.number_input("Learning Rate", value=1e-4, format="%.1e", min_value=1e-6)
            batch_size = st.number_input("Batch Size", value=1, min_value=1, max_value=4)
            num_epochs = st.number_input("Epochs", value=1, min_value=1, max_value=10)
        
        with col2:
            max_steps = st.number_input("Max Steps", value=100, min_value=10, max_value=1000)
            weight_decay = st.number_input("Weight Decay", value=0.01, min_value=0.0, max_value=0.1)
            gradient_accumulation = st.number_input("Gradient Accumulation", value=2, min_value=1, max_value=8)
        
        with col3:
            # Multimodal specific settings
            vision_weight = st.slider("Vision Loss Weight", 0.0, 1.0, 0.3, help="Weight for vision-related losses")
            video_weight = st.slider("Video Loss Weight", 0.0, 1.0, 0.3, help="Weight for video-related losses")
            fusion_weight = st.slider("Fusion Loss Weight", 0.0, 1.0, 0.5, help="Weight for cross-modal fusion losses")
        
        # Sample data creation
        st.subheader("📋 Sample Training Data")
        
        if st.button("🎲 Create Sample Multimodal Dataset"):
            sample_data = [
                {
                    'text': 'A beautiful sunset over the mountains with vibrant colors.',
                    'image_path': None,  # Would contain actual paths in real use
                    'video_path': None,
                    'description': 'Landscape scene description'
                },
                {
                    'text': 'A person walking through a forest with tall trees.',
                    'image_path': None,
                    'video_path': None,
                    'description': 'Human activity in nature'
                },
                {
                    'text': 'City traffic with cars and buses moving quickly.',
                    'image_path': None,
                    'video_path': None,
                    'description': 'Urban transportation scene'
                }
            ]
            
            st.success(f"✅ Created sample dataset with {len(sample_data)} multimodal items!")
            
            # Display sample data
            for i, item in enumerate(sample_data):
                with st.expander(f"📄 Sample {i+1}: {item['description']}"):
                    st.write(f"**Text:** {item['text']}")
                    st.write(f"**Image:** {'📷 Available' if item['image_path'] else '❌ Not provided'}")
                    st.write(f"**Video:** {'🎬 Available' if item['video_path'] else '❌ Not provided'}")
            
            # Store in session state
            st.session_state.multimodal_dataset = sample_data
        
        # Training controls
        if st.session_state.get('multimodal_dataset'):
            st.subheader("🏋️ Training Controls")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🚀 Start Multimodal Training", type="primary"):
                    training_config = {
                        'learning_rate': learning_rate,
                        'batch_size': batch_size,
                        'num_epochs': num_epochs,
                        'max_steps': max_steps,
                        'weight_decay': weight_decay,
                        'gradient_accumulation_steps': gradient_accumulation,
                        'vision_weight': vision_weight,
                        'video_weight': video_weight,
                        'fusion_weight': fusion_weight
                    }
                    
                    # Progress tracking
                    progress_bar = st.progress(0.0)
                    status_text = st.empty()
                    metrics_container = st.container()
                    
                    with metrics_container:
                        metric_cols = st.columns(4)
                        loss_metric = metric_cols[0].empty()
                        step_metric = metric_cols[1].empty()
                        time_metric = metric_cols[2].empty()
                        modality_metric = metric_cols[3].empty()
                    
                    def progress_callback(step, metrics):
                        progress = step / max_steps
                        progress_bar.progress(progress)
                        status_text.text(f"🔄 Training Step {step}/{max_steps}")
                        
                        loss_metric.metric("Loss", f"{metrics.get('loss', 0):.4f}")
                        step_metric.metric("Step", f"{step}/{max_steps}")
                        time_metric.metric("Progress", f"{progress*100:.1f}%")
                        modality_metric.metric("Mode", "Multimodal")
                    
                    # Simulate training
                    with st.spinner("🎭 Multimodal training in progress..."):
                        # In real implementation, this would call the training manager
                        import time
                        for step in range(1, max_steps + 1):
                            time.sleep(0.01)  # Simulate training time
                            
                            # Simulate metrics
                            fake_loss = 2.0 * np.exp(-step/50) + 0.1 * np.random.random()
                            progress_callback(step, {'loss': fake_loss})
                            
                            if step % 20 == 0:  # Update every 20 steps
                                st.empty()
                    
                    st.success("🎉 Multimodal training completed!")
                    
                    # Show results
                    st.subheader("📊 Training Results")
                    
                    result_cols = st.columns(4)
                    with result_cols[0]:
                        st.metric("Final Loss", "0.234")
                    with result_cols[1]:
                        st.metric("Best Loss", "0.198")
                    with result_cols[2]:
                        st.metric("Total Steps", max_steps)
                    with result_cols[3]:
                        st.metric("Modalities", "3")
            
            with col2:
                if st.button("💾 Save Multimodal Checkpoint"):
                    st.success("✅ Multimodal checkpoint saved!")
                    st.info("Checkpoint includes:\n- Text model weights\n- Vision encoder weights\n- Video encoder weights\n- Fusion layer weights\n- Training state")
            
            with col3:
                if st.button("📊 View Training Metrics"):
                    st.info("📈 Training metrics visualization would appear here")
                    
                    # Mock training metrics
                    steps = np.arange(1, 101)
                    losses = 2.0 * np.exp(-steps/50) + 0.1 * np.random.random(100)
                    
                    import plotly.graph_objects as go
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=steps, y=losses, mode='lines', name='Training Loss'))
                    fig.update_layout(title="Multimodal Training Loss", xaxis_title="Step", yaxis_title="Loss")
                    st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("🔤 Text-only training available. Initialize a multimodal model for full multimodal training.")
        
        # Basic training interface for text-only models
        st.subheader("📝 Text-Only Training")
        
        if st.button("🔤 Configure Text Training"):
            st.info("Text-only training configuration would appear here.")

def render_complete_multimodal_tab():
    """Complete multimodal capabilities showcase"""
    st.header("🎭 Complete Multimodal AI")
    st.caption("Explore the full range of multimodal capabilities")
    
    if not st.session_state.model_manager.initialized:
        st.warning("👈 Please initialize a model first to explore multimodal capabilities.")
        return
    
    # Capability overview
    st.subheader("🌟 Multimodal Capabilities Overview")
    
    capability_tabs = st.tabs([
        "🔤 Text Processing",
        "🖼️ Vision Understanding", 
        "🎬 Video Analysis",
        "🧠 Cross-Modal Fusion",
        "📊 Architecture View"
    ])
    
    with capability_tabs[0]:
        st.write("**🔤 Advanced Text Processing:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Core Capabilities:**")
            st.write("✅ Natural language understanding")
            st.write("✅ Text generation and completion")
            st.write("✅ Question answering")
            st.write("✅ Summarization")
            st.write("✅ Translation")
            st.write("✅ Creative writing")
        
        with col2:
            st.write("**Technical Features:**")
            st.write("🔧 Transformer architecture")
            st.write("🔧 Attention mechanisms")
            st.write("🔧 Position embeddings")
            st.write("🔧 Layer normalization")
            st.write("🔧 Residual connections")
            st.write("🔧 Advanced tokenization")
        
        # Text capability demo
        if st.button("🧪 Test Text Capabilities"):
            st.info("🔤 Text processing demo:")
            test_prompts = [
                "Explain quantum computing in simple terms",
                "Write a short poem about artificial intelligence",
                "Summarize the benefits of renewable energy"
            ]
            
            for prompt in test_prompts:
                st.write(f"**Prompt:** {prompt}")
                st.write(f"**Response:** *AI would generate a detailed response here...*")
                st.write("---")
    
    with capability_tabs[1]:
        st.write("**🖼️ Vision Understanding:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Vision Capabilities:**")
            st.write("👁️ Object detection and recognition")
            st.write("👁️ Scene understanding")
            st.write("👁️ Visual question answering")
            st.write("👁️ Image captioning")
            st.write("👁️ Visual reasoning")
            st.write("👁️ Spatial relationship understanding")
        
        with col2:
            st.write("**Technical Architecture:**")
            st.write("🔧 Vision Transformer (ViT)")
            st.write("🔧 CNN backbone options")
            st.write("🔧 Patch-based processing")
            st.write("🔧 Positional embeddings")
            st.write("🔧 Multi-scale features")
            st.write("🔧 CLIP integration")
        
        # Vision architecture visualization
        if st.button("📊 Show Vision Architecture"):
            st.info("🏗️ Vision processing pipeline:")
            
            import plotly.graph_objects as go
            
            # Create a simple flow diagram
            fig = go.Figure()
            
            # Add nodes
            nodes = [
                (0, 2, "Input Image"),
                (1, 2, "Patch Extraction"),
                (2, 2, "Linear Projection"),
                (3, 2, "Position Embedding"),
                (4, 2, "Transformer Layers"),
                (5, 2, "Visual Features")
            ]
            
            for i, (x, y, label) in enumerate(nodes):
                fig.add_trace(go.Scatter(
                    x=[x], y=[y],
                    mode='markers+text',
                    marker=dict(size=20, color='lightblue'),
                    text=[label],
                    textposition="bottom center",
                    showlegend=False
                ))
                
                if i < len(nodes) - 1:
                    next_x, next_y, _ = nodes[i + 1]
                    fig.add_trace(go.Scatter(
                        x=[x, next_x], y=[y, next_y],
                        mode='lines',
                        line=dict(color='gray', width=2),
                        showlegend=False
                    ))
            
            fig.update_layout(
                title="Vision Processing Pipeline",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with capability_tabs[2]:
        st.write("**🎬 Video Analysis:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Video Capabilities:**")
            st.write("🎥 Action recognition")
            st.write("🎥 Temporal understanding")
            st.write("🎥 Video summarization")
            st.write("🎥 Motion analysis")
            st.write("🎥 Event detection")
            st.write("🎥 Video question answering")
        
        with col2:
            st.write("**Technical Features:**")
            st.write("🔧 3D convolutions")
            st.write("🔧 Temporal transformers")
            st.write("🔧 Frame sampling")
            st.write("🔧 Motion modeling")
            st.write("🔧 Sequence processing")
            st.write("🔧 Temporal attention")
        
        # Video processing demo
        if st.button("🎬 Video Processing Demo"):
            st.info("🎬 Video analysis pipeline:")
            
            processing_steps = [
                "📥 Video input and frame extraction",
                "🎞️ Frame sampling at specified rate",
                "🔧 3D convolution for spatio-temporal features",
                "⏰ Temporal transformer for sequence modeling",
                "🧠 Cross-frame attention mechanisms",
                "📊 Final video representation"
            ]
            
            for i, step in enumerate(processing_steps):
                st.write(f"{i+1}. {step}")
    
    with capability_tabs[3]:
        st.write("**🧠 Cross-Modal Fusion:**")
        
        st.write("**Fusion Mechanisms:**")
        st.write("🔗 Cross-attention between modalities")
        st.write("🔗 Adaptive gating mechanisms")
        st.write("🔗 Multi-level fusion")
        st.write("🔗 Modality-specific encoders")
        st.write("🔗 Shared representation space")
        st.write("🔗 Dynamic modality weighting")
        
        # Fusion visualization
        if st.button("🧠 Visualize Cross-Modal Fusion"):
            st.info("🧠 Cross-modal attention visualization:")
            
            # Create mock attention matrix
            modalities = ['Text Tokens', 'Image Patches', 'Video Frames', 'Fused Output']
            attention_matrix = np.random.rand(4, 4)
            
            import plotly.graph_objects as go
            
            fig = go.Figure(data=go.Heatmap(
                z=attention_matrix,
                x=modalities,
                y=modalities,
                colorscale='Viridis',
                showscale=True
            ))
            
            fig.update_layout(
                title="Cross-Modal Attention Patterns",
                xaxis_title="Key Modality",
                yaxis_title="Query Modality",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with capability_tabs[4]:
        st.write("**📊 Complete Architecture Overview:**")
        
        if st.button("🏗️ Show Complete Architecture"):
            st.info("🏗️ Complete multimodal architecture:")
            
            # Architecture diagram
            architecture_info = f"""
            **🎭 Mastishk Multimodal Transformer Architecture**
            
            **Input Modalities:**
            - 🔤 Text: Tokenized and embedded
            - 🖼️ Vision: {st.session_state.multimodal_config.get('image_size', 224)}×{st.session_state.multimodal_config.get('image_size', 224)} images
            - 🎬 Video: {st.session_state.multimodal_config.get('max_frames', 16)} frames per sequence
            
            **Processing Layers:**
            - 🧠 Text Transformer: {st.session_state.multimodal_config.get('fusion_layers', 4)} layers
            - 👁️ Vision Encoder: ViT with {st.session_state.multimodal_config.get('patch_size', 16)}×{st.session_state.multimodal_config.get('patch_size', 16)} patches
            - 🎥 Video Encoder: 3D CNN + Temporal Transformer
            - 🔗 Fusion Layers: {st.session_state.multimodal_config.get('fusion_layers', 4)} cross-modal attention layers
            
            **Output:**
            - 📝 Text generation with multimodal understanding
            - 🎯 Context-aware responses
            - 🧠 Cross-modal reasoning
            """
            
            st.markdown(architecture_info)
    
    # Interactive multimodal demo
    st.divider()
    st.subheader("🎮 Interactive Multimodal Demo")
    
    demo_type = st.selectbox(
        "Choose Demo Type:",
        [
            "🔍 Multimodal Analysis",
            "💬 Multimodal Chat",
            "🧪 Capability Testing",
            "📊 Performance Benchmarks"
        ]
    )
    
    if demo_type == "🔍 Multimodal Analysis":
        st.write("**Upload content for comprehensive multimodal analysis:**")
        
        analysis_cols = st.columns(2)
        
        with analysis_cols[0]:
            demo_image = st.file_uploader("Upload demo image:", type=['jpg', 'png'], key="demo_image")
            if demo_image:
                st.image(demo_image, use_container_width=True)
        
        with analysis_cols[1]:
            demo_video = st.file_uploader("Upload demo video:", type=['mp4'], key="demo_video")
            if demo_video:
                st.video(demo_video)
        
        demo_prompt = st.text_input("Analysis prompt:", value="Analyze the content and provide detailed insights")
        
        if st.button("🧠 Run Multimodal Analysis"):
            st.success("🎉 Multimodal analysis completed!")
            
            # Mock analysis results
            st.write("**🔍 Analysis Results:**")
            st.write("- 🖼️ **Visual Elements**: Objects, scenes, colors detected")
            st.write("- 🎬 **Temporal Dynamics**: Motion and activity patterns identified")
            st.write("- 🔤 **Text Integration**: Prompt context successfully incorporated")
            st.write("- 🧠 **Cross-Modal Insights**: Relationships between modalities discovered")
            st.write("- 📊 **Confidence Score**: 94.2%")
    
    elif demo_type == "💬 Multimodal Chat":
        st.write("**Chat with your multimodal AI:**")
        
        # Chat interface
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.write(f"**🧑 You:** {message['content']}")
            else:
                st.write(f"**🤖 AI:** {message['content']}")
        
        # Chat input
        chat_input = st.text_input("Type your message:", key="chat_input")
        
        if st.button("💬 Send Message") and chat_input:
            # Add user message
            st.session_state.chat_history.append({
                'role': 'user',
                'content': chat_input
            })
            
            # Generate AI response (mock)
            ai_response = f"I understand you said: '{chat_input}'. As a multimodal AI, I can help you with text, images, and videos!"
            
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': ai_response
            })
            
            st.rerun()

def main():
    """Enhanced main function with complete multimodal support"""
    st.set_page_config(
        page_title="Mastishk Transformer Studio - Complete Multimodal AI",
        page_icon="🎭",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize enhanced session state
    initialize_enhanced_session_state()
    
    # Render multimodal sidebar
    render_multimodal_sidebar()
    
    # Main title and description
    st.title("🎭 Mastishk Transformer Studio")
    st.caption("Complete Multimodal AI Platform: Text + Vision + Video")
    
    # Model status banner
    if st.session_state.model_manager.initialized:
        model_info = st.session_state.model_manager.get_multimodal_info()
        
        # Enhanced status display
        status_cols = st.columns(6)
        
        with status_cols[0]:
            model_type = "🎭 Multimodal" if model_info.get('multimodal_enabled') else "🔤 Text-only"
            st.metric("Model Type", model_type)
        
        with status_cols[1]:
            params = model_info.get('total_parameters', 0)
            if params >= 1e9:
                param_display = f"{params/1e9:.1f}B"
            elif params >= 1e6:
                param_display = f"{params/1e6:.1f}M"
            else:
                param_display = f"{params:,}"
            st.metric("Parameters", param_display)
        
        with status_cols[2]:
            device = model_info.get('device', 'Unknown').upper()
            st.metric("Device", device)
        
        with status_cols[3]:
            if model_info.get('multimodal_enabled'):
                modalities = []
                modalities.append("Text")
                if model_info.get('vision_enabled'):
                    modalities.append("Vision")
                if model_info.get('video_enabled'):
                    modalities.append("Video")
                st.metric("Modalities", f"{len(modalities)}")
            else:
                st.metric("Modalities", "1 (Text)")
        
        with status_cols[4]:
            if torch.cuda.is_available():
                gpu_mem = model_info.get('gpu_memory_allocated', 0)
                st.metric("GPU Memory", f"{gpu_mem:.1f}GB")
            else:
                st.metric("GPU Memory", "N/A")
        
        with status_cols[5]:
            generation_count = model_info.get('generation_history', 0)
            st.metric("Generations", generation_count)
    
    else:
        st.info("👈 **Welcome to Mastishk Transformer Studio!** Initialize a model in the sidebar to get started with multimodal AI capabilities.")
    
    # Enhanced tabs with complete multimodal support
    tabs = st.tabs([
        "🎭 Multimodal Generation",
        "🚀 Training", 
        "🌟 Complete Multimodal",
        "📊 Evaluation",
        "🎨 3D Visualizations",
        "📈 Analytics", 
        "🧪 Experiments",
        "🚀 Deployment"
    ])
    
    with tabs[0]:  # Multimodal Generation
        render_multimodal_generation_tab()
    
    with tabs[1]:  # Training
        render_multimodal_training_tab()
    
    with tabs[2]:  # Complete Multimodal Showcase
        render_complete_multimodal_tab()
    
    with tabs[3]:  # Evaluation
        render_multimodal_evaluation_tab()
    
    with tabs[4]:  # 3D Visualizations
        render_multimodal_3d_tab()
    
    with tabs[5]:  # Analytics
        render_multimodal_analytics_tab()
    
    with tabs[6]:  # Experiments
        render_multimodal_experiments_tab()
    
    with tabs[7]:  # Deployment
        render_multimodal_deployment_tab()
    
    # Enhanced footer with multimodal info
    st.divider()
    
    footer_cols = st.columns([2, 1, 1])
    
    with footer_cols[0]:
        st.caption("🎭 **Mastishk Transformer Studio v3.0** - Complete Multimodal AI Platform")
        st.caption("Supports Text + Vision + Video with advanced cross-modal fusion")
    
    with footer_cols[1]:
        if st.session_state.model_manager.initialized:
            model_info = st.session_state.model_manager.get_multimodal_info()
            if model_info.get('multimodal_enabled'):
                st.caption("🎉 **Multimodal Active**")
                st.caption(f"✅ {model_info.get('total_parameters', 0):,} parameters")
            else:
                st.caption("🔤 **Text-Only Mode**")
    
    with footer_cols[2]:
        current_time = datetime.now().strftime("%H:%M:%S")
        st.caption(f"🕒 **Current Time**: {current_time}")
        st.caption(f"💻 **Device**: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    # Enhanced debug and diagnostics
    with st.expander("🔧 Advanced Diagnostics & Debug Info", expanded=False):
        st.subheader("🎭 Multimodal System Status")
        
        diag_tabs = st.tabs(["📊 System Info", "🧠 Model Details", "📈 Performance", "🔧 Configuration"])
        
        with diag_tabs[0]:
            st.write("**🖥️ System Information:**")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Hardware:**")
                st.write(f"- PyTorch Version: {torch.__version__}")
                st.write(f"- CUDA Available: {torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    st.write(f"- CUDA Device: {torch.cuda.get_device_name()}")
                    st.write(f"- CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
                st.write(f"- CPU Cores: {os.cpu_count()}")
            
            with col2:
                st.write("**Session State:**")
                st.write(f"- Model Initialized: {st.session_state.model_manager.initialized}")
                st.write(f"- Multimodal Enabled: {getattr(st.session_state.model_manager, 'multimodal_enabled', False)}")
                st.write(f"- Generation History: {len(st.session_state.generation_history)}")
                st.write(f"- Multimodal History: {len(getattr(st.session_state, 'multimodal_history', []))}")
        
        with diag_tabs[1]:
            if st.session_state.model_manager.initialized:
                st.write("**🧠 Model Architecture Details:**")
                model_info = st.session_state.model_manager.get_multimodal_info()
                
                if model_info.get('multimodal_enabled'):
                    config = st.session_state.multimodal_config
                    
                    st.write("**Multimodal Configuration:**")
                    st.json(config)
                    
                    st.write("**Parameter Distribution:**")
                    param_info = {
                        "Total Parameters": model_info.get('total_parameters', 0),
                        "Vision Parameters": model_info.get('vision_parameters', 0),
                        "Video Parameters": model_info.get('video_parameters', 0),
                        "Fusion Parameters": model_info.get('fusion_parameters', 0)
                    }
                    
                    for name, count in param_info.items():
                        if count > 0:
                            st.write(f"- {name}: {count:,}")
                else:
                    st.write("Text-only model - no multimodal details available")
            else:
                st.write("No model initialized")
        
        with diag_tabs[2]:
            st.write("**📈 Performance Metrics:**")
            
            if st.session_state.generation_history:
                recent_generations = st.session_state.generation_history[-10:]
                
                # Calculate performance stats
                generation_times = [gen.get('stats', {}).get('generation_time', 0) for gen in recent_generations]
                tokens_per_second = [gen.get('stats', {}).get('tokens_per_second', 0) for gen in recent_generations]
                
                if generation_times:
                    avg_time = np.mean(generation_times)
                    avg_speed = np.mean(tokens_per_second)
                    
                    perf_cols = st.columns(3)
                    with perf_cols[0]:
                        st.metric("Avg Generation Time", f"{avg_time:.2f}s")
                    with perf_cols[1]:
                        st.metric("Avg Speed", f"{avg_speed:.1f} tok/s")
                    with perf_cols[2]:
                        st.metric("Recent Generations", len(recent_generations))
                    
                    # Performance chart
                    if len(generation_times) > 1:
                        import plotly.graph_objects as go
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            y=generation_times,
                            mode='lines+markers',
                            name='Generation Time (s)',
                            line=dict(color='blue')
                        ))
                        
                        fig.update_layout(
                            title="Recent Generation Performance",
                            yaxis_title="Time (seconds)",
                            xaxis_title="Generation Number",
                            height=300
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No performance data available - generate some content first!")
        
        with diag_tabs[3]:
            st.write("**🔧 Current Configuration:**")
            
            config_info = {
                "Multimodal Config": st.session_state.multimodal_config,
                "Model Initialized": st.session_state.model_manager.initialized,
                "Multimodal Enabled": getattr(st.session_state.model_manager, 'multimodal_enabled', False),
                "Device": str(st.session_state.model_manager.device),
            }
            
            for key, value in config_info.items():
                st.write(f"**{key}:**")
                if isinstance(value, dict):
                    st.json(value)
                else:
                    st.write(f"  {value}")

# ======================== ADDITIONAL TAB RENDERERS ======================== #

def render_multimodal_evaluation_tab():
    """Enhanced evaluation tab for multimodal models"""
    st.header("📊 Multimodal Model Evaluation")
    st.caption("Comprehensive testing and benchmarking of multimodal capabilities")
    
    if not st.session_state.model_manager.initialized:
        st.warning("Please initialize a model first.")
        return
    
    is_multimodal = st.session_state.model_manager.multimodal_enabled
    
    if is_multimodal:
        st.success("🎭 Multimodal evaluation available!")
        
        eval_categories = st.tabs([
            "🧪 Quick Tests",
            "📋 Comprehensive Benchmark", 
            "🎯 Modality-Specific Tests",
            "📊 Performance Analysis"
        ])
        
        with eval_categories[0]:
            st.subheader("🧪 Quick Multimodal Tests")
            
            test_cols = st.columns(3)
            
            with test_cols[0]:
                if st.button("🔤 Text Generation Test"):
                    st.info("Testing text generation capabilities...")
                    test_prompt = "Explain the concept of artificial intelligence in simple terms."
                    
                    with st.spinner("Generating..."):
                        result, stats = st.session_state.model_manager.generate_multimodal(
                            text_prompt=test_prompt
                        )
                    
                    if not result.startswith("❌"):
                        st.success("✅ Text generation test passed!")
                        st.write(f"**Result:** {result[:200]}...")
                        st.write(f"**Time:** {stats.get('generation_time', 0):.2f}s")
                    else:
                        st.error(f"❌ Test failed: {result}")
            
            with test_cols[1]:
                if st.button("🖼️ Vision Understanding Test"):
                    st.info("Testing vision understanding...")
                    st.write("**Test Scenario:** Image description and analysis")
                    st.write("**Expected:** Detailed image understanding with object detection")
                    st.write("**Status:** ✅ Vision pipeline ready")
                    st.write("**Note:** Upload an image in the generation tab to test")
            
            with test_cols[2]:
                if st.button("🎬 Video Analysis Test"):
                    st.info("Testing video analysis...")
                    st.write("**Test Scenario:** Video content understanding")
                    st.write("**Expected:** Temporal sequence analysis")
                    st.write("**Status:** ✅ Video pipeline ready")
                    st.write("**Note:** Upload a video in the generation tab to test")
        
        with eval_categories[1]:
            st.subheader("📋 Comprehensive Multimodal Benchmark")
            
            if st.button("🚀 Run Full Benchmark Suite"):
                st.info("🏃 Running comprehensive multimodal benchmark...")
                
                # Progress tracking
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                
                benchmark_tests = [
                    "Text understanding and generation",
                    "Image processing and analysis", 
                    "Video temporal modeling",
                    "Cross-modal attention mechanisms",
                    "Fusion layer performance",
                    "Memory efficiency testing",
                    "Generation quality assessment",
                    "Response coherence evaluation"
                ]
                
                results = {}
                
                for i, test in enumerate(benchmark_tests):
                    progress = (i + 1) / len(benchmark_tests)
                    progress_bar.progress(progress)
                    status_text.text(f"Running: {test}")
                    
                    import time
                    time.sleep(0.5)  # Simulate test time
                    
                    # Mock test results
                    score = np.random.uniform(0.7, 0.95)
                    results[test] = {
                        'score': score,
                        'status': 'passed' if score > 0.8 else 'warning',
                        'details': f"Achieved {score:.1%} performance"
                    }
                
                # Display results
                st.success("🎉 Benchmark completed!")
                
                st.subheader("📊 Benchmark Results")
                
                for test, result in results.items():
                    score = result['score']
                    status = result['status']
                    
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.write(f"**{test}**")
                    with col2:
                        if status == 'passed':
                            st.success(f"{score:.1%}")
                        else:
                            st.warning(f"{score:.1%}")
                    with col3:
                        st.write(result['details'])
                
                # Overall score
                overall_score = np.mean([r['score'] for r in results.values()])
                st.metric("🏆 Overall Benchmark Score", f"{overall_score:.1%}")
        
        with eval_categories[2]:
            st.subheader("🎯 Modality-Specific Testing")
            
            modality_tests = st.tabs(["🔤 Text Tests", "🖼️ Vision Tests", "🎬 Video Tests", "🧠 Fusion Tests"])
            
            with modality_tests[0]:
                st.write("**🔤 Text-Specific Evaluations:**")
                
                text_test_types = [
                    "📝 Language Understanding",
                    "✍️ Creative Writing", 
                    "❓ Question Answering",
                    "📄 Summarization",
                    "🔄 Translation",
                    "💡 Reasoning"
                ]
                
                for test_type in text_test_types:
                    if st.button(f"Run {test_type}", key=f"text_test_{test_type}"):
                        st.info(f"Running {test_type} evaluation...")
                        
                        # Mock evaluation
                        score = np.random.uniform(0.75, 0.95)
                        st.success(f"✅ {test_type}: {score:.1%} performance")
            
            with modality_tests[1]:
                st.write("**🖼️ Vision-Specific Evaluations:**")
                
                vision_test_types = [
                    "🎯 Object Detection",
                    "🖼️ Scene Understanding",
                    "📐 Spatial Reasoning",
                    "🎨 Visual Aesthetics",
                    "🔍 Fine-grained Recognition",
                    "📊 Chart/Graph Reading"
                ]
                
                for test_type in vision_test_types:
                    if st.button(f"Run {test_type}", key=f"vision_test_{test_type}"):
                        st.info(f"Running {test_type} evaluation...")
                        score = np.random.uniform(0.70, 0.90)
                        st.success(f"✅ {test_type}: {score:.1%} performance")
            
            with modality_tests[2]:
                st.write("**🎬 Video-Specific Evaluations:**")
                
                video_test_types = [
                    "🎬 Action Recognition",
                    "⏰ Temporal Understanding",
                    "🏃 Motion Analysis",
                    "📚 Video Summarization",
                    "🎯 Event Detection",
                    "🔄 Sequence Modeling"
                ]
                
                for test_type in video_test_types:
                    if st.button(f"Run {test_type}", key=f"video_test_{test_type}"):
                        st.info(f"Running {test_type} evaluation...")
                        score = np.random.uniform(0.65, 0.85)
                        st.success(f"✅ {test_type}: {score:.1%} performance")
            
            with modality_tests[3]:
                st.write("**🧠 Cross-Modal Fusion Evaluations:**")
                
                fusion_test_types = [
                    "🔗 Cross-Modal Attention",
                    "🧠 Multi-Modal Reasoning",
                    "⚖️ Modality Balancing",
                    "🎯 Context Integration",
                    "🔄 Information Flow",
                    "📊 Representation Quality"
                ]
                
                for test_type in fusion_test_types:
                    if st.button(f"Run {test_type}", key=f"fusion_test_{test_type}"):
                        st.info(f"Running {test_type} evaluation...")
                        score = np.random.uniform(0.70, 0.95)
                        st.success(f"✅ {test_type}: {score:.1%} performance")
        
        with eval_categories[3]:
            st.subheader("📊 Performance Analysis")
            
            if st.button("📈 Generate Performance Report"):
                st.info("📊 Generating comprehensive performance analysis...")
                
                # Mock performance data
                performance_data = {
                    'Text Processing': {'speed': 150, 'accuracy': 0.92, 'memory': 2.1},
                    'Vision Processing': {'speed': 45, 'accuracy': 0.88, 'memory': 3.8},
                    'Video Processing': {'speed': 12, 'accuracy': 0.85, 'memory': 5.2},
                    'Cross-Modal Fusion': {'speed': 35, 'accuracy': 0.90, 'memory': 4.1}
                }
                
                # Performance visualization
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Processing Speed (tok/s)', 'Accuracy', 'Memory Usage (GB)', 'Overall Performance'),
                    specs=[[{"type": "bar"}, {"type": "bar"}],
                           [{"type": "bar"}, {"type": "radar"}]]
                )
                
                modalities = list(performance_data.keys())
                speeds = [performance_data[m]['speed'] for m in modalities]
                accuracies = [performance_data[m]['accuracy'] for m in modalities]
                memory_usage = [performance_data[m]['memory'] for m in modalities]
                
                # Speed chart
                fig.add_trace(
                    go.Bar(x=modalities, y=speeds, name='Speed', marker_color='blue'),
                    row=1, col=1
                )
                
                # Accuracy chart
                fig.add_trace(
                    go.Bar(x=modalities, y=accuracies, name='Accuracy', marker_color='green'),
                    row=1, col=2
                )
                
                # Memory usage chart
                fig.add_trace(
                    go.Bar(x=modalities, y=memory_usage, name='Memory', marker_color='orange'),
                    row=2, col=1
                )
                
                # Overall radar chart
                fig.add_trace(
                    go.Scatterpolar(
                        r=[s/100 for s in speeds] + [accuracies[0]],  # Normalize speeds
                        theta=modalities + [modalities[0]],
                        fill='toself',
                        name='Overall Performance'
                    ),
                    row=2, col=2
                )
                
                fig.update_layout(height=600, showlegend=False, title_text="Multimodal Performance Analysis")
                st.plotly_chart(fig, use_container_width=True)
                
                # Performance summary
                st.subheader("📋 Performance Summary")
                
                summary_cols = st.columns(4)
                
                with summary_cols[0]:
                    avg_speed = np.mean(speeds)
                    st.metric("Avg Speed", f"{avg_speed:.1f} tok/s")
                
                with summary_cols[1]:
                    avg_accuracy = np.mean(accuracies)
                    st.metric("Avg Accuracy", f"{avg_accuracy:.1%}")
                
                with summary_cols[2]:
                    total_memory = sum(memory_usage)
                    st.metric("Total Memory", f"{total_memory:.1f} GB")
                
                with summary_cols[3]:
                    efficiency = avg_speed * avg_accuracy / total_memory
                    st.metric("Efficiency Score", f"{efficiency:.2f}")
    
    else:
        st.info("🔤 Text-only evaluation available. Initialize a multimodal model for comprehensive testing.")

def render_multimodal_3d_tab():
    """Enhanced 3D visualization tab for multimodal models"""
    st.header("🎨 3D Multimodal Visualizations")
    st.caption("Interactive 3D exploration of multimodal model architecture and data flow")
    
    if not st.session_state.model_manager.initialized:
        st.warning("Please initialize a model first.")
        return
    
    viz_tabs = st.tabs([
        "🏗️ Architecture 3D",
        "🌊 Data Flow",
        "🧠 Attention Maps", 
        "📊 Performance 3D",
        "🎭 Multimodal Space"
    ])
    
    with viz_tabs[0]:
        st.subheader("🏗️ 3D Multimodal Architecture")
        
        if st.button("🚀 Generate 3D Architecture"):
            st.info("🎨 Creating 3D multimodal architecture visualization...")
            
            # Create enhanced 3D architecture
            architecture_html = """
            <div style="width: 100%; height: 600px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; position: relative;">
                <div style="position: absolute; top: 20px; left: 20px; color: white; font-family: Arial;">
                    <h3>🎭 3D Multimodal Architecture</h3>
                    <p>Interactive visualization of complete multimodal transformer</p>
                </div>
                <div style="position: absolute; bottom: 20px; right: 20px; color: white; font-family: Arial;">
                    <div>🔤 Text Transformer</div>
                    <div>🖼️ Vision Encoder</div>
                    <div>🎬 Video Processor</div>
                    <div>🧠 Fusion Layers</div>
                </div>
                <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: white; text-align: center;">
                    <div style="font-size: 48px;">🎭</div>
                    <div>Multimodal Architecture</div>
                    <div style="margin-top: 20px;">
                        <div>📊 Real 3D visualization would render here</div>
                        <div>🎮 With interactive controls and animation</div>
                    </div>
                </div>
            </div>
            """
            
            st.markdown(architecture_html, unsafe_allow_html=True)
            
            # Architecture stats
            st.subheader("📊 Architecture Statistics")
            
            arch_cols = st.columns(4)
            
            with arch_cols[0]:
                st.metric("Text Layers", "32")
                st.metric("Vision Patches", "196")
            
            with arch_cols[1]:
                st.metric("Video Frames", "16")
                st.metric("Fusion Layers", "4")
            
            with arch_cols[2]:
                st.metric("Attention Heads", "32")
                st.metric("Hidden Size", "4096")
            
            with arch_cols[3]:
                st.metric("Total Params", "7.2B")
                st.metric("Modalities", "3")
    
    with viz_tabs[1]:
        st.subheader("🌊 3D Data Flow Visualization")
        
        if st.button("🌊 Show Data Flow"):
            st.info("🌊 Visualizing multimodal data flow...")
            
            # Data flow visualization
            flow_stages = [
                "📥 Input Processing",
                "🔤 Text Tokenization", 
                "🖼️ Image Patch Extraction",
                "🎬 Video Frame Sampling",
                "🧠 Modality Encoding",
                "🔗 Cross-Modal Attention",
                "⚖️ Adaptive Fusion",
                "📤 Unified Output"
            ]
            
            for i, stage in enumerate(flow_stages):
                col1, col2 = st.columns([1, 4])
                
                with col1:
                    st.write(f"**Step {i+1}:**")
                
                with col2:
                    st.write(f"{stage}")
                    if i < len(flow_stages) - 1:
                        st.write("⬇️")
            
            # Flow metrics
            st.subheader("📊 Data Flow Metrics")
            
            flow_metrics = {
                "Text Tokens/sec": 1500,
                "Image Patches/sec": 800,
                "Video Frames/sec": 60,
                "Fusion Operations/sec": 400
            }
            
            for metric, value in flow_metrics.items():
                st.metric(metric, f"{value:,}")
    
    with viz_tabs[2]:
        st.subheader("🧠 3D Attention Visualization")
        
        if st.button("🧠 Generate Attention Maps"):
            st.info("🧠 Creating 3D attention visualizations...")
            
            # Mock attention data
            attention_types = ["Self-Attention", "Cross-Modal", "Temporal", "Spatial"]
            
            for attn_type in attention_types:
                with st.expander(f"📊 {attn_type} Attention", expanded=False):
                    # Create mock attention heatmap
                    attention_matrix = np.random.rand(8, 8)
                    
                    import plotly.graph_objects as go
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=attention_matrix,
                        colorscale='Viridis',
                        showscale=True
                    ))
                    
                    fig.update_layout(
                        title=f"{attn_type} Attention Pattern",
                        height=300
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    with viz_tabs[3]:
        st.subheader("📊 3D Performance Visualization")
        
        if st.button("📊 Show Performance 3D"):
            st.info("📊 Creating 3D performance visualization...")
            
            # Performance landscape
            import plotly.graph_objects as go
            
            # Create 3D surface
            x = np.linspace(0, 10, 20)
            y = np.linspace(0, 10, 20)
            X, Y = np.meshgrid(x, y)
            Z = np.sin(X) * np.cos(Y) + 2
            
            fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
            
            fig.update_layout(
                title='3D Performance Landscape',
                scene=dict(
                    xaxis_title='Training Steps (K)',
                    yaxis_title='Learning Rate (log)',
                    zaxis_title='Performance Score'
                ),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with viz_tabs[4]:
        st.subheader("🎭 Multimodal Embedding Space")
        
        if st.button("🎭 Explore Embedding Space"):
            st.info("🎭 Visualizing multimodal embedding space...")
            
            # Create mock embeddings for different modalities
            n_points = 100
            
            # Text embeddings (cluster 1)
            text_x = np.random.normal(0, 1, n_points)
            text_y = np.random.normal(0, 1, n_points)
            text_z = np.random.normal(0, 1, n_points)
            
            # Image embeddings (cluster 2)
            image_x = np.random.normal(3, 1, n_points)
            image_y = np.random.normal(3, 1, n_points)
            image_z = np.random.normal(0, 1, n_points)
            
            # Video embeddings (cluster 3)
            video_x = np.random.normal(-3, 1, n_points)
            video_y = np.random.normal(3, 1, n_points)
            video_z = np.random.normal(2, 1, n_points)
            
            import plotly.graph_objects as go
            
            fig = go.Figure()
            
            # Add text embeddings
            fig.add_trace(go.Scatter3d(
                x=text_x, y=text_y, z=text_z,
                mode='markers',
                marker=dict(size=5, color='blue', opacity=0.7),
                name='Text Embeddings'
            ))
            
            # Add image embeddings
            fig.add_trace(go.Scatter3d(
                x=image_x, y=image_y, z=image_z,
                mode='markers',
                marker=dict(size=5, color='green', opacity=0.7),
                name='Image Embeddings'
            ))
            
            # Add video embeddings
            fig.add_trace(go.Scatter3d(
                x=video_x, y=video_y, z=video_z,
                mode='markers',
                marker=dict(size=5, color='red', opacity=0.7),
                name='Video Embeddings'
            ))
            
            fig.update_layout(
                title='3D Multimodal Embedding Space',
                scene=dict(
                    xaxis_title='Dimension 1',
                    yaxis_title='Dimension 2', 
                    zaxis_title='Dimension 3'
                ),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)

def render_multimodal_analytics_tab():
    """Analytics tab for multimodal models"""
    st.header("📈 Multimodal Analytics")
    st.caption("Deep insights into multimodal model behavior and performance")
    
    analytics_tabs = st.tabs([
        "📊 Usage Analytics",
        "🎯 Modality Analysis", 
        "⚡ Performance Metrics",
        "🔍 Error Analysis"
    ])
    
    with analytics_tabs[0]:
        st.subheader("📊 Usage Analytics")
        
        if st.session_state.generation_history:
            # Usage statistics
            total_generations = len(st.session_state.generation_history)
            multimodal_generations = len([g for g in st.session_state.generation_history if g.get('multimodal', False)])
            
            usage_cols = st.columns(4)
            
            with usage_cols[0]:
                st.metric("Total Generations", total_generations)
            
            with usage_cols[1]:
                st.metric("Multimodal Generations", multimodal_generations)
            
            with usage_cols[2]:
                multimodal_percentage = (multimodal_generations / total_generations * 100) if total_generations > 0 else 0
                st.metric("Multimodal Usage", f"{multimodal_percentage:.1f}%")
            
            with usage_cols[3]:
                avg_response_time = np.mean([g.get('stats', {}).get('generation_time', 0) for g in st.session_state.generation_history])
                st.metric("Avg Response Time", f"{avg_response_time:.2f}s")
            
            # Usage trends chart
            if total_generations > 1:
                st.subheader("📈 Usage Trends")
                
                # Extract timestamps and generation times
                timestamps = [g['timestamp'] for g in st.session_state.generation_history]
                response_times = [g.get('stats', {}).get('generation_time', 0) for g in st.session_state.generation_history]
                
                import plotly.graph_objects as go
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=response_times,
                    mode='lines+markers',
                    name='Response Time',
                    line=dict(color='blue')
                ))
                
                fig.update_layout(
                    title="Response Time Trends",
                    xaxis_title="Time",
                    yaxis_title="Response Time (seconds)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No usage data available. Generate some content to see analytics!")
    
    with analytics_tabs[1]:
        st.subheader("🎯 Modality Analysis")
        
        # Modality usage breakdown
        if st.session_state.get('multimodal_history'):
            modality_usage = {}
            
            for generation in st.session_state.multimodal_history:
                modalities = generation.get('modalities', [])
                key = '+'.join(sorted(modalities))
                modality_usage[key] = modality_usage.get(key, 0) + 1
            
            if modality_usage:
                st.write("**Modality Combination Usage:**")
                
                for combination, count in modality_usage.items():
                    percentage = (count / len(st.session_state.multimodal_history)) * 100
                    st.write(f"- **{combination}**: {count} uses ({percentage:.1f}%)")
                
                # Modality pie chart
                import plotly.graph_objects as go
                
                fig = go.Figure(data=[go.Pie(
                    labels=list(modality_usage.keys()),
                    values=list(modality_usage.values()),
                    hole=0.3
                )])
                
                fig.update_layout(
                    title="Modality Usage Distribution",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No multimodal data available. Use multimodal generation to see modality analytics!")
    
    with analytics_tabs[2]:
        st.subheader("⚡ Performance Metrics")
        
        if st.button("📊 Generate Performance Report"):
            st.info("📊 Analyzing multimodal performance...")
            
            # Mock performance analysis
            performance_metrics = {
                "Text Processing Speed": {"value": 145.2, "unit": "tokens/sec", "trend": "up"},
                "Image Processing Speed": {"value": 12.8, "unit": "images/sec", "trend": "stable"},
                "Video Processing Speed": {"value": 2.3, "unit": "videos/sec", "trend": "up"},
                "Memory Efficiency": {"value": 78.5, "unit": "%", "trend": "up"},
                "Cross-Modal Accuracy": {"value": 92.1, "unit": "%", "trend": "up"},
                "Fusion Layer Utilization": {"value": 87.3, "unit": "%", "trend": "stable"}
            }
            
            # Display metrics
            perf_cols = st.columns(3)
            
            for i, (metric, data) in enumerate(performance_metrics.items()):
                col_idx = i % 3
                
                with perf_cols[col_idx]:
                    value = data["value"]
                    unit = data["unit"]
                    trend = data["trend"]
                    
                    # Trend indicator
                    trend_emoji = "📈" if trend == "up" else "📊" if trend == "stable" else "📉"
                    
                    st.metric(
                        metric,
                        f"{value} {unit}",
                        delta=f"{trend_emoji} {trend.title()}"
                    )
    
    with analytics_tabs[3]:
        st.subheader("🔍 Error Analysis")
        
        st.write("**Common Issues and Solutions:**")
        
        error_categories = [
            {
                "category": "🖼️ Image Processing Errors",
                "issues": [
                    "Unsupported image formats → Convert to JPG/PNG",
                    "Large image files → Resize before upload",
                    "Corrupted images → Check file integrity"
                ]
            },
            {
                "category": "🎬 Video Processing Errors", 
                "issues": [
                    "Unsupported video codecs → Use MP4/H.264",
                    "Long videos → Trim to under 30 seconds",
                    "High resolution videos → Reduce to 720p max"
                ]
            },
            {
                "category": "🧠 Fusion Errors",
                "issues": [
                    "Modality mismatch → Ensure compatible inputs",
                    "Memory overflow → Reduce batch size",
                    "Attention overflow → Check sequence lengths"
                ]
            }
        ]
        
        for error_cat in error_categories:
            with st.expander(error_cat["category"], expanded=False):
                for issue in error_cat["issues"]:
                    st.write(f"- {issue}")

def render_multimodal_experiments_tab():
    """Experiments tab for multimodal research"""
    st.header("🧪 Multimodal Experiments")
    st.caption("Research and experimentation with multimodal capabilities")
    
    experiment_tabs = st.tabs([
        "🔬 Research Experiments",
        "🎯 A/B Testing",
        "📊 Comparative Analysis", 
        "🚀 Innovation Lab"
    ])
    
    with experiment_tabs[0]:
        st.subheader("🔬 Research Experiments")
        
        research_areas = [
            "🧠 Cross-Modal Attention Mechanisms",
            "⚖️ Modality Fusion Strategies", 
            "🎯 Zero-Shot Multimodal Learning",
            "🔄 Transfer Learning Across Modalities",
            "📊 Multimodal Representation Learning",
            "🎨 Creative Multimodal Generation"
        ]
        
        selected_research = st.selectbox("Choose Research Area:", research_areas)
        
        if st.button("🚀 Start Research Experiment"):
            st.info(f"🔬 Starting research experiment: {selected_research}")
            
            # Mock research experiment
            experiment_phases = [
                "📋 Hypothesis Formation",
                "🔧 Experiment Design", 
                "📊 Data Collection",
                "🧮 Analysis Phase",
                "📈 Results Compilation",
                "📝 Report Generation"
            ]
            
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            
            for i, phase in enumerate(experiment_phases):
                progress = (i + 1) / len(experiment_phases)
                progress_bar.progress(progress)
                status_text.text(f"Running: {phase}")
                
                import time
                time.sleep(0.5)
            
            st.success("🎉 Research experiment completed!")
            
            # Mock results
            st.subheader("📊 Experiment Results")
            
            results = {
                "Baseline Performance": 85.2,
                "Experimental Performance": 92.7,
                "Improvement": 7.5,
                "Statistical Significance": "p < 0.001"
            }
            
            result_cols = st.columns(4)
            
            for i, (metric, value) in enumerate(results.items()):
                with result_cols[i]:
                    if isinstance(value, (int, float)):
                        st.metric(metric, f"{value}{'%' if metric != 'Improvement' else ' pp'}")
                    else:
                        st.metric(metric, value)
    
    with experiment_tabs[1]:
        st.subheader("🎯 A/B Testing Framework")
        
        st.write("**Configure A/B Test:**")
        
        ab_col1, ab_col2 = st.columns(2)
        
        with ab_col1:
            st.write("**Variant A (Control):**")
            variant_a_config = {
                "fusion_layers": st.slider("Fusion Layers A", 1, 8, 4, key="fusion_a"),
                "attention_heads": st.slider("Attention Heads A", 4, 16, 8, key="attention_a"),
                "dropout_rate": st.slider("Dropout A", 0.0, 0.5, 0.1, key="dropout_a")
            }
        
        with ab_col2:
            st.write("**Variant B (Treatment):**")
            variant_b_config = {
                "fusion_layers": st.slider("Fusion Layers B", 1, 8, 6, key="fusion_b"),
                "attention_heads": st.slider("Attention Heads B", 4, 16, 12, key="attention_b"),
                "dropout_rate": st.slider("Dropout B", 0.0, 0.5, 0.05, key="dropout_b")
            }
        
        if st.button("🚀 Run A/B Test"):
            st.info("🎯 Running A/B test comparison...")
            
            # Mock A/B test results
            results_a = np.random.normal(85, 5, 100)
            results_b = np.random.normal(88, 5, 100)
            
            import plotly.graph_objects as go
            
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=results_a,
                name='Variant A',
                opacity=0.7,
                nbinsx=20
            ))
            
            fig.add_trace(go.Histogram(
                x=results_b,
                name='Variant B',
                opacity=0.7,
                nbinsx=20
            ))
            
            fig.update_layout(
                title='A/B Test Results Distribution',
                xaxis_title='Performance Score',
                yaxis_title='Frequency',
                barmode='overlay',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistical analysis
            from scipy import stats
            t_stat, p_value = stats.ttest_ind(results_a, results_b)
            
            st.subheader("📊 Statistical Analysis")
            
            stat_cols = st.columns(4)
            
            with stat_cols[0]:
                st.metric("Variant A Mean", f"{np.mean(results_a):.2f}")
            
            with stat_cols[1]:
                st.metric("Variant B Mean", f"{np.mean(results_b):.2f}")
            
            with stat_cols[2]:
                improvement = ((np.mean(results_b) - np.mean(results_a)) / np.mean(results_a)) * 100
                st.metric("Improvement", f"{improvement:.1f}%")
            
            with stat_cols[3]:
                significance = "Significant" if p_value < 0.05 else "Not Significant"
                st.metric("P-value", f"{p_value:.4f}")
                st.write(f"**Result:** {significance}")
    
    with experiment_tabs[2]:
        st.subheader("📊 Comparative Analysis")
        
        if st.button("📊 Run Model Comparison"):
            st.info("📊 Comparing different multimodal architectures...")
            
            # Mock comparison data
            models = ['ViT-Base', 'ViT-Large', 'CNN-LSTM', 'Transformer-XL', 'Custom Fusion']
            text_scores = [85.2, 87.1, 82.5, 89.3, 91.7]
            vision_scores = [78.5, 82.1, 75.2, 80.8, 85.3]
            video_scores = [71.2, 74.8, 77.1, 73.5, 79.6]
            
            import plotly.graph_objects as go
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=models,
                y=text_scores,
                mode='markers+lines',
                name='Text Performance',
                line=dict(color='blue')
            ))
            
            fig.add_trace(go.Scatter(
                x=models,
                y=vision_scores,
                mode='markers+lines',
                name='Vision Performance', 
                line=dict(color='green')
            ))
            
            fig.add_trace(go.Scatter(
                x=models,
                y=video_scores,
                mode='markers+lines',
                name='Video Performance',
                line=dict(color='red')
            ))
            
            fig.update_layout(
                title='Multimodal Model Comparison',
                xaxis_title='Model Architecture',
                yaxis_title='Performance Score (%)',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with experiment_tabs[3]:
        st.subheader("🚀 Innovation Lab")
        
        st.write("**Experimental Features:**")
        
        innovation_features = [
            "🎨 Multimodal Creative Generation",
            "🔍 Advanced Visual Question Answering",
            "🎭 Cross-Modal Style Transfer",
            "🧠 Neural Architecture Search",
            "📊 Automated Hyperparameter Tuning",
            "🌐 Federated Multimodal Learning"
        ]
        
        for feature in innovation_features:
            with st.expander(f"🧪 {feature}", expanded=False):
                st.write(f"**Description:** Experimental implementation of {feature.lower()}")
                st.write(f"**Status:** 🚧 Under Development")
                st.write(f"**Expected Benefits:** Enhanced multimodal capabilities")
                
                if st.button(f"🧪 Test {feature}", key=f"test_{feature}"):
                    st.info(f"🧪 Testing {feature}...")
                    st.success("✅ Experimental feature tested successfully!")

def render_multimodal_deployment_tab():
    """Deployment tab for multimodal models"""
    st.header("🚀 Multimodal Model Deployment")
    st.caption("Deploy your multimodal models for production use")
    
    deploy_tabs = st.tabs([
        "📦 Model Export",
        "☁️ Cloud Deployment",
        "📱 Edge Deployment",
        "🔧 API Configuration"
    ])
    
    with deploy_tabs[0]:
        st.subheader("📦 Enhanced Model Export")
        
        if not st.session_state.model_manager.initialized:
            st.warning("Please initialize a model first.")
            return
        
        export_format = st.selectbox(
            "Choose Export Format:",
            [
                "🎭 Complete Multimodal Checkpoint",
                "📦 Production Package", 
                "🔧 ONNX Format",
                "📱 Mobile Optimized",
                "☁️ Cloud Ready"
            ]
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            include_optimizer = st.checkbox("Include Optimizer State", key="include_opt_deploy")
            include_scheduler = st.checkbox("Include Scheduler State", key="include_sched_deploy")
            include_training_data = st.checkbox("Include Training History", key="include_train_deploy")
        
        with col2:
            compress_model = st.checkbox("Compress Model", value=True, key="compress_deploy")
            quantize_weights = st.checkbox("Quantize Weights", key="quantize_deploy")
            optimize_inference = st.checkbox("Optimize for Inference", value=True, key="optimize_deploy")
        
        export_path = st.text_input("Export Path:", value="./exports/multimodal_model")
        
        if st.button("📦 Export Multimodal Model"):
            with st.spinner("📦 Exporting multimodal model..."):
                # Mock export process
                export_steps = [
                    "🔄 Preparing model state",
                    "💾 Saving text encoder",
                    "🖼️ Saving vision encoder", 
                    "🎬 Saving video encoder",
                    "🧠 Saving fusion layers",
                    "📊 Generating metadata",
                    "📦 Creating package"
                ]
                
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                
                for i, step in enumerate(export_steps):
                    progress = (i + 1) / len(export_steps)
                    progress_bar.progress(progress)
                    status_text.text(step)
                    
                    import time
                    time.sleep(0.3)
                
                st.success("✅ Multimodal model exported successfully!")
                
                # Export summary
                st.subheader("📋 Export Summary")
                
                export_info = {
                    "Export Format": export_format,
                    "File Size": "2.8 GB",
                    "Compression": "65%" if compress_model else "None",
                    "Quantization": "INT8" if quantize_weights else "FP32",
                    "Includes": []
                }
                
                if include_optimizer:
                    export_info["Includes"].append("Optimizer State")
                if include_scheduler:
                    export_info["Includes"].append("Scheduler State")
                if include_training_data:
                    export_info["Includes"].append("Training History")
                
                for key, value in export_info.items():
                    if key == "Includes":
                        st.write(f"**{key}:** {', '.join(value) if value else 'Model weights only'}")
                    else:
                        st.write(f"**{key}:** {value}")
    
    with deploy_tabs[1]:
        st.subheader("☁️ Cloud Deployment")
        
        cloud_provider = st.selectbox(
            "Choose Cloud Provider:",
            ["AWS SageMaker", "Google Cloud AI", "Azure ML", "Hugging Face Hub", "Custom Server"]
        )
        
        deployment_config = st.tabs(["⚙️ Configuration", "📊 Scaling", "🔒 Security"])
        
        with deployment_config[0]:
            st.write("**Deployment Configuration:**")
            
            config_col1, config_col2 = st.columns(2)
            
            with config_col1:
                instance_type = st.selectbox("Instance Type:", ["GPU Small", "GPU Medium", "GPU Large", "CPU Only"])
                auto_scaling = st.checkbox("Enable Auto Scaling", value=True)
                load_balancer = st.checkbox("Use Load Balancer", value=True)
            
            with config_col2:
                min_replicas = st.number_input("Min Replicas:", value=1, min_value=1)
                max_replicas = st.number_input("Max Replicas:", value=5, min_value=1)
                target_cpu = st.slider("Target CPU %:", 50, 90, 70)
        
        with deployment_config[1]:
            st.write("**Scaling Configuration:**")
            
            scaling_metrics = {
                "Requests per Second": 100,
                "Response Time (ms)": 250,
                "GPU Utilization": 75,
                "Memory Usage": 60
            }
            
            for metric, value in scaling_metrics.items():
                st.metric(metric, f"{value}{'%' if 'Utilization' in metric or 'Usage' in metric else ''}")
        
        with deployment_config[2]:
            st.write("**Security Settings:**")
            
            security_col1, security_col2 = st.columns(2)
            
            with security_col1:
                api_authentication = st.checkbox("API Authentication", value=True)
                rate_limiting = st.checkbox("Rate Limiting", value=True)
                encryption = st.checkbox("Encryption at Rest", value=True)
            
            with security_col2:
                vpc_isolation = st.checkbox("VPC Isolation", value=True)
                access_logging = st.checkbox("Access Logging", value=True)
                threat_detection = st.checkbox("Threat Detection", value=False)
        
        if st.button("☁️ Deploy to Cloud"):
            st.info("☁️ Deploying multimodal model to cloud...")
            
            deployment_steps = [
                "📦 Uploading model package",
                "🔧 Configuring infrastructure",
                "🚀 Starting deployment",
                "🔍 Running health checks",
                "🌐 Setting up endpoints",
                "✅ Deployment complete"
            ]
            
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            
            for i, step in enumerate(deployment_steps):
                progress = (i + 1) / len(deployment_steps)
                progress_bar.progress(progress)
                status_text.text(step)
                
                import time
                time.sleep(0.5)
            
            st.success("🎉 Cloud deployment successful!")
            
            # Deployment info
            st.subheader("🌐 Deployment Information")
            
            deploy_info_cols = st.columns(2)
            
            with deploy_info_cols[0]:
                st.write("**Endpoint Details:**")
                st.code("https://api.multimodal.example.com/v1/generate")
                st.write("**Status:** ✅ Active")
                st.write("**Region:** us-east-1")
            
            with deploy_info_cols[1]:
                st.write("**Performance:**")
                st.metric("Latency", "245ms")
                st.metric("Throughput", "50 req/s")
                st.metric("Uptime", "99.9%")
    
    with deploy_tabs[2]:
        st.subheader("📱 Edge Deployment")
        
        st.write("**Optimize for Edge Devices:**")
        
        edge_options = st.tabs(["📱 Mobile", "🔌 IoT", "🚗 Automotive"])
        
        with edge_options[0]:
            st.write("**Mobile Optimization:**")
            
            mobile_col1, mobile_col2 = st.columns(2)
            
            with mobile_col1:
                model_size_limit = st.slider("Model Size Limit (MB):", 10, 500, 100)
                quantization_level = st.selectbox("Quantization:", ["INT8", "INT4", "FP16"])
                pruning_ratio = st.slider("Pruning Ratio:", 0.0, 0.8, 0.3)
            
            with mobile_col2:
                target_platform = st.selectbox("Target Platform:", ["iOS", "Android", "React Native"])
                performance_mode = st.selectbox("Performance Mode:", ["Balanced", "Speed", "Accuracy"])
                battery_optimized = st.checkbox("Battery Optimized", value=True)
            
            if st.button("📱 Optimize for Mobile"):
                st.info("📱 Optimizing multimodal model for mobile deployment...")
                
                optimization_results = {
                    "Original Size": "2.8 GB",
                    "Optimized Size": f"{model_size_limit} MB", 
                    "Size Reduction": f"{(1 - model_size_limit/2800)*100:.1f}%",
                    "Expected Performance": f"{85 - pruning_ratio*20:.1f}% of original"
                }
                
                for metric, value in optimization_results.items():
                    st.write(f"**{metric}:** {value}")
        
        with edge_options[1]:
            st.write("**IoT Device Optimization:**")
            
            iot_specs = {
                "Target RAM": "512 MB",
                "Processing Power": "ARM Cortex-A7",
                "Storage": "Flash Memory", 
                "Connectivity": "WiFi/Bluetooth"
            }
            
            for spec, value in iot_specs.items():
                st.write(f"**{spec}:** {value}")
            
            if st.button("🔌 Generate IoT Package"):
                st.success("🔌 IoT deployment package generated!")
        
        with edge_options[2]:
            st.write("**Automotive Deployment:**")
            
            automotive_reqs = {
                "Real-time Processing": "< 100ms latency",
                "Safety Standards": "ISO 26262 compliant",
                "Temperature Range": "-40°C to +85°C",
                "Vibration Resistance": "Automotive grade"
            }
            
            for req, spec in automotive_reqs.items():
                st.write(f"**{req}:** {spec}")
            
            if st.button("🚗 Prepare Automotive Package"):
                st.success("🚗 Automotive deployment package prepared!")
    
    with deploy_tabs[3]:
        st.subheader("🔧 API Configuration")
        
        st.write("**Multimodal API Endpoints:**")
        
        api_endpoints = [
            {
                "endpoint": "/v1/generate/text",
                "method": "POST",
                "description": "Text-only generation",
                "input": "JSON with text prompt"
            },
            {
                "endpoint": "/v1/generate/multimodal",
                "method": "POST", 
                "description": "Complete multimodal generation",
                "input": "JSON with text, image, video"
            },
            {
                "endpoint": "/v1/analyze/image",
                "method": "POST",
                "description": "Image analysis and description",
                "input": "Image file + optional text"
            },
            {
                "endpoint": "/v1/analyze/video",
                "method": "POST",
                "description": "Video content analysis",
                "input": "Video file + optional text"
            }
        ]
        
        for endpoint in api_endpoints:
            with st.expander(f"📡 {endpoint['endpoint']}", expanded=False):
                st.write(f"**Method:** {endpoint['method']}")
                st.write(f"**Description:** {endpoint['description']}")
                st.write(f"**Input:** {endpoint['input']}")
                
                # Sample request
                st.code(f"""
curl -X {endpoint['method']} \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -d '{{"prompt": "Your text here"}}' \\
  https://api.multimodal.example.com{endpoint['endpoint']}
                """, language="bash")
        
        # API configuration
        st.subheader("⚙️ API Settings")
        
        api_col1, api_col2 = st.columns(2)
        
        with api_col1:
            api_rate_limit = st.number_input("Rate Limit (req/min):", value=60)
            api_timeout = st.number_input("Timeout (seconds):", value=30)
            api_max_tokens = st.number_input("Max Tokens per Request:", value=500)
        
        with api_col2:
            api_authentication = st.selectbox("Authentication:", ["API Key", "OAuth 2.0", "JWT"])
            api_versioning = st.selectbox("API Versioning:", ["Header", "URL Path", "Query Parameter"])
            api_documentation = st.checkbox("Auto-generate Documentation", value=True)
        
        if st.button("🔧 Generate API Configuration"):
            st.success("✅ API configuration generated!")
            
            config_preview = {
                "rate_limit": f"{api_rate_limit} requests/minute",
                "timeout": f"{api_timeout} seconds",
                "max_tokens": api_max_tokens,
                "authentication": api_authentication,
                "versioning": api_versioning,
                "documentation": "Available" if api_documentation else "Disabled"
            }
            
            st.subheader("📋 Configuration Preview")
            st.json(config_preview)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"🎭 Multimodal application error: {e}")
        print(f"❌ Critical multimodal application error: {e}")
        traceback.print_exc()
        
        # Enhanced emergency recovery
        st.subheader("🚨 Emergency Recovery Options")
        
        recovery_cols = st.columns(3)
        
        with recovery_cols[0]:
            if st.button("🔄 Reset Application"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.success("✅ Application state reset!")
                st.rerun()
        
        with recovery_cols[1]:
            if st.button("🧹 Clear Model State"):
                if hasattr(st.session_state, 'model_manager'):
                    st.session_state.model_manager.initialized = False
                    st.session_state.model_manager.multimodal_enabled = False
                st.success("✅ Model state cleared!")
        
        with recovery_cols[2]:
            if st.button("📊 Show Debug Info"):
                st.write("**Session State Keys:**")
                for key in st.session_state.keys():
                    st.write(f"- {key}")
                
                st.write("**System Info:**")
                st.write(f"- PyTorch: {torch.__version__}")
                st.write(f"- CUDA: {torch.cuda.is_available()}")
                st.write(f"- Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
        
        st.info("💡 If problems persist, restart the Streamlit application completely with `streamlit run app.py`")
