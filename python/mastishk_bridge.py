#!/usr/bin/env python3
"""
Mastishk Transformer Bridge Service
Handles communication between the Node.js server and Python ML operations
"""

import sys
import json
import traceback
import os
import torch
import torch.nn as nn
from pathlib import Path
import importlib.util
from typing import Dict, Any, Optional
import threading
import queue
import time
from datetime import datetime

# Import the Mastishk transformer
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load the Mastishk transformer module
spec = importlib.util.spec_from_file_location(
    "mastishk_transformer", 
    os.path.join(os.path.dirname(__file__), "..", "attached_assets", "mastishk_transformer_1751122653406.py")
)
mastishk_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mastishk_module)

# Import Mastishk classes
MastishkConfig = mastishk_module.MastishkConfig
MastishkForCausalLM = getattr(mastishk_module, 'MastishkForCausalLM', None)

# If MastishkForCausalLM doesn't exist, we'll create a placeholder
if MastishkForCausalLM is None:
    class MastishkForCausalLM(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            # This would be replaced with actual model implementation
            
        def forward(self, input_ids, **kwargs):
            # Placeholder implementation
            batch_size, seq_len = input_ids.shape
            vocab_size = self.config.vocab_size
            logits = torch.randn(batch_size, seq_len, vocab_size)
            return type('Output', (), {'logits': logits})()
        
        def generate(self, input_ids, **kwargs):
            # Placeholder generation
            max_length = kwargs.get('max_length', 100)
            batch_size = input_ids.shape[0]
            
            # Simple random generation for demo
            for _ in range(max_length - input_ids.shape[1]):
                with torch.no_grad():
                    outputs = self.forward(input_ids)
                    next_token_logits = outputs.logits[:, -1, :]
                    
                    # Apply temperature
                    temperature = kwargs.get('temperature', 1.0)
                    next_token_logits = next_token_logits / temperature
                    
                    # Apply top_k
                    top_k = kwargs.get('top_k', 50)
                    if top_k > 0:
                        top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                        next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                        next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                    
                    # Sample next token
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    input_ids = torch.cat([input_ids, next_token], dim=-1)
                    
                    # Check for EOS token
                    if next_token.item() == self.config.eos_token_id:
                        break
            
            return input_ids

class MastishkBridge:
    def __init__(self):
        self.model: Optional[MastishkForCausalLM] = None
        self.config: Optional[MastishkConfig] = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.training_active = False
        self.training_thread = None
        self.training_queue = queue.Queue()
        
        # Send initialization message
        self.send_message('initialized', {'device': str(self.device)})
    
    def send_message(self, message_type: str, data: Any):
        """Send message to Node.js process"""
        message = {
            'type': message_type,
            'data': data,
            'timestamp': time.time()
        }
        print(json.dumps(message), flush=True)
    
    def log_error(self, error: Exception, context: str = ""):
        """Log error and send to Node.js"""
        error_msg = f"{context}: {str(error)}"
        self.send_message('error', {
            'error': error_msg,
            'traceback': traceback.format_exc()
        })
    
    def initialize_model(self, config_data: Dict[str, Any]):
        """Initialize the Mastishk model with given configuration"""
        try:
            # Create config
            self.config = MastishkConfig(**config_data)
            
            # Initialize model
            self.model = MastishkForCausalLM(self.config)
            self.model.to(self.device)
            
            # Initialize simple tokenizer (placeholder)
            class SimpleTokenizer:
                def __init__(self, vocab_size):
                    self.vocab_size = vocab_size
                    self.bos_token_id = 1
                    self.eos_token_id = 2
                    self.pad_token_id = 0
                
                def encode(self, text, **kwargs):
                    # Simple character-level tokenization for demo
                    tokens = [ord(c) % self.vocab_size for c in text]
                    if kwargs.get('add_special_tokens', True):
                        tokens = [self.bos_token_id] + tokens
                    return tokens
                
                def decode(self, tokens, **kwargs):
                    # Simple decoding
                    if isinstance(tokens, torch.Tensor):
                        tokens = tokens.tolist()
                    
                    # Remove special tokens
                    cleaned_tokens = [t for t in tokens if t not in [self.bos_token_id, self.eos_token_id, self.pad_token_id]]
                    
                    # Convert back to characters
                    try:
                        text = ''.join([chr(min(max(t, 32), 126)) for t in cleaned_tokens])
                        return text
                    except:
                        return f"Generated {len(cleaned_tokens)} tokens"
            
            self.tokenizer = SimpleTokenizer(self.config.vocab_size)
            
            # Calculate model parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            self.send_message('model_loaded', {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'device': str(self.device),
                'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
            })
            
        except Exception as e:
            self.log_error(e, "Model initialization")
            self.send_message('model_error', {'error': str(e)})
    
    def start_training(self, training_config: Dict[str, Any], data_path: str):
        """Start training process"""
        try:
            if self.model is None:
                raise ValueError("Model not initialized")
            
            if self.training_active:
                raise ValueError("Training already in progress")
            
            self.training_active = True
            
            # Start training in separate thread
            self.training_thread = threading.Thread(
                target=self._training_loop,
                args=(training_config, data_path)
            )
            self.training_thread.start()
            
        except Exception as e:
            self.log_error(e, "Training start")
            self.send_message('training_error', {'error': str(e)})
    
    def _training_loop(self, training_config: Dict[str, Any], data_path: str):
        """Main training loop"""
        try:
            max_steps = training_config.get('max_steps', 1000)
            learning_rate = training_config.get('learning_rate', 5e-4)
            
            # Setup optimizer
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=training_config.get('weight_decay', 0.01)
            )
            
            # Setup scheduler
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max_steps
            )
            
            # Load training data (simplified)
            training_data = self._load_training_data(data_path)
            
            self.model.train()
            
            for step in range(max_steps):
                if not self.training_active:
                    break
                
                # Simulate training step
                batch = self._get_training_batch(training_data)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch['input_ids'])
                
                # Calculate loss (simplified)
                loss = torch.nn.functional.cross_entropy(
                    outputs.logits.view(-1, outputs.logits.size(-1)),
                    batch['labels'].view(-1),
                    ignore_index=self.config.pad_token_id
                )
                
                # Backward pass
                loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    training_config.get('max_grad_norm', 1.0)
                )
                
                optimizer.step()
                scheduler.step()
                
                # Send progress update
                if step % 10 == 0:
                    progress = {
                        'step': step,
                        'loss': loss.item(),
                        'learningRate': scheduler.get_last_lr()[0],
                        'gpuUtilization': self._get_gpu_utilization(),
                        'memoryUsage': self._get_memory_usage(),
                    }
                    
                    self.send_message('training_progress', progress)
                
                # Small delay to prevent overwhelming
                time.sleep(0.1)
            
            self.training_active = False
            self.send_message('training_complete', {
                'final_step': step,
                'final_loss': loss.item() if 'loss' in locals() else 0.0
            })
            
        except Exception as e:
            self.training_active = False
            self.log_error(e, "Training loop")
            self.send_message('training_error', {'error': str(e)})
    
    def stop_training(self):
        """Stop training process"""
        self.training_active = False
        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.join(timeout=5.0)
        
        self.send_message('training_stopped', {})
    
    def generate_text(self, prompt: str, generation_config: Dict[str, Any]):
        """Generate text from prompt"""
        try:
            if self.model is None:
                raise ValueError("Model not initialized")
            
            start_time = time.time()
            
            # Tokenize input
            input_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
            input_ids = torch.tensor([input_tokens], device=self.device)
            
            # Set model to eval mode
            self.model.eval()
            
            with torch.no_grad():
                # Generate
                output_ids = self.model.generate(
                    input_ids,
                    max_length=generation_config.get('max_length', 100),
                    temperature=generation_config.get('temperature', 0.7),
                    top_k=generation_config.get('top_k', 50),
                    top_p=generation_config.get('top_p', 0.9),
                    do_sample=generation_config.get('do_sample', True),
                    pad_token_id=self.config.pad_token_id,
                    eos_token_id=self.config.eos_token_id
                )
            
            # Decode output
            generated_tokens = output_ids[0][len(input_tokens):]
            output_text = self.tokenizer.decode(generated_tokens)
            
            generation_time = time.time() - start_time
            
            result = {
                'output': output_text,
                'tokensGenerated': len(generated_tokens),
                'generationTime': generation_time
            }
            
            self.send_message('generation_complete', result)
            
        except Exception as e:
            self.log_error(e, "Text generation")
            self.send_message('generation_error', {'error': str(e)})
    
    def save_checkpoint(self, path: str, metadata: Dict[str, Any]):
        """Save model checkpoint"""
        try:
            if self.model is None:
                raise ValueError("Model not initialized")
            
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'config': self.config.__dict__,
                'metadata': metadata,
                'timestamp': datetime.now().isoformat()
            }
            
            torch.save(checkpoint, path)
            self.send_message('checkpoint_saved', {'path': path})
            
        except Exception as e:
            self.log_error(e, "Checkpoint save")
            self.send_message('checkpoint_error', {'error': str(e)})
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            # Load config
            self.config = MastishkConfig(**checkpoint['config'])
            
            # Recreate model
            self.model = MastishkForCausalLM(self.config)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            
            self.send_message('checkpoint_loaded', {
                'path': path,
                'metadata': checkpoint.get('metadata', {})
            })
            
        except Exception as e:
            self.log_error(e, "Checkpoint load")
            self.send_message('checkpoint_error', {'error': str(e)})
    
    def _load_training_data(self, data_path: str):
        """Load training data from path"""
        # Simplified data loading
        try:
            data_files = []
            data_dir = Path(data_path)
            
            if data_dir.is_dir():
                data_files = list(data_dir.glob('*.txt'))
            elif data_dir.is_file():
                data_files = [data_dir]
            
            texts = []
            for file_path in data_files:
                with open(file_path, 'r', encoding='utf-8') as f:
                    texts.append(f.read())
            
            return texts
            
        except Exception as e:
            self.log_error(e, "Data loading")
            return ["Sample training text for demonstration."]
    
    def _get_training_batch(self, training_data):
        """Get a training batch"""
        # Simplified batch creation
        import random
        
        text = random.choice(training_data)
        tokens = self.tokenizer.encode(text)
        
        # Truncate to reasonable length
        max_length = min(512, len(tokens))
        tokens = tokens[:max_length]
        
        # Create labels (shifted by one for causal LM)
        input_tokens = tokens[:-1]
        label_tokens = tokens[1:]
        
        # Pad to batch size
        input_ids = torch.tensor([input_tokens], device=self.device)
        labels = torch.tensor([label_tokens], device=self.device)
        
        return {
            'input_ids': input_ids,
            'labels': labels
        }
    
    def _get_gpu_utilization(self):
        """Get GPU utilization percentage"""
        try:
            if torch.cuda.is_available():
                return torch.cuda.utilization()
            return 0
        except:
            return 0
    
    def _get_memory_usage(self):
        """Get GPU memory usage in GB"""
        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024**3)
            return 0
        except:
            return 0
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_training()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.send_message('cleanup_complete', {})

def main():
    """Main bridge process"""
    bridge = MastishkBridge()
    
    try:
        # Process messages from Node.js
        for line in sys.stdin:
            try:
                message = json.loads(line.strip())
                message_type = message.get('type')
                data = message.get('data', {})
                
                if message_type == 'initialize_model':
                    bridge.initialize_model(data)
                elif message_type == 'start_training':
                    bridge.start_training(data['config'], data['dataPath'])
                elif message_type == 'stop_training':
                    bridge.stop_training()
                elif message_type == 'generate_text':
                    bridge.generate_text(data['prompt'], data['config'])
                elif message_type == 'save_checkpoint':
                    bridge.save_checkpoint(data['path'], data['metadata'])
                elif message_type == 'load_checkpoint':
                    bridge.load_checkpoint(data['path'])
                elif message_type == 'cleanup':
                    bridge.cleanup()
                    break
                    
            except json.JSONDecodeError:
                continue
            except Exception as e:
                bridge.log_error(e, f"Message processing: {message_type}")
                
    except KeyboardInterrupt:
        pass
    finally:
        bridge.cleanup()

if __name__ == '__main__':
    main()
