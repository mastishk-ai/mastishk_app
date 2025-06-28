"""
Training Service for Mastishk Transformer
Handles training operations and monitoring
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
import numpy as np
from datetime import datetime
import os

class TextDataset(Dataset):
    """Simple text dataset for training"""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Prepare all tokenized sequences
        self.tokenized_texts = []
        for text in texts:
            tokens = self.tokenizer.encode(text)
            
            # Split into chunks if too long
            for i in range(0, len(tokens), max_length):
                chunk = tokens[i:i + max_length]
                if len(chunk) > 10:  # Only keep reasonable length chunks
                    self.tokenized_texts.append(chunk)
    
    def __len__(self):
        return len(self.tokenized_texts)
    
    def __getitem__(self, idx):
        tokens = self.tokenized_texts[idx]
        
        # Pad to max_length
        if len(tokens) < self.max_length:
            tokens = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]
        
        # Create input and target sequences
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': (input_ids != self.tokenizer.pad_token_id).long()
        }

class TrainingService:
    """Handles model training operations"""
    
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.training_active = False
        self.current_step = 0
        self.progress_callback: Optional[Callable] = None
        
    def set_progress_callback(self, callback: Callable):
        """Set callback for training progress updates"""
        self.progress_callback = callback
    
    def prepare_data(self, data_path: str, batch_size: int = 2, max_length: int = 512):
        """Prepare training data"""
        # Load text files
        texts = []
        data_dir = Path(data_path)
        
        if data_dir.is_dir():
            for file_path in data_dir.glob('*.txt'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            texts.append(content)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        elif data_dir.is_file():
            with open(data_dir, 'r', encoding='utf-8') as f:
                texts.append(f.read())
        
        if not texts:
            texts = ["Sample training text for demonstration purposes."]
        
        # Create dataset
        dataset = TextDataset(texts, self.tokenizer, max_length)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0  # Avoid multiprocessing issues
        )
        
        return dataloader
    
    def train(self, config: Dict[str, Any], data_path: str):
        """Main training function"""
        try:
            self.training_active = True
            self.current_step = 0
            
            # Extract config
            learning_rate = config.get('learning_rate', 5e-4)
            max_steps = config.get('max_steps', 1000)
            batch_size = config.get('batch_size', 2)
            gradient_accumulation_steps = config.get('gradient_accumulation_steps', 4)
            max_grad_norm = config.get('max_grad_norm', 1.0)
            weight_decay = config.get('weight_decay', 0.01)
            warmup_steps = config.get('warmup_steps', 100)
            eval_steps = config.get('eval_steps', 100)
            save_steps = config.get('save_steps', 500)
            
            # Prepare data
            dataloader = self.prepare_data(data_path, batch_size)
            
            # Setup optimizer
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
            
            # Setup scheduler
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max_steps, eta_min=learning_rate * 0.1
            )
            
            # Training loop
            self.model.train()
            accumulated_loss = 0.0
            optimizer.zero_grad()
            
            data_iter = iter(dataloader)
            
            for step in range(max_steps):
                if not self.training_active:
                    break
                
                try:
                    # Get next batch
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        # Reset data iterator
                        data_iter = iter(dataloader)
                        batch = next(data_iter)
                    
                    # Move to device
                    input_ids = batch['input_ids'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    
                    # Forward pass
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    
                    # Calculate loss
                    logits = outputs.logits
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1),
                        ignore_index=self.tokenizer.pad_token_id
                    )
                    
                    # Scale loss for gradient accumulation
                    loss = loss / gradient_accumulation_steps
                    
                    # Backward pass
                    loss.backward()
                    
                    accumulated_loss += loss.item()
                    
                    # Update weights
                    if (step + 1) % gradient_accumulation_steps == 0:
                        # Clip gradients
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_grad_norm
                        )
                        
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        
                        # Send progress update
                        if self.progress_callback and step % 10 == 0:
                            avg_loss = accumulated_loss / gradient_accumulation_steps
                            
                            progress = {
                                'step': step,
                                'loss': avg_loss,
                                'learningRate': scheduler.get_last_lr()[0],
                                'gpuUtilization': self._get_gpu_utilization(),
                                'memoryUsage': self._get_memory_usage(),
                                'eta': self._estimate_eta(step, max_steps, time.time())
                            }
                            
                            self.progress_callback('training_progress', progress)
                            accumulated_loss = 0.0
                    
                    self.current_step = step
                    
                    # Small delay to prevent overwhelming
                    time.sleep(0.01)
                    
                except Exception as e:
                    print(f"Error in training step {step}: {e}")
                    continue
            
            # Training completed
            self.training_active = False
            if self.progress_callback:
                self.progress_callback('training_complete', {
                    'final_step': self.current_step,
                    'total_steps': max_steps
                })
                
        except Exception as e:
            self.training_active = False
            if self.progress_callback:
                self.progress_callback('training_error', {'error': str(e)})
            raise e
    
    def stop_training(self):
        """Stop training process"""
        self.training_active = False
    
    def _get_gpu_utilization(self):
        """Get GPU utilization"""
        try:
            if torch.cuda.is_available():
                return min(torch.cuda.utilization(), 100.0)
            return 0.0
        except:
            return 0.0
    
    def _get_memory_usage(self):
        """Get GPU memory usage in GB"""
        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated(self.device) / (1024**3)
            return 0.0
        except:
            return 0.0
    
    def _estimate_eta(self, current_step: int, total_steps: int, start_time: float):
        """Estimate time remaining"""
        if current_step == 0:
            return "Calculating..."
        
        elapsed = time.time() - start_time
        steps_per_second = current_step / elapsed
        remaining_steps = total_steps - current_step
        
        if steps_per_second > 0:
            eta_seconds = remaining_steps / steps_per_second
            hours = int(eta_seconds // 3600)
            minutes = int((eta_seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
        
        return "Unknown"
    
    def save_checkpoint(self, save_path: str, metadata: Dict[str, Any] = None):
        """Save training checkpoint"""
        try:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'current_step': self.current_step,
                'metadata': metadata or {},
                'timestamp': datetime.now().isoformat()
            }
            
            torch.save(checkpoint, save_path)
            return True
            
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
            return False
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.current_step = checkpoint.get('current_step', 0)
            return checkpoint.get('metadata', {})
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return None

class MetricsCollector:
    """Collects and manages training metrics"""
    
    def __init__(self):
        self.metrics_history = []
        self.start_time = None
    
    def start_collection(self):
        """Start metrics collection"""
        self.start_time = time.time()
        self.metrics_history = []
    
    def add_metric(self, step: int, loss: float, learning_rate: float, **kwargs):
        """Add a metric point"""
        metric = {
            'step': step,
            'loss': loss,
            'learning_rate': learning_rate,
            'timestamp': time.time(),
            **kwargs
        }
        self.metrics_history.append(metric)
    
    def get_latest_metrics(self, count: int = 100):
        """Get latest metrics"""
        return self.metrics_history[-count:] if self.metrics_history else []
    
    def get_metrics_summary(self):
        """Get summary of metrics"""
        if not self.metrics_history:
            return {}
        
        losses = [m['loss'] for m in self.metrics_history]
        
        return {
            'total_steps': len(self.metrics_history),
            'current_loss': losses[-1] if losses else 0.0,
            'min_loss': min(losses) if losses else 0.0,
            'avg_loss': np.mean(losses) if losses else 0.0,
            'training_time': time.time() - self.start_time if self.start_time else 0.0
        }
    
    def export_metrics(self, file_path: str):
        """Export metrics to file"""
        try:
            with open(file_path, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
            return True
        except Exception as e:
            print(f"Error exporting metrics: {e}")
            return False
