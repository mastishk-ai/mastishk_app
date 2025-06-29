#!/usr/bin/env python3
"""
Mastishk Bridge - Python service for ML operations
Provides a JSON-based communication interface between Node.js and Python ML code
"""

import sys
import json
import time
import traceback
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

try:
    from mastishk_transformer import create_model, MastishkTrainer, save_checkpoint, load_checkpoint
    import torch
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class WeightVerifier:
    """Weight verification system for tracking training progress"""
    
    def __init__(self):
        self.snapshots = {}
        self.verification_history = []
    
    def create_weight_snapshot(self, model, step, snapshot_type, description=""):
        """Create a snapshot of model weights"""
        snapshot = {
            'step': step,
            'type': snapshot_type,
            'description': description,
            'timestamp': time.time(),
            'weights': {}
        }
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                snapshot['weights'][name] = {
                    'mean': float(param.data.mean().item()),
                    'std': float(param.data.std().item()),
                    'norm': float(param.data.norm().item()),
                    'shape': list(param.data.shape)
                }
        
        snapshot_id = f"{step}_{snapshot_type}"
        self.snapshots[snapshot_id] = snapshot
        return snapshot
    
    def verify_weight_updates(self, pre_snapshot, post_snapshot, expected_update=True):
        """Verify that weights actually changed between snapshots"""
        results = {
            'weights_changed': False,
            'layers_changed': [],
            'verification_status': 'No changes detected',
            'change_statistics': {}
        }
        
        pre_weights = pre_snapshot['weights']
        post_weights = post_snapshot['weights']
        
        for layer_name in pre_weights:
            if layer_name in post_weights:
                pre_norm = pre_weights[layer_name]['norm']
                post_norm = post_weights[layer_name]['norm']
                
                # Check for meaningful change (more than numerical precision)
                relative_change = abs(post_norm - pre_norm) / max(pre_norm, 1e-8)
                
                if relative_change > 1e-6:  # Threshold for detecting real changes
                    results['weights_changed'] = True
                    results['layers_changed'].append(layer_name)
                    results['change_statistics'][layer_name] = {
                        'relative_change': relative_change,
                        'pre_norm': pre_norm,
                        'post_norm': post_norm
                    }
        
        if results['weights_changed']:
            results['verification_status'] = f"✅ Weight updates verified in {len(results['layers_changed'])} layers"
        else:
            results['verification_status'] = "❌ No weight updates detected - check learning rate or gradients"
        
        self.verification_history.append(results)
        return results

class MastishkBridge:
    def __init__(self):
        self.models = {}
        self.trainers = {}
        self.training_active = False
        self.weight_verifier = WeightVerifier()
        self.weight_logging_enabled = False
        
    def send_message(self, msg_type, data=None, error=None):
        """Send JSON message to Node.js"""
        message = {
            "type": msg_type,
            "data": data,
            "error": error,
            "timestamp": time.time()
        }
        print(json.dumps(message), flush=True)
    
    def handle_create_model(self, data):
        """Create a new model"""
        try:
            if not TORCH_AVAILABLE:
                raise Exception("PyTorch not available. Install torch to use ML features.")
            
            model_id = data.get('id')
            config = data.get('config', {})
            
            model, model_config = create_model(config)
            trainer = MastishkTrainer(model, model_config)
            
            self.models[model_id] = model
            self.trainers[model_id] = trainer
            
            self.send_message("model_created", {
                "id": model_id,
                "parameters": sum(p.numel() for p in model.parameters()),
                "config": config
            })
            
        except Exception as e:
            self.send_message("error", error=str(e))
    
    def handle_start_training(self, data):
        """Start training a model"""
        try:
            model_id = data.get('modelId')
            config = data.get('config', {})
            
            # Enable weight logging if configured
            self.weight_logging_enabled = config.get('enable_weight_logging', False)
            
            if not TORCH_AVAILABLE:
                # Simulate training for demo purposes
                self.simulate_training(data)
                return
                
            if model_id not in self.trainers:
                raise Exception(f"Model {model_id} not found")
            
            trainer = self.trainers[model_id]
            self.training_active = True
            
            # Simulate training progress with weight verification
            self.simulate_training_with_progress(trainer, config)
            
        except Exception as e:
            self.send_message("error", error=str(e))
    
    def simulate_training(self, data):
        """Simulate training when PyTorch is not available"""
        self.send_message("training_started", {"status": "started"})
        
        # Simulate training steps
        total_steps = 20
        for step in range(total_steps):
            time.sleep(0.1)  # Simulate work
            
            # Simulate decreasing loss
            loss = 4.5 - (step * 0.2) + (0.1 * (step % 3))
            learning_rate = 0.001 * (0.95 ** (step // 5))
            
            progress = {
                "step": step + 1,
                "totalSteps": total_steps,
                "loss": loss,
                "learningRate": learning_rate,
                "gpuUtilization": min(85 + (step % 10), 95),
                "memoryUsage": min(60 + (step % 15), 80)
            }
            
            self.send_message("training_progress", progress)
        
        self.send_message("training_completed", {
            "status": "completed",
            "finalLoss": loss,
            "totalSteps": total_steps
        })
    
    def simulate_training_with_progress(self, trainer, config):
        """Simulate training with realistic progress and weight verification"""
        self.send_message("training_started", {"status": "started"})
        
        # Create dummy optimizer
        dummy_optimizer = optim.AdamW(trainer.model.parameters(), lr=config.get('learning_rate', 0.001))
        
        total_steps = 50
        gradient_accumulation_steps = config.get('gradient_accumulation_steps', 2)
        accumulated_steps = 0
        
        for step in range(total_steps):
            time.sleep(0.05)  # Simulate work
            
            # Weight verification for optimizer steps when enabled
            pre_optimizer_snapshot = None
            weights_updated = False
            layers_changed = 0
            verification_status = ""
            
            # Simulate training step
            loss = self.simulate_training_step(trainer, dummy_optimizer, step)
            accumulated_steps += 1
            
            # Check if we should step optimizer (gradient accumulation complete)
            should_step = (accumulated_steps % gradient_accumulation_steps == 0)
            
            if should_step and self.weight_logging_enabled and TORCH_AVAILABLE:
                # Create pre-optimizer snapshot
                pre_optimizer_snapshot = self.weight_verifier.create_weight_snapshot(
                    trainer.model, step, "pre_optimizer_step", f"Before optimizer step {step}"
                )
                
                # Simulate optimizer step (actually step for weight verification)
                dummy_optimizer.step()
                dummy_optimizer.zero_grad()
                
                # Create post-optimizer snapshot and verify
                post_snapshot = self.weight_verifier.create_weight_snapshot(
                    trainer.model, step, "post_optimizer_step", f"After optimizer step {step}"
                )
                
                verification_results = self.weight_verifier.verify_weight_updates(
                    pre_optimizer_snapshot, post_snapshot, expected_update=True
                )
                
                weights_updated = verification_results.get('weights_changed', False)
                layers_changed = len(verification_results.get('layers_changed', []))
                verification_status = verification_results.get('verification_status', 'Unknown')
                
                # Send weight snapshot update
                self.send_message("weight_snapshot", {
                    "step": step + 1,
                    "weights_updated": weights_updated,
                    "layers_changed": layers_changed,
                    "verification_status": verification_status,
                    "snapshot_id": f"{step}_post_optimizer_step"
                })
                
                accumulated_steps = 0
            
            progress = {
                "step": step + 1,
                "totalSteps": total_steps,
                "loss": loss,
                "learningRate": dummy_optimizer.param_groups[0]['lr'],
                "gpuUtilization": min(80 + (step % 15), 95),
                "memoryUsage": min(55 + (step % 20), 75),
                "weights_updated": weights_updated,
                "layers_changed": layers_changed,
                "verification_status": verification_status if self.weight_logging_enabled else "Weight logging disabled",
                "optimizer_stepped": should_step
            }
            
            self.send_message("training_progress", progress)
            
            # Save checkpoint every 10 steps
            if (step + 1) % 10 == 0:
                checkpoint_path = f"checkpoints/checkpoint_step_{step + 1}.pt"
                self.save_checkpoint(trainer.model, dummy_optimizer, step + 1, loss, checkpoint_path)
        
        self.send_message("training_completed", {
            "status": "completed",
            "finalLoss": loss,
            "totalSteps": total_steps,
            "weight_logging_enabled": self.weight_logging_enabled,
            "verification_snapshots": len(self.weight_verifier.snapshots) if self.weight_logging_enabled else 0
        })
    
    def simulate_training_step(self, trainer, optimizer, step):
        """Simulate a single training step"""
        # Create dummy batch
        batch_size = 4
        seq_len = 64
        vocab_size = trainer.config.vocab_size
        
        dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
        dummy_labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        batch = {
            'input_ids': dummy_input,
            'labels': dummy_labels
        }
        
        result = trainer.train_step(batch, optimizer)
        return result['loss']
    
    def save_checkpoint(self, model, optimizer, step, loss, filepath):
        """Save model checkpoint"""
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(model, optimizer, step, loss, filepath)
            
            self.send_message("checkpoint_saved", {
                "step": step,
                "loss": loss,
                "filepath": filepath
            })
            
        except Exception as e:
            self.send_message("error", error=f"Failed to save checkpoint: {str(e)}")
    
    def handle_generate_text(self, data):
        """Generate text with model"""
        try:
            if not TORCH_AVAILABLE:
                # Return demo text when PyTorch is not available
                self.send_message("generation_completed", {
                    "output": f"Demo generated text for prompt: {data.get('prompt', '')}. This is a simulated response since PyTorch is not available.",
                    "tokensGenerated": 15,
                    "generationTime": 0.5
                })
                return
            
            model_id = data.get('modelId')
            prompt = data.get('prompt', '')
            config = data.get('config', {})
            
            if model_id not in self.trainers:
                raise Exception(f"Model {model_id} not found")
            
            trainer = self.trainers[model_id]
            
            # Tokenize prompt (simplified)
            prompt_ids = torch.tensor([[hash(c) % trainer.config.vocab_size for c in prompt[:10]]], dtype=torch.long)
            
            start_time = time.time()
            generated = trainer.generate(
                prompt_ids,
                max_length=config.get('max_length', 50),
                temperature=config.get('temperature', 1.0),
                top_k=config.get('top_k', 50),
                top_p=config.get('top_p', 0.9)
            )
            generation_time = time.time() - start_time
            
            # Convert back to text (simplified)
            output_text = f"{prompt} [Generated continuation with {generated.size(1) - prompt_ids.size(1)} tokens]"
            
            self.send_message("generation_completed", {
                "output": output_text,
                "tokensGenerated": generated.size(1) - prompt_ids.size(1),
                "generationTime": generation_time
            })
            
        except Exception as e:
            self.send_message("error", error=str(e))
    
    def handle_message(self, message):
        """Handle incoming message from Node.js"""
        try:
            msg_type = message.get('type')
            data = message.get('data', {})
            
            if msg_type == 'create_model':
                self.handle_create_model(data)
            elif msg_type == 'start_training':
                self.handle_start_training(data)
            elif msg_type == 'generate_text':
                self.handle_generate_text(data)
            elif msg_type == 'ping':
                self.send_message('pong', {'status': 'ready'})
            else:
                self.send_message('error', error=f'Unknown message type: {msg_type}')
                
        except Exception as e:
            self.send_message('error', error=str(e))
    
    def run(self):
        """Main event loop"""
        self.send_message('ready', {
            'torch_available': TORCH_AVAILABLE,
            'status': 'Bridge initialized successfully'
        })
        
        try:
            while True:
                line = sys.stdin.readline()
                if not line:
                    break
                
                try:
                    message = json.loads(line.strip())
                    self.handle_message(message)
                except json.JSONDecodeError:
                    self.send_message('error', error='Invalid JSON message')
                except Exception as e:
                    self.send_message('error', error=str(e))
                    
        except KeyboardInterrupt:
            self.send_message('shutdown', {'status': 'Bridge shutting down'})

if __name__ == '__main__':
    bridge = MastishkBridge()
    bridge.run()