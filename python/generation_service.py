"""
Text Generation Service for Mastishk Transformer
Handles text generation with various strategies and configurations
"""

import torch
import torch.nn.functional as F
import time
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from datetime import datetime
import json

class GenerationService:
    """Handles text generation operations"""
    
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.generation_history = []
    
    def generate(self, prompt: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate text from prompt with given configuration"""
        start_time = time.time()
        
        try:
            # Set model to eval mode
            self.model.eval()
            
            # Tokenize prompt
            input_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
            input_ids = torch.tensor([input_tokens], device=self.device)
            
            # Extract generation config
            max_length = config.get('max_length', 100)
            temperature = config.get('temperature', 0.7)
            top_p = config.get('top_p', 0.9)
            top_k = config.get('top_k', 50)
            repetition_penalty = config.get('repetition_penalty', 1.1)
            length_penalty = config.get('length_penalty', 1.0)
            no_repeat_ngram_size = config.get('no_repeat_ngram_size', 3)
            do_sample = config.get('do_sample', True)
            early_stopping = config.get('early_stopping', False)
            num_beams = config.get('num_beams', 1)
            
            # Generate text
            with torch.no_grad():
                if num_beams > 1:
                    output_ids = self._beam_search_generate(
                        input_ids, max_length, num_beams, temperature,
                        top_k, top_p, repetition_penalty, early_stopping
                    )
                else:
                    output_ids = self._sample_generate(
                        input_ids, max_length, temperature, top_k, top_p,
                        repetition_penalty, no_repeat_ngram_size, do_sample
                    )
            
            # Decode generated text
            generated_tokens = output_ids[0][len(input_tokens):]
            generated_text = self.tokenizer.decode(generated_tokens)
            
            generation_time = time.time() - start_time
            
            # Store in history
            generation_record = {
                'prompt': prompt,
                'generated_text': generated_text,
                'config': config,
                'tokens_generated': len(generated_tokens),
                'generation_time': generation_time,
                'timestamp': datetime.now().isoformat()
            }
            self.generation_history.append(generation_record)
            
            return {
                'output': generated_text,
                'tokensGenerated': len(generated_tokens),
                'generationTime': generation_time,
                'promptTokens': len(input_tokens),
                'totalTokens': len(input_tokens) + len(generated_tokens)
            }
            
        except Exception as e:
            raise Exception(f"Generation failed: {str(e)}")
    
    def _sample_generate(self, input_ids: torch.Tensor, max_length: int,
                        temperature: float, top_k: int, top_p: float,
                        repetition_penalty: float, no_repeat_ngram_size: int,
                        do_sample: bool) -> torch.Tensor:
        """Generate using sampling methods"""
        
        sequence = input_ids.clone()
        generated_tokens = []
        
        for _ in range(max_length - input_ids.shape[1]):
            # Get model predictions
            outputs = self.model(sequence)
            next_token_logits = outputs.logits[0, -1, :]
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits, sequence[0], repetition_penalty
                )
            
            # Apply no-repeat n-gram penalty
            if no_repeat_ngram_size > 0:
                next_token_logits = self._apply_no_repeat_ngram_penalty(
                    next_token_logits, sequence[0], no_repeat_ngram_size
                )
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                next_token_logits = self._top_k_filtering(next_token_logits, top_k)
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                next_token_logits = self._top_p_filtering(next_token_logits, top_p)
            
            # Sample next token
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            generated_tokens.append(next_token.item())
            sequence = torch.cat([sequence, next_token.unsqueeze(0)], dim=1)
            
            # Check for EOS token
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        return sequence
    
    def _beam_search_generate(self, input_ids: torch.Tensor, max_length: int,
                             num_beams: int, temperature: float, top_k: int,
                             top_p: float, repetition_penalty: float,
                             early_stopping: bool) -> torch.Tensor:
        """Generate using beam search"""
        
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        
        # Initialize beam search
        beam_scores = torch.zeros(batch_size, num_beams, device=self.device)
        beam_sequences = input_ids.unsqueeze(1).repeat(1, num_beams, 1)
        beam_sequences = beam_sequences.view(batch_size * num_beams, seq_len)
        
        done = [False for _ in range(batch_size)]
        
        for step in range(max_length - seq_len):
            # Get model predictions for all beams
            outputs = self.model(beam_sequences)
            next_token_logits = outputs.logits[:, -1, :]
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Get log probabilities
            next_token_scores = F.log_softmax(next_token_logits, dim=-1)
            
            # Add beam scores
            next_token_scores = next_token_scores.view(batch_size, num_beams, -1)
            next_token_scores = next_token_scores + beam_scores.unsqueeze(-1)
            
            # Reshape for top-k selection
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
            
            # Select top 2*num_beams candidates
            top_scores, top_indices = torch.topk(
                next_token_scores, 2 * num_beams, dim=-1
            )
            
            # Convert indices back to beam and token indices
            beam_indices = top_indices // vocab_size
            token_indices = top_indices % vocab_size
            
            # Update beams
            new_beam_sequences = []
            new_beam_scores = []
            
            for batch_idx in range(batch_size):
                if done[batch_idx]:
                    continue
                
                batch_beam_sequences = []
                batch_beam_scores = []
                
                for beam_idx in range(num_beams):
                    beam_id = beam_indices[batch_idx, beam_idx]
                    token_id = token_indices[batch_idx, beam_idx]
                    score = top_scores[batch_idx, beam_idx]
                    
                    # Get the sequence for this beam
                    old_beam_seq = beam_sequences[batch_idx * num_beams + beam_id]
                    new_beam_seq = torch.cat([old_beam_seq, token_id.unsqueeze(0)])
                    
                    batch_beam_sequences.append(new_beam_seq)
                    batch_beam_scores.append(score)
                
                new_beam_sequences.extend(batch_beam_sequences)
                new_beam_scores.extend(batch_beam_scores)
            
            # Update beam sequences and scores
            if new_beam_sequences:
                beam_sequences = torch.stack(new_beam_sequences)
                beam_scores = torch.stack(new_beam_scores).view(batch_size, num_beams)
            
            # Check for early stopping
            if early_stopping:
                # Simple early stopping: if all beams have EOS token
                eos_mask = (beam_sequences[:, -1] == self.tokenizer.eos_token_id)
                if eos_mask.all():
                    break
        
        # Return best sequence (first beam)
        return beam_sequences[:num_beams]
    
    def _apply_repetition_penalty(self, logits: torch.Tensor, 
                                 prev_tokens: torch.Tensor,
                                 penalty: float) -> torch.Tensor:
        """Apply repetition penalty to logits"""
        if penalty == 1.0:
            return logits
        
        # Get unique previous tokens
        unique_tokens = torch.unique(prev_tokens)
        
        # Apply penalty
        for token in unique_tokens:
            if logits[token] > 0:
                logits[token] = logits[token] / penalty
            else:
                logits[token] = logits[token] * penalty
        
        return logits
    
    def _apply_no_repeat_ngram_penalty(self, logits: torch.Tensor,
                                      prev_tokens: torch.Tensor,
                                      ngram_size: int) -> torch.Tensor:
        """Apply no-repeat n-gram penalty"""
        if ngram_size <= 0 or len(prev_tokens) < ngram_size:
            return logits
        
        # Get the last (ngram_size-1) tokens
        context = prev_tokens[-(ngram_size-1):].tolist()
        
        # Find all ngrams in the sequence
        banned_tokens = set()
        for i in range(len(prev_tokens) - ngram_size + 1):
            ngram = prev_tokens[i:i+ngram_size-1].tolist()
            if ngram == context:
                banned_tokens.add(prev_tokens[i+ngram_size-1].item())
        
        # Apply penalty to banned tokens
        for token in banned_tokens:
            logits[token] = float('-inf')
        
        return logits
    
    def _top_k_filtering(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Apply top-k filtering to logits"""
        if top_k <= 0:
            return logits
        
        # Get top-k values and indices
        top_k_values, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
        
        # Create a copy of logits with -inf for non-top-k values
        filtered_logits = torch.full_like(logits, float('-inf'))
        filtered_logits[top_k_indices] = top_k_values
        
        return filtered_logits
    
    def _top_p_filtering(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Apply top-p (nucleus) filtering to logits"""
        if top_p >= 1.0:
            return logits
        
        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        
        # Convert to probabilities
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        
        # Calculate cumulative probabilities
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Find the cutoff point
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False
        
        # Create mask for original indices
        indices_to_remove = sorted_indices_to_remove.scatter(
            0, sorted_indices, sorted_indices_to_remove
        )
        
        # Apply mask
        filtered_logits = logits.clone()
        filtered_logits[indices_to_remove] = float('-inf')
        
        return filtered_logits
    
    def get_generation_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent generation history"""
        return self.generation_history[-limit:] if self.generation_history else []
    
    def clear_history(self):
        """Clear generation history"""
        self.generation_history = []
    
    def export_history(self, file_path: str) -> bool:
        """Export generation history to file"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.generation_history, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error exporting history: {e}")
            return False
    
    def calculate_perplexity(self, text: str) -> float:
        """Calculate perplexity of given text"""
        try:
            self.model.eval()
            
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            input_ids = torch.tensor([tokens], device=self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids, labels=input_ids)
                loss = outputs.loss if hasattr(outputs, 'loss') else 0.0
                perplexity = torch.exp(loss).item()
            
            return perplexity
            
        except Exception as e:
            print(f"Error calculating perplexity: {e}")
            return float('inf')
