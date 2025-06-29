#!/usr/bin/env python3
"""
Comprehensive Checkpoint Resume Verification Test
This script demonstrates and verifies that training resumes from correct checkpoint steps
"""

import requests
import json
import time
import sys
from typing import Dict, Any, List, Optional

BASE_URL = "http://localhost:5000"

def make_request(method: str, endpoint: str, data: Dict = None) -> Dict:
    """Make API request with error handling"""
    url = f"{BASE_URL}{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        elif method == "PUT":
            response = requests.put(url, json=data)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return {}

def create_test_model() -> Optional[Dict]:
    """Create a test model for checkpoint testing"""
    print("🔧 Creating test model...")
    model_data = {
        "name": "Checkpoint Test Model",
        "config": {
            "vocab_size": 50257,
            "n_positions": 1024,
            "n_embd": 768,
            "n_layer": 12,
            "n_head": 12,
            "learning_rate": 0.0005
        }
    }
    
    result = make_request("POST", "/api/models", model_data)
    if result:
        print(f"✅ Model created: ID {result.get('id')}")
        return result
    return None

def start_initial_training(model_id: int) -> Optional[Dict]:
    """Start initial training that will create checkpoints"""
    print("🚀 Starting initial training...")
    training_config = {
        "modelId": model_id,
        "name": "Initial Training Run",
        "dataFiles": [],
        "batch_size": 8,
        "learning_rate": 0.0005,
        "max_steps": 100
    }
    
    result = make_request("POST", "/api/training/start", training_config)
    if result:
        print(f"✅ Training started: Run ID {result.get('id')}")
        return result
    return None

def wait_for_checkpoints(model_id: int, min_checkpoints: int = 2) -> List[Dict]:
    """Wait for training to create some checkpoints"""
    print(f"⏳ Waiting for at least {min_checkpoints} checkpoints...")
    
    for attempt in range(30):  # Wait up to 30 seconds
        checkpoints = make_request("GET", f"/api/models/{model_id}/checkpoints")
        if isinstance(checkpoints, list) and len(checkpoints) >= min_checkpoints:
            print(f"✅ Found {len(checkpoints)} checkpoints")
            for cp in checkpoints:
                print(f"   📁 {cp.get('name')} - Step {cp.get('step')}")
            return checkpoints
        
        time.sleep(1)
        print(f"   Attempt {attempt + 1}/30 - Found {len(checkpoints) if isinstance(checkpoints, list) else 0} checkpoints")
    
    print("❌ Timeout waiting for checkpoints")
    return []

def stop_training() -> bool:
    """Stop current training"""
    print("⏹️ Stopping training...")
    status = make_request("GET", "/api/training/status")
    
    if status.get("isTraining") and status.get("activeRun"):
        run_id = status["activeRun"]["id"]
        result = make_request("POST", f"/api/training/{run_id}/stop")
        if result:
            print("✅ Training stopped")
            return True
    
    print("ℹ️ No training to stop")
    return True

def resume_from_checkpoint(checkpoint_id: int, checkpoint_step: int) -> Optional[Dict]:
    """Resume training from specific checkpoint"""
    print(f"🔄 Resuming from checkpoint ID {checkpoint_id} at step {checkpoint_step}...")
    
    resume_config = {
        "config": {
            "batch_size": 8,
            "learning_rate": 0.0005,
            "max_steps": 150  # Continue beyond original checkpoint
        }
    }
    
    result = make_request("POST", f"/api/checkpoints/{checkpoint_id}/resume-training", resume_config)
    if result:
        print(f"✅ Resume training initiated: Run ID {result.get('id')}")
        return result
    return None

def verify_training_continuation(expected_start_step: int) -> bool:
    """Verify that training actually resumed from the expected step"""
    print(f"🔍 Verifying training resumed from step {expected_start_step}...")
    
    for attempt in range(10):
        status = make_request("GET", "/api/training/status")
        
        if status.get("isTraining") and status.get("activeRun"):
            active_run = status["activeRun"]
            current_step = active_run.get("currentStep", 0)
            run_name = active_run.get("name", "")
            
            print(f"   📊 Current step: {current_step}")
            print(f"   📝 Run name: {run_name}")
            
            # Check if the run name indicates resumption
            if "Resume from" in run_name:
                print("✅ Training run indicates checkpoint resumption")
            
            # Verify step is at or near expected start
            if current_step >= expected_start_step:
                print(f"✅ Training correctly resumed from step {expected_start_step}")
                print(f"   Current progress: {current_step}/{active_run.get('totalSteps', 'unknown')}")
                return True
            elif current_step > 0:
                print(f"⚠️ Training at step {current_step}, expected >= {expected_start_step}")
        
        time.sleep(2)
        print(f"   Verification attempt {attempt + 1}/10...")
    
    print("❌ Could not verify training continuation")
    return False

def run_comprehensive_test():
    """Run complete checkpoint resume verification test"""
    print("=" * 60)
    print("🧪 COMPREHENSIVE CHECKPOINT RESUME VERIFICATION TEST")
    print("=" * 60)
    
    # Step 1: Create test model
    model = create_test_model()
    if not model:
        print("❌ Failed to create test model")
        return False
    
    model_id = model["id"]
    
    # Step 2: Start initial training
    training_run = start_initial_training(model_id)
    if not training_run:
        print("❌ Failed to start initial training")
        return False
    
    # Step 3: Wait for checkpoints to be created
    checkpoints = wait_for_checkpoints(model_id, min_checkpoints=2)
    if not checkpoints:
        print("❌ No checkpoints created")
        return False
    
    # Step 4: Stop training
    if not stop_training():
        print("❌ Failed to stop training")
        return False
    
    # Wait a moment for training to fully stop
    time.sleep(2)
    
    # Step 5: Select a checkpoint to resume from
    selected_checkpoint = None
    for cp in checkpoints:
        if cp.get("step", 0) > 10:  # Choose a checkpoint with meaningful progress
            selected_checkpoint = cp
            break
    
    if not selected_checkpoint:
        selected_checkpoint = checkpoints[0]  # Fallback to first checkpoint
    
    checkpoint_id = selected_checkpoint["id"]
    checkpoint_step = selected_checkpoint["step"]
    
    print(f"🎯 Selected checkpoint: {selected_checkpoint['name']} at step {checkpoint_step}")
    
    # Step 6: Resume training from checkpoint
    resumed_run = resume_from_checkpoint(checkpoint_id, checkpoint_step)
    if not resumed_run:
        print("❌ Failed to resume training from checkpoint")
        return False
    
    # Step 7: Verify training continuation
    verification_success = verify_training_continuation(checkpoint_step)
    
    # Step 8: Final cleanup
    print("🧹 Cleaning up...")
    stop_training()
    
    print("=" * 60)
    if verification_success:
        print("✅ CHECKPOINT RESUME VERIFICATION: SUCCESS")
        print(f"   ✓ Training successfully resumed from step {checkpoint_step}")
        print("   ✓ Checkpoint system is working correctly")
    else:
        print("❌ CHECKPOINT RESUME VERIFICATION: FAILED")
        print("   ✗ Could not verify proper checkpoint resumption")
    print("=" * 60)
    
    return verification_success

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)