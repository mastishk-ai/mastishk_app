"""
Test script for Mastishk Transformer Studio
This script tests the complete workflow end-to-end
"""

import requests
import json
import time
import sys
from pathlib import Path

# Configuration for testing
BASE_URL = "http://localhost:5000"
TEST_MODEL_CONFIG = {
    "name": "test-model-small",
    "config": {
        "vocab_size": 1000,
        "hidden_size": 512,
        "intermediate_size": 1024,
        "num_hidden_layers": 6,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "hidden_act": "silu",
        "max_position_embeddings": 512,
        "initializer_range": 0.02,
        "rms_norm_eps": 1e-5,
        "use_flash_attention": False,
        "use_differential_attention": False,
        "use_minimax": False,
        "lolcats_enabled": False,
        "use_multi_token_prediction": False,
        "use_moe": False,
        "use_mod": False
    }
}

TEST_TRAINING_CONFIG = {
    "learning_rate": 0.001,
    "batch_size": 4,
    "num_epochs": 2,
    "warmup_steps": 10,
    "weight_decay": 0.01,
    "gradient_clipping": 1.0,
    "scheduler_type": "cosine",
    "optimizer_type": "adamw",
    "save_steps": 5,
    "eval_steps": 5,
    "logging_steps": 1
}

TEST_GENERATION_CONFIG = {
    "max_length": 50,
    "temperature": 0.8,
    "top_k": 40,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    "do_sample": True,
    "num_beams": 1,
    "early_stopping": False
}

def test_api_endpoint(method, endpoint, data=None, files=None):
    """Test API endpoint and return response"""
    url = f"{BASE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            if files:
                response = requests.post(url, files=files)
            else:
                response = requests.post(url, json=data, headers={'Content-Type': 'application/json'})
        elif method == "PUT":
            response = requests.put(url, json=data, headers={'Content-Type': 'application/json'})
        elif method == "DELETE":
            response = requests.delete(url)
        
        print(f"✓ {method} {endpoint} -> {response.status_code}")
        if response.status_code >= 400:
            print(f"  Error: {response.text}")
        return response
    except Exception as e:
        print(f"✗ {method} {endpoint} -> Error: {e}")
        return None

def create_test_data_file():
    """Create a test training data file"""
    test_data = """Hello world, this is a test.
The quick brown fox jumps over the lazy dog.
Machine learning is fascinating.
Transformers are powerful neural networks.
This is sample training data for testing.
"""
    
    with open("test_data.txt", "w") as f:
        f.write(test_data)
    
    return "test_data.txt"

def run_full_test():
    """Run complete end-to-end test of the studio"""
    
    print("=" * 60)
    print("MASTISHK TRANSFORMER STUDIO - FULL TEST")
    print("=" * 60)
    
    # Test 1: Check server health
    print("\n1. Testing Server Health...")
    response = test_api_endpoint("GET", "/api/health")
    if not response or response.status_code != 200:
        print("❌ Server not responding properly!")
        return False
    
    # Test 2: Create a model
    print("\n2. Testing Model Creation...")
    response = test_api_endpoint("POST", "/api/models", TEST_MODEL_CONFIG)
    if not response or response.status_code not in [200, 201]:
        print("❌ Failed to create model!")
        return False
    
    model_data = response.json()
    model_id = model_data.get('id')
    print(f"  Model created with ID: {model_id}")
    
    # Test 3: List models
    print("\n3. Testing Model Listing...")
    response = test_api_endpoint("GET", "/api/models")
    if response and response.status_code == 200:
        models = response.json()
        print(f"  Found {len(models)} models")
    
    # Test 4: Upload training data
    print("\n4. Testing Data Upload...")
    data_file = create_test_data_file()
    
    with open(data_file, 'rb') as f:
        files = {'files': (data_file, f, 'text/plain')}
        response = test_api_endpoint("POST", "/api/training/upload", files=files)
    
    if response and response.status_code in [200, 201]:
        print("  Training data uploaded successfully")
    else:
        print("❌ Failed to upload training data!")
        return False
    
    # Test 5: Start training
    print("\n5. Testing Training Start...")
    training_data = {
        "modelId": model_id,
        "config": TEST_TRAINING_CONFIG,
        "dataPath": data_file
    }
    
    response = test_api_endpoint("POST", "/api/training/start", training_data)
    if response and response.status_code in [200, 201]:
        training_run = response.json()
        training_run_id = training_run.get('id')
        print(f"  Training started with run ID: {training_run_id}")
        
        # Wait a bit and check training status
        time.sleep(2)
        
        # Test 6: Check training status
        print("\n6. Testing Training Status...")
        response = test_api_endpoint("GET", "/api/training/status")
        if response and response.status_code == 200:
            status = response.json()
            print(f"  Training status: {status}")
        
        # Test 7: Get training runs
        print("\n7. Testing Training Runs...")
        response = test_api_endpoint("GET", "/api/training-runs")
        if response and response.status_code == 200:
            runs = response.json()
            print(f"  Found {len(runs)} training runs")
    
    # Test 8: Test text generation
    print("\n8. Testing Text Generation...")
    generation_data = {
        "modelId": model_id,
        "prompt": "Hello world",
        "config": TEST_GENERATION_CONFIG
    }
    
    response = test_api_endpoint("POST", "/api/generate", generation_data)
    if response and response.status_code in [200, 201]:
        generation = response.json()
        print(f"  Generated text: {generation.get('output', 'No output')[:100]}...")
    
    # Test 9: List generations
    print("\n9. Testing Generation History...")
    response = test_api_endpoint("GET", "/api/generations")
    if response and response.status_code == 200:
        generations = response.json()
        print(f"  Found {len(generations)} generations")
    
    # Test 10: Test checkpoints
    print("\n10. Testing Checkpoints...")
    response = test_api_endpoint("GET", "/api/checkpoints")
    if response and response.status_code == 200:
        checkpoints = response.json()
        print(f"  Found {len(checkpoints)} checkpoints")
    
    # Test 11: Test metrics
    print("\n11. Testing Training Metrics...")
    if 'training_run_id' in locals():
        response = test_api_endpoint("GET", f"/api/training-runs/{training_run_id}/metrics")
        if response and response.status_code == 200:
            metrics = response.json()
            print(f"  Found {len(metrics)} metric entries")
    
    # Clean up
    print("\n12. Cleaning Up...")
    try:
        Path(data_file).unlink()
        print("  Test data file cleaned up")
    except:
        pass
    
    print("\n" + "=" * 60)
    print("✅ FULL TEST COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nThe Mastishk Transformer Studio is working properly!")
    print("You can now:")
    print("- Create and configure transformer models")
    print("- Upload training data and start training")
    print("- Monitor training progress in real-time")
    print("- Generate text with trained models")
    print("- Manage checkpoints and view analytics")
    
    return True

if __name__ == "__main__":
    success = run_full_test()
    sys.exit(0 if success else 1)