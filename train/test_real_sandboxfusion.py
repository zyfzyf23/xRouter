#!/usr/bin/env python3

import json
import time

import requests

SANDBOX_FUSION_SERVERS = "ip-10-3-76-98"
SANDBOX_FUSION_PORT = 8080

def test_real_sandboxfusion():
    """Test the real SandboxFusion server"""
    
    print("Testing Real SandboxFusion Server")
    print("=" * 50)
    
    # Test 1: Check if server is running
    try:
        response = requests.get(f"http://{SANDBOX_FUSION_SERVERS}:{SANDBOX_FUSION_PORT}/v1/ping", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running!")
            print(f"Ping response: {response.text}")
        else:
            print(f"❌ Server responded with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return False
    
    print()
    
    # Test 2: Basic code execution
    code_request = {
        "code": 'print("Hello from SandboxFusion!")',
        "language": "python",
        "run_timeout": 10,
        "compile_timeout": 10,
        "memory_limit_MB": 512
    }
    
    try:
        response = requests.post(
            f"http://{SANDBOX_FUSION_SERVERS}:{SANDBOX_FUSION_PORT}/run_code",
            json=code_request,
            timeout=15
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Code execution successful!")
            print(f"Status: {result['status']}")
            print(f"Message: {result['message']}")
            if result['run_result']:
                print(f"Output: {result['run_result']['stdout']}")
                print(f"Error: {result['run_result']['stderr']}")
        else:
            print(f"❌ Code execution failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Code execution test failed: {e}")
        return False
    
    print()
    
    # Test 3: Function with error handling
    code_request2 = {
        "code": '''
def add(a, b):
    return a + b

result = add(5, 3)
print(f"5 + 3 = {result}")
print(undefined_variable)  # This will cause an error
''',
        "language": "python",
        "run_timeout": 10,
        "compile_timeout": 10,
        "memory_limit_MB": 512
    }
    
    try:
        response = requests.post(
            f"http://{SANDBOX_FUSION_SERVERS}:{SANDBOX_FUSION_PORT}/run_code",
            json=code_request2,
            timeout=15
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Error handling test successful!")
            print(f"Status: {result['status']}")
            print(f"Message: {result['message']}")
            if result['run_result']:
                print(f"Output: {result['run_result']['stdout']}")
                print(f"Error: {result['run_result']['stderr']}")
        else:
            print(f"❌ Error handling test failed: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
    
    print()
    
    # Test 4: Test with stdin
    code_request3 = {
        "code": '''
name = input("Enter your name: ")
print(f"Hello, {name}!")
''',
        "language": "python",
        "stdin": "SandboxFusion User",
        "run_timeout": 10,
        "compile_timeout": 10,
        "memory_limit_MB": 512
    }
    
    try:
        response = requests.post(
            f"http://{SANDBOX_FUSION_SERVERS}:{SANDBOX_FUSION_PORT}/run_code",
            json=code_request3,
            timeout=15
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Stdin test successful!")
            print(f"Status: {result['status']}")
            if result['run_result']:
                print(f"Output: {result['run_result']['stdout']}")
        else:
            print(f"❌ Stdin test failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Stdin test failed: {e}")
    
    
    print("🎉 SandboxFusion server is working correctly!")
    return True

if __name__ == "__main__":
    test_real_sandboxfusion() 