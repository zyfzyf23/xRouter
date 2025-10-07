#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Router API Test Client

This module provides test clients for the Router API server. It demonstrates
how to interact with the router using OpenAI-compatible client calls.

Features:
- OpenAI-compatible client usage
- Test cases for different types of queries
- Conversation history support
- Metadata inspection and cost tracking
- Error handling and retry logic

Usage:
    # Start the server first:
    python serve_router.py --port 8000
    
    # Then run tests:
    python test_router.py --url http://localhost:8000
    
    # Or run specific test:
    python test_router.py --test simple_math --url http://localhost:8000
"""

import argparse
import json
import time
from typing import Any, Dict, List, Optional

import openai
import requests


class RouterTestClient:
    """
    Test client for the Router API that mimics OpenAI client usage.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000/v1", api_key: str = "dummy"):
        """
        Initialize the test client.
        
        Args:
            base_url: Base URL of the router API
            api_key: API key (dummy for local testing)
        """
        self.base_url = base_url
        self.api_key = api_key
        
        # Initialize OpenAI client with custom base URL
        self.client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        
        # Test server connectivity
        self._test_connectivity()
    
    def _test_connectivity(self):
        """Test if the server is accessible"""
        try:
            # Test health endpoint
            health_url = self.base_url.replace("/v1", "/health")
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                print(f"✅ Server connectivity test passed: {health_url}")
            else:
                print(f"⚠️  Server responded with status {response.status_code}")
        except Exception as e:
            print(f"❌ Server connectivity test failed: {e}")
            print("Make sure the server is running: python serve_router.py")
    
    def chat_completion(self, 
                       messages: List[Dict[str, str]], 
                       model: str = "router",
                       max_tokens: Optional[int] = None,
                       temperature: float = 1.0,
                       show_metadata: bool = True) -> Dict[str, Any]:
        """
        Send a chat completion request to the router.
        
        Args:
            messages: List of chat messages
            model: Model name (typically "router")
            max_tokens: Maximum tokens for response
            temperature: Sampling temperature
            show_metadata: Whether to print metadata
            
        Returns:
            Response dictionary with content and metadata
        """
        try:
            start_time = time.time()
            
            # Make the request
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": msg["role"], "content": msg["content"]} for msg in messages],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Extract response content
            content = response.choices[0].message.content
            response_time = time.time() - start_time
            
            # Extract router metadata if available
            router_metadata = {}
            if hasattr(response, 'router_metadata'):
                router_metadata = response.router_metadata
            elif hasattr(response, 'model_extra') and 'router_metadata' in response.model_extra:
                router_metadata = response.model_extra['router_metadata']
            
            # Try to get metadata from the raw response
            raw_response = response.model_dump() if hasattr(response, 'model_dump') else response.__dict__
            if 'router_metadata' in raw_response:
                router_metadata = raw_response['router_metadata']
            
            result = {
                "content": content,
                "response_time": response_time,
                "usage": response.usage.model_dump() if hasattr(response.usage, 'model_dump') else response.usage.__dict__,
                "router_metadata": router_metadata
            }
            
            if show_metadata:
                self._print_response_info(result)
            
            return result
            
        except Exception as e:
            print(f"❌ Error in chat completion: {e}")
            return {
                "error": str(e),
                "content": None,
                "response_time": time.time() - start_time if 'start_time' in locals() else 0
            }
    
    def _print_response_info(self, result: Dict[str, Any]):
        """Print formatted response information"""
        print(f"\n📝 Response: {result['content']}")
        print(f"⏱️  Response Time: {result['response_time']:.2f}s")
        
        if result.get('usage'):
            usage = result['usage']
            print(f"🔢 Token Usage: {usage.get('total_tokens', 'N/A')} total")
            print(f"   Input: {usage.get('prompt_tokens', 'N/A')}, Output: {usage.get('completion_tokens', 'N/A')}")
        
        if result.get('router_metadata'):
            metadata = result['router_metadata']
            print(f"🤖 Model Used: {metadata.get('model_used', 'Unknown')}")
            print(f"💰 Total Cost: ${metadata.get('total_cost', 0):.6f}")
            print(f"🎯 Strategy: {metadata.get('routing_strategy', 'Unknown')}")
            
            if metadata.get('call_history'):
                print(f"📚 Call History: {len(metadata['call_history'])} calls")
                for i, call in enumerate(metadata['call_history']):
                    print(f"   {i+1}. {call.get('model_id', 'Unknown')} - ${call.get('cost', 0):.6f}")
    
    def test_simple_query(self) -> Dict[str, Any]:
        """Test a simple query that should be answered directly"""
        print("\n" + "="*50)
        print("🧪 Test: Simple Query")
        print("="*50)
        
        messages = [
            {"role": "user", "content": "What is 2 + 2?"}
        ]
        
        return self.chat_completion(messages)
    
    def test_math_problem(self) -> Dict[str, Any]:
        """Test a math problem that might require a specialized model"""
        print("\n" + "="*50)
        print("🧪 Test: Math Problem")
        print("="*50)
        
        messages = [
            {"role": "user", "content": "Solve this equation step by step: 3x² + 5x - 2 = 0"}
        ]
        
        return self.chat_completion(messages)
    
    def test_coding_task(self) -> Dict[str, Any]:
        """Test a coding task that should route to a coding model"""
        print("\n" + "="*50)
        print("🧪 Test: Coding Task")
        print("="*50)
        
        messages = [
            {"role": "user", "content": "Write a Python function to implement binary search on a sorted list. Include error handling and type hints."}
        ]
        
        return self.chat_completion(messages)
    
    def test_reasoning_task(self) -> Dict[str, Any]:
        """Test a complex reasoning task"""
        print("\n" + "="*50)
        print("🧪 Test: Complex Reasoning")
        print("="*50)
        
        messages = [
            {"role": "user", "content": "You have 100 doors, all closed. You go through 100 passes. On the first pass, you visit every door and toggle it (if it's closed, you open it; if it's open, you close it). On the second pass, you visit every 2nd door and toggle it. On the third pass, you visit every 3rd door and toggle it. This continues until the 100th pass, where you visit only the 100th door and toggle it. After all 100 passes, which doors are open?"}
        ]
        
        return self.chat_completion(messages)
    
    def test_conversation(self) -> Dict[str, Any]:
        """Test a multi-turn conversation"""
        print("\n" + "="*50)
        print("🧪 Test: Multi-turn Conversation")
        print("="*50)
        
        # First message
        messages = [
            {"role": "user", "content": "I'm planning a birthday party for 20 people. Can you help me create a checklist?"}
        ]
        
        result1 = self.chat_completion(messages, show_metadata=False)
        print(f"Assistant: {result1['content']}")
        
        # Follow-up message
        messages.append({"role": "assistant", "content": result1['content']})
        messages.append({"role": "user", "content": "Great! Now can you estimate the total budget if I want to spend around $15 per person?"})
        
        result2 = self.chat_completion(messages)
        
        return result2
    
    def test_creative_task(self) -> Dict[str, Any]:
        """Test a creative writing task"""
        print("\n" + "="*50)
        print("🧪 Test: Creative Writing")
        print("="*50)
        
        messages = [
            {"role": "user", "content": "Write a short story (2-3 paragraphs) about a robot who discovers they can dream. Make it thought-provoking and emotionally engaging."}
        ]
        
        return self.chat_completion(messages, temperature=0.8)
    
    def test_with_system_prompt(self) -> Dict[str, Any]:
        """Test with a custom system prompt"""
        print("\n" + "="*50)
        print("🧪 Test: Custom System Prompt")
        print("="*50)
        
        messages = [
            {"role": "system", "content": "You are a helpful math tutor. Always explain your reasoning step by step and check your work."},
            {"role": "user", "content": "What's the derivative of x³ + 2x² - 5x + 3?"}
        ]
        
        return self.chat_completion(messages)
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test cases and return summary"""
        print("\n🚀 Running Router API Test Suite")
        print("="*60)
        
        test_results = {}
        
        # Test cases
        test_cases = [
            ("simple_query", self.test_simple_query),
            ("math_problem", self.test_math_problem),
            ("coding_task", self.test_coding_task),
            ("reasoning_task", self.test_reasoning_task),
            ("conversation", self.test_conversation),
            ("creative_task", self.test_creative_task),
            ("system_prompt", self.test_with_system_prompt)
        ]
        
        total_cost = 0
        total_time = 0
        success_count = 0
        
        for test_name, test_func in test_cases:
            try:
                result = test_func()
                test_results[test_name] = result
                
                if result.get('content'):
                    success_count += 1
                    
                # Accumulate costs and times
                if result.get('router_metadata'):
                    total_cost += result['router_metadata'].get('total_cost', 0)
                total_time += result.get('response_time', 0)
                
            except Exception as e:
                print(f"❌ Test {test_name} failed: {e}")
                test_results[test_name] = {"error": str(e)}
        
        # Print summary
        print("\n" + "="*60)
        print("📊 Test Summary")
        print("="*60)
        print(f"✅ Successful Tests: {success_count}/{len(test_cases)}")
        print(f"💰 Total Cost: ${total_cost:.6f}")
        print(f"⏱️  Total Time: {total_time:.2f}s")
        print(f"📈 Average Time per Test: {total_time/len(test_cases):.2f}s")
        
        return test_results
    
    def get_available_models(self) -> Dict[str, Any]:
        """Get list of available models"""
        try:
            models = self.client.models.list()
            return {"models": [model.id for model in models.data]}
        except Exception as e:
            print(f"❌ Error getting models: {e}")
            return {"error": str(e)}
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get server information"""
        try:
            # Get server root info
            root_url = self.base_url.replace("/v1", "")
            response = requests.get(root_url)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Server returned status {response.status_code}"}
                
        except Exception as e:
            return {"error": str(e)}


def main():
    """Main entry point for test client"""
    parser = argparse.ArgumentParser(description="Router API Test Client")
    parser.add_argument("--url", default="http://localhost:8800/v1", 
                       help="Base URL of the router API")
    parser.add_argument("--test", choices=[
        "simple_query", "math_problem", "coding_task", "reasoning_task",
        "conversation", "creative_task", "system_prompt", "all"
    ], default="all", help="Specific test to run")
    parser.add_argument("--models", action="store_true", 
                       help="List available models")
    parser.add_argument("--info", action="store_true",
                       help="Get server information")
    
    args = parser.parse_args()
    
    # Initialize client
    print(f"🔌 Connecting to Router API: {args.url}")
    client = RouterTestClient(args.url)
    
    # Handle specific requests
    if args.models:
        print("\n📋 Available Models:")
        models = client.get_available_models()
        if "models" in models:
            for model in models["models"]:
                print(f"  - {model}")
        else:
            print(f"❌ Error: {models.get('error', 'Unknown error')}")
        return
    
    if args.info:
        print("\n🔍 Server Information:")
        info = client.get_server_info()
        print(json.dumps(info, indent=2))
        return
    
    # Run tests
    if args.test == "all":
        client.run_all_tests()
    elif args.test == "simple_query":
        client.test_simple_query()
    elif args.test == "math_problem":
        client.test_math_problem()
    elif args.test == "coding_task":
        client.test_coding_task()
    elif args.test == "reasoning_task":
        client.test_reasoning_task()
    elif args.test == "conversation":
        client.test_conversation()
    elif args.test == "creative_task":
        client.test_creative_task()
    elif args.test == "system_prompt":
        client.test_with_system_prompt()


if __name__ == "__main__":
    main()