#!/usr/bin/env python3
"""Quick test of async LLM methods."""

import asyncio
import sys
import time
from models.llm_manager import LLMManager

async def test_single_request():
    """Test single async request."""
    print("Testing single async request with DeepSeek...")

    async with LLMManager(provider="deepseek") as llm:
        start = time.time()
        response = await llm.generate_async(
            prompt="What is 2+2? Answer in one word.",
            temperature=0,
            max_tokens=10
        )
        elapsed = time.time() - start

        print(f"  Response: {response.text}")
        print(f"  Success: {response.success}")
        print(f"  Time: {elapsed:.2f}s")

        return response.success

async def test_parallel_requests():
    """Test parallel async requests."""
    print("\nTesting 5 parallel async requests with DeepSeek...")

    async with LLMManager(provider="deepseek") as llm:
        start = time.time()

        # Create 5 concurrent requests
        tasks = [
            llm.generate_async(
                prompt=f"What is {i}+{i}? Answer in one word.",
                temperature=0,
                max_tokens=10
            )
            for i in range(1, 6)
        ]

        # Run all in parallel
        responses = await asyncio.gather(*tasks)
        elapsed = time.time() - start

        print(f"  Completed {len(responses)} requests in {elapsed:.2f}s")
        print(f"  Average: {elapsed/len(responses):.2f}s per request")
        print(f"  Success rate: {sum(1 for r in responses if r.success)}/{len(responses)}")

        # Show responses
        for i, response in enumerate(responses, 1):
            if response.success:
                print(f"    {i}+{i} = {response.text.strip()}")

        return all(r.success for r in responses)

async def main():
    """Run tests."""
    print("=" * 60)
    print("Async LLM Test Suite")
    print("=" * 60)

    try:
        # Test 1: Single request
        success1 = await test_single_request()

        # Test 2: Parallel requests (demonstrates async benefit)
        success2 = await test_parallel_requests()

        print("\n" + "=" * 60)
        if success1 and success2:
            print("✅ All tests passed!")
            return 0
        else:
            print("❌ Some tests failed")
            return 1

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
