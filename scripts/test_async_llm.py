#!/usr/bin/env python3
"""
Test script for async LLM methods.
This is a simple test to verify the async implementation works.
"""

import asyncio
from models.llm_manager import LLMManager
from config import Config


async def test_async_anthropic():
    """Test async Anthropic generation."""
    print("Testing Anthropic async generation...")

    async with LLMManager(provider="anthropic") as llm:
        response = await llm.generate_async(
            prompt="Say hello in one word",
            temperature=0.7,
            max_tokens=10
        )

        if response.success:
            print(f"Success! Response: {response.text}")
            print(f"Model: {response.model}")
            if response.metadata:
                print(f"Usage: {response.metadata.get('usage')}")
        else:
            print(f"Error: {response.error}")

    print()


async def test_async_groq():
    """Test async Groq generation."""
    print("Testing Groq async generation...")

    async with LLMManager(provider="groq") as llm:
        response = await llm.generate_async(
            prompt="Say hello in one word",
            temperature=0.7,
            max_tokens=10
        )

        if response.success:
            print(f"Success! Response: {response.text}")
            print(f"Model: {response.model}")
            if response.metadata:
                print(f"Usage: {response.metadata.get('usage')}")
        else:
            print(f"Error: {response.error}")

    print()


async def test_async_deepseek():
    """Test async DeepSeek generation."""
    print("Testing DeepSeek async generation...")

    async with LLMManager(provider="deepseek") as llm:
        response = await llm.generate_async(
            prompt="Say hello in one word",
            temperature=0.7,
            max_tokens=10
        )

        if response.success:
            print(f"Success! Response: {response.text}")
            print(f"Model: {response.model}")
            if response.metadata:
                print(f"Usage: {response.metadata.get('usage')}")
        else:
            print(f"Error: {response.error}")

    print()


async def test_parallel_requests():
    """Test parallel async requests."""
    print("Testing parallel async requests...")

    # Determine which provider to use
    provider = Config.LLM_PROVIDER

    async with LLMManager(provider=provider) as llm:
        # Create 3 concurrent requests
        tasks = [
            llm.generate_async(prompt=f"Count to {i}", max_tokens=20)
            for i in range(1, 4)
        ]

        # Run all tasks concurrently
        responses = await asyncio.gather(*tasks)

        for i, response in enumerate(responses, 1):
            if response.success:
                print(f"Request {i} succeeded: {response.text[:50]}...")
            else:
                print(f"Request {i} failed: {response.error}")

    print()


async def main():
    """Run all tests."""
    print("=" * 60)
    print("Async LLM Manager Test Suite")
    print("=" * 60)
    print()

    # Test based on configured provider
    if Config.LLM_PROVIDER == "anthropic" and Config.ANTHROPIC_API_KEY:
        await test_async_anthropic()
    elif Config.LLM_PROVIDER == "groq" and Config.GROQ_API_KEY:
        await test_async_groq()
    elif Config.LLM_PROVIDER == "deepseek" and Config.DEEPSEEK_API_KEY:
        await test_async_deepseek()
    else:
        print(f"Provider '{Config.LLM_PROVIDER}' not configured or no API key found.")
        print("Please set API keys in .env file.")
        return

    # Test parallel requests
    await test_parallel_requests()

    print("=" * 60)
    print("Tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
