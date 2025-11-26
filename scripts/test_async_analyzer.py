#!/usr/bin/env python3
"""
Test script for async analyzer functionality.
Demonstrates usage of the new async methods.
"""

import asyncio
from core.analyzer import ExerciseAnalyzer
from models.llm_manager import LLMManager
from config import Config

async def test_async_analyzer():
    """Test async analysis methods."""

    # Create sample exercises
    exercises = [
        {
            "id": "test_ex_1",
            "text": "Design a Mealy machine that detects the sequence 101.",
            "analyzed": False
        },
        {
            "id": "test_ex_2",
            "text": "Convert the following DFA to a minimal DFA using the partition algorithm.",
            "analyzed": False
        },
        {
            "id": "test_ex_3",
            "text": "Given a Boolean function F(A,B,C), convert from SOP to POS form.",
            "analyzed": False
        }
    ]

    print("=" * 60)
    print("Testing Async ExerciseAnalyzer")
    print("=" * 60)
    print()

    # Initialize LLM manager with async context manager
    async with LLMManager(provider=Config.LLM_PROVIDER) as llm:
        # Initialize analyzer with LLM manager
        analyzer = ExerciseAnalyzer(llm_manager=llm, language="en")

        print(f"[INFO] Using provider: {Config.LLM_PROVIDER}")
        print(f"[INFO] Using model: {llm.primary_model}")
        print()

        # Test 1: Single async analysis with retry
        print("[TEST 1] Testing _analyze_exercise_with_retry_async()")
        print("-" * 60)
        try:
            result = await analyzer._analyze_exercise_with_retry_async(
                exercise_text=exercises[0]["text"],
                course_name="Computer Architecture",
                previous_exercise=None
            )
            print(f"Result: topic={result.topic}, confidence={result.confidence:.2f}")
            print(f"  Procedures: {[p.name for p in result.procedures]}")
        except Exception as e:
            print(f"ERROR: {e}")
        print()

        # Test 2: Async batch merge
        print("[TEST 2] Testing merge_exercises_async()")
        print("-" * 60)
        try:
            merged = await analyzer.merge_exercises_async(
                exercises=exercises,
                batch_size=2,
                show_progress=True,
                skip_analyzed=False
            )
            print(f"Merged {len(exercises)} exercises into {len(merged)} complete exercises")
            for i, ex in enumerate(merged, 1):
                analysis = ex.get("analysis")
                if analysis:
                    print(f"  {i}. Topic: {analysis.topic}, Confidence: {analysis.confidence:.2f}")
                    print(f"     Procedures: {[p.name for p in analysis.procedures]}")
        except Exception as e:
            print(f"ERROR: {e}")
        print()

    print("=" * 60)
    print("Test complete!")
    print("=" * 60)

def main():
    """Run async test."""
    try:
        asyncio.run(test_async_analyzer())
    except KeyboardInterrupt:
        print("\n[INFO] Test interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
