#!/usr/bin/env python3
"""Test confidence threshold filtering with artificially high threshold to demonstrate filtering."""

import os
import sys

# Verify GROQ_API_KEY is set
if "GROQ_API_KEY" not in os.environ:
    print("Error: GROQ_API_KEY environment variable not set")
    print("Please set it: export GROQ_API_KEY=your_key_here")
    sys.exit(1)

# Set a high threshold to test filtering
os.environ["EXAMINA_MIN_CONFIDENCE"] = "0.85"

from models.llm_manager import LLMManager
from core.analyzer import ExerciseAnalyzer
from storage.database import Database
from config import Config

print(f"Testing Confidence Threshold Filtering with High Threshold")
print(f"=" * 80)
print(f"Minimum confidence threshold: {Config.MIN_ANALYSIS_CONFIDENCE}")
print(f"(Artificially high to demonstrate filtering)")
print(f"=" * 80)

# Get exercises from ADE course
with Database() as db:
    exercises = db.get_exercises_by_course('B006802')

if not exercises:
    print("No exercises found in database for course B006802")
    exit(1)

print(f"\nFound {len(exercises)} exercises in database")
print(f"\nRunning topic and core loop discovery with confidence filtering...\n")

# Initialize analyzer with Groq
llm = LLMManager(provider="groq")
analyzer = ExerciseAnalyzer(llm)

# Run discovery (this will apply confidence filtering)
result = analyzer.discover_topics_and_knowledge_items('B006802')

print("\n" + "=" * 80)
print("RESULTS:")
print("=" * 80)
print(f"Original exercises: {result['original_count']}")
print(f"Merged exercises: {result['merged_count']}")
print(f"Accepted exercises: {result['accepted_count']}")
print(f"Skipped (low confidence): {result['low_confidence_skipped']}")
print(f"\nTopics discovered: {len(result['topics'])}")
print(f"Core loops discovered: {len(result['knowledge_items'])}")

# Show confidence scores for all merged exercises
print("\n" + "-" * 80)
print("All confidence scores:")
for i, ex in enumerate(result['merged_exercises']):
    analysis = ex.get('analysis')
    if analysis:
        skipped = " [SKIPPED]" if ex.get('low_confidence_skipped') else " [ACCEPTED]"
        print(f"  Exercise {i+1}: confidence={analysis.confidence:.2f}{skipped}")

print("\n" + "=" * 80)
print("Test complete!")
