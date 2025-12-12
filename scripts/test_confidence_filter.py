#!/usr/bin/env python3
"""Test confidence threshold filtering for exercise analysis."""

import os
import sys

# Verify GROQ_API_KEY is set
if "GROQ_API_KEY" not in os.environ:
    print("Error: GROQ_API_KEY environment variable not set")
    print("Please set it: export GROQ_API_KEY=your_key_here")
    sys.exit(1)

from models.llm_manager import LLMManager
from core.analyzer import ExerciseAnalyzer
from storage.database import Database
from config import Config

print(f"Testing Confidence Threshold Filtering")
print(f"=" * 80)
print(f"Minimum confidence threshold: {Config.MIN_ANALYSIS_CONFIDENCE}")
print(f"=" * 80)

# Initialize database to ensure migrations run
with Database() as db:
    db.initialize()

# Get exercises from ADE course
with Database() as db:
    exercises = db.get_exercises_by_course('B006802')

if not exercises:
    print("No exercises found in database for course B006802")
    print("Please run PDF ingestion first.")
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

if result['topics']:
    print("\n" + "-" * 80)
    print("Topics found:")
    for topic_name, topic_data in result['topics'].items():
        print(f"  - {topic_name}: {topic_data['exercise_count']} exercises")

if result['knowledge_items']:
    print("\n" + "-" * 80)
    print("Core loops found:")
    for loop_id, loop_data in result['knowledge_items'].items():
        print(f"  - {loop_data['name']}: {loop_data['exercise_count']} exercises")

# Show some examples of confidence scores from merged exercises
print("\n" + "-" * 80)
print("Sample confidence scores from merged exercises:")
for i, ex in enumerate(result['merged_exercises'][:10]):
    analysis = ex.get('analysis')
    if analysis:
        skipped = " [SKIPPED]" if ex.get('low_confidence_skipped') else ""
        print(f"  Exercise {i+1}: confidence={analysis.confidence:.2f}{skipped}")

print("\n" + "=" * 80)
print("Test complete!")
