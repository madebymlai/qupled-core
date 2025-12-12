#!/usr/bin/env python3
"""Debug analyzer to see what LLM returns for each exercise."""

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

# Get first few exercises from ADE
with Database() as db:
    exercises = db.get_exercises_by_course('B006802')[:5]

llm = LLMManager(provider="groq")
analyzer = ExerciseAnalyzer(llm)

for i, ex in enumerate(exercises):
    print(f"\n{'='*80}")
    print(f"Exercise {i+1}: {ex['id']}")
    print(f"Text preview: {ex['text'][:150]}...")
    print('='*80)

    # Analyze
    analysis = analyzer.analyze_exercise(
        ex['text'],
        "Computer Architecture",
        None
    )

    print(f"is_valid_exercise: {analysis.is_valid_exercise}")
    print(f"is_fragment: {analysis.is_fragment}")
    print(f"topic: {analysis.topic}")
    print(f"knowledge_item_name: {analysis.knowledge_item_name}")
    print(f"knowledge_item_id: {analysis.knowledge_item_id}")
    print(f"difficulty: {analysis.difficulty}")
    print(f"confidence: {analysis.confidence}")
    print(f"procedure steps: {len(analysis.procedure or [])}")
