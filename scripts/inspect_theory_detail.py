#!/usr/bin/env python3
"""Get detailed view of theory-heavy exercises."""

import sqlite3
import json
from pathlib import Path

db_path = Path(__file__).parent / "data" / "examina.db"

with sqlite3.connect(str(db_path)) as conn:
    conn.row_factory = sqlite3.Row

    # Get the AL exercise that mentions "definizione" (definition)
    cursor = conn.execute("""
        SELECT *
        FROM exercises
        WHERE course_code = 'B006807'
        AND (text LIKE '%deﬁnizione%' OR text LIKE '%definizione%' OR text LIKE '%Deﬁnire%')
        LIMIT 3
    """)

    print("=== LINEAR ALGEBRA THEORY QUESTIONS ===\n")
    for i, row in enumerate(cursor, 1):
        print(f"\n{'='*80}")
        print(f"Exercise {i}: {row['id'][:50]}...")
        print(f"{'='*80}")
        print(f"Difficulty: {row['difficulty']}")
        print(f"Topic ID: {row['topic_id']}")
        print(f"Core Loop: {row['knowledge_item_id']}")
        print(f"Tags: {row['tags']}")
        print(f"\nFull Text:")
        print(row['text'])
        print(f"\nMetadata: {row['analysis_metadata']}")

    # Get PC exercises with "proprietà" (properties) - theory about concurrency
    cursor = conn.execute("""
        SELECT *
        FROM exercises
        WHERE course_code = 'B018757'
        AND (text LIKE '%proprietà%' OR text LIKE '%property%' OR text LIKE '%assioma%' OR text LIKE '%axiom%')
        LIMIT 2
    """)

    print("\n\n=== CONCURRENT PROGRAMMING THEORY QUESTIONS ===\n")
    for i, row in enumerate(cursor, 1):
        print(f"\n{'='*80}")
        print(f"Exercise {i}: {row['id'][:50]}...")
        print(f"{'='*80}")
        print(f"Difficulty: {row['difficulty']}")
        print(f"Topic ID: {row['topic_id']}")
        print(f"Core Loop: {row['knowledge_item_id']}")
        print(f"Tags: {row['tags']}")
        print(f"\nFull Text:")
        print(row['text'][:800])
        print("..." if len(row['text']) > 800 else "")
        print(f"\nMetadata: {row['analysis_metadata']}")
