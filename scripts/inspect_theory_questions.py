#!/usr/bin/env python3
"""Inspect exercises to identify theory questions."""

import sqlite3
import json
from pathlib import Path

db_path = Path(__file__).parent / "data" / "examina.db"

# Look for potential theory questions in each course
courses = [
    ("B006802", "ADE - Computer Architecture"),
    ("B006807", "AL - Linear Algebra"),
    ("B018757", "PC - Concurrent Programming")
]

with sqlite3.connect(str(db_path)) as conn:
    conn.row_factory = sqlite3.Row

    for course_code, course_name in courses:
        print(f"\n{'='*80}")
        print(f"{course_name}")
        print(f"{'='*80}")

        # Get a few sample exercises
        cursor = conn.execute("""
            SELECT id, text, difficulty, topic_id, knowledge_item_id, tags, analysis_metadata
            FROM exercises
            WHERE course_code = ?
            LIMIT 5
        """, (course_code,))

        for i, row in enumerate(cursor, 1):
            print(f"\n--- Exercise {i} ---")
            print(f"ID: {row['id'][:50]}...")
            print(f"Text: {row['text'][:300]}...")
            print(f"Difficulty: {row['difficulty']}")
            print(f"Tags: {row['tags']}")

            # Look for keywords suggesting theory questions
            text_lower = row['text'].lower()
            keywords = {
                'definition': ['definisci', 'definizione', 'define', 'definition', 'cos\'è', 'what is'],
                'theorem': ['teorema', 'theorem', 'dimostrazione', 'proof', 'dimostra'],
                'axiom': ['assioma', 'axiom', 'postulato', 'postulate', 'proprietà', 'property'],
                'explanation': ['spiega', 'explain', 'illustra', 'illustrate', 'descrivi', 'describe'],
                'concept': ['perché', 'why', 'come funziona', 'how does', 'cosa significa', 'what does'],
            }

            detected = []
            for category, words in keywords.items():
                for word in words:
                    if word in text_lower:
                        detected.append(category)
                        break

            if detected:
                print(f"THEORY INDICATORS: {', '.join(set(detected))}")
            else:
                print("THEORY INDICATORS: None (likely procedural)")

        # Look at core loops to understand what types exist
        print(f"\n--- Core Loops in {course_code} ---")
        cursor = conn.execute("""
            SELECT DISTINCT cl.id, cl.name, COUNT(e.id) as ex_count
            FROM knowledge_items cl
            JOIN topics t ON cl.topic_id = t.id
            LEFT JOIN exercises e ON cl.id = e.knowledge_item_id
            WHERE t.course_code = ?
            GROUP BY cl.id
            ORDER BY ex_count DESC
            LIMIT 5
        """, (course_code,))

        for row in cursor:
            print(f"  - {row['name']}: {row['ex_count']} exercises")
