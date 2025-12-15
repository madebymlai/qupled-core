#!/usr/bin/env python3
"""Quick database inspection script."""

import sqlite3
from pathlib import Path

db_path = Path(__file__).parent / "data" / "examina.db"

with sqlite3.connect(str(db_path)) as conn:
    conn.row_factory = sqlite3.Row

    print("=== COURSES ===")
    cursor = conn.execute("SELECT code, name FROM courses")
    for row in cursor:
        print(f"{row['code']}: {row['name']}")

    print("\n=== EXERCISES COUNT BY COURSE ===")
    cursor = conn.execute(
        "SELECT course_code, COUNT(*) as count FROM exercises GROUP BY course_code"
    )
    for row in cursor:
        print(f"{row['course_code']}: {row['count']} exercises")

    print("\n=== EXERCISES TABLE SCHEMA ===")
    cursor = conn.execute("PRAGMA table_info(exercises)")
    for row in cursor:
        print(f"{row[1]}: {row[2]}")

    print("\n=== SAMPLE EXERCISE (ADE) ===")
    cursor = conn.execute("""
        SELECT id, text, difficulty, topic_id, knowledge_item_id, tags, analysis_metadata
        FROM exercises
        WHERE course_code = 'ADE'
        LIMIT 1
    """)
    row = cursor.fetchone()
    if row:
        print(f"ID: {row['id'][:50]}")
        print(f"Text: {row['text'][:200]}...")
        print(f"Difficulty: {row['difficulty']}")
        print(f"Topic ID: {row['topic_id']}")
        print(f"Core Loop ID: {row['knowledge_item_id']}")
        print(f"Tags: {row['tags']}")
        print(
            f"Metadata: {row['analysis_metadata'][:200] if row['analysis_metadata'] else None}..."
        )

    print("\n=== TOPICS BY COURSE ===")
    cursor = conn.execute("""
        SELECT c.code, c.name as course_name, t.name as topic_name, COUNT(e.id) as exercise_count
        FROM courses c
        LEFT JOIN topics t ON c.code = t.course_code
        LEFT JOIN exercises e ON t.id = e.topic_id
        GROUP BY c.code, t.id
        ORDER BY c.code, topic_name
    """)
    current_course = None
    for row in cursor:
        if row["code"] != current_course:
            print(f"\n{row['code']} ({row['course_name']}):")
            current_course = row["code"]
        if row["topic_name"]:
            print(f"  - {row['topic_name']}: {row['exercise_count']} exercises")
