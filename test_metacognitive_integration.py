#!/usr/bin/env python3
"""Test metacognitive tips integration in learn command."""

from core.tutor import Tutor
from storage.database import Database

def test_metacognitive_integration():
    """Test that metacognitive tips are displayed correctly."""

    # Get a core loop from ADE course
    with Database() as db:
        cursor = db.conn.execute("""
            SELECT cl.id, cl.name
            FROM core_loops cl
            JOIN topics t ON cl.topic_id = t.id
            LEFT JOIN exercises e ON e.core_loop_id = cl.id
            WHERE t.course_code = 'B006802'
            AND e.id IS NOT NULL
            GROUP BY cl.id, cl.name
            LIMIT 1
        """)
        row = cursor.fetchone()

        if not row:
            print("No core loops with exercises found in ADE course")
            return

        core_loop_id, core_loop_name = row
        print(f"Testing with Core Loop: {core_loop_name}")
        print(f"ID: {core_loop_id}")
        print("-" * 80)

    # Test learning without metacognitive tips
    print("\n=== Test 1: Learn WITHOUT metacognitive tips ===")
    tutor_en = Tutor(language="en")
    response1 = tutor_en.learn(
        course_code="B006802",
        core_loop_id=core_loop_id,
        explain_concepts=False,
        adaptive=False,
        include_metacognitive=False
    )

    print(f"Success: {response1.success}")
    print(f"Includes metacognitive: {response1.metadata.get('includes_metacognitive', False)}")
    print(f"Content length: {len(response1.content)} chars")

    if "LEARNING STRATEGIES" in response1.content:
        print("✗ Unexpected: Found metacognitive section")
    else:
        print("✓ Correct: No metacognitive section")

    # Test learning WITH metacognitive tips (English)
    print("\n=== Test 2: Learn WITH metacognitive tips (English) ===")
    response2 = tutor_en.learn(
        course_code="B006802",
        core_loop_id=core_loop_id,
        explain_concepts=False,
        adaptive=False,
        include_metacognitive=True
    )

    print(f"Success: {response2.success}")
    print(f"Includes metacognitive: {response2.metadata.get('includes_metacognitive', False)}")
    print(f"Content length: {len(response2.content)} chars")

    if "LEARNING STRATEGIES" in response2.content:
        print("✓ Found metacognitive section")
    else:
        print("✗ Missing metacognitive section")

    # Check for key sections
    sections_found = []
    if "problem-solving framework" in response2.content.lower():
        sections_found.append("Framework")
    if "study tips" in response2.content.lower():
        sections_found.append("Study Tips")
    if "self-assessment" in response2.content.lower():
        sections_found.append("Self-Assessment")
    if "retrieval practice" in response2.content.lower():
        sections_found.append("Retrieval Practice")

    print(f"Sections found: {', '.join(sections_found) if sections_found else 'None'}")

    # Test with Italian
    print("\n=== Test 3: Learn WITH metacognitive tips (Italian) ===")
    tutor_it = Tutor(language="it")
    response3 = tutor_it.learn(
        course_code="B006802",
        core_loop_id=core_loop_id,
        explain_concepts=False,
        adaptive=False,
        include_metacognitive=True
    )

    print(f"Success: {response3.success}")
    if "STRATEGIE DI APPRENDIMENTO" in response3.content:
        print("✓ Found Italian metacognitive section")
    else:
        print("✗ Missing Italian metacognitive section")

    # Show a preview of the metacognitive section
    print("\n=== Preview of Metacognitive Section (English) ===")
    if "LEARNING STRATEGIES" in response2.content:
        start = response2.content.find("LEARNING STRATEGIES")
        # Find the end (next section or end of content)
        end = start + 1500
        preview = response2.content[start:end]
        print(preview)
        print("...")

if __name__ == "__main__":
    test_metacognitive_integration()
