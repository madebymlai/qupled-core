#!/usr/bin/env python3
"""Test solution display feature in learn command."""

from core.tutor import Tutor
from storage.database import Database

def test_solution_display():
    """Test that solutions are displayed when available."""

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
        print(f"Testing with Core Loop: {core_loop_name} (ID: {core_loop_id})")
        print("-" * 80)

    # Test learning without solutions
    print("\n=== Testing learn without solutions ===")
    tutor = Tutor(language="en")
    response = tutor.learn(
        course_code="B006802",
        core_loop_id=core_loop_id,
        explain_concepts=False,
        adaptive=False,
        show_solutions=False
    )

    print(f"Success: {response.success}")
    print(f"Has solutions in metadata: {response.metadata.get('has_solutions', False)}")
    print(f"Solutions count: {response.metadata.get('solutions_count', 0)}")
    print(f"Content length: {len(response.content)} chars")

    # Test learning with solutions
    print("\n=== Testing learn with solutions ===")
    response = tutor.learn(
        course_code="B006802",
        core_loop_id=core_loop_id,
        explain_concepts=False,
        adaptive=False,
        show_solutions=True
    )

    print(f"Success: {response.success}")
    print(f"Has solutions in metadata: {response.metadata.get('has_solutions', False)}")
    print(f"Solutions count: {response.metadata.get('solutions_count', 0)}")
    print(f"Content length: {len(response.content)} chars")

    # Check if "OFFICIAL SOLUTIONS" appears in content
    if "OFFICIAL SOLUTIONS" in response.content or "SOLUZIONI UFFICIALI" in response.content:
        print("✓ Solution section found in content")
    else:
        print("✗ Solution section NOT found (this is expected if no exercises have solutions)")

    # Show a preview of the content
    print("\n=== Content Preview (last 500 chars) ===")
    print(response.content[-500:])

if __name__ == "__main__":
    test_solution_display()
