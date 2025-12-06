"""
Test exercise splitter with LLM-provided regex patterns.

Tests:
1. Real PDFs from each course
2. Synthetic edge cases (Cyrillic, Greek, combined format, bullets)
"""
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock

# Add examina to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.exercise_splitter import (
    _find_all_markers,
    _build_hierarchy,
    MarkerPattern,
    MarkerType,
)


@dataclass
class TestCase:
    """Test case for exercise splitter."""
    name: str
    text: str
    pattern: MarkerPattern
    expected_parents: int
    expected_subs: int
    expected_total: int


# Test cases for various formats
TEST_CASES = [
    # Latin lettered sub-markers (current Italian format)
    TestCase(
        name="Italian exam with numbered subs",
        text="""
Esercizio 1
Dato un sistema, calcolare:
1) La prima cosa
2) La seconda cosa
3) La terza cosa

Esercizio 2
Un altro problema:
1) Punto uno
2) Punto due

Soluzione
1) Risposta uno
2) Risposta due
""",
        pattern=MarkerPattern(
            exercise_pattern=r"Esercizio\s+(\d+)",
            sub_pattern=r"(\d+)\s*\)",
            solution_pattern="Soluzione",
        ),
        expected_parents=2,
        expected_subs=5,  # 3 + 2 (excluding solutions)
        expected_total=7,
    ),

    # Cyrillic lettered sub-markers
    TestCase(
        name="Russian exam with Cyrillic letters",
        text="""
Задание 1
Решите задачу:
а) Первый пункт
б) Второй пункт
в) Третий пункт

Задание 2
Другая задача:
а) Пункт один
б) Пункт два
""",
        pattern=MarkerPattern(
            exercise_pattern=r"Задание\s+(\d+)",
            sub_pattern=r"([а-я])\s*\)",
            solution_pattern=None,
        ),
        expected_parents=2,
        expected_subs=5,
        expected_total=7,
    ),

    # Greek lettered sub-markers
    TestCase(
        name="Greek exam with Greek letters",
        text="""
Άσκηση 1
Λύστε την εξίσωση:
α) Πρώτο
β) Δεύτερο
γ) Τρίτο

Άσκηση 2
Άλλη άσκηση:
α) Ένα
β) Δύο
""",
        pattern=MarkerPattern(
            exercise_pattern=r"Άσκηση\s+(\d+)",
            sub_pattern=r"([α-ω])\s*\)",
            solution_pattern=None,
        ),
        expected_parents=2,
        expected_subs=5,
        expected_total=7,
    ),

    # Combined format (1a, 1b, 2a, 2b) - no keyword, just combined markers
    # For this format, exercise_pattern should match a dummy that won't match,
    # and sub_pattern captures both parent number and sub letter
    TestCase(
        name="Combined format 1a), 1b), 2a)",
        text="""
This exam uses combined numbering.

1a) First question part a
1b) First question part b
1c) First question part c

2a) Second question part a
2b) Second question part b

Answer: See solutions below.
""",
        pattern=MarkerPattern(
            exercise_pattern=r"NOMATCH_PLACEHOLDER_ZZZZZ",  # Won't match anything - combined format has no separate parents
            sub_pattern=r"(\d+)([a-z])\s*\)",  # 2 groups: parent + sub
            solution_pattern="Answer",
        ),
        expected_parents=0,  # No separate parent markers
        expected_subs=5,  # All are combined sub-markers
        expected_total=5,
    ),

    # Bulleted format
    TestCase(
        name="Bulleted sub-questions",
        text="""
Exercise 1
Answer the following:
- First bullet point
- Second bullet point
- Third bullet point

Exercise 2
More questions:
• Fourth bullet
• Fifth bullet
""",
        pattern=MarkerPattern(
            exercise_pattern=r"Exercise\s+(\d+)",
            sub_pattern=r"[-•]\s+",
            solution_pattern=None,
        ),
        expected_parents=2,
        expected_subs=5,
        expected_total=7,
    ),

    # Roman numerals
    TestCase(
        name="Roman numeral sub-questions",
        text="""
Problem 1
Calculate:
i) First part
ii) Second part
iii) Third part

Problem 2
Prove:
i) First proof
ii) Second proof
""",
        pattern=MarkerPattern(
            exercise_pattern=r"Problem\s+(\d+)",
            sub_pattern=r"([ivx]+)\s*\)",
            solution_pattern=None,
        ),
        expected_parents=2,
        expected_subs=5,
        expected_total=7,
    ),

    # Dot delimiter
    TestCase(
        name="Dot delimiter sub-questions",
        text="""
Question 1
Answer these:
a. First item
b. Second item
c. Third item

Question 2
More items:
a. Item one
b. Item two
""",
        pattern=MarkerPattern(
            exercise_pattern=r"Question\s+(\d+)",
            sub_pattern=r"([a-z])\s*\.",
            solution_pattern=None,
        ),
        expected_parents=2,
        expected_subs=5,
        expected_total=7,
    ),

    # Parenthesized markers
    TestCase(
        name="Parenthesized (a), (b) format",
        text="""
Exercise 1
Solve:
(a) Part a
(b) Part b
(c) Part c

Exercise 2
Evaluate:
(a) First
(b) Second
""",
        pattern=MarkerPattern(
            exercise_pattern=r"Exercise\s+(\d+)",
            sub_pattern=r"\(([a-z])\)",
            solution_pattern=None,
        ),
        expected_parents=2,
        expected_subs=5,
        expected_total=7,
    ),

    # No sub-markers
    TestCase(
        name="No sub-questions",
        text="""
Esercizio 1
Prima domanda completa.

Esercizio 2
Seconda domanda completa.

Esercizio 3
Terza domanda completa.
""",
        pattern=MarkerPattern(
            exercise_pattern=r"Esercizio\s+(\d+)",
            sub_pattern=None,
            solution_pattern=None,
        ),
        expected_parents=3,
        expected_subs=0,
        expected_total=3,
    ),

]

# Special test case for restart detection - requires _build_hierarchy
RESTART_TEST = TestCase(
    name="Numbered subs with page header restart",
    text="""
Esercizio 1
Calcolare:
1) Prima cosa
2) Seconda cosa
3) Terza cosa
4) Quarta cosa

--- Page Break ---
Instructions:
1) Write your name
2) Show your work
3) Check answers

Esercizio 2
Altro problema:
1) Un punto
2) Due punti
""",
    pattern=MarkerPattern(
        exercise_pattern=r"Esercizio\s+(\d+)",
        sub_pattern=r"(\d+)\s*\)",
        solution_pattern=None,
    ),
    expected_parents=2,
    # After _build_hierarchy restart detection:
    # Ex1: 4 subs (1,2,3,4), Ex2: 2 subs (1,2)
    # The restart sequence (1,2,3 after 4) is filtered in _build_hierarchy
    expected_subs=6,
    expected_total=8,
)


def test_pattern_matching():
    """Test pattern matching for all test cases."""
    print("=" * 60)
    print("EXERCISE SPLITTER PATTERN TESTS")
    print("=" * 60)

    passed = 0
    failed = 0

    for tc in TEST_CASES:
        print(f"\n--- {tc.name} ---")

        # Find markers
        markers, solution_ranges = _find_all_markers(tc.text, tc.pattern)

        # Count by type
        parents = [m for m in markers if m.marker_type == MarkerType.PARENT]
        subs = [m for m in markers if m.marker_type == MarkerType.SUB]

        parent_count = len(parents)
        sub_count = len(subs)
        total_count = len(markers)

        # Check results
        parent_ok = parent_count == tc.expected_parents
        sub_ok = sub_count == tc.expected_subs
        total_ok = total_count == tc.expected_total

        if parent_ok and sub_ok and total_ok:
            print(f"  ✓ PASS: {parent_count} parents, {sub_count} subs, {total_count} total")
            passed += 1
        else:
            print(f"  ✗ FAIL:")
            if not parent_ok:
                print(f"    Parents: got {parent_count}, expected {tc.expected_parents}")
            if not sub_ok:
                print(f"    Subs: got {sub_count}, expected {tc.expected_subs}")
            if not total_ok:
                print(f"    Total: got {total_count}, expected {tc.expected_total}")

            # Debug: show found markers
            print(f"    Found markers:")
            for m in markers[:10]:  # Show first 10
                print(f"      {m.marker_type.value}: '{m.marker_text.strip()[:30]}' num={m.number}")
            if len(markers) > 10:
                print(f"      ... and {len(markers) - 10} more")

            failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


def test_restart_detection():
    """Test restart detection filters page headers."""
    print("\n" + "=" * 60)
    print("RESTART DETECTION TEST")
    print("=" * 60)

    tc = RESTART_TEST
    print(f"\n--- {tc.name} ---")

    # Find markers (raw - includes restart sequences)
    markers, solution_ranges = _find_all_markers(tc.text, tc.pattern)

    print(f"  Raw markers found: {len(markers)}")

    # Build hierarchy (applies restart detection)
    hierarchy = _build_hierarchy(markers, tc.text)

    # Count actual exercises after hierarchy building
    total_exercises = 0
    for root in hierarchy:
        total_exercises += 1  # Count parent
        total_exercises += len(root.children)  # Count children

    print(f"  After hierarchy: {len(hierarchy)} parents, {total_exercises} total exercises")

    # Check: should have 2 parents with 4+2=6 subs = 8 total
    # But hierarchy returns roots only, so we need to count differently
    parents_ok = len(hierarchy) == tc.expected_parents
    children_count = sum(len(r.children) for r in hierarchy)
    children_ok = children_count == tc.expected_subs

    if parents_ok and children_ok:
        print(f"  ✓ PASS: {len(hierarchy)} parents, {children_count} children")
        return True
    else:
        print(f"  ✗ FAIL:")
        print(f"    Parents: got {len(hierarchy)}, expected {tc.expected_parents}")
        print(f"    Children: got {children_count}, expected {tc.expected_subs}")
        for root in hierarchy:
            print(f"    - Parent {root.marker.number}: {len(root.children)} children")
            for child in root.children:
                print(f"      - Sub {child.marker.number}")
        return False


def test_hierarchy_building():
    """Test hierarchy building (parent-child relationships)."""
    print("\n" + "=" * 60)
    print("HIERARCHY BUILDING TESTS")
    print("=" * 60)

    # Simple hierarchy test
    text = """
Esercizio 1
Context for exercise 1.
1) Sub question one
2) Sub question two

Esercizio 2
Context for exercise 2.
1) Another sub
"""
    pattern = MarkerPattern(
        exercise_pattern=r"Esercizio\s+(\d+)",
        sub_pattern=r"(\d+)\s*\)",
        solution_pattern=None,
    )

    markers, _ = _find_all_markers(text, pattern)
    hierarchy = _build_hierarchy(markers, text)

    print(f"\nHierarchy test:")
    print(f"  Root exercises: {len(hierarchy)}")

    for root in hierarchy:
        print(f"  - Exercise {root.marker.number}: {len(root.children)} children")
        for child in root.children:
            print(f"    - Sub {child.marker.number}")

    # Verify structure
    if len(hierarchy) == 2 and len(hierarchy[0].children) == 2 and len(hierarchy[1].children) == 1:
        print("  ✓ PASS: Correct hierarchy structure")
        return True
    else:
        print("  ✗ FAIL: Incorrect hierarchy")
        return False


if __name__ == "__main__":
    import sys

    pattern_ok = test_pattern_matching()
    restart_ok = test_restart_detection()
    hierarchy_ok = test_hierarchy_building()

    all_ok = pattern_ok and restart_ok and hierarchy_ok
    if all_ok:
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed")
        sys.exit(1)
