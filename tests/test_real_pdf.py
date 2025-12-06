"""
Test exercise splitter with real PDF files using actual LLM.
"""
import sys
import os
from pathlib import Path

# Add examina to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.pdf_processor import PDFProcessor
from core.exercise_splitter import ExerciseSplitter
from models.llm_manager import LLMManager

# LLM provider to use for testing
LLM_PROVIDER = "deepseek"


def test_pdf(pdf_path: str, course_code: str, expected_min: int = 5):
    """Test a single PDF file."""
    print(f"\n{'='*60}")
    print(f"Testing: {Path(pdf_path).name}")
    print("=" * 60)

    # Process PDF
    processor = PDFProcessor()
    pdf_content = processor.process_pdf(Path(pdf_path))

    print(f"  Pages: {len(pdf_content.pages)}")
    print(f"  Total text: {sum(len(p.text) for p in pdf_content.pages)} chars")

    # Initialize LLM with deepseek provider
    llm = LLMManager(provider=LLM_PROVIDER)
    print(f"  LLM: {llm.provider} / {llm.fast_model}")

    # Split with smart splitter
    splitter = ExerciseSplitter()
    exercises = splitter.split_pdf_smart(pdf_content, course_code, llm)

    print(f"\n  Results:")
    print(f"    Exercises found: {len(exercises)}")

    # Count parents vs subs
    parents = [e for e in exercises if not e.is_sub_question]
    subs = [e for e in exercises if e.is_sub_question]
    with_solutions = [e for e in exercises if e.solution]

    print(f"    Parent exercises: {len(parents)}")
    print(f"    Sub-questions: {len(subs)}")
    print(f"    With solutions: {len(with_solutions)}")

    # Show first few exercises
    print(f"\n  Sample exercises:")
    for i, ex in enumerate(exercises[:5]):
        preview = ex.text[:100].replace("\n", " ").strip()
        sol_marker = " [+sol]" if ex.solution else ""
        sub_marker = f" (sub of {ex.parent_exercise_number})" if ex.is_sub_question else ""
        print(f"    {i+1}. [{ex.exercise_number}]{sub_marker}{sol_marker}: {preview}...")

    # Verify minimum
    if len(exercises) >= expected_min:
        print(f"\n  ✓ PASS: Found {len(exercises)} exercises (>= {expected_min})")
        return True
    else:
        print(f"\n  ✗ FAIL: Found only {len(exercises)} exercises (expected >= {expected_min})")
        return False


if __name__ == "__main__":
    TEST_PDFS = [
        # (path, course_code, expected_min_exercises)
        # ADE exam with 2 exercises and 12 sub-questions (14 total)
        ("/home/laimk/git/examina-cloud/test-data/ADE-ESAMI/Prova intermedia 2024-01-29 - SOLUZIONI v4.pdf", "ADE", 10),
        # ADE exam with 4 exercises and 3 sub-questions (7 total)
        ("/home/laimk/git/examina-cloud/test-data/ADE-ESAMI/Compito - Prima Prova Intermedia 10-02-2020 - Soluzioni.pdf", "ADE", 5),
        # AL exam with combined format (1a, 1b, 2a, 2b, etc.)
        ("/home/laimk/git/examina-cloud/test-data/AL-ESAMI/20120612 - appello.pdf", "AL", 3),
        # SO exam (replacing PC PDF which is image-only)
        ("/home/laimk/git/examina-cloud/test-data/SO-ESAMI/SOfebbraio2020.pdf", "SO", 3),
    ]

    passed = 0
    failed = 0

    for pdf_path, course_code, expected_min in TEST_PDFS:
        if Path(pdf_path).exists():
            if test_pdf(pdf_path, course_code, expected_min):
                passed += 1
            else:
                failed += 1
        else:
            print(f"\n⚠ Skipping (not found): {pdf_path}")

    print(f"\n{'='*60}")
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)
