"""
Comprehensive tests for detection_utils module.

Tests cover:
- Numbered point detection (various formats)
- Transformation keyword detection (English and Italian)
- Procedure type classification
- Edge cases and error handling
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.detection_utils import (
    detect_numbered_points,
    detect_transformation_keywords,
    classify_procedure_type,
    extract_exercise_metadata,
    _roman_to_int,
)


# ============================================================================
# Test detect_numbered_points()
# ============================================================================


def test_detect_numbered_points_simple():
    """Test basic numeric point detection."""
    text = """
    1. Design Mealy machine
    2. Minimize with implication table
    3. Transform to Moore equivalent
    """
    points = detect_numbered_points(text)

    assert len(points) == 3, f"Expected 3 points, got {len(points)}"
    assert points[0]["point_number"] == 1
    assert points[1]["point_number"] == 2
    assert points[2]["point_number"] == 3
    assert "Design Mealy machine" in points[0]["text"]
    assert "Minimize with implication table" in points[1]["text"]
    assert "Transform to Moore equivalent" in points[2]["text"]
    print("✓ test_detect_numbered_points_simple passed")


def test_detect_numbered_points_letters():
    """Test letter-based point detection."""
    text = """
    a) First step
    b) Second step
    c) Third step
    """
    points = detect_numbered_points(text)

    assert len(points) == 3, f"Expected 3 points, got {len(points)}"
    assert points[0]["point_number"] == "a"
    assert points[1]["point_number"] == "b"
    assert points[2]["point_number"] == "c"
    assert "First step" in points[0]["text"]
    print("✓ test_detect_numbered_points_letters passed")


def test_detect_numbered_points_roman():
    """Test Roman numeral detection."""
    text = """
    I. First section
    II. Second section
    III. Third section
    IV. Fourth section
    """
    points = detect_numbered_points(text)

    assert len(points) == 4, f"Expected 4 points, got {len(points)}"
    assert points[0]["point_number"] == 1  # I
    assert points[1]["point_number"] == 2  # II
    assert points[2]["point_number"] == 3  # III
    assert points[3]["point_number"] == 4  # IV
    print("✓ test_detect_numbered_points_roman passed")


def test_detect_numbered_points_italian():
    """Test Italian-style point detection."""
    text = """
    Punto 1: Disegnare l'automa di Mealy
    Punto 2: Minimizzare con la tabella delle implicazioni
    Esercizio 1.a: Trasformare in Moore
    """
    points = detect_numbered_points(text)

    assert len(points) >= 2, f"Expected at least 2 points, got {len(points)}"
    # Check for "Punto" patterns
    punto_points = [p for p in points if "italian" in p["pattern_type"]]
    assert len(punto_points) >= 2, f"Expected at least 2 Italian-style points"
    print("✓ test_detect_numbered_points_italian passed")


def test_detect_numbered_points_nested():
    """Test nested numbering detection."""
    text = """
    1.1 First sub-point
    1.2 Second sub-point
    2.1 Another sub-point
    """
    points = detect_numbered_points(text)

    assert len(points) == 3, f"Expected 3 points, got {len(points)}"
    assert points[0]["point_number"] == "1.1"
    assert points[1]["point_number"] == "1.2"
    assert points[2]["point_number"] == "2.1"
    print("✓ test_detect_numbered_points_nested passed")


def test_detect_numbered_points_mixed():
    """Test mixed numbering styles."""
    text = """
    1. Main point
    a) Sub-point one
    b) Sub-point two
    2. Another main point
    """
    points = detect_numbered_points(text)

    assert len(points) >= 4, f"Expected at least 4 points, got {len(points)}"
    print("✓ test_detect_numbered_points_mixed passed")


def test_detect_numbered_points_empty():
    """Test with empty text."""
    points = detect_numbered_points("")
    assert len(points) == 0
    print("✓ test_detect_numbered_points_empty passed")


def test_detect_numbered_points_no_points():
    """Test text without numbered points."""
    text = "This is a simple exercise without numbered points."
    points = detect_numbered_points(text)
    assert len(points) == 0
    print("✓ test_detect_numbered_points_no_points passed")


# ============================================================================
# Test detect_transformation_keywords() - English
# ============================================================================


def test_detect_transformation_english_basic():
    """Test basic English transformation detection."""
    text = "Transform the Mealy machine to Moore equivalent"
    results = detect_transformation_keywords(text, language="en")

    assert len(results) > 0, "Expected at least one transformation"
    assert results[0]["type"] == "transformation"
    assert "mealy" in results[0]["source_format"].lower()
    assert "moore" in results[0]["target_format"].lower()
    assert results[0]["confidence"] > 0.7
    print("✓ test_detect_transformation_english_basic passed")


def test_detect_transformation_english_convert():
    """Test 'convert' keyword detection."""
    text = "Convert the DFA into an NFA"
    results = detect_transformation_keywords(text, language="en")

    assert len(results) > 0
    assert "dfa" in results[0]["source_format"].lower()
    assert "nfa" in results[0]["target_format"].lower()
    print("✓ test_detect_transformation_english_convert passed")


def test_detect_transformation_english_arrow():
    """Test arrow notation detection."""
    text = "Mealy → Moore transformation"
    results = detect_transformation_keywords(text, language="en")

    assert len(results) > 0
    assert "mealy" in results[0]["source_format"].lower()
    assert "moore" in results[0]["target_format"].lower()
    assert results[0]["confidence"] >= 0.9  # Arrow notation should have high confidence
    print("✓ test_detect_transformation_english_arrow passed")


def test_detect_transformation_english_derive():
    """Test 'derive' keyword detection."""
    text = "Derive a Moore machine from the given Mealy machine"
    results = detect_transformation_keywords(text, language="en")

    assert len(results) > 0
    # With 'derive', source and target are swapped
    assert "moore" in results[0]["target_format"].lower()
    assert "mealy" in results[0]["source_format"].lower()
    print("✓ test_detect_transformation_english_derive passed")


def test_detect_transformation_english_equivalent():
    """Test 'equivalent' keyword detection."""
    text = "Design a Moore equivalent automaton"
    results = detect_transformation_keywords(text, language="en")

    assert len(results) > 0
    assert "moore" in results[0]["target_format"].lower()
    print("✓ test_detect_transformation_english_equivalent passed")


# ============================================================================
# Test detect_transformation_keywords() - Italian
# ============================================================================


def test_detect_transformation_italian_basic():
    """Test basic Italian transformation detection."""
    text = "Disegnare l'automa di Moore equivalente a quello di Mealy"
    results = detect_transformation_keywords(text, language="it")

    assert len(results) > 0, "Expected at least one transformation"
    # Should detect "equivalente" pattern
    print("✓ test_detect_transformation_italian_basic passed")


def test_detect_transformation_italian_trasformare():
    """Test 'trasformare' keyword detection."""
    text = "Trasformare l'automa di Mealy in un automa di Moore"
    results = detect_transformation_keywords(text, language="it")

    assert len(results) > 0
    assert "mealy" in results[0]["source_format"].lower()
    assert "moore" in results[0]["target_format"].lower()
    print("✓ test_detect_transformation_italian_trasformare passed")


def test_detect_transformation_italian_ricavare():
    """Test 'ricavare' keyword detection."""
    text = "Ricavare un automa di Moore dall'automa di Mealy dato"
    results = detect_transformation_keywords(text, language="it")

    assert len(results) > 0
    # 'ricavare' should swap source/target like 'derive'
    print("✓ test_detect_transformation_italian_ricavare passed")


def test_detect_transformation_italian_arrow():
    """Test arrow notation works in Italian too."""
    text = "Trasformazione Mealy → Moore"
    results = detect_transformation_keywords(text, language="it")

    assert len(results) > 0
    assert "mealy" in results[0]["source_format"].lower()
    assert "moore" in results[0]["target_format"].lower()
    print("✓ test_detect_transformation_italian_arrow passed")


# ============================================================================
# Test detect_transformation_keywords() - Edge Cases
# ============================================================================


def test_detect_transformation_no_keywords():
    """Test text without transformation keywords."""
    text = "Design a simple counter circuit"
    results = detect_transformation_keywords(text, language="en")

    assert len(results) == 0
    print("✓ test_detect_transformation_no_keywords passed")


def test_detect_transformation_empty():
    """Test with empty text."""
    results = detect_transformation_keywords("", language="en")
    assert len(results) == 0
    print("✓ test_detect_transformation_empty passed")


def test_detect_transformation_multiple():
    """Test detection of multiple transformations."""
    text = """
    1. Transform Mealy to Moore
    2. Convert DFA to NFA
    3. Derive regex from automaton
    """
    results = detect_transformation_keywords(text, language="en")

    assert len(results) >= 2, f"Expected at least 2 transformations, got {len(results)}"
    print("✓ test_detect_transformation_multiple passed")


# ============================================================================
# Test classify_procedure_type()
# ============================================================================


def test_classify_procedure_type_design():
    """Test classification of design procedures."""
    text = "Design a Mealy machine that recognizes the pattern"
    proc_type = classify_procedure_type(text, language="en")

    assert proc_type == "design", f"Expected 'design', got '{proc_type}'"
    print("✓ test_classify_procedure_type_design passed")


def test_classify_procedure_type_transformation():
    """Test classification of transformation procedures."""
    text = "Transform the given Mealy machine to Moore equivalent"
    proc_type = classify_procedure_type(text, language="en")

    assert proc_type == "transformation", f"Expected 'transformation', got '{proc_type}'"
    print("✓ test_classify_procedure_type_transformation passed")


def test_classify_procedure_type_minimization():
    """Test classification of minimization procedures."""
    text = "Minimize the automaton using implication table"
    proc_type = classify_procedure_type(text, language="en")

    assert proc_type == "minimization", f"Expected 'minimization', got '{proc_type}'"
    print("✓ test_classify_procedure_type_minimization passed")


def test_classify_procedure_type_verification():
    """Test classification of verification procedures."""
    text = "Verify that the two automata are equivalent"
    proc_type = classify_procedure_type(text, language="en")

    assert proc_type == "verification", f"Expected 'verification', got '{proc_type}'"
    print("✓ test_classify_procedure_type_verification passed")


def test_classify_procedure_type_analysis():
    """Test classification of analysis procedures."""
    text = "Calculate the number of states in the minimal DFA"
    proc_type = classify_procedure_type(text, language="en")

    assert proc_type == "analysis", f"Expected 'analysis', got '{proc_type}'"
    print("✓ test_classify_procedure_type_analysis passed")


def test_classify_procedure_type_implementation():
    """Test classification of implementation procedures."""
    text = "Implement the circuit using logic gates"
    proc_type = classify_procedure_type(text, language="en")

    assert proc_type == "implementation", f"Expected 'implementation', got '{proc_type}'"
    print("✓ test_classify_procedure_type_implementation passed")


def test_classify_procedure_type_italian_design():
    """Test Italian procedure classification - design."""
    text = "Disegnare un automa di Mealy che riconosce il pattern"
    proc_type = classify_procedure_type(text, language="it")

    assert proc_type == "design", f"Expected 'design', got '{proc_type}'"
    print("✓ test_classify_procedure_type_italian_design passed")


def test_classify_procedure_type_italian_transformation():
    """Test Italian procedure classification - transformation."""
    text = "Trasformare l'automa di Mealy in Moore equivalente"
    proc_type = classify_procedure_type(text, language="it")

    assert proc_type == "transformation", f"Expected 'transformation', got '{proc_type}'"
    print("✓ test_classify_procedure_type_italian_transformation passed")


def test_classify_procedure_type_default():
    """Test default classification for ambiguous text."""
    text = "Some random text without clear procedure keywords"
    proc_type = classify_procedure_type(text, language="en")

    # Should default to 'analysis'
    assert proc_type == "analysis", f"Expected 'analysis' (default), got '{proc_type}'"
    print("✓ test_classify_procedure_type_default passed")


# ============================================================================
# Test extract_exercise_metadata()
# ============================================================================


def test_extract_exercise_metadata_complete():
    """Test complete metadata extraction."""
    text = """
    1. Design a Mealy machine for pattern recognition
    2. Minimize using implication table
    3. Transform to Moore equivalent
    """
    metadata = extract_exercise_metadata(text, language="en")

    assert "points" in metadata
    assert "transformations" in metadata
    assert "procedure_types" in metadata
    assert "is_multi_step" in metadata
    assert "step_count" in metadata

    assert len(metadata["points"]) == 3
    assert metadata["is_multi_step"] is True
    assert metadata["step_count"] == 3

    # Check procedure types
    assert 1 in metadata["procedure_types"]
    assert metadata["procedure_types"][1] == "design"
    assert metadata["procedure_types"][2] == "minimization"
    assert metadata["procedure_types"][3] == "transformation"

    print("✓ test_extract_exercise_metadata_complete passed")


def test_extract_exercise_metadata_single_step():
    """Test metadata for single-step exercise."""
    text = "Design a simple counter using D flip-flops"
    metadata = extract_exercise_metadata(text, language="en")

    assert metadata["is_multi_step"] is False
    assert metadata["step_count"] == 0  # No numbered points
    print("✓ test_extract_exercise_metadata_single_step passed")


# ============================================================================
# Test _roman_to_int() helper
# ============================================================================


def test_roman_to_int():
    """Test Roman numeral conversion."""
    assert _roman_to_int("I") == 1
    assert _roman_to_int("II") == 2
    assert _roman_to_int("III") == 3
    assert _roman_to_int("IV") == 4
    assert _roman_to_int("V") == 5
    assert _roman_to_int("VI") == 6
    assert _roman_to_int("IX") == 9
    assert _roman_to_int("X") == 10
    assert _roman_to_int("XII") == 12
    print("✓ test_roman_to_int passed")


# ============================================================================
# Integration Tests
# ============================================================================


def test_integration_realistic_exercise():
    """Test with realistic exercise text."""
    text = """
    Esercizio 1.3

    Dato il seguente automa di Mealy:
    [diagram here]

    1. Minimizzare l'automa usando la tabella delle implicazioni
    2. Trasformare l'automa minimizzato in un automa di Moore equivalente
    3. Verificare che i due automi sono equivalenti
    """

    # Detect points
    points = detect_numbered_points(text)
    assert len(points) >= 3

    # Detect transformations
    transformations = detect_transformation_keywords(text, language="it")
    assert len(transformations) > 0

    # Classify procedures
    for point in points:
        if point["point_number"] in [1, 2, 3]:
            proc_type = classify_procedure_type(point["text"], language="it")
            assert proc_type in ["minimization", "transformation", "verification"]

    # Full metadata
    metadata = extract_exercise_metadata(text, language="it")
    assert metadata["is_multi_step"] is True
    assert metadata["step_count"] >= 3

    print("✓ test_integration_realistic_exercise passed")


# ============================================================================
# Main test runner
# ============================================================================


def run_all_tests():
    """Run all tests and report results."""
    tests = [
        # Numbered points tests
        test_detect_numbered_points_simple,
        test_detect_numbered_points_letters,
        test_detect_numbered_points_roman,
        test_detect_numbered_points_italian,
        test_detect_numbered_points_nested,
        test_detect_numbered_points_mixed,
        test_detect_numbered_points_empty,
        test_detect_numbered_points_no_points,
        # Transformation detection - English
        test_detect_transformation_english_basic,
        test_detect_transformation_english_convert,
        test_detect_transformation_english_arrow,
        test_detect_transformation_english_derive,
        test_detect_transformation_english_equivalent,
        # Transformation detection - Italian
        test_detect_transformation_italian_basic,
        test_detect_transformation_italian_trasformare,
        test_detect_transformation_italian_ricavare,
        test_detect_transformation_italian_arrow,
        # Transformation edge cases
        test_detect_transformation_no_keywords,
        test_detect_transformation_empty,
        test_detect_transformation_multiple,
        # Procedure classification
        test_classify_procedure_type_design,
        test_classify_procedure_type_transformation,
        test_classify_procedure_type_minimization,
        test_classify_procedure_type_verification,
        test_classify_procedure_type_analysis,
        test_classify_procedure_type_implementation,
        test_classify_procedure_type_italian_design,
        test_classify_procedure_type_italian_transformation,
        test_classify_procedure_type_default,
        # Metadata extraction
        test_extract_exercise_metadata_complete,
        test_extract_exercise_metadata_single_step,
        # Helper functions
        test_roman_to_int,
        # Integration
        test_integration_realistic_exercise,
    ]

    passed = 0
    failed = 0
    errors = []

    print("\n" + "=" * 70)
    print("Running Detection Utils Test Suite")
    print("=" * 70 + "\n")

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            failed += 1
            errors.append(f"{test.__name__}: {str(e)}")
            print(f"✗ {test.__name__} FAILED: {str(e)}")
        except Exception as e:
            failed += 1
            errors.append(f"{test.__name__}: {type(e).__name__}: {str(e)}")
            print(f"✗ {test.__name__} ERROR: {type(e).__name__}: {str(e)}")

    print("\n" + "=" * 70)
    print(f"Test Results: {passed} passed, {failed} failed out of {len(tests)} total")
    print("=" * 70)

    if errors:
        print("\nFailed Tests:")
        for error in errors:
            print(f"  - {error}")

    return passed, failed


if __name__ == "__main__":
    passed, failed = run_all_tests()
    sys.exit(0 if failed == 0 else 1)
