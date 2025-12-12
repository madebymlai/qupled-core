#!/usr/bin/env python3
"""
Test script for multi-procedure analyzer functionality.
Tests both new multi-procedure format and backward compatibility with old format.
"""

import json
from core.analyzer import AnalysisResult, ProcedureInfo


def test_new_format_multi_procedure():
    """Test new format with multiple procedures."""
    print("=" * 60)
    print("TEST 1: New format with MULTIPLE procedures")
    print("=" * 60)

    procedures = [
        ProcedureInfo(
            name="Mealy Machine Design",
            type="design",
            steps=["Step 1", "Step 2", "Step 3"],
            point_number=1,
            transformation=None
        ),
        ProcedureInfo(
            name="Mealy to Moore Transformation",
            type="transformation",
            steps=["Step A", "Step B"],
            point_number=2,
            transformation={
                "source_format": "Mealy Machine",
                "target_format": "Moore Machine"
            }
        ),
        ProcedureInfo(
            name="State Minimization",
            type="minimization",
            steps=["Step X", "Step Y", "Step Z"],
            point_number=3,
            transformation=None
        )
    ]

    result = AnalysisResult(
        is_valid_exercise=True,
        is_fragment=False,
        should_merge_with_previous=False,
        topic="Sequential Circuits",
        difficulty="medium",
        variations=["FSM", "State Machines"],
        confidence=0.9,
        procedures=procedures
    )

    print(f"\n✓ Created AnalysisResult with {len(result.procedures)} procedures")
    print(f"\nProcedures:")
    for i, proc in enumerate(result.procedures, 1):
        print(f"  {i}. {proc.name}")
        print(f"     - Type: {proc.type}")
        print(f"     - Point: {proc.point_number}")
        print(f"     - Steps: {len(proc.steps)}")
        if proc.transformation:
            print(f"     - Transformation: {proc.transformation['source_format']} → {proc.transformation['target_format']}")

    print(f"\n✓ Backward compatibility (properties):")
    print(f"  - knowledge_item_name: {result.knowledge_item_name}")
    print(f"  - knowledge_item_id: {result.knowledge_item_id}")
    print(f"  - procedure steps: {len(result.procedure) if result.procedure else 0}")

    assert result.knowledge_item_name == "Mealy Machine Design", "Primary core loop name should be first procedure"
    assert result.knowledge_item_id == "mealy_machine_design", "Primary core loop ID should be normalized"
    assert len(result.procedure) == 3, "Primary procedure should have 3 steps"

    print("\n✓ All assertions passed!")


def test_new_format_single_procedure():
    """Test new format with single procedure."""
    print("\n" + "=" * 60)
    print("TEST 2: New format with SINGLE procedure")
    print("=" * 60)

    procedures = [
        ProcedureInfo(
            name="Karnaugh Map Simplification",
            type="minimization",
            steps=["Draw K-map", "Group terms", "Write minimal SOP"],
            point_number=None,
            transformation=None
        )
    ]

    result = AnalysisResult(
        is_valid_exercise=True,
        is_fragment=False,
        should_merge_with_previous=False,
        topic="Boolean Algebra",
        difficulty="easy",
        variations=["K-maps", "3-variable"],
        confidence=0.85,
        procedures=procedures
    )

    print(f"\n✓ Created AnalysisResult with {len(result.procedures)} procedure")
    print(f"\nProcedure:")
    print(f"  1. {result.procedures[0].name}")
    print(f"     - Type: {result.procedures[0].type}")

    print(f"\n✓ Backward compatibility (properties):")
    print(f"  - knowledge_item_name: {result.knowledge_item_name}")
    print(f"  - knowledge_item_id: {result.knowledge_item_id}")

    assert result.knowledge_item_name == "Karnaugh Map Simplification"
    assert result.knowledge_item_id == "karnaugh_map_simplification"

    print("\n✓ All assertions passed!")


def test_old_format_compatibility():
    """Test backward compatibility with old single-procedure format."""
    print("\n" + "=" * 60)
    print("TEST 3: OLD format (simulated from LLM response)")
    print("=" * 60)

    # Simulate old LLM response format
    old_format_data = {
        "is_valid_exercise": True,
        "is_fragment": False,
        "should_merge_with_previous": False,
        "topic": "Sequential Circuits",
        "knowledge_item_name": "Finite State Machine Design",
        "procedure": ["Define states", "Define transitions", "Draw diagram"],
        "difficulty": "hard",
        "variations": ["Moore machine"],
        "confidence": 0.75
    }

    print("\nSimulated old LLM response:")
    print(json.dumps(old_format_data, indent=2))

    # This is what the parser would create (simulating the conversion logic)
    procedures = [
        ProcedureInfo(
            name=old_format_data["knowledge_item_name"],
            type="other",  # Unknown type in old format
            steps=old_format_data.get("procedure", []),
            point_number=None,
            transformation=None
        )
    ]

    result = AnalysisResult(
        is_valid_exercise=old_format_data.get("is_valid_exercise", True),
        is_fragment=old_format_data.get("is_fragment", False),
        should_merge_with_previous=old_format_data.get("should_merge_with_previous", False),
        topic=old_format_data.get("topic"),
        difficulty=old_format_data.get("difficulty"),
        variations=old_format_data.get("variations", []),
        confidence=old_format_data.get("confidence", 0.5),
        procedures=procedures
    )

    print(f"\n✓ Converted to new format with {len(result.procedures)} procedure")
    print(f"\n✓ Backward compatibility (properties):")
    print(f"  - knowledge_item_name: {result.knowledge_item_name}")
    print(f"  - knowledge_item_id: {result.knowledge_item_id}")
    print(f"  - procedure: {result.procedure}")

    assert result.knowledge_item_name == "Finite State Machine Design"
    assert result.knowledge_item_id == "finite_state_machine_design"
    assert result.procedure == ["Define states", "Define transitions", "Draw diagram"]

    print("\n✓ All assertions passed!")


def test_empty_procedures():
    """Test with empty procedures list."""
    print("\n" + "=" * 60)
    print("TEST 4: Empty procedures list")
    print("=" * 60)

    result = AnalysisResult(
        is_valid_exercise=True,
        is_fragment=False,
        should_merge_with_previous=False,
        topic="Unknown",
        difficulty=None,
        variations=[],
        confidence=0.0,
        procedures=[]
    )

    print(f"\n✓ Created AnalysisResult with {len(result.procedures)} procedures")
    print(f"\n✓ Backward compatibility (should all be None):")
    print(f"  - knowledge_item_name: {result.knowledge_item_name}")
    print(f"  - knowledge_item_id: {result.knowledge_item_id}")
    print(f"  - procedure: {result.procedure}")

    assert result.knowledge_item_name is None
    assert result.knowledge_item_id is None
    assert result.procedure is None

    print("\n✓ All assertions passed!")


def test_transformation_detection():
    """Test transformation type detection."""
    print("\n" + "=" * 60)
    print("TEST 5: Transformation detection and tagging")
    print("=" * 60)

    procedures = [
        ProcedureInfo(
            name="SOP to POS Conversion",
            type="transformation",
            steps=["Apply De Morgan's law", "Distribute terms", "Simplify"],
            point_number=1,
            transformation={
                "source_format": "SOP",
                "target_format": "POS"
            }
        ),
        ProcedureInfo(
            name="Boolean Expression Minimization",
            type="minimization",
            steps=["Use Boolean algebra laws", "Factor out common terms"],
            point_number=2,
            transformation=None
        )
    ]

    result = AnalysisResult(
        is_valid_exercise=True,
        is_fragment=False,
        should_merge_with_previous=False,
        topic="Boolean Algebra",
        difficulty="medium",
        variations=[],
        confidence=0.88,
        procedures=procedures
    )

    print(f"\n✓ Created AnalysisResult with {len(result.procedures)} procedures")

    # Simulate tag generation (like in CLI)
    tags = []
    for proc in result.procedures:
        tags.append(proc.type)
        if proc.transformation:
            src = proc.transformation.get('source_format', '').lower().replace(' ', '_')
            tgt = proc.transformation.get('target_format', '').lower().replace(' ', '_')
            tags.append(f"transform_{src}_to_{tgt}")

    print(f"\n✓ Generated tags: {tags}")

    assert "transformation" in tags
    assert "minimization" in tags
    assert "transform_sop_to_pos" in tags

    print("\n✓ All assertions passed!")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MULTI-PROCEDURE ANALYZER TEST SUITE")
    print("=" * 60)

    try:
        test_new_format_multi_procedure()
        test_new_format_single_procedure()
        test_old_format_compatibility()
        test_empty_procedures()
        test_transformation_detection()

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nSummary:")
        print("  ✓ Multi-procedure support works correctly")
        print("  ✓ Single-procedure support works correctly")
        print("  ✓ Backward compatibility with old format works")
        print("  ✓ Empty procedures handled gracefully")
        print("  ✓ Transformation detection and tagging works")
        print("=" * 60 + "\n")

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}\n")
        raise
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}\n")
        raise
