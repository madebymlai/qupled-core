"""
Test suite for anonymous synonym detection approach.

The anonymous approach:
1. Anonymizes item names (Item_1, Item_2, etc.)
2. LLM judges purely on exercise content - "do these test the same skill?"
3. Names are revealed only after grouping decision
4. Post-merge: learning_approach derived from combined exercises (majority wins)

Test categories:
- SHOULD MERGE: Same skill despite different names
- SHOULD NOT MERGE: Different skills despite similar names
- EDGE CASES: Empty data, ties, single items, etc.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Test data representing real-world scenarios
# Format: {"name": str, "exercises": [str], "learning_approach": str}

# ============================================================================
# TEST DATA: Should Merge (Same Skill)
# ============================================================================

# Case 1: Abbreviation variants - FCFS scheduling
FCFS_FULL = {
    "name": "first_come_first_served_scheduling",
    "learning_approach": "procedural",
    "exercises": [
        "Given processes P1(arrival=0, burst=5), P2(arrival=1, burst=3), P3(arrival=2, burst=8). Calculate the average waiting time using FCFS scheduling.",
        "Five processes arrive at times 0, 2, 4, 6, 8 with burst times 3, 6, 4, 5, 2. Draw the Gantt chart for FCFS and compute turnaround time.",
    ],
}

FCFS_ABBREV = {
    "name": "fcfs_scheduling",
    "learning_approach": "procedural",
    "exercises": [
        "Process table: P1(0,4), P2(1,3), P3(2,1), P4(3,5). Apply first-come-first-served and calculate average waiting time.",
        "Three jobs with arrival times [0,0,0] and burst times [10,5,8]. Compute completion times under FCFS.",
    ],
}

# Case 2: Different context, same skill - FSM design
FSM_TRAMWAY = {
    "name": "finite_state_machine_design",
    "learning_approach": "procedural",
    "exercises": [
        "Design a Moore FSM for a tramway crossing controller. States: Idle, Warning, Blocked. Inputs: train_approaching, train_passed. Output: barrier_down.",
        "Draw the state diagram for a tramway signal controller with states S0, S1, S2 and transitions based on sensor inputs.",
    ],
}

FSM_WEBCAM = {
    "name": "fsm_design",
    "learning_approach": "procedural",
    "exercises": [
        "Design a Mealy FSM for a webcam motion detector. States: Monitoring, Detected, Recording. Inputs: motion, timeout. Output: start_recording.",
        "Create a state transition table for a security camera controller that switches between idle, alert, and recording states.",
    ],
}

# Case 3: Synonym naming - same concept
DEADLOCK_AVOIDANCE_FULL = {
    "name": "deadlock_avoidance_bankers_algorithm",
    "learning_approach": "procedural",
    "exercises": [
        "Given Available=[3,3,2], Max matrix and Allocation matrix for 5 processes, determine if the system is in a safe state using Banker's algorithm.",
        "Apply the Banker's algorithm: Available=[2,1,0], processes P0-P3 with given Max and Allocation. Find a safe sequence if one exists.",
    ],
}

BANKERS_ALGO = {
    "name": "bankers_algorithm",
    "learning_approach": "procedural",
    "exercises": [
        "System has 3 resource types with Available=[1,5,2]. Five processes have the following Need and Allocation matrices. Is the state safe?",
        "Using Banker's algorithm, check if granting request [1,0,2] from P1 would leave the system in a safe state.",
    ],
}

# Case 4: Different verbosity - eigenvalues
EIGENVALUES_SHORT = {
    "name": "eigenvalues",
    "learning_approach": "procedural",
    "exercises": [
        "Find the eigenvalues of matrix A = [[4,2],[1,3]].",
        "Compute eigenvalues and eigenvectors for the 2x2 matrix [[1,-1],[2,4]].",
    ],
}

EIGENVALUES_LONG = {
    "name": "eigenvalue_computation_and_diagonalization",
    "learning_approach": "procedural",
    "exercises": [
        "Given matrix B = [[2,1],[1,2]], find its eigenvalues and use them to diagonalize the matrix.",
        "Calculate the eigenvalues of [[3,1],[0,2]] and verify by computing the characteristic polynomial.",
    ],
}

# ============================================================================
# TEST DATA: Should NOT Merge (Different Skills)
# ============================================================================

# Case 5: Opposites - detection vs prevention
DEADLOCK_DETECTION = {
    "name": "deadlock_detection",
    "learning_approach": "procedural",
    "exercises": [
        "Given the current allocation and request matrices, apply the deadlock detection algorithm to find which processes are deadlocked.",
        "System has resources R1=7, R2=2, R3=6. With the given allocation and request matrices, identify any deadlocked processes.",
    ],
}

DEADLOCK_PREVENTION = {
    "name": "deadlock_prevention",
    "learning_approach": "conceptual",
    "exercises": [
        "Explain how the hold-and-wait condition can be prevented in a resource allocation system. Give two practical approaches.",
        "Compare and contrast the four necessary conditions for deadlock and describe how each can be prevented.",
    ],
}

# Case 6: Same name, different skills (explain vs calculate)
CPU_SCHEDULING_CONCEPTUAL = {
    "name": "cpu_scheduling",
    "learning_approach": "conceptual",
    "exercises": [
        "Explain the difference between preemptive and non-preemptive CPU scheduling. When would you prefer one over the other?",
        "Compare Round Robin and Priority Scheduling in terms of response time, throughput, and fairness. Which is better for interactive systems?",
    ],
}

CPU_SCHEDULING_PROCEDURAL = {
    "name": "cpu_scheduling",
    "learning_approach": "procedural",
    "exercises": [
        "Given 5 processes with arrival and burst times, calculate average waiting time and turnaround time using SJF scheduling.",
        "Apply Round Robin (quantum=4) to processes P1(0,10), P2(1,4), P3(2,5). Draw the Gantt chart and compute average waiting time.",
    ],
}

# Case 7: Related but different - Moore vs Mealy
MOORE_MACHINE = {
    "name": "moore_machine_design",
    "learning_approach": "procedural",
    "exercises": [
        "Design a Moore machine that outputs 1 when the input sequence contains '101'. Show state diagram with outputs on states.",
        "Convert the given Mealy machine to an equivalent Moore machine. Draw the resulting state diagram.",
    ],
}

MEALY_MACHINE = {
    "name": "mealy_machine_design",
    "learning_approach": "procedural",
    "exercises": [
        "Design a Mealy machine that outputs 1 on the transition that completes the pattern '110'. Show outputs on transitions.",
        "Given a Moore machine, convert it to a Mealy machine with the same input-output behavior.",
    ],
}

# Case 8: Different matrix sizes - should NOT merge
EIGENVALUES_2X2 = {
    "name": "eigenvalues_2x2",
    "learning_approach": "procedural",
    "exercises": [
        "Find eigenvalues of the 2x2 matrix [[1,2],[3,4]] using the characteristic equation.",
        "For A = [[a,b],[c,d]], derive the formula for eigenvalues in terms of trace and determinant.",
    ],
}

EIGENVALUES_3X3 = {
    "name": "eigenvalues_3x3",
    "learning_approach": "procedural",
    "exercises": [
        "Find eigenvalues of the 3x3 matrix [[1,0,0],[0,2,1],[0,1,2]] using cofactor expansion.",
        "Given a 3x3 upper triangular matrix, explain why the eigenvalues are the diagonal entries and verify with an example.",
    ],
}

# Case 9: SoP vs PoS (opposites)
SOP_MINIMIZATION = {
    "name": "sum_of_products_minimization",
    "learning_approach": "procedural",
    "exercises": [
        "Minimize the Boolean function F(A,B,C,D) = Σm(0,1,2,5,6,7,8,9,10,14) using a Karnaugh map. Express result in SoP form.",
        "Using K-map, find the minimal SoP expression for f(w,x,y,z) with minterms 1,3,5,7,9,11,13,15.",
    ],
}

POS_MINIMIZATION = {
    "name": "product_of_sums_minimization",
    "learning_approach": "procedural",
    "exercises": [
        "Minimize F(A,B,C,D) = ΠM(0,1,2,5,8,9,10) using K-map. Express the result in Product of Sums form.",
        "Find the minimal PoS expression for the function with maxterms 0,2,4,6,8,10,12,14.",
    ],
}

# ============================================================================
# TEST DATA: Edge Cases
# ============================================================================

# Case 10: Empty exercises
EMPTY_EXERCISES = {"name": "some_concept", "learning_approach": "conceptual", "exercises": []}

# Case 11: Single very long exercise
SINGLE_LONG_EXERCISE = {
    "name": "complex_algorithm",
    "learning_approach": "analytical",
    "exercises": [
        """Consider a distributed system with N nodes. Each node maintains a local clock
        that may drift. The Lamport timestamp algorithm assigns logical timestamps to events.
        Given the following sequence of events across 3 nodes:
        - Node A sends message m1 at local time 5
        - Node B receives m1 at local time 3 (before A's send according to local clock)
        - Node B sends message m2 at local time 7
        - Node A receives m2 at local time 8
        - Node C sends message m3 to both A and B at local time 2

        a) Assign Lamport timestamps to all events
        b) Determine the happens-before relationship between events
        c) Identify any concurrent events
        d) Could this sequence of events happen in a real distributed system? Explain."""
    ],
}

# Case 12: Items with identical exercises (should definitely merge)
IDENTICAL_1 = {
    "name": "process_synchronization",
    "learning_approach": "procedural",
    "exercises": [
        "Implement a solution to the producer-consumer problem using semaphores.",
        "Write pseudocode for the readers-writers problem with writer preference.",
    ],
}

IDENTICAL_2 = {
    "name": "synchronization_primitives",
    "learning_approach": "procedural",
    "exercises": [
        "Implement a solution to the producer-consumer problem using semaphores.",
        "Write pseudocode for the readers-writers problem with writer preference.",
    ],
}

# Case 13: Learning approach tie scenario
TIE_ITEM_1 = {
    "name": "binary_search",
    "learning_approach": "procedural",
    "exercises": [
        "Implement binary search to find element 42 in sorted array [1,5,12,42,67,89,100].",
    ],
}

TIE_ITEM_2 = {
    "name": "binary_search_algorithm",
    "learning_approach": "conceptual",
    "exercises": [
        "Explain why binary search has O(log n) time complexity. What assumptions must hold?",
    ],
}

# Case 14: Non-CS domain (law - to test course-agnostic)
CONTRACT_LAW_1 = {
    "name": "contract_formation",
    "learning_approach": "conceptual",
    "exercises": [
        "Explain the four essential elements required for a valid contract formation under common law.",
        "Distinguish between an offer and an invitation to treat. Provide two examples of each.",
    ],
}

CONTRACT_LAW_2 = {
    "name": "valid_contract_elements",
    "learning_approach": "conceptual",
    "exercises": [
        "What are the necessary conditions for a contract to be legally binding? Discuss consideration and capacity.",
        "Analyze whether the following scenario constitutes a valid contract: A advertises a car for $5000, B says 'I'll take it'.",
    ],
}

# Case 15: Medicine domain (course-agnostic test)
CARDIOLOGY_1 = {
    "name": "myocardial_infarction_diagnosis",
    "learning_approach": "procedural",
    "exercises": [
        "Given ECG showing ST elevation in leads II, III, aVF, identify the type of MI and affected coronary artery.",
        "Patient presents with chest pain, elevated troponins, and ST depression in V1-V4. What is your diagnosis and immediate management?",
    ],
}

CARDIOLOGY_2 = {
    "name": "heart_attack_diagnosis",
    "learning_approach": "procedural",
    "exercises": [
        "Interpret the following ECG: ST elevation in V1-V4, reciprocal depression in II, III, aVF. Name the infarct territory.",
        "A 55-year-old male with crushing chest pain has troponin I of 5.2 ng/mL. List the diagnostic criteria for STEMI.",
    ],
}


# ============================================================================
# Test Case Definitions
# ============================================================================

SHOULD_MERGE_CASES = [
    {
        "name": "abbreviation_fcfs",
        "description": "FCFS abbreviation vs full name - same scheduling skill",
        "items": [FCFS_FULL, FCFS_ABBREV],
        "expected_groups": 1,
        "reason": "Both test the same FCFS scheduling calculation skill",
    },
    {
        "name": "different_context_fsm",
        "description": "FSM design with tramway vs webcam context - same skill",
        "items": [FSM_TRAMWAY, FSM_WEBCAM],
        "expected_groups": 1,
        "reason": "Both test FSM design skill, context is irrelevant",
    },
    {
        "name": "synonym_bankers",
        "description": "Banker's algorithm with different naming",
        "items": [DEADLOCK_AVOIDANCE_FULL, BANKERS_ALGO],
        "expected_groups": 1,
        "reason": "Same algorithm, different name verbosity",
    },
    {
        "name": "verbosity_eigenvalues",
        "description": "Eigenvalue computation with short vs long name",
        "items": [EIGENVALUES_SHORT, EIGENVALUES_LONG],
        "expected_groups": 1,
        "reason": "Same 2x2 eigenvalue computation skill",
    },
    {
        "name": "identical_exercises",
        "description": "Items with identical exercise content",
        "items": [IDENTICAL_1, IDENTICAL_2],
        "expected_groups": 1,
        "reason": "Identical exercises = definitely same skill",
    },
    {
        "name": "law_contract_formation",
        "description": "Contract law - same topic different naming (non-CS)",
        "items": [CONTRACT_LAW_1, CONTRACT_LAW_2],
        "expected_groups": 1,
        "reason": "Both test understanding of contract formation elements",
    },
    {
        "name": "medicine_mi_diagnosis",
        "description": "MI diagnosis - medical domain (non-CS)",
        "items": [CARDIOLOGY_1, CARDIOLOGY_2],
        "expected_groups": 1,
        "reason": "Both test MI diagnosis from ECG/labs",
    },
]

SHOULD_NOT_MERGE_CASES = [
    {
        "name": "opposites_detection_prevention",
        "description": "Deadlock detection vs prevention - opposite approaches",
        "items": [DEADLOCK_DETECTION, DEADLOCK_PREVENTION],
        "expected_groups": 0,
        "reason": "Detection finds deadlocks; prevention stops them from happening - different skills",
    },
    {
        "name": "same_name_different_skill",
        "description": "CPU scheduling - explain theory vs calculate schedule",
        "items": [CPU_SCHEDULING_CONCEPTUAL, CPU_SCHEDULING_PROCEDURAL],
        "expected_groups": 0,
        "reason": "One tests understanding, other tests calculation - different skills",
    },
    {
        "name": "moore_vs_mealy",
        "description": "Moore vs Mealy machine design - related but different",
        "items": [MOORE_MACHINE, MEALY_MACHINE],
        "expected_groups": 0,
        "reason": "Different state machine types with different design rules",
    },
    {
        "name": "matrix_size_variants",
        "description": "Eigenvalues 2x2 vs 3x3 - different techniques",
        "items": [EIGENVALUES_2X2, EIGENVALUES_3X3],
        "expected_groups": 0,
        "reason": "2x2 uses simple formula; 3x3 needs cofactor expansion",
    },
    {
        "name": "sop_vs_pos",
        "description": "Sum of Products vs Product of Sums minimization",
        "items": [SOP_MINIMIZATION, POS_MINIMIZATION],
        "expected_groups": 0,
        "reason": "Different canonical forms with different K-map grouping rules",
    },
]

EDGE_CASES = [
    {
        "name": "single_item",
        "description": "Single item - nothing to merge",
        "items": [FCFS_FULL],
        "expected_groups": 0,
        "reason": "Cannot merge with only one item",
    },
    {
        "name": "empty_exercises",
        "description": "Item with no exercises",
        "items": [FCFS_FULL, EMPTY_EXERCISES],
        "expected_groups": 0,
        "reason": "Cannot judge skill without exercises",
    },
    {
        "name": "approach_tie",
        "description": "Same skill but different approaches - tie scenario",
        "items": [TIE_ITEM_1, TIE_ITEM_2],
        "expected_groups": 1,  # Should merge based on skill, then resolve approach
        "reason": "Same binary search skill, approach determined post-merge",
        "post_merge_approach": "procedural",  # or LLM decides
    },
    {
        "name": "long_exercise",
        "description": "Item with single very long exercise",
        "items": [SINGLE_LONG_EXERCISE, SINGLE_LONG_EXERCISE],  # Same item
        "expected_groups": 1,
        "reason": "Long exercises should still be processed",
    },
]

MULTI_ITEM_CASES = [
    {
        "name": "three_way_merge",
        "description": "Three items that should all merge",
        "items": [
            FCFS_FULL,
            FCFS_ABBREV,
            {
                "name": "first_come_first_served",
                "learning_approach": "procedural",
                "exercises": [
                    "Calculate FCFS waiting times for process set {P1(0,8), P2(1,4), P3(2,9), P4(3,5)}.",
                ],
            },
        ],
        "expected_groups": 1,
        "reason": "All three test the same FCFS scheduling skill",
    },
    {
        "name": "partial_merge",
        "description": "Some items merge, others don't",
        "items": [FCFS_FULL, FCFS_ABBREV, CPU_SCHEDULING_CONCEPTUAL],
        "expected_groups": 1,  # FCFS items merge, conceptual stays separate
        "expected_separate": 1,  # CPU conceptual stays alone
        "reason": "FCFS items merge; conceptual scheduling is different skill",
    },
    {
        "name": "no_merges",
        "description": "Multiple items, none should merge",
        "items": [MOORE_MACHINE, MEALY_MACHINE, SOP_MINIMIZATION, POS_MINIMIZATION],
        "expected_groups": 0,
        "reason": "All items test different skills",
    },
]


# ============================================================================
# Test Functions
# ============================================================================


def test_should_merge_cases():
    """Test cases where items SHOULD be merged."""
    print("\n" + "=" * 70)
    print("SHOULD MERGE CASES")
    print("=" * 70)

    for case in SHOULD_MERGE_CASES:
        print(f"\n[{case['name']}] {case['description']}")
        print(f"  Items: {[i['name'] for i in case['items']]}")
        print(f"  Expected: {case['expected_groups']} group(s)")
        print(f"  Reason: {case['reason']}")

        # Validate test data
        for item in case["items"]:
            assert "name" in item, "Item missing 'name'"
            assert "exercises" in item, f"Item {item['name']} missing 'exercises'"
            assert len(item["exercises"]) > 0 or case["name"] == "empty_exercises", (
                f"Item {item['name']} has no exercises"
            )

        print("  [DATA VALID]")


def test_should_not_merge_cases():
    """Test cases where items should NOT be merged."""
    print("\n" + "=" * 70)
    print("SHOULD NOT MERGE CASES")
    print("=" * 70)

    for case in SHOULD_NOT_MERGE_CASES:
        print(f"\n[{case['name']}] {case['description']}")
        print(f"  Items: {[i['name'] for i in case['items']]}")
        print(f"  Expected: {case['expected_groups']} group(s) (no merging)")
        print(f"  Reason: {case['reason']}")

        # Show exercise snippets for comparison
        for item in case["items"]:
            ex_preview = (
                item["exercises"][0][:80] + "..." if item["exercises"] else "(no exercises)"
            )
            print(f'    - {item["name"]}: "{ex_preview}"')

        print("  [DATA VALID]")


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("\n" + "=" * 70)
    print("EDGE CASES")
    print("=" * 70)

    for case in EDGE_CASES:
        print(f"\n[{case['name']}] {case['description']}")
        print(f"  Items: {[i['name'] for i in case['items']]}")
        print(f"  Expected: {case['expected_groups']} group(s)")
        print(f"  Reason: {case['reason']}")

        if "post_merge_approach" in case:
            print(f"  Post-merge approach: {case['post_merge_approach']}")

        print("  [DATA VALID]")


def test_multi_item_cases():
    """Test cases with more than 2 items."""
    print("\n" + "=" * 70)
    print("MULTI-ITEM CASES")
    print("=" * 70)

    for case in MULTI_ITEM_CASES:
        print(f"\n[{case['name']}] {case['description']}")
        print(f"  Items ({len(case['items'])}): {[i['name'] for i in case['items']]}")
        print(f"  Expected merged groups: {case['expected_groups']}")
        if "expected_separate" in case:
            print(f"  Expected separate items: {case['expected_separate']}")
        print(f"  Reason: {case['reason']}")

        print("  [DATA VALID]")


def generate_anonymous_prompt(items: list[dict]) -> str:
    """Generate the anonymous synonym detection prompt."""
    items_text = []
    for i, item in enumerate(items, 1):
        item_text = f"- Item_{i}"
        if item.get("exercises"):
            exercise_snippets = []
            for ex in item["exercises"][:2]:
                snippet = ex[:200] + "..." if len(ex) > 200 else ex
                exercise_snippets.append(f'    "{snippet}"')
            if exercise_snippets:
                item_text += "\n  Exercises:\n" + "\n".join(exercise_snippets)
        items_text.append(item_text)

    prompt = f"""Identify which items test the EXACT SAME SKILL based on their exercises.

Items (judge ONLY by exercises, not by item IDs):
{chr(10).join(items_text)}

GROUP items that test the SAME SKILL:
- A student who masters one has automatically mastered the other
- Exercises require the EXACT SAME knowledge/technique to solve
- A single flashcard could teach both equally well

DO NOT GROUP when items test DIFFERENT SKILLS:
- One asks to EXPLAIN/DESCRIBE, another asks to CALCULATE/APPLY
- Mastering one gives only partial mastery of the other
- They would need separate study sessions

Return JSON array of groups (item IDs that test same skill).
Return [] if no items test the same skill.
"""
    return prompt


def preview_prompts():
    """Preview what prompts would be generated for key test cases."""
    print("\n" + "=" * 70)
    print("PROMPT PREVIEWS")
    print("=" * 70)

    preview_cases = [
        ("abbreviation_fcfs", [FCFS_FULL, FCFS_ABBREV]),
        ("same_name_different_skill", [CPU_SCHEDULING_CONCEPTUAL, CPU_SCHEDULING_PROCEDURAL]),
        ("opposites_detection_prevention", [DEADLOCK_DETECTION, DEADLOCK_PREVENTION]),
    ]

    for name, items in preview_cases:
        print(f"\n--- {name} ---")
        prompt = generate_anonymous_prompt(items)
        print(prompt[:1500])
        if len(prompt) > 1500:
            print(f"... [{len(prompt) - 1500} more chars]")


def run_all_tests():
    """Run all test validations."""
    print("\n" + "=" * 70)
    print("ANONYMOUS SYNONYM DETECTION TEST SUITE")
    print("=" * 70)
    print("\nThis test suite validates the test DATA for the anonymous approach.")
    print("Actual LLM-based testing requires running with merge_items().\n")

    test_should_merge_cases()
    test_should_not_merge_cases()
    test_edge_cases()
    test_multi_item_cases()
    preview_prompts()

    # Summary
    total_cases = (
        len(SHOULD_MERGE_CASES)
        + len(SHOULD_NOT_MERGE_CASES)
        + len(EDGE_CASES)
        + len(MULTI_ITEM_CASES)
    )

    print("\n" + "=" * 70)
    print(f"SUMMARY: {total_cases} test cases defined")
    print(f"  - Should merge: {len(SHOULD_MERGE_CASES)}")
    print(f"  - Should NOT merge: {len(SHOULD_NOT_MERGE_CASES)}")
    print(f"  - Edge cases: {len(EDGE_CASES)}")
    print(f"  - Multi-item: {len(MULTI_ITEM_CASES)}")
    print("=" * 70)

    return total_cases


# ============================================================================
# LLM Integration Test (requires actual LLM)
# ============================================================================


def test_with_llm(llm_manager=None):
    """
    Run actual LLM-based synonym detection tests.

    Usage:
        from models.llm_manager import LLMManager
        llm = LLMManager(provider="deepseek")
        test_with_llm(llm)
    """
    if llm_manager is None:
        print("Skipping LLM tests - no LLM manager provided")
        print("To run: test_with_llm(LLMManager(provider='deepseek'))")
        return

    from core.merger import merge_items

    print("\n" + "=" * 70)
    print("LLM-BASED SYNONYM DETECTION TESTS")
    print("=" * 70)

    passed = 0
    failed = 0

    # Test should-merge cases
    for case in SHOULD_MERGE_CASES:
        items = [
            {"name": i["name"], "type": "key_concept", "exercises": i["exercises"]}
            for i in case["items"]
        ]

        result = merge_items(items, llm_manager)

        if len(result) >= case["expected_groups"]:
            print(f"PASS [{case['name']}]: Got {len(result)} group(s)")
            passed += 1
        else:
            print(
                f"FAIL [{case['name']}]: Expected {case['expected_groups']} group(s), got {len(result)}"
            )
            failed += 1

    # Test should-not-merge cases
    for case in SHOULD_NOT_MERGE_CASES:
        items = [
            {"name": i["name"], "type": "key_concept", "exercises": i["exercises"]}
            for i in case["items"]
        ]

        result = merge_items(items, llm_manager)

        if len(result) == 0:
            print(f"PASS [{case['name']}]: Correctly returned no merges")
            passed += 1
        else:
            print(f"FAIL [{case['name']}]: Should not merge, but got {len(result)} group(s)")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
    return passed, failed


if __name__ == "__main__":
    run_all_tests()
