#!/usr/bin/env python3
"""
Quick sanity test for anonymous synonym detection.
Tests 4 critical cases that must pass.

Usage:
    python tests/test_synonym_quick.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.llm_manager import LLMManager

# Critical test cases
CRITICAL_TESTS = [
    {
        "name": "FCFS_abbreviation",
        "description": "FCFS vs first_come_first_served - SHOULD MERGE",
        "items": [
            {
                "name": "fcfs_scheduling",
                "exercises": [
                    "Calculate FCFS waiting time for P1(0,5), P2(1,3), P3(2,8).",
                ],
            },
            {
                "name": "first_come_first_served_scheduling",
                "exercises": [
                    "Apply first-come-first-served to P1(0,4), P2(1,3), P3(2,1). Find avg wait.",
                ],
            },
        ],
        "should_merge": True,
    },
    {
        "name": "detection_vs_prevention",
        "description": "Deadlock detection vs prevention - SHOULD NOT MERGE",
        "items": [
            {
                "name": "deadlock_detection",
                "exercises": [
                    "Apply deadlock detection algorithm to find deadlocked processes.",
                ],
            },
            {
                "name": "deadlock_prevention",
                "exercises": [
                    "Explain how to prevent the hold-and-wait condition.",
                ],
            },
        ],
        "should_merge": False,
    },
    {
        "name": "same_name_different_skill",
        "description": "CPU scheduling: explain vs calculate - SHOULD NOT MERGE",
        "items": [
            {
                "name": "cpu_scheduling",
                "exercises": [
                    "Explain preemptive vs non-preemptive scheduling differences.",
                ],
            },
            {
                "name": "cpu_scheduling",
                "exercises": [
                    "Calculate SJF schedule for 5 processes with given burst times.",
                ],
            },
        ],
        "should_merge": False,
    },
    {
        "name": "different_context_same_skill",
        "description": "FSM design tramway vs webcam - SHOULD MERGE",
        "items": [
            {
                "name": "fsm_tramway",
                "exercises": [
                    "Design Moore FSM for tramway crossing. States: Idle, Warning, Blocked.",
                ],
            },
            {
                "name": "fsm_webcam",
                "exercises": [
                    "Design Mealy FSM for webcam motion detector. States: Monitoring, Recording.",
                ],
            },
        ],
        "should_merge": True,
    },
]


def generate_prompt(items):
    """Generate anonymous prompt."""
    items_text = []
    for i, item in enumerate(items, 1):
        item_text = f"- Item_{i}"
        for ex in item["exercises"]:
            item_text += f'\n  Exercise: "{ex}"'
        items_text.append(item_text)

    return f"""Identify which items test the EXACT SAME SKILL based on exercises.

{chr(10).join(items_text)}

GROUP items testing SAME SKILL (mastering one = mastering other).
DO NOT GROUP items testing DIFFERENT SKILLS (explain vs calculate = different).

Return JSON: array of groups like [["Item_1", "Item_2"]] or [] if no same skill.
"""


def run_quick_test():
    """Run the 4 critical tests."""
    print("Loading LLM (deepseek)...")
    llm = LLMManager(provider="deepseek")

    passed = 0
    failed = 0

    for test in CRITICAL_TESTS:
        print(f"\n{'=' * 50}")
        print(f"TEST: {test['name']}")
        print(f"  {test['description']}")

        prompt = generate_prompt(test["items"])
        response = llm.generate(prompt=prompt, temperature=0.0, json_mode=True)

        try:
            result = json.loads(response.text) if response else []
            # Handle both {"groups": [...]} and [...] formats
            if isinstance(result, dict) and "groups" in result:
                groups = result["groups"]
            elif isinstance(result, list):
                groups = result
            else:
                groups = []
        except Exception:
            groups = []

        has_merge = len(groups) > 0

        if has_merge == test["should_merge"]:
            print(f"  PASS: Got {'merge' if has_merge else 'no merge'} as expected")
            passed += 1
        else:
            print(
                f"  FAIL: Expected {'merge' if test['should_merge'] else 'no merge'}, got {'merge' if has_merge else 'no merge'}"
            )
            print(f"  Response: {response.text if response else 'None'}")
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"RESULTS: {passed}/4 passed")

    if failed == 0:
        print("All critical tests passed!")
    else:
        print(f"WARNING: {failed} critical test(s) failed!")

    return failed == 0


if __name__ == "__main__":
    success = run_quick_test()
    sys.exit(0 if success else 1)
