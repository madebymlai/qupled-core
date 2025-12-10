#!/usr/bin/env python3
"""
Run anonymous synonym detection tests against actual LLM.

Usage:
    python tests/run_synonym_llm_test.py [--provider deepseek|anthropic]
"""

import sys
import os
import argparse
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.llm_manager import LLMManager
from test_anonymous_synonym_detection import (
    SHOULD_MERGE_CASES,
    SHOULD_NOT_MERGE_CASES,
    EDGE_CASES,
    MULTI_ITEM_CASES,
    generate_anonymous_prompt,
)


def run_single_test(items: list[dict], llm: LLMManager, verbose: bool = False) -> dict:
    """
    Run synonym detection on a set of items.

    Returns:
        {"groups": [...], "raw_response": str}
    """
    prompt = generate_anonymous_prompt(items)

    if verbose:
        print(f"\n--- PROMPT ---\n{prompt[:800]}...")

    response = llm.generate(
        prompt=prompt,
        temperature=0.0,
        json_mode=True,
    )

    raw_text = response.text if response else ""

    if verbose:
        print(f"\n--- RESPONSE ---\n{raw_text}")

    try:
        result = json.loads(raw_text)
        # Handle both {"groups": [...]} and [...] formats
        if isinstance(result, dict) and "groups" in result:
            groups = result["groups"]
        elif isinstance(result, list):
            groups = result
        else:
            groups = []
    except json.JSONDecodeError:
        groups = []

    return {
        "groups": groups,
        "raw_response": raw_text,
    }


def run_test_suite(llm: LLMManager, verbose: bool = False):
    """Run the full test suite."""

    results = {
        "passed": 0,
        "failed": 0,
        "details": [],
    }

    print("\n" + "="*70)
    print("SHOULD MERGE CASES")
    print("="*70)

    for case in SHOULD_MERGE_CASES:
        items = case["items"]
        expected = case["expected_groups"]

        result = run_single_test(items, llm, verbose)
        num_groups = len(result["groups"])

        # For should-merge, we expect at least 1 group
        passed = num_groups >= expected

        status = "PASS" if passed else "FAIL"
        print(f"\n[{status}] {case['name']}: {case['description']}")
        print(f"  Items: {[i['name'] for i in items]}")
        print(f"  Expected: >= {expected} group(s), Got: {num_groups}")
        print(f"  Groups: {result['groups']}")

        if passed:
            results["passed"] += 1
        else:
            results["failed"] += 1

        results["details"].append({
            "case": case["name"],
            "category": "should_merge",
            "passed": passed,
            "expected": expected,
            "actual": num_groups,
            "groups": result["groups"],
        })

    print("\n" + "="*70)
    print("SHOULD NOT MERGE CASES")
    print("="*70)

    for case in SHOULD_NOT_MERGE_CASES:
        items = case["items"]

        result = run_single_test(items, llm, verbose)
        num_groups = len(result["groups"])

        # For should-not-merge, we expect 0 groups
        passed = num_groups == 0

        status = "PASS" if passed else "FAIL"
        print(f"\n[{status}] {case['name']}: {case['description']}")
        print(f"  Items: {[i['name'] for i in items]}")
        print(f"  Expected: 0 groups, Got: {num_groups}")
        if num_groups > 0:
            print(f"  UNEXPECTED Groups: {result['groups']}")

        if passed:
            results["passed"] += 1
        else:
            results["failed"] += 1

        results["details"].append({
            "case": case["name"],
            "category": "should_not_merge",
            "passed": passed,
            "expected": 0,
            "actual": num_groups,
            "groups": result["groups"],
        })

    print("\n" + "="*70)
    print("EDGE CASES")
    print("="*70)

    for case in EDGE_CASES:
        items = case["items"]
        expected = case["expected_groups"]

        # Skip single-item case (can't run without modification)
        if len(items) < 2:
            print(f"\n[SKIP] {case['name']}: {case['description']} (single item)")
            continue

        result = run_single_test(items, llm, verbose)
        num_groups = len(result["groups"])

        passed = num_groups == expected

        status = "PASS" if passed else "FAIL"
        print(f"\n[{status}] {case['name']}: {case['description']}")
        print(f"  Items: {[i['name'] for i in items]}")
        print(f"  Expected: {expected} group(s), Got: {num_groups}")
        print(f"  Groups: {result['groups']}")

        if passed:
            results["passed"] += 1
        else:
            results["failed"] += 1

        results["details"].append({
            "case": case["name"],
            "category": "edge_case",
            "passed": passed,
            "expected": expected,
            "actual": num_groups,
            "groups": result["groups"],
        })

    print("\n" + "="*70)
    print("MULTI-ITEM CASES")
    print("="*70)

    for case in MULTI_ITEM_CASES:
        items = case["items"]
        expected = case["expected_groups"]

        result = run_single_test(items, llm, verbose)
        num_groups = len(result["groups"])

        # For multi-item, check if we got the expected number of merged groups
        passed = num_groups == expected

        status = "PASS" if passed else "FAIL"
        print(f"\n[{status}] {case['name']}: {case['description']}")
        print(f"  Items ({len(items)}): {[i['name'] for i in items]}")
        print(f"  Expected: {expected} group(s), Got: {num_groups}")
        print(f"  Groups: {result['groups']}")

        if passed:
            results["passed"] += 1
        else:
            results["failed"] += 1

        results["details"].append({
            "case": case["name"],
            "category": "multi_item",
            "passed": passed,
            "expected": expected,
            "actual": num_groups,
            "groups": result["groups"],
        })

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    total = results["passed"] + results["failed"]
    print(f"Passed: {results['passed']}/{total}")
    print(f"Failed: {results['failed']}/{total}")

    if results["failed"] > 0:
        print("\nFailed cases:")
        for detail in results["details"]:
            if not detail["passed"]:
                print(f"  - {detail['case']} ({detail['category']}): expected {detail['expected']}, got {detail['actual']}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run synonym detection LLM tests")
    parser.add_argument("--provider", choices=["deepseek", "anthropic"], default="deepseek",
                       help="LLM provider to use (default: deepseek)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show full prompts and responses")
    args = parser.parse_args()

    print(f"Using LLM provider: {args.provider}")

    llm = LLMManager(provider=args.provider)

    results = run_test_suite(llm, verbose=args.verbose)

    # Exit with error code if any tests failed
    sys.exit(0 if results["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
