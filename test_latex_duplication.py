#!/usr/bin/env python3
"""Test script to check LaTeX duplication in review exercise generation."""

import re
import sys
from core.review_engine import ReviewEngine, ExerciseExample
from models.llm_manager import LLMManager

# Common LaTeX duplication patterns to detect
DUPLICATION_PATTERNS = [
    r'([A-Z])\1(?![a-z])',  # VV, WW, KK (single letter doubled)
    r'(\$[^$]+\$)\1',        # $x$x$ (whole expression doubled)
    r'(mathbb\{[A-Z]\})\1',  # mathbb{R}mathbb{R}
    r'(\\[a-z]+)\1',         # \alpha\alpha
]

def check_for_duplication(text: str) -> list[str]:
    """Check text for LaTeX duplication patterns."""
    issues = []
    for pattern in DUPLICATION_PATTERNS:
        matches = re.findall(pattern, text)
        if matches:
            issues.append(f"Pattern '{pattern}' matched: {matches[:3]}")
    return issues


def test_exercise_generation(engine: ReviewEngine, test_name: str, examples: list[ExerciseExample]):
    """Generate an exercise and check for duplication."""
    print(f"\n{'='*60}")
    print(f"Test: {test_name}")
    print('='*60)

    exercise = engine.generate_exercise(
        knowledge_item_name=test_name,
        learning_approach="conceptual",
        examples=examples,
    )

    print(f"\nExercise text:\n{exercise.exercise_text[:500]}...")
    print(f"\nExpected answer:\n{exercise.expected_answer[:300]}...")

    # Check for duplication
    issues = check_for_duplication(exercise.exercise_text)
    issues.extend(check_for_duplication(exercise.expected_answer))

    if issues:
        print(f"\n❌ DUPLICATION DETECTED:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print(f"\n✅ No duplication detected")
        return True


def main():
    # Initialize LLM
    llm = LLMManager()
    engine = ReviewEngine(llm)

    # Test cases with LaTeX-heavy content
    test_cases = [
        {
            "name": "Applicazione Lineare",
            "examples": [
                ExerciseExample(
                    text="Dare la definizione di applicazione lineare tra due spazi vettoriali $V$ e $W$ sul campo $\\mathbb{R}$.",
                    solution="Un'applicazione $f: V \\to W$ è lineare se $f(u+v) = f(u) + f(v)$ e $f(\\alpha v) = \\alpha f(v)$.",
                    source_type="exam"
                ),
            ]
        },
        {
            "name": "Autovalori e Autovettori",
            "examples": [
                ExerciseExample(
                    text="Sia $A$ una matrice $3 \\times 3$. Trovare gli autovalori di $A$ sapendo che $\\det(A - \\lambda I) = 0$.",
                    solution="Gli autovalori si trovano risolvendo $\\det(A - \\lambda I) = 0$.",
                    source_type="exam"
                ),
            ]
        },
        {
            "name": "SR Latch",
            "examples": [
                ExerciseExample(
                    text="Si consideri un latch SR con ingressi $S$ e $R$. Spiegare perché $S=1, R=1$ non è ammissibile.",
                    solution="Con $S=1$ e $R=1$, entrambe le uscite $Q$ e $\\bar{Q}$ sono forzate a 0, violando la condizione $Q \\neq \\bar{Q}$.",
                    source_type="exam"
                ),
            ]
        },
        {
            "name": "Limiti e Derivate",
            "examples": [
                ExerciseExample(
                    text="Calcolare $\\lim_{x \\to 0} \\frac{\\sin(x)}{x}$ usando la regola di L'Hôpital.",
                    solution="$\\lim_{x \\to 0} \\frac{\\sin(x)}{x} = \\lim_{x \\to 0} \\frac{\\cos(x)}{1} = 1$",
                    source_type="exam"
                ),
            ]
        },
    ]

    results = []
    for tc in test_cases:
        passed = test_exercise_generation(engine, tc["name"], tc["examples"])
        results.append((tc["name"], passed))

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    passed = sum(1 for _, p in results if p)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    for name, p in results:
        status = "✅" if p else "❌"
        print(f"  {status} {name}")

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
