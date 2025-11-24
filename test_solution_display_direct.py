#!/usr/bin/env python3
"""Direct test of solution formatting."""

from core.tutor import Tutor

def test_solution_formatting():
    """Test the solution formatting directly."""

    # Create mock exercises with solutions
    exercises_with_solutions = [
        {
            'exercise_number': 'Ex 1.2',
            'solution': 'This is the solution to exercise 1.2.\n\nStep 1: First do this.\nStep 2: Then do that.\nStep 3: Finally, verify the result.',
            'source_pdf': 'exam_2023.pdf'
        },
        {
            'exercise_number': 'Ex 2.5',
            'solution': 'Solution for exercise 2.5:\n- Calculate the value\n- Apply the formula\n- Get the final answer',
            'source_pdf': 'exam_2024.pdf'
        }
    ]

    # Test English formatting
    print("=== Testing English formatting ===")
    tutor_en = Tutor(language="en")
    formatted_en = tutor_en._format_official_solutions(exercises_with_solutions)
    print(formatted_en)
    print("\n" + "=" * 80 + "\n")

    # Test Italian formatting
    print("=== Testing Italian formatting ===")
    tutor_it = Tutor(language="it")
    formatted_it = tutor_it._format_official_solutions(exercises_with_solutions)
    print(formatted_it)

if __name__ == "__main__":
    test_solution_formatting()
