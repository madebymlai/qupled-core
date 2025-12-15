"""
SM-2 Algorithm Demonstration

This script demonstrates how the SM-2 spaced repetition algorithm works
with various quality scores and shows example progressions.
"""

from core.sm2 import SM2Algorithm


def print_header(title):
    """Print a formatted header."""
    print(f"\n{'=' * 70}")
    print(f"{title:^70}")
    print(f"{'=' * 70}\n")


def print_review(review_num, quality, result):
    """Print review information in a formatted way."""
    print(f"Review {review_num}:")
    print(f"  Quality:        {quality} / 5")
    print(f"  New EF:         {result['new_ef']}")
    print(f"  New Interval:   {result['new_interval']} days")
    print(f"  New Repetition: {result['new_repetition']}")
    print(f"  Next Review:    {result['next_review_date'].strftime('%Y-%m-%d')}")
    print()


def demo_perfect_recall():
    """Demonstrate perfect recall progression (all quality 5)."""
    print_header("Perfect Recall Progression (Quality 5)")

    sm2 = SM2Algorithm()
    ef = 2.5
    interval = 0
    rep = 0

    qualities = [5, 5, 5, 5, 5]

    for i, quality in enumerate(qualities, 1):
        result = sm2.calculate_next_review(quality, ef, interval, rep)
        print_review(i, quality, result)

        ef = result["new_ef"]
        interval = result["new_interval"]
        rep = result["new_repetition"]

    print(f"Summary after {len(qualities)} reviews:")
    print(f"  - Started with 0 days interval")
    print(f"  - Ended with {interval} days interval")
    print(f"  - Progression: 1 → 6 → 15 → 38 → 95 days")
    print(f"  - Exponential growth after 2nd review")


def demo_mixed_quality():
    """Demonstrate mixed quality progression."""
    print_header("Mixed Quality Progression (5, 4, 3, 5, 4)")

    sm2 = SM2Algorithm()
    ef = 2.5
    interval = 0
    rep = 0

    qualities = [5, 4, 3, 5, 4]
    quality_names = ["Perfect", "Good", "Fair", "Perfect", "Good"]

    for i, (quality, name) in enumerate(zip(qualities, quality_names), 1):
        result = sm2.calculate_next_review(quality, ef, interval, rep)
        print(f"Review {i} ({name}):")
        print(f"  Quality:        {quality} / 5")
        print(f"  New EF:         {result['new_ef']}")
        print(f"  New Interval:   {result['new_interval']} days")
        print(f"  New Repetition: {result['new_repetition']}")
        print()

        ef = result["new_ef"]
        interval = result["new_interval"]
        rep = result["new_repetition"]

    print(f"Summary:")
    print(f"  - Quality 3 reduced EF from 2.5 to {ef}")
    print(f"  - Despite lower EF, intervals still grow")
    print(f"  - Final interval: {interval} days")


def demo_failure_and_recovery():
    """Demonstrate what happens when you fail (quality < 3)."""
    print_header("Failure and Recovery (5, 5, 2, 5, 5)")

    sm2 = SM2Algorithm()
    ef = 2.5
    interval = 0
    rep = 0

    qualities = [5, 5, 2, 5, 5]
    quality_names = ["Perfect", "Perfect", "FAILED", "Perfect", "Perfect"]

    for i, (quality, name) in enumerate(zip(qualities, quality_names), 1):
        result = sm2.calculate_next_review(quality, ef, interval, rep)
        print(f"Review {i} ({name}):")
        print(f"  Quality:        {quality} / 5")
        print(f"  New EF:         {result['new_ef']}")
        print(f"  New Interval:   {result['new_interval']} days")
        print(f"  New Repetition: {result['new_repetition']}")

        if quality < 3:
            print(f"  ⚠️  RESET! Back to day 1")

        print()

        ef = result["new_ef"]
        interval = result["new_interval"]
        rep = result["new_repetition"]

    print(f"Summary:")
    print(f"  - Quality < 3 resets progress back to day 1")
    print(f"  - But EF is preserved (and reduced)")
    print(f"  - Need to rebuild repetition count")
    print(f"  - Final EF: {ef} (lower than starting 2.5)")


def demo_quality_to_interval_mapping():
    """Show how different quality scores affect intervals."""
    print_header("Quality Score Impact on Intervals")

    sm2 = SM2Algorithm()

    print("Starting from: EF=2.5, Interval=6 days, Repetition=2\n")

    for quality in range(6):
        result = sm2.calculate_next_review(
            quality=quality, current_ef=2.5, current_interval=6, repetition=2
        )

        quality_names = {
            0: "Complete blackout",
            1: "Incorrect, vaguely familiar",
            2: "Incorrect, remembered after seeing",
            3: "Correct with difficulty",
            4: "Correct with hesitation",
            5: "Perfect recall",
        }

        print(f"Quality {quality} ({quality_names[quality]}):")
        print(f"  New EF:       {result['new_ef']}")
        print(f"  New Interval: {result['new_interval']} days")
        print(f"  New Rep:      {result['new_repetition']}")

        if quality < 3:
            print(f"  Status:       ⚠️  RESET")
        else:
            print(f"  Status:       ✓ Progress")

        print()


def demo_performance_to_quality():
    """Demonstrate converting quiz performance to quality scores."""
    print_header("Quiz Performance → Quality Score Conversion")

    sm2 = SM2Algorithm()

    scenarios = [
        {
            "name": "Perfect - Fast & No Hints",
            "correct": True,
            "time_taken": 60,
            "hint_used": False,
            "expected_time": 180,
        },
        {
            "name": "Good - Normal Speed",
            "correct": True,
            "time_taken": 150,
            "hint_used": False,
            "expected_time": 180,
        },
        {
            "name": "Good - With Hint",
            "correct": True,
            "time_taken": 150,
            "hint_used": True,
            "expected_time": 180,
        },
        {
            "name": "Fair - Slow but Correct",
            "correct": True,
            "time_taken": 300,
            "hint_used": False,
            "expected_time": 180,
        },
        {
            "name": "Poor - Incorrect",
            "correct": False,
            "time_taken": 120,
            "hint_used": False,
            "expected_time": 180,
        },
    ]

    for scenario in scenarios:
        quality = sm2.get_review_quality_from_score(
            correct=scenario["correct"],
            time_taken=scenario["time_taken"],
            hint_used=scenario["hint_used"],
            expected_time=scenario["expected_time"],
        )

        time_ratio = scenario["time_taken"] / scenario["expected_time"]

        print(f"{scenario['name']}:")
        print(f"  Correct:      {scenario['correct']}")
        print(
            f"  Time:         {scenario['time_taken']}s / {scenario['expected_time']}s ({time_ratio:.1%})"
        )
        print(f"  Hint Used:    {scenario['hint_used']}")
        print(f"  → Quality:    {quality} / 5")
        print()


def demo_mastery_levels():
    """Demonstrate mastery level progression."""
    print_header("Mastery Level Progression")

    sm2 = SM2Algorithm()

    scenarios = [
        {"name": "Brand New", "rep": 0, "interval": 0, "correct": 0, "total": 0},
        {"name": "First Review", "rep": 1, "interval": 1, "correct": 1, "total": 1},
        {"name": "Learning Phase", "rep": 2, "interval": 6, "correct": 2, "total": 2},
        {"name": "Reviewing Phase", "rep": 5, "interval": 15, "correct": 5, "total": 6},
        {"name": "Mastered", "rep": 12, "interval": 45, "correct": 12, "total": 13},
        {
            "name": "Many Reviews, Low Accuracy",
            "rep": 12,
            "interval": 45,
            "correct": 6,
            "total": 12,
        },
    ]

    for scenario in scenarios:
        level = sm2.get_mastery_level(
            repetition=scenario["rep"],
            interval_days=scenario["interval"],
            correct_count=scenario["correct"],
            total_count=scenario["total"],
        )

        if scenario["total"] > 0:
            accuracy = (scenario["correct"] / scenario["total"]) * 100
        else:
            accuracy = 0

        print(f"{scenario['name']}:")
        print(f"  Repetition:   {scenario['rep']}")
        print(f"  Interval:     {scenario['interval']} days")
        print(f"  Accuracy:     {accuracy:.0f}% ({scenario['correct']}/{scenario['total']})")
        print(f"  → Level:      {level.upper()}")
        print()


def demo_long_term_retention():
    """Simulate long-term learning with realistic study patterns."""
    print_header("Long-Term Retention Simulation (6 months)")

    sm2 = SM2Algorithm()
    ef = 2.5
    interval = 0
    rep = 0

    # Simulate realistic performance over time
    # Usually good (4-5), occasional struggles (3), rare failures (2)
    qualities = [5, 4, 5, 3, 5, 4, 5, 5, 4, 5]

    print("Simulating a student learning a difficult concept:\n")

    total_days = 0
    for i, quality in enumerate(qualities, 1):
        result = sm2.calculate_next_review(quality, ef, interval, rep)

        total_days += result["new_interval"]

        quality_emoji = {5: "🌟", 4: "✓", 3: "~", 2: "✗", 1: "✗", 0: "✗"}

        print(
            f"Review {i:2d} (Day {total_days:3d}): "
            f"Quality {quality} {quality_emoji[quality]}  →  "
            f"Next in {result['new_interval']:3d} days "
            f"(EF: {result['new_ef']})"
        )

        ef = result["new_ef"]
        interval = result["new_interval"]
        rep = result["new_repetition"]

        if total_days > 180:  # 6 months
            break

    print(f"\nAfter ~6 months:")
    print(f"  Total reviews:    {i}")
    print(f"  Current interval: {interval} days")
    print(f"  Consecutive reps: {rep}")
    print(f"  Easiness factor:  {ef}")
    print(f"\n  → Material is well-retained with minimal review effort!")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print(" " * 15 + "SM-2 SPACED REPETITION ALGORITHM")
    print(" " * 20 + "Interactive Demonstration")
    print("=" * 70)

    demo_perfect_recall()
    demo_mixed_quality()
    demo_failure_and_recovery()
    demo_quality_to_interval_mapping()
    demo_performance_to_quality()
    demo_mastery_levels()
    demo_long_term_retention()

    print("\n" + "=" * 70)
    print(" " * 20 + "Demonstration Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. Perfect recall (5) → Exponential interval growth")
    print("  2. Good recall (4) → Steady progress")
    print("  3. Fair recall (3) → Slower growth, lower EF")
    print("  4. Poor recall (<3) → Reset to day 1, rebuild progress")
    print("  5. EF adjusts to item difficulty (1.3 - 2.5 range)")
    print("  6. Mastery levels track learning progress")
    print("\n")


if __name__ == "__main__":
    main()
