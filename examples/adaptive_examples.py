#!/usr/bin/env python3
"""
Example demonstrations of the Adaptive Teaching System.
Shows how depth and prerequisites are automatically selected based on mastery level.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.adaptive_teaching import AdaptiveTeachingManager
from storage.database import Database
from rich.console import Console
from rich.table import Table

console = Console()


def example_depth_selection():
    """Example: Automatic depth selection based on mastery level."""
    console.print("\n[bold cyan]Example 1: Adaptive Depth Selection[/bold cyan]\n")

    # Simulated mastery levels
    scenarios = [
        {
            "student": "Alice (Beginner)",
            "mastery": 0.15,
            "expected_depth": "basic",
            "show_prereqs": True,
        },
        {
            "student": "Bob (Intermediate)",
            "mastery": 0.55,
            "expected_depth": "medium",
            "show_prereqs": False,  # Unless recent failures
        },
        {
            "student": "Carol (Advanced)",
            "mastery": 0.85,
            "expected_depth": "advanced",
            "show_prereqs": False,
        },
    ]

    table = Table(title="Adaptive Depth Selection Examples", show_header=True)
    table.add_column("Student", style="cyan")
    table.add_column("Mastery", justify="right", style="yellow")
    table.add_column("Recommended Depth", style="green")
    table.add_column("Show Prerequisites", justify="center", style="magenta")

    for scenario in scenarios:
        mastery_pct = f"{int(scenario['mastery'] * 100)}%"
        prereqs = "✓" if scenario["show_prereqs"] else "✗"

        table.add_row(scenario["student"], mastery_pct, scenario["expected_depth"], prereqs)

    console.print(table)
    console.print("\n[dim]The system automatically adjusts based on student performance![/dim]\n")


def example_learning_path(course_code: str = "B006802"):
    """Example: Personalized learning path generation."""
    console.print("\n[bold cyan]Example 2: Personalized Learning Path[/bold cyan]\n")

    try:
        with AdaptiveTeachingManager() as atm:
            learning_path = atm.get_personalized_learning_path(course_code, limit=5)

        if not learning_path:
            console.print("[yellow]No learning path available. Take some quizzes first![/yellow]\n")
            return

        console.print(f"[bold]Learning Path for {course_code}[/bold]\n")

        for i, item in enumerate(learning_path, 1):
            action_icons = {"review": "🔄", "strengthen": "💪", "learn": "📖", "practice": "✍️"}
            icon = action_icons.get(item["action"], "•")

            console.print(
                f"{i}. {icon} [bold]{item['action'].upper()}[/bold]: {item['knowledge_item']}"
            )
            console.print(f"   Topic: {item['topic']}")
            console.print(f"   Reason: {item['reason']}")
            console.print(f"   Time: {item['estimated_time']} minutes")

            if "mastery" in item:
                mastery_pct = int(item["mastery"] * 100)
                console.print(f"   Current mastery: {mastery_pct}%")

            console.print()

        total_time = sum(item["estimated_time"] for item in learning_path)
        console.print(f"[dim]Total estimated time: {total_time} minutes[/dim]\n")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]\n")


def example_knowledge_gaps(course_code: str = "B006802"):
    """Example: Knowledge gap detection."""
    console.print("\n[bold cyan]Example 3: Knowledge Gap Detection[/bold cyan]\n")

    try:
        with AdaptiveTeachingManager() as atm:
            gaps = atm.detect_knowledge_gaps(course_code)

        if not gaps:
            console.print("[green]✓ No knowledge gaps detected![/green]\n")
            return

        console.print(f"[bold]Knowledge Gaps Analysis for {course_code}[/bold]\n")

        # Group by severity
        high_gaps = [g for g in gaps if g["severity"] == "high"]
        medium_gaps = [g for g in gaps if g["severity"] == "medium"]

        if high_gaps:
            console.print("[bold red]⚠️  High Priority Gaps:[/bold red]\n")
            for gap in high_gaps:
                mastery_pct = int(gap["mastery"] * 100)
                console.print(f"  • {gap['gap']} ({gap['topic']}) - {mastery_pct}% mastery")
                console.print(f"    Recommendation: {gap['recommendation']}\n")

        if medium_gaps:
            console.print("[bold yellow]⚡ Medium Priority Gaps:[/bold yellow]\n")
            for gap in medium_gaps:
                mastery_pct = int(gap["mastery"] * 100)
                console.print(f"  • {gap['gap']} - {mastery_pct}% mastery\n")

        console.print(f"[dim]Total gaps found: {len(gaps)}[/dim]\n")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]\n")


def example_adaptive_recommendations(course_code: str = "B006802", knowledge_item_name: str = None):
    """Example: Get adaptive recommendations for a core loop."""
    console.print("\n[bold cyan]Example 4: Adaptive Recommendations[/bold cyan]\n")

    if not knowledge_item_name:
        # Get first available core loop
        with Database() as db:
            loops = db.get_knowledge_items_by_course(course_code)
            if not loops:
                console.print("[yellow]No core loops found for this course![/yellow]\n")
                return
            knowledge_item_name = loops[0]["name"]

    try:
        with AdaptiveTeachingManager() as atm:
            recommendations = atm.get_adaptive_recommendations(course_code, knowledge_item_name)

        console.print(f"[bold]Recommendations for: {knowledge_item_name}[/bold]\n")

        mastery_pct = int(recommendations["current_mastery"] * 100)
        mastery_emoji = (
            "🟢"
            if recommendations["current_mastery"] >= 0.7
            else "🟡"
            if recommendations["current_mastery"] >= 0.3
            else "🔴"
        )

        console.print(f"{mastery_emoji} Current Mastery: {mastery_pct}%")
        console.print(f"📚 Recommended Depth: [bold]{recommendations['depth']}[/bold]")
        console.print(
            f"📖 Show Prerequisites: {'Yes' if recommendations['show_prerequisites'] else 'No'}"
        )
        console.print(f"✍️  Practice Exercises: {recommendations['practice_count']}")

        if recommendations["focus_areas"]:
            console.print(f"\n🎯 Focus Areas:")
            for area in recommendations["focus_areas"]:
                console.print(f"   • {area}")

        if recommendations["next_review"]:
            console.print(f"\n📅 Next Review: {recommendations['next_review']}")

        console.print()

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]\n")


def example_mastery_summary(course_code: str = "B006802"):
    """Example: Overall mastery summary."""
    console.print("\n[bold cyan]Example 5: Mastery Summary[/bold cyan]\n")

    try:
        with AdaptiveTeachingManager() as atm:
            summary = atm.get_mastery_summary(course_code)

        console.print(f"[bold]Mastery Summary for {course_code}[/bold]\n")

        # Overall statistics
        console.print(f"Total Topics: {summary['total_topics']}")
        console.print(f"  🟢 Mastered: {summary['mastered_topics']}")
        console.print(f"  🟡 In Progress: {summary['in_progress_topics']}")
        console.print(f"  🔴 Weak: {summary['weak_topics']}")
        console.print(f"  ⚪ Not Started: {summary['not_started_topics']}")

        overall_pct = int(summary["overall_mastery"] * 100)
        console.print(f"\n📊 Overall Mastery: {overall_pct}%")

        if summary["topic_details"]:
            console.print(f"\n[bold]Topic Breakdown:[/bold]\n")

            # Show top 5 weakest topics
            weak_topics = [t for t in summary["topic_details"] if t["mastery"] > 0][:5]
            if weak_topics:
                for topic in weak_topics:
                    mastery_pct = int(topic["mastery"] * 100)
                    status_icons = {
                        "mastered": "🟢",
                        "in_progress": "🟡",
                        "weak": "🔴",
                        "not_started": "⚪",
                    }
                    icon = status_icons.get(topic["status"], "•")
                    console.print(f"  {icon} {topic['topic_name']}: {mastery_pct}%")

        console.print()

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]\n")


if __name__ == "__main__":
    console.print("\n[bold magenta]" + "=" * 40 + "[/bold magenta]")
    console.print("[bold magenta]Adaptive Teaching System Examples[/bold magenta]")
    console.print("[bold magenta]" + "=" * 40 + "[/bold magenta]\n")

    # Run examples
    example_depth_selection()

    # The following examples require actual course data
    # Uncomment and specify a valid course code to run them

    # example_learning_path("B006802")
    # example_knowledge_gaps("B006802")
    # example_adaptive_recommendations("B006802")
    # example_mastery_summary("B006802")

    console.print(
        "[dim]To run the other examples, uncomment them and provide a valid course code.[/dim]\n"
    )
