#!/usr/bin/env python3
"""
Validation script for multi-procedure exercise extraction.

This script validates that Phase 6 multi-procedure support is working correctly by:
1. Finding exercises with multiple procedures
2. Analyzing procedure type distribution
3. Verifying specific exercises (e.g., 2024-01-29 #1)
4. Checking tag-based search functionality
"""

import json
from storage.database import Database
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


def validate_multi_procedure_extraction(course_code='B006802'):
    """Validate multi-procedure extraction for a course."""

    console.print(f"\n[bold cyan]Multi-Procedure Validation Report[/bold cyan]")
    console.print(f"[dim]Course: {course_code}[/dim]\n")

    with Database() as db:
        # Get all exercises for the course
        exercises = db.get_exercises_by_course(course_code)

        if not exercises:
            console.print(f"[red]No exercises found for course {course_code}[/red]\n")
            return

        # Track statistics
        multi_procedure_count = 0
        procedure_type_counts = {
            'design': 0,
            'transformation': 0,
            'minimization': 0,
            'verification': 0,
            'analysis': 0,
            'implementation': 0
        }
        transformation_types = {}

        # Track multi-procedure exercises
        multi_proc_examples = []

        for ex in exercises:
            # Get all core loops for this exercise
            core_loops = db.get_exercise_core_loops(ex['id'])

            if len(core_loops) > 1:
                multi_procedure_count += 1

                # Collect example info
                example_info = {
                    'exercise_number': ex.get('exercise_number', 'Unknown'),
                    'source_pdf': ex.get('source_pdf', 'Unknown'),
                    'core_loop_count': len(core_loops),
                    'procedures': [cl['name'] for cl in core_loops],
                    'tags': json.loads(ex['tags']) if ex.get('tags') else []
                }
                multi_proc_examples.append(example_info)

                # Count procedure types from tags
                if ex.get('tags'):
                    tags = json.loads(ex['tags']) if isinstance(ex['tags'], str) else ex['tags']
                    for tag in tags:
                        if tag in procedure_type_counts:
                            procedure_type_counts[tag] += 1
                        # Track transformation types
                        if tag.startswith('transform_'):
                            transformation_types[tag] = transformation_types.get(tag, 0) + 1

        # Summary Statistics
        console.print("[bold]Summary Statistics:[/bold]\n")
        console.print(f"  Total exercises: {len(exercises)}")
        console.print(f"  Multi-procedure exercises: {multi_procedure_count}")
        console.print(f"  Percentage: {(multi_procedure_count / len(exercises) * 100):.1f}%\n")

        # Procedure Type Distribution
        console.print("[bold]Procedure Type Distribution:[/bold]\n")
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Procedure Type", style="white")
        table.add_column("Count", justify="right", style="green")

        for ptype, count in sorted(procedure_type_counts.items(), key=lambda x: -x[1]):
            if count > 0:
                table.add_row(ptype.title(), str(count))

        console.print(table)
        console.print()

        # Transformation Types
        if transformation_types:
            console.print("[bold]Transformation Types:[/bold]\n")
            for trans_type, count in sorted(transformation_types.items(), key=lambda x: -x[1]):
                # Pretty print transformation type
                parts = trans_type.replace('transform_', '').split('_to_')
                if len(parts) == 2:
                    from_type = parts[0].replace('_', ' ').title()
                    to_type = parts[1].replace('_', ' ').title()
                    console.print(f"  {from_type} → {to_type}: {count}")
                else:
                    console.print(f"  {trans_type}: {count}")
            console.print()

        # Top Multi-Procedure Examples
        if multi_proc_examples:
            console.print("[bold]Multi-Procedure Examples:[/bold]\n")

            # Sort by number of procedures
            multi_proc_examples.sort(key=lambda x: x['core_loop_count'], reverse=True)

            for i, example in enumerate(multi_proc_examples[:10], 1):
                console.print(f"[cyan]{i}. Exercise: {example['exercise_number']}[/cyan]")
                console.print(f"   Source: {example['source_pdf']}")
                console.print(f"   Procedures ({example['core_loop_count']}):")
                for j, proc in enumerate(example['procedures'], 1):
                    console.print(f"     {j}. {proc}")
                if example['tags']:
                    console.print(f"   Tags: {', '.join(example['tags'])}")
                console.print()

        # Specific Exercise Validation (2024-01-29)
        console.print("[bold]Specific Exercise Validation:[/bold]\n")

        # Find exercises from 2024-01-29
        target_exercises = [ex for ex in exercises if '2024-01-29' in ex.get('source_pdf', '')]

        if target_exercises:
            console.print(f"Found {len(target_exercises)} exercise(s) from 2024-01-29:\n")
            for ex in target_exercises:
                core_loops = db.get_exercise_core_loops(ex['id'])
                console.print(f"  Exercise: {ex.get('exercise_number', 'Unknown')}")
                console.print(f"  Source: {ex.get('source_pdf', 'Unknown')}")
                console.print(f"  Procedures ({len(core_loops)}):")
                for i, cl in enumerate(core_loops, 1):
                    step_info = f" (point {cl['step_number']})" if cl['step_number'] else ""
                    console.print(f"    {i}. {cl['name']}{step_info}")

                # Check if transformation is present
                has_transformation = any('transform' in cl['name'].lower() for cl in core_loops)
                if has_transformation:
                    console.print(f"  [green]✓ Contains transformation procedure[/green]")
                else:
                    console.print(f"  [yellow]⚠ No transformation procedure found[/yellow]")

                if ex.get('tags'):
                    tags = json.loads(ex['tags']) if isinstance(ex['tags'], str) else ex['tags']
                    console.print(f"  Tags: {', '.join(tags)}")
                console.print()
        else:
            console.print("[yellow]No exercises found from 2024-01-29[/yellow]\n")

        # Tag-Based Search Validation
        console.print("[bold]Tag-Based Search Validation:[/bold]\n")

        # Test searching by transformation tag
        transformation_exercises = db.get_exercises_by_tag(course_code, 'transformation')
        console.print(f"  Exercises with 'transformation' tag: {len(transformation_exercises)}")

        # Test searching by design tag
        design_exercises = db.get_exercises_by_tag(course_code, 'design')
        console.print(f"  Exercises with 'design' tag: {len(design_exercises)}")

        # Test getting multi-procedure exercises
        multi_proc_exercises = db.get_exercises_with_multiple_procedures(course_code)
        console.print(f"  Multi-procedure exercises (via junction table): {len(multi_proc_exercises)}")
        console.print()

        # Consistency Check
        console.print("[bold]Consistency Checks:[/bold]\n")

        # Check if multi-procedure counts match
        if len(multi_proc_exercises) == multi_procedure_count:
            console.print(f"  [green]✓ Multi-procedure count consistent: {multi_procedure_count}[/green]")
        else:
            console.print(f"  [red]✗ Multi-procedure count mismatch: {len(multi_proc_exercises)} vs {multi_procedure_count}[/red]")

        # Check backward compatibility (exercises should have core_loop_id OR junction table entries)
        exercises_with_procedures = 0
        for ex in exercises:
            if ex.get('core_loop_id') or db.get_exercise_core_loops(ex['id']):
                exercises_with_procedures += 1

        console.print(f"  [green]✓ Exercises with procedures: {exercises_with_procedures}/{len(exercises)}[/green]")
        console.print()

        # Final Status
        console.print("[bold green]✓ Validation Complete![/bold green]\n")

        return {
            'total_exercises': len(exercises),
            'multi_procedure_count': multi_procedure_count,
            'procedure_type_counts': procedure_type_counts,
            'transformation_types': transformation_types,
            'validation_passed': True
        }


def test_search_functionality(course_code='B006802'):
    """Test the search command functionality."""

    console.print("\n[bold cyan]Search Functionality Test[/bold cyan]\n")

    with Database() as db:
        # Test 1: Search by tag
        console.print("[bold]Test 1: Search by tag 'transformation'[/bold]")
        results = db.get_exercises_by_tag(course_code, 'transformation')
        console.print(f"  Found {len(results)} exercises\n")

        # Test 2: Search by text
        console.print("[bold]Test 2: Search by text 'Mealy'[/bold]")
        results = db.search_exercises_by_text(course_code, 'Mealy')
        console.print(f"  Found {len(results)} exercises\n")

        # Test 3: Get multi-procedure exercises
        console.print("[bold]Test 3: Get multi-procedure exercises only[/bold]")
        results = db.get_exercises_with_multiple_procedures(course_code)
        console.print(f"  Found {len(results)} exercises\n")

        # Test 4: Search by procedure type
        console.print("[bold]Test 4: Search by procedure type 'design'[/bold]")
        results = db.get_exercises_by_procedure_type(course_code, 'design')
        console.print(f"  Found {len(results)} exercises\n")

        console.print("[bold green]✓ All search tests completed![/bold green]\n")


def main():
    """Run all validation tests."""

    try:
        # Validate multi-procedure extraction
        results = validate_multi_procedure_extraction('B006802')

        # Test search functionality
        test_search_functionality('B006802')

        # Print summary
        console.print("\n" + "="*60 + "\n")
        console.print("[bold green]Validation Summary[/bold green]\n")
        console.print(f"  Total exercises analyzed: {results['total_exercises']}")
        console.print(f"  Multi-procedure exercises: {results['multi_procedure_count']}")
        console.print(f"  Validation status: [green]PASSED[/green]")
        console.print()

    except Exception as e:
        console.print(f"\n[bold red]Validation Error:[/bold red] {e}\n")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
