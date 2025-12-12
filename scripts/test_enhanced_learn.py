#!/usr/bin/env python3
"""Test enhanced learning functionality."""

from core.tutor import Tutor
from rich.console import Console
from rich.markdown import Markdown

console = Console()

def test_enhanced_learn():
    """Test enhanced learn with Moore Machine Design."""

    # Initialize tutor with English
    tutor = Tutor(language="en")

    console.print("\n[bold cyan]Testing Enhanced Learning System[/bold cyan]")
    console.print("[yellow]Core Loop: Moore Machine Design[/yellow]")
    console.print("[yellow]Depth: medium[/yellow]")
    console.print("[yellow]Include Prerequisites: Yes[/yellow]\n")

    # Call enhanced learn method
    response = tutor.learn(
        course_code="B006802",
        knowledge_item_id="moore_machine_design",
        explain_concepts=True,
        depth="medium"
    )

    if response.success:
        console.print("[bold green]✓ Learning explanation generated successfully![/bold green]\n")

        # Display the content as markdown
        md = Markdown(response.content)
        console.print(md)

        console.print(f"\n[dim]Metadata: {response.metadata}[/dim]")
    else:
        console.print(f"[bold red]✗ Failed:[/bold red] {response.content}")

if __name__ == "__main__":
    test_enhanced_learn()
