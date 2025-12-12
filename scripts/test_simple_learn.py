#!/usr/bin/env python3
"""Test simple learning without prerequisites."""

from core.tutor import Tutor
from rich.console import Console
from rich.markdown import Markdown

console = Console()

def test_simple_learn():
    """Test learn without prerequisites first."""

    tutor = Tutor(language="en")

    console.print("\n[bold cyan]Testing Enhanced Learning (No Prerequisites)[/bold cyan]")
    console.print("[yellow]Core Loop: conversione_mealy_moore[/yellow]\n")

    # Call with explain_concepts=False to skip prerequisites
    response = tutor.learn(
        course_code="B006802",
        knowledge_item_id="conversione_mealy_moore",
        explain_concepts=False,
        depth="basic"  # Use basic depth for faster response
    )

    if response.success:
        console.print("[bold green]✓ Success![/bold green]\n")
        console.print("[dim]" + response.content[:500] + "...[/dim]")
        console.print(f"\n[dim]Full length: {len(response.content)} chars[/dim]")
        console.print(f"[dim]Metadata: {response.metadata}[/dim]")
    else:
        console.print(f"[bold red]✗ Failed:[/bold red] {response.content}")

if __name__ == "__main__":
    test_simple_learn()
