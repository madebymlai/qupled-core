"""
Theory Concept Visualization for Examina.
Renders concept dependency graphs in multiple formats (ASCII, Mermaid, JSON).
"""

import json
from typing import Dict, List
from core.concept_graph import ConceptGraph


class ConceptVisualizer:
    """Visualizes concept dependency graphs."""

    def __init__(self):
        pass

    def render_ascii(self, graph: ConceptGraph, max_width: int = 80) -> str:
        """
        Render graph as ASCII art for CLI display.

        Uses topological sort to order concepts, shows dependencies with arrows.

        Args:
            graph: ConceptGraph to visualize
            max_width: Maximum width for text wrapping

        Returns:
            ASCII art string
        """
        learning_order = graph.topological_sort()
        if not learning_order:
            return "Error: Cycle detected in concept graph!"

        lines = []
        lines.append("=" * max_width)
        lines.append("Concept Dependency Graph")
        lines.append("=" * max_width)
        lines.append("")

        # Group concepts by level (number of prerequisites in chain)
        levels = self._compute_levels(graph, learning_order)

        # Show concepts in learning order, grouped by level
        current_level = 0
        for i, concept_id in enumerate(learning_order, 1):
            concept = graph.concepts[concept_id]
            prereqs = graph.get_prerequisites(concept_id)
            level = levels.get(concept_id, 0)

            # Show level header if level changed
            if level > current_level:
                lines.append("")
                lines.append(f"--- Level {level} (requires Level {level - 1}) ---")
                lines.append("")
                current_level = level
            elif level == 0 and i == 1:
                lines.append("--- Level 0 (Foundation - no prerequisites) ---")
                lines.append("")
                current_level = 0

            # Concept info
            concept_str = f"[{i}] {concept.name}"
            if concept.exercise_count > 0:
                concept_str += f" ({concept.exercise_count} exercise{'s' if concept.exercise_count != 1 else ''})"

            lines.append(concept_str)

            # Show prerequisites with arrows
            if prereqs:
                prereq_names = [f"[{learning_order.index(p.id) + 1}] {p.name}" for p in prereqs]
                lines.append(f"    ↑ Requires: {', '.join(prereq_names)}")

            # Show dependents
            dependents = graph.get_dependents(concept_id)
            if dependents:
                dep_count = len(dependents)
                lines.append(f"    ↓ Enables: {dep_count} concept{'s' if dep_count != 1 else ''}")

            lines.append("")

        # Summary
        lines.append("=" * max_width)
        lines.append(f"Total: {len(graph.concepts)} concepts, {len(graph.edges)} dependencies")
        lines.append(f"Maximum depth: {max(levels.values()) if levels else 0} levels")
        lines.append("=" * max_width)

        return "\n".join(lines)

    def _compute_levels(self, graph: ConceptGraph, learning_order: List[str]) -> Dict[str, int]:
        """
        Compute the level of each concept (longest path from foundation).

        Args:
            graph: ConceptGraph
            learning_order: Topologically sorted concept IDs

        Returns:
            Dict mapping concept_id to level (0 = foundation)
        """
        levels = {}

        # Process in topological order
        for concept_id in learning_order:
            prereqs = graph.get_prerequisites(concept_id)
            if not prereqs:
                # Foundation concept
                levels[concept_id] = 0
            else:
                # Level = max(prerequisite levels) + 1
                prereq_levels = [levels.get(p.id, 0) for p in prereqs]
                levels[concept_id] = max(prereq_levels) + 1

        return levels

    def render_mermaid(self, graph: ConceptGraph) -> str:
        """
        Render graph as Mermaid diagram syntax (for future web display).

        https://mermaid.js.org/

        Args:
            graph: ConceptGraph to visualize

        Returns:
            Mermaid syntax string
        """
        lines = []
        lines.append("graph TD")
        lines.append("")

        # Add nodes
        for concept_id, concept in graph.concepts.items():
            safe_id = concept_id.replace(" ", "_").replace("-", "_").replace(".", "_")
            # Escape quotes in name
            safe_name = concept.name.replace('"', "'")
            node_label = f"{safe_name}"
            if concept.exercise_count > 0:
                node_label += f"\\n({concept.exercise_count})"

            lines.append(f'    {safe_id}["{node_label}"]')

        lines.append("")

        # Add edges (prerequisite -> dependent)
        for src, dst in graph.edges:
            safe_src = src.replace(" ", "_").replace("-", "_").replace(".", "_")
            safe_dst = dst.replace(" ", "_").replace("-", "_").replace(".", "_")
            lines.append(f"    {safe_src} --> {safe_dst}")

        lines.append("")

        # Add styling
        lines.append("    classDef default fill:#f9f,stroke:#333,stroke-width:2px")

        return "\n".join(lines)

    def export_json(self, graph: ConceptGraph) -> str:
        """
        Export graph as JSON for API/web consumption.

        Args:
            graph: ConceptGraph to export

        Returns:
            JSON string
        """
        # Compute levels for additional metadata
        learning_order = graph.topological_sort()
        levels = self._compute_levels(graph, learning_order) if learning_order else {}

        data = {
            "metadata": {
                "total_concepts": len(graph.concepts),
                "total_dependencies": len(graph.edges),
                "max_depth": max(levels.values()) if levels else 0,
                "has_cycles": learning_order is None,
            },
            "concepts": [
                {
                    "id": c.id,
                    "name": c.name,
                    "description": c.description,
                    "exercise_count": c.exercise_count,
                    "prerequisites": c.prerequisites,
                    "level": levels.get(c.id, 0),
                    "learning_order": learning_order.index(c.id) + 1
                    if learning_order and c.id in learning_order
                    else None,
                }
                for c in graph.concepts.values()
            ],
            "edges": [{"from": src, "to": dst} for src, dst in graph.edges],
        }

        if learning_order:
            data["learning_order"] = learning_order

        return json.dumps(data, indent=2)

    def render_learning_path(self, graph: ConceptGraph, target_concept_id: str) -> str:
        """
        Render the learning path to reach a specific concept.

        Shows all prerequisites and the order to learn them.

        Args:
            graph: ConceptGraph
            target_concept_id: ID of target concept

        Returns:
            ASCII art showing learning path
        """
        if target_concept_id not in graph.concepts:
            return f"Error: Concept '{target_concept_id}' not found in graph."

        target = graph.concepts[target_concept_id]
        lines = []
        lines.append(f"Learning Path to: {target.name}")
        lines.append("=" * 60)
        lines.append("")

        # Find all prerequisites recursively
        visited = set()
        path = []

        def collect_prerequisites(concept_id: str):
            if concept_id in visited:
                return
            visited.add(concept_id)

            prereqs = graph.get_prerequisites(concept_id)
            for prereq in prereqs:
                collect_prerequisites(prereq.id)

            path.append(concept_id)

        collect_prerequisites(target_concept_id)

        if len(path) == 1:
            lines.append("This is a foundation concept with no prerequisites!")
        else:
            lines.append(f"You need to learn {len(path) - 1} concept(s) first:")
            lines.append("")

            for i, concept_id in enumerate(path, 1):
                concept = graph.concepts[concept_id]
                is_target = concept_id == target_concept_id

                prefix = "→" if is_target else f"{i}."
                marker = " ← YOU ARE HERE" if is_target else ""

                lines.append(
                    f"{prefix} {concept.name} ({concept.exercise_count} exercises){marker}"
                )

                if not is_target:
                    # Show what this enables
                    next_in_path = [
                        graph.concepts[pid]
                        for pid in path[i:]
                        if pid in graph.get_dependents(concept_id)
                    ]
                    if next_in_path:
                        lines.append(f"   (enables: {', '.join([c.name for c in next_in_path])})")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)
