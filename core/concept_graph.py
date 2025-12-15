"""
Theory Concept Dependency Graph Builder for Examina.
Dynamically discovers prerequisite relationships between concepts using LLM.
Fully generic - works for ANY subject (CS, Math, Physics, Chemistry, etc.).
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class Concept:
    """Represents a theory concept."""

    id: str  # e.g., "eigenvalues"
    name: str  # e.g., "Eigenvalues and Eigenvectors"
    description: Optional[str] = None
    exercise_count: int = 0
    prerequisites: List[str] = field(default_factory=list)  # List of concept IDs

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, Concept) and self.id == other.id


@dataclass
class ConceptGraph:
    """Directed acyclic graph of concept dependencies."""

    concepts: Dict[str, Concept] = field(default_factory=dict)
    edges: List[Tuple[str, str]] = field(default_factory=list)  # (prerequisite_id, concept_id)

    def add_concept(self, concept: Concept):
        """Add concept to graph."""
        self.concepts[concept.id] = concept

    def add_dependency(self, prerequisite_id: str, concept_id: str):
        """Add prerequisite relationship."""
        edge = (prerequisite_id, concept_id)
        if edge not in self.edges:
            self.edges.append(edge)
            # Also add to concept's prerequisites list
            if concept_id in self.concepts:
                if prerequisite_id not in self.concepts[concept_id].prerequisites:
                    self.concepts[concept_id].prerequisites.append(prerequisite_id)

    def get_prerequisites(self, concept_id: str) -> List[Concept]:
        """Get all direct prerequisites for a concept."""
        prereq_ids = [src for src, dst in self.edges if dst == concept_id]
        return [self.concepts[id] for id in prereq_ids if id in self.concepts]

    def get_dependents(self, concept_id: str) -> List[Concept]:
        """Get all concepts that depend on this one."""
        dependent_ids = [dst for src, dst in self.edges if src == concept_id]
        return [self.concepts[id] for id in dependent_ids if id in self.concepts]

    def topological_sort(self) -> Optional[List[str]]:
        """
        Return learning order (prerequisites first).
        Uses Kahn's algorithm for topological sort.
        Returns None if cycle detected.
        """
        # Build adjacency list and in-degree map
        adj = {id: [] for id in self.concepts}
        in_degree = {id: 0 for id in self.concepts}

        for src, dst in self.edges:
            if src in adj and dst in adj:  # Both concepts exist
                adj[src].append(dst)
                in_degree[dst] += 1

        # Start with concepts that have no prerequisites
        queue = [id for id in self.concepts if in_degree[id] == 0]
        result = []

        while queue:
            # Sort to ensure deterministic ordering
            queue.sort()
            current = queue.pop(0)
            result.append(current)

            for neighbor in adj[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # If result doesn't contain all concepts, there's a cycle
        return result if len(result) == len(self.concepts) else None

    def detect_cycles(self) -> List[List[str]]:
        """Detect cycles in the graph (should be none for valid DAG)."""
        visited = set()
        rec_stack = set()
        cycles = []

        def dfs(node: str, path: List[str]):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for src, dst in self.edges:
                if src == node:
                    if dst in rec_stack:
                        # Cycle found
                        cycle_start = path.index(dst)
                        cycle = path[cycle_start:] + [dst]
                        if cycle not in cycles:
                            cycles.append(cycle)
                    elif dst not in visited:
                        dfs(dst, path[:])

            rec_stack.remove(node)

        for concept_id in self.concepts:
            if concept_id not in visited:
                dfs(concept_id, [])

        return cycles


class ConceptGraphBuilder:
    """Builds concept dependency graphs from course data."""

    def __init__(self, llm_manager):
        """Initialize with LLM manager for prerequisite discovery.

        Args:
            llm_manager: LLMManager instance for querying prerequisites
        """
        self.llm = llm_manager
        self._cache = {}
        logger.info("ConceptGraphBuilder initialized with LLM-based prerequisite discovery")

    def build_from_course(self, course_code: str) -> ConceptGraph:
        """
        Build concept graph for a course.

        1. Extract all theory concepts from exercises
        2. Use LLM to discover prerequisites for each concept
        3. Build directed graph

        Args:
            course_code: Course code to build graph for

        Returns:
            ConceptGraph with all concepts and dependencies
        """
        from storage.database import Database

        graph = ConceptGraph()

        with Database() as db:
            # Get all theory exercises
            cursor = db.conn.execute(
                """
                SELECT id, text, theory_metadata, exercise_type
                FROM exercises
                WHERE course_code = ?
                AND (exercise_type = 'theory' OR exercise_type = 'proof' OR exercise_type = 'hybrid')
            """,
                (course_code,),
            )

            exercises = cursor.fetchall()
            concept_exercises = {}

            logger.info(f"Found {len(exercises)} theory exercises for {course_code}")

            # Extract concepts from exercises
            for ex_id, text, metadata_json, ex_type in exercises:
                if not metadata_json:
                    continue

                try:
                    metadata = (
                        json.loads(metadata_json)
                        if isinstance(metadata_json, str)
                        else metadata_json
                    )

                    # Try to extract concept information from metadata
                    concept_id = metadata.get("concept_id")
                    concept_name = metadata.get("concept_name")

                    # If not in metadata, try to extract from exercise text using LLM
                    if not concept_id or not concept_name:
                        concept_info = self._extract_concept_from_text(text, course_code)
                        if concept_info:
                            concept_id = concept_info["id"]
                            concept_name = concept_info["name"]

                    if concept_id and concept_name:
                        if concept_id not in concept_exercises:
                            concept_exercises[concept_id] = {
                                "name": concept_name,
                                "exercises": [],
                                "description": metadata.get("description"),
                            }
                        concept_exercises[concept_id]["exercises"].append((ex_id, text))
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse metadata for exercise {ex_id}: {e}")
                    continue

            logger.info(f"Extracted {len(concept_exercises)} unique concepts")

            # Create concept nodes
            for concept_id, data in concept_exercises.items():
                concept = Concept(
                    id=concept_id,
                    name=data["name"],
                    description=data.get("description"),
                    exercise_count=len(data["exercises"]),
                )
                graph.add_concept(concept)

            # Discover prerequisites using LLM
            logger.info("Discovering prerequisites using LLM...")
            for concept_id, concept in graph.concepts.items():
                prerequisites = self._discover_prerequisites(
                    concept.name, list(graph.concepts.values())
                )

                for prereq_id in prerequisites:
                    if prereq_id in graph.concepts:
                        graph.add_dependency(prereq_id, concept_id)
                        logger.debug(f"Added dependency: {prereq_id} -> {concept_id}")

        logger.info(
            f"Built graph with {len(graph.concepts)} concepts and {len(graph.edges)} dependencies"
        )
        return graph

    def _extract_concept_from_text(
        self, exercise_text: str, course_code: str
    ) -> Optional[Dict[str, str]]:
        """
        Extract concept information from exercise text using LLM.
        Fallback for exercises without proper metadata.

        Args:
            exercise_text: Exercise text
            course_code: Course code for context

        Returns:
            Dict with 'id' and 'name' or None
        """
        # Cache key
        cache_key = f"extract:{course_code}:{exercise_text[:100]}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        prompt = f"""Given this theory exercise, identify the main concept it tests.

Exercise:
{exercise_text[:500]}

Respond with ONLY the concept name (e.g., "Eigenvalues", "Boolean Algebra", "State Machines").
If no clear concept, respond with "Unknown"."""

        response = self.llm.generate(prompt=prompt, temperature=0.0, max_tokens=50)

        if not response.success or response.text.strip().lower() == "unknown":
            return None

        concept_name = response.text.strip()
        concept_id = concept_name.lower().replace(" ", "_").replace("-", "_")

        result = {"id": concept_id, "name": concept_name}
        self._cache[cache_key] = result
        return result

    def _discover_prerequisites(self, concept_name: str, all_concepts: List[Concept]) -> List[str]:
        """
        Use LLM to discover which concepts are prerequisites.

        IMPORTANT: This is GENERIC - works for any subject/concept.

        Args:
            concept_name: Name of the concept to find prerequisites for
            all_concepts: List of all concepts in the course

        Returns:
            List of prerequisite concept IDs
        """
        # Cache key
        concept_names = sorted([c.name for c in all_concepts])
        cache_key = f"prereq:{concept_name}:{','.join(concept_names)}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Build prompt
        other_concepts = [c for c in all_concepts if c.name != concept_name]
        if not other_concepts:
            return []

        concept_list = "\n".join([f"- {c.name}" for c in other_concepts])

        prompt = f"""Given the concept "{concept_name}", which of these concepts are DIRECT prerequisites (must be understood first)?

Available concepts:
{concept_list}

List ONLY the concept names that are direct prerequisites, one per line. If none, respond with "None".

Rules:
- Only list DIRECT prerequisites (not transitive)
- Concept must be logically required before understanding "{concept_name}"
- Be conservative - only list clear dependencies
- Do not infer prerequisites not in the list

Prerequisites:"""

        response = self.llm.generate(prompt=prompt, temperature=0.0, max_tokens=200)

        if not response.success:
            logger.warning(f"Failed to get prerequisites for {concept_name}")
            return []

        # Parse response
        lines = response.text.strip().split("\n")
        prerequisites = []

        for line in lines:
            line = line.strip(" -•*").strip()
            if line.lower() == "none":
                break

            # Find matching concept (fuzzy matching)
            for concept in other_concepts:
                # Check if concept name is in line (case insensitive)
                if concept.name.lower() in line.lower():
                    if concept.id not in prerequisites:
                        prerequisites.append(concept.id)
                    break

        # Cache result
        self._cache[cache_key] = prerequisites
        logger.debug(f"Discovered prerequisites for {concept_name}: {prerequisites}")
        return prerequisites
