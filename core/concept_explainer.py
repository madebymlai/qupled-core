"""
Concept explainer module for foundational learning.
Identifies prerequisites and generates deep explanations.
"""

from typing import List, Optional
from dataclasses import dataclass
from models.llm_manager import LLMManager


@dataclass
class Concept:
    """Represents a foundational concept."""

    name: str
    description: str
    importance: int  # 1=required, 2=helpful, 3=optional
    examples: List[str]
    analogies: List[str]
    common_misconceptions: List[str]


@dataclass
class ConceptExplanation:
    """Generated explanation for a concept."""

    concept_name: str
    explanation: str
    prerequisites: List[str]
    examples: List[str]
    related_concepts: List[str]


class ConceptExplainer:
    """Explains foundational concepts for learning."""

    # Concept hierarchy: knowledge_item_name → required concepts
    CONCEPT_HIERARCHY = {
        # Finite State Machines (FSM) related
        "moore_machine_design": [
            Concept(
                name="Finite State Machine (FSM)",
                description="A computational model with finite states, transitions, and memory of only the current state",
                importance=1,
                examples=[
                    "Traffic light: states = {Red, Yellow, Green}",
                    "Vending machine: states = {Idle, CoinInserted, DispensingItem}",
                    "Light switch: states = {On, Off}",
                ],
                analogies=[
                    "Like a simple robot that follows a flowchart",
                    "Like a game character with different modes (idle, running, jumping)",
                    "Like a door that can be open or closed, nothing in between",
                ],
                common_misconceptions=[
                    "FSMs can remember past events (they only know current state)",
                    "FSMs can count to infinity (they have finite states)",
                    "FSMs are only for hardware (used in software, protocols, UIs too)",
                ],
            ),
            Concept(
                name="Moore vs Mealy Machines",
                description="Two FSM types differing in when outputs are produced",
                importance=1,
                examples=[
                    "Moore: Traffic light (output = light color, depends only on state)",
                    "Mealy: Vending machine (output = 'dispense item', depends on coin input)",
                    "Moore: Digital clock display (time shown = current state)",
                    "Mealy: ATM (output depends on button pressed + current screen)",
                ],
                analogies=[
                    "Moore: Actor that wears different costumes (state) and audience sees costume",
                    "Mealy: Actor that responds differently based on costume AND what you say",
                    "Moore: Room with colored walls (you see color = state)",
                    "Mealy: Echo chamber that modifies your voice based on where you stand AND what you say",
                ],
                common_misconceptions=[
                    "Moore machines are more powerful (they're equivalent in expressiveness)",
                    "Mealy machines are always better (Moore can be simpler for some problems)",
                    "Outputs in Moore machines can't change (they change with state transitions)",
                ],
            ),
            Concept(
                name="State Transitions",
                description="Rules defining how system moves from one state to another based on inputs",
                importance=1,
                examples=[
                    "Light switch: if state=Off AND input=press → state=On",
                    "Password checker: if state=2CharsEntered AND input=correctChar → state=3CharsEntered",
                    "Elevator: if state=Floor2 AND input=buttonFloor5 → state=MovingUp",
                ],
                analogies=[
                    "Like rules in a board game: if you're on space X and roll Y, move to space Z",
                    "Like choosing paths in a 'choose your own adventure' book",
                ],
                common_misconceptions=[
                    "Transitions happen automatically (they need triggers/inputs)",
                    "Multiple states can be active (only one state at a time)",
                    "Transitions are instantaneous (in hardware they take time)",
                ],
            ),
        ],
        "mealy_machine_design": [
            Concept(
                name="Finite State Machine (FSM)",
                description="A computational model with finite states, transitions, and memory of only the current state",
                importance=1,
                examples=["Traffic light controller", "Vending machine", "Elevator controller"],
                analogies=["Like a simple robot following a flowchart"],
                common_misconceptions=["FSMs can remember unlimited history"],
            ),
            Concept(
                name="Moore vs Mealy Machines",
                description="Mealy outputs depend on current state AND input, Moore only on state",
                importance=1,
                examples=[
                    "Mealy: ATM (output depends on button + screen state)",
                    "Moore: Digital clock (output = time shown = state)",
                ],
                analogies=["Mealy: reactive assistant (responds to you based on context)"],
                common_misconceptions=["Mealy is always more efficient than Moore"],
            ),
        ],
        "conversione_mealy_moore": [
            Concept(
                name="Moore vs Mealy Equivalence",
                description="Any Mealy machine can be converted to Moore and vice versa",
                importance=1,
                examples=["Mealy with 3 states → Moore with 3-9 states (depends on transitions)"],
                analogies=["Like translating between languages: meaning preserved, words differ"],
                common_misconceptions=["Conversion always creates more states"],
            ),
            Concept(
                name="Output Timing",
                description="Moore outputs are stable (state-based), Mealy can react faster (input-based)",
                importance=2,
                examples=[
                    "Moore: Output stable for entire clock cycle",
                    "Mealy: Output can change within clock cycle",
                ],
                analogies=["Moore: Sign that shows one message per room"],
                common_misconceptions=["Faster always means better"],
            ),
        ],
        "minimizzazione_con_tabella_delle_implicazioni": [
            Concept(
                name="State Equivalence",
                description="Two states are equivalent if they produce same outputs and transition to equivalent states",
                importance=1,
                examples=[
                    "States A and B both output '1' and both go to C on input '0'",
                    "If A and B behave identically for all possible input sequences, they're equivalent",
                ],
                analogies=[
                    "Like two identical doors: if they look the same and lead to same places, keep only one"
                ],
                common_misconceptions=[
                    "States with same output are always equivalent (transitions matter too)"
                ],
            ),
            Concept(
                name="Implication Table Method",
                description="Systematic algorithm to find all equivalent state pairs",
                importance=1,
                examples=[
                    "Compare all state pairs (A-B, A-C, B-C, etc.)",
                    "Mark pairs as different if outputs differ or if they lead to different states",
                ],
                analogies=[
                    "Like playing 'spot the difference' game: mark everything that differs, what remains is same"
                ],
                common_misconceptions=["One pass is enough (need to iterate until no changes)"],
            ),
        ],
        # Linear Algebra
        "gauss_elimination": [
            Concept(
                name="System of Linear Equations",
                description="Multiple linear equations solved simultaneously",
                importance=1,
                examples=["2x + 3y = 8 and x - y = 1"],
                analogies=["Like finding intersection of lines on a graph"],
                common_misconceptions=["Systems always have exactly one solution"],
            ),
            Concept(
                name="Row Operations",
                description="Operations that don't change solution: swap rows, multiply row, add rows",
                importance=1,
                examples=["Row2 = Row2 - 2*Row1"],
                analogies=["Like simplifying fractions: looks different, same value"],
                common_misconceptions=["Order of operations doesn't matter"],
            ),
        ],
        "determinante_matrice": [
            Concept(
                name="Matrix Determinant",
                description="Scalar value encoding properties of matrix (invertibility, volume scaling)",
                importance=1,
                examples=["det([[1,2],[3,4]]) = 1*4 - 2*3 = -2"],
                analogies=["Like checking if a transformation squishes space to zero volume"],
                common_misconceptions=["Determinant measures matrix size"],
            )
        ],
        # Concurrent Programming
        "sincronizzazione_semafori": [
            Concept(
                name="Race Condition",
                description="Bug where program behavior depends on timing of uncontrolled events",
                importance=1,
                examples=["Two threads incrementing shared counter without locks"],
                analogies=[
                    "Like two people editing same document simultaneously, changes conflict"
                ],
                common_misconceptions=["Race conditions are rare in practice"],
            ),
            Concept(
                name="Semaphore",
                description="Synchronization primitive: counter with atomic wait/signal operations",
                importance=1,
                examples=["Binary semaphore (mutex): 0 or 1", "Counting semaphore: 0 to N"],
                analogies=["Like a ticket system: wait() takes ticket, signal() returns ticket"],
                common_misconceptions=["Semaphores prevent all deadlocks"],
            ),
        ],
    }

    def __init__(self, llm_manager: Optional[LLMManager] = None, language: str = "en"):
        """Initialize concept explainer.

        Args:
            llm_manager: LLM manager for generating explanations
            language: Output language (any ISO 639-1 code, e.g., "en", "de", "zh")
        """
        self.llm = llm_manager or LLMManager(provider="anthropic")
        self.language = language

    def _language_instruction(self, action: str = "Respond") -> str:
        """Generate dynamic language instruction for any language."""
        return f"{action} in {self.language.upper()} language."

    def get_prerequisites(self, knowledge_item_name: str) -> List[Concept]:
        """Get prerequisite concepts for a core loop.

        Args:
            knowledge_item_name: Name of core loop (e.g., 'moore_machine_design')

        Returns:
            List of prerequisite concepts
        """
        # Normalize name for lookup
        normalized_name = knowledge_item_name.lower().replace(" ", "_")

        # Try direct match
        if normalized_name in self.CONCEPT_HIERARCHY:
            return self.CONCEPT_HIERARCHY[normalized_name]

        # Try fuzzy match
        for key in self.CONCEPT_HIERARCHY.keys():
            if key in normalized_name or normalized_name in key:
                return self.CONCEPT_HIERARCHY[key]

        # No prerequisites found
        return []

    def explain_concept(
        self, concept: Concept, depth: str = "medium", include_examples: bool = True
    ) -> str:
        """Generate detailed explanation for a concept.

        Args:
            concept: Concept to explain
            depth: Explanation depth (basic, medium, advanced)
            include_examples: Whether to include examples

        Returns:
            Formatted explanation string
        """
        sections = []

        # Header
        importance_labels = {1: "REQUIRED", 2: "HELPFUL", 3: "OPTIONAL"}
        importance = importance_labels.get(concept.importance, "REQUIRED")
        sections.append(f"**{concept.name}** [{importance}]")
        sections.append(f"{concept.description}")
        sections.append("")

        # Analogies
        if concept.analogies and depth in ["medium", "advanced"]:
            sections.append("**Think of it like:**")
            for analogy in concept.analogies[:2]:  # Limit to 2 analogies
                sections.append(f"  • {analogy}")
            sections.append("")

        # Examples
        if include_examples and concept.examples:
            sections.append("**Examples:**")
            for example in concept.examples[:3]:  # Limit to 3 examples
                sections.append(f"  • {example}")
            sections.append("")

        # Common misconceptions
        if concept.common_misconceptions and depth in ["medium", "advanced"]:
            sections.append("**⚠️ Common Misconceptions:**")
            for misconception in concept.common_misconceptions:
                sections.append(f"  ✗ {misconception}")
            sections.append("")

        return "\n".join(sections)

    def explain_prerequisites(self, knowledge_item_name: str, depth: str = "medium") -> str:
        """Generate formatted explanation of all prerequisites.

        Args:
            knowledge_item_name: Core loop name
            depth: Explanation depth

        Returns:
            Formatted prerequisite explanations
        """
        prerequisites = self.get_prerequisites(knowledge_item_name)

        if not prerequisites:
            return ""

        sections = []

        # Header
        if self.language == "it":
            sections.append("## 📚 CONCETTI FONDAMENTALI")
            sections.append("Prima di imparare questa procedura, è importante comprendere:")
        else:
            sections.append("## 📚 FOUNDATIONAL CONCEPTS")
            sections.append("Before learning this procedure, it's important to understand:")
        sections.append("")

        # Required concepts first
        required = [c for c in prerequisites if c.importance == 1]
        helpful = [c for c in prerequisites if c.importance == 2]
        optional = [c for c in prerequisites if c.importance == 3]

        for concepts, label in [
            (required, "Required"),
            (helpful, "Helpful"),
            (optional, "Optional"),
        ]:
            if not concepts:
                continue

            if len(concepts) > 0 and depth == "advanced":
                label_it = {"Required": "Obbligatori", "Helpful": "Utili", "Optional": "Opzionali"}
                header = label_it.get(label, label) if self.language == "it" else label
                sections.append(f"### {header}:")
                sections.append("")

            for concept in concepts:
                explanation = self.explain_concept(concept, depth=depth, include_examples=True)
                sections.append(explanation)

        return "\n".join(sections)

    def generate_deep_explanation(
        self, knowledge_item_name: str, procedure_steps: List[str], topic_name: str
    ) -> str:
        """Generate LLM-based deep explanation with reasoning.

        Args:
            knowledge_item_name: Name of core loop
            procedure_steps: List of procedure steps
            topic_name: Topic name for context

        Returns:
            Deep explanation with WHY reasoning
        """
        prerequisites = self.get_prerequisites(knowledge_item_name)
        prereq_context = ""
        if prerequisites:
            prereq_names = [c.name for c in prerequisites if c.importance == 1]
            prereq_context = f"\nPrerequisite concepts: {', '.join(prereq_names)}"

        prompt = f"""{self._language_instruction("Respond")}

You are an expert educator explaining a technical procedure to students.

TOPIC: {topic_name}
PROCEDURE: {knowledge_item_name}{prereq_context}

PROCEDURE STEPS:
{self._format_steps(procedure_steps)}

Your task is to provide a DEEP, PEDAGOGICAL explanation that:

1. **EXPLAINS THE WHY** - For each step, explain:
   - WHY this step is necessary (what problem does it solve?)
   - WHY this method works (what's the underlying reasoning?)
   - WHY we do it in this order (why not earlier/later?)

2. **SHOWS THE HOW** - For each step:
   - HOW to perform the step concretely
   - HOW to recognize when you've done it correctly
   - HOW to debug if something goes wrong

3. **TEACHES DECISION-MAKING** - Explain:
   - WHEN to use this procedure (what problems does it solve?)
   - WHEN NOT to use it (what are the limitations?)
   - HOW to recognize which approach to use

4. **HIGHLIGHTS COMMON MISTAKES** - For each step:
   - What mistakes do students typically make?
   - Why do these mistakes happen?
   - How to avoid them?

5. **BUILDS INTUITION** - Use:
   - Analogies to familiar concepts
   - Visual descriptions (describe what to imagine)
   - Concrete examples throughout

Make your explanation:
- Start from first principles (don't assume prior knowledge)
- Be conversational and engaging
- Use "you" to address the student
- Build complexity gradually
- Include practical tips from experience

Structure your response as:
1. Big Picture (What is this and why does it matter?)
2. Step-by-Step Breakdown (WHY + HOW for each step)
3. Common Pitfalls (Mistakes and how to avoid them)
4. When to Use This (Decision-making guidance)
5. Practice Tips (How to master this skill)
"""

        response = self.llm.generate(
            prompt=prompt, model=self.llm.primary_model, temperature=0.4, max_tokens=3000
        )

        if response.success:
            return response.text
        else:
            return f"Failed to generate deep explanation: {response.error}"

    def _format_steps(self, steps: List[str]) -> str:
        """Format procedure steps."""
        if not steps:
            return "No steps provided."
        return "\n".join([f"{i + 1}. {step}" for i, step in enumerate(steps)])
