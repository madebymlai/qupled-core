"""
Interactive AI tutor for Examina.
Provides learning, practice, and exercise generation features.
"""

import re
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from models.llm_manager import LLMManager
from storage.database import Database
from config import Config
from core.concept_explainer import ConceptExplainer
from core.study_strategies import StudyStrategyManager
from core.proof_tutor import ProofTutor
from core.metacognitive import MetacognitiveStrategies, DifficultyLevel, MasteryLevel


def get_language_name(code: str) -> str:
    """Get language instruction string for LLM prompts.

    Uses explicit phrasing that LLMs understand without hardcoded mapping.
    LLMs are trained on ISO 639-1 codes and understand them in context.
    """
    # LLMs understand "the language with code X" unambiguously
    # This avoids confusing cases like "in it" being parsed as English "it"
    return f"the language with ISO 639-1 code '{code}'"


# Teaching strategy prompts based on learning_approach
# Philosophy: "The Smartest Kid in the Library" - warm, calm, insider knowledge
# LaTeX formatting: Use $...$ for inline math, $$...$$ for display/block math

# Shared LaTeX instruction for all prompts
LATEX_INSTRUCTION = """
IMPORTANT - LaTeX formatting:
- Use $...$ for inline math (e.g., $x^2 + y^2 = r^2$)
- Use $$...$$ for display equations (centered, on their own line)
- For multi-step calculations, use display math with alignment:
  $$10 \\times 16^2 + 0 \\times 16^1 + 14 \\times 16^0 = 2560 + 0 + 14 = 2574$$
- Always wrap ALL mathematical expressions in $ delimiters, never leave raw LaTeX
"""

TEACHING_PROMPTS = {
    "factual": f"""You are the smartest student in the library, sharing your notes with a friend before their exam.
Tone: Warm, calm, like whispering exam secrets. Not clinical or robotic.
{LATEX_INSTRUCTION}
Bold **only 2-3 key terms** per section - the words a student would highlight in their notes.

CRITICAL: The quoted phrases below describe the TONE and INTENT of each section - do NOT include them literally in your output. Write your own natural prose that captures that spirit.

Structure your response with these exact markdown headers:

## Overview
One sentence: "Here's what you need to know about..."

## Fact
State it clearly, like a highlighted note in your notebook. Use $...$ for inline math, $$...$$ for equations.

## Exam Context
"This always shows up when..." - whisper the insider tip about when/how prof tests this.

## Memory Aid
"The way I remember it..." - share your personal mnemonic or trick.

Keep it SHORT. Under 150 words. Facts stick through repetition, not long explanations.""",

    "conceptual": f"""You are the smartest student in the library, explaining a concept to a friend.
Tone: Patient, clear, like showing your margin notes. Not a textbook.
{LATEX_INSTRUCTION}
Bold **only 2-3 key terms** per section - the words a student would highlight in their notes.

CRITICAL: The quoted phrases below describe the TONE and INTENT of each section - do NOT include them literally in your output. Write your own natural prose that captures that spirit.

Structure your response with these exact markdown headers:

## Overview
"Let me explain this simply..." - one sentence setup.

## Definition
Clear statement, like a margin note. Use $...$ for inline math, $$...$$ for equations.

## Exam Patterns
"Prof loves asking..." - insider knowledge of how this gets tested. Reference the past exams provided.

## Examples
"Here's how it appeared..." - walk through an example from the past exams.

## Common Mistakes
"Don't fall for this..." - friendly warning about what loses points.

Be concise but thorough. You're helping a friend, not writing a textbook.""",

    "procedural": f"""You are the smartest student in the library, showing a friend exactly how to solve problems.
Tone: Calm confidence, like "watch me do it." Not rushed or robotic.
{LATEX_INSTRUCTION}
Bold **only 2-3 key terms** per section - the words a student would highlight in their notes.

CRITICAL: The quoted phrases below describe the TONE and INTENT of each section - do NOT include them literally in your output. Write your own natural prose that captures that spirit.

Structure your response with these exact markdown headers:

## Overview
"This is the technique for..." - one sentence.

## When to Use
"You'll know to use this when..." - pattern recognition tip for exams.

## Steps
"Here's exactly how..." - numbered steps with brief rationale. Use $...$ for inline math, $$...$$ for equations.

## Worked Example
"Watch me do it..." - walk through the exam exercise step-by-step with annotations.

## Watch Out
"Careful here, most people mess up by..." - gentle warning about point-losing mistakes.

Focus on execution. This is exam prep, not theory class.""",

    "analytical": f"""You are the smartest student in the library, showing a friend how to think through hard problems.
Tone: Strategic, like sharing exam hacks. Not academic or preachy.
{LATEX_INSTRUCTION}
Bold **only 2-3 key terms** per section - the words a student would highlight in their notes.

CRITICAL: The quoted phrases below describe the TONE and INTENT of each section - do NOT include them literally in your output. Write your own natural prose that captures that spirit.

Structure your response with these exact markdown headers:

## Overview
"These questions want you to think about..." - frame the challenge.

## Problem Types
"Prof usually frames it like..." - pattern recognition from past exams.

## Approach
"The trick is to..." - insider strategy for breaking down these problems.

## Worked Example
"Here's a full-marks answer..." - show the gold standard from past exams.

## Scoring Tips
"To get all the points..." - exam hacks for maximizing score.

This is about cracking the exam, not philosophical depth."""
}

# Section types per learning_approach
SECTIONS_BY_APPROACH = {
    "factual": ["overview", "fact", "exam_context", "memory_aid"],
    "conceptual": ["overview", "definition", "exam_patterns", "examples", "common_mistakes"],
    "procedural": ["overview", "when_to_use", "steps", "worked_example", "watch_out"],
    "analytical": ["overview", "problem_types", "approach", "worked_example", "scoring_tips"]
}

# Prompt version for cache invalidation - bump when prompts change
SECTION_PROMPT_VERSION = 1

# Section-by-section prompts for waterfall learn mode
# Each section is generated independently with focused prompts
SECTION_PROMPTS = {
    "procedural": {
        "overview": f"""You are the smartest student in the library, helping a friend before their exam.
Section 1 of 5: OVERVIEW

{LATEX_INSTRUCTION}
Bold **only 2-3 key terms** - what a student would highlight.

Write 50-100 words covering:
- What problem does this procedure solve?
- When would you recognize to use it in an exam?

Keep it conversational: "You know when you see X? That's when you use this."
Do NOT include steps yet - just set up WHY this matters.""",

        "when_to_use": f"""You are the smartest student in the library, helping a friend before their exam.
Section 2 of 5: WHEN TO USE

{LATEX_INSTRUCTION}
Bold **only 2-3 key terms**.

Write 50-100 words covering:
- Pattern recognition: what clues in exam questions signal this procedure?
- What does the setup look like? What keywords appear?

The student already read the Overview. Don't re-introduce - jump to specifics.
"When you see X, Y, or Z in the problem, that's your cue to use this."
""",

        "steps": f"""You are the smartest student in the library, helping a friend before their exam.
Section 3 of 5: STEPS

{LATEX_INSTRUCTION}
Bold **only 2-3 key terms** per step.

Write 200-400 words. For EACH step:
1. **Step N: [Name]** - What to do (clear instruction)
2. WHY it works (the reasoning, not just "because")
3. How to verify you did it right

The student knows WHEN to use this. Now teach HOW.
Be thorough - this is the core learning. Take your time.
Use proper LaTeX for all math expressions.""",

        "worked_example": f"""You are the smartest student in the library, helping a friend before their exam.
Section 4 of 5: WORKED EXAMPLE

{LATEX_INSTRUCTION}
Bold **only 2-3 key terms**.

Write 300-500 words walking through the exam exercise step by step.
Show your work like you're solving it on the board:
- Write out each calculation
- Reference the step numbers as you go ("Applying Step 2...")
- Point out where students often mess up

The student knows the steps. Do NOT re-list them before starting.
But DO reference step numbers as you work: "Now in Step 3, we..."

This should feel like watching someone solve it, not reading a solution manual.""",

        "watch_out": f"""You are the smartest student in the library, helping a friend before their exam.
Section 5 of 5: WATCH OUT

{LATEX_INSTRUCTION}
Bold **only 2-3 key terms**.

Write 150-250 words covering 2-3 biggest mistakes students make:
For each mistake:
- The mistake itself (what goes wrong)
- Why it happens (the trap)
- How to avoid it (the fix)

The student knows the steps. Reference specific step numbers when relevant.
Be specific to THIS procedure, not generic exam advice."""
    },

    "conceptual": {
        "overview": f"""You are the smartest student in the library, explaining a concept to a friend.
Section 1 of 4: OVERVIEW

{LATEX_INSTRUCTION}
Bold **only 1-2 key terms**.

Write 30-50 words - ONE sentence summary.
"This is about..." - set up what they're about to learn.
Keep it ultra-brief. The definition comes next.""",

        "definition": f"""You are the smartest student in the library, explaining a concept to a friend.
Section 2 of 4: DEFINITION

{LATEX_INSTRUCTION}
Bold **only 2-3 key terms**.

Write 150-250 words covering:
- The formal definition (precise, like a textbook)
- The intuition (plain language, like margin notes)
- An analogy if it helps

The student read the overview. Now give them the real content.
Use proper LaTeX for mathematical definitions.""",

        "exam_patterns": f"""You are the smartest student in the library, explaining a concept to a friend.
Section 3 of 4: EXAM PATTERNS

{LATEX_INSTRUCTION}
Bold **only 2-3 key terms**.

Write 150-250 words covering:
- How professors test this concept
- Common question formats you'll see
- What they're really asking for

The student knows the definition. Don't redefine it.
"Prof loves asking..." - share the insider knowledge.""",

        "common_mistakes": f"""You are the smartest student in the library, explaining a concept to a friend.
Section 4 of 4: COMMON MISTAKES

{LATEX_INSTRUCTION}
Bold **only 2-3 key terms**.

Write 150-250 words covering:
- 2-3 mistakes students make with this concept
- Why each mistake happens
- How to avoid it

The student knows the definition and exam patterns.
"Don't fall for this..." - friendly warning about point-losing errors."""
    },

    "factual": {
        "fact": f"""You are the smartest student in the library, sharing notes before an exam.
Section 1 of 3: THE FACT

{LATEX_INSTRUCTION}
Bold **the key fact itself**.

Write 20-50 words. State it clearly and memorably.
Like a highlighted note in your notebook.
Just the fact - context comes next.""",

        "context": f"""You are the smartest student in the library, sharing notes before an exam.
Section 2 of 3: CONTEXT

{LATEX_INSTRUCTION}
Bold **only 1-2 key terms**.

Write 50-100 words covering:
- When/why this fact matters
- Where it appears in exams
- What it connects to

The student knows the fact. Now tell them why it's important.
"This always shows up when..." - the insider tip.""",

        "memory_aid": f"""You are the smartest student in the library, sharing notes before an exam.
Section 3 of 3: MEMORY AID

{LATEX_INSTRUCTION}

Write 50-100 words with a mnemonic or memory trick.
- An acronym, rhyme, or visual association
- How YOU remember it

The student knows the fact and context.
"The way I remember it..." - share your trick."""
    },

    "analytical": {
        "overview": f"""You are the smartest student in the library, showing a friend how to crack hard problems.
Section 1 of 4: OVERVIEW

{LATEX_INSTRUCTION}
Bold **only 1-2 key terms**.

Write 50-100 words covering:
- What type of problem is this?
- What makes it challenging?

"These questions want you to think about..." - frame the challenge.
Don't solve anything yet - just set up what they'll face.""",

        "approach": f"""You are the smartest student in the library, showing a friend how to crack hard problems.
Section 2 of 4: APPROACH

{LATEX_INSTRUCTION}
Bold **only 2-3 key terms**.

Write 150-250 words covering:
- How to think about this type of problem
- What framework or strategy to use
- Key questions to ask yourself

The student knows the problem type. Now teach the thinking.
"The trick is to..." - share the strategic insight.""",

        "worked_example": f"""You are the smartest student in the library, showing a friend how to crack hard problems.
Section 3 of 4: WORKED EXAMPLE

{LATEX_INSTRUCTION}
Bold **only 2-3 key terms**.

Write 300-500 words walking through the exam exercise.
Show the full solution with your reasoning visible:
- Apply the approach from the previous section
- Reference the strategy as you go ("Using the framework...")
- Show how to structure a full-marks answer

The student knows the approach. Do NOT re-explain it.
But DO reference it: "Applying our strategy of..."

This is the gold standard - show what excellence looks like.""",

        "scoring_tips": f"""You are the smartest student in the library, showing a friend how to crack hard problems.
Section 4 of 4: SCORING TIPS

{LATEX_INSTRUCTION}
Bold **only 2-3 key terms**.

Write 100-200 words covering:
- How to maximize your score
- What graders look for
- Partial credit strategies

The student has seen the worked example.
"To get all the points..." - exam hacks for the win."""
    }
}

# Map which sections need context from previous sections
SECTION_CONTEXT_DEPENDENCIES = {
    "procedural": {
        "worked_example": "steps",  # worked example needs steps content
        "watch_out": "steps",       # watch out references steps
    },
    "analytical": {
        "worked_example": "approach",  # worked example needs approach content
    },
    # conceptual and factual don't need context passing
}


def parse_markdown_sections(markdown: str, learning_approach: str) -> List[Dict[str, Any]]:
    """Parse LLM markdown output into sections array.

    Args:
        markdown: Raw markdown from LLM with ## headers
        learning_approach: The learning approach used (for section type mapping)

    Returns:
        List of {type, content} dicts
    """
    # Split by ## headers
    parts = re.split(r'^## ', markdown, flags=re.MULTILINE)

    if len(parts) <= 1:
        # No headers found - return as single content section
        return [{"type": "content", "content": markdown.strip()}]

    sections = []
    # First part is any content before first header (usually empty)
    if parts[0].strip():
        sections.append({"type": "preamble", "content": parts[0].strip()})

    # Parse each section
    for part in parts[1:]:
        lines = part.split('\n', 1)
        header = lines[0].strip()
        content = lines[1].strip() if len(lines) > 1 else ""

        # Convert header to section type (e.g., "Worked Example" -> "worked_example")
        section_type = header.lower().replace(' ', '_')

        sections.append({
            "type": section_type,
            "content": content
        })

    return sections


@dataclass
class TutorResponse:
    """Response from tutor."""
    content: str
    success: bool
    metadata: Optional[Dict[str, Any]] = None


class Tutor:
    """AI tutor for learning core loops and practicing exercises."""

    def __init__(self, llm_manager: Optional[LLMManager] = None, language: str = "en"):
        """Initialize tutor.

        Args:
            llm_manager: LLM manager instance
            language: Output language (any ISO 639-1 code, e.g., "en", "de", "zh")
        """
        self.llm = llm_manager or LLMManager(provider=Config.LLM_PROVIDER)
        self.language = language
        self.concept_explainer = ConceptExplainer(llm_manager=self.llm, language=language)
        self.strategy_manager = StudyStrategyManager(language=language)
        self.proof_tutor = ProofTutor(llm_manager=self.llm, language=language)
        self.metacognitive = MetacognitiveStrategies()

    def _language_instruction(self, action: str = "Respond") -> str:
        """Generate dynamic language instruction for any language.

        Args:
            action: The action verb (e.g., "Respond", "Create", "Explain")

        Returns:
            Language instruction string that works for any ISO 639-1 code
        """
        # LLM understands any ISO 639-1 language code
        return f"{action} in {self.language.upper()} language."

    def learn(self, course_code: str,
              knowledge_item_id: str,
              explain_concepts: bool = True,
              depth: str = "medium",
              adaptive: bool = True,
              include_study_strategy: bool = False,
              show_solutions: bool = True,
              include_metacognitive: bool = True,
              learning_type: str = "conceptual",  # procedural, conceptual, factual, analytical
              show_theory: Optional[bool] = None,
              show_worked_examples: Optional[bool] = None,
              max_theory_sections: Optional[int] = None,
              max_worked_examples: Optional[int] = None,
              knowledge_item_data: Optional[Dict[str, Any]] = None,
              exercises_data: Optional[List[Dict[str, Any]]] = None) -> TutorResponse:
        """Explain a core loop with theory → worked examples → practice flow.

        Args:
            course_code: Course code
            knowledge_item_id: Core loop ID to learn
            explain_concepts: Whether to include prerequisite concepts (default: True)
            depth: Explanation depth - basic, medium, advanced (default: medium)
            adaptive: Enable adaptive teaching (auto-select depth and prerequisites based on mastery)
            include_study_strategy: Whether to include metacognitive study strategy (default: False)
            show_solutions: Whether to show official solutions for exercises (default: True)
            include_metacognitive: Whether to include metacognitive study tips (default: True)
            learning_type: Type of learning content (procedural, conceptual, factual, analytical)
            show_theory: Whether to show theory materials (default: from Config.SHOW_THEORY_BY_DEFAULT)
            show_worked_examples: Whether to show worked examples (default: from Config.SHOW_WORKED_EXAMPLES_BY_DEFAULT)
            max_theory_sections: Max theory sections to show (default: from Config.MAX_THEORY_SECTIONS_IN_LEARN)
            max_worked_examples: Max worked examples to show (default: from Config.MAX_WORKED_EXAMPLES_IN_LEARN)
            knowledge_item_data: Optional dict with knowledge_item data (name, procedure, topic_name, topic_id).
                           When provided, skips SQLite query - useful for cloud integration with PostgreSQL.
            exercises_data: Optional list of example exercises. When provided with knowledge_item_data,
                           skips SQLite exercises query.

        Returns:
            TutorResponse with explanation
        """
        # Apply Config defaults for Phase 10 parameters if not explicitly set
        if show_theory is None:
            show_theory = Config.SHOW_THEORY_BY_DEFAULT
        if show_worked_examples is None:
            show_worked_examples = Config.SHOW_WORKED_EXAMPLES_BY_DEFAULT
        if max_theory_sections is None:
            max_theory_sections = Config.MAX_THEORY_SECTIONS_IN_LEARN
        if max_worked_examples is None:
            max_worked_examples = Config.MAX_WORKED_EXAMPLES_IN_LEARN

        # Use provided data or query SQLite
        if knowledge_item_data is not None:
            # Cloud integration: data provided directly (PostgreSQL)
            knowledge_item_dict = knowledge_item_data
            topic_id = knowledge_item_data.get('topic_id')
            examples = exercises_data[:3] if exercises_data else []
            # Cloud doesn't have theory materials in SQLite, skip them
            theory_materials = []
            worked_examples = []
            exercises_with_solutions = []
            if show_solutions and exercises_data:
                for ex in examples:
                    if ex.get('solution') and ex.get('solution').strip():
                        exercises_with_solutions.append({
                            'exercise_number': ex.get('exercise_number', 'Unknown'),
                            'solution': ex.get('solution'),
                            'source_pdf': ex.get('source_pdf', '')
                        })
        else:
            # Local SQLite mode
            with Database() as db:
                # Get core loop details
                knowledge_item = db.conn.execute("""
                    SELECT cl.*, t.name as topic_name, t.id as topic_id
                    FROM knowledge_items cl
                    JOIN topics t ON cl.topic_id = t.id
                    WHERE cl.id = ? AND t.course_code = ?
                """, (knowledge_item_id, course_code)).fetchone()

                if not knowledge_item:
                    return TutorResponse(
                        content="Core loop not found.",
                        success=False
                    )

                topic_id = knowledge_item['topic_id']

                # Get learning materials for this topic (theory and worked examples)
                # Always fetch materials (default flow), but respect show flags and limits
                theory_materials = []
                worked_examples = []

                if show_theory:
                    theory_materials = db.get_learning_materials_by_topic(
                        topic_id=topic_id,
                        material_type='theory',
                        limit=max_theory_sections
                    )

                if show_worked_examples:
                    worked_examples = db.get_learning_materials_by_topic(
                        topic_id=topic_id,
                        material_type='worked_example',
                        limit=max_worked_examples
                    )

                # Get example exercises (with solutions if available)
                exercises = db.get_exercises_by_course(course_code)
                examples = [ex for ex in exercises if ex.get('knowledge_item_id') == knowledge_item_id][:3]

                # Track exercises with solutions for later display
                exercises_with_solutions = []
                if show_solutions:
                    for ex in examples:
                        if ex.get('solution') and ex.get('solution').strip():
                            exercises_with_solutions.append({
                                'exercise_number': ex.get('exercise_number', 'Unknown'),
                                'solution': ex.get('solution'),
                                'source_pdf': ex.get('source_pdf', '')
                            })

            knowledge_item_dict = dict(knowledge_item)

        # Check if this is a proof exercise (check first example)
        if examples and self.proof_tutor.is_proof_exercise(examples[0].get('text', '')):
            # Use proof-specific learning
            return self._learn_proof(course_code, knowledge_item_id, examples[0], explain_concepts, depth, adaptive)
        knowledge_item_name = knowledge_item_dict.get('name', '')

        # Adaptive teaching: Auto-select depth and prerequisites based on mastery
        adaptive_recommendations = None
        if adaptive:
            from core.adaptive_teaching import AdaptiveTeachingManager

            with AdaptiveTeachingManager() as atm:
                adaptive_recommendations = atm.get_adaptive_recommendations(
                    course_code, knowledge_item_name
                )

                # Override depth and explain_concepts if adaptive
                depth = adaptive_recommendations['depth']
                explain_concepts = adaptive_recommendations['show_prerequisites']

        # Get prerequisite concepts
        prerequisite_text = ""
        if explain_concepts:
            prerequisite_text = self.concept_explainer.explain_prerequisites(
                knowledge_item_name, depth=depth
            )

        # Build enhanced learning prompt with deep reasoning
        prompt = self._build_enhanced_learn_prompt(
            knowledge_item=knowledge_item_dict,
            examples=examples,
            depth=depth,
            learning_type=learning_type,
        )

        # Call LLM for deep explanation
        response = self.llm.generate(
            prompt=prompt,
            model=self.llm.primary_model,
            temperature=0.3,
            max_tokens=3500  # Increased for deeper explanations
        )

        if not response.success:
            return TutorResponse(
                content=f"Failed to generate explanation: {response.error}",
                success=False
            )

        # Combine prerequisite concepts with LLM explanation
        full_content = []

        # 1. Show theory materials first (if any)
        if theory_materials:
            full_content.append(self._display_theory_materials(theory_materials, self.language))
            full_content.append("\n" + "=" * 60 + "\n")

        # 2. Show worked examples (if any)
        if worked_examples:
            full_content.append(self._display_worked_examples(worked_examples, self.language))
            full_content.append("\n" + "=" * 60 + "\n")

        # 3. Show prerequisite concepts
        if prerequisite_text:
            full_content.append(prerequisite_text)
            full_content.append("\n" + "=" * 60 + "\n")

        # 4. Show LLM-generated explanation
        full_content.append(response.text)

        # Add study strategy if requested
        if include_study_strategy:
            strategy = self.strategy_manager.get_strategy_for_knowledge_item(
                knowledge_item_name,
                difficulty=depth
            )
            if strategy:
                full_content.append("\n" + "=" * 60 + "\n")
                full_content.append(self.strategy_manager.format_strategy_output(strategy, knowledge_item_name))

        # Add metacognitive learning strategies if requested
        if include_metacognitive:
            full_content.append("\n" + "=" * 60 + "\n")
            full_content.append(self._format_metacognitive_tips(knowledge_item_name, depth, knowledge_item_dict))

        # Add adaptive recommendations at end
        if adaptive and adaptive_recommendations:
            full_content.append("\n" + "=" * 60 + "\n")
            full_content.append(self._format_adaptive_recommendations(adaptive_recommendations))

        # Add official solutions section if available
        if exercises_with_solutions:
            full_content.append("\n" + "=" * 60 + "\n")
            full_content.append(self._format_official_solutions(exercises_with_solutions))

        return TutorResponse(
            content="\n".join(full_content),
            success=True,
            metadata={
                "knowledge_item": knowledge_item_id,
                "examples_count": len(examples),
                "includes_prerequisites": explain_concepts,
                "depth": depth,
                "adaptive": adaptive,
                "recommendations": adaptive_recommendations,
                "includes_study_strategy": include_study_strategy,
                "includes_metacognitive": include_metacognitive,
                "has_solutions": len(exercises_with_solutions) > 0,
                "solutions_count": len(exercises_with_solutions),
                "theory_materials_count": len(theory_materials),
                "worked_examples_count": len(worked_examples),
                "has_theory": len(theory_materials) > 0,
                "has_worked_examples": len(worked_examples) > 0
            }
        )

    def learn_knowledge_item(
        self,
        knowledge_item: Dict[str, Any],
        exercises: List[Dict[str, Any]],
        notes: Optional[List[str]] = None,
        parent_exercise_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Teach a KnowledgeItem using learning_approach-specific prompts.

        Cloud-first method: data passed directly from PostgreSQL, no SQLite.

        Args:
            knowledge_item: KnowledgeItem dict with id, name, knowledge_type, learning_approach, content
            exercises: List of linked exercise dicts for examples
            notes: Optional list of user's note content strings (PRO users)
            parent_exercise_context: Optional parent exercise text for sub-questions

        Returns:
            Dict with sections array and metadata
        """
        import json

        # Get learning_approach (default to conceptual)
        learning_approach = knowledge_item.get('learning_approach', 'conceptual')
        if learning_approach not in TEACHING_PROMPTS:
            learning_approach = 'conceptual'

        # Get the teaching strategy prompt
        strategy_prompt = TEACHING_PROMPTS[learning_approach]

        # Select best exercise for worked example (prefer exams)
        example_exercise = self._select_example_exercise(exercises) if exercises else None

        # Build the LLM prompt
        prompt = self._build_knowledge_item_prompt(
            knowledge_item=knowledge_item,
            strategy_prompt=strategy_prompt,
            example_exercise=example_exercise,
            notes=notes,
            parent_exercise_context=parent_exercise_context,
        )

        # Call LLM
        response = self.llm.generate(
            prompt=prompt,
            model=self.llm.primary_model,
            temperature=0.3,
            max_tokens=2000
        )

        if not response.success:
            # Return fallback response
            return {
                "sections": [{
                    "type": "fallback",
                    "content": f"Could not generate explanation: {response.error}"
                }],
                "raw_content": "",
                "learning_approach": learning_approach,
                "error": True
            }

        # Parse markdown into sections
        sections = parse_markdown_sections(response.text, learning_approach)

        return {
            "sections": sections,
            "raw_content": response.text,
            "learning_approach": learning_approach,
            "using_notes": bool(notes),
            "has_parent_context": bool(parent_exercise_context),
            "error": False
        }

    def learn_section(
        self,
        knowledge_item: Dict[str, Any],
        section_name: str,
        section_index: int,
        exercises: List[Dict[str, Any]],
        previous_section_content: Optional[str] = None,
        notes: Optional[List[str]] = None,
        parent_exercise_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a single section for waterfall learn mode.

        Each section is generated independently with a focused prompt.

        Args:
            knowledge_item: KnowledgeItem dict with id, name, knowledge_type, learning_approach, content
            section_name: Name of section to generate (e.g., "overview", "steps", "worked_example")
            section_index: Index of this section (0-based)
            exercises: List of linked exercise dicts for examples
            previous_section_content: Optional content from a previous section (for context dependencies)
            notes: Optional list of user's note content strings (PRO users)
            parent_exercise_context: Optional parent exercise text for sub-questions

        Returns:
            Dict with section content and metadata
        """
        import json

        # Get learning_approach (default to conceptual)
        learning_approach = knowledge_item.get('learning_approach', 'conceptual').lower()
        if learning_approach not in SECTION_PROMPTS:
            learning_approach = 'conceptual'

        # Get section prompts for this approach
        approach_prompts = SECTION_PROMPTS.get(learning_approach, {})

        # Get the specific section prompt
        section_prompt = approach_prompts.get(section_name)
        if not section_prompt:
            return {
                "content": f"Unknown section: {section_name}",
                "section_name": section_name,
                "section_index": section_index,
                "error": True
            }

        # Get total sections for this approach
        sections_list = list(approach_prompts.keys())
        total_sections = len(sections_list)

        # Select example exercise for worked example section
        example_exercise = None
        if "example" in section_name.lower() and exercises:
            example_exercise = self._select_example_exercise(exercises)

        # Build the prompt
        prompt = self._build_section_prompt(
            knowledge_item=knowledge_item,
            section_prompt=section_prompt,
            section_name=section_name,
            example_exercise=example_exercise,
            previous_section_content=previous_section_content,
            notes=notes,
            parent_exercise_context=parent_exercise_context,
        )

        # Call LLM
        response = self.llm.generate(
            prompt=prompt,
            model=self.llm.primary_model,
            temperature=0.3,
            max_tokens=1500  # Sufficient for individual sections
        )

        if not response.success:
            return {
                "content": f"Could not generate section: {response.error}",
                "section_name": section_name,
                "section_index": section_index,
                "total_sections": total_sections,
                "learning_approach": learning_approach,
                "error": True
            }

        return {
            "content": response.text,
            "section_name": section_name,
            "section_index": section_index,
            "total_sections": total_sections,
            "is_last": section_index == total_sections - 1,
            "learning_approach": learning_approach,
            "error": False
        }

    def _build_section_prompt(
        self,
        knowledge_item: Dict[str, Any],
        section_prompt: str,
        section_name: str,
        example_exercise: Optional[Dict[str, Any]],
        previous_section_content: Optional[str],
        notes: Optional[List[str]],
        parent_exercise_context: Optional[str],
    ) -> str:
        """Build LLM prompt for a single section."""
        import json

        # Build language instruction
        if self.language and self.language.lower() != "en":
            lang_name = get_language_name(self.language)
            language_instruction = f"IMPORTANT: You MUST respond entirely in {lang_name}. Do not respond in English.\n\n"
        else:
            language_instruction = "Respond in English.\n\n"

        # Start with language instruction and section prompt
        prompt_parts = [
            language_instruction + section_prompt,
            "",
            f"Knowledge Item: {knowledge_item.get('name', 'Unknown')}",
            f"Type: {knowledge_item.get('knowledge_type', 'unknown')}",
        ]

        # Add content if available
        content = knowledge_item.get('content')
        if content:
            if isinstance(content, dict):
                content_str = json.dumps(content, indent=2)
            else:
                content_str = str(content)
            prompt_parts.append(f"Content: {content_str}")

        # Add previous section content if this section depends on it
        if previous_section_content:
            prompt_parts.append("")
            prompt_parts.append("CONTEXT FROM PREVIOUS SECTION:")
            prompt_parts.append("The student has already read this content:")
            prompt_parts.append("---")
            prompt_parts.append(previous_section_content)
            prompt_parts.append("---")
            prompt_parts.append("Reference this when relevant (e.g., step numbers, key concepts).")

        # Add example exercise for worked example sections
        if example_exercise:
            prompt_parts.append("")
            prompt_parts.append("EXAM EXERCISE TO SOLVE:")
            prompt_parts.append(f"Source: {example_exercise.get('source_pdf', 'Unknown')}")
            prompt_parts.append(example_exercise.get('text', example_exercise.get('content', '')))

            # Add solution if available (for reference)
            solution = example_exercise.get('solution')
            if solution:
                prompt_parts.append("")
                prompt_parts.append("Official solution (use as reference):")
                prompt_parts.append(solution)

        # Add parent exercise context for sub-questions
        if parent_exercise_context:
            prompt_parts.append("")
            prompt_parts.append("This is a sub-question. Full exercise context:")
            prompt_parts.append(parent_exercise_context)

        # Add user's notes (PRO feature)
        if notes:
            prompt_parts.append("")
            prompt_parts.append("Student's notes on this topic:")
            for note in notes[:3]:
                note_text = note[:1500] if len(note) > 1500 else note
                prompt_parts.append(note_text)
            prompt_parts.append("")
            prompt_parts.append("Incorporate relevant parts if they help.")

        return "\n".join(prompt_parts)

    def get_sections_for_approach(self, learning_approach: str) -> List[str]:
        """Get list of section names for a learning approach.

        Args:
            learning_approach: The learning approach (procedural, conceptual, factual, analytical)

        Returns:
            List of section names in order
        """
        approach = learning_approach.lower()
        if approach not in SECTION_PROMPTS:
            approach = 'conceptual'
        return list(SECTION_PROMPTS[approach].keys())

    def get_section_context_dependency(self, learning_approach: str, section_name: str) -> Optional[str]:
        """Check if a section needs content from a previous section.

        Args:
            learning_approach: The learning approach
            section_name: The section to check

        Returns:
            Name of section to get context from, or None
        """
        approach = learning_approach.lower()
        dependencies = SECTION_CONTEXT_DEPENDENCIES.get(approach, {})
        return dependencies.get(section_name)

    def _select_example_exercise(self, exercises: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Select best exercise for worked example.

        Prioritizes: exam > exercise_sheet > homework
        """
        if not exercises:
            return None

        priority = {"exam": 1, "exercise_sheet": 2, "homework": 3}
        sorted_ex = sorted(
            exercises,
            key=lambda e: priority.get(e.get('source_type', ''), 99)
        )

        # Get top tier (all with same best source_type)
        best_type = sorted_ex[0].get('source_type')
        top_tier = [e for e in sorted_ex if e.get('source_type') == best_type]

        # Random pick within top tier for variety
        return random.choice(top_tier)

    def _build_knowledge_item_prompt(
        self,
        knowledge_item: Dict[str, Any],
        strategy_prompt: str,
        example_exercise: Optional[Dict[str, Any]],
        notes: Optional[List[str]],
        parent_exercise_context: Optional[str],
    ) -> str:
        """Build LLM prompt for teaching a KnowledgeItem."""
        import json

        # Build language instruction - always include to ensure correct response language
        if self.language and self.language.lower() != "en":
            lang_name = get_language_name(self.language)
            language_instruction = f"IMPORTANT: You MUST respond entirely in {lang_name}. Do not respond in English.\n\n"
        else:
            language_instruction = "Respond in English.\n\n"

        # Start with language instruction and strategy prompt
        prompt_parts = [
            language_instruction + strategy_prompt,
            "",
            f"Knowledge Item: {knowledge_item.get('name', 'Unknown')}",
            f"Type: {knowledge_item.get('knowledge_type', 'unknown')}",
        ]

        # Add content if available
        content = knowledge_item.get('content')
        if content:
            if isinstance(content, dict):
                content_str = json.dumps(content, indent=2)
            else:
                content_str = str(content)
            prompt_parts.append(f"Content: {content_str}")

        # Add example exercise for worked example
        if example_exercise:
            prompt_parts.append("")
            prompt_parts.append("Example exercise from past exams:")
            prompt_parts.append(f"Source: {example_exercise.get('source_pdf', 'Unknown')}")
            prompt_parts.append(example_exercise.get('text', example_exercise.get('content', '')))

            # Add solution if available
            solution = example_exercise.get('solution')
            if solution:
                prompt_parts.append("")
                prompt_parts.append("Official solution:")
                prompt_parts.append(solution)

        # Add parent exercise context for sub-questions
        if parent_exercise_context:
            prompt_parts.append("")
            prompt_parts.append("This is a sub-question. Full exercise context:")
            prompt_parts.append(parent_exercise_context)

        # Add user's notes (PRO feature)
        if notes:
            prompt_parts.append("")
            prompt_parts.append("The student has uploaded their own notes on this topic:")
            for note in notes[:3]:  # Limit to 3 notes
                # Truncate long notes
                note_text = note[:2000] if len(note) > 2000 else note
                prompt_parts.append(note_text)
            prompt_parts.append("")
            prompt_parts.append("Incorporate relevant parts of their notes in your explanation.")

        return "\n".join(prompt_parts)

    def practice(self, course_code: str, topic: Optional[str] = None,
                 difficulty: Optional[str] = None) -> TutorResponse:
        """Get a practice exercise.

        Args:
            course_code: Course code
            topic: Optional topic filter
            difficulty: Optional difficulty filter (easy|medium|hard)

        Returns:
            TutorResponse with exercise
        """
        with Database() as db:
            # Get exercises
            exercises = db.get_exercises_by_course(course_code)

            # Filter by topic if specified
            if topic:
                topic_id = db.conn.execute(
                    "SELECT id FROM topics WHERE course_code = ? AND name LIKE ?",
                    (course_code, f"%{topic}%")
                ).fetchone()

                if topic_id:
                    exercises = [ex for ex in exercises if ex.get('topic_id') == topic_id[0]]

            # Filter by difficulty if specified
            if difficulty:
                exercises = [ex for ex in exercises if ex.get('difficulty') == difficulty]

            # Filter out exercises without core loops
            exercises = [ex for ex in exercises if ex.get('knowledge_item_id')]

            if not exercises:
                return TutorResponse(
                    content="No exercises found matching criteria.",
                    success=False
                )

            # Pick random exercise
            exercise = random.choice(exercises)
            exercise_id = exercise['id']

            # Check if there are linked worked examples
            linked_materials = db.get_materials_for_exercise(exercise_id)
            worked_example_hints = [
                m for m in linked_materials
                if m.get('material_type') == 'worked_example'
            ]

        # Format exercise content with hints
        content = exercise['text']
        if worked_example_hints:
            content += self._format_worked_example_hints(worked_example_hints, self.language)

        return TutorResponse(
            content=content,
            success=True,
            metadata={
                "exercise_id": exercise['id'],
                "knowledge_item_id": exercise.get('knowledge_item_id'),
                "difficulty": exercise.get('difficulty'),
                "topic_id": exercise.get('topic_id'),
                "has_worked_example_hints": len(worked_example_hints) > 0,
                "worked_example_count": len(worked_example_hints)
            }
        )

    def check_answer(self, exercise_id: str, user_answer: str,
                    provide_hints: bool = False) -> TutorResponse:
        """Check user's answer and provide feedback.

        Args:
            exercise_id: Exercise ID
            user_answer: User's answer
            provide_hints: Whether to provide hints if wrong

        Returns:
            TutorResponse with feedback
        """
        with Database() as db:
            # Get exercise and core loop
            exercise = db.conn.execute("""
                SELECT e.*, cl.procedure, cl.name as knowledge_item_name
                FROM exercises e
                LEFT JOIN knowledge_items cl ON e.knowledge_item_id = cl.id
                WHERE e.id = ?
            """, (exercise_id,)).fetchone()

            if not exercise:
                return TutorResponse(
                    content="Exercise not found.",
                    success=False
                )

        # Build evaluation prompt
        prompt = self._build_evaluation_prompt(
            exercise_text=exercise['text'],
            user_answer=user_answer,
            procedure=exercise['procedure'],
            provide_hints=provide_hints
        )

        # Call LLM
        response = self.llm.generate(
            prompt=prompt,
            model=self.llm.primary_model,
            temperature=0.3,
            max_tokens=1500
        )

        if not response.success:
            return TutorResponse(
                content=f"Failed to evaluate answer: {response.error}",
                success=False
            )

        return TutorResponse(
            content=response.text,
            success=True,
            metadata={
                "exercise_id": exercise_id,
                "has_hints": provide_hints
            }
        )

    def generate(self, course_code: str, knowledge_item_id: str,
                difficulty: str = "medium") -> TutorResponse:
        """Generate a new exercise variation.

        Args:
            course_code: Course code
            knowledge_item_id: Core loop to generate for
            difficulty: Difficulty level (easy|medium|hard)

        Returns:
            TutorResponse with new exercise
        """
        with Database() as db:
            # Get core loop
            knowledge_item = db.conn.execute("""
                SELECT cl.*, t.name as topic_name
                FROM knowledge_items cl
                JOIN topics t ON cl.topic_id = t.id
                WHERE cl.id = ? AND t.course_code = ?
            """, (knowledge_item_id, course_code)).fetchone()

            if not knowledge_item:
                return TutorResponse(
                    content="Core loop not found.",
                    success=False
                )

            # Get example exercises
            exercises = db.get_exercises_by_course(course_code)
            examples = [ex for ex in exercises if ex.get('knowledge_item_id') == knowledge_item_id][:5]

            if not examples:
                return TutorResponse(
                    content="No example exercises found for this core loop.",
                    success=False
                )

        # Build generation prompt
        prompt = self._build_generation_prompt(
            knowledge_item=dict(knowledge_item),
            examples=examples,
            difficulty=difficulty
        )

        # Call LLM
        response = self.llm.generate(
            prompt=prompt,
            model=self.llm.primary_model,
            temperature=0.7,  # Higher temperature for creativity
            max_tokens=1500
        )

        if not response.success:
            return TutorResponse(
                content=f"Failed to generate exercise: {response.error}",
                success=False
            )

        return TutorResponse(
            content=response.text,
            success=True,
            metadata={
                "knowledge_item": knowledge_item_id,
                "difficulty": difficulty,
                "based_on_examples": len(examples)
            }
        )

    def _build_learn_prompt(self, knowledge_item: Dict[str, Any],
                           examples: List[Dict[str, Any]]) -> str:
        """Build prompt for learning explanation."""
        prompt = f"""{self._language_instruction("Respond")}

You are an AI tutor helping students learn problem-solving procedures.

TOPIC: {knowledge_item.get('topic_name', 'Unknown')}
CORE PROCEDURE: {knowledge_item['name']}

SOLVING STEPS:
{self._format_procedure(knowledge_item.get('procedure'))}

EXAMPLE EXERCISES:
{self._format_examples(examples)}

Your task:
1. Explain the theory behind this procedure
2. Walk through each step with clear explanations
3. Show how to apply it using the examples
4. Provide tips and common mistakes to avoid

Make it pedagogical and clear for students learning this for the first time.
"""
        return prompt

    def _build_enhanced_learn_prompt(self, knowledge_item: Dict[str, Any],
                                     examples: List[Dict[str, Any]],
                                     depth: str = "medium",
                                     learning_type: str = "conceptual") -> str:
        """Build enhanced prompt with type-specific learning approach.

        Args:
            knowledge_item: Core loop data
            examples: Example exercises
            depth: Explanation depth (basic, medium, advanced)
            learning_type: Type of learning content (procedural, conceptual, factual, analytical)
        """
        depth_instructions = {
            "basic": "Keep explanations simple and concise. Focus on the core concepts.",
            "medium": "Provide balanced explanations with WHY reasoning and practical examples.",
            "advanced": "Give comprehensive explanations with deep reasoning, edge cases, and optimization strategies."
        }

        depth_instruction = depth_instructions.get(depth, depth_instructions["medium"])

        # Build type-specific structure instructions
        if learning_type == "procedural":
            structure = """
Provide a clear, flowing explanation using this structure:

## Big Picture
What problem does this procedure solve? Why does it matter? When would you use it?

## How It Works
For each step, write a flowing explanation that naturally weaves in the reasoning.
Don't use labels like "WHAT/WHY/HOW" - just explain as a good tutor would.

Format each step as:
**Step 1: [Step Name]**
[Flowing explanation with reasoning naturally embedded. Explain what you do, why it works, and how to verify correctness.]

## Worked Example
Walk through a concrete example applying all steps.

## Common Mistakes
What typically goes wrong and why? How to avoid or catch these errors?

## When to Use This
What signals tell you this procedure applies? What are its limitations?"""

        elif learning_type == "factual":
            structure = """
Provide a clear explanation using this structure:

## Key Facts
Present the essential facts clearly and concisely. Use bullet points or numbered lists.

## Context & Background
Explain the historical or contextual background that makes these facts meaningful.

## Memory Aids
Provide mnemonics, acronyms, or memory techniques to help remember key information.

## Connections
How do these facts connect to other topics or concepts? Show relationships.

## Quick Reference
A summary table or list for fast recall during study or exams."""

        elif learning_type == "analytical":
            structure = """
Provide a clear explanation using this structure:

## Central Question or Thesis
What is the main argument, question, or issue being analyzed?

## Key Perspectives
Present different viewpoints or interpretations:
- **Perspective A**: [View and supporting arguments]
- **Perspective B**: [Counter-view and supporting arguments]

## Evidence & Cases
What evidence, examples, or case studies support each perspective?

## Critical Analysis
How do you evaluate these perspectives? What are the strengths and weaknesses?

## Forming Your Own View
Guide the student on how to develop and defend their own position.

## Key Arguments to Remember
Summary of the most important arguments for exam preparation."""

        else:  # conceptual (default)
            structure = """
Provide a clear, flowing explanation using this structure:

## Definition
What is this concept? Give a clear, formal definition.

## Intuition
Explain it in plain language. Use analogies or real-world examples.

## Why It Matters
Why is this concept important? How is it used in practice?

## Key Points
What are the essential things to remember about this concept?

## Related Concepts
How does this connect to other concepts? What are prerequisites and extensions?

## Common Misconceptions
What do students often get wrong about this concept?"""

        prompt = f"""{self._language_instruction("Respond")}

You are an expert educator helping students learn.

TOPIC: {knowledge_item.get('topic_name', 'Unknown')}
CONCEPT/PROCEDURE: {knowledge_item['name']}
LEARNING TYPE: {learning_type}
EXPLANATION DEPTH: {depth}

CONTENT OUTLINE:
{self._format_procedure(knowledge_item.get('procedure'))}

EXAMPLE EXERCISES:
{self._format_examples(examples)}

{depth_instruction}
{structure}

Guidelines:
- Write conversationally, as if tutoring in person
- Use concrete examples, not just abstract descriptions
- Use "you" language to engage the student
- Build from simple to complex
- Keep it flowing and readable
"""
        return prompt

    def _build_evaluation_prompt(self, exercise_text: str, user_answer: str,
                                 procedure: Optional[str], provide_hints: bool) -> str:
        """Build prompt for answer evaluation."""
        hint_instruction = ""
        if provide_hints:
            hint_instruction = "\n3. Provide progressive hints to guide them toward the solution"

        prompt = f"""{self._language_instruction("Respond")}

You are an AI tutor evaluating a student's answer.

EXERCISE:
{exercise_text}

STUDENT'S ANSWER:
{user_answer}

EXPECTED PROCEDURE:
{self._format_procedure(procedure)}

Your task:
1. Evaluate if the answer is correct or partially correct
2. Identify what's right and what's wrong{hint_instruction}
4. Be encouraging and constructive

Respond in a friendly, pedagogical tone.
"""
        return prompt

    def _build_generation_prompt(self, knowledge_item: Dict[str, Any],
                                 examples: List[Dict[str, Any]],
                                 difficulty: str) -> str:
        """Build prompt for exercise generation."""
        prompt = f"""{self._language_instruction("Create an exercise")}

You are creating a new practice exercise.

PROCEDURE TO PRACTICE: {knowledge_item['name']}
TOPIC: {knowledge_item.get('topic_name', 'Unknown')}
DIFFICULTY: {difficulty}

SOLVING STEPS:
{self._format_procedure(knowledge_item.get('procedure'))}

EXAMPLE EXERCISES:
{self._format_examples(examples, limit=3)}

Your task:
Create a NEW exercise that:
1. Tests the same procedure/core loop
2. Has similar structure to the examples
3. Matches the requested difficulty level
4. Uses different numbers/scenarios than the examples
5. Is clear and well-formatted

Generate ONLY the exercise text, not the solution.
"""
        return prompt

    def _display_theory_materials(self, theory_materials: List[Dict[str, Any]],
                                  language: str) -> str:
        """Display theory materials for a topic.

        Args:
            theory_materials: List of theory material dictionaries
            language: Output language ("en" or "it")

        Returns:
            Formatted string with theory materials
        """
        language_headers = {
            "it": {
                "title": "MATERIALI TEORICI",
                "intro": "Prima di iniziare con gli esercizi, esaminiamo la teoria di base:",
                "source": "Fonte",
                "page": "pagina"
            },
            "en": {
                "title": "THEORY MATERIALS",
                "intro": "Before starting with exercises, let's review the foundational theory:",
                "source": "Source",
                "page": "page"
            }
        }

        headers = language_headers.get(language, language_headers["en"])
        lines = [f"\n{headers['title']}\n"]
        lines.append(headers['intro'])
        lines.append("")

        for i, material in enumerate(theory_materials, 1):
            title = material.get('title', f"Theory Section {i}")
            content = material.get('content', '')
            source_pdf = material.get('source_pdf', '')
            page_number = material.get('page_number')

            # Add title
            lines.append(f"## {title}")
            lines.append("")

            # Add source info
            source_info = f"[{headers['source']}: {source_pdf}"
            if page_number:
                source_info += f", {headers['page']} {page_number}"
            source_info += "]"
            lines.append(source_info)
            lines.append("")

            # Add content
            lines.append(content)
            lines.append("")

            # Add separator if not last item
            if i < len(theory_materials):
                lines.append("-" * 40)
                lines.append("")

        return "\n".join(lines)

    def _display_worked_examples(self, worked_examples: List[Dict[str, Any]],
                                 language: str) -> str:
        """Display worked examples for a topic.

        Args:
            worked_examples: List of worked example dictionaries
            language: Output language ("en" or "it")

        Returns:
            Formatted string with worked examples
        """
        language_headers = {
            "it": {
                "title": "ESEMPI RISOLTI",
                "intro": "Ora vediamo come applicare questa teoria attraverso esempi risolti passo dopo passo:",
                "example": "Esempio Risolto",
                "source": "Fonte",
                "page": "pagina",
                "note": "Nota: Questo e un esempio completo che mostra come risolvere questo tipo di problema."
            },
            "en": {
                "title": "WORKED EXAMPLES",
                "intro": "Now let's see how to apply this theory through step-by-step worked examples:",
                "example": "Worked Example",
                "source": "Source",
                "page": "page",
                "note": "Note: This is a complete example showing how to solve this type of problem."
            }
        }

        headers = language_headers.get(language, language_headers["en"])
        lines = [f"\n{headers['title']}\n"]
        lines.append(headers['intro'])
        lines.append("")

        for i, material in enumerate(worked_examples, 1):
            title = material.get('title', f"{headers['example']} {i}")
            content = material.get('content', '')
            source_pdf = material.get('source_pdf', '')
            page_number = material.get('page_number')

            # Add title
            lines.append(f"### {title}")
            lines.append("")

            # Add source info
            source_info = f"[{headers['source']}: {source_pdf}"
            if page_number:
                source_info += f", {headers['page']} {page_number}"
            source_info += "]"
            lines.append(source_info)
            lines.append("")

            # Add content (the worked solution)
            lines.append(content)
            lines.append("")

            # Add helpful note
            lines.append(f"[{headers['note']}]")
            lines.append("")

            # Add separator if not last item
            if i < len(worked_examples):
                lines.append("-" * 40)
                lines.append("")

        return "\n".join(lines)

    def _format_worked_example_hints(self, worked_examples: List[Dict[str, Any]],
                                     language: str) -> str:
        """Format hints about available worked examples for an exercise.

        Args:
            worked_examples: List of worked example material dictionaries
            language: Output language ("en" or "it")

        Returns:
            Formatted string with hints about worked examples
        """
        language_headers = {
            "it": {
                "hint": "Suggerimento: Per un problema simile, consulta",
                "these_worked_examples": "questi esempi risolti",
                "this_worked_example": "questo esempio risolto"
            },
            "en": {
                "hint": "Hint: For a similar problem, see",
                "these_worked_examples": "these worked examples",
                "this_worked_example": "this worked example"
            }
        }

        headers = language_headers.get(language, language_headers["en"])
        lines = ["\n\n---"]

        if len(worked_examples) == 1:
            title = worked_examples[0].get('title', 'Worked Example')
            lines.append(f"\n{headers['hint']} {headers['this_worked_example']}: \"{title}\"")
        else:
            lines.append(f"\n{headers['hint']} {headers['these_worked_examples']}:")
            for material in worked_examples:
                title = material.get('title', 'Worked Example')
                lines.append(f"  - \"{title}\"")

        return "\n".join(lines)

    def _format_procedure(self, procedure: Optional[str]) -> str:
        """Format procedure steps for display."""
        if not procedure:
            return "No procedure available."

        try:
            import json
            steps = json.loads(procedure)
            if isinstance(steps, list):
                return "\n".join([f"{i+1}. {step}" for i, step in enumerate(steps)])
        except:
            pass

        return str(procedure)

    def _format_examples(self, examples: List[Dict[str, Any]], limit: int = 3) -> str:
        """Format example exercises."""
        if not examples:
            return "No examples available."

        formatted = []
        for i, ex in enumerate(examples[:limit], 1):
            text = ex['text'][:300] + "..." if len(ex['text']) > 300 else ex['text']
            formatted.append(f"Example {i}:\n{text}\n")

        return "\n".join(formatted)

    def _format_adaptive_recommendations(self, recommendations: Dict[str, Any]) -> str:
        """Format adaptive teaching recommendations.

        Args:
            recommendations: Recommendations dictionary from AdaptiveTeachingManager

        Returns:
            Formatted string with recommendations
        """
        language_headers = {
            "it": {
                "title": "📚 RACCOMANDAZIONI DI STUDIO PERSONALIZZATE",
                "mastery": "Livello di padronanza attuale",
                "practice": "Esercizi consigliati per la pratica",
                "focus": "Aree su cui concentrarsi",
                "next_review": "Prossima revisione programmata"
            },
            "en": {
                "title": "📚 PERSONALIZED STUDY RECOMMENDATIONS",
                "mastery": "Current mastery level",
                "practice": "Recommended practice exercises",
                "focus": "Focus areas",
                "next_review": "Next scheduled review"
            }
        }

        headers = language_headers.get(self.language, language_headers["en"])
        lines = [f"\n{headers['title']}\n"]

        # Current mastery
        mastery = recommendations.get('current_mastery', 0.0)
        mastery_pct = int(mastery * 100)
        mastery_emoji = "🟢" if mastery >= 0.7 else "🟡" if mastery >= 0.3 else "🔴"
        lines.append(f"{mastery_emoji} {headers['mastery']}: {mastery_pct}%")

        # Practice recommendations
        practice_count = recommendations.get('practice_count', 3)
        lines.append(f"\n✍️  {headers['practice']}: {practice_count}")

        # Focus areas
        focus_areas = recommendations.get('focus_areas', [])
        if focus_areas:
            lines.append(f"\n🎯 {headers['focus']}:")
            for area in focus_areas:
                lines.append(f"   • {area}")

        # Next review
        next_review = recommendations.get('next_review')
        if next_review:
            from datetime import datetime
            try:
                review_date = datetime.fromisoformat(next_review)
                lines.append(f"\n📅 {headers['next_review']}: {review_date.strftime('%Y-%m-%d')}")
            except:
                pass

        return "\n".join(lines)

    def _format_official_solutions(self, exercises_with_solutions: List[Dict[str, Any]]) -> str:
        """Format official solutions section.

        Args:
            exercises_with_solutions: List of exercises with solutions

        Returns:
            Formatted string with solutions
        """
        language_headers = {
            "it": {
                "title": "SOLUZIONI UFFICIALI",
                "available": "Sono disponibili soluzioni ufficiali per alcuni esercizi di esempio:",
                "exercise": "Esercizio",
                "from": "da",
                "note": "Nota: Questa soluzione e stata estratta automaticamente dal PDF dell'esame."
            },
            "en": {
                "title": "OFFICIAL SOLUTIONS",
                "available": "Official solutions are available for some example exercises:",
                "exercise": "Exercise",
                "from": "from",
                "note": "Note: This solution was automatically extracted from the exam PDF."
            }
        }

        headers = language_headers.get(self.language, language_headers["en"])
        lines = [f"\n{headers['title']}\n"]
        lines.append(headers['available'])
        lines.append("")

        for i, ex_sol in enumerate(exercises_with_solutions, 1):
            exercise_num = ex_sol.get('exercise_number', f'#{i}')
            source_pdf = ex_sol.get('source_pdf', '')
            solution = ex_sol.get('solution', '')

            lines.append(f"{headers['exercise']} {exercise_num}")
            if source_pdf:
                lines.append(f"({headers['from']} {source_pdf})")
            lines.append("")
            lines.append(solution)
            lines.append("")
            lines.append(f"[{headers['note']}]")
            lines.append("")

            if i < len(exercises_with_solutions):
                lines.append("-" * 40)
                lines.append("")

        return "\n".join(lines)

    def _format_metacognitive_tips(self, knowledge_item_name: str, depth: str,
                                   knowledge_item_dict: Dict[str, Any]) -> str:
        """Format metacognitive learning strategies section.

        Args:
            knowledge_item_name: Name of the core loop
            depth: Difficulty level (basic, medium, advanced)
            knowledge_item_dict: Core loop dictionary with metadata

        Returns:
            Formatted string with metacognitive strategies
        """
        language_headers = {
            "it": {
                "title": "STRATEGIE DI APPRENDIMENTO",
                "framework": "Strategia consigliata per problemi di questo tipo",
                "tips": "Consigli per studiare efficacemente",
                "self_assessment": "Domande di autovalutazione",
                "retrieval": "Tecniche di recupero per rafforzare la memoria"
            },
            "en": {
                "title": "LEARNING STRATEGIES",
                "framework": "Recommended problem-solving framework",
                "tips": "Study tips for effective learning",
                "self_assessment": "Self-assessment prompts",
                "retrieval": "Retrieval practice techniques"
            }
        }

        headers = language_headers.get(self.language, language_headers["en"])
        lines = [f"\n{headers['title']}\n"]

        # Map depth to difficulty level
        difficulty_map = {
            "basic": DifficultyLevel.EASY,
            "medium": DifficultyLevel.MEDIUM,
            "advanced": DifficultyLevel.HARD
        }
        difficulty = difficulty_map.get(depth, DifficultyLevel.MEDIUM)

        # Assume NEW mastery for now (could be enhanced with actual mastery data)
        mastery = MasteryLevel.NEW

        # 1. Problem-solving framework
        # Determine problem type from core loop name/description
        problem_type = self._infer_problem_type(knowledge_item_name)
        framework = self.metacognitive.get_problem_solving_framework(problem_type)

        lines.append(f"### {headers['framework']}: {framework.name}\n")
        lines.append(framework.description)
        lines.append("")
        for step in framework.steps:
            lines.append(f"  {step}")
        lines.append("")

        # 2. Study tips (show top 3 most relevant)
        tips = self.metacognitive.get_study_tips(knowledge_item_name, difficulty, mastery)
        if tips:
            lines.append(f"### {headers['tips']}:\n")
            for i, tip in enumerate(tips[:3], 1):
                lines.append(f"{i}. **{tip.tip}**")
                lines.append(f"   Why: {tip.why}")
                lines.append(f"   When: {tip.when_to_use}")
                lines.append("")

        # 3. Self-assessment prompts (show top 2)
        assessments = self.metacognitive.get_self_assessment_prompts(knowledge_item_name, mastery)
        if assessments:
            lines.append(f"### {headers['self_assessment']}:\n")
            for i, assessment in enumerate(assessments[:2], 1):
                lines.append(f"{i}. {assessment.prompt}")
            lines.append("")

        # 4. Retrieval practice suggestions
        # Assume 0 hours since last review (immediate)
        retrieval_techniques = self.metacognitive.get_retrieval_practice_suggestions(
            time_since_last_review=0,
            mastery=mastery
        )
        if retrieval_techniques:
            lines.append(f"### {headers['retrieval']}:\n")
            for technique in retrieval_techniques[:2]:
                lines.append(f"**{technique.technique}**: {technique.description}")
                lines.append(f"Example: {technique.example}")
                lines.append("")

        return "\n".join(lines)

    def _infer_problem_type(self, knowledge_item_name: str) -> str:
        """Infer problem type from core loop name.

        Args:
            knowledge_item_name: Name of the core loop

        Returns:
            Problem type string (design, theory, proof, debugging, etc.)
        """
        name_lower = knowledge_item_name.lower()

        if any(keyword in name_lower for keyword in ['prove', 'proof', 'theorem', 'lemma']):
            return "proof"
        elif any(keyword in name_lower for keyword in ['design', 'construct', 'build', 'create']):
            return "design"
        elif any(keyword in name_lower for keyword in ['debug', 'verify', 'check', 'validate']):
            return "debugging"
        elif any(keyword in name_lower for keyword in ['theory', 'concept', 'definition', 'explain']):
            return "theory"
        elif any(keyword in name_lower for keyword in ['implement', 'code', 'program']):
            return "implementation"
        else:
            return "general"

    def _learn_proof(self, course_code: str, knowledge_item_id: str, example_exercise: Dict[str, Any],
                     explain_concepts: bool, depth: str, adaptive: bool) -> TutorResponse:
        """Handle proof-specific learning.

        Args:
            course_code: Course code
            knowledge_item_id: Core loop ID
            example_exercise: Example proof exercise
            explain_concepts: Whether to include prerequisites
            depth: Explanation depth
            adaptive: Whether to use adaptive teaching

        Returns:
            TutorResponse with proof explanation
        """
        exercise_text = example_exercise.get('text', '')
        exercise_id = example_exercise.get('id', '')

        # Get proof-specific explanation
        proof_explanation = self.proof_tutor.learn_proof(course_code, exercise_id, exercise_text)

        # Optionally add prerequisite concepts
        full_content = []

        if explain_concepts:
            # Extract core loop name for concept explanation
            with Database() as db:
                knowledge_item = db.conn.execute(
                    "SELECT name FROM knowledge_items WHERE id = ?",
                    (knowledge_item_id,)
                ).fetchone()
                if knowledge_item:
                    knowledge_item_name = knowledge_item['name']
                    prerequisite_text = self.concept_explainer.explain_prerequisites(
                        knowledge_item_name, depth=depth
                    )
                    if prerequisite_text:
                        full_content.append(prerequisite_text)
                        full_content.append("\n" + "=" * 60 + "\n")

        full_content.append(proof_explanation)

        return TutorResponse(
            content="\n".join(full_content),
            success=True,
            metadata={
                "knowledge_item": knowledge_item_id,
                "is_proof": True,
                "depth": depth,
                "includes_prerequisites": explain_concepts
            }
        )
