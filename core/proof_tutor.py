"""
Proof-specific tutoring module for Examina.
Handles proof learning, practice, technique identification, and verification.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from models.llm_manager import LLMManager
from storage.database import Database


@dataclass
class ProofTechnique:
    """Information about a proof technique."""
    name: str  # direct, contradiction, induction, construction, contrapositive
    description: str
    when_to_use: str
    common_mistakes: List[str]


@dataclass
class ProofAnalysis:
    """Analysis of a proof exercise."""
    is_proof: bool
    proof_type: str  # mathematical, logical, algorithmic
    technique_suggested: str
    premise: str
    goal: str
    key_concepts: List[str]
    difficulty: str


class ProofTutor:
    """Specialized tutor for proof-based exercises."""

    # Proof technique patterns
    PROOF_TECHNIQUES = {
        "direct": ProofTechnique(
            name="direct",
            description="Direct proof: Start from premises and apply logical steps to reach the conclusion",
            when_to_use="Use when there's a clear path from hypothesis to conclusion using definitions and known theorems",
            common_mistakes=[
                "Assuming what you need to prove",
                "Circular reasoning",
                "Missing intermediate steps"
            ]
        ),
        "contradiction": ProofTechnique(
            name="contradiction",
            description="Proof by contradiction: Assume the opposite of what you want to prove and derive a contradiction",
            when_to_use="Use when proving 'impossibility' statements or when direct proof seems difficult",
            common_mistakes=[
                "Not clearly stating the contradiction assumption",
                "Deriving a falsehood that doesn't contradict the assumption",
                "Forgetting to conclude the original statement"
            ]
        ),
        "induction": ProofTechnique(
            name="induction",
            description="Mathematical induction: Prove base case and inductive step (P(n) → P(n+1))",
            when_to_use="Use for statements involving natural numbers or recursive structures",
            common_mistakes=[
                "Not proving the base case",
                "Not clearly stating the inductive hypothesis",
                "Circular reasoning in inductive step"
            ]
        ),
        "construction": ProofTechnique(
            name="construction",
            description="Constructive proof: Show existence by explicitly constructing an example",
            when_to_use="Use for existence proofs where you can build the object",
            common_mistakes=[
                "Not verifying the constructed object satisfies all requirements",
                "Construction not being well-defined",
                "Missing edge cases"
            ]
        ),
        "contrapositive": ProofTechnique(
            name="contrapositive",
            description="Proof by contrapositive: Prove 'not Q implies not P' instead of 'P implies Q'",
            when_to_use="Use when the contrapositive is easier to prove than the direct implication",
            common_mistakes=[
                "Confusing contrapositive with converse",
                "Not proving the full contrapositive statement"
            ]
        )
    }

    def __init__(self, llm_manager: Optional[LLMManager] = None, language: str = "en"):
        """Initialize proof tutor.

        Args:
            llm_manager: LLM manager instance
            language: Output language (any ISO 639-1 code, e.g., "en", "de", "zh")
        """
        self.llm = llm_manager or LLMManager(provider="anthropic")
        self.language = language

    def _language_instruction(self, action: str = "Respond") -> str:
        """Generate dynamic language instruction for any language."""
        return f"{action} in {self.language.upper()} language."

    def _lang_instruction(self) -> str:
        """Generate language instruction phrase for any language."""
        return f"in {self.language.upper()} language"

    def is_proof_exercise(self, exercise_text: str) -> bool:
        """Check if exercise is a proof question.

        Args:
            exercise_text: Exercise text

        Returns:
            True if this is a proof exercise
        """
        # Keywords that indicate proof exercises
        # Use more specific patterns to avoid false positives
        import re

        text_lower = exercise_text.lower()

        # Italian proof keywords - more specific patterns
        italian_patterns = [
            r'\bdimostraz',  # dimostrazione, dimostrazioni
            r'\bdimostra[rt]',  # dimostrare, dimostrare
            r'\bsi dimostri',  # formal proof request
            r'\bprova che\b',  # prove that (not just "prova" = test)
            r'\bprovare che\b',  # to prove that
        ]

        # English proof keywords
        english_patterns = [
            r'\bprove\b',  # prove (not proven, disprove, etc.)
            r'\bproof\b',  # proof
            r'\bshow that\b',  # show that
            r'\bdemonstrate that\b',  # demonstrate that
            r'\bverify that\b',  # verify that (in proof context)
        ]

        # Check all patterns
        all_patterns = italian_patterns + english_patterns
        for pattern in all_patterns:
            if re.search(pattern, text_lower):
                return True

        return False

    def analyze_proof(self, course_code: str, exercise_text: str) -> ProofAnalysis:
        """Analyze a proof exercise to identify technique and structure.

        Args:
            course_code: Course code
            exercise_text: Exercise text

        Returns:
            ProofAnalysis with proof structure
        """
        prompt = f"""{self._language_instruction("Respond")}

You are an expert mathematics and computer science educator analyzing a proof exercise.

EXERCISE:
{exercise_text}

Your task: Analyze this proof exercise and provide:

1. **Proof Type**: mathematical, logical, or algorithmic
2. **Suggested Technique**: Which proof technique would work best?
   - direct: Direct proof from premises to conclusion
   - contradiction: Assume opposite and derive contradiction
   - induction: Base case + inductive step
   - construction: Construct explicit example
   - contrapositive: Prove not Q → not P instead of P → Q

3. **Structure**:
   - What is given (premise/hypothesis)?
   - What needs to be proven (goal/conclusion)?
   - What are the key concepts involved?

4. **Difficulty**: easy, medium, or hard

Respond in JSON format:
{{
    "is_proof": true/false,
    "proof_type": "mathematical|logical|algorithmic",
    "technique_suggested": "direct|contradiction|induction|construction|contrapositive",
    "premise": "what is given",
    "goal": "what to prove",
    "key_concepts": ["concept1", "concept2", ...],
    "difficulty": "easy|medium|hard"
}}
"""

        response = self.llm.generate(
            prompt=prompt,
            model=self.llm.primary_model,
            temperature=0.2,
            max_tokens=1000
        )

        if not response.success:
            # Fallback analysis
            return ProofAnalysis(
                is_proof=self.is_proof_exercise(exercise_text),
                proof_type="mathematical",
                technique_suggested="direct",
                premise="Unknown",
                goal="Unknown",
                key_concepts=[],
                difficulty="medium"
            )

        # Parse JSON response
        try:
            import json
            # Extract JSON from response (may have markdown code blocks)
            text = response.text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            data = json.loads(text)

            return ProofAnalysis(
                is_proof=data.get("is_proof", True),
                proof_type=data.get("proof_type", "mathematical"),
                technique_suggested=data.get("technique_suggested", "direct"),
                premise=data.get("premise", "Unknown"),
                goal=data.get("goal", "Unknown"),
                key_concepts=data.get("key_concepts", []),
                difficulty=data.get("difficulty", "medium")
            )
        except Exception as e:
            print(f"[WARNING] Failed to parse proof analysis: {e}")
            return ProofAnalysis(
                is_proof=True,
                proof_type="mathematical",
                technique_suggested="direct",
                premise="Unknown",
                goal="Unknown",
                key_concepts=[],
                difficulty="medium"
            )

    def learn_proof(self, course_code: str, exercise_id: str, exercise_text: str) -> str:
        """Generate proof-specific learning explanation.

        Args:
            course_code: Course code
            exercise_id: Exercise ID
            exercise_text: Exercise text

        Returns:
            Formatted proof explanation
        """
        # First analyze the proof
        analysis = self.analyze_proof(course_code, exercise_text)

        # Get proof technique info
        technique = self.PROOF_TECHNIQUES.get(analysis.technique_suggested)

        prompt = f"""{self._language_instruction("Respond")}

You are an expert educator teaching a student how to approach and solve a PROOF exercise.

PROOF EXERCISE:
{exercise_text}

ANALYSIS:
- Proof Type: {analysis.proof_type}
- Premise (Given): {analysis.premise}
- Goal (To Prove): {analysis.goal}
- Suggested Technique: {analysis.technique_suggested}

PROOF TECHNIQUE GUIDANCE:
{technique.description if technique else "Direct proof approach"}

When to use: {technique.when_to_use if technique else "N/A"}

Common Mistakes to Avoid:
{chr(10).join('- ' + m for m in technique.common_mistakes) if technique else "N/A"}

Your task: Provide a comprehensive, step-by-step explanation of how to approach and solve this proof.

Structure your response as follows:

## 1. UNDERSTANDING THE PROBLEM
- What are we given (premises/hypotheses)?
- What do we need to prove (conclusion)?
- What key concepts or definitions are involved?
- What is the logical structure (implication, equivalence, existence, etc.)?

## 2. PROOF STRATEGY
- Why is {analysis.technique_suggested} proof the best approach here?
- What's the overall game plan?
- What sub-goals or lemmas might we need?

## 3. STEP-BY-STEP PROOF
Walk through the proof step by step:
- **Step 1**: [Clear description]
  - WHY: [Explain the reasoning]
  - HOW: [Show the formal manipulation]

- **Step 2**: [Next step]
  - WHY: [Reasoning]
  - HOW: [Formalization]

[Continue for all steps...]

## 4. KEY INSIGHTS
- What's the critical insight that makes this proof work?
- How do the pieces connect together?
- What mathematical principles are at play?

## 5. VERIFICATION
- How can we verify our proof is correct?
- What should we check?
- Are there edge cases?

## 6. COMMON MISTAKES
- What mistakes do students typically make on this type of proof?
- How can you avoid them?
- Warning signs that something went wrong?

## 7. PRACTICE TIPS
- How to practice this proof technique?
- Similar problems to try?
- How to build intuition?

Make your explanation clear, pedagogical, and mathematically rigorous while remaining accessible.
"""

        response = self.llm.generate(
            prompt=prompt,
            model=self.llm.primary_model,
            temperature=0.3,
            max_tokens=3500
        )

        if not response.success:
            return f"Error generating proof explanation: {response.error}"

        return response.text

    def practice_proof(self, course_code: str, exercise_text: str,
                      user_attempt: str, provide_hints: bool = True) -> Dict[str, Any]:
        """Evaluate a proof attempt and provide feedback.

        Args:
            course_code: Course code
            exercise_text: Exercise text
            user_attempt: Student's proof attempt
            provide_hints: Whether to provide hints

        Returns:
            Dictionary with feedback and evaluation
        """
        # Analyze the proof
        analysis = self.analyze_proof(course_code, exercise_text)

        hint_instruction = ""
        if provide_hints:
            hint_instruction = """
If the proof has errors, provide progressive hints:
- Hint 1: Point to the area with the problem (without revealing the solution)
- Hint 2: Explain what's wrong conceptually
- Hint 3: Suggest the correct approach
"""

        prompt = f"""{self._language_instruction("Respond")}

You are an expert educator evaluating a student's proof.

EXERCISE:
{exercise_text}

EXPECTED APPROACH:
- Technique: {analysis.technique_suggested}
- Premise: {analysis.premise}
- Goal: {analysis.goal}

STUDENT'S PROOF ATTEMPT:
{user_attempt}

Your task: Evaluate the student's proof thoroughly.

Provide:

## 1. EVALUATION
- Is the proof correct? (Yes/Partially/No)
- Score: 0-100%

## 2. STRENGTHS
What did the student do well?
- Correct logical steps
- Good structure
- Proper use of definitions/theorems

## 3. WEAKNESSES & ERRORS
What needs improvement?
- Logical gaps
- Incorrect reasoning
- Missing steps
- Circular reasoning
- Unjustified claims

## 4. DETAILED FEEDBACK
Go through the proof step by step:
- Step X: ✓ Correct / ✗ Error - [explanation]

## 5. CONCEPTUAL UNDERSTANDING
Does the student understand:
- The underlying concepts?
- Why the technique works?
- The logical structure?

{hint_instruction}

## 6. SUGGESTIONS FOR IMPROVEMENT
How can the student improve?
- What to study
- What to practice
- Resources that might help

Be encouraging but rigorous. Focus on helping the student learn, not just grading.
"""

        response = self.llm.generate(
            prompt=prompt,
            model=self.llm.primary_model,
            temperature=0.3,
            max_tokens=2500
        )

        if not response.success:
            return {
                "is_correct": False,
                "score": 0.0,
                "feedback": f"Error evaluating proof: {response.error}",
                "technique": analysis.technique_suggested
            }

        # Simple heuristic to determine correctness from feedback
        feedback_lower = response.text.lower()
        is_correct = (
            "correct" in feedback_lower and
            "incorrect" not in feedback_lower and
            ("100%" in feedback_lower or "fully correct" in feedback_lower)
        )

        # Try to extract score
        score = 0.5  # Default partial credit
        if is_correct:
            score = 1.0
        elif "no" in feedback_lower or "incorrect" in feedback_lower:
            score = 0.0

        return {
            "is_correct": is_correct,
            "score": score,
            "feedback": response.text,
            "technique": analysis.technique_suggested,
            "analysis": analysis
        }

    def get_technique_explanation(self, technique_name: str) -> str:
        """Get detailed explanation of a proof technique.

        Args:
            technique_name: Name of the technique

        Returns:
            Formatted explanation
        """
        technique = self.PROOF_TECHNIQUES.get(technique_name)
        if not technique:
            return f"Unknown proof technique: {technique_name}"

        output = [
            f"# {technique.name.title()} Proof",
            "",
            f"**Description**: {technique.description}",
            "",
            f"**When to Use**: {technique.when_to_use}",
            "",
            "**Common Mistakes**:",
        ]

        for mistake in technique.common_mistakes:
            output.append(f"  - {mistake}")

        return "\n".join(output)

    def suggest_technique(self, exercise_text: str) -> str:
        """Suggest the best proof technique for an exercise.

        Args:
            exercise_text: Exercise text

        Returns:
            Suggested technique name
        """
        # Quick heuristic checks
        text_lower = exercise_text.lower()

        # Induction keywords
        if any(keyword in text_lower for keyword in ["for all n", "ogni n", "induzione", "induction"]):
            return "induction"

        # Contradiction keywords
        if any(keyword in text_lower for keyword in ["impossibil", "non esiste", "does not exist", "cannot"]):
            return "contradiction"

        # Existence keywords
        if any(keyword in text_lower for keyword in ["esiste", "exists", "trova", "find", "construct"]):
            return "construction"

        # Default to direct
        return "direct"

    def get_proof_guidance(self, exercise_text: str, technique: str) -> Dict[str, Any]:
        """Get step-by-step proof guidance.

        Args:
            exercise_text: Exercise text
            technique: Proof technique to use

        Returns:
            Dictionary with steps and guidance
        """
        prompt = f"""Provide step-by-step guidance for proving this statement using {technique} proof.

Exercise: {exercise_text}

Proof Technique: {technique}

Provide a structured outline of proof steps ({self._lang_instruction()}):
1. What to assume/start with
2. Key logical steps to take
3. What definitions/theorems to use
4. How to reach the conclusion

Format as numbered steps. Be specific but don't give away the full solution.
Each step should guide thinking, not provide complete answers.
"""

        response = self.llm.generate(
            prompt=prompt,
            model=self.llm.fast_model,
            temperature=0.5,
            max_tokens=1000
        )

        if not response.success:
            return {"success": False, "error": response.error}

        # Parse response into steps
        steps = []
        for line in response.text.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                # Remove numbering
                clean_line = line.lstrip('0123456789.-•) ').strip()
                if clean_line:
                    steps.append(clean_line)

        return {
            "success": True,
            "steps": steps,
            "technique": technique
        }

    def get_hint_for_step(self, exercise_text: str, technique: str, step_number: int) -> str:
        """Get a hint for a specific proof step.

        Args:
            exercise_text: Exercise text
            technique: Proof technique being used
            step_number: Which step needs a hint

        Returns:
            Hint text
        """
        prompt = f"""Provide a helpful hint for step {step_number} of this proof.

Exercise: {exercise_text}
Technique: {technique}
Step: {step_number}

Give a hint that helps the student think through this step without giving away the answer ({self._lang_instruction()}).
The hint should:
- Point to relevant definitions or theorems
- Suggest what to consider or look for
- NOT provide the complete step

Keep it brief (2-3 sentences).
"""

        response = self.llm.generate(
            prompt=prompt,
            model=self.llm.fast_model,
            temperature=0.5,
            max_tokens=200
        )

        if response.success:
            return response.text.strip()
        return f"Error generating hint: {response.error}"

    def get_full_proof(self, exercise_text: str, technique: str) -> str:
        """Get complete proof solution.

        Args:
            exercise_text: Exercise text
            technique: Proof technique to use

        Returns:
            Full proof in markdown format
        """
        prompt = f"""Provide a complete, rigorous proof for this statement using {technique} proof.

Exercise: {exercise_text}

Proof Technique: {technique}

Provide a complete formal proof ({self._lang_instruction()}) with:
1. Clear statement of what we're proving
2. Setup (definitions, assumptions, given information)
3. Step-by-step proof with justifications
4. Clear conclusion

Format in markdown with proper mathematical notation.
Be rigorous and educational - explain WHY each step follows.
"""

        response = self.llm.generate(
            prompt=prompt,
            model=self.llm.primary_model,
            temperature=0.3,
            max_tokens=2000
        )

        if response.success:
            return response.text.strip()
        return f"Error generating proof: {response.error}"
