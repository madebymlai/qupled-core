"""
Metacognitive Learning Strategies Module

This module implements research-backed metacognitive strategies to enhance learning:
1. Study tips tailored to topic/difficulty
2. Problem-solving frameworks
3. Self-assessment prompts
4. Retrieval practice suggestions
5. Spaced repetition optimization

Based on cognitive science research on effective learning strategies.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class DifficultyLevel(Enum):
    """Exercise difficulty levels."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class MasteryLevel(Enum):
    """Student mastery levels (from SM-2 algorithm)."""
    NEW = "new"  # Never attempted
    LEARNING = "learning"  # EF < 2.5
    REVIEWING = "reviewing"  # EF >= 2.5, not yet mastered
    MASTERED = "mastered"  # Consistent high performance


@dataclass
class StudyTip:
    """A targeted study tip for effective learning."""
    category: str  # e.g., "preparation", "during_study", "after_study", "metacognitive"
    tip: str
    why: str  # Explanation of why this tip works (cognitive science backing)
    when_to_use: str  # When is this tip most effective?


@dataclass
class ProblemSolvingFramework:
    """A structured framework for solving problems."""
    name: str
    steps: List[str]
    description: str
    best_for: List[str]  # Types of problems this framework works well for
    example: Optional[str] = None


@dataclass
class SelfAssessment:
    """Self-assessment prompts to check understanding."""
    prompt: str
    category: str  # e.g., "understanding", "application", "analysis"
    difficulty: DifficultyLevel


@dataclass
class RetrievalPractice:
    """Retrieval practice suggestion to strengthen memory."""
    technique: str
    description: str
    example: str
    optimal_timing: str  # When to use this technique for best results


class MetacognitiveStrategies:
    """Provides metacognitive learning strategies based on context."""

    def __init__(self):
        """Initialize metacognitive strategies."""
        self._problem_solving_frameworks = self._init_frameworks()
        self._retrieval_techniques = self._init_retrieval_techniques()

    def _init_frameworks(self) -> List[ProblemSolvingFramework]:
        """Initialize problem-solving frameworks."""
        return [
            ProblemSolvingFramework(
                name="Polya's Four-Step Method",
                steps=[
                    "1. Understand the problem - What is being asked? What are the givens?",
                    "2. Devise a plan - What strategy or approach will you use?",
                    "3. Execute the plan - Carry out your strategy step by step",
                    "4. Review/Reflect - Does your answer make sense? Can you verify it?"
                ],
                description="Classic problem-solving framework applicable to any domain",
                best_for=["math problems", "design problems", "algorithmic problems"],
                example="When designing a Mealy machine: (1) Understand input/output specs, (2) Plan state transitions, (3) Draw diagram, (4) Verify with test cases"
            ),
            ProblemSolvingFramework(
                name="IDEAL Framework",
                steps=[
                    "I - Identify the problem",
                    "D - Define the constraints and goals",
                    "E - Explore possible strategies",
                    "A - Act on the best strategy",
                    "L - Look back and evaluate"
                ],
                description="Metacognitive framework emphasizing strategy exploration",
                best_for=["complex design problems", "optimization problems", "multi-step problems"]
            ),
            ProblemSolvingFramework(
                name="Feynman Technique",
                steps=[
                    "1. Explain the concept in simple terms (as if teaching a child)",
                    "2. Identify gaps in your explanation",
                    "3. Go back to source material to fill gaps",
                    "4. Simplify and use analogies"
                ],
                description="Effective for deep conceptual understanding",
                best_for=["theory questions", "conceptual understanding", "proof problems"]
            ),
            ProblemSolvingFramework(
                name="Rubber Duck Debugging",
                steps=[
                    "1. State the problem out loud",
                    "2. Explain what you expect to happen",
                    "3. Walk through your solution step by step",
                    "4. Spot the error or gap in reasoning"
                ],
                description="Verbalization technique to find errors and deepen understanding",
                best_for=["debugging", "finding mistakes", "verification"]
            )
        ]

    def _init_retrieval_techniques(self) -> List[RetrievalPractice]:
        """Initialize retrieval practice techniques."""
        return [
            RetrievalPractice(
                technique="Free Recall",
                description="Close your notes and write down everything you remember about the topic",
                example="After studying FSM minimization, close your notes and list all steps from memory",
                optimal_timing="Immediately after study session and 24 hours later"
            ),
            RetrievalPractice(
                technique="Self-Quizzing",
                description="Create questions for yourself and answer them without looking at notes",
                example="Quiz yourself: 'What's the difference between Mealy and Moore machines?' Answer without notes",
                optimal_timing="Use during study breaks and before exams"
            ),
            RetrievalPractice(
                technique="Teach-Back Method",
                description="Explain the concept to someone else (or pretend to)",
                example="Explain to a friend how to convert an NFA to a DFA",
                optimal_timing="After initial learning and before important reviews"
            ),
            RetrievalPractice(
                technique="Elaborative Interrogation",
                description="Ask yourself 'why' and 'how' questions about the material",
                example="Why does Moore machine minimization differ from Mealy? How are they similar?",
                optimal_timing="While reading and reviewing material"
            ),
            RetrievalPractice(
                technique="Interleaved Practice",
                description="Mix different types of problems instead of blocking by topic",
                example="Do one FSM problem, then a Boolean algebra problem, then back to FSM",
                optimal_timing="During practice sessions and review"
            )
        ]

    def get_study_tips(self, topic: str, difficulty: DifficultyLevel,
                       mastery: MasteryLevel) -> List[StudyTip]:
        """
        Get personalized study tips based on topic, difficulty, and current mastery.

        Args:
            topic: The topic being studied
            difficulty: Difficulty level of the material
            mastery: Current mastery level

        Returns:
            List of relevant study tips
        """
        tips = []

        # Preparation tips (before studying)
        if mastery == MasteryLevel.NEW:
            tips.append(StudyTip(
                category="preparation",
                tip="Start with a quick overview before diving into details",
                why="Advance organizers help your brain create a mental scaffold for new information",
                when_to_use="Before your first study session on a new topic"
            ))
            tips.append(StudyTip(
                category="preparation",
                tip="Set a specific learning goal for this session",
                why="Specific goals improve focus and help you monitor progress",
                when_to_use="At the start of every study session"
            ))

        # During study tips
        if difficulty == DifficultyLevel.HARD or mastery == MasteryLevel.LEARNING:
            tips.append(StudyTip(
                category="during_study",
                tip="Break the problem into smaller sub-problems",
                why="Cognitive load theory: working memory can only hold 4-7 items at once",
                when_to_use="When feeling overwhelmed by problem complexity"
            ))
            tips.append(StudyTip(
                category="during_study",
                tip="Draw diagrams or visualizations",
                why="Dual coding theory: visual + verbal encoding strengthens memory",
                when_to_use="For procedural and design problems"
            ))

        if mastery in [MasteryLevel.REVIEWING, MasteryLevel.MASTERED]:
            tips.append(StudyTip(
                category="during_study",
                tip="Practice explaining WHY each step works, not just HOW",
                why="Deep processing creates stronger, more flexible knowledge representations",
                when_to_use="When reviewing familiar material"
            ))

        # After study tips
        tips.append(StudyTip(
            category="after_study",
            tip="Test yourself immediately after studying (retrieval practice)",
            why="The 'testing effect': retrieval strengthens memory more than re-reading",
            when_to_use="Within 10 minutes of finishing a study session"
        ))

        if difficulty != DifficultyLevel.EASY:
            tips.append(StudyTip(
                category="after_study",
                tip="Sleep on it - review again tomorrow",
                why="Memory consolidation happens during sleep; spaced repetition beats cramming",
                when_to_use="After learning complex material"
            ))

        # Metacognitive tips
        tips.append(StudyTip(
            category="metacognitive",
            tip="Rate your confidence after each problem (1-5 scale)",
            why="Metacognitive monitoring improves self-awareness and study efficiency",
            when_to_use="After every practice problem"
        ))

        if mastery == MasteryLevel.LEARNING:
            tips.append(StudyTip(
                category="metacognitive",
                tip="Identify your specific error patterns",
                why="Pattern recognition helps you catch mistakes before they happen",
                when_to_use="When reviewing incorrect answers"
            ))

        return tips

    def get_problem_solving_framework(self, problem_type: str) -> ProblemSolvingFramework:
        """
        Get the most appropriate problem-solving framework for a problem type.

        Args:
            problem_type: Type of problem (e.g., "design", "theory", "proof")

        Returns:
            Most relevant problem-solving framework
        """
        # Match problem type to framework
        if problem_type in ["theory", "conceptual", "proof"]:
            return self._problem_solving_frameworks[2]  # Feynman Technique
        elif problem_type in ["design", "implementation"]:
            return self._problem_solving_frameworks[0]  # Polya's method
        elif problem_type in ["debugging", "verification"]:
            return self._problem_solving_frameworks[3]  # Rubber Duck
        else:
            return self._problem_solving_frameworks[1]  # IDEAL (general purpose)

    def get_self_assessment_prompts(self, topic: str,
                                    mastery: MasteryLevel) -> List[SelfAssessment]:
        """
        Get self-assessment prompts to check understanding.

        Args:
            topic: The topic being assessed
            mastery: Current mastery level

        Returns:
            List of self-assessment prompts
        """
        prompts = []

        # Understanding level (Bloom's taxonomy)
        prompts.append(SelfAssessment(
            prompt=f"Can you explain {topic} in your own words without looking at your notes?",
            category="understanding",
            difficulty=DifficultyLevel.EASY
        ))

        # Application level
        prompts.append(SelfAssessment(
            prompt=f"Can you solve a new problem involving {topic} that you haven't seen before?",
            category="application",
            difficulty=DifficultyLevel.MEDIUM
        ))

        # Analysis level
        if mastery in [MasteryLevel.REVIEWING, MasteryLevel.MASTERED]:
            prompts.append(SelfAssessment(
                prompt=f"Can you compare {topic} to related concepts and explain the key differences?",
                category="analysis",
                difficulty=DifficultyLevel.MEDIUM
            ))
            prompts.append(SelfAssessment(
                prompt=f"Can you identify when NOT to use {topic}?",
                category="analysis",
                difficulty=DifficultyLevel.HARD
            ))

        # Creation level
        if mastery == MasteryLevel.MASTERED:
            prompts.append(SelfAssessment(
                prompt=f"Can you create a new problem involving {topic} and solve it?",
                category="creation",
                difficulty=DifficultyLevel.HARD
            ))

        return prompts

    def get_retrieval_practice_suggestions(self,
                                          time_since_last_review: int,
                                          mastery: MasteryLevel) -> List[RetrievalPractice]:
        """
        Get retrieval practice suggestions based on timing and mastery.

        Args:
            time_since_last_review: Hours since last review
            mastery: Current mastery level

        Returns:
            List of appropriate retrieval techniques
        """
        suggestions = []

        # Immediate retrieval (< 1 hour)
        if time_since_last_review < 1:
            suggestions.append(self._retrieval_techniques[0])  # Free Recall
            suggestions.append(self._retrieval_techniques[1])  # Self-Quizzing

        # Short-term retrieval (1-24 hours)
        elif time_since_last_review < 24:
            suggestions.append(self._retrieval_techniques[2])  # Teach-Back
            suggestions.append(self._retrieval_techniques[3])  # Elaborative Interrogation

        # Long-term retrieval (> 24 hours)
        else:
            suggestions.append(self._retrieval_techniques[1])  # Self-Quizzing
            suggestions.append(self._retrieval_techniques[4])  # Interleaved Practice

        # For advanced learners, always include interleaved practice
        if mastery in [MasteryLevel.REVIEWING, MasteryLevel.MASTERED]:
            if self._retrieval_techniques[4] not in suggestions:
                suggestions.append(self._retrieval_techniques[4])

        return suggestions
