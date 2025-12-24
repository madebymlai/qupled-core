"""
Examina Core - AI-powered exam preparation library.

Main components:
- Analyzer: Exercise analysis and procedure extraction
- Tutor: AI tutoring with adaptive explanations
- ReviewEngine: Answer evaluation and exercise generation
"""

from core.analyzer import AnalysisResult, ExerciseAnalyzer
from core.answer_evaluator import RecallEvaluationResult
from core.note_splitter import NoteSection, NoteSplitter
from core.review_engine import (
    ExerciseExample,
    GeneratedExercise,
    ReviewEngine,
    ReviewEvaluation,
    calculate_mastery,
)
from core.tutor import Tutor

__all__ = [
    "ExerciseAnalyzer",
    "AnalysisResult",
    "Tutor",
    "RecallEvaluationResult",
    "NoteSplitter",
    "NoteSection",
    # Review Mode v2
    "ReviewEngine",
    "GeneratedExercise",
    "ReviewEvaluation",
    "ExerciseExample",
    "calculate_mastery",
]
