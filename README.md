# Examina Core

Lightweight business logic for AI-powered exam preparation.

## Architecture

```
examina (this repo)   - Core logic, no heavy deps
    ↓ imported by
examina-cloud         - Web platform (FastAPI + React + PostgreSQL)
examina-cli           - Local CLI (ChromaDB, vector search)
```

**Rule**: Cloud and CLI import from core, never reimplement.

## Modules

| Module | Purpose |
|--------|---------|
| `core/analyzer.py` | Exercise analysis, knowledge extraction |
| `core/tutor.py` | Knowledge item teaching (Learn mode) |
| `core/review_engine.py` | Exercise generation & answer evaluation (Review mode) |
| `core/fsrs_scheduler.py` | FSRS spaced repetition algorithm |
| `core/answer_evaluator.py` | Student answer evaluation |
| `core/pdf_processor.py` | PDF text extraction |
| `core/exercise_splitter.py` | Exercise detection and splitting |
| `core/note_splitter.py` | Note/summary splitting |
| `core/solution_separator.py` | Separate exercises from solutions |
| `core/merger.py` | Merge duplicate knowledge items |
| `core/features.py` | Feature extraction for ML |
| `core/active_learning.py` | Active learning strategies |

## Installation

```bash
pip install git+https://github.com/madebymlai/examina.git
```

## Usage

```python
from core.analyzer import ExerciseAnalyzer
from core.tutor import Tutor
from core.review_engine import ReviewEngine
from core.fsrs_scheduler import FSRSScheduler, ReviewResult
from models.llm_manager import LLMManager

llm = LLMManager()

# Analyze exercise
analyzer = ExerciseAnalyzer(llm_manager=llm, language="en")
result = analyzer.analyze_exercise(
    exercise_text="Prove that the sum of two even numbers is even.",
    course_name="Discrete Math"
)

# Learn mode - teach a knowledge item
tutor = Tutor(llm_manager=llm, language="en")
explanation = tutor.learn_knowledge_item(
    knowledge_item={"name": "Even Numbers", "learning_approach": "conceptual"},
    exercises=[...]
)

# Review mode - generate exercise and evaluate answer
engine = ReviewEngine(llm)
exercise = engine.generate_exercise(
    knowledge_item_name="Even Numbers",
    learning_approach="conceptual",
    examples=[...]
)
result = engine.evaluate_answer(
    exercise_text=exercise.exercise_text,
    expected_answer=exercise.expected_answer,
    student_answer="user's answer",
    exercise_type=exercise.exercise_type
)

# FSRS scheduling
scheduler = FSRSScheduler()
review_result = scheduler.review(
    card_state=current_card_state,
    rating=3,  # 1=Again, 2=Hard, 3=Good, 4=Easy
)
next_review_date = review_result.next_review_date
```

## LLM Providers

| Provider | Best For |
|----------|----------|
| DeepSeek | Analysis, reasoning |
| Groq | Fast responses |
| Anthropic | Premium explanations |

## Related

- [examina-cloud](https://github.com/madebymlai/examina-cloud) - Web platform
- [examina-cli](https://github.com/madebymlai/examina-cli) - Local CLI
