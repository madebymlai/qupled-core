# Phase 10: Learning Materials - Design Principles

## Overview

Phase 10 adds learning materials (lecture notes, slides, theory, worked examples) as first-class content, not forced into exercise format.

**Goal:** Make notes/slides genuinely usable as theory + examples, without breaking or slowing down the existing exam pipeline.

---

## 1. Smart Splitter: Pure Classifier/Segmenter

### Contract

**Input:** Raw text segments/pages from a PDF (with positional info)

**Output:** List of content segments, each with:
- `kind`: theory | worked_example | practice
- Positional info (page, character offsets)
- Confidence score (0.0-1.0)
- Plain text content

### Responsibilities

✅ **What the splitter DOES:**
- Classify: "What type of content is this?"
- Segment: "Where does each segment start and end?"
- Score: "How confident am I?"

❌ **What the splitter DOES NOT do:**
- Decide what to store in database (that's ingestion's job)
- Link to topics (that's analyzer's job)
- Format for display (that's tutor's job)

### Design Principle

> "Given this text, what learning/practice items exist here, what type are they, and where are they?"

The splitter is a **content classifier**, not a mini-ingestion pipeline.

### Implementation Contract

```python
@dataclass
class ContentSegment:
    """Pure classification output - no storage decisions."""
    kind: str  # 'theory' | 'worked_example' | 'practice'
    text: str
    page_number: int
    start_char: int
    end_char: int
    confidence: float
    title: Optional[str] = None

class SmartExerciseSplitter:
    def split_pdf_content(self, pdf_content, course_code) -> List[ContentSegment]:
        """Returns classified segments. Ingestion decides what to do with them."""
        pass
```

---

## 2. Ingestion Modes: Document Type Semantics

### Flag Meaning

`--material-type exams|notes` answers: **"What kind of document is this?"**

NOT: "What algorithm should we run?"

### Mode: exams

**Meaning:** PDFs that are mostly problem sets / past exams / homework

**Goal:** Maximize high-quality exercises

**Primary path:** Pattern-based splitting (fast, free, proven)

**Optional enhancement:** `--smart-split` for messy edge cases

**Algorithm choice:**
```
if material_type == 'exams':
    if smart_split:
        use SmartExerciseSplitter  # optional enhancer
    else:
        use ExerciseSplitter  # default, fast
```

### Mode: notes

**Meaning:** PDFs that are mostly lecture notes, slides, mixed theory/examples

**Goal:** Maximize high-quality learning_materials (theory + worked examples) + any practice

**Primary path:** Smart splitting (necessary for unstructured content)

**Algorithm choice:**
```
if material_type == 'notes':
    use SmartExerciseSplitter  # always, regardless of --smart-split flag
    # Because notes need classification, not just pattern matching
```

### User Experience

The flag describes **document semantics**, not **implementation details**:

```bash
# "I have exam PDFs" (structured problem sets)
examina ingest --course ADE --zip exams.zip --material-type exams

# "I have lecture notes" (unstructured theory/examples)
examina ingest --course ADE --zip notes.zip --material-type notes
```

---

## 3. Topic-Aware Linking: Symmetric Treatment

### Design Principle

> Treat learning materials and exercises **symmetrically** at the topic level.

Both are "things associated with topics" that use similar detection logic.

### Distinct Linking Responsibilities

**1. Content → Topics** (Many-to-many)
- Question: "Which topics does this content belong to?"
- Applies to: BOTH exercises AND learning materials
- Same logic: Topic detection from content text
- Tables: `exercise_core_loops`, `material_topics`

**2. Worked Examples → Exercises** (Many-to-many with type)
- Question: "Which exercises is this worked example particularly relevant to?"
- Applies to: Only worked_example materials
- Logic: Semantic matching within same topics
- Table: `material_exercise_links`

### Linking Strategy

**Phase A: Topic Detection** (Same for both)
```python
# For exercises (existing)
detected_topics = analyze_exercise(exercise_text)
link_exercise_to_topics(exercise_id, detected_topics)

# For materials (new, symmetric)
detected_topics = analyze_learning_material(material_text)
link_material_to_topics(material_id, detected_topics)
```

**Phase B: Cross-Linking** (Explicit relationships)
```python
# For each worked_example material:
#   1. Get its topics
#   2. Find exercises with same topics
#   3. Rank by semantic similarity
#   4. Link top 3-5 most similar
link_worked_examples_to_exercises(course_code)
```

### Anti-Patterns to Avoid

❌ Over-linking: Not everything to everything
❌ Hidden logic: Linking should be explicit in database, not tutor heuristics
❌ Opaque matching: Keep similarity thresholds configurable

### Success Criteria

✅ Explicit relationship layer in database
✅ Symmetric topic detection for exercises and materials
✅ Clear separation: topic detection ≠ cross-linking

---

## 4. Tutor: Explicit Theory → Example → Practice Flow

### Design Principle

> Theory → worked example → practice should be the **default learning script**, not an optional enhancement.

### Learning Flow Structure

```
┌─────────────────────────────────────┐
│ 1. TOPIC ENTRY                      │
│    User selects topic/core loop     │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│ 2. THEORY FIRST                     │
│    Show learning_materials          │
│    type='theory' for this topic     │
│    (High-signal, small set)         │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│ 3. WORKED EXAMPLES                  │
│    Show 1-2 worked_example          │
│    materials for this topic         │
│    (Step-by-step illustrations)     │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│ 4. PRACTICE                         │
│    Show exercises for this topic    │
│    Link to worked examples as hints │
└─────────────────────────────────────┘
```

### Implementation Requirements

**First-Class Flow:**
- NOT: "sometimes show notes if available"
- YES: "this IS the learning script"

**Configurable (in spirit):**
- How many theory snippets? (default: 2-3)
- Include worked examples? (default: yes)
- Max worked examples? (default: 1-2)
- Even if hardcoded initially, design for future configuration

**Fallback Gracefully:**
- No theory? Skip to worked examples
- No worked examples? Skip to exercises
- No materials at all? Existing exercise-only behavior

### Code Structure

```python
def learn(topic_id, show_theory=True, show_worked_examples=True,
          max_theory=3, max_examples=2):
    """
    Configurable learning flow with theory → example → practice.

    Args:
        show_theory: Whether to include theory materials (future: mode control)
        show_worked_examples: Whether to include examples
        max_theory: Max theory sections to show
        max_examples: Max worked examples to show
    """
    if show_theory:
        theory = fetch_theory_materials(topic_id, limit=max_theory)
        display_theory(theory)

    if show_worked_examples:
        examples = fetch_worked_examples(topic_id, limit=max_examples)
        display_worked_examples(examples)

    # Always show practice (existing behavior)
    exercises = fetch_exercises(topic_id)
    display_exercises(exercises, linked_examples=examples)
```

### Future Modes (Design for)

While not implemented yet, design should allow:
- **Deep Learning Mode:** Full theory + examples + practice
- **Quick Drill Mode:** Skip theory, go straight to practice
- **Example-Only Mode:** Just worked examples, no theory

---

## 5. Success Criteria: No Regression + Notes Coverage

### For Exams/Problem Sets

**Requirement:** Zero regression in exercise extraction quality

**Validation:**
- Structured exam PDFs still use pattern-based path primarily
- Exercise count, quality, metadata unchanged from Phase 9
- No performance degradation (speed, cost)

**Test Cases:**
- ADE course (B006802) - existing exercises should still work
- SO course - Q+A detection should still work
- Any course with clean, numbered exercises

### For Notes/Slides

**Requirement:** Reasonable coverage with low false positives

**Coverage Goals:**
- Theory sections: Detect 70%+ of actual theory content
- Worked examples: Detect 60%+ of examples with solutions
- Practice: Extract any problems found (even if minority of content)

**False Positive Goals:**
- Don't misclassify non-exercise content as exercises (<10% error rate)
- Don't treat headers/instructions as theory sections

**Test Cases:**
- Lecture notes with clear sections (theory, examples, problems)
- Slides with mixed content
- PDFs with embedded figures and diagrams

### Overall Phase 10 Purpose

> "Make notes/slides genuinely usable as theory + examples, without breaking or slowing down the existing exam pipeline."

**Success means:**
1. ✅ Exam ingestion works exactly as before (no regression)
2. ✅ Notes ingestion creates useful learning materials (new capability)
3. ✅ Tutor provides structured learning flow (theory → example → practice)
4. ✅ All changes are provider-agnostic, configurable, web-ready

---

## Responsibility Matrix

| Component | Responsibility | Does NOT Do |
|-----------|---------------|-------------|
| **Smart Splitter** | Classify content into theory/worked_example/practice segments | Store in database, link to topics, format for display |
| **Ingestion** | Turn segments into database records (exercises + materials) | Classify content types, link to topics |
| **Analyzer** | Map exercises and materials to topics; link materials to exercises | Extract content from PDFs, display to users |
| **Tutor** | Orchestrate theory → example → practice learning experience | Store content, detect topics, classify segments |

---

## Configuration Points

All Phase 10 features should respect existing config patterns:

```python
# config.py additions (examples)
LEARNING_MATERIALS_ENABLED = True
MAX_THEORY_SECTIONS_IN_LEARN = 3
MAX_WORKED_EXAMPLES_IN_LEARN = 2
MATERIAL_TOPIC_SIMILARITY_THRESHOLD = 0.85
WORKED_EXAMPLE_EXERCISE_SIMILARITY_THRESHOLD = 0.70
SHOW_THEORY_BY_DEFAULT = True
SHOW_WORKED_EXAMPLES_BY_DEFAULT = True
```

---

## Migration Path

Phase 10 is **additive**, not destructive:

1. Existing exercises continue to work (no schema changes to exercises table)
2. New learning_materials tables are independent
3. Pattern-based splitting remains default for exams
4. Tutor falls back gracefully when no materials exist

**Backward compatibility:** All existing features (analyze, quiz, progress) work unchanged.
