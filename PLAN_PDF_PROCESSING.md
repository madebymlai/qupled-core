# PDF Processing Rework - LLM-Based Exercise Extraction

## Overview

Replace regex-based exercise splitting with LLM-powered extraction for accurate exercise boundary detection regardless of PDF format.

## Current Architecture

```
PDF File
    ↓
PDFProcessor (pymupdf) → extracts text per page
    ↓
ExerciseSplitter (regex) → finds exercise markers
    ↓
List[Exercise] → stored in database
```

**Problems:**
- Regex only matches specific patterns (Esercizio N, Exercise N, etc.)
- Misses exercises with non-standard numbering
- Can't handle complex layouts or implicit boundaries

## New Architecture

```
PDF File
    ↓
PDFProcessor (pymupdf) → extracts text per page
    ↓
LLMExerciseSplitter (DeepSeek) → identifies exercise boundaries
    ↓
List[Exercise] → stored in database
```

---

## Phase 1: Core LLM Splitter

### 1.1 Create LLMExerciseSplitter class

**File:** `core/llm_exercise_splitter.py`

```python
class LLMExerciseSplitter:
    """LLM-powered exercise boundary detection."""

    def __init__(self, llm_manager: LLMManager):
        self.llm = llm_manager

    def split_pdf_content(self, pdf_content: PDFContent, course_code: str) -> List[Exercise]:
        """Split PDF into exercises using LLM."""
        # Combine all pages into one text with page markers
        full_text = self._prepare_text_with_markers(pdf_content)

        # Ask LLM to identify exercise boundaries
        boundaries = self._detect_boundaries(full_text)

        # Extract exercises based on boundaries
        exercises = self._extract_exercises(full_text, boundaries, pdf_content, course_code)

        return exercises
```

### 1.2 Prompt Design

**System prompt:**
```
You are an expert at analyzing academic documents. Your task is to identify individual exercises/problems in exam papers and homework sheets.

An exercise is a complete problem that a student needs to solve. It may contain:
- A problem statement
- Sub-questions (a, b, c or i, ii, iii)
- Given data or constraints
- Required calculations

Sub-questions belong to their parent exercise - do NOT split them.
```

**User prompt template:**
```
Analyze this academic document and identify each distinct exercise/problem.

For each exercise found, provide:
1. exercise_number: The number/label (e.g., "1", "2", "I", "A")
2. start_marker: The exact text that starts the exercise (first 50 chars)
3. end_marker: The exact text that ends the exercise (last 50 chars)

Document:
---
{text}
---

Return JSON array:
[
  {"exercise_number": "1", "start_marker": "...", "end_marker": "..."},
  ...
]
```

### 1.3 Response Parsing

- Parse JSON response from LLM
- Handle malformed JSON (retry with simpler prompt)
- Validate boundaries exist in original text
- Fall back to regex if LLM fails completely

---

## Phase 2: Integration

### 2.1 Update Worker Task

**File:** `examina-cloud/backend/worker/tasks/ingest.py`

```python
# Replace:
from core.exercise_splitter import ExerciseSplitter

# With:
from core.llm_exercise_splitter import LLMExerciseSplitter
from models.llm_manager import LLMManager

# In ingest_pdf_task:
llm = LLMManager(provider="deepseek")
splitter = LLMExerciseSplitter(llm_manager=llm)
exercises = splitter.split_pdf_content(pdf_content, course_code)
```

### 2.2 Keep Regex as Fallback

```python
def split_pdf_content(self, pdf_content, course_code):
    try:
        # Try LLM first
        exercises = self._llm_split(pdf_content, course_code)
        if exercises:
            return exercises
    except Exception as e:
        logger.warning(f"LLM splitting failed: {e}")

    # Fallback to regex
    regex_splitter = ExerciseSplitter()
    return regex_splitter.split_pdf_content(pdf_content, course_code)
```

---

## Phase 3: Optimization

### 3.1 Cost Optimization

**Batch processing for long PDFs:**
- If PDF > 20 pages, process in chunks of 10 pages
- Reduces context window usage
- Allows parallel processing

**Token estimation:**
- Estimate tokens before sending
- If > 50k tokens, split into batches

### 3.2 Caching

- Cache LLM responses by PDF content hash
- If same PDF uploaded twice, reuse extraction
- Store in Redis with 24h TTL

### 3.3 Async Processing

- Already using Celery workers
- No changes needed for async

---

## Phase 4: Quality Improvements

### 4.1 Confidence Scoring

LLM returns confidence for each boundary:
```json
{
  "exercise_number": "1",
  "start_marker": "...",
  "confidence": 0.95
}
```

Log low-confidence extractions for review.

### 4.2 Exercise Validation

After extraction, validate:
- Each exercise has minimum content (> 50 chars)
- No duplicate exercise numbers
- Exercises don't overlap
- Page numbers are consistent

### 4.3 Image Handling

- Detect if PDF has images/diagrams
- Associate images with correct exercises based on position
- Flag exercises that may have missing visual content

---

## Phase 5: Testing

### 5.1 Test Dataset

Create test PDFs covering:
- [ ] Italian exams (Esercizio N)
- [ ] English exams (Problem N)
- [ ] Numbered only (1., 2., 3.)
- [ ] Roman numerals (I, II, III)
- [ ] Letter-based (A, B, C)
- [ ] No explicit markers (paragraph-based)
- [ ] Mixed languages
- [ ] Multi-page exercises
- [ ] Sub-questions (a, b, c)

### 5.2 Accuracy Metrics

Track:
- Correct exercise count: target > 95%
- Boundary accuracy: target > 90%
- False positives (splitting too much)
- False negatives (merging separate exercises)

### 5.3 A/B Testing

- Run both regex and LLM in parallel
- Compare results
- Log discrepancies for analysis

---

## Implementation Order

1. **Week 1: Core**
   - [ ] Create `LLMExerciseSplitter` class
   - [ ] Design and test prompts with sample PDFs
   - [ ] Implement JSON parsing with error handling

2. **Week 2: Integration**
   - [ ] Update worker task to use new splitter
   - [ ] Add fallback to regex
   - [ ] Deploy to staging

3. **Week 3: Optimization**
   - [ ] Add caching layer
   - [ ] Implement batch processing for large PDFs
   - [ ] Monitor costs and accuracy

4. **Week 4: Polish**
   - [ ] Add confidence scoring
   - [ ] Improve prompts based on real-world results
   - [ ] Document edge cases

---

## Cost Estimate

| Scenario | PDFs/month | Cost/PDF | Monthly Cost |
|----------|------------|----------|--------------|
| Light    | 100        | $0.01    | $1           |
| Medium   | 1,000      | $0.01    | $10          |
| Heavy    | 10,000     | $0.01    | $100         |

DeepSeek pricing makes this very affordable.

---

## Files to Create/Modify

### New Files (examina-core)
- `core/llm_exercise_splitter.py` - Main LLM splitter class
- `core/prompts/exercise_splitting.py` - Prompt templates
- `tests/test_llm_exercise_splitter.py` - Unit tests

### Modified Files (examina-cloud)
- `backend/worker/tasks/ingest.py` - Use new splitter
- `backend/app/core/config.py` - Add config for LLM splitting

### Keep (no changes)
- `core/exercise_splitter.py` - Keep as fallback
- `core/pdf_processor.py` - Text extraction unchanged

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| LLM API downtime | Fallback to regex |
| Unexpected costs | Token limits, caching |
| Slow processing | Async workers, already handled |
| Bad LLM output | JSON validation, retry logic |
| Privacy concerns | DeepSeek processes text only, no storage |

---

## Success Criteria

- [ ] Exercise detection accuracy > 95%
- [ ] Processing time < 30 seconds for 20-page PDF
- [ ] Cost < $0.05 per PDF average
- [ ] Zero user complaints about incorrect splitting
