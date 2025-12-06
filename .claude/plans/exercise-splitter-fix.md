# ExerciseSplitter Fix: Context-Aware Sub-Exercise Extraction

## Problem

Testing on `Prova intermedia 2024-01-29 - SOLUZIONI v4.pdf`:
- **Expected**: Sub-exercises WITH parent context
- **Actual**: 51 exercises detected, many without context or are junk

**Root causes:**

1. **Sub-exercises lose parent context**
   - Parent exercise has sub-parts
   - Each sub-part is detected separately WITHOUT the setup/context
   - Result: sub-question without knowing WHAT it's asking about

2. **Repeated page headers detected as exercises**
   - Every page has numbered items that match structural patterns
   - These get split as junk exercises

3. **Page-by-page splitting loses multi-page content**
   - Exercise spanning pages 1-3 gets truncated at page boundaries

4. **Hardcoded regex patterns fail edge cases**
   - Can't handle all languages/formats
   - OCR errors break pattern matching
   - Non-standard marker formats missed

## Solution: LLM-Based Document-Level Post-Processing

### New Architecture

```
Phase 0: LLM MARKER DETECTION (1 call)
──────────────────────────────────────────────
- Send full document text to LLM (fast model)
- LLM identifies parent and sub markers (language-agnostic)
- Returns structured JSON with positions
- No hardcoded patterns - LLM understands exam structure

Phase 1: BUILD HIERARCHY
──────────────────────────────────────────────
- Group markers into parent-child relationships
- Parent: exercise at page 1, pos 0
  ├─ Sub: sub-question at pos 200
  ├─ Sub: sub-question at pos 400
  └─ Sub: sub-question at pos 500
- Calculate text ranges for each marker

Phase 2: EXTRACT CONTEXT
──────────────────────────────────────────────
- For each parent exercise:
  - Context = text between parent marker and first sub-marker
  - This includes: problem setup, diagrams, definitions

Phase 3: EXPAND SUB-EXERCISES
──────────────────────────────────────────────
- For each sub-exercise:
  - Full text = Parent context + Sub-exercise text
  - Exercise ID = "{parent_num}.{sub_num}"

Phase 4: OUTPUT
──────────────────────────────────────────────
- Flat list of context-rich exercises
- Each sub-exercise has full context for standalone understanding
```

### Example

**Input:**
```
Esercizio 1
Si consideri il seguente automa A = (Q, Σ, δ, q0, F) dove:
Q = {q0, q1, q2}, Σ = {a, b}, ...
[diagram]

1. Determinare se l'automa è deterministico
2. Costruire la tabella di transizione
3. Minimizzare l'automa

Esercizio 2
...
```

**Output (3 exercises):**
```
Exercise 1.1:
"Esercizio 1
Si consideri il seguente automa A = (Q, Σ, δ, q0, F) dove:
Q = {q0, q1, q2}, Σ = {a, b}, ...
[diagram]

1. Determinare se l'automa è deterministico"

Exercise 1.2:
"Esercizio 1
Si consideri il seguente automa A = (Q, Σ, δ, q0, F) dove:
Q = {q0, q1, q2}, Σ = {a, b}, ...
[diagram]

2. Costruire la tabella di transizione"

Exercise 1.3:
"Esercizio 1
Si consideri il seguente automa A = (Q, Σ, δ, q0, F) dove:
Q = {q0, q1, q2}, Σ = {a, b}, ...
[diagram]

3. Minimizzare l'automa"
```

---

## Implementation Steps

### Step 1: Define data structures
**File:** `/home/laimk/git/examina/core/exercise_splitter.py`
**Complexity:** Low

```python
from dataclasses import dataclass
from enum import Enum

class MarkerType(Enum):
    PARENT = "parent"      # Main exercise marker
    SUB = "sub"            # Sub-question marker

@dataclass
class Marker:
    """A detected exercise marker in the document."""
    marker_type: MarkerType
    marker_text: str       # The actual marker text (e.g., "Exercise 1", "a)")
    number: str            # Extracted number/letter ("1", "a", etc.)
    start_position: int    # Character position where marker starts
    question_start: int    # Character position where question text begins

@dataclass
class ExerciseNode:
    """Hierarchical exercise structure."""
    marker: Marker
    context: str           # Setup text (for parents)
    question_text: str     # The actual question
    children: list["ExerciseNode"]
    parent: "ExerciseNode | None"
```

### Step 2: LLM detects PATTERN, we find all markers
**File:** `/home/laimk/git/examina/core/exercise_splitter.py`
**Complexity:** Medium

**Key insight:** LLM identifies the PATTERN (e.g., "Esercizio"), we search FULL document.
This handles long documents (>15k chars) without missing markers.

```python
@dataclass
class MarkerPattern:
    """Pattern for exercise markers."""
    keyword: str           # e.g., "Esercizio", "Exercise", "Problem"
    has_sub_markers: bool  # Whether document has sub-questions
    sub_pattern: str | None  # e.g., "numbered" (1., 2.) or "lettered" (a), b))

def _detect_pattern_with_llm(self, full_text: str, llm_manager: "LLMManager") -> MarkerPattern | None:
    """
    Use LLM to detect the PATTERN used for exercise markers.
    LLM only needs to see sample text to identify the pattern.
    """
    prompt = '''Analyze this academic exam document and identify the PATTERN used for exercise markers.

TASK: Identify:
1. What word/phrase marks the start of main exercises? (e.g., "Exercise", "Problem", "Question")
2. Are there sub-questions within exercises? If so, what format? (e.g., "1.", "a)", "(i)")

Return JSON:
{
  "parent_keyword": "the exact word that marks exercises (null if none)",
  "has_sub_markers": true/false,
  "sub_format": "numbered" or "lettered" or "roman" or null
}

Return ONLY the JSON object.'''

    # Only need first ~10k chars to detect pattern
    text_for_llm = full_text[:10000] if len(full_text) > 10000 else full_text

    response = llm_manager.generate(
        prompt + f'\n\nDOCUMENT TEXT:\n"""\n{text_for_llm}\n"""',
        temperature=0.0,
        model="fast"
    )

    try:
        data = json.loads(response)
        if not data.get("parent_keyword"):
            return None
        return MarkerPattern(
            keyword=data["parent_keyword"],
            has_sub_markers=data.get("has_sub_markers", False),
            sub_pattern=data.get("sub_format"),
        )
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Failed to parse LLM pattern response: {e}")
        return None

def _find_all_markers(self, full_text: str, pattern: MarkerPattern) -> list[Marker]:
    """
    Find ALL markers in full document using detected pattern.
    No truncation - searches entire text.
    """
    markers = []

    # Build regex for parent markers: "keyword + number"
    parent_regex = re.compile(
        rf'(?:^|\n)\s*{re.escape(pattern.keyword)}\s+(\d+)',
        re.IGNORECASE | re.MULTILINE
    )

    for match in parent_regex.finditer(full_text):
        markers.append(Marker(
            marker_type=MarkerType.PARENT,
            marker_text=match.group(0).strip(),
            number=match.group(1),
            start_position=match.start(),
            question_start=match.end(),
        ))

    # Find sub-markers if pattern indicates they exist
    if pattern.has_sub_markers:
        sub_regex = self._get_sub_regex(pattern.sub_pattern)
        if sub_regex:
            for match in sub_regex.finditer(full_text):
                markers.append(Marker(
                    marker_type=MarkerType.SUB,
                    marker_text=match.group(0).strip(),
                    number=match.group(1),
                    start_position=match.start(),
                    question_start=match.end(),
                ))

    # Sort by position
    markers.sort(key=lambda m: m.start_position)
    return markers

def _get_sub_regex(self, sub_format: str | None) -> re.Pattern | None:
    """Get regex pattern for sub-markers based on format."""
    patterns = {
        "numbered": re.compile(r'(?:^|\n)\s*(\d+)\.\s+(?=[A-Z])', re.MULTILINE),
        "lettered": re.compile(r'(?:^|\n)\s*([a-z])\)\s+(?=[A-Z])', re.MULTILINE | re.IGNORECASE),
        "roman": re.compile(r'(?:^|\n)\s*\(([ivxlcdm]+)\)\s+', re.MULTILINE | re.IGNORECASE),
    }
    return patterns.get(sub_format)
```

### Step 3: Build hierarchy from markers
**File:** `/home/laimk/git/examina/core/exercise_splitter.py`
**Complexity:** Medium

```python
def _build_hierarchy(
    self,
    markers: list[Marker],
    full_text: str
) -> list[ExerciseNode]:
    """
    Build parent-child hierarchy from flat marker list.
    """
    if not markers:
        return []

    roots = []  # Top-level exercises
    current_parent = None

    for i, marker in enumerate(markers):
        # Calculate text range (from question_start until next marker or end)
        next_pos = markers[i + 1].start_position if i + 1 < len(markers) else len(full_text)
        text_range = full_text[marker.question_start:next_pos].strip()

        if marker.marker_type == MarkerType.PARENT:
            # Start new parent exercise
            node = ExerciseNode(
                marker=marker,
                context="",  # Will be filled when first child found
                question_text=text_range,
                children=[],
                parent=None,
            )
            roots.append(node)
            current_parent = node

        elif marker.marker_type == MarkerType.SUB and current_parent:
            # Add as child of current parent
            node = ExerciseNode(
                marker=marker,
                context="",  # Children don't have their own context
                question_text=text_range,
                children=[],
                parent=current_parent,
            )
            current_parent.children.append(node)

            # If this is first child, extract context from parent
            if len(current_parent.children) == 1:
                # Context = parent text BEFORE first sub-marker
                context_end = marker.start_position - current_parent.marker.question_start
                current_parent.context = current_parent.question_text[:context_end].strip()

    return roots
```

### Step 5: Expand to flat exercises with context
**File:** `/home/laimk/git/examina/core/exercise_splitter.py`
**Complexity:** Medium

```python
def _expand_exercises(
    self,
    roots: list[ExerciseNode],
    source_pdf: str,
    course_code: str,
) -> list[Exercise]:
    """
    Expand hierarchy into flat list of exercises.
    Each sub-exercise gets parent context prepended.
    """
    exercises = []

    for parent in roots:
        if parent.children:
            # Parent has sub-exercises: expand each with context
            for child in parent.children:
                full_text = f"{parent.context}\n\n{child.question_text}"
                exercise_num = f"{parent.marker.number}.{child.marker.number}"

                exercises.append(self._create_exercise(
                    text=full_text,
                    exercise_number=exercise_num,
                    source_pdf=source_pdf,
                    course_code=course_code,
                    page_num=child.marker.page_num,
                ))
        else:
            # Parent has no sub-exercises: use as single exercise
            exercises.append(self._create_exercise(
                text=parent.question_text,
                exercise_number=parent.marker.number,
                source_pdf=source_pdf,
                course_code=course_code,
                page_num=parent.marker.page_num,
            ))

    return exercises
```

### Step 6: Update main split function
**File:** `/home/laimk/git/examina/core/exercise_splitter.py`
**Complexity:** Medium

```python
def split_pdf_content(
    self,
    pdf_content: PDFContent,
    course_code: str,
    llm_manager: "LLMManager | None" = None
) -> list[Exercise]:
    """
    Split PDF into exercises with context-aware sub-exercise extraction.

    Args:
        pdf_content: Extracted PDF content
        course_code: Course code for ID generation
        llm_manager: LLM manager for pattern detection (optional)
    """
    self.exercise_counter = 0

    # Build full document text
    full_text = "\n".join(page.text for page in pdf_content.pages)
    source_pdf = pdf_content.file_path.name

    # Phase 0: Detect pattern (LLM or fallback to regex)
    markers = []
    if llm_manager:
        pattern = self._detect_pattern_with_llm(full_text, llm_manager)
        if pattern:
            markers = self._find_all_markers(full_text, pattern)

    # Fallback: use existing regex-based detection
    if not markers:
        markers = self._detect_markers_with_regex(full_text)

    if not markers:
        # No markers found - fall back to page-based splitting
        return self._split_unstructured(pdf_content, course_code)

    # Phase 1: Build hierarchy
    roots = self._build_hierarchy(markers, full_text)

    # Phase 2-3: Expand to flat exercises with context
    exercises = self._expand_exercises(roots, source_pdf, course_code)

    # Phase 4: Validate and filter
    exercises = [ex for ex in exercises if self.validate_exercise(ex)]

    return exercises

def _detect_markers_with_regex(self, full_text: str) -> list[Marker]:
    """
    Fallback: detect markers using existing regex patterns.
    Used when LLM is not available or fails.
    """
    # Use existing _detect_exercise_pattern from current code
    pattern = self._detect_exercise_pattern(full_text)
    if not pattern:
        return []

    markers = []
    for match in pattern.finditer(full_text):
        markers.append(Marker(
            marker_type=MarkerType.PARENT,
            marker_text=match.group(0).strip(),
            number=match.group(1) if match.groups() else "1",
            start_position=match.start(),
            question_start=match.end(),
        ))

    return markers
```

### Step 7: Handle repeated headers
**File:** `/home/laimk/git/examina/core/exercise_splitter.py`
**Complexity:** Low

```python
def _detect_repeated_headers(self, pdf_content: PDFContent) -> set[str]:
    """
    Detect text fragments that appear on multiple pages.
    These are likely headers/instructions to filter out.
    """
    from collections import Counter

    # Extract first N lines from each page
    page_headers = []
    for page in pdf_content.pages:
        lines = page.text.strip().split('\n')[:10]
        page_headers.append(set(line.strip() for line in lines if line.strip()))

    # Find lines appearing on 3+ pages
    all_lines = [line for page_lines in page_headers for line in page_lines]
    line_counts = Counter(all_lines)

    return {line for line, count in line_counts.items()
            if count >= 3 and len(line) < 200}
```

### Step 8: Fallback for unstructured PDFs
**File:** `/home/laimk/git/examina/core/exercise_splitter.py`
**Complexity:** Low

```python
def _split_unstructured(self, pdf_content: PDFContent, course_code: str) -> list[Exercise]:
    """
    Fallback for PDFs without clear exercise markers.
    Uses structural patterns or treats each page as exercise.
    """
    exercises = []
    source_pdf = pdf_content.file_path.name

    for i, page in enumerate(pdf_content.pages):
        text = page.text.strip()
        if not text or len(text) < 100:
            continue

        if self._is_instruction_page(text):
            continue

        exercises.append(self._create_exercise(
            text=text,
            exercise_number=str(i + 1),
            source_pdf=source_pdf,
            course_code=course_code,
            page_num=i,
        ))

    return exercises
```

---

## Testing Plan

### Test 1: Context propagation
```python
# Verify sub-exercises have parent context
pdf_path = "test-data/ADE-ESAMI/Prova intermedia 2024-01-29 - SOLUZIONI v4.pdf"
exercises = splitter.split_pdf_content(pdf_content, "ADE")

# Each sub-exercise should contain the parent setup
for ex in exercises:
    if "." in ex.exercise_number:  # Is sub-exercise
        assert "Si consideri" in ex.text or "dato" in ex.text.lower()
```

### Test 2: Exercise count
```python
# Should have more exercises than parent count (sub-exercises expanded)
# But NOT 51 junk exercises
assert 6 < len(exercises) < 30  # Reasonable range
```

### Test 3: No junk exercises
```python
for ex in exercises:
    assert len(ex.text) > 100
    assert "soluzioni E procedimenti" not in ex.text[:50]
```

### Test 4: Multi-page exercises
```python
# Exercises spanning pages should have complete content
# (Test with specific PDF that has multi-page exercises)
```

---

## Edge Cases

1. **No sub-exercises**: Parent exercise used as-is
2. **Nested sub-exercises** (1.a, 1.b): Current design handles one level
3. **PDFs without "Esercizio N"**: Falls back to unstructured splitting
4. **Very long context**: May need truncation (consider max_context_length)
5. **Images/diagrams in context**: Preserved in text (may appear as [image] or blank)

---

## Files Changed

| File | Changes |
|------|---------|
| `/home/laimk/git/examina/core/exercise_splitter.py` | Major refactor |

---

## Estimated Impact

| Metric | Before | After |
|--------|--------|-------|
| Exercises detected | 51 (junk) | ~15-20 (meaningful) |
| Context per exercise | Missing | Full parent context |
| Multi-page support | Broken | Working |
| LLM analysis quality | Poor | Good (has context) |
