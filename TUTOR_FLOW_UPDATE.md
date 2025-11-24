# Tutor Learning Flow Update

## Summary

Updated `/home/laimk/git/Examina/core/tutor.py` to implement a **theory → worked example → practice** learning flow. This enhancement provides students with foundational theory and worked examples before presenting practice exercises.

## Changes Made

### 1. Enhanced `learn()` Method

**Location:** Lines 44-225

**Key Changes:**
- Added fetching of theory materials and worked examples from the database
- Modified SQL query to include `topic_id`
- Added calls to `db.get_learning_materials_by_topic()` for both theory and worked examples
- Updated content assembly to display materials in proper pedagogical order
- Enhanced metadata to track material counts

**New Flow:**
```
OLD: Fetch exercises → Show exercises → Explain concepts

NEW: Fetch topic →
     1. Show theory materials (if any)
     2. Show worked examples (if any)
     3. Show prerequisite concepts (if enabled)
     4. Show LLM-generated explanation
     5. Show study strategies (if enabled)
     6. Show metacognitive tips (if enabled)
     7. Show official solutions (if available)
```

### 2. New Helper Method: `_display_theory_materials()`

**Location:** Lines 585-643

**Purpose:** Display theory materials for a topic in a clear, educational format.

**Features:**
- Bilingual support (English/Italian)
- Formatted title and introduction
- Source information (PDF and page number)
- Clean separation between multiple theory sections

**Output Example:**
```
THEORY MATERIALS

Before starting with exercises, let's review the foundational theory:

## Introduction to Finite State Machines
[Source: lecture_notes.pdf, page 15]

A finite state machine (FSM) is a mathematical model...
```

### 3. New Helper Method: `_display_worked_examples()`

**Location:** Lines 645-726

**Purpose:** Display worked examples showing complete step-by-step solutions.

**Features:**
- Bilingual support (English/Italian)
- Clear title and introduction emphasizing "how it's done"
- Source information
- Helpful note explaining the purpose
- Separation between multiple examples

**Output Example:**
```
WORKED EXAMPLES

Now let's see how to apply this theory through step-by-step worked examples:

### Worked Example: Designing a 2-bit Counter
[Source: exam_2022_solutions.pdf, page 3]

Problem: Design a FSM that counts from 0 to 3...
Solution:
1. Define states: S0 (count=0), S1 (count=1)...

[Note: This is a complete example showing how to solve this type of problem.]
```

### 4. Enhanced `practice()` Method

**Location:** Lines 227-293

**Key Changes:**
- Added fetching of linked materials for exercises
- Filter for worked examples specifically
- Added hints about available worked examples
- Enhanced metadata to track hint availability

**New Feature:** When a practice exercise has linked worked examples, students see:
```
[Exercise text here]

---

Hint: For a similar problem, see this worked example: "Example: FSM Design Pattern"
```

### 5. New Helper Method: `_format_worked_example_hints()`

**Location:** Lines 728-764

**Purpose:** Format hints about worked examples linked to an exercise.

**Features:**
- Bilingual support
- Handles single or multiple worked examples
- Clean formatting with clear separator
- References materials by title

### 6. Updated Metadata

**Added fields in `learn()` response:**
- `theory_materials_count`: Number of theory materials displayed
- `worked_examples_count`: Number of worked examples displayed
- `has_theory`: Boolean indicating theory materials presence
- `has_worked_examples`: Boolean indicating worked examples presence

**Added fields in `practice()` response:**
- `has_worked_example_hints`: Boolean indicating if hints are shown
- `worked_example_count`: Number of linked worked examples

## Design Principles

### 1. Pedagogical Flow
Implements the proven "explain → show → do" teaching pattern:
- **Explain:** Theory materials provide foundational understanding
- **Show:** Worked examples demonstrate application
- **Do:** Practice exercises allow students to apply knowledge

### 2. Optional Enhancement
- Theory and worked examples are **optional** - they enhance learning but don't break existing functionality
- If no materials exist, the tutor proceeds with existing behavior
- Backward compatible with exercise-only learning

### 3. Bilingual Support
All new sections support both English and Italian:
- Headers and labels translated
- Consistent terminology
- Respects tutor's language setting

### 4. Source Attribution
All materials include:
- Source PDF filename
- Page number (when available)
- Clear separation from LLM-generated content

## Database Integration

**Methods Used:**
- `db.get_learning_materials_by_topic(topic_id, material_type)` - Fetch theory/worked examples
- `db.get_materials_for_exercise(exercise_id)` - Find linked materials for exercises

**Material Types:**
- `'theory'` - Foundational theoretical content
- `'worked_example'` - Complete worked solutions

## Testing

**Test Script:** `/home/laimk/git/Examina/test_tutor_flow.py`

**Test Coverage:**
1. ✅ Theory materials fetching
2. ✅ Worked examples fetching
3. ✅ Content display formatting
4. ✅ Metadata tracking
5. ✅ Fallback behavior (no materials)
6. ✅ Practice hints integration
7. ✅ Bilingual support

**Results:**
- All code compiles without errors
- Methods work correctly with and without materials
- Backward compatibility maintained
- No breaking changes to existing functionality

## What Users Will See

### Example 1: Learning with Full Materials

```
THEORY MATERIALS

Before starting with exercises, let's review the foundational theory:

## Finite State Machine Fundamentals
[Source: lecture_notes.pdf, page 12]

[Theory content here...]

============================================================

WORKED EXAMPLES

Now let's see how to apply this theory through step-by-step worked examples:

### Example: Mealy Machine Design
[Source: exam_solutions.pdf, page 5]

[Complete worked solution here...]

============================================================

[LLM-generated explanation and procedure...]

============================================================

[Study strategies, metacognitive tips, official solutions...]
```

### Example 2: Learning without Materials (Fallback)

```
[LLM-generated explanation and procedure...]

============================================================

[Study strategies, metacognitive tips, official solutions...]
```

### Example 3: Practice with Hints

```
Exercise 42: Design a finite state machine that...

---

Hint: For a similar problem, see this worked example: "Mealy Machine Design Pattern"
```

## Status

✅ **READY FOR TESTING**

The implementation is complete and ready for integration testing with actual learning materials. The feature:
- Maintains backward compatibility
- Handles edge cases (no materials)
- Provides clear pedagogical structure
- Supports bilingual operation
- Includes comprehensive metadata

## Next Steps

To fully utilize this feature:

1. **Add learning materials** to the database using the material extraction pipeline
2. **Link materials to topics** via the `material_topics` table
3. **Link worked examples to exercises** via the `material_exercise_links` table
4. **Test with real course content** to verify formatting and flow
5. **Gather user feedback** on the pedagogical effectiveness

## Files Modified

- `/home/laimk/git/Examina/core/tutor.py` - Main implementation (4 new methods, 2 enhanced methods)
- `/home/laimk/git/Examina/test_tutor_flow.py` - Test script (created)
