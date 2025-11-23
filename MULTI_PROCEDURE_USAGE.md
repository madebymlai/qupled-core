# Multi-Procedure Exercise Support - User Guide

## Overview

Examina now supports exercises with multiple procedures (e.g., design + transformation + minimization). This allows the system to accurately represent complex exam questions that require students to perform multiple related tasks.

### What are Multi-Procedure Exercises?

Many exam exercises consist of multiple steps, each requiring a different type of problem-solving approach. For example:

**Example Exercise:**
> 1. Design a Mealy machine for X
> 2. Transform the Mealy machine to an equivalent Moore machine
> 3. Minimize the resulting Moore machine

This single exercise covers **three distinct procedures**:
- **Design** (Mealy Machine Design)
- **Transformation** (Mealy→Moore Transformation)
- **Minimization** (FSM Minimization)

Phase 6 of Examina extracts and tracks all procedures per exercise, enabling:
- More accurate search and filtering
- Better quiz customization (practice specific procedure types)
- Improved progress tracking (mastery per procedure type)

---

## CLI Usage

### 1. View Course Information

The `info` command displays all exercises and their procedures:

```bash
examina info --course ADE
```

**Output:**
```
Architectural Design of Embedded Systems
Architetture di Elaboratori Digitali

Code: B006802
Acronym: ADE
Level: Bachelor (L-31)

Status:
  Topics discovered: 14
  Exercises ingested: 27

Multi-Procedure Exercises:
  5/27 exercises cover multiple procedures

Top Examples:
  • Exercise 2024-01-29_1: 3 procedures
    - Mealy Machine Design
    - Mealy→Moore Transformation (point 2)
    - FSM Minimization (point 3)
```

---

### 2. Search Exercises by Tag

The new `search` command allows filtering exercises by procedure tags:

```bash
# Find all transformation exercises
examina search --course ADE --tag transformation

# Find all design exercises
examina search --course ADE --tag design

# Find specific transformation types
examina search --course ADE --tag transform_mealy_to_moore
```

**Output:**
```
Search Results
Course: Architectural Design of Embedded Systems (ADE)
Filter: tag 'transformation'

Found 8 exercise(s):

Exercise: 2024-01-29_1
  Procedures (3):
    1. Mealy Machine Design
    2. Mealy→Moore Transformation (point 2)
    3. FSM Minimization (point 3)
  Tags: design, transformation, minimization, transform_mealy_to_moore
  Difficulty: medium
  Source: 2024-01-29.pdf (page 1)
```

**Available Tags:**
- `design` - Design exercises (create from scratch)
- `transformation` - Format conversion exercises
- `minimization` - State/logic minimization exercises
- `verification` - Correctness verification exercises
- `analysis` - Analysis and evaluation exercises
- `implementation` - Implementation exercises
- `transform_X_to_Y` - Specific transformation types (e.g., `transform_mealy_to_moore`)

---

### 3. Search by Text Content

Search for specific keywords in exercise text:

```bash
# Find exercises mentioning "garage door"
examina search --course ADE --text "garage door"

# Find exercises from a specific exam
examina search --course ADE --text "2024-01-29"
```

---

### 4. Find Multi-Procedure Exercises Only

Filter to show only exercises with 2+ procedures:

```bash
examina search --course ADE --multi-only

# Combine with other filters
examina search --course ADE --tag transformation --multi-only
```

---

### 5. Quiz Filtering by Procedure

The `quiz` command now supports filtering by procedure type:

```bash
# Practice only transformation exercises
examina quiz --course ADE --procedure transformation --questions 5

# Practice only design exercises
examina quiz --course ADE --procedure design --questions 10

# Combine with difficulty
examina quiz --course ADE --procedure minimization --difficulty hard --questions 3
```

**Available Procedure Types:**
- `design`
- `transformation`
- `verification`
- `minimization`
- `analysis`
- `implementation`

---

## Database API

If you're building features on top of Examina, you can use these database methods:

### Get All Procedures for an Exercise

```python
from storage.database import Database

with Database() as db:
    # Get all core loops for an exercise
    core_loops = db.get_exercise_core_loops(exercise_id)

    for cl in core_loops:
        print(f"{cl['name']} - Step {cl['step_number']}")
```

**Returns:**
```python
[
    {
        'id': 'mealy_machine_design_xyz',
        'name': 'Mealy Machine Design',
        'step_number': 1,
        'procedure': ['Identify inputs/outputs', 'Define states', ...]
    },
    {
        'id': 'mealy_to_moore_transformation_abc',
        'name': 'Mealy→Moore Transformation',
        'step_number': 2,
        'procedure': ['Create state table', 'Map transitions', ...]
    }
]
```

---

### Search Exercises by Tag

```python
with Database() as db:
    # Find all transformation exercises
    exercises = db.get_exercises_by_tag('ADE', 'transformation')

    # Find specific transformation type
    exercises = db.get_exercises_by_tag('ADE', 'transform_mealy_to_moore')
```

---

### Search Exercises by Text

```python
with Database() as db:
    # Search by text content
    exercises = db.search_exercises_by_text('ADE', 'garage door')
```

---

### Get Multi-Procedure Exercises

```python
with Database() as db:
    # Get exercises with 2+ procedures
    multi_exercises = db.get_exercises_with_multiple_procedures('ADE')

    for ex in multi_exercises:
        print(f"{ex['exercise_number']}: {ex['core_loop_count']} procedures")
```

---

### Filter by Procedure Type

```python
with Database() as db:
    # Get all design exercises
    design_exercises = db.get_exercises_by_procedure_type('ADE', 'design')

    # Get all transformation exercises
    transform_exercises = db.get_exercises_by_procedure_type('ADE', 'transformation')
```

---

## Tag System

### Automatic Tag Generation

When exercises are analyzed, the system automatically generates tags based on:

1. **Procedure Type** - The general category (design, transformation, etc.)
2. **Transformation Details** - Specific transformation types (e.g., `transform_mealy_to_moore`)

### Tag Format

Tags are stored as JSON arrays in the `exercises.tags` column:

```json
["design", "transformation", "minimization", "transform_mealy_to_moore"]
```

### Custom Tags

You can add custom tags via the database:

```python
with Database() as db:
    # Add custom tags
    db.update_exercise_tags(exercise_id, [
        'design',
        'transformation',
        'advanced',
        'exam_2024'
    ])
```

---

## Examples

### Example 1: Find All Mealy→Moore Transformation Exercises

```bash
examina search --course ADE --tag transform_mealy_to_moore
```

### Example 2: Practice Only Multi-Step Exercises

```bash
examina quiz --course ADE --multi-only --questions 5
```

### Example 3: Search for FSM Minimization in Hard Exercises

```bash
examina search --course ADE --tag minimization --text "minimize"
```

### Example 4: Get Exercises from Specific Exam

```bash
examina search --course ADE --text "2024-01-29"
```

---

## Backend: Junction Table Architecture

### Database Schema

Multi-procedure support uses a many-to-many relationship:

**Tables:**
- `exercises` - Exercise data
- `core_loops` - Procedure definitions
- `exercise_core_loops` (junction table) - Links exercises to multiple procedures

**Junction Table Schema:**
```sql
CREATE TABLE exercise_core_loops (
    exercise_id TEXT NOT NULL,
    core_loop_id TEXT NOT NULL,
    step_number INTEGER,  -- Which point in the exercise (1, 2, 3, etc.)
    PRIMARY KEY (exercise_id, core_loop_id)
)
```

### Backward Compatibility

The legacy `exercises.core_loop_id` column is retained for backward compatibility. New code should use the junction table via `get_exercise_core_loops()`.

---

## Validation

To validate that multi-procedure extraction is working correctly, run:

```bash
python validate_multi_procedure.py
```

This script:
- Counts multi-procedure exercises
- Shows procedure type distribution
- Validates specific exercises (e.g., 2024-01-29 #1)
- Tests search functionality
- Checks database consistency

---

## Known Limitations

1. **Requires Re-Analysis** - Existing exercises need to be re-analyzed to populate multi-procedure data
2. **Language-Specific** - Detection works best with English and Italian keywords
3. **Fuzzy Matching** - Some edge cases may require manual tag correction

---

## Future Enhancements

- **Semantic Search** - Use embeddings to find similar procedures
- **Dependency Tracking** - Link procedures that depend on each other
- **Procedure Templates** - Pre-defined procedure templates for common patterns
- **Visual Procedure Maps** - Show relationships between procedures in exercises

---

## Support

For issues or questions:
1. Check the validation script output
2. Review the analyzer logs
3. Inspect the database with `examina info --course <CODE>`
4. Search for similar exercises with `examina search`

---

**Last Updated:** Phase 6.5 - Multi-Procedure Search & Validation
