# Multi-Procedure Search - Quick Reference

## Common Commands

### 1. Search by Procedure Type

```bash
# Find all transformation exercises
examina search --course ADE --tag transformation

# Find all design exercises
examina search --course ADE --tag design

# Find all minimization exercises
examina search --course ADE --tag minimization

# Find all analysis exercises
examina search --course ADE --tag analysis

# Find all verification exercises
examina search --course ADE --tag verification
```

---

### 2. Search by Specific Transformation

```bash
# Find Mealy→Moore transformations
examina search --course ADE --tag transform_macchina_di_mealy_to_macchina_di_moore

# Find Moore→Mealy transformations
examina search --course ADE --tag transform_automa_di_moore_to_automa_di_mealy

# Find binary conversion exercises
examina search --course ADE --tag transform_binario
```

---

### 3. Search by Text Content

```bash
# Find exercises mentioning "Mealy"
examina search --course ADE --text "Mealy"

# Find exercises from specific exam
examina search --course ADE --text "2024-01-29"

# Find exercises about FSM
examina search --course ADE --text "finite state"
```

---

### 4. Find Multi-Procedure Exercises

```bash
# All multi-procedure exercises
examina search --course ADE --multi-only

# Multi-procedure transformations only
examina search --course ADE --tag transformation --multi-only

# Multi-procedure design exercises
examina search --course ADE --tag design --multi-only
```

---

### 5. Quiz with Procedure Filtering

```bash
# Practice only transformations
examina quiz --course ADE --procedure transformation --questions 5

# Practice only design exercises
examina quiz --course ADE --procedure design --questions 10

# Hard minimization exercises
examina quiz --course ADE --procedure minimization --difficulty hard --questions 3
```

---

### 6. View Course Information

```bash
# See all multi-procedure exercises
examina info --course ADE

# Will show:
# - Total exercise count
# - Multi-procedure exercise count
# - Top examples with procedure lists
```

---

### 7. Validate Multi-Procedure Extraction

```bash
# Run comprehensive validation
python validate_multi_procedure.py

# Shows:
# - Summary statistics
# - Procedure type distribution
# - Transformation types
# - Top examples
# - Search functionality tests
```

---

## Python API Quick Reference

### Get All Procedures for an Exercise

```python
from storage.database import Database

with Database() as db:
    core_loops = db.get_exercise_core_loops(exercise_id)

    for cl in core_loops:
        print(f"{cl['name']} - Step {cl['step_number']}")
```

---

### Search by Tag

```python
with Database() as db:
    # All transformation exercises
    exercises = db.get_exercises_by_tag('ADE', 'transformation')

    # Specific transformation type
    exercises = db.get_exercises_by_tag('ADE', 'transform_mealy_to_moore')
```

---

### Search by Text

```python
with Database() as db:
    exercises = db.search_exercises_by_text('ADE', 'garage door')
```

---

### Get Multi-Procedure Exercises

```python
with Database() as db:
    multi_exercises = db.get_exercises_with_multiple_procedures('ADE')

    for ex in multi_exercises:
        print(f"{ex['exercise_number']}: {ex['core_loop_count']} procedures")
```

---

### Filter by Procedure Type

```python
with Database() as db:
    design_exercises = db.get_exercises_by_procedure_type('ADE', 'design')
```

---

## Available Tags

### Procedure Types
- `design` - Create from scratch
- `transformation` - Convert between formats
- `minimization` - Reduce states/logic
- `verification` - Check correctness
- `analysis` - Evaluate performance
- `implementation` - Code implementation

### Specific Transformations (Examples)
- `transform_macchina_di_mealy_to_macchina_di_moore`
- `transform_automa_di_moore_to_automa_di_mealy`
- `transform_binario_puro_to_decimale`
- `transform_forma_sop_to_circuito_logico`
- `transform_numero_decimale_to_ieee754_binario`

---

## Common Use Cases

### Case 1: Find all exercises for a specific procedure type
```bash
examina search --course ADE --tag transformation --limit 10
```

### Case 2: Practice a specific transformation
```bash
examina quiz --course ADE --procedure transformation --questions 5
```

### Case 3: Find complex multi-step exercises
```bash
examina search --course ADE --multi-only --limit 5
```

### Case 4: Find exercises from a specific exam
```bash
examina search --course ADE --text "2024-01-29"
```

### Case 5: Validate procedure extraction
```bash
python validate_multi_procedure.py
```

---

## Output Format

### Search Results
```
Exercise: <number>
  Procedures (<count>):
    1. <procedure_name> (point <step>)
    2. <procedure_name> (point <step>)
    ...
  Tags: <tag1>, <tag2>, ...
  Difficulty: <level>
  Source: <pdf_name> (page <number>)
```

### Info Command
```
Multi-Procedure Exercises:
  <count>/<total> exercises cover multiple procedures

Top Examples:
  • Exercise <number>: <count> procedures
    - <procedure_name> (point <step>)
    - <procedure_name> (point <step>)
    ... and <n> more
```

---

## Performance Tips

1. **Limit results** - Use `--limit N` for faster searches
2. **Specific tags** - More specific tags = faster results
3. **Index benefits** - Tag searches are faster than text searches
4. **Cache hits** - Repeated searches benefit from SQLite caching

---

## Troubleshooting

### No results found?
- Check tag spelling (case-sensitive)
- Try broader search (e.g., just "transformation" instead of full tag)
- Use `--text` instead of `--tag` for fuzzy matching
- Check if exercises are analyzed (`examina info --course <CODE>`)

### Wrong exercises returned?
- Use more specific tags (e.g., `transform_mealy_to_moore` instead of just `transformation`)
- Combine multiple filters (`--tag design --multi-only`)
- Use text search for exact phrases (`--text "exact phrase"`)

### Performance slow?
- Add `--limit` to restrict results
- Use tag search instead of text search when possible
- Run `examina deduplicate` to clean up duplicate topics/loops

---

**Last Updated:** Phase 6.5 - Multi-Procedure Search & Validation
