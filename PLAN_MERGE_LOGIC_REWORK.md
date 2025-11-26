# Plan: Rework Topic/Procedure Merge Logic

## Problem Statement

The current analysis creates incorrect topic/procedure groupings:
1. **Topic Fragmentation**: Related concepts become separate topics instead of grouping under a parent topic
2. **Over-merging Procedures**: Distinct procedures get merged when they should stay separate
3. **No Context Awareness**: LLM doesn't know existing topics/procedures, creates duplicates

## Core Issues (Generalized)

| Issue Type | Description |
|------------|-------------|
| **Too Granular Topics** | Technique X becomes its own topic instead of being under broader topic Y |
| **Related Concepts Split** | Two related concepts A and B become separate topics when they should share one |
| **Procedures Merged** | Two distinct methods M1 and M2 get merged because they're in same domain |

---

## Solution Architecture

### Phase 1: Topic Granularity Rules (Core Layer)

**Location**: `examina/core/analyzer.py`

**Problem**: LLM creates topics at inconsistent granularity levels

**Solution**: Define topic granularity criteria in prompt (domain-agnostic)

```
TOPIC GRANULARITY RULES:

Topics should be at the CHAPTER/UNIT level of a course, NOT:
- Individual techniques (too specific)
- Entire course domains (too broad)

GRANULARITY TEST:
Ask: "Would this topic have 3-10 related procedures/methods?"
- If YES → Good topic level
- If NO (only 1 procedure) → Too specific, find parent topic
- If NO (50+ procedures) → Too broad, needs splitting

TOPIC NAMING:
- Name should describe a PROBLEM DOMAIN, not a single technique
- Multiple techniques that solve similar problems → ONE topic
- Different problem types → DIFFERENT topics

Example pattern (not hardcoded):
- "Technique for solving X" → Topic should be "X Problems" or "X Analysis"
- "Method A" and "Method B" both for "X" → Same topic, different procedures
```

**Implementation**:
1. Modify `_build_analysis_prompt()` in `analyzer.py`
2. Add granularity test criteria
3. Emphasize topic = problem domain, procedure = solution method

---

### Phase 2: Procedure Distinction (Core Layer)

**Location**: `examina/core/analyzer.py`

**Problem**: Distinct procedures get merged because they share the same topic/domain

**Solution**: Add procedure identity rules based on semantic meaning

```
PROCEDURE IDENTITY RULES:

Two procedures are THE SAME if:
- They solve the EXACT same problem type
- They use the SAME algorithm/method
- Only differ in language or phrasing

Two procedures are DIFFERENT if:
- They solve different problem types (even in same domain)
- They use different algorithms/methods
- One transforms A→B, another transforms B→A
- One designs X, another verifies X

PROCEDURE TYPE CLASSIFICATION:
- design: Create something new (design a circuit, design an automaton)
- transformation: Convert from format A to format B
- verification: Check if something is correct/minimal/valid
- analysis: Calculate metrics, performance, properties
- minimization: Reduce/simplify something

CRITICAL: Procedures of different types should NEVER merge, even if related
- "Design X" ≠ "Verify X" ≠ "Minimize X"
- "A→B Conversion" ≠ "B→A Conversion"
```

**Implementation**:
1. Add procedure type classification in LLM prompt
2. Enforce type-based separation in merge logic
3. Use procedure type + normalized name for matching

---

### Phase 3: Context Injection (Cloud Layer)

**Location**: `examina-cloud/backend/worker/tasks/analyze.py`

**Problem**: LLM doesn't know what topics/procedures already exist → creates duplicates

**Solution**: Pass existing entities to LLM for context-aware analysis

```python
# Before analysis, fetch existing topics and procedures for this course
existing_topics = session.query(Topic).filter(
    Topic.course_code == course_code,
    Topic.user_id == user_id
).all()

existing_procedures = session.query(CoreLoop).filter(
    CoreLoop.user_id == user_id
).all()

# Build context with both names and types
topic_context = [{"name": t.name, "procedure_count": len(t.exercises)} for t in existing_topics]
procedure_context = [{"name": p.name, "type": p.procedure.get("type", "unknown")} for p in existing_procedures]

# Pass to analyzer
result = analyzer.analyze_exercise(
    exercise_text=exercise.text,
    course_name=course_name,
    existing_topics=topic_context,       # NEW
    existing_procedures=procedure_context # NEW
)
```

**Prompt addition** (dynamic, not hardcoded):
```
EXISTING TOPICS in this course:
{dynamically_inserted_list_of_existing_topics}

EXISTING PROCEDURES:
{dynamically_inserted_list_of_existing_procedures_with_types}

CONTEXT RULES:
1. PREFER existing topic if exercise fits semantically
2. PREFER existing procedure if solving same problem with same method
3. Create NEW topic only if exercise doesn't fit ANY existing topic
4. Create NEW procedure only if method is genuinely different
5. When in doubt, USE EXISTING over creating new
```

**Implementation**:
1. Modify `ExerciseAnalyzer.analyze_exercise()` signature to accept context
2. Modify `_build_analysis_prompt()` to dynamically inject context
3. Update cloud layer to fetch and pass context before each analysis batch
4. Cache context during batch analysis (don't re-fetch for each exercise)

---

### Phase 4: Smart Normalization (Cloud Layer)

**Location**: `examina-cloud/backend/worker/tasks/analyze.py`

**Problem**: Minor variations create duplicates (singular vs plural, abbreviations)

**Solution**: Enhanced normalization that handles common variations

```python
def normalize_for_matching(name: str, language: str = "auto") -> str:
    """
    Normalize for matching (case-insensitive, plural-insensitive).

    Handles:
    - Case: "Moore Machine" = "moore machine"
    - Plurals: "Machines" = "Machine" (both EN and IT)
    - Common suffixes: "Design" vs "Designing"
    - Whitespace/punctuation: "A - B" = "A-B" = "A B"
    """
    normalized = name.strip().lower()

    # Normalize whitespace and punctuation
    normalized = re.sub(r'[\s\-_]+', ' ', normalized)
    normalized = re.sub(r'[^\w\s]', '', normalized)

    # Generic plural handling (works for most EN/IT words)
    # Remove trailing 's', 'es', 'i', 'e' when they look like plurals
    if len(normalized) > 4:
        # English plurals
        if normalized.endswith('ies'):
            normalized = normalized[:-3] + 'y'
        elif normalized.endswith('es') and normalized[-3] in 'sxzh':
            normalized = normalized[:-2]
        elif normalized.endswith('s') and normalized[-2] not in 'su':
            normalized = normalized[:-1]
        # Italian plurals
        elif normalized.endswith('i') and normalized[-2] in 'aeiou':
            normalized = normalized[:-1] + 'o'  # circuiti → circuito

    return normalized.strip()


def names_match(name1: str, name2: str) -> bool:
    """Check if two names refer to the same entity."""
    return normalize_for_matching(name1) == normalize_for_matching(name2)
```

**Implementation**:
1. Create generic `normalize_for_matching()` function
2. Keep original name for display in DB
3. Use normalized form only for duplicate detection
4. No hardcoded domain terms - works with any language/domain

---

### Phase 5: Procedure Type Enforcement (Both Layers)

**Problem**: No clear distinction between procedure types in merge logic

**Solution**: Use procedure type as part of identity check

```python
# In cloud layer merge logic
def get_or_create_core_loop(
    session: Session,
    procedure_name: str,
    procedure_type: str,  # NEW: design, transformation, verification, etc.
    procedure_steps: List[str],
    topic_id: uuid.UUID,
    user_id: uuid.UUID,
) -> tuple[CoreLoop, bool]:
    """
    Match requires BOTH:
    1. Normalized name matches
    2. Procedure type matches

    This prevents merging "Design X" with "Verify X"
    """
    normalized = normalize_for_matching(procedure_name)

    core_loop = session.query(CoreLoop).filter(
        func.lower(func.trim(CoreLoop.name)) == normalized,
        CoreLoop.procedure["type"].astext == procedure_type,  # Type must match!
        CoreLoop.user_id == user_id,
    ).first()

    # ... rest of logic
```

**Type-based merge rules**:
```
MERGE ALLOWED (same type + similar name):
- "Design X" (design) = "X Design" (design) ✓

MERGE BLOCKED (different types):
- "Design X" (design) ≠ "Verify X" (verification) ✗
- "A→B Conversion" (transformation) ≠ "B→A Conversion" (transformation) ✗
  (different transformation direction = different procedure)
```

---

## Implementation Order

### Step 1: Core Prompt Enhancement (examina-core)
**File**: `core/analyzer.py`
**Changes**:
- Add topic granularity rules to prompt (chapter-level, not technique-level)
- Add procedure identity rules (when to merge vs separate)
- Add procedure type classification requirement
- Modify `analyze_exercise()` signature to accept existing context

**Complexity**: Medium (prompt changes + signature change)

### Step 2: Context Passing (examina-cloud)
**File**: `backend/worker/tasks/analyze.py`
**Changes**:
- Fetch existing topics/procedures before analysis batch
- Pass context to analyzer
- Enhanced normalization function
- Type-aware procedure matching

**Complexity**: Medium (DB queries + logic changes)

### Step 3: Testing
**Approach**:
- Create test exercises covering edge cases:
  - Two techniques that should share a topic
  - Two procedures that should stay separate
  - Naming variations (plural, language)
- Verify grouping logic works generically (not just for specific domains)

### Step 4: Optional Enhancements
- Similarity threshold for fuzzy matching (configurable)
- Manual merge/split UI in frontend
- Audit log for merge decisions (debugging)

---

## Decision Points for Discussion

### Q1: Topic Granularity Target

**Option A**: Chapter-level (recommended)
- ~5-10 topics per course
- Many procedures per topic (3-10)
- Topics = what you'd see in course syllabus

**Option B**: Section-level
- ~15-25 topics per course
- Fewer procedures per topic (1-5)
- More granular organization

### Q2: Procedure Matching Strategy

**Option A**: Type + Normalized Name (recommended)
- Safe, predictable
- LLM controls initial assignment
- No semantic guessing

**Option B**: Semantic Similarity
- More aggressive deduplication
- Risk of wrong merges
- Needs tuning per domain

### Q3: Handling Existing Data

**Option A**: Forward-only (recommended initially)
- New logic applies to new analyses
- Existing data unchanged
- Lower risk

**Option B**: Migration
- Run cleanup script on existing data
- More work, higher risk
- Better long-term data quality

---

## Success Criteria (Generic)

After implementation, these patterns should work:

| Input Pattern | Expected Behavior |
|---------------|-------------------|
| Two exercises using same technique | Same topic, same procedure |
| Two exercises in same domain, different techniques | Same topic, different procedures |
| Two exercises using techniques that are inverses | Same topic, different procedures (A→B ≠ B→A) |
| Second exercise where topic already exists | Reuses existing topic |
| Naming variation (plural/singular) | Matches existing, no duplicate |
| Design vs Verify vs Minimize same thing | Three separate procedures |

The system should work **without any domain-specific configuration**.

---

## Phase 6: Concept Handling (Theory Questions)

**Problem**: Not all exercises have procedures - some are purely theoretical (definitions, explanations, proofs)

**Current State**: `AnalysisResult` already has:
- `exercise_type`: procedural | theory | proof | hybrid
- `theory_category`: definition | theorem | axiom | property | explanation | derivation | concept
- `concept_id`, `prerequisite_concepts`

**Solution**: Apply same merge logic to Concepts as to Procedures

### Concept Identity Rules

```
Two concepts are THE SAME if:
- They refer to the same theoretical entity (definition, theorem, property)
- Same concept_id after normalization
- Same theory_category

Two concepts are DIFFERENT if:
- Different theoretical entities (even if related)
- Different theory_category (definition of X ≠ proof of X)
- Different prerequisite chains

EXAMPLES:
- "Definition of X" = "Definizione di X" (same concept, different language) ✓
- "Definition of X" ≠ "Theorem about X" (different category) ✗
- "Theorem A" ≠ "Corollary of A" (different concepts) ✗
```

### Concept-Topic Relationship

```
TOPIC GRANULARITY for theory:
- Topic should still be chapter/unit level
- Multiple related concepts belong under ONE topic
- Example: "Eigenvalue Definition", "Eigenvalue Properties", "Diagonalization Theorem"
  → All under topic "Eigenvalues and Diagonalization"

CONCEPT vs PROCEDURE:
- Procedural exercise: topic + procedure(s)
- Theory exercise: topic + concept(s)
- Hybrid exercise: topic + procedure(s) + concept(s)
```

### Implementation

```python
def get_or_create_concept(
    session: Session,
    concept_name: str,
    concept_type: str,  # definition, theorem, property, etc.
    content: dict,
    prerequisites: List[str],
    topic_id: uuid.UUID,
    user_id: uuid.UUID,
) -> tuple[Concept, bool]:
    """
    Match requires BOTH:
    1. Normalized name matches
    2. Concept type matches
    """
    normalized = normalize_for_matching(concept_name)

    concept = session.query(Concept).filter(
        func.lower(func.trim(Concept.name)) == normalized,
        Concept.concept_type == concept_type_enum,  # Type must match!
        Concept.topic_id == topic_id,
        Concept.user_id == user_id,
    ).first()

    # ... rest of logic
```

### Context for Theory Questions

When passing context to LLM, include existing concepts:

```python
# Also fetch existing concepts for context
existing_concepts = session.query(Concept).filter(
    Concept.user_id == user_id
).all()

concept_context = [
    {"name": c.name, "type": c.concept_type.value, "topic": c.topic.name}
    for c in existing_concepts
]
```

**Prompt addition**:
```
EXISTING CONCEPTS in this course:
{dynamically_inserted_list_of_concepts_with_types}

For THEORY questions:
- Match existing concept if asking about same theoretical entity
- Create new concept only if genuinely new theoretical entity
- Same topic can have multiple concepts (definition, properties, theorems about it)
```

---

## Updated Success Criteria

| Exercise Type | Expected Behavior |
|---------------|-------------------|
| Procedural (same technique) | Same topic, same procedure |
| Procedural (different techniques, same domain) | Same topic, different procedures |
| Theory (definition of X) | Topic + Concept (type=definition) |
| Theory (theorem about X) | Same topic + different Concept (type=theorem) |
| Theory (explain why X) | Same topic + Concept (type=explanation) |
| Proof exercise | Topic + Concept (type=proof) |
| Hybrid (compute X + explain why) | Topic + Procedure + Concept |
| Two definitions in same area | Same topic, both concepts preserved |

The system handles **both procedural and theoretical content** uniformly.
