# Plan: Adaptive Teaching Based on Mastery Level

## Goal
Transform Examina from "exercises with spaced repetition" into a true intelligent tutoring system where content, depth, and difficulty adapt to each student's mastery level in real-time.

## Current State
- SM-2 algorithm tracks mastery at exercise/core_loop level
- `adaptive_teaching.py` has useful methods (get_recommended_depth, detect_knowledge_gaps, get_personalized_learning_path) but they're **mostly unused**
- Tutor makes one-time depth decision at session start, no dynamic adaptation
- Quiz selects exercises randomly, ignoring mastery distribution

## Implementation Phases

### Phase 1: Mastery Aggregation Foundation
**Goal:** Create proper hierarchy with cascade updates

**Tasks:**
1. Create `core/mastery_aggregator.py`:
   - `aggregate_core_loop_mastery(course_code, core_loop_id)` - from exercises
   - `aggregate_topic_mastery(course_code, topic_id)` - from core loops
   - `aggregate_course_mastery(course_code)` - from topics
   - `update_mastery_cascade(exercise_id)` - trigger after quiz attempt

2. Add database queries to `storage/database.py`:
   - `get_exercises_by_mastery_level(course_code, level)` - bucket query
   - `get_mastery_distribution(course_code)` - stats for dashboard
   - `get_weak_core_loops(course_code, threshold=0.5)` - for gap detection

3. Update `core/quiz_engine.py`:
   - Call `update_mastery_cascade()` after each attempt
   - Use full SM-2 algorithm (not simplified intervals)

**Files:** `core/mastery_aggregator.py` (new), `storage/database.py`, `core/quiz_engine.py`

---

### Phase 2: Mastery-Based Content Selection
**Goal:** Select quiz exercises based on mastery, not random

**Tasks:**
1. Add mastery-aware selection to `QuizEngine._select_exercises()`:
   ```
   Distribution for balanced quiz:
   - 40% weak (mastery < 0.4) - reinforce gaps
   - 40% learning (0.4-0.7) - build proficiency
   - 20% strong (> 0.7) - maintain mastery
   ```

2. Add `--adaptive` flag to quiz command:
   - Without flag: current random behavior
   - With flag: mastery-based selection

3. Implement difficulty progression within quiz:
   - Start with medium difficulty
   - If 2 correct in a row → increase difficulty
   - If 2 wrong in a row → decrease difficulty

**Files:** `core/quiz_engine.py`, `cli.py`

---

### Phase 3: Dynamic Adaptation During Learning
**Goal:** Adjust tutor behavior based on real-time performance

**Tasks:**
1. Track performance during quiz session:
   - Accumulate correct/wrong ratio
   - Track time per question
   - Detect struggling (long time + wrong answer)

2. Update `core/tutor.py` practice mode:
   - After each wrong answer, offer simpler explanation
   - After 3 correct, offer to skip ahead
   - Show mastery progress bar

3. Add real-time feedback:
   - "You're at 45% mastery on this topic"
   - "Great progress! Mastery improved to 62%"
   - "Consider reviewing prerequisites first"

4. Integrate `get_personalized_learning_path()`:
   - Show recommended next steps after quiz
   - Suggest what to learn next based on gaps

**Files:** `core/tutor.py`, `core/quiz_engine.py`, `core/adaptive_teaching.py`

---

### Phase 4: Prerequisite Integration
**Goal:** Respect and enforce prerequisite mastery

**Tasks:**
1. Connect prerequisites to mastery:
   - Check prerequisite mastery before teaching advanced topics
   - Warn if prerequisites are weak

2. Modify content selection:
   - Filter out advanced exercises if prerequisites < 0.5 mastery
   - Recommend prerequisite practice when detected

3. Add to `learn` command:
   - "Warning: Your mastery of [X] is low. Learn that first?"
   - `--force` flag to skip prerequisite check

**Files:** `core/adaptive_teaching.py`, `core/tutor.py`, `cli.py`

---

## File Changes Summary

| File | Changes |
|------|---------|
| `core/mastery_aggregator.py` | NEW - Hierarchy aggregation + cascade updates |
| `storage/database.py` | Add mastery bucket queries (3 methods) |
| `core/quiz_engine.py` | Mastery-based selection + cascade trigger |
| `core/tutor.py` | Dynamic adaptation + real-time feedback |
| `core/adaptive_teaching.py` | Integrate existing methods into flow |
| `cli.py` | Add --adaptive flag, show mastery feedback |

## Success Criteria

1. **Phase 1 Complete:** Mastery updates cascade automatically after each quiz attempt
2. **Phase 2 Complete:** Quiz exercises selected by mastery distribution, not random
3. **Phase 3 Complete:** Tutor adjusts depth mid-session based on performance
4. **Phase 4 Complete:** Prerequisites checked and enforced before advanced content

## Estimated Effort

| Phase | Effort | Priority |
|-------|--------|----------|
| Phase 1 | 2-3 hours | HIGH (foundation) |
| Phase 2 | 2-3 hours | HIGH (core feature) |
| Phase 3 | 3-4 hours | MEDIUM (polish) |
| Phase 4 | 2-3 hours | MEDIUM (enhancement) |

Total: ~10-13 hours of implementation

## Start Point

Begin with Phase 1 → Create `core/mastery_aggregator.py` with cascade update logic.
