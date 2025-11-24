# Examina - TODO List

## Phase 3 - AI Analysis ✅ COMPLETED

**Done:**
- ✅ Intelligent splitter (filters instructions, works for all formats)
- ✅ AI analysis with Groq
- ✅ Rate limit handling with exponential retry
- ✅ Database + Vector store
- ✅ Topic and core loop discovery

**Future improvements (low priority):**
- [x] Topic/core loop deduplication - Database-aware similarity matching + cleanup command
- [x] Confidence threshold filtering - Filter low-confidence analyses (default 0.5)
- [x] Resume failed analysis - Checkpoint system with --force flag
- [x] Batch processing optimization - 7-8x speedup with parallel processing
- [x] Caching LLM responses - File-based cache with TTL, 100% hit rate on re-runs
- [x] Deduplicate command - Merge existing duplicates with `examina deduplicate`
- [ ] Language-aware deduplication - Match "Finite State Machines" with "Macchine a Stati Finiti"
- [ ] Translation dictionary for English/Italian topic matching
- [ ] Semantic similarity instead of string similarity (use embeddings)
- [ ] Provider-agnostic rate limiting tracker

## Phase 4 - Tutor Features ✅ COMPLETED

**Done:**
- ✅ **Add Anthropic Claude Sonnet 4.5** - Better rate limits, higher quality (14 topics, 23 core loops found!)
- ✅ **Analyze with Anthropic** - Successfully analyzed all 27 ADE exercises including SR Latch
- ✅ **Language switch (Italian/English)** - Added `--lang` flag to all commands (analyze, learn, practice, generate)
- ✅ **Tutor class** - Created core/tutor.py with learning, practice, and generation features
- ✅ **Learn command** - Explains core loops with theory, procedure, examples, and tips
- ✅ **Practice command** - Interactive practice with AI feedback and hints
- ✅ **Generate command** - Creates new exercise variations based on examples

**Tested:**
- All commands work with both English and Italian
- Learn: Generated comprehensive Moore machine tutorial
- Generate: Created new garage door control exercise
- Practice: Interactive answer evaluation with helpful feedback

## Phase 5 - Quiz System ✅ COMPLETED

**Implemented using parallel agents in ~4 hours total execution time!**

### 5.1 Database Schema ✅
- ✅ Added `quiz_sessions` table - Session metadata and scores
- ✅ Added `quiz_attempts` table - Individual question attempts
- ✅ Added `exercise_reviews` table - SM-2 spaced repetition data
- ✅ Added `topic_mastery` table - Aggregated mastery per topic
- ✅ Implemented database migrations with backward compatibility
- ✅ Added 5 performance indexes for fast queries
- ✅ Added 19 helper methods to Database class

### 5.2 SM-2 Algorithm ✅
- ✅ Created `core/sm2.py` with SM-2 implementation (572 lines)
- ✅ Implemented quality scoring (0-5 based on correctness, speed, hints)
- ✅ Implemented interval calculation (1d → 6d → exponential with EF)
- ✅ Implemented easiness factor adjustment (1.3-2.5 range)
- ✅ Implemented mastery level progression (new → learning → reviewing → mastered)
- ✅ Added comprehensive documentation and examples
- ✅ Created 33 unit tests (100% pass rate)

### 5.3 Quiz Session Management ✅
- ✅ Created `core/quiz.py` with QuizManager (620 lines)
- ✅ Created `core/quiz_engine.py` with QuizEngine class
- ✅ Implemented `create_quiz()` - Supports random, topic, core_loop, review types
- ✅ Implemented smart question selection with prioritization
- ✅ Implemented `submit_answer()` - AI evaluation + SM-2 update
- ✅ Implemented `complete_quiz()` - Final scoring and mastery updates
- ✅ Full integration with Tutor class for AI feedback

### 5.4 CLI Commands ✅
- ✅ Implemented `examina quiz` with all filters (topic, difficulty, core_loop)
- ✅ Added `--review-only` flag for spaced repetition mode
- ✅ Added `--questions N` flag for custom quiz length
- ✅ Interactive quiz flow with Rich UI (panels, colors, spinners)
- ✅ Multi-line answer input (double Enter to submit)
- ✅ Implemented `examina progress` with breakdowns
- ✅ Implemented `examina suggest` for study recommendations
- ✅ Full `--lang` support (Italian/English) for all quiz commands

### 5.5 Progress Tracking & Analytics ✅
- ✅ Created `core/analytics.py` with ProgressAnalytics class
- ✅ Implemented `get_course_summary()` - Overall progress stats
- ✅ Implemented `get_topic_breakdown()` - Per-topic mastery
- ✅ Implemented `get_weak_areas()` - Identify struggling topics
- ✅ Implemented `get_due_reviews()` - SM-2 scheduled reviews
- ✅ Implemented `get_study_suggestions()` - Personalized recommendations
- ✅ Beautiful Rich visualizations (progress bars, tables, color-coded status)

### 5.6 Testing ✅
- ✅ 33 unit tests for SM-2 algorithm (100% pass)
- ✅ Test suite for QuizManager
- ✅ Integration test capabilities
- ✅ Demo scripts (demo_sm2.py) for verification

### Documentation ✅
- ✅ Created `QUIZ_API_REFERENCE.md` - Complete API reference
- ✅ Created `QUIZ_MANAGER_README.md` - Implementation guide
- ✅ Updated `README.md` with Phase 5 features
- ✅ Kept `PHASE_5_PLAN.md` for reference

**Achievement: Completed in ~4 hours using 4 parallel agents (vs. estimated 35-45 hours)**
**Performance gain: 9-11x faster than sequential implementation!**

## Phase 6 - Multi-Core-Loop Support 🚧 IN PROGRESS

**Goal:** Extract ALL procedures from multi-step exercises (e.g., "1. Design Mealy, 2. Transform to Moore, 3. Minimize")

**Implemented using 3 parallel agents:**

### 6.1 Database Schema ✅ COMPLETED
- ✅ Created `exercise_core_loops` junction table (many-to-many relationship)
- ✅ Added `tags` column to exercises for flexible search
- ✅ Implemented automatic migration from legacy `core_loop_id` column
- ✅ Added 4 new Database helper methods:
  - `link_exercise_to_core_loop()` - Link exercise to multiple core loops
  - `get_exercise_core_loops()` - Get all core loops for an exercise
  - `get_exercises_by_core_loop()` - Updated to use junction table
  - `get_exercises_with_multiple_procedures()` - Find multi-step exercises
- ✅ Full backward compatibility with existing code

### 6.2 Detection Logic ✅ COMPLETED
- ✅ Created `core/detection_utils.py` (455 lines)
- ✅ Implemented `detect_numbered_points()` - 8 pattern types (numeric, letters, Roman, Italian)
- ✅ Implemented `detect_transformation_keywords()` - 15 transformation patterns (English + Italian)
- ✅ Implemented `classify_procedure_type()` - 6 categories (design, transformation, verification, minimization, analysis, implementation)
- ✅ Bilingual support (English/Italian)
- ✅ Confidence scoring for fuzzy matches
- ✅ 33 unit tests (100% pass rate)

### 6.3 Analyzer Updates ✅ COMPLETED
- ✅ Updated LLM prompt to extract ALL procedures per exercise
- ✅ Added multi-procedure JSON response format
- ✅ Implemented procedure-specific transformation detection
- ✅ Added automatic tag generation (e.g., `transform_mealy_to_moore`)
- ✅ Updated result processing to link exercises to multiple core loops
- ✅ Full backward compatibility with old single-procedure format
- ✅ Test suite with 5 test cases (100% pass)

### 6.4 Quiz & Search Updates 🔜 TODO
- [ ] Update quiz selection queries to use junction table
- [ ] Add `--procedure` filter flag to quiz command
- [ ] Update `info` command to show all procedures per exercise
- [ ] Add search by tags functionality

### 6.5 Testing & Validation 🔜 TODO
- [ ] Re-analyze B006802 (ADE) exercises with multi-procedure extraction
- [ ] Verify Exercise 1 from 2024-01-29 now maps to "Mealy→Moore Transformation"
- [ ] Test quiz filtering by procedure type
- [ ] Validate tag-based search

### 6.6 Documentation ✅ COMPLETED
- ✅ Created `MULTI_PROCEDURE_IMPLEMENTATION_SUMMARY.md`
- ✅ Created `MULTI_PROCEDURE_ARCHITECTURE.md`
- ✅ Created example LLM responses (`example_multi_procedure_llm_response.json`)
- [ ] Update README.md with Phase 6 features

**Status:** Core implementation complete (6.1-6.3), CLI/testing remaining (6.4-6.5)

## Phase 7 - Enhanced Learning System 🚧 IN PROGRESS

**Problem:** Current `learn` command assumes prior knowledge and doesn't deeply explain WHY and HOW.

**Goal:** Create a comprehensive teaching system that:
- Explains foundational concepts (doesn't assume prior knowledge)
- Provides detailed reasoning for each algorithm step
- Teaches metacognitive strategies (how to learn effectively)
- Adapts explanations to student's understanding level

### 7.1 Deep Theory Explanations ✅ COMPLETED
- [x] Created `core/concept_explainer.py` module
- [x] Added prerequisite concepts detection (CONCEPT_HIERARCHY mapping)
- [x] Implemented concept explanations with examples, analogies, and misconceptions
- [x] Added foundational theory before procedures
- [x] Implemented progressive complexity (basic/medium/advanced depth levels)
- [x] Enhanced Tutor.learn() to include prerequisite concepts
- [x] Updated CLI with --depth and --no-concepts flags

**Features:**
- Prerequisite concept system for FSM, Linear Algebra, and Concurrent Programming topics
- Detailed concept explanations with analogies and common misconceptions
- Three depth levels: basic (concise), medium (balanced), advanced (comprehensive)
- Automatic prerequisite injection before procedural explanations

### 7.2 Step-by-Step Reasoning ✅ PARTIALLY COMPLETE
- [x] Enhanced procedure explanations with WHY for each step
- [x] Added common mistakes and how to avoid them
- [x] Included decision-making logic ("when to use this technique")
- [x] Created 5-section explanation structure (Big Picture, Step-by-Step, Pitfalls, Decision-Making, Practice Strategy)
- [ ] Provide worked examples with detailed reasoning (LLM-generated per request)
- [ ] Interactive questioning to check understanding (future enhancement)

### 7.3 Metacognitive Learning Strategies 🔜 TODO
- [ ] Create study strategies module
- [ ] Add learning tips per topic/difficulty
- [ ] Teach problem-solving frameworks
- [ ] Include self-assessment prompts
- [ ] Provide retrieval practice suggestions

### 7.4 Adaptive Teaching 🔜 TODO
- [ ] Track student's understanding level per topic
- [ ] Adjust explanation depth based on mastery
- [ ] Detect knowledge gaps and fill them proactively
- [ ] Recommend personalized learning paths

## Phase 8 - Automatic Topic Splitting ✅ COMPLETED

**Problem:** LLM analysis sometimes creates generic topics (e.g., "Algebra Lineare") with too many diverse core loops (30+), making them difficult to study effectively.

**Solution:** Fully LLM-driven post-processing to automatically detect and split generic topics into specific, focused subtopics.

### 8.1 Configuration ✅ COMPLETED
- ✅ Added `GENERIC_TOPIC_THRESHOLD` setting (default: 10 core loops)
- ✅ Added `TOPIC_CLUSTER_MIN/MAX` settings (4-6 subtopics)
- ✅ Added `TOPIC_SPLITTING_ENABLED` toggle
- ✅ Environment variable support for customization

### 8.2 Database Methods ✅ COMPLETED
- ✅ Implemented `split_topic()` in database.py
- ✅ Transaction-safe with automatic rollback on failure
- ✅ Preserves all exercise-core_loop relationships
- ✅ Optional deletion of empty original topics
- ✅ Returns detailed statistics (topics created, loops moved, errors)

### 8.3 Detection & Clustering ✅ COMPLETED
- ✅ Implemented `detect_generic_topics()` in analyzer.py
- ✅ Detection criteria: >10 core loops OR topic name matches course name
- ✅ Implemented `cluster_core_loops_for_topic()` for LLM-based clustering
- ✅ Semantic similarity grouping of core loops
- ✅ Full validation (ensures all core loops assigned exactly once)
- ✅ Cluster count validation (4-6 subtopics)

### 8.4 CLI Command ✅ COMPLETED
- ✅ Created `split-topics` command
- ✅ Added `--dry-run` flag for preview without changes
- ✅ Added `--force` flag to skip confirmation prompts
- ✅ Added `--delete-old` flag to remove empty original topics
- ✅ Multi-language support (Italian/English)
- ✅ Rich UI with detailed progress and statistics

### 8.5 Testing & Validation ✅ COMPLETED
- ✅ Successfully tested on AL course (B006807)
- ✅ Split "Algebra Lineare" (30 core loops) into 6 focused topics:
  - Sottospazi Vettoriali e Basi (10 loops)
  - Applicazioni Lineari e Trasformazioni (6 loops)
  - Diagonalizzazione e Autovalori (5 loops)
  - Cambi di Base e Basi Ortonormali (3 loops)
  - Matrici Parametriche e Determinanti (3 loops)
  - Teoria e Problemi Integrati (3 loops)
- ✅ All 30 core loops preserved and properly categorized
- ✅ No data loss or broken relationships

**Features:**
- No hardcoding - fully LLM-driven for any subject/language
- Safe - transaction-based with rollback on failure
- Transparent - dry-run mode shows preview before applying
- Scalable - configurable thresholds and cluster sizes
- Validated - ensures all core loops assigned exactly once

**Achievement: Generic topic problem completely resolved! 🎉**

## Known Issues
- Groq free tier rate limit (30 req/min) prevents analyzing large courses in one run
- Splitter may over-split on some edge cases (needs more real-world testing)
- API timeouts with long prompts (enhanced learn with prerequisites may timeout - use --no-concepts flag)
- Deduplication may occasionally merge semantically different items with similar names (e.g., "Mealy Machine" vs "Moore Machine" have 0.92 similarity)
- Topic splitting with `--delete-old` may fail due to foreign key constraints if topic has references (safe to skip deletion)
