# Examina - TODO

## Active Development

### Phase 7 - Enhanced Learning System ✅ COMPLETED

**Core Features:**
- ✅ Deep theory explanations with prerequisite concepts
- ✅ Step-by-step reasoning with WHY for each step
- ✅ Three depth levels (basic, medium, advanced)
- ✅ Metacognitive learning strategies module (`core/metacognitive.py`)
- ✅ Study tips per topic/difficulty (context-aware, research-backed)
- ✅ Problem-solving frameworks (Polya, IDEAL, Feynman, Rubber Duck)
- ✅ Self-assessment prompts (Bloom's taxonomy)
- ✅ Retrieval practice suggestions (5 techniques)
- ✅ Interactive proof practice mode (`prove` command)

**Future Enhancements:**
- [x] Integrate metacognitive tips into `learn` command UI ✅ (completed 2025-11-24)
- [x] Display separated solutions during learning (when available) ✅ (completed 2025-11-24)
- [x] Adaptive teaching based on mastery level ✅ (completed 2025-11-25)
- [x] Track student understanding per topic ✅ (completed 2025-11-25)
- [x] Detect knowledge gaps and fill proactively ✅ (completed 2025-11-25)
- [x] Personalized learning paths ✅ (completed 2025-11-25)

## High Priority Improvements

### Phase 3 - AI Analysis
- [x] **Handle exam files with solutions** ✅ - Implemented generic solution separator (`separate-solutions` command)
  - LLM-based Q+A detection (works for any format/language)
  - Automatic separation with confidence scoring
  - Tested on SO course (10 Q+A detected, 4 separated successfully)
  - Tested on ADE SOLUZIONI (correctly identified 16 question-only exercises)
- [x] **Provider-agnostic rate limiting tracker** ✅ (completed 2025-11-24)
  - Sliding window algorithm (60-second windows)
  - Thread-safe with persistent caching
  - Works for ALL providers (anthropic, groq, ollama, openai, future)
  - CLI command: `examina rate-limits`
- [x] **Analysis Performance Optimization** ✅ (completed 2025-11-24)
  - ✅ **Option 1 Complete**: Increased batch size (10 → 30) - 40% faster
  - ✅ **Option 2 Complete**: Async/await with asyncio - 1.1-5x faster (v0.13.0)
  - ✅ **Option 3 Complete**: Procedure pattern caching - 100% cache hit rate (v0.14.0)
    - Embedding-based similarity matching with text validation
    - Thread-safe for async/parallel analysis
    - CLI command: `examina pattern-cache` (--stats, --build, --clear)
    - 26.2 exercises/second with cached patterns
  - **Remaining option (deferred to web):**
    - ~~Option 4: Stream processing pipeline~~ → Moved to Phase 2 (Web API Layer)
  - **Current performance**: 26+ ex/s with cache hits, 0.39-0.44 ex/s for new exercises
  - **Recommendation**: Build cache first (`pattern-cache --build`), then analyze

### Phase 6 - Multi-Core-Loop Support
- [x] **Clean up orphaned core loops** - ✅ Added `--clean-orphans` flag to deduplicate command
- [x] **Fix mis-categorized exercises** ✅ - Re-analyzed ADE course (B006802), created 123 exercise-core_loop linkages
- [x] **Bilingual procedure deduplication** ✅ (completed 2025-11-24)
  - LLM-based translation detection (ANY language pair)
  - Removed 85 hardcoded translation pairs
  - Automatic cross-language merging
- [x] **Automatic language detection** ✅ (completed 2025-11-24)
  - LLM-based language detection for procedures/topics
  - ISO 639-1 code mapping
  - Database columns: `language` in core_loops and topics
  - CLI command: `examina detect-languages`
- [x] **Strictly monolingual analysis mode** ✅ (completed 2025-11-24)
  - Added `--monolingual` flag to analyze command
  - Automatic primary language detection (from first 5 exercises)
  - LLM-based procedure translation to primary language
  - Prevents cross-language duplicate procedures
  - All tests pass (4/4), fully documented

### Phase 9 - Theory & Proof Support
- [x] **Interactive proof practice mode** ✅ - Already implemented (`prove` command)
- [x] **Tune theory detection threshold** ✅ (completed 2025-11-24)
  - Lowered from 2 keywords → 1 keyword
  - Added explicit prompt notes that 1 keyword is sufficient
  - Expected to improve detection from 55% to 70%+
- [ ] Re-analyze existing exercises with Phase 9 detection
- [ ] Build theory concept dependency visualization

### Phase 10 - Learning Materials as First-Class Content 🚧 IN PROGRESS

**Goal:** Lecture notes and slides become first-class learning materials, not forced into exercise format.

**Design Document:** See [PHASE10_DESIGN.md](PHASE10_DESIGN.md) for complete design principles and contracts.

**Conceptual Model:**
- Topics / Core Loops → Abstract concepts ("FSM minimization", "Moore machine design")
- Learning Materials → Theory sections, worked examples, references from notes/slides
- Exercises → Practice problems / exam-style questions

**Learning Flow:** Topic → theory → worked example → practice (default learning script, not optional)

**Key Design Principles:**
1. **Smart splitter = pure classifier** (segment/classify content, don't store)
2. **Ingestion modes = document type** (`--material-type exams|notes` describes PDF, not algorithm)
3. **Topic linking = symmetric treatment** (materials and exercises use same detection logic)
4. **Tutor flow = explicit, configurable** (theory → example → practice is first-class, not "sometimes show notes")
5. **Success = no regression + notes coverage** (exam pipeline unchanged, notes become usable)

**Status:**

- [x] **Database Schema** ✅ (completed 2025-11-24)
  - `learning_materials` table (id, course_code, material_type, title, content, source_pdf, page_number)
  - `material_topics` join table (many-to-many: materials ↔ topics)
  - `material_exercise_links` table (link worked examples to practice exercises)
  - Database methods: store, link, retrieve materials and relationships

- [x] **Smart Splitter as Content Classifier** ✅ (completed 2025-11-24)
  - Updated `_build_detection_prompt()` to classify as theory/worked_example/practice_exercise
  - Updated `_parse_detection_response()` to return `DetectedContent` objects
  - Updated `split_pdf_content()` to return BOTH exercises and learning_materials
  - Pattern-based splitting preserved for structured exams (no regression)

- [x] **Ingestion Mode: --material-type flag** ✅ (completed 2025-11-24)
  - Added `--material-type exams|notes` flag to `ingest` command
  - For `--material-type exams`: Pattern-based (default), optional `--smart-split`
  - For `--material-type notes`: Smart splitting enabled automatically
  - Stores learning_materials in database via new methods

- [x] **Topic-Aware Material Linker** ✅ (completed 2025-11-24)
  - `analyze_learning_material()` - mirrors exercise analysis
  - `link_materials_to_topics()` - semantic matching to existing topics
  - `link_worked_examples_to_exercises()` - similarity-based linking
  - CLI command: `examina link-materials --course CODE`

- [x] **Tutor: Theory → Worked Example → Practice Flow** ✅ (completed 2025-11-24)
  - Enhanced `learn()` method with theory and worked example display
  - Configurable parameters: show_theory, show_worked_examples, max_theory_sections, max_worked_examples
  - Defaults from Config (SHOW_THEORY_BY_DEFAULT, MAX_THEORY_SECTIONS_IN_LEARN, etc.)
  - Bilingual support (en/it) for theory and example display
  - Graceful fallback when no materials exist

**Refinements Completed (2025-11-24):**

- [x] **Configuration System** ✅
  - Added 7 Phase 10 config constants (LEARNING_MATERIALS_ENABLED, SHOW_THEORY_BY_DEFAULT, etc.)
  - All configurable via environment variables
  - Moved hardcoded 0.3 threshold to Config.WORKED_EXAMPLE_EXERCISE_SIMILARITY_THRESHOLD

- [x] **Tutor Configurability** ✅
  - Added show_theory, show_worked_examples, max_theory_sections, max_worked_examples parameters
  - Theory → example → practice is default flow (not conditional)
  - Documented in docstring and implementation

- [x] **Database Method Enhancement** ✅
  - Added limit parameter to get_learning_materials_by_topic()
  - Tutor respects max limits when fetching materials

**Design Constraints:**
- ✅ No regression on existing exercise-based features (analysis, quiz, spaced repetition)
- ✅ Default behavior for structured exams remains fast (pattern-based)
- ✅ Smart splitting and learning materials are additive enhancements
- ✅ Many-to-many relationships (materials ↔ topics, materials ↔ exercises)

**Testing & Validation:**
- [x] **End-to-End Testing** ✅ (completed 2025-11-24)
  - Tested with 3-page Italian lecture notes (Appunti-AE-3pages.pdf)
  - Successfully extracted 3 exercises, 20 materials (19 theory, 1 worked example)
  - Verified Theory → LLM Explanation flow in learn command
  - All design principles validated in production use
  - See PHASE10_TESTING.md for detailed test results

**Remaining Cleanup:**
- [x] **Service Interface for SaaS Readiness** ✅ (completed 2025-11-24)
  - Fixed hardcoded providers in CLI (cli.py:1196) and Tutor (tutor.py:37)
  - Added consistent `--provider` flag to all commands (learn, prove, practice, etc.)
  - Created stateless `ExaminaService` class wrapping core operations (core/service.py)
  - Documented service interface for future web layer integration
  - Goal achieved: Thin web layer can instantiate services with user preferences

---

### Phase 11 - Provider Routing Architecture ✅ COMPLETED (2025-11-24)

**Goal:** Task-based routing to optimize cost and quality across different LLM providers.

**Implementation:**

- [x] **Task Type System** (`core/task_types.py`)
  - Three task categories: BULK_ANALYSIS, INTERACTIVE, PREMIUM
  - Different providers optimized for different use cases

- [x] **Provider Router** (`core/provider_router.py`)
  - Profile-based routing (Free/Pro/Local)
  - Automatic fallback handling (API key missing only)
  - Fail-fast design for transparency (no runtime fallback)

- [x] **Provider Profiles** (`config/provider_profiles.yaml`)
  - **Free:** DeepSeek for bulk, Groq for interactive, premium disabled
  - **Pro:** DeepSeek for bulk, Anthropic for interactive/premium
  - **Local:** Ollama only (privacy mode)

- [x] **DeepSeek Integration**
  - Added DeepSeek provider support (671B MoE model)
  - $0.14/M tokens (10-20x cheaper than Anthropic)
  - No rate limiting (high/unlimited RPM)
  - Primary provider for bulk operations in Free/Pro profiles

- [x] **CLI Integration**
  - Added `--profile [free|pro|local]` flag to commands
  - Backward compatible `--provider` flag still works
  - Service layer supports both routing and direct provider usage

- [x] **Testing & Validation**
  - All routing tests pass (6/6 scenarios)
  - Both profile-based and direct provider usage tested
  - DeepSeek API verified working

**Cost Impact:**
- Bulk operations: 10-20x cost reduction vs Anthropic
- Free tier sustainable with DeepSeek + Groq
- Pro tier balances cost (DeepSeek bulk) + quality (Anthropic interactive)

**Design Decisions:**
- Fail-fast on provider failures (no silent fallback for cost/quality control)
- Fallback only when API key missing (configuration issue, not runtime issue)
- Profile-based routing encourages cost-aware usage patterns

---

### Feature Tiers (Free vs Pro)

**Design Decision:** Some features require significant LLM tokens and should be gated.

**Free Tier Features:**
- ✅ Exam ingestion (pattern-based splitting - no LLM)
- ✅ Basic analysis (with procedure cache)
- ✅ Quiz and learning modes
- ✅ Progress tracking

**Pro Tier Features:**
- 📋 **Note/lecture ingestion** - Requires LLM-based smart splitting (high token cost)
- 📋 **Advanced explanations** - Uses Anthropic for premium quality
- 📋 **Unlimited analysis** - No rate limiting on bulk operations

**Rationale:**
- Notes are unstructured → require LLM to classify (theory/example/exercise)
- Exams are structured → pattern-based splitting is free and fast
- This keeps free tier sustainable while monetizing heavy LLM usage

---

## Future: Web Application Migration 🌐

**IMPORTANT DESIGN PRINCIPLE:** All new code must be web-ready.

### Repository Strategy

**Two-Repo Architecture:**

#### 📂 Public Repo: `examina` (Current) - Open Source Tool
**What stays here:**
- ✅ CLI tool (`cli.py`)
- ✅ Core analysis logic (`core/`, `storage/`, `models/`, `utils/`)
- ✅ Local database (SQLite)
- ✅ Local usage (BYO API keys)
- ✅ MIT License (portfolio piece)
- ✅ Documentation and examples

**Target audience:** Nerds, developers, power users, students who want to self-host

**Value proposition:** "Run Examina locally for free with your own API keys"

#### 🔒 Private Repo: `examina-cloud` (Future) - SaaS Platform
**What goes there:**
- 🔐 Web app (frontend + backend API)
- 🔐 Authentication and authorization
- 🔐 Billing and subscription management
- 🔐 Multi-user and team features
- 🔐 Deployment scripts and infrastructure
- 🔐 Monitoring, logging, and analytics
- 🔐 Proprietary features and optimizations

**Target audience:** Students who want convenience, non-technical users, institutions

**Value proposition:** "Hosted Examina with accounts, payments, and polished UI"

### Migration Roadmap (Long-term)

- [ ] **Phase 0: Repository Setup**
  - Create private GitHub repo: `examina-cloud`
  - Set up repo structure (frontend/, backend/, infrastructure/)
  - Configure GitHub secrets (API keys, deployment configs)
  - Set up dev environment (Docker, local dev stack)
  - Invite team members (if any)

- [ ] **Phase 1: Core Library Extraction**
  - Ensure all business logic is in `core/` (framework-agnostic)
  - Create Python package: `examina-core` (installable via pip)
  - Version and publish to PyPI (private or public)
  - Private repo can import: `from examina_core import Analyzer, Tutor`

- [ ] **Phase 2: API Layer** (Private Repo)
  - FastAPI REST API
  - Wrap `examina-core` functions as API endpoints
  - JWT authentication
  - Rate limiting (per-user, not per-provider)
  - Multi-tenancy (user_id isolation)
  - **Background Job System:**
    - Celery + Redis for async task queue
    - Background workers for long-running analysis
    - Job status polling endpoint (`GET /jobs/{id}`)
    - WebSocket for real-time progress updates
    - Streaming pipeline (exercise → analyze → dedupe → store)
    - Retry logic with exponential backoff

- [ ] **Phase 3: Frontend** (Private Repo)
  - React/Vue web UI
  - File upload and PDF processing
  - Interactive quiz interface
  - Progress dashboards
  - Payment integration (Stripe/Paddle)

- [ ] **Phase 4: Database Migration** (Private Repo)
  - User accounts system
  - PostgreSQL with user_id foreign keys
  - Data isolation per user
  - Migration tools from SQLite (for self-hosted users)

- [ ] **Phase 5: Deployment** (Private Repo)
  - Docker containers
  - Kubernetes/AWS ECS
  - CI/CD pipeline (GitHub Actions)
  - Monitoring (Sentry, DataDog)
  - CDN for static assets

### Web-Ready Design Guidelines

**All new code MUST follow these principles:**

1. **No Hardcoding** ✓ (already enforced)
   - No hardcoded course codes, provider names, or configuration
   - All settings via environment variables or database

2. **Separation of Concerns**
   - Business logic in `core/` (reusable in web)
   - CLI-specific code in `cli.py` only
   - Database operations in `storage/` (abstract layer)

3. **Stateless Operations**
   - No global state or singletons
   - Pass dependencies explicitly (dependency injection)
   - Functions should be pure where possible

4. **Multi-User Ready**
   - Plan for `user_id` column in tables
   - Avoid assumptions of single-user
   - Consider data isolation and permissions

5. **Async-Friendly**
   - Avoid blocking operations where possible
   - Consider async/await patterns
   - Use connection pooling for databases

6. **API-First Thinking**
   - Functions should accept/return structured data (dicts, dataclasses)
   - Avoid print() - use proper logging
   - Return error codes, not sys.exit()

## Low Priority / Future

- [ ] Language detection for procedures - Automatically detect and merge equivalent procedures
- [ ] Concept normalization - Handle variations like "autovalori_autovettori" vs "autovalori_e_autovettori"
- [ ] Interactive merge review for deduplication - Manual approve/reject
- [ ] Merge history tracking - Allow undo operations
- [ ] Core loop similarity tuning - Review 95 ADE merges (might be legitimate)

## Known Issues

- **Groq rate limit**: Free tier (30 req/min) prevents analyzing large courses in one run
- **API timeouts**: Enhanced learn with prerequisites may timeout - use `--no-concepts` flag
- **Topic splitting**: `--delete-old` may fail due to foreign key constraints if topic has references

## Notes

For completed phases and detailed implementation history, see [CHANGELOG.md](CHANGELOG.md).
