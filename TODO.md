# Examina - TODO

> `[ ]` = pending, `[x]` = complete, `📋` = planned

> CLI tool and examina-core library tasks. For web app, see `examina-cloud/TODO.md`

## ✅ Completed: Business Logic Extraction (v0.15.0)

- [x] Create core/dto/mastery.py - DTOs for mastery calculation
- [x] Create core/dto/progress.py - DTOs for progress tracking
- [x] Create core/ports/mastery_repository.py - Abstract interface
- [x] Create core/progress_analyzer.py - Database-agnostic business logic
- [x] Create core/answer_evaluator.py - Unified answer evaluation
- [x] Add tests/test_progress_analyzer.py - 23 unit tests
- [x] Add tests/test_answer_evaluator.py - 17 unit tests
- [x] Shadow mode rollout complete (2025-11-25)

---

## Pending

### Known Issues
- **Groq rate limit**: Free tier (30 req/min) prevents analyzing large courses in one run
- **API timeouts**: Enhanced learn with prerequisites may timeout - use `--no-concepts` flag
- **Topic splitting**: `--delete-old` may fail due to foreign key constraints if topic has references
- **examina-core packaging**: `from config import Config` fails when installed as package

### Low Priority
- [ ] Concept normalization - Minor: some topics use underscores vs spaces (cosmetic)
- [ ] Interactive merge review for deduplication - Manual approve/reject (web feature)
- [ ] Merge history tracking - Allow undo operations (web feature)

### Long-Term Goals
- [ ] **Community Patterns** - Aggregate pattern data across users on same course/prof
  - "Based on 47 students' uploads, Prof. Rossi loves FSM problems (89%)"
  - Requires: privacy consent, enough user volume, anonymous aggregation
  - Prerequisite: Multi-tenant web app with significant user base

---

## Feature Tiers (Free vs Pro)

**Free Tier:**
- ✅ Exam/exercise ingestion (pattern-based splitting - no LLM)
- ✅ Basic analysis (with procedure cache)
- ✅ Quiz and learning modes
- ✅ Progress tracking

**Pro Tier:**
- 📋 Note/lecture ingestion (smart splitting with LLM, 50 page limit)
- 📋 Advanced explanations (Anthropic for premium quality)
- 📋 Unlimited analysis (no rate limiting)

---

## Completed Phases (Archive)

<details>
<summary>Phase 3 - AI Analysis ✅</summary>

- Handle exam files with solutions (solution separator)
- Provider-agnostic rate limiting tracker
- Analysis Performance Optimization (26+ ex/s with cache)
</details>

<details>
<summary>Phase 6 - Multi-Core-Loop Support ✅</summary>

- Clean up orphaned core loops
- Bilingual procedure deduplication
- Automatic language detection
- Monolingual analysis mode
</details>

<details>
<summary>Phase 7 - Enhanced Learning System ✅</summary>

- Deep theory explanations with prerequisite concepts
- Three depth levels (basic, medium, advanced)
- Metacognitive learning strategies
- Problem-solving frameworks (Polya, IDEAL, Feynman)
- Interactive proof practice mode
</details>

<details>
<summary>Phase 9 - Theory & Proof Support ✅</summary>

- Interactive proof practice mode (`prove` command)
- Theory detection threshold tuning
- Concept map visualization (`concept-map` command)
</details>

<details>
<summary>Phase 10 - Learning Materials ✅</summary>

- Database schema for learning_materials
- Smart splitter as content classifier
- `--material-type exams|notes` flag
- Topic-aware material linker
- Tutor: Theory → Worked Example → Practice flow
- Service interface for SaaS readiness
</details>

<details>
<summary>Phase 11 - Provider Routing ✅</summary>

- Task type system (BULK_ANALYSIS, INTERACTIVE, PREMIUM)
- Provider router with profiles (Free/Pro/Local)
- DeepSeek integration ($0.14/M tokens)
- `--profile [free|pro|local]` flag
</details>

---

## Web Migration

See `examina-cloud/TODO.md` for:
- Phase 2: API Layer ✅
- Phase 4: Frontend ✅ (React + React Query)
- Phase 5: Docker/Deploy ✅ (CI/CD, healthchecks)
- Phase 6: Production (SSL, backups, monitoring)

---

## Notes

For detailed implementation history, see [CHANGELOG.md](CHANGELOG.md).
