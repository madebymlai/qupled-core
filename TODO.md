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
- [ ] Adaptive teaching based on mastery level
- [ ] Track student understanding per topic
- [ ] Detect knowledge gaps and fill proactively
- [ ] Personalized learning paths

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
- [ ] Strictly monolingual analysis mode - Ensure procedures extracted in only one language

### Phase 9 - Theory & Proof Support
- [x] **Interactive proof practice mode** ✅ - Already implemented (`prove` command)
- [ ] Re-analyze existing exercises with Phase 9 detection
- [ ] Tune theory detection threshold (2 keywords → 1 keyword)
- [ ] Build theory concept dependency visualization

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
  - FastAPI/Flask REST API
  - Wrap `examina-core` functions as API endpoints
  - JWT authentication
  - Rate limiting (per-user, not per-provider)
  - Multi-tenancy (user_id isolation)

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
