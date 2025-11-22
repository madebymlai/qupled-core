# Examina - TODO List

## Phase 3 - AI Analysis ✅ COMPLETED

**Done:**
- ✅ Intelligent splitter (filters instructions, works for all formats)
- ✅ AI analysis with Groq
- ✅ Rate limit handling with exponential retry
- ✅ Database + Vector store
- ✅ Topic and core loop discovery

**Future improvements (low priority):**
- [ ] Provider-agnostic rate limiting tracker
- [ ] Resume failed analysis
- [ ] Batch processing optimization
- [ ] Topic/core loop deduplication
- [ ] Confidence thresholds
- [ ] Caching LLM responses

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

## Phase 5 - Quiz System
- [ ] Implement `quiz` command - Generate quizzes from exercises
- [ ] Spaced repetition algorithm (SM-2)
- [ ] Progress tracking and analytics

## Known Issues
- Groq free tier rate limit (30 req/min) prevents analyzing large courses in one run
- Splitter may over-split on some edge cases (needs more real-world testing)
- Topics can be duplicated with slight variations in naming
