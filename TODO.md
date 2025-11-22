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

## Phase 4 - Tutor Features (CURRENT)

### Completed
- ✅ **Add Anthropic Claude Sonnet 4.5** - Better rate limits, higher quality (14 topics, 23 core loops found!)
- ✅ **Analyze with Anthropic** - Successfully analyzed all 27 ADE exercises including SR Latch

### In Progress
- [ ] **Add language switch (Italian/English)** - Allow user to choose output language for analysis and tutor
  - Add `--lang` flag to analyze command
  - Store language preference in config
  - Update prompts to request responses in selected language

### Next Steps
- [ ] Implement `learn` command - Interactive tutor for learning core loops
- [ ] Implement `practice` command - Practice exercises with hints
- [ ] Implement `generate` command - Generate new similar exercises

## Phase 5 - Quiz System
- [ ] Implement `quiz` command - Generate quizzes from exercises
- [ ] Spaced repetition algorithm (SM-2)
- [ ] Progress tracking and analytics

## Known Issues
- Groq free tier rate limit (30 req/min) prevents analyzing large courses in one run
- Splitter may over-split on some edge cases (needs more real-world testing)
- Topics can be duplicated with slight variations in naming
