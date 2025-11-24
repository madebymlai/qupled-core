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
- [ ] Provider-agnostic rate limiting tracker

### Phase 6 - Multi-Core-Loop Support
- [x] **Clean up orphaned core loops** - ✅ Added `--clean-orphans` flag to deduplicate command
- [x] **Fix mis-categorized exercises** ✅ - Re-analyzed ADE course (B006802), created 123 exercise-core_loop linkages
  - Note: Some edge cases remain (1 pure Moore exercise still linked to Mealy due to LLM caching)
  - All exercises now properly categorized and linked to core loops
- [ ] Bilingual procedure deduplication - Merge duplicate procedures across languages
- [ ] Strictly monolingual analysis mode - Ensure procedures extracted in only one language
- [ ] Cross-language procedure similarity - Use embeddings to match equivalent procedures

### Phase 9 - Theory & Proof Support
- [ ] Re-analyze existing 75 exercises with Phase 9 detection
- [ ] Tune theory detection threshold (2 keywords → 1 keyword)
- [ ] Add interactive proof practice mode
- [ ] Build theory concept dependency visualization

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
