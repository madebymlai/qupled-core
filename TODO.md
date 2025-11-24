# Examina - TODO

## Active Development

### Phase 7 - Enhanced Learning System 🚧 IN PROGRESS

**Completed:**
- ✅ Deep theory explanations with prerequisite concepts
- ✅ Step-by-step reasoning with WHY for each step
- ✅ Three depth levels (basic, medium, advanced)

**In Progress:**
- [ ] Metacognitive learning strategies module
- [ ] Study tips per topic/difficulty
- [ ] Problem-solving frameworks
- [ ] Self-assessment prompts
- [ ] Retrieval practice suggestions

**Planned:**
- [ ] Adaptive teaching based on mastery level
- [ ] Track student understanding per topic
- [ ] Detect knowledge gaps and fill proactively
- [ ] Personalized learning paths

## High Priority Improvements

### Phase 3 - AI Analysis
- [ ] **Handle exam files with solutions** - Parse PDFs that include both questions AND solutions in the same document (currently assumes questions-only format)
- [ ] Provider-agnostic rate limiting tracker

### Phase 6 - Multi-Core-Loop Support
- [ ] **Clean up orphaned core loops** - Remove core loops not linked to any exercises
  - ADE: 20 orphaned (out of 32 total)
  - AL: 36 orphaned (out of 47 total)
  - PC: 10 orphaned (out of 22 total)
  - Add `--clean-orphans` flag to deduplicate command
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
