# Option 3: Procedure Pattern Caching - Implementation Plan

**Goal:** Cache common procedure patterns to avoid redundant LLM calls, achieving high-effort, very-high-gain performance improvement.

**Expected Impact:** 50-80% reduction in LLM calls for courses with repetitive exercises.

---

## 1. Current Architecture Analysis

### 1.1 Current Flow
```
Exercise Text
    ↓
[LLM Analysis] ← EXPENSIVE (every exercise)
    ↓
ProcedureInfo(name, type, steps)
    ↓
[Deduplication] ← Happens AFTER all exercises analyzed
    ↓
Core Loop Storage
```

**Problem:** Each exercise requires LLM call, even for identical procedure patterns.

### 1.2 Current Components
- **ExerciseAnalyzer** (`core/analyzer.py`):
  - `_analyze_exercise_with_retry()` - Makes LLM call for each exercise
  - `_build_analysis_prompt()` - Constructs analysis prompt
  - Returns `AnalysisResult` with `List[ProcedureInfo]`

- **SemanticMatcher** (`core/semantic_matcher.py`):
  - Already has embedding infrastructure (`all-MiniLM-L6-v2`)
  - `cosine_similarity()` for comparing embeddings
  - Used for deduplication

- **Database Schema**:
  ```sql
  core_loops (
      id TEXT PRIMARY KEY,           -- normalized name
      topic_id INTEGER,
      name TEXT NOT NULL,            -- display name
      description TEXT,
      procedure TEXT NOT NULL,       -- JSON: List[str] steps
      difficulty_avg REAL,
      exercise_count INTEGER
  )
  ```

---

## 2. Proposed Architecture

### 2.1 New Flow
```
Exercise Text
    ↓
[Pattern Matcher] ← NEW: Check cache first
    ↓
    ├─ CACHE HIT → Return cached procedure (skip LLM)
    │              95% confidence
    ↓
    └─ CACHE MISS → [LLM Analysis] → Store in cache
                     (only 20-50% of exercises)
```

### 2.2 Cache Strategy

**Hybrid Approach:**
1. **Exercise Text Embedding** (fast, fuzzy matching)
2. **Keyword Extraction** (instant, high precision for common patterns)
3. **Fallback to LLM** (when confidence < threshold)

---

## 3. Implementation Phases

### Phase 1: Cache Infrastructure (2-3 hours)

#### 3.1 Create ProcedureCache Class
**File:** `core/procedure_cache.py`

```python
class ProcedureCache:
    """
    Cache for common procedure patterns.

    Supports two cache modes:
    1. Exercise-text-based: Match full exercise text via embedding
    2. Keyword-based: Extract keywords and match to known patterns
    """

    def __init__(self, db: Database, semantic_matcher: SemanticMatcher):
        self.db = db
        self.semantic_matcher = semantic_matcher
        self.cache = {}  # In-memory cache
        self.embeddings = {}  # Exercise text → embedding

    def lookup(self, exercise_text: str) -> Optional[CacheHit]:
        """
        Try to find matching procedure in cache.

        Returns:
            CacheHit with confidence score, or None if no match
        """

    def add(self, exercise_text: str, procedures: List[ProcedureInfo]):
        """Add newly analyzed exercise to cache."""

    def build_from_database(self, course_code: str = None):
        """Build cache from existing analyzed exercises."""
```

**Data Structures:**
```python
@dataclass
class CacheHit:
    procedures: List[ProcedureInfo]
    confidence: float  # 0.0-1.0
    match_type: str   # 'exact', 'embedding', 'keyword'
    source_exercise_id: str
```

#### 3.2 Database Schema Addition
**New table:** `procedure_cache_entries`

```sql
CREATE TABLE procedure_cache_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    course_code TEXT,              -- NULL = global cache
    pattern_hash TEXT NOT NULL,    -- Hash of normalized exercise text
    exercise_text_sample TEXT,     -- First 500 chars for inspection
    procedures_json TEXT NOT NULL, -- Cached procedures
    embedding BLOB,                -- Vector embedding (optional)
    match_count INTEGER DEFAULT 0, -- How many times this was matched
    confidence_avg REAL DEFAULT 1.0, -- Average confidence when matched
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_matched_at TIMESTAMP,

    UNIQUE(course_code, pattern_hash)
)
```

**Indexes:**
```sql
CREATE INDEX idx_cache_course ON procedure_cache_entries(course_code);
CREATE INDEX idx_cache_hash ON procedure_cache_entries(pattern_hash);
CREATE INDEX idx_cache_matches ON procedure_cache_entries(match_count DESC);
```

#### 3.3 Configuration
**File:** `config.py` additions

```python
# Procedure Caching (Option 3)
PROCEDURE_CACHE_ENABLED = os.getenv("EXAMINA_CACHE_ENABLED", "true").lower() == "true"
PROCEDURE_CACHE_MIN_CONFIDENCE = float(os.getenv("EXAMINA_CACHE_MIN_CONFIDENCE", "0.85"))
PROCEDURE_CACHE_SCOPE = os.getenv("EXAMINA_CACHE_SCOPE", "course")  # 'course' or 'global'
PROCEDURE_CACHE_MODE = os.getenv("EXAMINA_CACHE_MODE", "hybrid")  # 'embedding', 'keyword', 'hybrid'
PROCEDURE_CACHE_EMBEDDING_THRESHOLD = float(os.getenv("EXAMINA_CACHE_EMBEDDING_THRESHOLD", "0.92"))
```

---

### Phase 2: Pattern Matching Logic (3-4 hours)

#### 2.1 Exercise Text Normalization
```python
def normalize_exercise_text(text: str) -> str:
    """
    Normalize exercise text for consistent matching.

    - Remove whitespace variations
    - Normalize numbers (preserve structure)
    - Remove latex artifacts
    - Lowercase
    """
```

#### 2.2 Embedding-Based Matching
```python
def match_by_embedding(self, exercise_text: str, threshold: float = 0.92) -> Optional[CacheHit]:
    """
    Match exercise by semantic similarity of full text.

    Process:
    1. Generate embedding for exercise_text
    2. Compare with all cached embeddings (vectorized)
    3. If cosine_similarity >= threshold → HIT
    4. Return highest-scoring match

    Performance:
    - O(n) where n = cache size
    - Fast with numpy vectorization
    - ~0.1ms per comparison with optimized implementation
    """
```

**Optimization:** Use FAISS for large caches (>1000 entries)

#### 2.3 Keyword-Based Matching
```python
KEYWORD_PATTERNS = {
    "moore_to_mealy": [
        r"moore.*mealy",
        r"convert.*moore.*to.*mealy",
        r"trasforma.*moore.*in.*mealy"
    ],
    "mealy_to_moore": [...],
    "fsm_minimization": [...],
    "truth_table_design": [...],
    # ... add common patterns
}

def match_by_keywords(self, exercise_text: str) -> Optional[CacheHit]:
    """
    Match exercise by detecting known keyword patterns.

    Pros:
    - Instant matching (regex)
    - High precision for common patterns
    - No embedding overhead

    Cons:
    - Requires pattern maintenance
    - Won't match novel phrasings
    """
```

#### 2.4 Hybrid Strategy
```python
def lookup(self, exercise_text: str) -> Optional[CacheHit]:
    """
    Hybrid matching strategy:

    1. Try keyword matching (instant, high precision)
       → If match with confidence >= 0.95: RETURN

    2. Try embedding matching (fast, fuzzy)
       → If match with confidence >= threshold: RETURN

    3. Return None → Fall back to LLM
    """
```

---

### Phase 3: Integration with Analyzer (2-3 hours)

#### 3.1 Modify ExerciseAnalyzer
**File:** `core/analyzer.py`

**Current:**
```python
def _analyze_exercise_with_retry(self, exercise_text: str, ...):
    # ALWAYS call LLM
    result = self.llm.generate(...)
```

**New:**
```python
def _analyze_exercise_with_retry(self, exercise_text: str, ...):
    # Try cache first
    if self.procedure_cache and Config.PROCEDURE_CACHE_ENABLED:
        cache_hit = self.procedure_cache.lookup(exercise_text)

        if cache_hit and cache_hit.confidence >= Config.PROCEDURE_CACHE_MIN_CONFIDENCE:
            # CACHE HIT - Skip LLM!
            self.cache_stats['hits'] += 1
            return self._build_result_from_cache(cache_hit, exercise_text)

    # CACHE MISS - Call LLM
    self.cache_stats['misses'] += 1
    result = self.llm.generate(...)

    # Add to cache for future
    if self.procedure_cache:
        self.procedure_cache.add(exercise_text, result.procedures)

    return result
```

#### 3.2 Async Support
```python
async def _analyze_exercise_with_retry_async(self, exercise_text: str, ...):
    # Cache lookup is synchronous (fast)
    if self.procedure_cache and Config.PROCEDURE_CACHE_ENABLED:
        cache_hit = self.procedure_cache.lookup(exercise_text)

        if cache_hit and cache_hit.confidence >= threshold:
            self.cache_stats['hits'] += 1
            return self._build_result_from_cache(cache_hit, exercise_text)

    # Async LLM call for cache miss
    self.cache_stats['misses'] += 1
    result = await self.llm.generate_async(...)

    # Add to cache
    if self.procedure_cache:
        self.procedure_cache.add(exercise_text, result.procedures)

    return result
```

---

### Phase 4: Cache Management (1-2 hours)

#### 4.1 Build Cache from Existing Data
**CLI Command:** `examina build-procedure-cache`

```python
def build_procedure_cache(course_code: str = None, scope: str = "course"):
    """
    Build procedure cache from existing analyzed exercises.

    Process:
    1. Query all analyzed exercises (where analyzed=1)
    2. For each exercise:
       - Extract procedures from analysis_metadata
       - Generate embedding
       - Add to cache
    3. Save cache to database

    Performance:
    - ~100 exercises/sec (embedding generation is bottleneck)
    - For 1000 exercises: ~10 seconds
    """
```

#### 4.2 Cache Statistics
```python
@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    total_lookups: int = 0
    avg_confidence: float = 0.0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total_lookups if self.total_lookups > 0 else 0.0

    @property
    def llm_calls_saved(self) -> int:
        return self.hits
```

**Display in CLI:**
```
📊 Procedure Cache Statistics:
   Cache hits: 85
   Cache misses: 15
   Hit rate: 85.0%
   💰 Saved ~85 LLM calls!
   Avg confidence: 0.94
```

#### 4.3 Cache Maintenance Commands
```bash
# Build cache from existing data
examina build-procedure-cache --course B006802

# Clear cache
examina clear-procedure-cache --course B006802

# View cache statistics
examina cache-stats --course B006802

# Export cache for sharing
examina export-cache --course B006802 --output cache.json

# Import cache
examina import-cache --input cache.json
```

---

## 4. Performance Impact Analysis

### 4.1 Current Performance (v0.13.0)
```
DeepSeek + async mode:
- 27 exercises in 61.4s
- 0.44 ex/s
- 27 LLM calls (100% of exercises)
```

### 4.2 Expected Performance with Caching

**Assumptions:**
- Cache hit rate: 70% (conservative for repetitive courses)
- Cache lookup time: 5ms (embedding comparison)
- LLM call time: 2000ms (current average)

**Calculation:**
```
Without cache: 27 exercises × 2000ms = 54,000ms = 54s
With cache:
  - 19 hits × 5ms = 95ms
  - 8 misses × 2000ms = 16,000ms
  Total: 16,095ms ≈ 16s

Speedup: 54s → 16s = 3.4x faster
```

**Real-world estimate:**
- **Best case** (90% hit rate): 5-6x faster
- **Typical** (70% hit rate): 3-4x faster
- **Worst case** (30% hit rate): 1.5-2x faster

### 4.3 Combined with Async (v0.13.0)
```
Async + Caching:
- Async provides 1.12x speedup on cache misses
- Cache reduces number of LLM calls by 70%

Combined: DeepSeek baseline 68.8s
→ With async: 61.4s (1.12x)
→ With cache: 20.6s (3.3x from baseline)
→ With async + cache: 18.4s (3.7x from baseline)

Target: 1.5 ex/s (27 exercises in ~18s)
```

---

## 5. Implementation Checklist

### Phase 1: Infrastructure ✓ (2-3 hours)
- [ ] Create `core/procedure_cache.py` with `ProcedureCache` class
- [ ] Define `CacheHit` dataclass
- [ ] Add database schema for `procedure_cache_entries` table
- [ ] Add configuration variables to `config.py`
- [ ] Write unit tests for cache class

### Phase 2: Matching Logic ✓ (3-4 hours)
- [ ] Implement `normalize_exercise_text()`
- [ ] Implement `match_by_embedding()` with cosine similarity
- [ ] Define `KEYWORD_PATTERNS` for common procedures
- [ ] Implement `match_by_keywords()` with regex
- [ ] Implement hybrid `lookup()` strategy
- [ ] Test matching accuracy on sample exercises

### Phase 3: Analyzer Integration ✓ (2-3 hours)
- [ ] Modify `ExerciseAnalyzer.__init__()` to accept cache
- [ ] Update `_analyze_exercise_with_retry()` with cache lookup
- [ ] Update `_analyze_exercise_with_retry_async()` with cache lookup
- [ ] Implement `_build_result_from_cache()` helper
- [ ] Add `cache_stats` tracking
- [ ] Test sync and async modes with cache

### Phase 4: Management & CLI ✓ (1-2 hours)
- [ ] Implement `build-procedure-cache` command
- [ ] Implement `clear-procedure-cache` command
- [ ] Implement `cache-stats` command
- [ ] Implement `export-cache` / `import-cache` commands
- [ ] Add cache statistics to analyze output
- [ ] Document cache usage in README

### Phase 5: Testing & Validation ✓ (2-3 hours)
- [ ] Test on B006802 (27 exercises)
- [ ] Measure cache hit rate
- [ ] Measure performance improvement
- [ ] Test cache persistence across runs
- [ ] Test cache with different courses
- [ ] Verify no accuracy regression

### Phase 6: Documentation ✓ (1 hour)
- [ ] Update CHANGELOG.md with v0.14.0
- [ ] Update TODO.md marking Option 3 complete
- [ ] Add PROCEDURE_CACHING.md user guide
- [ ] Update CLI help text
- [ ] Add performance benchmarks

---

## 6. Edge Cases & Risks

### 6.1 Edge Cases

**1. Cache Stale Data**
- **Problem:** Cached procedures may be incorrect if analysis improves
- **Solution:** Version cache entries, invalidate on schema changes
- **Mitigation:** Add `cache_version` field, clear cache on major updates

**2. Course-Specific vs Global Cache**
- **Problem:** Some procedures are course-specific (domain terminology)
- **Solution:** Support both scopes via `PROCEDURE_CACHE_SCOPE` config
- **Recommendation:** Default to `course` scope for accuracy

**3. Multi-Language Exercises**
- **Problem:** Same procedure in different languages won't match
- **Solution:**
  - Option A: Multilingual embeddings (paraphrase-multilingual-MiniLM-L12-v2)
  - Option B: Translate before matching (use existing monolingual mode)
- **Recommendation:** Option A (multilingual embeddings)

**4. Novel Exercise Patterns**
- **Problem:** Genuinely new procedures always miss cache
- **Solution:** This is expected behavior - cache grows over time
- **Mitigation:** Pre-populate cache with common patterns

**5. False Positives (Wrong Match)**
- **Problem:** High similarity but different procedure
- **Solution:** Use conservative threshold (0.92 default)
- **Validation:** Log low-confidence matches for review

### 6.2 Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Cache mismatches reduce accuracy | Medium | High | Conservative threshold (0.92), log mismatches |
| Embedding generation slows analysis | Low | Medium | Async embedding, batch processing |
| Cache grows too large | Low | Low | Prune low-match-count entries, TTL |
| False sense of coverage | Medium | Low | Report cache hit rate, allow manual review |

---

## 7. Testing Strategy

### 7.1 Unit Tests
```python
def test_cache_exact_match():
    """Exact duplicate exercise should hit cache with 1.0 confidence."""

def test_cache_fuzzy_match():
    """Similar exercise should hit cache with >0.92 confidence."""

def test_cache_miss_novel_pattern():
    """Novel procedure should miss cache."""

def test_cache_keyword_match():
    """Known keyword pattern should hit cache instantly."""

def test_cache_persistence():
    """Cache should persist across analyzer instances."""
```

### 7.2 Integration Tests
```python
def test_analyze_with_cache_enabled():
    """Analyze 27 exercises with cache, verify hit rate > 50%."""

def test_cache_accuracy():
    """Verify cached results match LLM results."""

def test_async_mode_with_cache():
    """Verify async mode works correctly with cache."""
```

### 7.3 Performance Benchmarks
```bash
# Baseline (no cache)
time examina analyze --course B006802 --force --provider deepseek

# With cache (first run - builds cache)
time examina analyze --course B006802 --force --provider deepseek --use-cache

# With cache (second run - uses cache)
time examina analyze --course B006802 --force --provider deepseek --use-cache

# Expected results:
# Baseline: 68.8s
# First run: 70s (slight overhead building cache)
# Second run: 20s (3.4x faster with 70% hit rate)
```

---

## 8. Timeline Estimate

**Total: 11-18 hours** (1.5-2 days of focused work)

| Phase | Time | Description |
|-------|------|-------------|
| Phase 1 | 2-3h | Cache infrastructure & database |
| Phase 2 | 3-4h | Matching logic (embedding + keywords) |
| Phase 3 | 2-3h | Analyzer integration (sync + async) |
| Phase 4 | 1-2h | CLI commands & management |
| Phase 5 | 2-3h | Testing & validation |
| Phase 6 | 1h | Documentation |

**Dependencies:**
- Existing: SemanticMatcher, embeddings infrastructure ✓
- New: sentence-transformers (already installed) ✓
- Optional: FAISS for large caches (install if needed)

---

## 9. Success Criteria

### 9.1 Performance Goals
- [ ] 3-4x speedup on repetitive courses (70% hit rate)
- [ ] 5-6x speedup on highly repetitive courses (90% hit rate)
- [ ] 1.5-2x speedup on diverse courses (30% hit rate)
- [ ] Cache lookup time < 10ms per exercise
- [ ] No accuracy regression (match LLM results)

### 9.2 Functional Goals
- [ ] Cache persists across runs
- [ ] Cache statistics displayed to user
- [ ] Course-specific and global cache modes work
- [ ] Cache can be exported/imported for sharing
- [ ] CLI commands for cache management functional

### 9.3 Quality Goals
- [ ] Conservative matching (no false positives)
- [ ] Clear logging of cache hits/misses
- [ ] Graceful degradation if cache fails
- [ ] Backward compatible (can disable caching)

---

## 10. Future Enhancements (Post-v0.14.0)

### 10.1 Adaptive Cache
- Machine learning to predict cache hit likelihood
- Dynamic threshold adjustment based on course characteristics
- Automatic cache pruning of low-quality entries

### 10.2 Distributed Cache
- Share cache across users (community-driven)
- Privacy-preserving: hash exercise text, share procedures
- Versioned cache updates from central repository

### 10.3 Procedure Templates
- Extract generalized procedure templates
- Match new exercises to templates with variable substitution
- Example: "Convert X to Y" template for any FSM conversion

### 10.4 Multi-Modal Caching
- Cache image-based exercises (diagram matching)
- Perceptual hashing for similar diagrams
- OCR + semantic matching for handwritten exercises

---

## 11. Recommendation

**Proceed with implementation?** ✅ YES

**Rationale:**
1. **High ROI:** 11-18 hours investment for 3-4x speedup
2. **Proven approach:** Leverages existing embedding infrastructure
3. **Low risk:** Conservative matching, graceful fallback to LLM
4. **Incremental:** Can ship Phase 1-4, defer advanced features
5. **Compound benefit:** Stacks with async (1.12x) for 3.7x total speedup

**Recommended approach:**
- Start with Phase 1-3 (core functionality)
- Test on B006802 to validate hit rate
- If successful (>50% hit rate), complete Phase 4-6
- Ship as v0.14.0 with comprehensive documentation

**Alternative (if short on time):**
- Implement keyword-only matching (simpler, 4-6 hours)
- Skip embedding matching initially
- Still achieve 40-60% hit rate on common patterns
- Upgrade to hybrid later if needed

---

## Appendix A: Code Snippets

### A.1 Cache Lookup Integration

```python
# core/analyzer.py - Modified _analyze_exercise_with_retry()

def _analyze_exercise_with_retry(self, exercise_text: str,
                                 course_name: str,
                                 previous_text: Optional[str] = None,
                                 max_retries: int = 2) -> ExerciseAnalysis:
    """Analyze exercise with retry logic and procedure caching."""

    # CACHE LOOKUP (NEW)
    if self.procedure_cache and Config.PROCEDURE_CACHE_ENABLED:
        cache_hit = self.procedure_cache.lookup(exercise_text)

        if cache_hit and cache_hit.confidence >= Config.PROCEDURE_CACHE_MIN_CONFIDENCE:
            self.cache_stats['hits'] += 1

            # Build result from cache (skip LLM entirely)
            return ExerciseAnalysis(
                is_valid_exercise=True,
                is_fragment=False,
                should_merge_with_previous=False,
                topic=cache_hit.topic,
                difficulty=cache_hit.difficulty,
                variations=cache_hit.variations,
                confidence=cache_hit.confidence,
                procedures=cache_hit.procedures,
                # ... other fields
            )

    # CACHE MISS - Proceed with LLM (existing code)
    self.cache_stats['misses'] += 1

    for attempt in range(max_retries):
        try:
            result = self.llm.generate(
                prompt=self._build_analysis_prompt(exercise_text, course_name, previous_text),
                temperature=0.1,
                json_mode=True
            )

            if result.success:
                analysis = self._parse_analysis_result(result.text, exercise_text)

                # ADD TO CACHE (NEW)
                if self.procedure_cache and analysis.procedures:
                    self.procedure_cache.add(
                        exercise_text=exercise_text,
                        topic=analysis.topic,
                        difficulty=analysis.difficulty,
                        variations=analysis.variations,
                        procedures=analysis.procedures,
                        confidence=analysis.confidence
                    )

                return analysis

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))
                continue
            raise

    return self._default_analysis_result()
```

### A.2 Cache Statistics Display

```python
# core/analyzer.py - Add to discover_topics_and_core_loops()

def discover_topics_and_core_loops(self, course_code: str, ...):
    """Analyze exercises with cache statistics tracking."""

    # ... existing code ...

    # NEW: Display cache statistics
    if self.procedure_cache and Config.PROCEDURE_CACHE_ENABLED:
        stats = self.cache_stats

        print(f"\n📊 Procedure Cache Statistics:")
        print(f"   Cache hits: {stats['hits']}")
        print(f"   Cache misses: {stats['misses']}")
        print(f"   Hit rate: {stats['hits'] / (stats['hits'] + stats['misses']) * 100:.1f}%")
        print(f"   💰 Saved ~{stats['hits']} LLM calls!")

    # ... existing code ...
```

---

**End of Plan**

*This plan provides a complete roadmap for implementing Option 3: Procedure Pattern Caching with precision, comprehensive testing, and clear success criteria.*
