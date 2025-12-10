# Sub-Detection Refactor: Boundary-Based Approach

## Summary

Remove unreliable `sub_triggers` feature and replace with boundary-based sub-question detection. Instead of relying on DeepSeek to detect "ambiguity" (which it fails at), we get parent `end_marker` from Sonnet first, then find subs within those boundaries. This eliminates the ambiguity problem entirely.

## Current Flow (Problem)

```
1. DeepSeek: patterns (exercise_pattern, sub_patterns, sub_triggers)
2. Find ALL markers (parent + sub) using patterns
3. Filter subs using sub_triggers ← BROKEN (DeepSeek doesn't understand ambiguity)
4. Build hierarchy
5. Sonnet: end_marker + context_summary
```

## New Flow

```
1. DeepSeek: patterns (exercise_pattern, sub_patterns) - NO sub_triggers
2. Find parent markers only
3. Sonnet call 1: end_marker for parents
4. Find subs within [parent_start, parent_end_marker] boundaries
5. Build hierarchy (rough sub text)
6. Sonnet call 2: end_marker for subs + context_summary for parents
```

## Files to Modify

- `/home/laimk/git/examina/core/exercise_splitter.py` - main changes

## Steps

### Step 1: Update LLM prompt - remove sub_triggers

Location: `_detect_pattern_with_llm()` (~line 244)

Changes:
- Remove `sub_triggers` from prompt text
- Remove `sub_triggers` from JSON output format
- Update `MarkerPattern` dataclass to remove `sub_triggers` field

### Step 2: Remove sub_triggers filtering logic

Location: `_find_all_markers()` (~line 888-953)

Changes:
- Remove `trigger_regexes` building logic
- Remove trigger phrase checking in sub-marker loop
- Simplify to just find parent markers (sub detection moves later)

### Step 3: Create new function `_get_parent_end_markers()`

New function for Sonnet call 1.

```python
def _get_parent_end_markers(
    parent_markers: List[Marker],
    full_text: str,
    llm_manager: "LLMManager",
) -> Dict[str, str]:
    """Get end_marker for each parent exercise.

    Returns dict mapping parent number to end_marker text.
    """
```

- Takes list of parent markers
- Extracts text for each parent (from marker to next parent or end)
- Asks Sonnet for end_marker for each
- Returns mapping of number → end_marker

### Step 4: Create new function `_find_sub_markers_in_boundaries()`

```python
def _find_sub_markers_in_boundaries(
    parent_markers: List[Marker],
    parent_end_positions: Dict[str, int],  # from end_marker
    sub_patterns: List[str],
    full_text: str,
) -> List[Marker]:
    """Find sub-markers within parent boundaries.

    Only matches sub_patterns between parent start and parent end_marker position.
    """
```

- For each parent, find end position from end_marker
- Apply sub_patterns only within [parent_start, end_position]
- Return list of sub markers

### Step 5: Update `_get_second_pass_results()`

Location: ~line 614

Changes:
- Rename or keep as is (now only called after hierarchy built)
- Still returns end_marker for subs + context_summary for parents
- No changes to logic, just when it's called

### Step 6: Update main `split_pdf_smart()` flow

Location: ~line 1480

New flow:
```python
# Step 1: Detect patterns (no sub_triggers)
detection = _detect_pattern_with_llm(text_sample, llm_manager)

# Step 2: Find parent markers only
parent_markers = _find_parent_markers(full_text, detection.pattern)

# Step 3: Sonnet call 1 - get parent end_markers
parent_end_markers = _get_parent_end_markers(parent_markers, full_text, second_pass_llm)
parent_end_positions = _find_end_marker_positions(full_text, parent_end_markers)

# Step 4: Find subs within boundaries
sub_markers = _find_sub_markers_in_boundaries(
    parent_markers, parent_end_positions, detection.pattern.sub_patterns, full_text
)

# Step 5: Combine and build hierarchy
all_markers = sorted(parent_markers + sub_markers, key=lambda m: m.start_position)
hierarchy = _build_hierarchy(all_markers, full_text)

# Step 6: Sonnet call 2 - end_marker for subs + context_summary
if second_pass_llm:
    results = _get_second_pass_results(hierarchy, second_pass_llm)
    _apply_second_pass_results(hierarchy, results)

# Step 7: Expand to flat list
exercises = _expand_exercises(hierarchy, ...)
```

### Step 7: Clean up unused code

- Remove `sub_triggers` from `MarkerPattern` dataclass
- Remove trigger-related helper code
- Update any tests that reference sub_triggers

## Edge Cases

- Parent with no subs: Works - no subs found within boundary, stays standalone
- Parent with text after subs: Works - Sonnet call 2 trims last sub properly
- Ambiguous patterns (both numbered): Works - boundary prevents confusion
- No sub_patterns returned: Works - skip sub detection entirely

## Dependencies

- Step 3 depends on Step 2 (need parent markers)
- Step 4 depends on Step 3 (need end positions)
- Step 5 depends on Step 4 (need all markers)
- Step 6 depends on Step 5 (need hierarchy)

## Testing

1. Run existing test PDFs through new flow
2. Verify sub-questions detected correctly
3. Verify ambiguous patterns (like `ambiguous_numbered` test) now work
4. Verify text-after-sub trimmed properly
