# Concept Naming Specificity Fix (COMPLETED)

## Summary
Fixed the analyzer to produce specific concept names instead of broad categories. Multiple root causes identified and fixed.

## Root Causes
1. CONTEXT RULES rule 4 said "USE EXISTING names over creating new ones" - caused generic name cascade
2. VARIATION section asked LLM to output `parent_name` - triggered `create_abstract_parents` merging
3. `content` and `variation_parameter` fields in JSON schema were unused cruft

## Files Modified
- `core/analyzer.py` - Multiple prompt changes

## Changes Made
1. Changed CONTEXT RULES rule 4 from "When in doubt, USE EXISTING names over creating new ones" to "Only reuse existing name if it's a close match, not a broad category"
2. Added `parent_context` parameter to `analyze_exercise` and `_build_analysis_prompt` methods
3. Removed VARIATION section entirely (including `parent_name`, `variation_parameter`)
4. Removed unused `content` field from JSON schema

## Commits
- `a93c2ed` - fix: prevent generic concept names by discouraging broad category reuse
- `aaf6992` - refactor: remove VARIATION section from analyzer prompt
- `fd07c1e` - refactor: remove unused content field from analyzer prompt

## Results
- Before: Generic names like `concurrent_programming`, `concurrency_control`
- After: Specific names like `Java Monitor Synchronization`, `semaphore_based_dining_philosophers`, `producer_consumer_problem_semaphore_solution`

## Testing
Re-analyzed PC course - 11 knowledge items now have specific technique+problem names
