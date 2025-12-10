# Concept Naming Specificity Fix

## Summary
Fix the analyzer to produce specific concept names instead of broad categories. Root cause: CONTEXT RULES rule 4 said "USE EXISTING names over creating new ones" which caused the first generic name to cascade to all subsequent exercises.

## Files Modified
- `core/analyzer.py` - Changed CONTEXT RULES rule 4 (line 377) + added parent_context parameter

## Changes Made
1. Changed CONTEXT RULES rule 4 from "When in doubt, USE EXISTING names over creating new ones" to "Only reuse existing name if it's a close match, not a broad category"
2. Added `parent_context` parameter to `analyze_exercise` and `_build_analysis_prompt` methods for sub-question context

## Results
- Before: Generic names like `concurrent_programming`, `concurrency_control`
- After: Specific names like `Java Monitor Synchronization`, `Semaphore-based Synchronization Problems`, `dining_philosophers_semaphore_solution`

## Testing
Re-analyzed PC course with force=True - new analyses produce specific technique-based names
