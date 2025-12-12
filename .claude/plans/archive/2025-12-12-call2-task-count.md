# Call 3 Validation: Fallback for Single Sub-Question

## Summary
Add validation after Call 3: if only 1 sub-question returned when `has_sub_questions=true`, treat exercise as standalone. This prevents weird single-sub exercises like `[4.1]` when Call 3 fails to detect multiple subs.

## Root Cause
Exercise 4 has 5 `o` bullet sub-questions, but Call 3 returned only 1 because:
- `o` looks like Italian word "or", not a bullet marker
- LLM interpreted "calculate in these representations" as ONE task with conditions

## Files to Modify
- `/home/laimk/git/examina/core/exercise_splitter.py`

## Steps

### Step 1: Add validation after Call 3 in split_pdf_smart
After `_get_sub_start_markers_parallel`, filter out exercises where only 1 sub was found:

```python
# Call 3: Sub-question start markers (parallel, only for exercises with subs)
if boundaries_with_subs:
    exercises_for_call3 = [
        (b.number, full_text[b.start_pos:exercise_analysis[b.number].end_pos])
        for b in boundaries_with_subs
    ]
    explicit_subs = asyncio.run(
        _get_sub_start_markers_parallel(exercises_for_call3, second_pass_llm)
    )

    # Validation: if only 1 sub found, Call 3 likely failed - treat as standalone
    for ex_num in list(explicit_subs.keys()):
        if len(explicit_subs[ex_num]) == 1:
            logger.warning(f"Exercise {ex_num}: expected multiple subs, got 1 - treating as standalone")
            del explicit_subs[ex_num]
else:
    explicit_subs = {}
```

### Step 2: Update boundaries_with_subs after validation
The `boundaries_with_subs` list is used for Calls 4 and 5. After validation removes some exercises from `explicit_subs`, update the list:

```python
# Update boundaries_with_subs to only include exercises that still have subs
boundaries_with_subs = [b for b in boundaries_with_subs if b.number in explicit_subs]
```

### Step 3: Test
```bash
cd /home/laimk/git/examina && rm -rf data/cache/llm/*.json && python3 << 'EOF'
# Test with ADE 2020
# Exercise 4: Call 3 returns 1 → validation fails → standalone
# Expected: [4] instead of [4.1]
EOF
```

## Edge Cases
- Exercise with legitimately 1 sub-question → treated as standalone (acceptable loss)
- Exercise with 0 subs returned → already filtered by `if subs` in results_dict

## Expected Result
- Exercise 4: No more `[4.1]` single sub
- Either detected as 5 subs (if Call 3 improves) or treated as standalone `[4]`
