"""
Post-processor for knowledge item validation.

Uses LLM to validate extracted items via coherence-based filtering.
Also detects synonyms for deduplication.
"""
import json
import logging
from typing import Any

from models.llm_manager import LLMManager

logger = logging.getLogger(__name__)


def detect_synonyms(
    names: list[str],
    llm: LLMManager,
) -> list[list[str]]:
    """
    Use LLM to detect synonym groups among knowledge item names.

    Returns list of groups, where each group contains synonymous names.
    Example: [["Mealy machine", "Mealy automaton"], ["DFS", "depth-first search"]]

    Args:
        names: List of knowledge item names to check
        llm: LLMManager instance

    Returns:
        List of synonym groups (each group has 2+ names that mean the same thing)
    """
    if len(names) < 2:
        return []

    # Dedupe names for the prompt
    unique_names = list(set(names))
    if len(unique_names) < 2:
        return []

    prompt = f"""Given these knowledge item names from a course, identify groups of DUPLICATES (different names for the SAME concept).

Names:
{chr(10).join(f"- {name}" for name in unique_names)}

MERGE AGGRESSIVELY - group items if they refer to the same underlying concept:

1. MERGE if same base concept with different suffixes:
   - "mips_calculation" = "mips_calculation_procedure" = "mips_calculation_method"
   - "boolean_function" = "boolean_function_minimization" (if minimization is THE main procedure)

2. MERGE abbreviations with full names:
   - "DFS" = "depth_first_search"
   - "CPI" = "cycles_per_instruction"

3. MERGE alternate phrasings:
   - "flip_flop_output_waveform" = "flip_flop_waveform_analysis"
   - "decimal_to_binary" = "binary_conversion_from_decimal"

4. DO NOT MERGE if truly different concepts:
   - "d_flip_flop" vs "d_latch" (different components)
   - "boolean_function" vs "karnaugh_map" (different concepts, even if related)

Return a JSON array of arrays, where each inner array contains names that should be merged.
Only include groups with 2+ names. Names not in any group are unique concepts.

Return ONLY the JSON array, no explanation. Return [] if no duplicates found."""

    try:
        response = llm.generate(
            prompt=prompt,
            temperature=0.0,
            json_mode=True,
        )

        if response and response.text:
            groups = json.loads(response.text)
            # Validate structure
            if isinstance(groups, list) and all(isinstance(g, list) for g in groups):
                # Filter to groups with 2+ items
                valid_groups = [g for g in groups if len(g) >= 2]
                if valid_groups:
                    logger.info(f"Detected {len(valid_groups)} synonym groups: {valid_groups}")
                return valid_groups

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse synonym detection response: {e}")
    except Exception as e:
        logger.warning(f"Synonym detection failed: {e}")

    return []


def filter_and_organize_knowledge(
    llm: LLMManager,
    items: list[dict],
    existing_parents: list[str],
    existing_topics: list[str] | None = None,  # Deprecated, ignored
) -> dict:
    """
    Filter extracted knowledge items using LLM.

    Uses coherence-based filtering:
    - Filter out outliers (context/scenario items, not academic concepts)
    - Normalize parent names to existing ones

    Args:
        llm: LLMManager instance
        items: List of extracted knowledge items with name, type, parent_name
        existing_parents: List of parent names already in the course
        existing_topics: DEPRECATED - ignored, kept for backward compatibility

    Returns:
        {
            "valid_items": [...],      # Items that passed filtering
            "filtered_items": [...],   # Items removed (with idx for debugging)
            "filtered_indices": [...], # Indices of filtered items
            "inferred_topics": [],     # DEPRECATED - always empty
        }
    """
    if not items:
        return {
            "valid_items": [],
            "filtered_items": [],
            "filtered_indices": [],
            "inferred_topics": [],
        }

    # Add index to each item for reliable matching (LLM may modify names)
    indexed_items = [{"idx": i, **item} for i, item in enumerate(items)]

    prompt = f"""Analyze these extracted knowledge items and filter out outliers.

ITEMS:
{json.dumps(indexed_items, indent=2, ensure_ascii=False)}

EXISTING PARENT CONCEPTS IN COURSE:
{json.dumps(existing_parents, ensure_ascii=False) if existing_parents else "[]"}

TASK:
1. AGGRESSIVELY Flag OUTLIERS using these tests:

   TEST A - Academic vs Context: "Is this item the SUBJECT of study, or just the SETTING/CONTEXT of a word problem?"
   TEST B - Textbook test: "Would this appear as a chapter heading or section title in this course's textbook?"
   TEST C - Lecture test: "Would a professor put this on a lecture slide as a concept to teach?"
   TEST D - Abstraction test: "Is this a general theoretical concept, or a specific real-world instance?"

   If ANY test fails → FILTER the item out

   Keep ONLY items where ALL tests pass

2. Normalize PARENTS - if an item suggests a parent similar to an existing one, use the existing name

CRITICAL: Each item has an "idx" field. You MUST return this idx unchanged for matching.

RETURN JSON:
{{
    "valid_indices": [0, 2, 4],
    "filtered_indices": [1, 3]
}}

NOTE: Just return the indices of valid vs filtered items.
"""

    try:
        response = llm.generate(
            prompt=prompt,
            temperature=0.3,
            json_mode=True,
        )

        if response and response.text:
            result = json.loads(response.text)

            # Get valid indices from result
            valid_indices = set(result.get("valid_indices", range(len(items))))
            filtered_indices = result.get("filtered_indices", [])

            # If LLM only returned filtered_indices, compute valid_indices
            if "valid_indices" not in result and filtered_indices:
                filtered_set = set(filtered_indices)
                valid_indices = {i for i in range(len(items)) if i not in filtered_set}

            # Build valid_items (original items that passed)
            valid_items = [items[i] for i in range(len(items)) if i in valid_indices]

            # Build filtered_items for logging
            filtered_items = [
                {"name": items[i].get("name", "unknown"), "idx": i}
                for i in filtered_indices
                if i < len(items)
            ]

            # Log filtered items for debugging
            if filtered_items:
                for item in filtered_items:
                    logger.info(f"Filtered: {item.get('name')} (idx={item.get('idx')})")

            return {
                "valid_items": valid_items,
                "filtered_items": filtered_items,
                "filtered_indices": list(filtered_indices),
                "inferred_topics": [],  # Deprecated
            }
        else:
            logger.warning("Post-processor got empty response, returning all items")
            return {
                "valid_items": items,
                "filtered_items": [],
                "filtered_indices": [],
                "inferred_topics": [],
            }

    except json.JSONDecodeError as e:
        logger.error(f"Post-processor JSON parse error: {e}")
        return {
            "valid_items": items,
            "filtered_items": [],
            "filtered_indices": [],
            "inferred_topics": [],
        }
    except Exception as e:
        logger.error(f"Post-processor error: {e}")
        return {
            "valid_items": items,
            "filtered_items": [],
            "filtered_indices": [],
            "inferred_topics": [],
        }
