"""
Post-processor for knowledge item merging.

Groups equivalent knowledge items by skill and picks canonical names.
"""
import json
import logging

from models.llm_manager import LLMManager

logger = logging.getLogger(__name__)


def group_items_by_skill(
    items: list[dict],
    llm: LLMManager,
) -> list[list[dict]]:
    """
    Group knowledge items that test the same skill.

    LLM sees item names + exercises to judge skill equivalence.
    Names help distinguish different skills, exercises help catch synonyms.

    Args:
        items: List of dicts with keys: name, exercises (list of exercise snippets)
        llm: LLMManager instance

    Returns:
        List of groups, where each group is a list of item dicts that test the same skill.
        Example: [[{name: "fsm_table", ...}, {name: "transition_table", ...}], ...]
    """
    if len(items) < 2:
        return []

    # Dedupe by name, keeping first occurrence
    seen_names: set[str] = set()
    unique_items: list[dict] = []
    for item in items:
        name = item.get("name", "")
        if name and name not in seen_names:
            seen_names.add(name)
            unique_items.append(item)

    if len(unique_items) < 2:
        return []

    # Check if we have exercise context
    has_exercises = any(item.get("exercises") for item in unique_items)
    if not has_exercises:
        logger.info("No exercises provided, skipping skill grouping")
        return []

    # Build prompt with item names + exercises
    # Names help distinguish different skills, exercises help catch synonyms
    items_text = []
    for item in unique_items:
        name = item.get("name", "unknown")
        item_text = f"- {name}"
        if item.get("exercises"):
            exercise_snippets = [f'    "{ex}"' for ex in item["exercises"]]
            if exercise_snippets:
                item_text += "\n  Exercises:\n" + "\n".join(exercise_snippets)
        items_text.append(item_text)

    prompt = f"""Identify which items test the EXACT SAME SKILL based on their exercises.

Items:
{chr(10).join(items_text)}

SAME SKILL (should group):
- Both require the EXACT SAME technique to solve
- A student who masters one has automatically mastered the other
- A single flashcard could teach both equally well

DIFFERENT SKILLS (should NOT group):
- One asks to EXPLAIN/DESCRIBE, another asks to CALCULATE/APPLY
- Mastering one gives only partial mastery of the other
- They would need separate study sessions

Return JSON: {{"groups": [["item_name_1", "item_name_2"], ["item_name_3", "item_name_4"]]}}
Return {{"groups": []}} if no items test the same skill."""

    # Build name -> item lookup
    name_to_item = {item.get("name", ""): item for item in unique_items}

    try:
        logger.info(f"Grouping {len(unique_items)} items by skill")
        response = llm.generate(prompt=prompt, temperature=0.0, json_mode=True)

        if not response or not response.text:
            logger.warning("Empty response from grouping LLM call")
            return []

        result = json.loads(response.text)

        # Handle both {"groups": [...]} and direct [...] format
        if isinstance(result, dict) and "groups" in result:
            groups = result["groups"]
        elif isinstance(result, list):
            groups = result
        else:
            logger.warning(f"Unexpected grouping response format: {type(result)}")
            return []

        if not groups:
            logger.info("No skill groups detected")
            return []

        # Convert item names back to actual items
        result_groups: list[list[dict]] = []
        for group in groups:
            if not isinstance(group, list) or len(group) < 2:
                continue
            group_items = [name_to_item[name] for name in group if name in name_to_item]
            if len(group_items) >= 2:
                result_groups.append(group_items)
                logger.info(f"Skill group: {[item['name'] for item in group_items]}")

        return result_groups

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse grouping response: {e}")
        return []
    except Exception as e:
        logger.warning(f"Grouping LLM call failed: {e}")
        return []


def get_canonical_name(
    group_names: list[str],
    llm: LLMManager,
) -> str:
    """
    Pick the most descriptive canonical name from a group of equivalent names.

    Args:
        group_names: List of equivalent knowledge item names
        llm: LLMManager instance

    Returns:
        The chosen canonical name (falls back to first name on error)
    """
    if not group_names:
        return ""
    if len(group_names) == 1:
        return group_names[0]

    prompt = f"""Pick the most descriptive name from: {json.dumps(group_names)}

Return JSON: {{"canonical": "chosen_name"}}"""

    try:
        response = llm.generate(prompt=prompt, temperature=0.0, json_mode=True)

        if response and response.text:
            result = json.loads(response.text)
            canonical_name = result.get("canonical", group_names[0])
            # Validate canonical is in group
            if canonical_name not in group_names:
                canonical_name = group_names[0]
            return canonical_name
        else:
            return group_names[0]

    except Exception as e:
        logger.warning(f"Canonical selection failed: {e}, using first name")
        return group_names[0]


def get_learning_approach(
    exercises: list[str],
    llm: LLMManager,
) -> str | None:
    """
    Derive learning approach from exercises.

    Args:
        exercises: List of exercise text snippets
        llm: LLMManager instance

    Returns:
        Learning approach string or None
    """
    if not exercises:
        return None

    prompt = f"""Based on these exercises, pick the most appropriate learning approach.

Exercises:
{chr(10).join(f'- "{ex}"' for ex in exercises[:6])}

Options: procedural, conceptual, factual, analytical
- procedural = exercise asks to APPLY steps/calculate/solve
- conceptual = exercise asks to EXPLAIN/compare/reason why
- factual = exercise asks to RECALL specific facts/definitions
- analytical = exercise asks to ANALYZE/evaluate/critique

Return JSON: {{"learning_approach": "procedural"}}"""

    try:
        response = llm.generate(prompt=prompt, temperature=0.0, json_mode=True)

        if response and response.text:
            result = json.loads(response.text)
            approach = result.get("learning_approach")
            valid_approaches = ["procedural", "conceptual", "factual", "analytical"]
            if approach in valid_approaches:
                return approach
        return None

    except Exception as e:
        logger.warning(f"Learning approach derivation failed: {e}")
        return None


def merge_items(
    items: list[tuple[str, str]] | list[dict],
    llm: LLMManager,
) -> list[tuple[str, str, list[str], str | None]]:
    """
    Merge equivalent knowledge items by grouping by skill and picking canonical names.

    Args:
        items: List of dicts with keys: name, type, exercises, learning_approach
        llm: LLMManager instance

    Returns:
        List of (canonical_name, type, member_names, learning_approach) tuples
    """
    if len(items) < 2:
        return []

    # Normalize input format
    normalized_items: list[dict] = []
    for item in items:
        if isinstance(item, tuple):
            normalized_items.append({
                "name": item[0],
                "type": item[1],
                "exercises": [],
                "learning_approach": None,
            })
        elif isinstance(item, dict):
            normalized_items.append({
                "name": item.get("name", ""),
                "type": item.get("type", "key_concept"),
                "exercises": item.get("exercises", []),
                "learning_approach": item.get("learning_approach"),
            })

    # Use new function to group by skill
    groups = group_items_by_skill(normalized_items, llm)

    if not groups:
        return []

    # Process each group
    all_results: list[tuple[str, str, list[str], str | None]] = []

    for group_items in groups:
        group_names = [item["name"] for item in group_items]

        # Get canonical name
        canonical_name = get_canonical_name(group_names, llm)

        # Get learning approach from combined exercises
        combined_exercises = []
        for item in group_items:
            combined_exercises.extend(item.get("exercises", []))
        learning_approach = get_learning_approach(combined_exercises, llm)

        # Get type from first item
        item_type = group_items[0].get("type", "key_concept")

        all_results.append((canonical_name, item_type, group_names, learning_approach))
        logger.info(f"Skill group: {group_names} -> canonical='{canonical_name}', approach={learning_approach}")

    return all_results


# Deprecated alias for backward compatibility
detect_synonyms = merge_items
