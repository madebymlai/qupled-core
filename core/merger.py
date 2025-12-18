"""
Post-processor for knowledge item merging.

Groups equivalent knowledge items by skill and picks canonical names.
Uses description-based approach for better accuracy.
"""

import json
import logging

from models.llm_manager import LLMManager
from core.analyzer import LEARNING_APPROACHES

logger = logging.getLogger(__name__)


def classify_item(
    new_item: dict,
    existing_groups: list[dict],
    llm: LLMManager,
    confidence_threshold: float = 0.7,
) -> dict:
    """
    Classify a new item into an existing group or mark as NEW.

    O(1) per item instead of O(N) pairwise comparisons.

    Args:
        new_item: Dict with 'name' and 'description'
        existing_groups: List of dicts with 'id', 'name', 'description'
        llm: LLMManager instance
        confidence_threshold: Minimum confidence to accept match (default 0.7)

    Returns:
        {
            "group_id": int or None (if NEW),
            "is_new": bool,
            "confidence": float (0.0-1.0)
        }
    """
    if not existing_groups:
        return {"group_id": None, "is_new": True, "confidence": 1.0}

    # Format groups for prompt
    groups_text = "\n".join(
        f"{i + 1}. {g['name']} - {g['description']}"
        for i, g in enumerate(existing_groups)
    )

    system = "You are a teacher organizing study materials."

    prompt = f"""Classify this item into an existing group or mark as NEW.

Existing groups:
{groups_text}

New item: {new_item['description']}

Same group = tests the **SAME** skill, would go on the same flashcard.
NEW = tests a **DIFFERENT** skill, needs separate study.

Return JSON: {{"group": 1, "confidence": 0.95}}
Or if new concept: {{"group": "NEW", "confidence": 0.95}}"""

    try:
        response = llm.generate(
            prompt=prompt,
            model="deepseek-reasoner",
            system=system,
            temperature=0.0,
            json_mode=True,
        )

        if not response or not response.text:
            logger.warning("Empty response from classify_item")
            return {"group_id": None, "is_new": True, "confidence": 0.5}

        result = json.loads(response.text)
        group = result.get("group")
        confidence = float(result.get("confidence", 0.5))

        # Handle NEW
        if group == "NEW" or group is None:
            return {"group_id": None, "is_new": True, "confidence": confidence}

        # Handle group match
        group_idx = int(group) - 1  # Convert 1-indexed to 0-indexed
        if 0 <= group_idx < len(existing_groups):
            # Check confidence threshold
            if confidence >= confidence_threshold:
                return {
                    "group_id": existing_groups[group_idx]["id"],
                    "is_new": False,
                    "confidence": confidence,
                }
            else:
                # Low confidence, treat as new
                logger.info(f"Low confidence {confidence} < {confidence_threshold}, treating as new")
                return {"group_id": None, "is_new": True, "confidence": confidence}

        return {"group_id": None, "is_new": True, "confidence": confidence}

    except Exception as e:
        logger.warning(f"classify_item failed: {e}")
        return {"group_id": None, "is_new": True, "confidence": 0.5}


def classify_items(
    new_items: list[dict],
    existing_groups: list[dict],
    llm: LLMManager,
    confidence_threshold: float = 0.7,
) -> tuple[list[dict], list[tuple[int, int]]]:
    """
    Classify new items into existing groups using O(N) classification.

    Processes items incrementally, then regenerates names/descriptions
    for groups that changed at end of batch.

    Args:
        new_items: List of dicts with 'id', 'name', 'description'
        existing_groups: List of dicts with 'id', 'name', 'description', 'items'
        llm: LLMManager instance
        confidence_threshold: Minimum confidence to accept match

    Returns:
        Tuple of:
        - Updated groups list (with new items added and descriptions regenerated)
        - List of (item_id, group_id) assignments for merging
    """
    if not new_items:
        return existing_groups, []

    # Track which groups changed
    changed_group_ids: set[int] = set()
    assignments: list[tuple[int, int]] = []

    # Build working copy of groups
    groups = [
        {
            "id": g["id"],
            "name": g["name"],
            "description": g["description"],
            "items": g.get("items", []).copy(),
        }
        for g in existing_groups
    ]
    group_by_id = {g["id"]: g for g in groups}

    # Process each new item
    for item in new_items:
        result = classify_item(item, groups, llm, confidence_threshold)

        if result["is_new"]:
            # Create new group with this item
            new_group = {
                "id": item["id"],  # Use item's ID as group ID
                "name": item["name"],
                "description": item["description"],
                "items": [item],
            }
            groups.append(new_group)
            group_by_id[new_group["id"]] = new_group
            logger.info(f"New group created: {item['name']}")
        else:
            # Add to existing group
            group_id = result["group_id"]
            group = group_by_id.get(group_id)
            if group:
                group["items"].append(item)
                changed_group_ids.add(group_id)
                assignments.append((item["id"], group_id))
                logger.info(f"Item '{item['name']}' -> group '{group['name']}' (conf: {result['confidence']:.2f})")

    # Regenerate name/description for changed groups
    for group_id in changed_group_ids:
        group = group_by_id.get(group_id)
        if not group or len(group["items"]) < 2:
            continue

        # Get canonical name
        item_names = [item["name"] for item in group["items"]]
        group["name"] = get_canonical_name(item_names, llm)

        # Regenerate description
        item_descriptions = [item["description"] for item in group["items"] if item.get("description")]
        if item_descriptions:
            group["description"] = regenerate_description(item_descriptions, llm)

        logger.info(f"Regenerated group: {group['name']} ({len(group['items'])} items)")

    return groups, assignments


def group_items(
    items: list[dict],
    llm: LLMManager,
) -> list[list[int]]:
    """
    Group items that describe the same task using anonymous IDs.

    Uses descriptions only - names are hidden from LLM to prevent bias.
    Returns indices into the original items list.

    Args:
        items: List of dicts with key: description
        llm: LLMManager instance

    Returns:
        List of index groups, e.g., [[0, 2], [1, 3, 4]]
    """
    if len(items) < 2:
        return []

    # Check if we have descriptions
    has_descriptions = any(item.get("description") for item in items)
    if not has_descriptions:
        logger.info("No descriptions provided, skipping grouping")
        return []

    # Build prompt with anonymous IDs - LLM never sees names
    items_text = [f"- Item {i + 1}: {item['description']}" for i, item in enumerate(items)]

    system = "You are a teacher grouping concepts for students to study."

    prompt = f"""Which test the same concept? Take your time and think carefully.

{chr(10).join(items_text)}

Same concept = would go on the same flashcard/study topic.

Return JSON: {{"reasoning": "...", "groups": [[1, 2]]}}
Return {{"reasoning": "...", "groups": []}} if all different."""

    try:
        logger.info(f"Grouping {len(items)} items by description")
        response = llm.generate(prompt=prompt, model="deepseek-reasoner", system=system)

        if not response or not response.text:
            logger.warning("Empty response from grouping LLM call")
            return []

        result = json.loads(response.text)

        # Handle {"groups": [...]} format
        groups = result.get("groups", [])
        if not groups:
            logger.info("No groups detected")
            return []

        # Convert 1-indexed to 0-indexed
        result_groups: list[list[int]] = []
        for group in groups:
            # Handle {"items": [...], "reason": "..."} format
            if isinstance(group, dict):
                group_items = group.get("items", [])
                reason = group.get("reason", "")
            elif isinstance(group, list):
                group_items = group
                reason = ""
            else:
                continue
            if len(group_items) < 2:
                continue
            # Convert to 0-indexed and validate
            indices = [
                idx - 1 for idx in group_items if isinstance(idx, int) and 0 < idx <= len(items)
            ]
            if len(indices) >= 2:
                result_groups.append(indices)
                logger.info(f"Group: items {indices}" + (f" - {reason}" if reason else ""))

        return result_groups

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse grouping response: {e}")
        return []
    except Exception as e:
        logger.warning(f"Grouping LLM call failed: {e}")
        return []


def regenerate_description(
    descriptions: list[str],
    llm: LLMManager,
) -> str:
    """
    Write the most representative description from multiple descriptions.

    Uses R1 with textbook editor mindset to write concise chapter subtitles.
    Stabilizes at ~100 chars and condenses verbose descriptions.

    Args:
        descriptions: List of description strings from merged items
        llm: LLMManager instance

    Returns:
        Most general description (falls back to longest on error)
    """
    if not descriptions:
        return ""
    if len(descriptions) == 1:
        return descriptions[0]

    # Filter empty descriptions
    descriptions = [d for d in descriptions if d]
    if not descriptions:
        return ""
    if len(descriptions) == 1:
        return descriptions[0]

    prompt = f"""Pick the most representative skill description:

{chr(10).join(f"- {d}" for d in descriptions)}

Return JSON: {{"description": "..."}}"""

    try:
        response = llm.generate(
            prompt=prompt,
            system="You are a textbook editor.",
            model="deepseek-reasoner",
            temperature=0.0,
            json_mode=True,
        )
        if response and response.text:
            result = json.loads(response.text)
            return result.get("description", max(descriptions, key=len))
        return max(descriptions, key=len)
    except Exception as e:
        logger.warning(f"Description selection failed: {e}")
        return max(descriptions, key=len)


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

    approaches_desc = "\n".join(f"- {k} = {v}" for k, v in LEARNING_APPROACHES.items())
    approaches_keys = "|".join(LEARNING_APPROACHES.keys())

    prompt = f"""Based on these exercises, pick the most appropriate learning approach.

Exercises:
{chr(10).join(f'- "{ex}"' for ex in exercises[:6])}

Options: {approaches_keys}
{approaches_desc}

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
) -> list[tuple[str, str, list[str], str | None, str | None]]:
    """
    Merge equivalent knowledge items by grouping by skill and picking canonical names.

    Args:
        items: List of dicts with keys: name, type, exercises, learning_approach, description
        llm: LLMManager instance

    Returns:
        List of (canonical_name, type, member_names, learning_approach, description) tuples
    """
    if len(items) < 2:
        return []

    # Normalize input format
    normalized_items: list[dict] = []
    for item in items:
        if isinstance(item, tuple):
            normalized_items.append(
                {
                    "name": item[0],
                    "type": item[1],
                    "exercises": [],
                    "learning_approach": None,
                    "description": None,
                }
            )
        elif isinstance(item, dict):
            normalized_items.append(
                {
                    "name": item.get("name", ""),
                    "type": item.get("type", "key_concept"),
                    "exercises": item.get("exercises", []),
                    "learning_approach": item.get("learning_approach"),
                    "description": item.get("description"),
                }
            )

    # Use new function to group by skill
    groups = group_items_by_skill(normalized_items, llm)

    if not groups:
        return []

    # Process each group
    all_results: list[tuple[str, str, list[str], str | None, str | None]] = []

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

        # Regenerate unified description from merged items
        group_descriptions = [item.get("description") for item in group_items if item.get("description")]
        description = regenerate_description(group_descriptions, llm) if group_descriptions else None

        all_results.append((canonical_name, item_type, group_names, learning_approach, description))
        logger.info(
            f"Skill group: {group_names} -> canonical='{canonical_name}', approach={learning_approach}"
        )

    return all_results
