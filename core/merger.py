"""
Post-processor for knowledge item merging.

Groups equivalent knowledge items by skill and picks canonical names.
Uses description-based approach for better accuracy.
Supports hierarchical categories to prevent sibling merging.
Supports active learning to reduce LLM calls by 70-90%.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from models.llm_manager import LLMManager

if TYPE_CHECKING:
    from core.active_learning import ActiveClassifier

logger = logging.getLogger(__name__)


def assign_category(
    item: dict,
    existing_categories: list[str],
    llm: LLMManager,
) -> tuple[str, bool]:
    """
    Assign item to existing category or create new one.

    Categories are discovered from content, not hardcoded.

    Args:
        item: Dict with 'name' and 'description'
        existing_categories: List of category names already in this course
        llm: LLMManager instance

    Returns:
        Tuple of (category_name, is_new)
    """
    if not existing_categories:
        # First item - generate category from description
        return _generate_category(item, llm), True

    categories_text = "\n".join(f"- {c}" for c in existing_categories)

    prompt = f"""Assign this item to a broad sub-topic.

Existing sub-topics:
{categories_text}

Item: {item.get('description', item.get('display_name', item.get('name', '')))}

Pick the best fitting broad sub-topic, or suggest NEW if none fit.
NEW must be broad (1-2 words), not a course name.

Return JSON: {{"category": "sub-topic name (lowercase)", "is_new": false}}
Or: {{"category": "new sub-topic name (lowercase)", "is_new": true}}"""

    try:
        response = llm.generate(
            prompt=prompt,
            model="deepseek-reasoner",
            system="You are organizing study materials.",
            temperature=0.0,
            json_mode=True,
        )

        if response and response.text:
            result = json.loads(response.text)
            category = result.get("category", existing_categories[0])
            is_new = result.get("is_new", False)

            # Validate existing category
            if not is_new and category not in existing_categories:
                # LLM hallucinated, pick closest or create new
                is_new = True

            return category, is_new

        return existing_categories[0], False

    except Exception as e:
        logger.warning(f"Category assignment failed: {e}")
        return existing_categories[0] if existing_categories else "General", False


def _generate_category(item: dict, llm: LLMManager) -> str:
    """Generate category for first item in course."""
    prompt = f"""What broad sub-topic does this item belong to?

Item: {item.get('description', item.get('display_name', item.get('name', '')))}

Return a broad sub-topic (1-2 words), NOT the course or subject name.

Return JSON: {{"category": "sub-topic name (lowercase)"}}"""

    try:
        response = llm.generate(
            prompt=prompt,
            model="deepseek-reasoner",
            system="You are organizing study materials.",
            temperature=0.0,
            json_mode=True,
        )

        if response and response.text:
            result = json.loads(response.text)
            return result.get("category", "General")

        return "General"

    except Exception as e:
        logger.warning(f"Category generation failed: {e}")
        return "General"


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

    # Format groups for prompt (use display_name if available for LLM readability)
    groups_text = "\n".join(
        f"{i + 1}. {g.get('display_name', g['name'])} - {g['description']}"
        for i, g in enumerate(existing_groups)
    )

    system = "You are a teacher organizing study materials."

    prompt = f"""Classify this item into an existing group or mark as NEW.

Existing groups:
{groups_text}

New item: {new_item['description']}

Same group = tests the **SAME** skill, would go on the same flashcard.
NEW = tests a **DIFFERENT** skill, needs separate study.

Return JSON: {{"group": <group_number>, "confidence": <0.0-1.0>}}
Or: {{"group": "NEW", "confidence": <0.0-1.0>}}"""

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
    active_classifier: ActiveClassifier | None = None,
) -> tuple[list[dict], list[tuple[int, int]]]:
    """
    Classify new items into existing groups using O(N) classification.

    Supports category-aware classification to prevent sibling merging:
    - Items are first assigned to a category
    - Classification happens only within the same category
    - Low confidence within category = sibling, don't merge

    When active_classifier is provided, uses ML predictions to skip LLM calls
    for high-confidence cases (70-90% reduction in LLM calls).

    Args:
        new_items: List of dicts with 'id', 'name', 'description', optional 'category'
        existing_groups: List of dicts with 'id', 'name', 'description', 'items', optional 'category'
        llm: LLMManager instance
        confidence_threshold: Minimum confidence to accept match
        active_classifier: Optional ActiveClassifier for ML-based predictions

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
            "display_name": g.get("display_name", g["name"]),  # Preserve for LLM
            "description": g["description"],
            "category": g.get("category"),
            "items": g.get("items", []).copy(),
        }
        for g in existing_groups
    ]
    group_by_id = {g["id"]: g for g in groups}

    # Get existing categories
    existing_categories = list(set(g["category"] for g in groups if g.get("category")))

    # LLM classify function for active learning fallback
    def llm_classify_fn(item: dict, candidate_groups: list[dict]) -> dict:
        return classify_item(item, candidate_groups, llm, confidence_threshold)

    # Process each new item
    for item in new_items:
        # Step 1: Assign category (if not already set)
        item_category = item.get("category")
        if not item_category:
            item_category, _ = assign_category(item, existing_categories, llm)
            item["category"] = item_category
            if item_category not in existing_categories:
                existing_categories.append(item_category)

        # Step 2: Filter groups to same category
        same_category_groups = [g for g in groups if g.get("category") == item_category]

        # Step 3: Classify within category (using active learning if available)
        if same_category_groups:
            if active_classifier:
                # Use active learning - may skip LLM calls
                al_result = active_classifier.classify(item, same_category_groups, llm_classify_fn)
                result = {
                    "is_new": al_result.is_new,
                    "group_id": al_result.group_id,
                    "confidence": al_result.confidence,
                }
            else:
                # Direct LLM call
                result = classify_item(item, same_category_groups, llm, confidence_threshold)
        else:
            result = {"is_new": True, "group_id": None, "confidence": 1.0}

        if result["is_new"]:
            # Create new group with this item (in this category)
            new_group = {
                "id": item["id"],
                "name": item["name"],
                "display_name": item.get("display_name", item["name"]),
                "description": item["description"],
                "category": item_category,
                "items": [item],
            }
            groups.append(new_group)
            group_by_id[new_group["id"]] = new_group
            logger.info(f"New group created: {item['name']} (category: {item_category})")
        else:
            # Add to existing group
            group_id = result["group_id"]
            group = group_by_id.get(group_id)
            if group:
                group["items"].append(item)
                changed_group_ids.add(group_id)
                assignments.append((item["id"], group_id))
                logger.info(
                    f"Item '{item['name']}' -> group '{group['name']}' "
                    f"(category: {item_category}, conf: {result['confidence']:.2f})"
                )

    # Regenerate name/description for changed groups
    for group_id in changed_group_ids:
        group = group_by_id.get(group_id)
        if not group or len(group["items"]) < 2:
            continue

        # Get canonical name (keep snake_case for storage/dedup)
        item_names = [item["name"] for item in group["items"]]
        group["name"] = get_canonical_name(item_names, llm)

        # Regenerate description
        item_descriptions = [item["description"] for item in group["items"] if item.get("description")]
        if item_descriptions:
            group["description"] = regenerate_description(item_descriptions, llm)

        logger.info(f"Regenerated group: {group['name']} ({len(group['items'])} items)")

    # Log active learning stats if used
    if active_classifier:
        stats = active_classifier.get_stats()
        logger.info(
            f"Active learning stats: "
            f"LLM calls={stats['llm_calls']}, "
            f"predictions={stats['predictions']}, "
            f"transitive={stats['transitive_inferences']}, "
            f"LLM rate={stats['llm_call_rate']:.1%}"
        )

    return groups, assignments


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


