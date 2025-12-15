"""
Detection utilities for identifying exercise patterns and transformation procedures.

This module provides intelligent detection logic for:
- Numbered points and sections in exercises
- Transformation/conversion keywords
- Procedure type classification
"""

import re
from typing import List, Dict, Any


def detect_numbered_points(text: str) -> List[Dict[str, Any]]:
    """
    Detect numbered points/sections in exercise text.

    Patterns to detect:
    - "1.", "2.", "3." (numeric with period)
    - "a)", "b)", "c)" (letters with parenthesis)
    - "1-", "2-", "3-" (numeric with dash)
    - Roman numerals: "I.", "II.", "III."
    - Italian: "Punto 1", "Esercizio 1.a"

    Args:
        text: Exercise text to analyze

    Returns:
        List of dicts with {
            'point_number': int or str,
            'text': str,  # text of this point
            'start_pos': int,
            'end_pos': int,
            'pattern_type': str  # type of numbering pattern
        }
    """
    points = []

    # Pattern definitions (ordered by priority)
    patterns = [
        # Italian patterns
        (r"(?:^|\n)\s*Punto\s+(\d+)\.?\s*[:\-]?\s*", "italian_punto"),
        (r"(?:^|\n)\s*Esercizio\s+(\d+)\.([a-z0-9])\b", "italian_esercizio_sub"),
        (r"(?:^|\n)\s*Esercizio\s+(\d+)\b", "italian_esercizio"),
        # Numeric with period: "1.", "2.", "3."
        (r"(?:^|\n)\s*(\d+)\.\s+(?=[A-Z])", "numeric_period"),
        # Letter with parenthesis: "a)", "b)", "c)"
        (r"(?:^|\n)\s*([a-z])\)\s+", "letter_paren"),
        # Numeric with dash: "1-", "2-", "3-"
        (r"(?:^|\n)\s*(\d+)\-\s+", "numeric_dash"),
        # Roman numerals with period: "I.", "II.", "III.", "IV."
        (r"(?:^|\n)\s*((?:I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII)+)\.\s+", "roman"),
        # Nested numbering: "1.1", "1.2", "2.1"
        (r"(?:^|\n)\s*(\d+)\.(\d+)\s+", "nested_numeric"),
    ]

    # Find all matches for each pattern
    all_matches = []

    for pattern, pattern_type in patterns:
        for match in re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE):
            start_pos = match.start()
            end_pos = match.end()

            # Check if this match overlaps with an existing one
            overlaps = False
            for existing in all_matches:
                existing_end = existing["start_pos"] + len(existing["match_text"])
                if start_pos < existing_end and end_pos > existing["start_pos"]:
                    overlaps = True
                    break

            if overlaps:
                continue  # Skip overlapping matches (earlier patterns take priority)

            # Extract the point number/identifier
            if pattern_type == "italian_esercizio_sub":
                point_number = f"{match.group(1)}.{match.group(2)}"
            elif pattern_type == "nested_numeric":
                point_number = f"{match.group(1)}.{match.group(2)}"
            else:
                point_number = match.group(1)

            # Convert roman numerals to integers
            if pattern_type == "roman":
                point_number = _roman_to_int(point_number)
            # Convert numeric strings to integers
            elif pattern_type in [
                "numeric_period",
                "numeric_dash",
                "italian_punto",
                "italian_esercizio",
            ]:
                try:
                    point_number = int(point_number)
                except ValueError:
                    pass  # Keep as string if conversion fails

            all_matches.append(
                {
                    "point_number": point_number,
                    "start_pos": start_pos,
                    "pattern_type": pattern_type,
                    "match_text": match.group(0),
                }
            )

    # Sort matches by position
    all_matches.sort(key=lambda x: x["start_pos"])

    # Extract text for each point
    for i, match_data in enumerate(all_matches):
        start_pos = match_data["start_pos"]

        # End position is the start of next point, or end of text
        if i + 1 < len(all_matches):
            end_pos = all_matches[i + 1]["start_pos"]
        else:
            end_pos = len(text)

        # Extract text (excluding the numbering prefix)
        match_end = start_pos + len(match_data["match_text"])
        point_text = text[match_end:end_pos].strip()

        points.append(
            {
                "point_number": match_data["point_number"],
                "text": point_text,
                "start_pos": start_pos,
                "end_pos": end_pos,
                "pattern_type": match_data["pattern_type"],
            }
        )

    return points


def _roman_to_int(roman: str) -> int:
    """Convert Roman numeral to integer.

    Args:
        roman: Roman numeral string (e.g., "IV", "XII")

    Returns:
        Integer value
    """
    roman_values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}

    roman = roman.upper()
    total = 0
    prev_value = 0

    for char in reversed(roman):
        value = roman_values.get(char, 0)
        if value < prev_value:
            total -= value
        else:
            total += value
        prev_value = value

    return total


def detect_transformation_keywords(text: str, language: str = "en") -> List[Dict[str, Any]]:
    """
    Detect transformation/conversion procedures in text.

    Keywords to detect:
    English:
    - "transform X to Y", "convert X into Y"
    - "derive Y from X", "obtain Y from X"
    - "equivalent Y", "Y equivalent to X"
    - "X → Y", "X to Y conversion"

    Italian:
    - "trasformare X in Y", "convertire X in Y"
    - "ricavare Y da X", "ottenere Y partendo da X"
    - "Y equivalente", "equivalente a X"
    - "trasformazione da X a Y"

    Args:
        text: Text to analyze
        language: Language code ('en' or 'it')

    Returns:
        List of dicts with {
            'type': 'transformation',
            'source_format': str,  # e.g., "mealy"
            'target_format': str,  # e.g., "moore"
            'matched_pattern': str,
            'confidence': float  # 0.0-1.0
        }
    """
    results = []

    # Known format types (case-insensitive)
    format_keywords = {
        "en": [
            "mealy",
            "moore",
            "dfa",
            "nfa",
            "pda",
            "turing",
            "fsm",
            "automaton",
            "regular expression",
            "regex",
            "grammar",
            "cfg",
            "cnf",
            "dnf",
            "state machine",
            "circuit",
            "boolean",
            "truth table",
            "karnaugh",
        ],
        "it": [
            "mealy",
            "moore",
            "dfa",
            "nfa",
            "pda",
            "turing",
            "automa",
            "espressione regolare",
            "regex",
            "grammatica",
            "circuito",
            "booleano",
            "tabella di verità",
            "karnaugh",
            "mappa di karnaugh",
        ],
    }

    # Transformation patterns by language
    patterns = {
        "en": [
            # "transform X to/into Y"
            (
                r"transform(?:ing)?\s+(?:the\s+)?(\w+(?:\s+\w+)?)\s+(?:to|into)\s+(?:a\s+|an\s+)?(\w+(?:\s+\w+)?)",
                0.9,
            ),
            # "convert X to/into Y"
            (
                r"convert(?:ing)?\s+(?:the\s+)?(\w+(?:\s+\w+)?)\s+(?:to|into)\s+(?:a\s+|an\s+)?(\w+(?:\s+\w+)?)",
                0.9,
            ),
            # "derive Y from X"
            (
                r"deriv(?:e|ing)\s+(?:a\s+|an\s+)?(\w+(?:\s+\w+)?)\s+from\s+(?:the\s+)?(\w+(?:\s+\w+)?)",
                0.85,
            ),
            # "obtain Y from X"
            (
                r"obtain(?:ing)?\s+(?:a\s+|an\s+)?(\w+(?:\s+\w+)?)\s+from\s+(?:the\s+)?(\w+(?:\s+\w+)?)",
                0.85,
            ),
            # "X equivalent" - X is the target format
            (r"(?:^|\s)(\w+)\s+equivalent(?:\s+(?:to|automaton|machine))?", 0.8),
            # "X → Y" or "X -> Y"
            (r"(\w+(?:\s+\w+)?)\s*(?:→|->)\s*(\w+(?:\s+\w+)?)", 0.95),
            # "X to Y conversion"
            (r"(\w+(?:\s+\w+)?)\s+to\s+(\w+(?:\s+\w+)?)\s+conversion", 0.9),
        ],
        "it": [
            # "trasformare X in Y"
            (
                r"trasform(?:are|azione|ando|a)\s+(?:l\'|il\s+)?(?:automa\s+)?(?:di\s+)?(\w+)\s+in\s+(?:un\s+)?(?:automa\s+)?(?:di\s+)?(\w+)",
                0.9,
            ),
            # "convertire X in Y"
            (
                r"convert(?:ire|endo|ito)\s+(?:l\'|il\s+)?(?:automa\s+)?(?:di\s+)?(\w+)\s+in\s+(?:un\s+)?(?:automa\s+)?(?:di\s+)?(\w+)",
                0.9,
            ),
            # "ricavare Y da X"
            (
                r"ricav(?:are|ando|a)\s+(?:un\s+)?(?:automa\s+)?(?:di\s+)?(\w+)\s+dall?\'?(?:automa\s+)?(?:di\s+)?(\w+)",
                0.85,
            ),
            # "ottenere Y partendo da X"
            (
                r"otten(?:ere|endo)\s+(?:un\s+|una\s+)?(\w+(?:\s+\w+)?)\s+partendo\s+da(?:l|ll\'|lla)?\s+(\w+(?:\s+\w+)?)",
                0.85,
            ),
            # "Y equivalente a X"
            (r"(\w+(?:\s+\w+)?)\s+equivalente\s+(?:a|al|alla)?\s+(\w+(?:\s+\w+)?)", 0.8),
            # "equivalente a X"
            (r"equivalente\s+(?:a|al|alla)\s+(\w+(?:\s+\w+)?)", 0.75),
            # "X → Y" or "X -> Y"
            (r"(\w+(?:\s+\w+)?)\s*(?:→|->)\s*(\w+(?:\s+\w+)?)", 0.95),
            # "trasformazione da X a Y"
            (
                r"trasformazione\s+da(?:l|ll\'|lla)?\s+(\w+(?:\s+\w+)?)\s+a(?:l|ll\'|lla)?\s+(\w+(?:\s+\w+)?)",
                0.9,
            ),
        ],
    }

    # Get patterns for specified language
    lang_patterns = patterns.get(language, patterns["en"])
    lang_formats = format_keywords.get(language, format_keywords["en"])

    # Search for transformation patterns
    for pattern, base_confidence in lang_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            groups = match.groups()

            # Handle different pattern structures
            if len(groups) == 2:
                source_format = groups[1] if "from" in pattern or "da" in pattern else groups[0]
                target_format = groups[0] if "from" in pattern or "da" in pattern else groups[1]
            elif len(groups) == 1:
                # Single format (e.g., "equivalent X")
                source_format = None
                target_format = groups[0]
            else:
                continue

            # Clean up format names
            source_format = source_format.strip().lower() if source_format else None
            target_format = target_format.strip().lower() if target_format else None

            # Validate formats against known keywords
            source_valid = source_format is None or any(kw in source_format for kw in lang_formats)
            target_valid = target_format is None or any(kw in target_format for kw in lang_formats)

            # Adjust confidence based on format recognition
            confidence = base_confidence
            if source_valid and target_valid:
                confidence = min(1.0, confidence + 0.05)
            elif not source_valid and not target_valid:
                confidence = max(0.5, confidence - 0.2)

            results.append(
                {
                    "type": "transformation",
                    "source_format": source_format,
                    "target_format": target_format,
                    "matched_pattern": match.group(0),
                    "confidence": confidence,
                    "position": match.start(),
                }
            )

    # Sort by position and remove duplicates
    results.sort(key=lambda x: x["position"])

    # Remove duplicate transformations (keep highest confidence)
    seen = {}
    unique_results = []
    for result in results:
        key = (result["source_format"], result["target_format"])
        if key not in seen or result["confidence"] > seen[key]["confidence"]:
            seen[key] = result

    unique_results = list(seen.values())
    unique_results.sort(key=lambda x: x.get("position", 0))

    # Remove position field before returning
    for result in unique_results:
        result.pop("position", None)

    return unique_results


def classify_procedure_type(point_text: str, language: str = "en") -> str:
    """
    Classify the type of procedure described in a point.

    Categories:
    - "design" - designing from scratch
    - "transformation" - converting between formats
    - "verification" - proving/checking properties
    - "minimization" - optimization procedures
    - "analysis" - analyzing/calculating properties
    - "implementation" - circuit/code implementation

    Uses keyword matching and heuristics.

    Args:
        point_text: Text of the exercise point
        language: Language code ('en' or 'it')

    Returns:
        Procedure type category
    """
    text_lower = point_text.lower()

    # Define keyword patterns by category and language
    keywords = {
        "en": {
            "transformation": ["transform", "convert", "derive", "equivalent", "obtain", "→", "->"],
            "design": [
                "design",
                "construct",
                "build",
                "create",
                "develop",
                "draw",
                "sketch",
                "implement from scratch",
            ],
            "verification": [
                "verify",
                "prove",
                "check",
                "validate",
                "demonstrate",
                "show that",
                "confirm",
                "test whether",
            ],
            "minimization": [
                "minimize",
                "optimize",
                "reduce",
                "simplify",
                "implication table",
                "state reduction",
                "equivalent states",
            ],
            "analysis": [
                "analyze",
                "calculate",
                "determine",
                "find",
                "compute",
                "identify",
                "evaluate",
                "count",
            ],
            "implementation": [
                "implement",
                "code",
                "program",
                "circuit",
                "logic gates",
                "truth table",
                "hardware",
            ],
        },
        "it": {
            "transformation": ["trasform", "convert", "ricav", "equivalente", "otten", "→", "->"],
            "design": ["disegn", "costrui", "crea", "svilupp", "progett", "realizza", "schema"],
            "verification": [
                "verific",
                "dimostr",
                "controlla",
                "valid",
                "mostra che",
                "conferma",
                "testa se",
            ],
            "minimization": [
                "minimizza",
                "ottimizza",
                "riduci",
                "semplifica",
                "tabella delle implicazioni",
                "riduzione degli stati",
                "stati equivalenti",
            ],
            "analysis": [
                "analizza",
                "calcola",
                "determina",
                "trova",
                "individua",
                "valuta",
                "conta",
            ],
            "implementation": [
                "implementa",
                "codifica",
                "programma",
                "circuito",
                "porte logiche",
                "tabella di verità",
                "hardware",
            ],
        },
    }

    # Get keywords for language
    lang_keywords = keywords.get(language, keywords["en"])

    # Score each category
    scores = {category: 0 for category in lang_keywords.keys()}

    for category, kw_list in lang_keywords.items():
        for keyword in kw_list:
            if keyword in text_lower:
                # Weight longer keywords more (they're more specific)
                weight = len(keyword.split())

                # Boost weight for high-priority keywords
                if category == "verification" and keyword in ["verify", "verific"]:
                    weight *= 2
                elif category == "design" and keyword in ["design", "disegn"]:
                    weight *= 1.5

                scores[category] += weight

    # Return category with highest score, or 'analysis' as default
    if max(scores.values()) == 0:
        return "analysis"  # Default if no keywords match

    return max(scores.items(), key=lambda x: x[1])[0]


def extract_exercise_metadata(text: str, language: str = "en") -> Dict[str, Any]:
    """
    Extract comprehensive metadata from exercise text.

    This is a convenience function that combines all detection utilities
    to provide a complete analysis of exercise structure.

    Args:
        text: Exercise text
        language: Language code ('en' or 'it')

    Returns:
        Dict with:
            - 'points': List of detected numbered points
            - 'transformations': List of detected transformations
            - 'procedure_types': Dict mapping point numbers to procedure types
            - 'is_multi_step': Boolean indicating if exercise has multiple steps
            - 'step_count': Number of detected steps
    """
    # Detect numbered points
    points = detect_numbered_points(text)

    # Detect transformations in full text
    transformations = detect_transformation_keywords(text, language)

    # Classify procedure type for each point
    procedure_types = {}
    for point in points:
        point_num = point["point_number"]
        proc_type = classify_procedure_type(point["text"], language)
        procedure_types[point_num] = proc_type

    # Determine if multi-step
    is_multi_step = len(points) > 1

    return {
        "points": points,
        "transformations": transformations,
        "procedure_types": procedure_types,
        "is_multi_step": is_multi_step,
        "step_count": len(points),
    }
