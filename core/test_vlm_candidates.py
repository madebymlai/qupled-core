#!/usr/bin/env python3
"""
Test multiple VLM candidates in parallel on exercise extraction.

Usage:
    python test_vlm_candidates.py [--test NAME] [--pdf PATH] [--pages 1,2,3]

Examples:
    python test_vlm_candidates.py --test ex10_11     # Test Ex 10/11 split logic
    python test_vlm_candidates.py --test al_full     # Full AL exam
    python test_vlm_candidates.py --pdf custom.pdf --pages 1,2
"""

import argparse
import base64
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

from config import Config

# VLM candidates to test
CANDIDATES = [
    "anthropic/claude-3-haiku",             # Claude 3 Haiku
    "google/gemini-2.5-flash",              # Gemini 2.5 Flash
    "x-ai/grok-4.1-fast",                   # Grok 4.1 Fast (reasoning: optional)
    # "z-ai/glm-4.6v",                      # GLM-4.6V - too slow (180s)
    # "openai/gpt-5-mini",                  # GPT-5 Mini - too slow (80s)
    # "google/gemini-2.5-flash-preview",     # removed
    # "google/gemini-2.0-flash-001",         # removed
    # "qwen/qwen2.5-vl-72b-instruct",        # removed
    # "zhipu-ai/glm-4.1v-thinking-preview",  # older GLM
    # "anthropic/claude-sonnet-4",           # expensive
]

# Predefined test cases
TEST_DATA_DIR = Path("/home/laimk/git/qupled-cloud/test-data")

TEST_CASES = {
    # === KEY SPLIT TESTS ===
    # ADE Ex 3 = shared timing table (should NOT split)
    "shared_table": {
        "pdf": TEST_DATA_DIR / "ADE-EXAMS/Compito 2018-01-25 - TESTO.pdf",
        "pages": [3],  # Page with Ex 3 timing table
        "description": "ADE: Ex 3 timing table (should NOT split - all fill ONE table)",
        "expected": {
            "3|1": "STANDALONE (fill timing table t1-t10 for DFF-S, D-Latch, JKFF-D)",
        },
    },
    # AL Ex 3 should split into 3-4 parts (different skills)
    "al_split": {
        "pdf": TEST_DATA_DIR / "AL-EXAMS/EsempioAlgebraLineareSecondoCompitino19-20.pdf",
        "pages": [3],  # Page with Ex 3
        "description": "AL: Ex 3 should SPLIT (a-d are different skills)",
        "expected": {
            "3|1": "SPLIT into 3.1-3.4 (det, basis, diagonalize, orthonormal)",
        },
    },
    # === FULL EXAMS ===
    "al_full": {
        "pdf": TEST_DATA_DIR / "AL-EXAMS/EsempioAlgebraLineareSecondoCompitino19-20.pdf",
        "pages": [1, 2, 3],
        "description": "Full AL exam 19-20 (3 pages)",
    },
    "al_2012": {
        "pdf": TEST_DATA_DIR / "AL-EXAMS/20120612 - appello.pdf",
        "pages": [1, 2],
        "description": "AL exam 2012-06-12",
    },
    "physics_full": {
        "pdf": TEST_DATA_DIR / "PHYSICS-EXAMS/mechanics-thermo-ksu-2022.pdf",
        "pages": [1, 2, 3],
        "description": "Full physics exam (mechanics/thermo)",
    },
    "anatomy": {
        "pdf": TEST_DATA_DIR / "ANATOMY-EXAMS/anatomy-physiology-lakeshore.pdf",
        "pages": [1, 2],
        "description": "Anatomy exam",
    },
}

# Load prompts from exercise_scanner
from exercise_scanner import (
    EXERCISE_EXTRACTION_PROMPT,
    EXERCISE_EXTRACTION_SYSTEM,
    render_page_to_image,
)


def resize_image(image_bytes: bytes, max_size: int = 2048) -> bytes:
    """Resize image if either dimension exceeds max_size."""
    from PIL import Image
    import io

    img = Image.open(io.BytesIO(image_bytes))
    width, height = img.size

    if width <= max_size and height <= max_size:
        return image_bytes

    if width > height:
        new_width = max_size
        new_height = int(height * max_size / width)
    else:
        new_height = max_size
        new_width = int(width * max_size / height)

    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_model(
    model: str,
    content: list[dict],
    timeout: int = 120,
) -> dict[str, Any]:
    """Test a single model and return results."""
    start_time = time.time()

    try:
        # Build request payload
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": EXERCISE_EXTRACTION_SYSTEM},
                {"role": "user", "content": content},
            ],
            "temperature": 0.0,
            "max_tokens": 16000,
        }

        # Grok: reasoning off (faster, same results)
        if "grok" in model.lower():
            payload["reasoning"] = {"enabled": False}

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {Config.OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()

        result = response.json()
        text = result["choices"][0]["message"]["content"]
        elapsed = time.time() - start_time

        # Parse JSON
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if lines[-1] == "```" else "\n".join(lines[1:])
            text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                except json.JSONDecodeError as e2:
                    return {
                        "model": model,
                        "success": False,
                        "error": str(e2),
                        "raw": text[:1000],
                        "elapsed": elapsed,
                    }
            else:
                return {
                    "model": model,
                    "success": False,
                    "error": "No JSON found",
                    "raw": text[:1000],
                    "elapsed": elapsed,
                }

        exercises = data.get("exercises", [])

        # Extract exercise numbers for quick comparison
        ex_nums = [ex.get("exercise_number") for ex in exercises]

        return {
            "model": model,
            "success": True,
            "exercise_count": len(exercises),
            "exercise_numbers": ex_nums,
            "exercises": exercises,
            "elapsed": elapsed,
        }

    except requests.Timeout:
        return {
            "model": model,
            "success": False,
            "error": "Timeout",
            "elapsed": timeout,
        }
    except Exception as e:
        return {
            "model": model,
            "success": False,
            "error": str(e),
            "elapsed": time.time() - start_time,
        }


def print_results(results: list[dict], expected: dict = None, verbose: bool = False):
    """Print comparison of results from all models."""
    print("\n" + "=" * 80)
    print("VLM CANDIDATE COMPARISON")
    print("=" * 80)

    for r in results:
        model_name = r["model"].split("/")[-1]
        print(f"\n{'─' * 60}")
        print(f"MODEL: {model_name}")
        print(f"{'─' * 60}")

        if not r["success"]:
            print(f"  ERROR: {r['error']}")
            if "raw" in r:
                print(f"  Raw response:\n{r['raw'][:500]}")
            continue

        print(f"  Time: {r['elapsed']:.1f}s")
        print(f"  Exercises: {r['exercise_count']}")
        print(f"  Numbers: {r['exercise_numbers']}")

        if verbose:
            print("\n  DETAILS:")
            for ex in r["exercises"]:
                num = ex.get("exercise_number", "?")
                text = ex.get("text", "")[:100].replace("\n", " ")
                print(f"    [{num}] {text}...")

    # Summary comparison
    print("\n" + "=" * 80)
    print("SUMMARY - Exercise Numbers per Model")
    print("=" * 80)

    for r in results:
        if r["success"]:
            model_name = r["model"].split("/")[-1][:25]
            nums = ", ".join(str(n) for n in r["exercise_numbers"])
            print(f"  {model_name:<25} | {nums}")

    # Key tests based on expected values
    if expected:
        print("\n" + "=" * 80)
        print("KEY TESTS (based on expected)")
        print("=" * 80)

        for ex_key, expectation in expected.items():
            # Support "original|renumbered" format
            ex_alternatives = ex_key.split("|")
            should_split = "SPLIT" in expectation.upper()

            print(f"\n  Ex {ex_alternatives[0]}: {'Should SPLIT' if should_split else 'Should NOT split (standalone)'}")
            print(f"    Expected: {expectation}")
            if len(ex_alternatives) > 1:
                print(f"    (Also checking renumbered as: {', '.join(ex_alternatives[1:])})")

            for r in results:
                if not r["success"]:
                    continue

                model_name = r["model"].split("/")[-1][:25]
                ex_nums = [str(n) for n in r["exercise_numbers"]]

                # Check any of the alternative exercise numbers
                matched_num = None
                has_subs = False
                has_standalone = False

                for ex_num in ex_alternatives:
                    if any(n.startswith(f"{ex_num}.") for n in ex_nums):
                        has_subs = True
                        matched_num = ex_num
                        break
                    if ex_num in ex_nums:
                        has_standalone = True
                        matched_num = ex_num

                if should_split:
                    # We WANT splits
                    if has_subs:
                        subs = [n for n in ex_nums if n.startswith(f"{matched_num}.") or n == matched_num]
                        print(f"    {model_name:<25} | PASS - Split: {subs}")
                    elif has_standalone:
                        print(f"    {model_name:<25} | FAIL - Kept as standalone (Ex {matched_num})")
                    else:
                        print(f"    {model_name:<25} | N/A - Not found")
                else:
                    # We DON'T want splits
                    if has_subs:
                        subs = [n for n in ex_nums if n.startswith(f"{matched_num}.") or n == matched_num]
                        print(f"    {model_name:<25} | FAIL - Incorrectly split: {subs}")
                    elif has_standalone:
                        print(f"    {model_name:<25} | PASS - Kept as standalone (Ex {matched_num})")
                    else:
                        print(f"    {model_name:<25} | N/A - Not found")


def main():
    parser = argparse.ArgumentParser(description="Test VLM candidates on exercise extraction")
    parser.add_argument("--test", "-t", type=str, choices=list(TEST_CASES.keys()),
                        help="Predefined test case name")
    parser.add_argument("--pdf", type=Path, default=None,
                        help="Path to PDF file (overrides --test)")
    parser.add_argument("--pages", type=str, default=None,
                        help="Comma-separated page numbers (1-indexed)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed exercise text")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model names to test (overrides CANDIDATES)")
    parser.add_argument("--list", "-l", action="store_true",
                        help="List available test cases")
    parser.add_argument("--all-pages", "-a", action="store_true",
                        help="Test with all PDF pages instead of specific pages")
    parser.add_argument("--exclude", "-x", type=str, default=None,
                        help="Comma-separated model names to exclude (partial match)")
    args = parser.parse_args()

    # List test cases
    if args.list:
        print("Available test cases:")
        for name, tc in TEST_CASES.items():
            print(f"  {name:<20} - {tc['description']}")
        sys.exit(0)

    if not Config.OPENROUTER_API_KEY:
        print("ERROR: OPENROUTER_API_KEY not configured")
        sys.exit(1)

    # Determine PDF and pages
    if args.pdf:
        pdf_path = args.pdf
        pages = [int(p.strip()) for p in args.pages.split(",")] if args.pages else [1]
        description = f"Custom: {pdf_path.name}"
        expected = {}
    elif args.test:
        tc = TEST_CASES[args.test]
        pdf_path = tc["pdf"]
        pages = tc["pages"]
        description = tc["description"]
        expected = tc.get("expected", {})
    else:
        # Default to shared_table test
        tc = TEST_CASES["shared_table"]
        pdf_path = tc["pdf"]
        pages = tc["pages"]
        description = tc["description"]
        expected = tc.get("expected", {})

    # Override with all pages if flag set
    if args.all_pages:
        import fitz
        doc = fitz.open(pdf_path)
        pages = list(range(1, len(doc) + 1))
        doc.close()

    # Load models
    models = CANDIDATES
    if args.models:
        models = [m.strip() for m in args.models.split(",")]

    # Exclude models
    if args.exclude:
        excludes = [x.strip().lower() for x in args.exclude.split(",")]
        models = [m for m in models if not any(x in m.lower() for x in excludes)]

    print(f"TEST: {description}")
    print(f"PDF: {pdf_path}")
    print(f"Pages: {pages}")
    if expected:
        print("EXPECTED:")
        for ex_num, exp in expected.items():
            print(f"  Ex {ex_num}: {exp}")
    print(f"\nModels: {len(models)}")
    for m in models:
        print(f"  - {m}")

    # Render pages
    print("\nRendering pages...")
    page_images = []
    for page_num in pages:
        img_bytes = render_page_to_image(pdf_path, page_num, dpi=150)
        img_bytes = resize_image(img_bytes, max_size=2048)
        page_images.append(img_bytes)
    print(f"  Rendered {len(page_images)} pages")

    # Build content (shared across all models)
    flowchart_path = Path(__file__).parent / "assets" / "flowchart.png"
    content = [{"type": "text", "text": EXERCISE_EXTRACTION_PROMPT}]

    if flowchart_path.exists():
        with open(flowchart_path, "rb") as f:
            flowchart_bytes = f.read()
        flowchart_b64 = base64.b64encode(flowchart_bytes).decode("utf-8")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{flowchart_b64}"},
        })
        print(f"  Loaded flowchart")

    for img_bytes in page_images:
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })

    # Test all models in parallel
    print(f"\nTesting {len(models)} models in parallel...")
    results = []

    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        futures = {
            executor.submit(test_model, model, content): model
            for model in models
        }

        for future in as_completed(futures):
            model = futures[future]
            model_name = model.split("/")[-1]
            try:
                result = future.result()
                status = "OK" if result["success"] else f"FAIL: {result.get('error', 'unknown')}"
                print(f"  {model_name}: {status} ({result['elapsed']:.1f}s)")
                results.append(result)
            except Exception as e:
                print(f"  {model_name}: EXCEPTION: {e}")
                results.append({
                    "model": model,
                    "success": False,
                    "error": str(e),
                    "elapsed": 0,
                })

    # Sort results by model name for consistent output
    results.sort(key=lambda r: r["model"])

    # Print comparison
    print_results(results, expected=expected, verbose=args.verbose)


if __name__ == "__main__":
    main()
