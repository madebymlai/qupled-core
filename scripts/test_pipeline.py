#!/usr/bin/env python3
"""
Pipeline Test Suite for examina-core.

Tests parsing, exercise splitting, and knowledge item extraction
across diverse exam PDFs from multiple academic domains.

Usage:
    python test_pipeline.py --smoke              # 1 PDF per course (~2 min)
    python test_pipeline.py --course ADE         # All PDFs in one course
    python test_pipeline.py --all                # All 43 PDFs (~15-30 min)
    python test_pipeline.py --pdf path/to.pdf    # Single PDF
    python test_pipeline.py --golden             # Regression tests with exact counts
    python test_pipeline.py --help               # Full usage guide
"""

import argparse
import json
import signal
import sys
import textwrap
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

# Add examina to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.pdf_processor import PDFProcessor, PDFContent
from core.exercise_splitter import ExerciseSplitter, Exercise
from core.analyzer import ExerciseAnalyzer
from core.post_processor import group_items_by_skill, get_canonical_name


def generate_item_description(name: str, exercises: list[dict], llm) -> str:
    """Generate skill-only description from exercises."""
    if not exercises:
        return ""

    exercises_text = []
    for ex in exercises[:6]:
        if ex.get("is_sub"):
            exercises_text.append(ex.get("text", ""))
        else:
            exercises_text.append(ex.get("context", "") or ex.get("text", ""))

    is_single = len(exercises_text) == 1
    exercise_word = "exercise" if is_single else "exercises"
    this_these = "this" if is_single else "these"

    prompt = f"""Describe the skill tested by {this_these} {exercise_word}.

{"Exercise" if is_single else "Exercises"}:
{chr(10).join(f"- {t}" for t in exercises_text)}

Write a description that:
- Includes the specific concept tested
- Excludes facts, data, scenarios, or question format
- Would fit any exercise testing this specific concept

**Be concise.**
**NO preambles about skill or exercise.**

Return JSON: {{"description": "..."}}"""

    try:
        response = llm.generate(prompt=prompt, temperature=0.0, json_mode=True)
        result = json.loads(response.text)
        return result.get("description", exercises_text[0][:100] if exercises_text else "")
    except Exception:
        return exercises_text[0][:100] if exercises_text else ""


def group_items_by_description(items: list[dict], llm) -> list[list[int]]:
    """Group items by skill using descriptions only (anonymous IDs)."""
    if len(items) < 2:
        return []

    items_text = [f"- Item {i+1}: {item['description']}" for i, item in enumerate(items)]

    prompt = f"""Which describe the same task?

{chr(10).join(items_text)}

Same task = **SAME** topic AND **SAME** action.
Different topics are NEVER the same task, even with same action.

Return JSON: {{"groups": [[1, 2]]}}
Return {{"groups": []}} if all different."""

    try:
        response = llm.generate(prompt=prompt, temperature=0.0, json_mode=True)
        result = json.loads(response.text)
        groups = result.get("groups", [])
        return [[idx - 1 for idx in group] for group in groups if len(group) >= 2]
    except Exception:
        return []
from models.llm_manager import LLMManager

# =============================================================================
# Configuration
# =============================================================================

TEST_DATA_PATH = Path("/home/laimk/git/examina-cloud/test-data")
TEST_RESULTS_PATH = Path(__file__).parent.parent / "test-results"

COURSES = {
    "ADE-EXAMS": "Architettura degli Elaboratori",
    "AL-EXAMS": "Algebra Lineare",
    "ANATOMY-EXAMS": "Anatomy and Physiology",
    "MACROECONOMICS-EXAMS": "Macroeconomics",
    "ORGANIC-CHEM-EXAMS": "Organic Chemistry",
    "PC-EXAMS": "Programmazione Concorrente",
    "PHYSICS-EXAMS": "Physics 101",
    "SO-EXAMS": "Sistemi Operativi",
}

GOLDEN_TESTS_FILE = Path(__file__).parent / "golden-tests.yaml"

def load_golden_tests() -> tuple[list[dict], int]:
    """Load golden tests from YAML config file."""
    if not GOLDEN_TESTS_FILE.exists():
        print(f"Warning: {GOLDEN_TESTS_FILE} not found, using empty list")
        return [], 1

    with open(GOLDEN_TESTS_FILE) as f:
        config = yaml.safe_load(f)

    tolerance = config.get("tolerance", 1)
    tests = config.get("tests", [])
    return tests, tolerance

VALID_APPROACHES = ["procedural", "conceptual", "factual", "analytical", "hybrid"]

# =============================================================================
# ANSI Colors
# =============================================================================

class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    @classmethod
    def disable(cls):
        cls.GREEN = cls.RED = cls.YELLOW = cls.CYAN = cls.DIM = cls.BOLD = cls.RESET = ""


def green(text: str) -> str:
    return f"{Colors.GREEN}{text}{Colors.RESET}"

def red(text: str) -> str:
    return f"{Colors.RED}{text}{Colors.RESET}"

def yellow(text: str) -> str:
    return f"{Colors.YELLOW}{text}{Colors.RESET}"

def cyan(text: str) -> str:
    return f"{Colors.CYAN}{text}{Colors.RESET}"

def dim(text: str) -> str:
    return f"{Colors.DIM}{text}{Colors.RESET}"

def bold(text: str) -> str:
    return f"{Colors.BOLD}{text}{Colors.RESET}"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TestResult:
    """Result of testing a single PDF."""
    pdf_path: str
    status: str = ""  # PASS, FAIL, ERROR
    error_stage: str = ""  # parse, split, analyze, full
    pages: int = 0
    language: str = ""
    exercises: int = 0
    sub_questions: int = 0
    parents: int = 0
    with_solutions: int = 0
    error: Optional[str] = None
    warnings: list = field(default_factory=list)
    duration: float = 0.0
    exercise_details: list = field(default_factory=list)
    skill_groups: list = field(default_factory=list)  # Merge results


@dataclass
class TestSummary:
    """Summary of all test results."""
    total: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    warnings: list = field(default_factory=list)
    results: list = field(default_factory=list)
    duration: float = 0.0


# =============================================================================
# Pipeline Runners
# =============================================================================

class PipelineTester:
    """Runs pipeline tests on PDFs."""

    def __init__(self, lang: str = "en", verbose: bool = False, quiet: bool = False, debug: bool = False):
        self.lang = lang
        self.verbose = verbose
        self.quiet = quiet
        self.debug = debug
        self.processor = PDFProcessor()
        self.llm = None
        self.splitter = None
        self.analyzer = None
        self._interrupted = False

    def _init_llm(self):
        """Lazy init LLM (expensive)."""
        if self.llm is None:
            self.llm = LLMManager(provider="deepseek", quiet=not self.debug)
            self.splitter = ExerciseSplitter()

    def _init_analyzer(self):
        """Lazy init analyzer."""
        self._init_llm()
        if self.analyzer is None:
            self.analyzer = ExerciseAnalyzer(llm_manager=self.llm, language=self.lang)

    def test_parse(self, pdf_path: Path) -> TestResult:
        """Stage 1: Test PDF parsing."""
        result = TestResult(pdf_path=str(pdf_path))
        start = time.time()

        try:
            pdf_content = self.processor.process_pdf(pdf_path)

            # Structural assertions
            assert pdf_content.total_pages > 0, "PDF has no pages"
            full_text = "".join(p.text for p in pdf_content.pages)
            assert len(full_text) > 100, "PDF text too short"

            result.pages = pdf_content.total_pages
            result.status = "PASS"

        except AssertionError as e:
            result.status = "FAIL"
            result.error_stage = "parse"
            result.error = str(e)
        except Exception as e:
            result.status = "ERROR"
            result.error_stage = "parse"
            result.error = f"Parse failed: {str(e)}"

        result.duration = time.time() - start
        return result

    def test_split(self, pdf_path: Path, course_name: str) -> TestResult:
        """Stage 1-2: Test PDF parsing + exercise splitting."""
        result = TestResult(pdf_path=str(pdf_path))
        start = time.time()

        try:
            self._init_llm()

            # Parse
            pdf_content = self.processor.process_pdf(pdf_path)
            result.pages = pdf_content.total_pages

            # Split
            exercises = self.splitter.split_pdf_smart(pdf_content, course_name, self.llm)

            # Structural assertions
            assert len(exercises) >= 1, "No exercises found"
            assert all(ex.text.strip() for ex in exercises), "Empty exercise text"

            # Count results
            parents = [e for e in exercises if not e.is_sub_question]
            subs = [e for e in exercises if e.is_sub_question]
            with_sol = [e for e in exercises if e.solution]

            result.exercises = len(exercises)
            result.parents = len(parents)
            result.sub_questions = len(subs)
            result.with_solutions = len(with_sol)
            result.status = "PASS"

            # Warnings
            generic_count = sum(1 for e in exercises if e.exercise_number and
                               (e.exercise_number.startswith("exercise_") or
                                e.exercise_number.startswith("question_")))
            if generic_count > 0:
                result.warnings.append(f"{generic_count} exercises with generic numbers")

            # Exercise details for verbose output
            for ex in exercises:
                result.exercise_details.append({
                    "number": ex.exercise_number,
                    "is_sub": ex.is_sub_question,
                    "parent": ex.parent_exercise_number,
                    "has_solution": bool(ex.solution),
                    "text_preview": self._truncate(ex.text, 100),
                    "text_full": ex.text,  # Full text for analyzer
                    "context": getattr(ex, "parent_context", "") or "",
                })

        except AssertionError as e:
            result.status = "FAIL"
            result.error_stage = "split"
            result.error = str(e)
        except Exception as e:
            result.status = "ERROR"
            result.error_stage = "split"
            result.error = f"Split failed: {str(e)}"

        result.duration = time.time() - start
        return result

    def test_analyze(self, pdf_path: Path, course_name: str) -> TestResult:
        """Stage 1-3: Test full pipeline including analysis (KI names, no grouping)."""
        result = self.test_split(pdf_path, course_name)
        if result.status != "PASS":
            return result

        start = time.time()

        try:
            self._init_analyzer()

            # Analyze each exercise to get KI names
            for ex in result.exercise_details:
                is_sub = ex.get("is_sub", False)
                if not is_sub and ex.get("context"):
                    text_for_analysis = ex.get("context", "")
                else:
                    text_for_analysis = ex.get("text_full", ex.get("text_preview", ""))

                analysis = self.analyzer.analyze_exercise(
                    exercise_text=text_for_analysis,
                    course_name=course_name,
                    exercise_context=None,
                    is_sub_question=is_sub
                )

                ki_name = None
                if analysis.knowledge_items:
                    ki_name = analysis.knowledge_items[0].name
                ex["ki_name"] = ki_name or f"unknown_{ex.get('number', '?')}"

            result.status = "PASS"

        except Exception as e:
            result.status = "ERROR"
            result.error_stage = "analyze"
            result.error = f"Analyze failed: {str(e)}"

        result.duration += time.time() - start
        return result

    def test_full(self, pdf_path: Path, course_name: str) -> TestResult:
        """Stage 1-4: Full pipeline including analysis + merge/skill grouping."""
        result = self.test_analyze(pdf_path, course_name)
        if result.status != "PASS":
            return result

        start = time.time()

        try:
            self._init_llm()

            # KI names already populated by test_analyze()
            # Group by knowledge_item name, collect exercise snippets
            ki_exercises: dict[str, list[dict]] = {}
            for ex in result.exercise_details:
                ki_name = ex.get("ki_name", "unknown")
                if ki_name not in ki_exercises:
                    ki_exercises[ki_name] = []

                # Build exercise snippet for merger
                # Sub-questions: text only (no parent context to avoid bias)
                # Standalone: context only (summary)
                if ex.get("is_sub"):
                    snippet = ex.get("text_preview", "")
                else:
                    snippet = ex.get("context", "") or ex.get("text_preview", "")

                ki_exercises[ki_name].append({
                    "number": ex.get("number"),
                    "is_sub": ex.get("is_sub"),
                    "context": ex.get("context", ""),
                    "text": ex.get("text_preview", ""),
                    "snippet": snippet,
                })

            # Build items with descriptions (new approach)
            items = []
            for name, exs in ki_exercises.items():
                description = generate_item_description(name, exs, self.llm)
                items.append({
                    "name": name,
                    "description": description,
                    "exercises": exs,
                })

            if len(items) >= 2:
                # Run skill grouping with descriptions (anonymous IDs)
                group_indices = group_items_by_description(items, self.llm)

                # Format skill groups with exercise context
                for indices in group_indices:
                    group_items_list = [items[i] for i in indices]
                    group_names = [item["name"] for item in group_items_list]
                    canonical = get_canonical_name(group_names, self.llm)

                    members = []
                    for item in group_items_list:
                        members.append({
                            "name": item["name"],
                            "description": item["description"],
                            "exercises": item["exercises"],
                        })

                    result.skill_groups.append({
                        "canonical": canonical,
                        "members": members,
                    })

            result.status = "PASS"

        except Exception as e:
            result.status = "ERROR"
            result.error_stage = "full"
            result.error = f"Full pipeline failed: {str(e)}"

        result.duration += time.time() - start
        return result

    def _truncate(self, text: str, max_len: int = 80) -> str:
        """Truncate text showing start...end."""
        text = text.replace("\n", " ").strip()
        if len(text) <= max_len:
            return text
        half = (max_len - 5) // 2
        return f"{text[:half]}...{text[-half:]}"

    def _get_course_name(self, folder: str) -> str:
        """Get course name from folder."""
        return COURSES.get(folder, folder)

    def _get_course_folder(self, pdf_path: Path) -> str:
        """Extract course folder from PDF path."""
        for folder in COURSES.keys():
            if folder in str(pdf_path):
                return folder
        return pdf_path.parent.name


# =============================================================================
# Test Runner
# =============================================================================

class TestRunner:
    """Coordinates test execution and output."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.tester = PipelineTester(
            lang=args.lang,
            verbose=args.verbose,
            quiet=args.quiet,
            debug=args.debug,
        )
        self.summary = TestSummary()
        self._interrupted = False
        self._start_time = None
        self._times = []  # Track PDF processing times for ETA
        self._total_llm_hits = 0
        self._total_llm_misses = 0

        # Set up interrupt handler
        signal.signal(signal.SIGINT, self._handle_interrupt)

    def _handle_interrupt(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        self._interrupted = True
        print(yellow("\n\nInterrupted! Saving partial results..."))

    def run(self) -> int:
        """Run tests and return exit code."""
        self._start_time = time.time()

        # Collect PDFs to test
        pdfs = self._collect_pdfs()
        if not pdfs:
            print(red("No PDFs found to test!"))
            return 1

        if self.args.dry_run:
            self._print_dry_run(pdfs)
            return 0

        # Run tests
        total = len(pdfs)
        for i, (pdf_path, course_folder) in enumerate(pdfs):
            if self._interrupted:
                break

            course_name = self.tester._get_course_name(course_folder)
            self._print_progress(i, total, pdf_path)

            # Accumulate and reset cache stats for per-PDF tracking
            if self.tester.llm:
                stats = self.tester.llm.get_cache_stats()
                self._total_llm_hits += stats['hits']
                self._total_llm_misses += stats['misses']
                self.tester.llm.reset_cache_stats()

            # Run appropriate test level
            if self.args.parse:
                result = self.tester.test_parse(pdf_path)
            elif self.args.full:
                result = self.tester.test_full(pdf_path, course_name)
            elif self.args.analyze:
                result = self.tester.test_analyze(pdf_path, course_name)
            else:  # Default: split
                result = self.tester.test_split(pdf_path, course_name)

            self._record_result(result)
            self._times.append(result.duration)

            if not self.args.quiet:
                self._print_result(result)

        # Capture final PDF's LLM stats
        if self.tester.llm:
            stats = self.tester.llm.get_cache_stats()
            self._total_llm_hits += stats['hits']
            self._total_llm_misses += stats['misses']

        # Print summary
        self.summary.duration = time.time() - self._start_time
        self._print_summary()

        # Auto-save last 3 runs
        self._save_recent()

        # Save results permanently if requested
        if self.args.save:
            self._save_results()

        # Save failed list for --rerun-failed
        self._save_failed_list()

        return 0 if self.summary.failed == 0 and self.summary.errors == 0 else 1

    def run_golden(self) -> int:
        """Run golden regression tests."""
        self._start_time = time.time()
        print(bold("=== GOLDEN REGRESSION TESTS ===\n"))

        golden_tests, tolerance = load_golden_tests()
        if not golden_tests:
            print(red("No golden tests found!"))
            return 1

        all_passed = True

        for test in golden_tests:
            pdf_path = TEST_DATA_PATH / test["path"]
            if not pdf_path.exists():
                print(red(f"MISSING: {test['path']}"))
                result = TestResult(pdf_path=str(pdf_path), status="ERROR", error="File not found")
                self._record_result(result)
                all_passed = False
                continue

            course_folder = self.tester._get_course_folder(pdf_path)
            course_name = self.tester._get_course_name(course_folder)

            print(f"Testing: {test['path']}")
            result = self.tester.test_split(pdf_path, course_name)

            if result.status != "PASS":
                print(red(f"  FAIL: {result.error}"))
                self._record_result(result)
                all_passed = False
                continue

            # Compare counts with tolerance
            expected = test["expected"]
            actual = {
                "total": result.exercises,
                "parents": result.parents,
                "subs": result.sub_questions,
                "with_solutions": result.with_solutions,
            }

            errors = []
            for key in expected:
                exp_val = expected[key]
                act_val = actual[key]
                diff = abs(exp_val - act_val)
                if diff > tolerance:
                    errors.append(f"{key}: expected {exp_val}, got {act_val}")

            if errors:
                print(red(f"  FAIL: {', '.join(errors)}"))
                result.status = "FAIL"
                result.error = ", ".join(errors)
                all_passed = False
            else:
                print(green(f"  PASS: total={actual['total']}, parents={actual['parents']}, subs={actual['subs']}"))

            self._record_result(result)

        print()
        self.summary.duration = time.time() - self._start_time

        # Auto-save last 3 runs
        self._save_recent()

        if all_passed:
            print(green("All golden tests passed!"))
            return 0
        else:
            print(red("Some golden tests failed!"))
            return 1

    def _collect_pdfs(self) -> list:
        """Collect PDFs based on run mode."""
        pdfs = []

        if self.args.pdf:
            # Single PDF mode
            pdf_path = Path(self.args.pdf)
            if pdf_path.exists():
                folder = self.tester._get_course_folder(pdf_path)
                pdfs.append((pdf_path, folder))
            else:
                print(red(f"PDF not found: {self.args.pdf}"))

        elif self.args.rerun_failed:
            # Rerun failed from last run
            failed_file = TEST_RESULTS_PATH / ".last-failed"
            if failed_file.exists():
                for line in failed_file.read_text().strip().split("\n"):
                    if line:
                        pdf_path = Path(line)
                        if pdf_path.exists():
                            folder = self.tester._get_course_folder(pdf_path)
                            pdfs.append((pdf_path, folder))
            else:
                print(yellow("No .last-failed file found. Run --all first."))

        elif self.args.course:
            # Single course mode
            course_key = None
            for key in COURSES.keys():
                if self.args.course.upper() in key.upper():
                    course_key = key
                    break

            if course_key:
                course_dir = TEST_DATA_PATH / course_key
                if course_dir.exists():
                    for pdf in sorted(course_dir.glob("*.pdf")):
                        if not self._should_skip(pdf):
                            pdfs.append((pdf, course_key))
            else:
                print(red(f"Course not found: {self.args.course}"))

        elif self.args.smoke:
            # Smoke test: 1 PDF per course
            for folder in sorted(COURSES.keys()):
                course_dir = TEST_DATA_PATH / folder
                if course_dir.exists():
                    pdf_files = sorted(course_dir.glob("*.pdf"))
                    if pdf_files:
                        pdf = pdf_files[0]
                        if not self._should_skip(pdf):
                            pdfs.append((pdf, folder))

        else:  # --all (default)
            for folder in sorted(COURSES.keys()):
                course_dir = TEST_DATA_PATH / folder
                if course_dir.exists():
                    for pdf in sorted(course_dir.glob("*.pdf")):
                        if not self._should_skip(pdf):
                            pdfs.append((pdf, folder))

        return pdfs

    def _should_skip(self, pdf_path: Path) -> bool:
        """Check if PDF should be skipped."""
        if self.args.skip:
            skip_list = self.args.skip if isinstance(self.args.skip, list) else [self.args.skip]
            for skip in skip_list:
                if skip in str(pdf_path):
                    return True
        return False

    def _print_progress(self, current: int, total: int, pdf_path: Path):
        """Print progress indicator with ETA."""
        if self.args.quiet:
            return

        # Calculate ETA
        eta_str = ""
        if self._times:
            avg_time = sum(self._times) / len(self._times)
            remaining = (total - current) * avg_time
            if remaining > 60:
                eta_str = f"(~{remaining/60:.0f} min remaining) "
            else:
                eta_str = f"(~{remaining:.0f}s remaining) "

        rel_path = pdf_path.relative_to(TEST_DATA_PATH) if TEST_DATA_PATH in pdf_path.parents else pdf_path.name
        print(f"\n{bold(f'[{current+1}/{total}]')} {cyan(eta_str)}{rel_path}")

    def _print_result(self, result: TestResult):
        """Print result for a single PDF."""
        if self.args.failures_only and result.status == "PASS":
            return

        print(f"  Pages: {result.pages}")
        if result.exercises > 0:
            print(f"  Exercises: {result.exercises}")
            print(f"  Sub-questions: {result.sub_questions}")
            if result.with_solutions > 0:
                print(f"  With solutions: {result.with_solutions}")

        if result.status == "PASS":
            print(f"  Status: {green('PASS')}")
        elif result.status == "FAIL":
            print(f"  Status: {red('FAIL')} - {result.error}")
        else:
            print(f"  Status: {red('ERROR')} - {result.error}")

        # Show LLM cache stats summary (if not --debug which shows all messages)
        if self.tester.llm and not self.args.debug:
            stats = self.tester.llm.get_cache_stats()
            if stats['total'] > 0:
                print(f"  LLM calls: {stats['total']} ({stats['hits']} cached, {stats['misses']} new)")

        for warning in result.warnings:
            print(f"  {yellow('Warning')}: {warning}")

        # Verbose: show exercise details
        if self.args.verbose and result.exercise_details:
            print()
            current_parent = None
            for ex in result.exercise_details:
                if ex["is_sub"]:
                    prefix = "    "
                    num = f"{ex['parent']}.{ex['number']}" if ex['parent'] else ex['number']
                else:
                    prefix = "  "
                    num = ex['number']
                    if ex.get("context"):
                        print(f"{prefix}Context: \"{ex['context'][:60]}...\"")

                sol_marker = " [+sol]" if ex["has_solution"] else ""
                print(f"{prefix}{num}{sol_marker}: \"{ex['text_preview']}\"")

        # Show skill groups (--full mode)
        if result.skill_groups:
            print(f"\n  {cyan('Skill Groups')} ({len(result.skill_groups)}):")
            for group in result.skill_groups:
                canonical = group["canonical"]
                print(f"    {bold(canonical)}")
                for member in group["members"]:
                    name = member["name"]
                    print(f"      ← {name}")
                    for ex in member.get("exercises", []):
                        num = ex.get("number", "?")
                        if ex.get("is_sub"):
                            ctx = ex.get("context", "")
                            txt = ex.get("text", "")
                            if ctx:
                                print(f"        [{num}] {dim(self._truncate_output(ctx, 50))}")
                                print(f"             \"{self._truncate_output(txt, 50)}\"")
                            else:
                                print(f"        [{num}] \"{self._truncate_output(txt, 50)}\"")
                        else:
                            txt = ex.get("text", "")
                            print(f"        [{num}] \"{self._truncate_output(txt, 60)}\"")

    def _truncate_output(self, text: str, max_len: int = 60) -> str:
        """Truncate text for output."""
        text = text.replace("\n", " ").strip()
        if len(text) <= max_len:
            return text
        return f"{text[:max_len-3]}..."

    def _record_result(self, result: TestResult):
        """Record result in summary."""
        self.summary.total += 1
        self.summary.results.append(result)

        if result.status == "PASS":
            self.summary.passed += 1
        elif result.status == "FAIL":
            self.summary.failed += 1
        else:
            self.summary.errors += 1

        self.summary.warnings.extend(result.warnings)

    def _print_summary(self):
        """Print test summary."""
        print(f"\n{bold('=== SUMMARY ===')}")
        print(f"Passed: {green(str(self.summary.passed))}/{self.summary.total}")

        # Show failures with stage breakdown
        failed_results = [r for r in self.summary.results if r.status != "PASS"]
        if failed_results:
            # Group by error stage
            by_stage = {}
            for r in failed_results:
                stage = r.error_stage or "unknown"
                if stage not in by_stage:
                    by_stage[stage] = []
                by_stage[stage].append(r)

            total_failed = len(failed_results)
            stage_summary = ", ".join(f"{len(v)} {k}" for k, v in sorted(by_stage.items()))
            print(f"Failed: {red(str(total_failed))}/{self.summary.total} ({stage_summary})")
        print(f"Duration: {self.summary.duration:.1f}s")

        # Aggregate stats
        total_exercises = sum(r.exercises for r in self.summary.results)
        total_subs = sum(r.sub_questions for r in self.summary.results)
        passed_count = len([r for r in self.summary.results if r.status == "PASS"])
        if passed_count > 0:
            avg_exercises = total_exercises / passed_count
            print(f"\nExercises: {total_exercises} total ({total_subs} subs), avg {avg_exercises:.1f}/PDF")

        # LLM stats (accumulated across all PDFs)
        total_llm = self._total_llm_hits + self._total_llm_misses
        if total_llm > 0:
            hit_rate = (self._total_llm_hits / total_llm) * 100
            print(f"LLM calls: {total_llm} ({self._total_llm_hits} cached, {self._total_llm_misses} new, {hit_rate:.0f}% hit rate)")

        if self.summary.warnings:
            print(f"\n{yellow('Warnings')}:")
            # Deduplicate warnings
            seen = set()
            for w in self.summary.warnings:
                if w not in seen:
                    print(f"  - {w}")
                    seen.add(w)

        if self._interrupted:
            print(yellow("\n(Interrupted - partial results)"))

    def _print_dry_run(self, pdfs: list):
        """Print what would be run without running."""
        print(bold("=== DRY RUN ===\n"))
        print(f"Would test {len(pdfs)} PDFs:\n")
        for pdf_path, folder in pdfs:
            rel_path = pdf_path.relative_to(TEST_DATA_PATH)
            print(f"  {rel_path}")
        print()

    def _get_mode(self) -> str:
        """Get current test mode string."""
        if self.args.golden:
            return "golden"
        elif self.args.parse:
            return "parse"
        elif self.args.analyze:
            return "analyze"
        elif self.args.full:
            return "full"
        elif self.args.smoke:
            return "smoke"
        else:
            return "split"

    def _save_results(self):
        """Save results to file (same format as _save_recent)."""
        TEST_RESULTS_PATH.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        mode = self._get_mode()

        if self.args.json:
            filename = f"{timestamp}_{mode}.json"
            filepath = TEST_RESULTS_PATH / filename
            data = {
                "timestamp": timestamp,
                "mode": mode,
                "summary": {
                    "total": self.summary.total,
                    "passed": self.summary.passed,
                    "failed": self.summary.failed,
                    "errors": self.summary.errors,
                    "duration": self.summary.duration,
                },
                "results": [
                    {
                        "pdf": r.pdf_path,
                        "status": r.status,
                        "pages": r.pages,
                        "exercises": r.exercises,
                        "subs": r.sub_questions,
                        "exercise_details": r.exercise_details,
                        "skill_groups": r.skill_groups,
                        "error": r.error,
                        "duration": r.duration,
                    }
                    for r in self.summary.results
                ],
            }
            filepath.write_text(json.dumps(data, indent=2))
        else:
            # Use same format as _save_recent
            filename = f"{timestamp}_{mode}.txt"
            filepath = TEST_RESULTS_PATH / filename
            lines = self._format_results_text(timestamp, mode)
            filepath.write_text("\n".join(lines))

        print(f"\nResults saved to: {filepath}")

    def _format_results_text(self, timestamp: str, mode: str) -> list:
        """Format results as text lines (shared by _save_results and _save_recent)."""
        lines = [
            f"Pipeline Test Results - {timestamp}",
            f"Mode: {mode}",
            f"Total: {self.summary.total}, Passed: {self.summary.passed}, Failed: {self.summary.failed}",
            f"Duration: {self.summary.duration:.1f}s",
            "",
        ]

        for r in self.summary.results:
            pdf_name = Path(r.pdf_path).name
            status = "PASS" if r.status == "PASS" else f"FAIL: {r.error}"

            # Mode: parse - just pages
            if self.args.parse:
                lines.append(f"═══ {pdf_name} ═══")
                lines.append(f"{status} | {r.pages} pages")
                lines.append("")

            # Mode: split (default) - exercises breakdown, no KI names
            elif not self.args.analyze and not self.args.full:
                lines.append(f"═══ {pdf_name} ═══")
                lines.append(f"{status} | {r.pages} pages | {r.exercises} exercises ({r.sub_questions} subs)")

                if r.exercise_details:
                    lines.append("")
                    # Group by parent exercise
                    exercises_by_parent = {}
                    standalone = []
                    parent_info = {}  # Store parent exercise info (context, etc.)

                    for ex in r.exercise_details:
                        if ex["is_sub"]:
                            parent = ex.get("parent", "?")
                            if parent not in exercises_by_parent:
                                exercises_by_parent[parent] = []
                            exercises_by_parent[parent].append(ex)
                        else:
                            parent_info[ex['number']] = ex
                            # Only add to standalone if no subs will claim this parent
                            standalone.append(ex)

                    # Print exercises with subs first (grouped)
                    printed_parents = set()
                    for parent_num, subs in sorted(exercises_by_parent.items()):
                        printed_parents.add(parent_num)
                        parent_ex = parent_info.get(parent_num, {})
                        sol = " [+sol]" if parent_ex.get("has_solution") else ""

                        lines.append(f"  EXERCISE {parent_num}:{sol}")
                        # Show parent context if available
                        if parent_ex.get("context"):
                            ctx = parent_ex["context"][:60] + "..." if len(parent_ex.get("context", "")) > 60 else parent_ex.get("context", "")
                            lines.append(f"    {ctx}")
                        # Show sub-questions
                        for sub in subs:
                            sub_sol = " [+sol]" if sub["has_solution"] else ""
                            prefix = f"    - {sub['number']}:{sub_sol} "
                            wrapped = textwrap.fill(sub['text_preview'], width=78,
                                                    initial_indent=prefix,
                                                    subsequent_indent=" " * len(prefix))
                            # Add continuation marker at end of lines that wrap
                            wrap_lines = wrapped.split('\n')
                            if len(wrap_lines) > 1:
                                wrap_lines = [l + "-" if i < len(wrap_lines) - 1 else l
                                              for i, l in enumerate(wrap_lines)]
                            lines.append('\n'.join(wrap_lines))

                    # Print standalone exercises (no subs)
                    for parent_ex in standalone:
                        parent_num = parent_ex['number']
                        if parent_num in printed_parents:
                            continue  # Already printed as grouped
                        sol = " [+sol]" if parent_ex["has_solution"] else ""
                        prefix = f"  - Ex {parent_num}:{sol} "
                        wrapped = textwrap.fill(parent_ex['text_preview'], width=78,
                                                initial_indent=prefix,
                                                subsequent_indent=" " * len(prefix))
                        # Add continuation marker at end of lines that wrap
                        wrap_lines = wrapped.split('\n')
                        if len(wrap_lines) > 1:
                            wrap_lines = [l + "-" if i < len(wrap_lines) - 1 else l
                                          for i, l in enumerate(wrap_lines)]
                        lines.append('\n'.join(wrap_lines))
                lines.append("")

            # Mode: analyze - exercises with KI names, no skill groups
            elif self.args.analyze:
                lines.append(f"═══ {pdf_name} ═══")
                lines.append(f"{status} | {r.pages} pages | {r.exercises} exercises ({r.sub_questions} subs)")

                if r.exercise_details:
                    lines.append("")
                    for ex in r.exercise_details:
                        if ex["is_sub"]:
                            num = ex['number']
                            indent = "      "
                        else:
                            num = f"Ex {ex['number']}"
                            indent = "    "
                        sol = " [+sol]" if ex["has_solution"] else ""
                        ki_name = ex.get("ki_name", "")
                        ki_label = f" → {ki_name}" if ki_name else ""
                        lines.append(f"  - {num}{sol}{ki_label}:")
                        wrapped = textwrap.fill(ex['text_preview'], width=80,
                                                initial_indent=indent,
                                                subsequent_indent=indent)
                        lines.append(wrapped)
                lines.append("")

            # Mode: full - exercises with KI names + skill groups
            else:
                n_groups = len(r.skill_groups) if r.skill_groups else 0
                lines.append(f"═══ {pdf_name} ═══")
                lines.append(f"{status} | {r.pages} pages | {r.exercises} exercises → {n_groups} skill groups")

                if r.exercise_details:
                    lines.append("")
                    lines.append("Exercises:")
                    for ex in r.exercise_details:
                        if ex["is_sub"]:
                            num = ex['number']
                            indent = "        "
                        else:
                            num = f"Ex {ex['number']}"
                            indent = "      "
                        sol = " [+sol]" if ex["has_solution"] else ""
                        ki_name = ex.get("ki_name", "")
                        ki_label = f" → {ki_name}" if ki_name else ""
                        lines.append(f"    - {num}{sol}{ki_label}:")
                        wrapped = textwrap.fill(ex['text_preview'], width=80,
                                                initial_indent=indent,
                                                subsequent_indent=indent)
                        lines.append(wrapped)

                if r.skill_groups:
                    lines.append("")
                    lines.append("Skill Groups:")
                    for group in r.skill_groups:
                        canonical = group.get("canonical", "Unknown")
                        lines.append(f"    [{canonical}]")
                        for member in group.get("members", []):
                            name = member.get("name", "?")
                            lines.append(f"      • {name}")

                lines.append("")

        return lines

    def _save_failed_list(self):
        """Save list of failed PDFs for --rerun-failed."""
        TEST_RESULTS_PATH.mkdir(exist_ok=True)
        failed_file = TEST_RESULTS_PATH / ".last-failed"

        failed_pdfs = [r.pdf_path for r in self.summary.results if r.status != "PASS"]
        if failed_pdfs:
            failed_file.write_text("\n".join(failed_pdfs))
        elif failed_file.exists():
            failed_file.unlink()

    def _save_recent(self):
        """Auto-save to .recent/ directory, keeping last 3 runs."""
        recent_dir = TEST_RESULTS_PATH / ".recent"
        recent_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        mode = self._get_mode()
        filename = f"{timestamp}_{mode}.txt"

        lines = self._format_results_text(timestamp, mode)
        (recent_dir / filename).write_text("\n".join(lines))

        # Keep only last 3
        recent_files = sorted(recent_dir.glob("*.txt"), key=lambda p: p.stat().st_mtime)
        while len(recent_files) > 3:
            recent_files[0].unlink()
            recent_files.pop(0)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Pipeline Test Suite for examina-core",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --smoke              Quick test (1 PDF per course)
  %(prog)s --golden             Run regression tests (exact counts)
  %(prog)s --pdf path.pdf --full  Show skill groups/merges for single PDF
  %(prog)s --course ADE         Test all PDFs in ADE course
  %(prog)s --smoke --save       Save results permanently
  %(prog)s --rerun-failed       Rerun only failed tests
        """,
    )

    # Run modes
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--smoke", action="store_true", help="Quick test: 1 PDF per course")
    mode_group.add_argument("--course", metavar="NAME", help="Test all PDFs in one course")
    mode_group.add_argument("--all", action="store_true", help="Test all PDFs (default)")
    mode_group.add_argument("--pdf", metavar="PATH", help="Test single PDF file")
    mode_group.add_argument("--golden", action="store_true", help="Run golden regression tests")
    mode_group.add_argument("--rerun-failed", action="store_true", help="Rerun failed tests from last run")

    # Pipeline depth
    depth_group = parser.add_mutually_exclusive_group()
    depth_group.add_argument("--parse", action="store_true", help="Test parse only (stage 1)")
    depth_group.add_argument("--split", action="store_true", help="Test parse + split (stages 1-2, default)")
    depth_group.add_argument("--analyze", action="store_true", help="Test parse + split + analyze (stages 1-3)")
    depth_group.add_argument("--full", action="store_true", help="Full pipeline + skill grouping (shows merges)")

    # Output options
    parser.add_argument("--save", action="store_true", help="Save results to test-results/")
    parser.add_argument("--json", action="store_true", help="Output as JSON (with --save)")
    parser.add_argument("--failures-only", action="store_true", help="Show only failed tests")
    parser.add_argument("--timing", action="store_true", help="Show timing per PDF")
    parser.add_argument("--lang", default="en", help="Language for analysis (default: en)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run without running")
    parser.add_argument("--skip", action="append", metavar="PDF", help="Skip specific PDF(s)")
    parser.add_argument("-q", "--quiet", action="store_true", help="Summary only")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show full exercise details")
    parser.add_argument("--debug", action="store_true", help="Show all LLM cache messages")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")

    args = parser.parse_args()

    # Default to --all if no mode specified
    if not any([args.smoke, args.course, args.all, args.pdf, args.golden, args.rerun_failed]):
        args.all = True

    # Disable colors if requested or not a TTY
    if args.no_color or not sys.stdout.isatty():
        Colors.disable()

    return args


def main():
    """Main entry point."""
    args = parse_args()

    runner = TestRunner(args)

    if args.golden:
        exit_code = runner.run_golden()
    else:
        exit_code = runner.run()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
