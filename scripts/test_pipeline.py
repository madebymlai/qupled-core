#!/usr/bin/env python3
"""
Pipeline Test Suite for examina-core.

Tests parsing, exercise splitting, and knowledge item extraction
across diverse exam PDFs from multiple academic domains.

Defaults (ON):
- Active learning: ML classifier reduces LLM calls 70-90%
- Persist: Training data saved to .examina/training_cache.json
- Timeout: 300s per PDF (prevents hangs)

Quick Start:
    python test_pipeline.py --smoke              # Quick test (1 PDF/course)
    python test_pipeline.py --smoke --full -j 4  # Parallel processing
    python test_pipeline.py --all --sample 10    # Random 10 PDFs

CI Integration:
    python test_pipeline.py --smoke --junit results.xml  # JUnit XML output
    python test_pipeline.py --compare old.json new.json  # Detect regressions

Reports:
    python test_pipeline.py --smoke --html report.html   # HTML report (auto-opens)
    python test_pipeline.py --smoke --show-cost          # LLM cost estimate

Full Usage:
    python test_pipeline.py --help
"""

import argparse
import json
import random
import signal
import sys
import textwrap
import threading
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
from xml.dom import minidom
from xml.etree.ElementTree import Element, SubElement, tostring

import yaml

# Optional: tqdm for progress bar
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Add examina to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.analyzer import ExerciseAnalyzer, generate_item_description
from core.exercise_splitter import ExerciseSplitter
from core.merger import classify_items, get_canonical_name
from core.pdf_processor import PDFProcessor
from models.llm_manager import LLMManager

# Optional: Active learning imports (may not be installed)
try:
    from core.active_learning import ActiveClassifier
    from core.features import compute_embedding, cosine_similarity
    ACTIVE_LEARNING_AVAILABLE = True
except ImportError:
    ACTIVE_LEARNING_AVAILABLE = False

# =============================================================================
# Configuration
# =============================================================================

TEST_DATA_PATH = Path("/home/laimk/git/examina-cloud/test-data")
TEST_RESULTS_PATH = Path(__file__).parent.parent / "test-results"
TRAINING_CACHE_PATH = Path(__file__).parent.parent / ".examina" / "training_cache.json"

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

# LLM costs per 1M tokens (USD) - DeepSeek pricing
# https://api-docs.deepseek.com/quick_start/pricing/
LLM_COSTS = {
    "deepseek-chat": {"input": 0.28, "input_cached": 0.028, "output": 0.42},
    "deepseek-reasoner": {"input": 0.28, "input_cached": 0.028, "output": 0.42},
    "default": {"input": 0.28, "input_cached": 0.028, "output": 0.42},
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
    skill_groups: list = field(default_factory=list)  # Internal merge results
    knowledge_items: list = field(default_factory=list)  # KIs for cross-batch merge
    categories: list = field(default_factory=list)  # Discovered categories
    active_learning_stats: dict = field(default_factory=dict)  # ML stats
    feature_similarities: list = field(default_factory=list)  # Top similar pairs


@dataclass
class TestSummary:
    """Summary of all test results."""

    total: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    timeouts: int = 0
    warnings: list = field(default_factory=list)
    results: list = field(default_factory=list)
    duration: float = 0.0
    cross_batch_groups: list = field(default_factory=list)  # Cross-PDF merge results


# =============================================================================
# Pipeline Runners
# =============================================================================


class PipelineTester:
    """Runs pipeline tests on PDFs."""

    def __init__(
        self,
        lang: str = "en",
        verbose: bool = False,
        quiet: bool = False,
        debug: bool = False,
        use_active_learning: bool = True,
        show_features: bool = False,
        training_path: str | None = None,
    ):
        self.lang = lang
        self.verbose = verbose
        self.quiet = quiet
        self.debug = debug
        self.use_active_learning = use_active_learning
        self.show_features = show_features
        self.training_path = training_path
        self.processor = PDFProcessor()
        self.llm = None
        self.splitter = None
        self.analyzer = None
        self.active_classifier = None
        self._interrupted = False

    def _init_llm(self):
        """Lazy init LLM (expensive)."""
        if self.llm is None:
            self.llm = LLMManager(provider="deepseek", quiet=not self.debug)
            self.splitter = ExerciseSplitter()

    def _init_active_learning(self):
        """Lazy init active learning classifier with optional warm start."""
        if self.use_active_learning and ACTIVE_LEARNING_AVAILABLE:
            if self.active_classifier is None:
                self.active_classifier = ActiveClassifier()

                # Warm start from training data
                if self.training_path and Path(self.training_path).exists():
                    try:
                        with open(self.training_path) as f:
                            data = json.load(f)
                        samples = self.active_classifier.import_training_data(data)
                        if not self.quiet:
                            print(f"  {cyan('Active Learning')}: Warm start ({samples} samples)")
                    except Exception as e:
                        if not self.quiet:
                            print(f"  {yellow('Warning')}: Failed to load training: {e}")
                            print(f"  {cyan('Active Learning')}: Cold start")
                elif not self.quiet:
                    print(f"  {cyan('Active Learning')}: Cold start")

    def _init_analyzer(self):
        """Lazy init analyzer."""
        self._init_llm()
        if self.analyzer is None:
            self.analyzer = ExerciseAnalyzer(llm_manager=self.llm, language=self.lang)

    def _compute_feature_matrix(self, items: list[dict]) -> list[dict]:
        """Compute embedding similarities between all items (for --show-features)."""
        if not ACTIVE_LEARNING_AVAILABLE or not items:
            return []

        # Compute embeddings for all items
        embeddings = {}
        for item in items:
            desc = item.get("description", "") or item.get("name", "")
            embeddings[item["id"]] = compute_embedding(desc)

        # Compute pairwise similarities
        similarities = []
        for i, item_a in enumerate(items):
            for item_b in items[i + 1 :]:
                emb_a = embeddings[item_a["id"]]
                emb_b = embeddings[item_b["id"]]
                sim = cosine_similarity(emb_a, emb_b)

                # Only show high similarity pairs
                if sim >= 0.5:
                    similarities.append({
                        "item_a": item_a["name"],
                        "item_b": item_b["name"],
                        "similarity": sim,
                    })

        # Sort by similarity descending
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities[:10]  # Top 10

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
            generic_count = sum(
                1
                for e in exercises
                if e.exercise_number
                and (
                    e.exercise_number.startswith("exercise_")
                    or e.exercise_number.startswith("question_")
                )
            )
            if generic_count > 0:
                result.warnings.append(f"{generic_count} exercises with generic numbers")

            # Exercise details for verbose output
            for ex in exercises:
                result.exercise_details.append(
                    {
                        "number": ex.exercise_number,
                        "is_sub": ex.is_sub_question,
                        "parent": ex.parent_exercise_number,
                        "has_solution": bool(ex.solution),
                        "text_preview": self._truncate(ex.text, 100),
                        "text_full": ex.text,  # Full text for analyzer
                        "context": getattr(ex, "exercise_context", "") or "",
                    }
                )

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
        """Stage 1-3: Test full pipeline including analysis (KI names + descriptions)."""
        result = self.test_split(pdf_path, course_name)
        if result.status != "PASS":
            return result

        start = time.time()

        try:
            self._init_analyzer()

            # Group exercises by KI name for description generation
            ki_exercises: dict[str, list[dict]] = {}

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
                    is_sub_question=is_sub,
                )

                ki_name = None
                learning_approach = None
                if analysis.knowledge_items:
                    ki = analysis.knowledge_items[0]
                    ki_name = ki.name
                    learning_approach = getattr(ki, 'learning_approach', None)

                ex["ki_name"] = ki_name or f"unknown_{ex.get('number', '?')}"
                ex["learning_approach"] = learning_approach

                # Collect exercises for description generation
                if ki_name:
                    if ki_name not in ki_exercises:
                        ki_exercises[ki_name] = []
                    ki_exercises[ki_name].append({
                        "text": ex.get("text_preview", ""),
                        "context": ex.get("context", ""),
                        "is_sub": is_sub,
                    })

            # Generate descriptions for each KI
            for ki_name, exs in ki_exercises.items():
                if exs:
                    description = generate_item_description(exs, self.llm)
                    # Store description back to exercises
                    for ex in result.exercise_details:
                        if ex.get("ki_name") == ki_name:
                            ex["ki_description"] = description

            result.status = "PASS"

        except Exception as e:
            result.status = "ERROR"
            result.error_stage = "analyze"
            result.error = f"Analyze failed: {str(e)}"

        result.duration += time.time() - start
        return result

    def test_full(
        self,
        pdf_path: Path,
        course_name: str,
        existing_items: list[dict] | None = None,
    ) -> TestResult:
        """Stage 1-4: Full pipeline including analysis + internal merge.

        Args:
            pdf_path: Path to PDF file
            course_name: Name of course for context
            existing_items: For sequential mode, items from previous PDFs to merge against

        Returns:
            TestResult with knowledge_items populated for cross-batch merge
        """
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
                if ex.get("is_sub"):
                    snippet = ex.get("text_preview", "")
                else:
                    snippet = ex.get("context", "") or ex.get("text_preview", "")

                ki_exercises[ki_name].append(
                    {
                        "number": ex.get("number"),
                        "is_sub": ex.get("is_sub"),
                        "context": ex.get("context", ""),
                        "text": ex.get("text_preview", ""),
                        "snippet": snippet,
                    }
                )

            # Build items with descriptions
            items = []
            for name, exs in ki_exercises.items():
                description = generate_item_description(exs, self.llm)
                items.append(
                    {
                        "id": len(items),  # Index as ID
                        "name": name,
                        "description": description,
                        "exercises": exs,
                        "pdf": pdf_path.name,
                    }
                )

            # Compute feature similarities (for --show-features)
            if self.show_features and len(items) >= 2:
                result.feature_similarities = self._compute_feature_matrix(items)

            # INTERNAL MERGE: Find duplicates within this PDF using classify_items
            if len(items) >= 2:
                # Init active learning if requested
                self._init_active_learning()

                # classify_items returns (groups, assignments)
                final_groups, _ = classify_items(
                    items, [], self.llm,
                    active_classifier=self.active_classifier,
                )

                # Capture active learning stats
                if self.active_classifier:
                    result.active_learning_stats = self.active_classifier.get_stats()

                # Collect categories discovered
                categories_seen = set()

                # Find groups with multiple items (merged)
                for group in final_groups:
                    group_items_list = group.get("items", [])

                    # Track categories
                    group_category = group.get("category")
                    if group_category:
                        categories_seen.add(group_category)

                    if len(group_items_list) < 2:
                        continue

                    group_names = [item["name"] for item in group_items_list]
                    canonical = group.get("name", get_canonical_name(group_names, self.llm))

                    members = []
                    for item in group_items_list:
                        members.append(
                            {
                                "name": item["name"],
                                "description": item.get("description", ""),
                                "category": item.get("category", ""),
                                "exercises": item.get("exercises", []),
                            }
                        )

                    result.skill_groups.append(
                        {
                            "canonical": canonical,
                            "description": group.get("description", ""),
                            "category": group_category,
                            "members": members,
                            "type": "internal",
                        }
                    )

                result.categories = list(categories_seen)

            # SEQUENTIAL MODE: Merge against existing items from previous PDFs
            if existing_items and len(items) >= 1:
                # Classify new items against existing groups
                existing_groups = [
                    {
                        "id": i,
                        "name": item["name"],
                        "description": item.get("description", ""),
                        "items": [item],
                    }
                    for i, item in enumerate(existing_items)
                ]

                final_groups, _ = classify_items(items, existing_groups, self.llm)

                # Find groups that contain items from both old and new
                for group in final_groups:
                    group_items_list = group.get("items", [])
                    if len(group_items_list) < 2:
                        continue

                    # Check if spans old and new PDFs
                    old_names = {item["name"] for item in existing_items}
                    new_names = {item["name"] for item in items}
                    group_names = [item["name"] for item in group_items_list]

                    has_old = any(name in old_names for name in group_names)
                    has_new = any(name in new_names for name in group_names)

                    if has_old and has_new:
                        canonical = group.get("name", get_canonical_name(group_names, self.llm))

                        members = []
                        for item in group_items_list:
                            members.append(
                                {
                                    "name": item["name"],
                                    "description": item.get("description", ""),
                                    "pdf": item.get("pdf", "?"),
                                }
                            )

                        result.skill_groups.append(
                            {
                                "canonical": canonical,
                                "members": members,
                                "type": "cross-batch-inline",
                            }
                        )

            # Store items for cross-batch (parallel mode uses this after all PDFs)
            result.knowledge_items = items
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

        # Determine training path (explicit or auto from cache)
        training_path = getattr(args, 'load_training', None)
        if not training_path and getattr(args, 'persist', False):
            if TRAINING_CACHE_PATH.exists():
                training_path = str(TRAINING_CACHE_PATH)

        self.tester = PipelineTester(
            lang=args.lang,
            verbose=args.verbose,
            quiet=args.quiet,
            debug=args.debug,
            use_active_learning=getattr(args, 'with_active_learning', True),
            show_features=getattr(args, 'show_features', False),
            training_path=training_path,
        )
        self.summary = TestSummary()
        self._interrupted = False
        self._start_time = None
        self._times = []  # Track PDF processing times for ETA
        self._total_llm_hits = 0
        self._total_llm_misses = 0

        # Cumulative active learning stats
        self._total_al_llm_calls = 0
        self._total_al_predictions = 0
        self._total_al_transitive = 0
        self._total_al_training = 0

        # Cross-batch tracking per course
        self._course_items: dict[str, list[dict]] = {}  # course_folder -> accumulated items

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

        # Determine processing mode (only matters for --full)
        is_sequential = self.args.sequential
        is_independent = self.args.independent
        # Default is parallel (Pro-like) with cross-batch

        # Run tests
        total = len(pdfs)
        prev_course = None

        # Determine if we should show progress bar
        show_progress = (
            TQDM_AVAILABLE and
            not self.args.quiet and
            not self.args.verbose and
            not getattr(self.args, 'no_progress', False) and
            sys.stdout.isatty()
        )

        pbar = None
        if show_progress:
            pbar = tqdm(
                total=total,
                desc="Testing",
                unit="pdf",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            )

        # Check for parallel mode (only for --full with -j > 1)
        use_parallel = (
            getattr(self.args, 'jobs', 1) > 1 and
            self.args.full and
            not is_sequential  # Sequential mode requires ordered processing
        )

        try:
            if use_parallel:
                # Parallel mode: process PDFs concurrently
                if not self.args.quiet:
                    print(f"Running with {self.args.jobs} parallel workers...")
                self._run_parallel(pdfs, pbar)
            else:
                # Sequential mode
                for i, (pdf_path, course_folder) in enumerate(pdfs):
                    if self._interrupted:
                        break

                    course_name = self.tester._get_course_name(course_folder)

                    # Cross-batch merge when course changes (parallel mode only)
                    if prev_course and prev_course != course_folder and self.args.full:
                        if not is_sequential and not is_independent:
                            self._run_cross_batch_merge(prev_course)

                    prev_course = course_folder

                    # Update progress (tqdm or text)
                    if pbar:
                        pbar.set_postfix({
                            "pdf": pdf_path.name[:20],
                            "pass": self.summary.passed,
                            "fail": self.summary.failed,
                        })
                    else:
                        self._print_progress(i, total, pdf_path)

                    # Accumulate and reset cache stats for per-PDF tracking
                    if self.tester.llm:
                        stats = self.tester.llm.get_cache_stats()
                        self._total_llm_hits += stats["hits"]
                        self._total_llm_misses += stats["misses"]
                        self.tester.llm.reset_cache_stats()

                    # Run appropriate test level (with timeout)
                    timeout = getattr(self.args, 'timeout', 300)
                    try:
                        if self.args.parse:
                            result = self._run_with_timeout(
                                self.tester.test_parse, timeout, pdf_path
                            )
                        elif self.args.full:
                            # Sequential mode: pass existing items for inline merge
                            existing_items = None
                            if is_sequential:
                                existing_items = self._course_items.get(course_folder, [])
                            result = self._run_with_timeout(
                                self.tester.test_full, timeout, pdf_path, course_name, existing_items
                            )

                            # Accumulate items for cross-batch (parallel) or next PDF (sequential)
                            if result.status == "PASS" and result.knowledge_items:
                                if course_folder not in self._course_items:
                                    self._course_items[course_folder] = []
                                self._course_items[course_folder].extend(result.knowledge_items)
                        elif self.args.analyze:
                            result = self._run_with_timeout(
                                self.tester.test_analyze, timeout, pdf_path, course_name
                            )
                        else:  # Default: split
                            result = self._run_with_timeout(
                                self.tester.test_split, timeout, pdf_path, course_name
                            )
                    except TimeoutError as e:
                        result = TestResult(
                            pdf_path=str(pdf_path),
                            status="TIMEOUT",
                            error=str(e),
                            duration=float(timeout),
                        )

                    self._record_result(result)
                    self._times.append(result.duration)

                    if pbar:
                        pbar.update(1)
                    elif not self.args.quiet:
                        self._print_result(result)

                # Final cross-batch merge for last course (sequential only)
                if self.args.full and prev_course and not is_sequential and not is_independent:
                    self._run_cross_batch_merge(prev_course)
        finally:
            if pbar:
                pbar.close()

        # Capture final PDF's LLM stats
        if self.tester.llm:
            stats = self.tester.llm.get_cache_stats()
            self._total_llm_hits += stats["hits"]
            self._total_llm_misses += stats["misses"]

        # Print summary
        self.summary.duration = time.time() - self._start_time
        self._print_summary()

        # Auto-save last 3 runs
        self._save_recent()

        # Save results permanently if requested
        if self.args.save:
            self._save_results()

        # Generate HTML report if requested
        if getattr(self.args, 'html', None):
            self._generate_html_report(self.args.html)

        # Generate JUnit XML report if requested
        if getattr(self.args, 'junit', None):
            self._generate_junit_report(self.args.junit)

        # Save training data if requested or persist is enabled
        self._save_training_data()

        # Save failed list for --rerun-failed
        self._save_failed_list()

        return 0 if self.summary.failed == 0 and self.summary.errors == 0 else 1

    def run_benchmark(self) -> int:
        """Run benchmark comparing with/without active learning."""
        print(bold("=== BENCHMARK MODE ===\n"))

        pdfs = self._collect_pdfs()
        if not pdfs:
            print(red("No PDFs found!"))
            return 1

        print(f"Testing {len(pdfs)} PDFs\n")

        # Run 1: Without active learning
        print(bold("Run 1: Without Active Learning"))
        print("=" * 40)
        self.tester.use_active_learning = False
        self.tester.active_classifier = None

        start1 = time.time()
        results_without = self._run_all_pdfs(pdfs)
        time_without = time.time() - start1
        llm_calls_without = self._total_llm_misses

        # Reset for run 2
        self._reset_benchmark_stats()

        # Run 2: With active learning
        print(f"\n{bold('Run 2: With Active Learning')}")
        print("=" * 40)
        self.tester.use_active_learning = True

        start2 = time.time()
        results_with = self._run_all_pdfs(pdfs)
        time_with = time.time() - start2
        llm_calls_with = self._total_al_llm_calls

        # Compare results
        self._print_benchmark_comparison(
            results_without, results_with,
            time_without, time_with,
            llm_calls_without, llm_calls_with,
        )

        return 0

    def _run_all_pdfs(self, pdfs: list) -> list:
        """Run all PDFs and return results."""
        results = []
        for i, (pdf_path, course_folder) in enumerate(pdfs):
            course_name = self.tester._get_course_name(course_folder)

            if not self.args.quiet:
                print(f"  [{i+1}/{len(pdfs)}] {pdf_path.name}")

            result = self.tester.test_full(pdf_path, course_name)
            results.append(result)
            self._record_result(result)

        return results

    def _run_with_timeout(self, func, timeout: int, *args, **kwargs):
        """Run function with timeout. Returns result or raises TimeoutError."""
        if timeout <= 0:
            return func(*args, **kwargs)

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                return future.result(timeout=timeout)
            except FuturesTimeoutError:
                raise TimeoutError(f"Exceeded {timeout}s timeout")

    def _run_parallel(self, pdfs: list, pbar=None) -> list:
        """Process PDFs in parallel using ThreadPoolExecutor."""
        results = []
        n_jobs = min(self.args.jobs, len(pdfs))

        # Thread-local storage for tester instances
        thread_local = threading.local()

        def get_thread_tester():
            """Get or create per-thread PipelineTester instance."""
            if not hasattr(thread_local, 'tester'):
                thread_local.tester = PipelineTester(
                    lang=self.tester.lang,
                    verbose=False,  # Disable verbose in threads
                    quiet=True,
                    debug=False,
                    use_active_learning=self.tester.use_active_learning,
                    show_features=self.tester.show_features,
                    training_path=self.tester.training_path,
                )
                # Share the active classifier (thread-safe with locks)
                thread_local.tester.active_classifier = self.tester.active_classifier
            return thread_local.tester

        def process_pdf(pdf_path, course_folder):
            """Process a single PDF in a thread."""
            tester = get_thread_tester()
            course_name = tester._get_course_name(course_folder)
            return tester.test_full(pdf_path, course_name)

        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            # Submit all tasks
            future_to_pdf = {
                executor.submit(process_pdf, pdf, course): (pdf, course)
                for pdf, course in pdfs
            }

            # Collect results as they complete
            timeout = getattr(self.args, 'timeout', 300)
            timeout_val = timeout if timeout > 0 else None

            for future in as_completed(future_to_pdf):
                pdf_path, course_folder = future_to_pdf[future]
                try:
                    result = future.result(timeout=timeout_val)
                    results.append(result)
                    self._record_result(result)
                    self._times.append(result.duration)

                    # Update progress bar if using tqdm
                    if pbar:
                        pbar.update(1)
                        pbar.set_postfix({
                            "pass": self.summary.passed,
                            "fail": self.summary.failed,
                        })

                except FuturesTimeoutError:
                    # Timeout result
                    result = TestResult(
                        pdf_path=str(pdf_path),
                        status="TIMEOUT",
                        error=f"Exceeded {timeout}s timeout",
                        duration=float(timeout),
                    )
                    results.append(result)
                    self._record_result(result)
                    if pbar:
                        pbar.update(1)

                except Exception as e:
                    # Record error result
                    result = TestResult(
                        pdf_path=str(pdf_path),
                        status="ERROR",
                        error=str(e),
                    )
                    results.append(result)
                    self._record_result(result)
                    if pbar:
                        pbar.update(1)

        return results

    def _reset_benchmark_stats(self):
        """Reset stats between benchmark runs."""
        self._total_llm_hits = 0
        self._total_llm_misses = 0
        self._total_al_llm_calls = 0
        self._total_al_predictions = 0
        self._total_al_transitive = 0
        self._total_al_training = 0
        self.summary = TestSummary()
        if self.tester.llm:
            self.tester.llm.reset_cache_stats()

    def _print_benchmark_comparison(
        self,
        results_without, results_with,
        time_without, time_with,
        llm_without, llm_with,
    ):
        """Print benchmark comparison."""
        print(f"\n{bold('=== BENCHMARK RESULTS ===')}\n")

        # Time comparison
        time_diff = time_without - time_with
        time_pct = (time_diff / time_without * 100) if time_without > 0 else 0
        print("Time:")
        print(f"  Without AL: {time_without:.1f}s")
        print(f"  With AL:    {time_with:.1f}s")
        if time_diff > 0:
            print(f"  Speedup:    {green(f'{time_pct:.0f}%')} ({time_diff:.1f}s saved)")
        else:
            print(f"  Difference: {yellow(f'{time_diff:.1f}s')}")

        # LLM calls comparison
        if llm_without > 0:
            llm_diff = llm_without - llm_with
            llm_pct = (llm_diff / llm_without * 100) if llm_without > 0 else 0
            print("\nLLM Calls (classification):")
            print(f"  Without AL: {llm_without}")
            print(f"  With AL:    {llm_with}")
            print(f"  Reduction:  {green(f'{llm_pct:.0f}%')} ({llm_diff} calls saved)")

        # Merge quality comparison
        merges_without = sum(len(r.skill_groups) for r in results_without)
        merges_with = sum(len(r.skill_groups) for r in results_with)
        print("\nMerge Groups Found:")
        print(f"  Without AL: {merges_without}")
        print(f"  With AL:    {merges_with}")
        if merges_without != merges_with:
            diff = merges_with - merges_without
            print(f"  Difference: {yellow(f'{diff:+d}')}")
        else:
            print(f"  Difference: {green('None (identical)')}")

        # KI counts
        kis_without = sum(len(r.knowledge_items) for r in results_without)
        kis_with = sum(len(r.knowledge_items) for r in results_with)
        print("\nKnowledge Items:")
        print(f"  Without AL: {kis_without}")
        print(f"  With AL:    {kis_with}")

    def run_compare(self) -> int:
        """Compare two result files and show regressions/improvements."""
        old_path, new_path = self.args.compare

        # Load files
        try:
            with open(old_path) as f:
                old_data = json.load(f)
            with open(new_path) as f:
                new_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"{red('Error')}: {e}")
            return 1

        print(bold("=== COMPARISON ===\n"))
        print(f"Old: {old_path}")
        print(f"New: {new_path}\n")

        # Compare summaries
        old_sum = old_data.get("summary", {})
        new_sum = new_data.get("summary", {})

        print(bold("Summary:"))
        self._compare_metric("Passed", old_sum.get("passed", 0), new_sum.get("passed", 0))
        self._compare_metric("Failed", old_sum.get("failed", 0), new_sum.get("failed", 0))
        self._compare_metric("Duration", old_sum.get("duration", 0), new_sum.get("duration", 0), suffix="s")

        # Find regressions and improvements
        old_by_pdf = {Path(r.get("pdf", "")).name: r for r in old_data.get("results", [])}
        new_by_pdf = {Path(r.get("pdf", "")).name: r for r in new_data.get("results", [])}

        regressions = []
        improvements = []

        for pdf_name, new_result in new_by_pdf.items():
            old_result = old_by_pdf.get(pdf_name)
            if not old_result:
                continue

            old_status = old_result.get("status", "")
            new_status = new_result.get("status", "")

            if old_status == "PASS" and new_status != "PASS":
                regressions.append({
                    "pdf": pdf_name,
                    "old_status": old_status,
                    "new_status": new_status,
                    "error": new_result.get("error"),
                })
            elif old_status != "PASS" and new_status == "PASS":
                improvements.append({
                    "pdf": pdf_name,
                    "old_status": old_status,
                    "new_status": new_status,
                })

        # Print regressions
        if regressions:
            print(f"\n{red('Regressions')} ({len(regressions)}):")
            for r in regressions:
                print(f"  {red('✗')} {r['pdf']}: {r['old_status']} → {r['new_status']}")
                if r.get("error"):
                    print(f"      Error: {r['error'][:60]}...")
        else:
            print(f"\n{green('No regressions')}")

        # Print improvements
        if improvements:
            print(f"\n{green('Improvements')} ({len(improvements)}):")
            for r in improvements:
                print(f"  {green('✓')} {r['pdf']}: {r['old_status']} → {r['new_status']}")

        # Compare KI counts by course
        print(f"\n{bold('KI Changes:')}")
        old_kis = self._count_kis_by_course(old_data.get("results", []))
        new_kis = self._count_kis_by_course(new_data.get("results", []))

        all_courses = set(old_kis.keys()) | set(new_kis.keys())
        for course in sorted(all_courses):
            old_count = old_kis.get(course, 0)
            new_count = new_kis.get(course, 0)
            self._compare_metric(f"  {course}", old_count, new_count, suffix=" KIs")

        return 1 if regressions else 0

    def _compare_metric(self, name: str, old: float, new: float, suffix: str = ""):
        """Print comparison of a metric."""
        diff = new - old
        if diff > 0:
            diff_str = yellow(f"+{diff:.1f}" if isinstance(diff, float) and not diff.is_integer() else f"+{int(diff)}")
        elif diff < 0:
            diff_str = green(f"{diff:.1f}" if isinstance(diff, float) and not diff.is_integer() else f"{int(diff)}")
        else:
            diff_str = dim("0")

        old_str = f"{old:.1f}" if isinstance(old, float) and not old.is_integer() else str(int(old) if isinstance(old, float) else old)
        new_str = f"{new:.1f}" if isinstance(new, float) and not new.is_integer() else str(int(new) if isinstance(new, float) else new)

        print(f"  {name}: {old_str} → {new_str}{suffix} ({diff_str})")

    def _count_kis_by_course(self, results: list) -> dict[str, int]:
        """Count KIs by course from results."""
        counts: dict[str, int] = {}
        for r in results:
            pdf_path = r.get("pdf", "")
            # Extract course from path
            for course in COURSES.keys():
                if course in pdf_path:
                    ki_count = len(r.get("exercise_details", []))
                    counts[course] = counts.get(course, 0) + ki_count
                    break
        return counts

    def _run_cross_batch_merge(self, course_folder: str):
        """Run cross-batch merge for accumulated items in a course (parallel mode)."""
        items = self._course_items.get(course_folder, [])
        if len(items) < 2:
            return

        if not self.args.quiet:
            print(f"\n{cyan('Cross-batch merge')} for {course_folder}: {len(items)} items")

        try:
            self.tester._init_llm()

            # Use classify_items to find groups
            final_groups, _ = classify_items(items, [], self.tester.llm)

            found_cross_pdf = False
            for group in final_groups:
                group_items_list = group.get("items", [])
                if len(group_items_list) < 2:
                    continue

                # Check if group spans multiple PDFs
                pdfs_in_group = set(item.get("pdf", "?") for item in group_items_list)
                if len(pdfs_in_group) < 2:
                    continue  # Internal merge, already handled

                found_cross_pdf = True
                group_names = [item["name"] for item in group_items_list]
                canonical = group.get("name", get_canonical_name(group_names, self.tester.llm))

                members = []
                for item in group_items_list:
                    members.append(
                        {
                            "name": item["name"],
                            "description": item.get("description", ""),
                            "pdf": item.get("pdf", "?"),
                        }
                    )

                self.summary.cross_batch_groups.append(
                    {
                        "course": course_folder,
                        "canonical": canonical,
                        "members": members,
                    }
                )

                if not self.args.quiet:
                    print(f"  {bold(canonical)}")
                    for m in members:
                        print(f"    ← {m['name']} ({m['pdf']})")

            if not found_cross_pdf and not self.args.quiet:
                print("  No cross-PDF duplicates found")

        except Exception as e:
            print(red(f"  Cross-batch merge failed: {e}"))

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
                print(
                    green(
                        f"  PASS: total={actual['total']}, parents={actual['parents']}, subs={actual['subs']}"
                    )
                )

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

        # Apply sampling if requested
        sample_n = getattr(self.args, 'sample', None)
        if sample_n and sample_n > 0 and len(pdfs) > sample_n:
            original_count = len(pdfs)
            seed = getattr(self.args, 'seed', None)
            if seed is not None:
                random.seed(seed)
            pdfs = random.sample(pdfs, sample_n)
            if not self.args.quiet:
                print(f"Sampled {sample_n} PDFs from {original_count} total" +
                      (f" (seed={seed})" if seed is not None else ""))

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
                eta_str = f"(~{remaining / 60:.0f} min remaining) "
            else:
                eta_str = f"(~{remaining:.0f}s remaining) "

        rel_path = (
            pdf_path.relative_to(TEST_DATA_PATH)
            if TEST_DATA_PATH in pdf_path.parents
            else pdf_path.name
        )
        print(f"\n{bold(f'[{current + 1}/{total}]')} {cyan(eta_str)}{rel_path}")

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
            if stats["total"] > 0:
                print(
                    f"  LLM calls: {stats['total']} ({stats['hits']} cached, {stats['misses']} new)"
                )

        for warning in result.warnings:
            print(f"  {yellow('Warning')}: {warning}")

        # Verbose: show exercise details
        if self.args.verbose and result.exercise_details:
            print()
            for ex in result.exercise_details:
                if ex["is_sub"]:
                    prefix = "    "
                    num = f"{ex['parent']}.{ex['number']}" if ex["parent"] else ex["number"]
                else:
                    prefix = "  "
                    num = ex["number"]
                    if ex.get("context"):
                        print(f'{prefix}Context: "{ex["context"][:60]}..."')

                sol_marker = " [+sol]" if ex["has_solution"] else ""
                print(f'{prefix}{num}{sol_marker}: "{ex["text_preview"]}"')

        # Show categories discovered
        if result.categories:
            print(f"\n  {cyan('Categories')}: {', '.join(sorted(result.categories))}")

        # Show skill groups (--full mode)
        if result.skill_groups:
            # Separate internal vs cross-batch-inline groups
            internal_groups = [g for g in result.skill_groups if g.get("type") == "internal"]
            cross_inline_groups = [
                g for g in result.skill_groups if g.get("type") == "cross-batch-inline"
            ]

            if internal_groups:
                print(
                    f"\n  {cyan('Merged Groups')} ({len(internal_groups)} within this PDF):"
                )
                for group in internal_groups:
                    canonical = group["canonical"]
                    category = group.get("category", "")
                    description = group.get("description", "")

                    # Header with category
                    cat_label = f" [{category}]" if category else ""
                    print(f"    {bold(canonical)}{dim(cat_label)}")

                    # Description
                    if description:
                        print(f"      {dim(self._truncate_output(description, 70))}")

                    # Members
                    for member in group["members"]:
                        name = member["name"]
                        member_desc = member.get("description", "")
                        print(f"      ← {name}")
                        if member_desc and self.args.verbose:
                            print(f"        {dim(self._truncate_output(member_desc, 60))}")

            if cross_inline_groups:
                print(
                    f"\n  {yellow('Cross-PDF Merges')} ({len(cross_inline_groups)} groups):"
                )
                for group in cross_inline_groups:
                    canonical = group["canonical"]
                    print(f"    {bold(canonical)}")
                    for member in group["members"]:
                        name = member["name"]
                        pdf = member.get("pdf", "?")
                        print(f"      ← {name} ({pdf})")

        # Show knowledge items with descriptions
        if result.knowledge_items:
            print(f"\n  {cyan('Knowledge Items')} ({len(result.knowledge_items)}):")
            for item in result.knowledge_items:
                name = item.get("name", "?")
                desc = item.get("description", "")
                cat = item.get("category", "")
                cat_label = f" [{cat}]" if cat else ""
                print(f"    • {name}{dim(cat_label)}")
                if desc and self.args.verbose:
                    print(f"      {dim(self._truncate_output(desc, 65))}")

        # Show active learning stats
        if result.active_learning_stats:
            stats = result.active_learning_stats
            total = stats.get("total", 0)
            if total > 0:
                llm_rate = stats.get("llm_call_rate", 1.0)
                pred_rate = stats.get("prediction_rate", 0.0)
                print(f"\n  {cyan('Active Learning')}:")
                print(f"    LLM calls: {stats.get('llm_calls', 0)}/{total} ({llm_rate:.0%})")
                print(f"    Predictions: {stats.get('predictions', 0)}/{total} ({pred_rate:.0%})")
                print(f"    Transitive: {stats.get('transitive_inferences', 0)}")
                print(f"    Training samples: {stats.get('training_samples', 0)}")

        # Show feature similarities (--show-features)
        if result.feature_similarities:
            print(f"\n  {cyan('Embedding Similarities')} (top pairs ≥0.5):")
            for pair in result.feature_similarities:
                sim = pair["similarity"]
                # Color code: green if likely match, yellow if borderline
                if sim >= 0.75:
                    sim_str = green(f"{sim:.2f}")
                elif sim >= 0.6:
                    sim_str = yellow(f"{sim:.2f}")
                else:
                    sim_str = dim(f"{sim:.2f}")
                name_a = self._truncate_output(pair["item_a"], 25)
                name_b = self._truncate_output(pair["item_b"], 25)
                print(f"    {sim_str}  {name_a} ↔ {name_b}")

    def _truncate_output(self, text: str, max_len: int = 60) -> str:
        """Truncate text for output."""
        text = text.replace("\n", " ").strip()
        if len(text) <= max_len:
            return text
        return f"{text[: max_len - 3]}..."

    def _record_result(self, result: TestResult):
        """Record result in summary."""
        self.summary.total += 1
        self.summary.results.append(result)

        if result.status == "PASS":
            self.summary.passed += 1
        elif result.status == "FAIL":
            self.summary.failed += 1
        elif result.status == "TIMEOUT":
            self.summary.timeouts += 1
        else:
            self.summary.errors += 1

        self.summary.warnings.extend(result.warnings)

        # Accumulate active learning stats
        if result.active_learning_stats:
            stats = result.active_learning_stats
            self._total_al_llm_calls += stats.get("llm_calls", 0)
            self._total_al_predictions += stats.get("predictions", 0)
            self._total_al_transitive += stats.get("transitive_inferences", 0)
            self._total_al_training += stats.get("training_samples", 0)

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
            print(
                f"\nExercises: {total_exercises} total ({total_subs} subs), avg {avg_exercises:.1f}/PDF"
            )

        # LLM stats (accumulated across all PDFs)
        total_llm = self._total_llm_hits + self._total_llm_misses
        if total_llm > 0:
            hit_rate = (self._total_llm_hits / total_llm) * 100
            print(
                f"LLM calls: {total_llm} ({self._total_llm_hits} cached, {self._total_llm_misses} new, {hit_rate:.0f}% hit rate)"
            )

        # Cost estimate if requested
        if getattr(self.args, 'show_cost', False) and self._total_llm_misses > 0:
            self._print_cost_estimate()

        # Active learning summary stats
        total_al = self._total_al_llm_calls + self._total_al_predictions + self._total_al_transitive
        if total_al > 0:
            al_reduction = 1 - (self._total_al_llm_calls / total_al) if total_al > 0 else 0
            print(f"\n{cyan('Active Learning Summary')}:")
            print(f"  Total classifications: {total_al}")
            print(f"  LLM calls: {self._total_al_llm_calls} ({self._total_al_llm_calls/total_al:.0%})")
            print(f"  ML predictions: {self._total_al_predictions} ({self._total_al_predictions/total_al:.0%})")
            print(f"  Transitive: {self._total_al_transitive} ({self._total_al_transitive/total_al:.0%})")
            print(f"  Training samples: {self._total_al_training}")
            print(f"  LLM reduction: {green(f'{al_reduction:.0%}')}")

        # Cross-batch merge summary (parallel mode)
        if self.summary.cross_batch_groups:
            print(
                f"\n{cyan('Cross-batch merges')}: {len(self.summary.cross_batch_groups)} groups found"
            )
            for group in self.summary.cross_batch_groups:
                print(f"  [{group['course']}] {group['canonical']}")
                for m in group["members"]:
                    print(f"    ← {m['name']} ({m['pdf']})")

        # Knowledge items per course summary
        if self._course_items and self.args.full:
            print(f"\n{cyan('Knowledge items per course')}:")
            for course, items in sorted(self._course_items.items()):
                print(f"  {course}: {len(items)} items")

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

    def _print_cost_estimate(self):
        """Print estimated LLM API cost based on call counts."""
        # Estimated average tokens per LLM call (conservative estimates)
        # Based on typical exam analysis prompts/responses
        AVG_INPUT_TOKENS = 1500   # ~1.5K input (prompt + context)
        AVG_OUTPUT_TOKENS = 500   # ~500 output (structured response)

        # Get pricing (default to deepseek-chat)
        rates = LLM_COSTS.get("default", {"input": 0.28, "output": 0.42})

        # Calculate estimated tokens
        llm_calls = self._total_llm_misses  # Only non-cached calls cost money
        est_input_tokens = llm_calls * AVG_INPUT_TOKENS
        est_output_tokens = llm_calls * AVG_OUTPUT_TOKENS

        # Calculate cost (per 1M tokens)
        input_cost = est_input_tokens / 1_000_000 * rates["input"]
        output_cost = est_output_tokens / 1_000_000 * rates["output"]
        total_cost = input_cost + output_cost

        # Cost per PDF (for non-cached PDFs)
        cost_per_pdf = total_cost / max(self.summary.total, 1)

        print(f"\n{cyan('Cost Estimate')} (DeepSeek pricing):")
        print(f"  LLM calls (non-cached): {llm_calls}")
        print(f"  Est. input tokens:  ~{est_input_tokens:,} (${input_cost:.4f})")
        print(f"  Est. output tokens: ~{est_output_tokens:,} (${output_cost:.4f})")
        print(f"  {bold('Total estimate:')} ${total_cost:.4f}")
        print(f"  Per PDF: ${cost_per_pdf:.4f}")

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

    def _save_training_data(self):
        """Save training data if requested or persist is enabled."""
        if not self.tester.active_classifier:
            return

        # Determine output path
        save_path = None
        if getattr(self.args, 'save_training', None):
            save_path = Path(self.args.save_training)
        elif getattr(self.args, 'persist', False):
            save_path = TRAINING_CACHE_PATH
            save_path.parent.mkdir(parents=True, exist_ok=True)

        if save_path:
            try:
                data = self.tester.active_classifier.export_training_data()
                with open(save_path, 'w') as f:
                    json.dump(data, f, indent=2)
                if not self.args.quiet:
                    print(f"Training data saved to {save_path} ({data['samples']} samples)")
            except Exception as e:
                print(f"{yellow('Warning')}: Failed to save training data: {e}")

    def _generate_html_report(self, output_path: str):
        """Generate HTML report with visual elements."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Pipeline Test Report - {timestamp}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white;
                     padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 2px solid #e0e0e0; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .summary {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 20px 0; }}
        .stat {{ background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; }}
        .stat-value {{ font-size: 32px; font-weight: bold; color: #333; }}
        .stat-label {{ color: #666; margin-top: 5px; }}
        .pass {{ color: #28a745; }}
        .fail {{ color: #dc3545; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #e0e0e0; }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        tr:hover {{ background: #f8f9fa; }}
        .sim-high {{ background: #28a745; color: white; }}
        .sim-med {{ background: #ffc107; }}
        .sim-low {{ background: #f8f9fa; }}
        .collapsible {{ cursor: pointer; padding: 10px; background: #f8f9fa;
                       border-radius: 4px; margin: 5px 0; }}
        .collapsible:hover {{ background: #e9ecef; }}
        .content {{ display: none; padding: 10px; background: #fff; border: 1px solid #e0e0e0; }}
        .content.show {{ display: block; }}
    </style>
</head>
<body>
<div class="container">
    <h1>Pipeline Test Report</h1>
    <p>Generated: {timestamp}</p>

    {self._html_summary_section()}
    {self._html_active_learning_section()}
    {self._html_results_section()}
    {self._html_similarity_section()}

</div>
<script>
document.querySelectorAll('.collapsible').forEach(el => {{
    el.addEventListener('click', () => {{
        el.nextElementSibling.classList.toggle('show');
    }});
}});
</script>
</body>
</html>"""

        Path(output_path).write_text(html)
        print(f"\nHTML report saved to: {output_path}")

        # Auto-open in browser (unless disabled or headless)
        if not getattr(self.args, 'no_open', False) and sys.stdout.isatty():
            try:
                webbrowser.open(f"file://{Path(output_path).absolute()}")
            except Exception:
                pass  # Silently fail if can't open browser

    def _generate_junit_report(self, output_path: str):
        """Generate JUnit XML report for CI integration."""
        # Group results by course
        by_course: dict[str, list] = {}
        for result in self.summary.results:
            course = "unknown"
            for c in COURSES.keys():
                if c in result.pdf_path:
                    course = c
                    break
            if course not in by_course:
                by_course[course] = []
            by_course[course].append(result)

        # Build XML
        testsuites = Element("testsuites")
        testsuites.set("name", "Pipeline Tests")
        testsuites.set("tests", str(self.summary.total))
        testsuites.set("failures", str(self.summary.failed))
        testsuites.set("errors", str(self.summary.errors + self.summary.timeouts))
        testsuites.set("time", f"{self.summary.duration:.1f}")

        for course, results in sorted(by_course.items()):
            suite = SubElement(testsuites, "testsuite")
            suite.set("name", course)
            suite.set("tests", str(len(results)))
            suite.set("failures", str(sum(1 for r in results if r.status == "FAIL")))
            suite.set("errors", str(sum(1 for r in results if r.status in ("ERROR", "TIMEOUT"))))
            suite.set("time", f"{sum(r.duration for r in results):.1f}")

            for result in results:
                testcase = SubElement(suite, "testcase")
                testcase.set("name", Path(result.pdf_path).name)
                testcase.set("classname", course)
                testcase.set("time", f"{result.duration:.1f}")

                if result.status == "FAIL":
                    failure = SubElement(testcase, "failure")
                    failure.set("message", result.error or "Test failed")
                    failure.text = result.error or ""
                elif result.status == "ERROR":
                    error = SubElement(testcase, "error")
                    error.set("message", result.error or "Error occurred")
                    error.text = result.error or ""
                elif result.status == "TIMEOUT":
                    error = SubElement(testcase, "error")
                    error.set("message", "Timeout")
                    error.text = result.error or "PDF processing timed out"

        # Pretty print
        xml_str = minidom.parseString(tostring(testsuites)).toprettyxml(indent="  ")
        # Remove XML declaration duplicate and extra blank lines
        lines = [line for line in xml_str.split("\n") if line.strip()]
        xml_str = "\n".join(lines)

        Path(output_path).write_text(xml_str)
        print(f"JUnit XML report saved to: {output_path}")

    def _html_summary_section(self) -> str:
        """Generate summary stats section."""
        return f"""
    <div class="summary">
        <div class="stat">
            <div class="stat-value">{self.summary.total}</div>
            <div class="stat-label">PDFs Tested</div>
        </div>
        <div class="stat">
            <div class="stat-value pass">{self.summary.passed}</div>
            <div class="stat-label">Passed</div>
        </div>
        <div class="stat">
            <div class="stat-value fail">{self.summary.failed + self.summary.errors}</div>
            <div class="stat-label">Failed</div>
        </div>
        <div class="stat">
            <div class="stat-value">{self.summary.duration:.1f}s</div>
            <div class="stat-label">Duration</div>
        </div>
    </div>"""

    def _html_active_learning_section(self) -> str:
        """Generate active learning stats section."""
        total = self._total_al_llm_calls + self._total_al_predictions + self._total_al_transitive
        if total == 0:
            return ""

        reduction = 1 - (self._total_al_llm_calls / total) if total > 0 else 0
        return f"""
    <h2>Active Learning Performance</h2>
    <div class="summary">
        <div class="stat">
            <div class="stat-value">{total}</div>
            <div class="stat-label">Classifications</div>
        </div>
        <div class="stat">
            <div class="stat-value">{self._total_al_llm_calls}</div>
            <div class="stat-label">LLM Calls</div>
        </div>
        <div class="stat">
            <div class="stat-value">{self._total_al_predictions}</div>
            <div class="stat-label">ML Predictions</div>
        </div>
        <div class="stat">
            <div class="stat-value pass">{reduction:.0%}</div>
            <div class="stat-label">LLM Reduction</div>
        </div>
    </div>"""

    def _html_results_section(self) -> str:
        """Generate per-PDF results section."""
        rows = []
        for r in self.summary.results:
            pdf_name = Path(r.pdf_path).name
            status_class = "pass" if r.status == "PASS" else "fail"
            n_kis = len(r.knowledge_items) if r.knowledge_items else 0
            n_groups = len(r.skill_groups) if r.skill_groups else 0

            rows.append(f"""
        <tr>
            <td>{pdf_name}</td>
            <td class="{status_class}">{r.status}</td>
            <td>{r.pages}</td>
            <td>{r.exercises}</td>
            <td>{n_kis}</td>
            <td>{n_groups}</td>
            <td>{r.duration:.1f}s</td>
        </tr>""")

        return f"""
    <h2>Results by PDF</h2>
    <table>
        <thead>
            <tr>
                <th>PDF</th>
                <th>Status</th>
                <th>Pages</th>
                <th>Exercises</th>
                <th>KIs</th>
                <th>Merged</th>
                <th>Time</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>"""

    def _html_similarity_section(self) -> str:
        """Generate similarity table section."""
        # Collect all similarity pairs across all results
        all_pairs = []
        for r in self.summary.results:
            if r.feature_similarities:
                for pair in r.feature_similarities:
                    all_pairs.append({
                        "pdf": Path(r.pdf_path).name,
                        **pair
                    })

        if not all_pairs:
            return ""

        # Sort by similarity
        all_pairs.sort(key=lambda x: x["similarity"], reverse=True)

        rows = []
        for pair in all_pairs[:20]:  # Top 20
            sim = pair["similarity"]
            sim_class = "sim-high" if sim >= 0.75 else "sim-med" if sim >= 0.6 else "sim-low"
            rows.append(f"""
        <tr>
            <td>{pair['pdf']}</td>
            <td>{pair['item_a'][:40]}</td>
            <td>{pair['item_b'][:40]}</td>
            <td class="{sim_class}" style="font-weight:bold">{sim:.2f}</td>
        </tr>""")

        return f"""
    <h2>Top Embedding Similarities</h2>
    <table>
        <thead>
            <tr>
                <th>PDF</th>
                <th>Item A</th>
                <th>Item B</th>
                <th>Similarity</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>"""

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
                lines.append(
                    f"{status} | {r.pages} pages | {r.exercises} exercises ({r.sub_questions} subs)"
                )

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
                            parent_info[ex["number"]] = ex
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
                            ctx = (
                                parent_ex["context"][:60] + "..."
                                if len(parent_ex.get("context", "")) > 60
                                else parent_ex.get("context", "")
                            )
                            lines.append(f"    {ctx}")
                        # Show sub-questions
                        for sub in subs:
                            sub_sol = " [+sol]" if sub["has_solution"] else ""
                            prefix = f"    - {sub['number']}:{sub_sol} "
                            wrapped = textwrap.fill(
                                sub["text_preview"],
                                width=78,
                                initial_indent=prefix,
                                subsequent_indent=" " * len(prefix),
                            )
                            # Add continuation marker at end of lines that wrap
                            wrap_lines = wrapped.split("\n")
                            if len(wrap_lines) > 1:
                                wrap_lines = [
                                    line + "-" if i < len(wrap_lines) - 1 else line
                                    for i, line in enumerate(wrap_lines)
                                ]
                            lines.append("\n".join(wrap_lines))

                    # Print standalone exercises (no subs)
                    for parent_ex in standalone:
                        parent_num = parent_ex["number"]
                        if parent_num in printed_parents:
                            continue  # Already printed as grouped
                        sol = " [+sol]" if parent_ex["has_solution"] else ""
                        prefix = f"  - Ex {parent_num}:{sol} "
                        wrapped = textwrap.fill(
                            parent_ex["text_preview"],
                            width=78,
                            initial_indent=prefix,
                            subsequent_indent=" " * len(prefix),
                        )
                        # Add continuation marker at end of lines that wrap
                        wrap_lines = wrapped.split("\n")
                        if len(wrap_lines) > 1:
                            wrap_lines = [
                                line + "-" if i < len(wrap_lines) - 1 else line
                                for i, line in enumerate(wrap_lines)
                            ]
                        lines.append("\n".join(wrap_lines))
                lines.append("")

            # Mode: analyze - exercises with KI names + descriptions
            elif self.args.analyze:
                lines.append(f"═══ {pdf_name} ═══")
                lines.append(
                    f"{status} | {r.pages} pages | {r.exercises} exercises ({r.sub_questions} subs)"
                )

                # Group by KI name for cleaner output
                ki_to_exercises: dict[str, list] = {}
                for ex in r.exercise_details:
                    ki_name = ex.get("ki_name", "unknown")
                    if ki_name not in ki_to_exercises:
                        ki_to_exercises[ki_name] = []
                    ki_to_exercises[ki_name].append(ex)

                if ki_to_exercises:
                    lines.append("")
                    lines.append("Knowledge Items:")
                    for ki_name, exercises in sorted(ki_to_exercises.items()):
                        # Get description from first exercise
                        ki_desc = exercises[0].get("ki_description", "")
                        lines.append(f"  • {ki_name} ({len(exercises)} exercises)")
                        if ki_desc:
                            lines.append(f"    {ki_desc[:70]}...")

                        # Show exercises under this KI
                        for ex in exercises:
                            num = ex.get("number", "?")
                            sol = " [+sol]" if ex.get("has_solution") else ""
                            preview = ex.get("text_preview", "")[:50]
                            lines.append(f"      [{num}]{sol} {preview}...")
                lines.append("")

            # Mode: full - KIs grouped by category + skill groups
            else:
                n_kis = len(r.knowledge_items) if r.knowledge_items else 0
                n_groups = len(r.skill_groups) if r.skill_groups else 0
                lines.append(f"═══ {pdf_name} ═══")
                lines.append(
                    f"{status} | {r.pages} pages | {r.exercises} exercises → {n_kis} KIs"
                )

                # Group KIs by category
                if r.knowledge_items:
                    lines.append("")
                    lines.append("Knowledge Items:")
                    cats_to_items: dict[str, list] = {}
                    for item in r.knowledge_items:
                        cat = item.get("category", "Uncategorized")
                        if cat not in cats_to_items:
                            cats_to_items[cat] = []
                        cats_to_items[cat].append(item)

                    for cat, items in sorted(cats_to_items.items()):
                        lines.append(f"  [{cat}]")
                        for item in items:
                            name = item.get("name", "?")
                            desc = item.get("description", "")[:60]
                            lines.append(f"    • {name}")
                            if desc:
                                lines.append(f"      {desc}...")

                # Merged skill groups
                if r.skill_groups:
                    lines.append("")
                    lines.append(f"Merged Groups ({n_groups}):")
                    for group in r.skill_groups:
                        canonical = group.get("canonical", "Unknown")
                        category = group.get("category", "")
                        cat_label = f" [{category}]" if category else ""
                        lines.append(f"  ► {canonical}{cat_label}")
                        for member in group.get("members", []):
                            name = member.get("name", "?")
                            lines.append(f"      ← {name}")

                # Feature similarities
                if r.feature_similarities:
                    lines.append("")
                    lines.append("Embedding Similarities (≥0.5):")
                    for pair in r.feature_similarities[:5]:  # Top 5
                        sim = pair["similarity"]
                        lines.append(
                            f"  {sim:.2f}  {pair['item_a'][:25]} ↔ {pair['item_b'][:25]}"
                        )

                # Active learning stats
                if r.active_learning_stats:
                    stats = r.active_learning_stats
                    total = stats.get("total", 0)
                    if total > 0:
                        llm_calls = stats.get("llm_calls", 0)
                        preds = stats.get("predictions", 0)
                        trans = stats.get("transitive_inferences", 0)
                        lines.append("")
                        lines.append(
                            f"Active Learning: {llm_calls} LLM, {preds} ML, {trans} transitive"
                        )

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
  %(prog)s --smoke                     Quick test (1 PDF/course)
  %(prog)s --smoke --full -j 4         Full pipeline, 4 parallel workers
  %(prog)s --all --sample 10           Random sample of 10 PDFs
  %(prog)s --course ADE --full         All PDFs in one course
  %(prog)s --golden                    Run regression tests

Performance:
  -j N, --jobs N           Parallel PDF processing (requires --full)
  --sample N               Random N PDFs (--seed for reproducibility)
  --timeout SEC            Per-PDF timeout (default: 300s, 0=none)

Output:
  --html PATH              HTML report (auto-opens in browser)
  --junit PATH             JUnit XML for CI (GitHub Actions, Jenkins)
  --json / --txt           Alternative formats with --save
  --show-cost              Estimate LLM costs (DeepSeek pricing)

CI/CD:
  --compare OLD NEW        Detect regressions (exit 1 if any)
  --junit PATH             Standard JUnit XML output
  --timeout SEC            Prevent stuck tests

Active Learning:
  --with-active-learning   ML reduces LLM calls 70-90%% (default ON)
  --persist                Auto-save training (default ON)
  --load-training PATH     Warm start from exported data
        """,
    )

    # Run modes
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--smoke", action="store_true", help="Quick test: 1 PDF per course")
    mode_group.add_argument("--course", metavar="NAME", help="Test all PDFs in one course")
    mode_group.add_argument("--all", action="store_true", help="Test all PDFs (default)")
    mode_group.add_argument("--pdf", metavar="PATH", help="Test single PDF file")
    mode_group.add_argument("--golden", action="store_true", help="Run golden regression tests")
    mode_group.add_argument(
        "--rerun-failed", action="store_true", help="Rerun failed tests from last run"
    )

    # Pipeline depth
    depth_group = parser.add_mutually_exclusive_group()
    depth_group.add_argument("--parse", action="store_true", help="Test parse only (stage 1)")
    depth_group.add_argument(
        "--split", action="store_true", help="Test parse + split (stages 1-2)"
    )
    depth_group.add_argument(
        "--analyze", action="store_true", help="Test parse + split + analyze (stages 1-3, default)"
    )
    depth_group.add_argument(
        "--full", action="store_true", help="Full pipeline + skill grouping (shows merges)"
    )

    # Processing mode (for --full with multiple PDFs)
    proc_group = parser.add_mutually_exclusive_group()
    proc_group.add_argument(
        "--parallel",
        action="store_true",
        help="Pro mode: Independent PDFs + cross-batch merge at end (default)",
    )
    proc_group.add_argument(
        "--sequential",
        action="store_true",
        help="Free mode: Sequential, each PDF sees previous items inline",
    )
    proc_group.add_argument(
        "--independent",
        action="store_true",
        help="Independent PDFs, no cross-batch merge (internal merge only)",
    )

    # Active learning (ON by default)
    al_group = parser.add_mutually_exclusive_group()
    al_group.add_argument(
        "--with-active-learning",
        action="store_true",
        default=True,
        help="Use active learning classifier (default: ON)",
    )
    al_group.add_argument(
        "--no-active-learning",
        action="store_true",
        help="Disable active learning (use pure LLM)",
    )
    parser.add_argument(
        "--show-features",
        action="store_true",
        help="Show embedding similarity and feature scores",
    )

    # Training data persistence (ON by default)
    parser.add_argument(
        "--load-training",
        metavar="PATH",
        help="Load training data from JSON file (warm start)",
    )
    parser.add_argument(
        "--save-training",
        metavar="PATH",
        help="Save training data to JSON file after run",
    )
    persist_group = parser.add_mutually_exclusive_group()
    persist_group.add_argument(
        "--persist",
        action="store_true",
        default=True,
        help="Auto-save/load training from cache (default: ON)",
    )
    persist_group.add_argument(
        "--no-persist",
        action="store_true",
        help="Disable training persistence (cold start each run)",
    )

    # Benchmark mode
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run twice (with/without AL) and compare performance",
    )

    # Output options
    parser.add_argument("--save", action="store_true", help="Save results to test-results/ (HTML by default)")
    parser.add_argument("--html", metavar="PATH", help="Generate HTML report at specified path")
    parser.add_argument("--no-open", action="store_true", help="Don't auto-open HTML report in browser")
    parser.add_argument("--json", action="store_true", help="Output as JSON instead of HTML (with --save)")
    parser.add_argument("--txt", action="store_true", help="Output as plain text instead of HTML (with --save)")
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("OLD", "NEW"),
        help="Compare two JSON result files and show regressions",
    )
    parser.add_argument("--failures-only", action="store_true", help="Show only failed tests")
    parser.add_argument("--timing", action="store_true", help="Show timing per PDF")
    parser.add_argument("--lang", default="en", help="Language for analysis (default: en)")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would run without running"
    )
    parser.add_argument("--skip", action="append", metavar="PDF", help="Skip specific PDF(s)")
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=1,
        metavar="N",
        help="Concurrent PDF processing threads (default: 1, requires --full)",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar")
    parser.add_argument("-q", "--quiet", action="store_true", help="Summary only")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show full exercise details")
    parser.add_argument("--debug", action="store_true", help="Show all LLM cache messages")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")

    # New v3 features
    parser.add_argument(
        "--sample",
        type=int,
        metavar="N",
        help="Randomly sample N PDFs from the test set",
    )
    parser.add_argument(
        "--seed",
        type=int,
        metavar="N",
        help="Random seed for --sample (for reproducibility)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        metavar="SEC",
        help="Timeout per PDF in seconds (default: 300, 0=no limit)",
    )
    parser.add_argument(
        "--junit",
        metavar="PATH",
        help="Export results to JUnit XML format (for CI)",
    )
    parser.add_argument(
        "--show-cost",
        action="store_true",
        help="Show estimated LLM API costs (DeepSeek pricing)",
    )

    args = parser.parse_args()

    # Default to --all if no mode specified
    if not any([args.smoke, args.course, args.all, args.pdf, args.golden, args.rerun_failed]):
        args.all = True

    # Default to --analyze if no depth specified (changed from --split)
    if not any([args.parse, args.split, args.analyze, args.full]):
        args.analyze = True

    # Handle active learning toggle
    if args.no_active_learning:
        args.with_active_learning = False

    # Handle persist toggle
    if args.no_persist:
        args.persist = False

    # Disable colors if requested or not a TTY
    if args.no_color or not sys.stdout.isatty():
        Colors.disable()

    return args


def main():
    """Main entry point."""
    args = parse_args()

    runner = TestRunner(args)

    if getattr(args, 'compare', None):
        exit_code = runner.run_compare()
    elif getattr(args, 'benchmark', False):
        exit_code = runner.run_benchmark()
    elif args.golden:
        exit_code = runner.run_golden()
    else:
        exit_code = runner.run()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
