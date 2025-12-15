#!/usr/bin/env python3
"""
Examina - AI-powered exam tutor system.
CLI interface for managing courses, ingesting exams, and studying.
"""

import asyncio
import click
import json
from rich.console import Console
from rich.table import Table

from config import Config
from storage.database import Database
from storage.file_manager import FileManager
from storage.vector_store import VectorStore
from core.pdf_processor import PDFProcessor
from core.exercise_splitter import ExerciseSplitter
from core.analyzer import ExerciseAnalyzer, AnalysisResult
from models.llm_manager import LLMManager
from scripts.study_context import study_plan

console = Console()


def get_effective_provider(provider, profile, task_type_value):
    """Helper to determine effective provider based on --provider and --profile flags.

    Args:
        provider: Provider from --provider flag (None if not specified)
        profile: Profile from --profile flag (None if not specified)
        task_type_value: Task type string (e.g., "bulk_analysis", "interactive")

    Returns:
        Provider name to use
    """
    if provider is not None:
        # Provider flag explicitly specified, use it
        return provider
    elif profile is not None:
        # Profile specified, use routing
        from core.provider_router import ProviderRouter
        from core.task_types import TaskType

        try:
            router = ProviderRouter()
            task_type = TaskType.from_string(task_type_value)
            effective = router.route(task_type, profile)
            console.print(f"[dim]Using profile '{profile}' → provider: {effective}[/dim]")
            return effective
        except Exception as e:
            console.print(f"[yellow]Warning: Routing failed: {e}[/yellow]")
            console.print(f"[dim]Falling back to default provider: {Config.LLM_PROVIDER}[/dim]")
            return Config.LLM_PROVIDER
    else:
        # Neither specified, use default
        return Config.LLM_PROVIDER


@click.group()
@click.version_option(version="0.1.0", prog_name="Examina")
def cli():
    """Examina - AI-powered exam tutor for mastering university courses."""
    pass


@cli.command()
def init():
    """Initialize Examina database and load course catalog."""
    console.print("\n[bold cyan]Initializing Examina...[/bold cyan]\n")

    try:
        # Create directories
        console.print("📁 Creating directories...")
        Config.ensure_dirs()
        console.print("   ✓ Directories created\n")

        # Initialize database
        console.print("🗄️  Creating database schema...")
        with Database() as db:
            db.initialize()
            console.print("   ✓ Database schema created\n")

            # Load courses from study_context.py
            console.print("📚 Loading course catalog...")
            courses_added = 0

            for degree_type, courses in study_plan.items():
                level = "bachelor" if "bachelor" in degree_type else "master"
                program = "L-31" if level == "bachelor" else "LM-18"

                for course in courses:
                    db.add_course(
                        code=course["code"],
                        name=course["name"],
                        original_name=course.get("original_name"),
                        acronym=course["acronym"],
                        degree_level=level,
                        degree_program=program,
                    )
                    courses_added += 1

            db.conn.commit()
            console.print(f"   ✓ Loaded {courses_added} courses\n")

        # Summary
        console.print("[bold green]✨ Examina initialized successfully![/bold green]\n")
        console.print(f"Database: {Config.DB_PATH}")
        console.print(f"Data directory: {Config.DATA_DIR}\n")
        console.print("Next steps:")
        console.print("  • examina courses - View available courses")
        console.print("  • examina ingest --course <CODE> --zip <FILE> - Import exam PDFs")
        console.print("  • examina --help - See all commands\n")

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}\n")
        raise click.Abort()


@cli.command()
@click.option(
    "--degree",
    type=click.Choice(["bachelor", "master", "all"]),
    default="all",
    help="Filter by degree level",
)
def courses(degree):
    """List all available courses."""
    try:
        with Database() as db:
            all_courses = db.get_all_courses()

        if not all_courses:
            console.print("\n[yellow]No courses found. Run 'examina init' first.[/yellow]\n")
            return

        # Filter by degree if specified
        if degree != "all":
            all_courses = [c for c in all_courses if c["degree_level"] == degree]

        # Create table
        table = Table(
            title=f"\n📚 Available Courses ({degree.title()})"
            if degree != "all"
            else "\n📚 Available Courses"
        )
        table.add_column("Code", style="cyan", no_wrap=True)
        table.add_column("Acronym", style="magenta")
        table.add_column("Name", style="white")
        table.add_column("Level", style="green")

        for course in all_courses:
            table.add_row(
                course["code"],
                course["acronym"] or "",
                course["name"],
                course["degree_level"].title(),
            )

        console.print(table)
        console.print(f"\nTotal: {len(all_courses)} courses\n")

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}\n")
        raise click.Abort()


@cli.command()
@click.option("--course", "-c", required=True, help="Course code (e.g., B006802 or ADE)")
def info(course):
    """Show detailed information about a course."""
    try:
        with Database() as db:
            # Try to find course by code or acronym
            all_courses = db.get_all_courses()
            found_course = None

            for c in all_courses:
                if c["code"] == course or c["acronym"] == course:
                    found_course = c
                    break

            if not found_course:
                console.print(f"\n[red]Course '{course}' not found.[/red]\n")
                console.print("Use 'examina courses' to see available courses.\n")
                return

            # Get topics and stats
            topics = db.get_topics_by_course(found_course["code"])
            exercises = db.get_exercises_by_course(found_course["code"])

            # Display info
            console.print(f"\n[bold cyan]{found_course['name']}[/bold cyan]")
            if found_course["original_name"]:
                console.print(f"[dim]{found_course['original_name']}[/dim]")

            console.print(f"\nCode: {found_course['code']}")
            console.print(f"Acronym: {found_course['acronym']}")
            console.print(
                f"Level: {found_course['degree_level'].title()} ({found_course['degree_program']})"
            )

            console.print(f"\n[bold]Status:[/bold]")
            console.print(f"  Topics discovered: {len(topics)}")
            console.print(f"  Exercises ingested: {len(exercises)}")

            if topics:
                console.print(f"\n[bold]Topics:[/bold]")
                for topic in topics:
                    knowledge_items = db.get_knowledge_items_by_topic(topic["id"])
                    console.print(f"  • {topic['name']} ({len(knowledge_items)} core loops)")

            # Show exercise type breakdown
            from core.proof_tutor import ProofTutor

            proof_tutor = ProofTutor()

            proof_count = 0
            procedural_count = 0
            theory_count = 0

            for ex in exercises:
                text = ex.get("text", "")
                is_proof = proof_tutor.is_proof_exercise(text)
                ex_tags = ex.get("tags")
                ex_tags_str = str(ex_tags) if ex_tags else "[]"

                if is_proof:
                    proof_count += 1
                elif any(
                    tag in ex_tags_str for tag in ["design", "transformation", "implementation"]
                ):
                    procedural_count += 1
                elif any(tag in ex_tags_str for tag in ["analysis", "verification"]):
                    theory_count += 1

            if proof_count > 0 or procedural_count > 0 or theory_count > 0:
                console.print(f"\n[bold]Exercise Type Breakdown:[/bold]")
                if procedural_count > 0:
                    console.print(
                        f"  Procedural: {procedural_count} ({procedural_count * 100 // len(exercises) if exercises else 0}%)"
                    )
                if theory_count > 0:
                    console.print(
                        f"  Theory: {theory_count} ({theory_count * 100 // len(exercises) if exercises else 0}%)"
                    )
                if proof_count > 0:
                    console.print(
                        f"  Proof: {proof_count} ({proof_count * 100 // len(exercises) if exercises else 0}%)"
                    )

            # Show multi-procedure statistics
            multi_proc_exercises = db.get_exercises_with_multiple_procedures(found_course["code"])
            if multi_proc_exercises:
                console.print(f"\n[bold]Multi-Procedure Exercises:[/bold]")
                console.print(
                    f"  {len(multi_proc_exercises)}/{len(exercises)} exercises cover multiple procedures"
                )

                # Show top 3 examples
                console.print(f"\n[bold]Top Examples:[/bold]")
                for ex in multi_proc_exercises[:3]:
                    # Get all core loops for this exercise
                    knowledge_items = db.get_exercise_knowledge_items(ex["id"])
                    console.print(
                        f"  • Exercise {ex['exercise_number'] or ex['id'][:8]}: {ex['knowledge_item_count']} procedures"
                    )
                    for cl in knowledge_items[:3]:  # Show first 3 procedures
                        step_info = f" (point {cl['step_number']})" if cl["step_number"] else ""
                        console.print(f"    - {cl['name']}{step_info}")
                    if ex["knowledge_item_count"] > 3:
                        console.print(f"    ... and {ex['knowledge_item_count'] - 3} more")

            console.print()

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}\n")
        raise click.Abort()


@cli.command()
@click.option("--course", "-c", required=True, help="Course code (e.g., B006802 or ADE)")
@click.option("--mermaid", is_flag=True, help="Output Mermaid diagram format (for rendering)")
@click.option("--show-mastery", is_flag=True, help="Show mastery levels for each concept")
def concept_map(course, mermaid, show_mastery):
    """Visualize topic and core loop concept dependencies."""
    from rich.tree import Tree

    try:
        with Database() as db:
            # Find course
            all_courses = db.get_all_courses()
            found_course = None
            for c in all_courses:
                if c["code"] == course or c["acronym"] == course:
                    found_course = c
                    break

            if not found_course:
                console.print(f"\n[red]Course '{course}' not found.[/red]\n")
                return

            course_code = found_course["code"]
            topics = db.get_topics_by_course(course_code)

            if not topics:
                console.print(f"\n[yellow]No topics found for {found_course['name']}.[/yellow]\n")
                return

            # Get mastery data if requested
            mastery_data = {}
            if show_mastery:
                from core.mastery_aggregator import MasteryAggregator

                aggregator = MasteryAggregator(db)
                weak_loops = aggregator.get_weak_knowledge_items(
                    course_code, threshold=1.0
                )  # Get all
                for wl in weak_loops:
                    mastery_data[wl["knowledge_item_id"]] = wl["mastery_score"]

            if mermaid:
                # Output Mermaid diagram format
                console.print("```mermaid")
                console.print("graph TD")
                console.print(f"    COURSE[{found_course['acronym'] or found_course['code']}]")

                for topic in topics:
                    topic_id = f"T{topic['id']}"
                    topic_name = topic["name"].replace(" ", "_")[:30]
                    console.print(f"    COURSE --> {topic_id}[{topic_name}]")

                    knowledge_items = db.get_knowledge_items_by_topic(topic["id"])
                    for cl in knowledge_items:
                        cl_id = f"CL{cl['id']}"
                        cl_name = cl["name"].replace(" ", "_")[:25]
                        console.print(f"    {topic_id} --> {cl_id}[{cl_name}]")

                console.print("```")

            else:
                # Rich tree view
                tree = Tree(
                    f"[bold cyan]{found_course['name']}[/bold cyan] ({found_course['acronym'] or course_code})",
                    guide_style="dim",
                )

                # Count stats
                total_loops = 0
                theory_count = 0
                procedural_count = 0

                for topic in topics:
                    knowledge_items = db.get_knowledge_items_by_topic(topic["id"])
                    total_loops += len(knowledge_items)

                    # Get exercise type distribution for this topic
                    topic_exercises = db.conn.execute(
                        """
                        SELECT exercise_type, COUNT(*) as count
                        FROM exercises
                        WHERE topic_id = ?
                        GROUP BY exercise_type
                    """,
                        (topic["id"],),
                    ).fetchall()

                    type_info = ""
                    for te in topic_exercises:
                        if te["exercise_type"] == "theory":
                            theory_count += te["count"]
                            type_info += f" [dim](T:{te['count']})[/dim]"
                        elif te["exercise_type"] == "procedural":
                            procedural_count += te["count"]

                    topic_branch = tree.add(
                        f"[bold yellow]{topic['name']}[/bold yellow] ({len(knowledge_items)} loops){type_info}"
                    )

                    for cl in knowledge_items:
                        # Build core loop label with mastery if available
                        cl_label = cl["name"]
                        if show_mastery and cl["id"] in mastery_data:
                            mastery = mastery_data[cl["id"]]
                            if mastery >= 0.7:
                                cl_label = f"[green]{cl['name']}[/green] ✓ {mastery:.0%}"
                            elif mastery >= 0.4:
                                cl_label = f"[yellow]{cl['name']}[/yellow] ◐ {mastery:.0%}"
                            else:
                                cl_label = f"[red]{cl['name']}[/red] ○ {mastery:.0%}"
                        elif show_mastery:
                            cl_label = f"[dim]{cl['name']}[/dim] (new)"

                        topic_branch.add(cl_label)

                console.print()
                console.print(tree)

                # Summary
                console.print()
                console.print(f"[dim]Summary: {len(topics)} topics, {total_loops} core loops[/dim]")
                if theory_count > 0 or procedural_count > 0:
                    console.print(
                        f"[dim]Exercise types: {procedural_count} procedural, {theory_count} theory[/dim]"
                    )
                console.print()

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}\n")
        import traceback

        traceback.print_exc()
        raise click.Abort()


@cli.command()
@click.option("--course", "-c", required=True, help="Course code (e.g., B006802 or ADE)")
@click.option("--tag", "-t", help="Search by procedure tag (design, transformation, etc.)")
@click.option("--text", help="Search by text in exercise content")
@click.option("--multi-only", is_flag=True, help="Only show multi-procedure exercises")
@click.option("--limit", "-l", type=int, default=20, help="Maximum number of results (default: 20)")
def search(course, tag, text, multi_only, limit):
    """Search exercises by tags or content."""
    try:
        with Database() as db:
            # Find course by code or acronym
            all_courses = db.get_all_courses()
            found_course = None
            for c in all_courses:
                if c["code"] == course or c["acronym"] == course:
                    found_course = c
                    break

            if not found_course:
                console.print(f"\n[red]Course '{course}' not found.[/red]\n")
                console.print("Use 'examina courses' to see available courses.\n")
                return

            course_code = found_course["code"]

            # Perform search
            exercises = []
            search_type = []

            if tag:
                exercises = db.get_exercises_by_tag(course_code, tag)
                search_type.append(f"tag '{tag}'")
            elif text:
                exercises = db.search_exercises_by_text(course_code, text)
                search_type.append(f"text '{text}'")
            else:
                # No filter specified, get all exercises
                exercises = db.get_exercises_by_course(course_code)
                search_type.append("all exercises")

            # Filter to multi-procedure only if requested
            if multi_only:
                # Get multi-procedure exercises
                multi_proc_exercises = db.get_exercises_with_multiple_procedures(course_code)
                multi_proc_ids = {ex["id"] for ex in multi_proc_exercises}
                exercises = [ex for ex in exercises if ex["id"] in multi_proc_ids]
                search_type.append("multi-procedure only")

            # Limit results
            exercises = exercises[:limit]

            # Display results
            console.print(f"\n[bold cyan]Search Results[/bold cyan]")
            console.print(f"[dim]Course: {found_course['name']} ({found_course['acronym']})[/dim]")
            console.print(f"[dim]Filter: {', '.join(search_type)}[/dim]\n")

            if not exercises:
                console.print("[yellow]No exercises found matching the search criteria.[/yellow]\n")
                return

            console.print(f"[bold]Found {len(exercises)} exercise(s):[/bold]\n")

            for ex in exercises:
                # Get all core loops for this exercise
                knowledge_items = db.get_exercise_knowledge_items(ex["id"])

                # Display exercise header
                exercise_id = ex.get("exercise_number") or ex.get("source_pdf", "Unknown")
                console.print(f"[cyan]Exercise: {exercise_id}[/cyan]")

                # Display procedures
                if len(knowledge_items) > 1:
                    console.print(f"  [yellow]Procedures ({len(knowledge_items)}):[/yellow]")
                    for i, cl in enumerate(knowledge_items, 1):
                        step_info = f" (point {cl['step_number']})" if cl["step_number"] else ""
                        console.print(f"    {i}. {cl['name']}{step_info}")
                elif len(knowledge_items) == 1:
                    console.print(f"  Core Loop: {knowledge_items[0]['name']}")
                else:
                    console.print(f"  [dim]No core loops assigned[/dim]")

                # Display tags if present
                if ex.get("tags"):
                    tags = json.loads(ex["tags"]) if isinstance(ex["tags"], str) else ex["tags"]
                    console.print(f"  Tags: {', '.join(tags)}")

                # Display difficulty
                if ex.get("difficulty"):
                    console.print(f"  Difficulty: {ex['difficulty']}")

                # Display source
                if ex.get("source_pdf"):
                    page_info = f" (page {ex['page_number']})" if ex.get("page_number") else ""
                    console.print(f"  [dim]Source: {ex['source_pdf']}{page_info}[/dim]")

                console.print()

            if len(exercises) == limit:
                console.print(
                    f"[dim]Showing first {limit} results. Use --limit to see more.[/dim]\n"
                )

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}\n")
        import traceback

        traceback.print_exc()
        raise click.Abort()


@cli.command()
@click.option("--course", "-c", required=True, help="Course code (e.g., B006802 or ADE)")
@click.option(
    "--zip",
    "-z",
    "zip_file",
    required=True,
    type=click.Path(exists=True),
    help="Path to ZIP file containing exam PDFs",
)
@click.option(
    "--material-type",
    type=click.Choice(["exams", "notes"]),
    default="exams",
    help="Type of material: exams (problem sets) or notes (lecture slides)",
)
@click.option(
    "--smart-split",
    is_flag=True,
    default=False,
    help="Use LLM-based splitting for unstructured materials (lecture notes, embedded examples). Costs API tokens.",
)
@click.option(
    "--provider",
    type=click.Choice(["anthropic", "groq", "ollama", "openai", "deepseek"]),
    default=None,
    help="LLM provider for smart splitting (overrides profile routing)",
)
@click.option(
    "--profile",
    type=click.Choice(["free", "pro", "local"]),
    default=None,
    help="Provider profile for routing (free/pro/local). Uses EXAMINA_PROVIDER_PROFILE if not specified.",
)
def ingest(course, zip_file, material_type, smart_split, provider, profile):
    """Ingest course materials (exams, homework, problem sets, lecture notes) for a course."""

    console.print(f"\n[bold cyan]Ingesting exams for {course}...[/bold cyan]\n")

    try:
        # Find course by code or acronym
        with Database() as db:
            all_courses = db.get_all_courses()
            found_course = None
            for c in all_courses:
                if c["code"] == course or c["acronym"] == course:
                    found_course = c
                    break

            if not found_course:
                console.print(f"[red]Course '{course}' not found.[/red]\n")
                console.print("Use 'examina courses' to see available courses.\n")
                return

            course_code = found_course["code"]
            console.print(f"Course: {found_course['name']} ({found_course['acronym']})\n")

        # Initialize components
        file_mgr = FileManager()
        pdf_processor = PDFProcessor()

        # Determine provider to use (provider flag overrides profile routing)
        effective_provider = provider
        if provider is None and profile is not None:
            # Use routing
            from core.provider_router import ProviderRouter
            from core.task_types import TaskType

            try:
                router = ProviderRouter()
                effective_provider = router.route(TaskType.BULK_ANALYSIS, profile)
                console.print(
                    f"[dim]Using profile '{profile}' → provider: {effective_provider}[/dim]"
                )
            except Exception as e:
                console.print(f"[yellow]Warning: Routing failed: {e}[/yellow]")
                console.print(f"[dim]Falling back to default provider: {Config.LLM_PROVIDER}[/dim]")
                effective_provider = Config.LLM_PROVIDER
        elif provider is None:
            # No provider or profile specified, use default
            effective_provider = Config.LLM_PROVIDER

        # Choose splitter based on material type
        use_smart_splitter = False
        if material_type == "notes":
            # Lecture notes always require smart splitting for content detection
            from core.smart_splitter import SmartExerciseSplitter
            from models.llm_manager import LLMManager

            console.print(
                f"[cyan]📚 Processing lecture notes with smart content detection ({effective_provider})[/cyan]"
            )
            console.print(
                "[dim]   Detecting theory sections, worked examples, and practice exercises[/dim]\n"
            )

            llm = LLMManager(provider=effective_provider)
            exercise_splitter = SmartExerciseSplitter(
                llm_manager=llm,
                enable_smart_detection=True,
                notes_mode=True,  # Process ALL pages with LLM for content classification
            )
            use_smart_splitter = True
        elif material_type == "exams" and smart_split:
            # Exams with --smart-split flag use hybrid approach
            from core.smart_splitter import SmartExerciseSplitter
            from models.llm_manager import LLMManager

            console.print(f"[cyan]🤖 Smart splitting enabled with {effective_provider}[/cyan]")
            console.print("[dim]   LLM-based detection for unstructured materials[/dim]\n")

            llm = LLMManager(provider=effective_provider)
            exercise_splitter = SmartExerciseSplitter(llm_manager=llm, enable_smart_detection=True)
            use_smart_splitter = True
        else:
            # Default: pattern-based splitting for exams (fast, free)
            exercise_splitter = ExerciseSplitter()

        # Extract PDFs from ZIP
        console.print("📦 Extracting ZIP file...")
        try:
            pdf_files = file_mgr.extract_zip(zip_file, course_code)
            console.print(f"   ✓ Found {len(pdf_files)} PDF(s)\n")
        except Exception as e:
            console.print(f"[red]Error extracting ZIP: {e}[/red]\n")
            raise click.Abort()

        if not pdf_files:
            console.print("[yellow]No PDF files found in ZIP.[/yellow]\n")
            return

        # Process each PDF
        total_exercises = 0
        total_theory_sections = 0
        total_worked_examples = 0
        processed_pdfs = 0

        for pdf_path in pdf_files:
            console.print(f"📄 Processing {pdf_path.name}...")

            # Check if scanned PDF
            if pdf_processor.is_scanned_pdf(pdf_path):
                console.print(
                    "   [yellow]⚠️  Scanned PDF detected - OCR not yet implemented[/yellow]"
                )
                console.print("   [dim]Skipping...[/dim]\n")
                continue

            try:
                # Extract content
                pdf_content = pdf_processor.process_pdf(pdf_path)
                console.print(f"   ✓ Extracted {pdf_content.total_pages} pages")

                # Split into exercises and learning materials
                learning_materials = []
                if use_smart_splitter:
                    # SmartExerciseSplitter returns SplitResult
                    split_result = exercise_splitter.split_pdf_content(pdf_content, course_code)
                    exercises = split_result.exercises
                    learning_materials = split_result.learning_materials

                    # Show smart splitting stats
                    if split_result.llm_based_count > 0:
                        console.print(
                            f"   ✓ Found {len(exercises)} exercise(s) "
                            f"(pattern: {split_result.pattern_based_count}, "
                            f"LLM: {split_result.llm_based_count})"
                        )
                        console.print(
                            f"   [dim]LLM processed {split_result.llm_pages_processed}/{split_result.total_pages} pages "
                            f"(est. cost: ${split_result.total_cost_estimate:.4f})[/dim]"
                        )
                    else:
                        console.print(f"   ✓ Found {len(exercises)} exercise(s) (pattern-based)")

                    # Show learning materials stats if any found
                    if learning_materials:
                        console.print(
                            f"   ✓ Found {split_result.theory_count} theory section(s), "
                            f"{split_result.worked_example_count} worked example(s)"
                        )
                else:
                    # Regular ExerciseSplitter returns List[Exercise]
                    exercises = exercise_splitter.split_pdf_content(pdf_content, course_code)

                # Filter valid exercises
                valid_exercises = [
                    ex for ex in exercises if exercise_splitter.validate_exercise(ex)
                ]
                if not use_smart_splitter or (
                    use_smart_splitter and len(valid_exercises) != len(exercises)
                ):
                    console.print(f"   ✓ {len(valid_exercises)} valid exercise(s) after filtering")

                # Store exercises in database
                with Database() as db:
                    for exercise in valid_exercises:
                        # Clean text
                        cleaned_text = exercise_splitter.clean_exercise_text(exercise.text)

                        # Store images if present
                        image_paths = []
                        if exercise.has_images:
                            for i, img_data in enumerate(exercise.image_data):
                                img_path = file_mgr.store_image(
                                    img_data, course_code, exercise.id, i
                                )
                                image_paths.append(str(img_path))

                        # Prepare exercise data
                        exercise_data = {
                            "id": exercise.id,
                            "course_code": course_code,
                            "topic_id": None,  # Will be filled in Phase 3 (AI analysis)
                            "knowledge_item_id": None,  # Will be filled in Phase 3
                            "source_pdf": pdf_path.name,
                            "page_number": exercise.page_number,
                            "exercise_number": exercise.exercise_number,
                            "text": cleaned_text,
                            "has_images": exercise.has_images,
                            "image_paths": image_paths if image_paths else None,
                            "latex_content": exercise.latex_content,
                            "difficulty": None,  # Will be analyzed in Phase 3
                            "variations": None,
                            "solution": None,
                            "analyzed": False,
                            "analysis_metadata": None,
                        }

                        db.add_exercise(exercise_data)

                    # Store learning materials if any
                    for material in learning_materials:
                        # Store images if present
                        material_image_paths = []
                        if material.has_images:
                            for i, img_data in enumerate(material.image_data):
                                img_path = file_mgr.store_image(
                                    img_data, course_code, material.id, i
                                )
                                material_image_paths.append(str(img_path))

                        # Store learning material
                        db.store_learning_material(
                            material_id=material.id,
                            course_code=course_code,
                            material_type=material.material_type,
                            content=material.content,
                            title=material.title,
                            source_pdf=pdf_path.name,
                            page_number=material.page_number,
                            has_images=material.has_images,
                            image_paths=material_image_paths if material_image_paths else None,
                            latex_content=material.latex_content,
                        )

                    db.conn.commit()

                total_exercises += len(valid_exercises)
                if use_smart_splitter:
                    total_theory_sections += sum(
                        1 for m in learning_materials if m.material_type == "theory"
                    )
                    total_worked_examples += sum(
                        1 for m in learning_materials if m.material_type == "worked_example"
                    )
                processed_pdfs += 1
                console.print(f"   ✓ Stored in database\n")

            except Exception as e:
                console.print(f"   [red]Error: {e}[/red]\n")
                continue

        # Summary
        console.print("[bold green]✨ Ingestion complete![/bold green]\n")
        console.print(f"Processed: {processed_pdfs} PDF(s)")
        console.print(f"Ingested: {total_exercises} exercise(s)", end="")
        if total_theory_sections > 0 or total_worked_examples > 0:
            console.print(
                f", {total_theory_sections} theory section(s), {total_worked_examples} worked example(s)"
            )
        else:
            console.print()
        console.print(f"\nNext steps:")
        console.print(f"  • examina info --course {course} - View course status")
        console.print(f"  • Phase 3: AI analysis to discover topics and core loops\n")

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}\n")
        import traceback

        traceback.print_exc()
        raise click.Abort()


@cli.command()
@click.option("--course", "-c", required=True, help="Course code (e.g., B006802 or ADE)")
@click.option("--limit", "-l", type=int, help="Limit number of exercises to analyze (for testing)")
@click.option(
    "--provider",
    "-p",
    type=click.Choice(["ollama", "groq", "anthropic", "openai", "deepseek"]),
    default=None,
    help="LLM provider (overrides profile routing)",
)
@click.option(
    "--profile",
    type=click.Choice(["free", "pro", "local"]),
    default=None,
    help="Provider profile for routing (free/pro/local). Uses EXAMINA_PROVIDER_PROFILE if not specified.",
)
@click.option(
    "--lang",
    type=click.Choice(["en", "it"]),
    default="en",
    help="Output language for analysis (default: en)",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Force re-analysis of all exercises (ignore existing analysis)",
)
@click.option(
    "--parallel/--sequential",
    default=True,
    help="Use parallel batch processing for better performance (default: parallel)",
)
@click.option(
    "--batch-size",
    "-b",
    type=int,
    default=None,
    help=f"Batch size for parallel processing (default: {Config.BATCH_SIZE})",
)
@click.option(
    "--monolingual",
    is_flag=True,
    help="Enable strictly monolingual mode - all procedures will be in single language (prevents cross-language duplicates)",
)
@click.option(
    "--async-mode",
    "use_async",
    is_flag=True,
    default=False,
    help="Use async processing for better performance (requires async-compatible provider)",
)
def analyze(
    course, limit, provider, profile, lang, force, parallel, batch_size, monolingual, use_async
):
    """Analyze exercises with AI to discover topics and core loops."""
    if use_async:
        # Run async version
        asyncio.run(
            analyze_async(
                course, limit, provider, profile, lang, force, parallel, batch_size, monolingual
            )
        )
    else:
        # Run sync version
        analyze_sync(
            course, limit, provider, profile, lang, force, parallel, batch_size, monolingual
        )


async def analyze_async(
    course, limit, provider, profile, lang, force, parallel, batch_size, monolingual
):
    """Asynchronous analysis implementation."""
    console.print(f"\n[bold cyan]Analyzing exercises for {course}...[/bold cyan]\n")
    console.print(f"[dim]Using async mode for improved performance[/dim]\n")

    try:
        # Find course
        with Database() as db:
            all_courses = db.get_all_courses()
            found_course = None
            for c in all_courses:
                if c["code"] == course or c["acronym"] == course:
                    found_course = c
                    break

            if not found_course:
                console.print(f"[red]Course '{course}' not found.[/red]\n")
                return

            course_code = found_course["code"]
            console.print(f"Course: {found_course['name']} ({found_course['acronym']})\n")

            # Check for exercises and analyze resume capability
            all_exercises = db.get_exercises_by_course(course_code)
            if not all_exercises:
                console.print("[yellow]No exercises found. Run 'examina ingest' first.[/yellow]\n")
                return

            analyzed_exercises = db.get_exercises_by_course(course_code, analyzed_only=True)
            unanalyzed_exercises = db.get_exercises_by_course(course_code, unanalyzed_only=True)

            # Display progress summary
            total_count = len(all_exercises)
            analyzed_count = len(analyzed_exercises)
            remaining_count = len(unanalyzed_exercises)

            console.print(f"Found {total_count} exercise fragments")
            console.print(f"  [green]Already analyzed: {analyzed_count}[/green]")
            console.print(f"  [yellow]Remaining: {remaining_count}[/yellow]\n")

            # Determine which exercises to analyze
            if force:
                console.print("[yellow]--force flag: Re-analyzing all exercises[/yellow]\n")
                exercises = all_exercises
                # Reset analyzed flag for all exercises
                db.conn.execute(
                    "UPDATE exercises SET analyzed = 0 WHERE course_code = ?", (course_code,)
                )
                db.conn.commit()
            elif remaining_count == 0:
                console.print(
                    "[green]All exercises already analyzed! Use --force to re-analyze.[/green]\n"
                )
                return
            else:
                if analyzed_count > 0:
                    console.print(
                        f"[cyan]Resuming analysis from checkpoint ({remaining_count} exercises remaining)...[/cyan]\n"
                    )
                exercises = all_exercises  # Need all for proper merging context

        # Determine provider to use (provider flag overrides profile routing)
        effective_provider = provider
        if provider is None and profile is not None:
            # Use routing
            from core.provider_router import ProviderRouter
            from core.task_types import TaskType

            try:
                router = ProviderRouter()
                effective_provider = router.route(TaskType.BULK_ANALYSIS, profile)
                console.print(
                    f"[dim]Using profile '{profile}' → provider: {effective_provider}[/dim]\n"
                )
            except Exception as e:
                console.print(f"[yellow]Warning: Routing failed: {e}[/yellow]")
                console.print(
                    f"[dim]Falling back to default provider: {Config.LLM_PROVIDER}[/dim]\n"
                )
                effective_provider = Config.LLM_PROVIDER
        elif provider is None:
            # No provider or profile specified, use default
            effective_provider = Config.LLM_PROVIDER

        # Initialize components with async context manager
        mode_str = f"language: {lang}, monolingual: {'ON' if monolingual else 'OFF'}"
        console.print(
            f"🤖 Initializing AI components (provider: {effective_provider}, {mode_str})..."
        )

        async with LLMManager(provider=effective_provider) as llm:
            # Initialize procedure cache for faster analysis (Option 3 - Performance)
            procedure_cache = None
            if Config.PROCEDURE_CACHE_ENABLED:
                from core.procedure_cache import ProcedureCache

                try:
                    # SemanticMatcher removed - using LLM-based detect_synonyms() instead
                    procedure_cache = ProcedureCache(db, semantic_matcher=None, user_id=None)
                    if Config.PROCEDURE_CACHE_PRELOAD:
                        procedure_cache.load_cache(course_code)
                    console.print(
                        f"   ✓ Procedure cache enabled ({len(procedure_cache._entries)} patterns)\n"
                    )
                except Exception as e:
                    console.print(f"   ⚠ Procedure cache unavailable: {e}\n")
                    procedure_cache = None

            analyzer = ExerciseAnalyzer(
                llm, language=lang, monolingual=monolingual, procedure_cache=procedure_cache
            )

            # Translation detector removed - names always extracted in English
            translation_detector = None

            # For embeddings, we still need Ollama (Groq/Anthropic don't provide embeddings)
            embed_llm = (
                LLMManager(provider="ollama")
                if effective_provider in ["groq", "anthropic"]
                else llm
            )
            vector_store = VectorStore(llm_manager=embed_llm)

            # Check if provider is ready
            if effective_provider == "ollama":
                console.print(f"   Checking {llm.primary_model}...")
                if not llm.check_model_available(llm.primary_model):
                    console.print(f"[red]Model {llm.primary_model} not found![/red]")
                    console.print(f"[yellow]Run: ollama pull {llm.primary_model}[/yellow]\n")
                    return
                console.print(f"   ✓ {llm.primary_model} ready\n")
            elif effective_provider == "groq":
                console.print(f"   Using Groq API with {llm.primary_model}")
                if not Config.GROQ_API_KEY:
                    console.print(f"[red]GROQ_API_KEY not set![/red]")
                    console.print(
                        f"[yellow]Get your free API key at: https://console.groq.com[/yellow]"
                    )
                    console.print(
                        f"[yellow]Then set it: export GROQ_API_KEY=your_key_here[/yellow]\n"
                    )
                    return
                console.print(f"   ✓ API key found\n")
            elif provider == "anthropic":
                console.print(f"   Using Anthropic API with {llm.primary_model}")
                if not Config.ANTHROPIC_API_KEY:
                    console.print(f"[red]ANTHROPIC_API_KEY not set![/red]")
                    console.print(
                        f"[yellow]Get your API key at: https://console.anthropic.com[/yellow]"
                    )
                    console.print(
                        f"[yellow]Then set it: export ANTHROPIC_API_KEY=your_key_here[/yellow]\n"
                    )
                    return
                console.print(f"   ✓ API key found\n")

            # Limit for testing
            if limit:
                exercises = exercises[:limit]
                console.print(f"[dim]Limited to {limit} exercises for testing[/dim]\n")

            # Analyze and merge using async method
            processing_mode = "async"
            console.print(
                f"🔍 Analyzing and merging exercise fragments ({processing_mode} mode)..."
            )
            batch_size_msg = batch_size if batch_size else Config.BATCH_SIZE
            console.print(f"[dim]Using batch size: {batch_size_msg}[/dim]")
            console.print("[dim]This may take a while...[/dim]\n")

            # Determine if we should skip analyzed exercises
            skip_analyzed = not force and analyzed_count > 0

            # Use async discovery method
            discovery_result = await analyzer.discover_topics_and_knowledge_items_async(
                course_code, batch_size=batch_size or Config.BATCH_SIZE, skip_analyzed=skip_analyzed
            )

            # Show progress summary
            if skip_analyzed:
                newly_analyzed = discovery_result["merged_count"]
                console.print(
                    f"✓ Analyzed {newly_analyzed} new exercises (skipped {analyzed_count} already analyzed)"
                )
                console.print(
                    f"  Total progress: {analyzed_count + newly_analyzed}/{total_count} exercises\n"
                )
            else:
                console.print(
                    f"✓ Merged {discovery_result['original_count']} fragments → {discovery_result['merged_count']} exercises\n"
                )

            # Display results
            topics = discovery_result["topics"]
            knowledge_items = discovery_result["knowledge_items"]

            if topics:
                console.print("[bold]📚 Discovered Topics:[/bold]")
                for topic_name, topic_data in topics.items():
                    console.print(
                        f"  • {topic_name} ({topic_data['exercise_count']} exercises, {len(topic_data['knowledge_items'])} core loops)"
                    )

            if knowledge_items:
                console.print(f"\n[bold]🔄 Discovered Core Loops:[/bold]")
                for loop_id, loop_data in knowledge_items.items():
                    console.print(
                        f"  • {loop_data['name']} ({loop_data['exercise_count']} exercises)"
                    )
                    if loop_data["procedure"]:
                        console.print(f"    [dim]Steps: {len(loop_data['procedure'])}[/dim]")

            # Store in database
            console.print(f"\n💾 Storing analysis results...")
            with Database() as db:
                # Get topic name mapping from analyzer (for deduplication)
                topic_name_mapping = getattr(analyzer, "topic_name_mapping", {})

                # Store topics (with language detection)
                for topic_name in topics.keys():
                    # Detect language if translation detector is available
                    topic_language = None
                    if monolingual and analyzer.primary_language:
                        # In monolingual mode, use the detected primary language
                        topic_language = analyzer.primary_language
                    elif translation_detector:
                        topic_language = translation_detector.detect_language(topic_name)
                        if topic_language == "unknown":
                            topic_language = None  # Store NULL instead of "unknown"

                    topic_id = db.add_topic(course_code, topic_name, language=topic_language)

                # Store core loops (with language detection)
                for loop_id, loop_data in knowledge_items.items():
                    # Get topic_id
                    topic_name = loop_data.get("topic")
                    if topic_name:
                        # Map topic name to canonical name (if deduplicated)
                        canonical_topic_name = topic_name
                        if canonical_topic_name in topic_name_mapping:
                            canonical_topic_name = topic_name_mapping[canonical_topic_name]

                        topic_rows = db.get_topics_by_course(course_code)
                        topic_id = next(
                            (t["id"] for t in topic_rows if t["name"] == canonical_topic_name), None
                        )

                        if topic_id:
                            # Detect language if translation detector is available
                            loop_language = None
                            if monolingual and analyzer.primary_language:
                                # In monolingual mode, use the detected primary language
                                loop_language = analyzer.primary_language
                            elif translation_detector:
                                loop_language = translation_detector.detect_language(
                                    loop_data["name"]
                                )
                                if loop_language == "unknown":
                                    loop_language = None  # Store NULL instead of "unknown"

                            db.add_knowledge_item(
                                loop_id=loop_id,
                                topic_id=topic_id,
                                name=loop_data["name"],
                                procedure=loop_data["procedure"],
                                description=None,
                                language=loop_language,
                            )

                # Get core loop ID mapping from analyzer (for deduplication)
                knowledge_item_id_mapping = getattr(analyzer, "knowledge_item_id_mapping", {})

                # Update exercises with analysis
                for merged_ex in discovery_result["merged_exercises"]:
                    # Update first exercise in merged group
                    first_id = merged_ex["merged_from"][0]

                    # Check if this exercise was skipped due to low confidence
                    if merged_ex.get("low_confidence_skipped"):
                        db.conn.execute(
                            """
                            UPDATE exercises
                            SET analyzed = 1, low_confidence_skipped = 1
                            WHERE id = ?
                        """,
                            (first_id,),
                        )
                        continue

                    analysis = merged_ex.get("analysis")
                    if analysis and analysis.topic:
                        # Map topic name to canonical name (if deduplicated)
                        canonical_topic_name = analysis.topic
                        if canonical_topic_name in topic_name_mapping:
                            canonical_topic_name = topic_name_mapping[canonical_topic_name]

                        # Get topic_id
                        topic_rows = db.get_topics_by_course(course_code)
                        topic_id = next(
                            (t["id"] for t in topic_rows if t["name"] == canonical_topic_name), None
                        )

                        # Get primary knowledge_item_id (first procedure) for backward compatibility
                        primary_knowledge_item_id = analysis.knowledge_item_id
                        if (
                            primary_knowledge_item_id
                            and primary_knowledge_item_id in knowledge_item_id_mapping
                        ):
                            primary_knowledge_item_id = knowledge_item_id_mapping[
                                primary_knowledge_item_id
                            ]

                        # Only update if primary_knowledge_item_id exists in deduplicated knowledge_items OR database
                        if (
                            primary_knowledge_item_id
                            and primary_knowledge_item_id not in knowledge_items
                        ):
                            # Check if it exists in database (may have been deduplicated to existing DB entry)
                            if not db.get_knowledge_item(primary_knowledge_item_id):
                                print(
                                    f"[DEBUG] Skipping exercise {first_id[:20]}... - knowledge_item_id '{primary_knowledge_item_id}' not found in deduplicated knowledge_items or database"
                                )
                                primary_knowledge_item_id = None

                        # Collect tags for flexible search
                        tags = []

                        # Process ALL procedures - link to junction table
                        if analysis.procedures:
                            for procedure_info in analysis.procedures:
                                proc_knowledge_item_id = (
                                    AnalysisResult._normalize_knowledge_item_id(procedure_info.name)
                                )

                                # Map to canonical ID if deduplicated
                                if (
                                    proc_knowledge_item_id
                                    and proc_knowledge_item_id in knowledge_item_id_mapping
                                ):
                                    proc_knowledge_item_id = knowledge_item_id_mapping[
                                        proc_knowledge_item_id
                                    ]

                                # Link exercise to core loop via junction table (check both new loops and DB)
                                if proc_knowledge_item_id and (
                                    proc_knowledge_item_id in knowledge_items
                                    or db.get_knowledge_item(proc_knowledge_item_id)
                                ):
                                    db.link_exercise_to_knowledge_item(
                                        exercise_id=first_id,
                                        knowledge_item_id=proc_knowledge_item_id,
                                        step_number=procedure_info.point_number,
                                    )

                                    # Collect tags
                                    tags.append(procedure_info.type)
                                    if procedure_info.transformation:
                                        src = (
                                            procedure_info.transformation.get("source_format", "")
                                            .lower()
                                            .replace(" ", "_")
                                        )
                                        tgt = (
                                            procedure_info.transformation.get("target_format", "")
                                            .lower()
                                            .replace(" ", "_")
                                        )
                                        tags.append(f"transform_{src}_to_{tgt}")

                        # Update exercise with primary core loop and metadata
                        db.update_exercise_analysis(
                            exercise_id=first_id,
                            topic_id=topic_id,
                            knowledge_item_id=primary_knowledge_item_id,
                            difficulty=analysis.difficulty,
                            variations=analysis.variations,
                            analyzed=True,
                        )

                        # Update tags
                        if tags:
                            db.update_exercise_tags(first_id, list(set(tags)))

                        # Phase 9.2: Update theory metadata if present
                        if analysis.exercise_type in ["theory", "proof", "hybrid"]:
                            db.update_exercise_theory_metadata(
                                exercise_id=first_id,
                                exercise_type=analysis.exercise_type,
                                theory_category=analysis.theory_category,
                                theorem_name=analysis.theorem_name,
                                concept_id=analysis.concept_id,
                                prerequisite_concepts=analysis.prerequisite_concepts,
                                theory_metadata=analysis.theory_metadata,
                            )

                db.conn.commit()
                console.print("   ✓ Stored in database\n")

            # Build vector store
            console.print("🧠 Building vector embeddings for RAG...")
            vector_store.add_exercises_batch(course_code, discovery_result["merged_exercises"])

            # Add core loops to vector store
            for loop_id, loop_data in knowledge_items.items():
                vector_store.add_knowledge_item(
                    course_code=course_code,
                    knowledge_item_id=loop_id,
                    name=loop_data["name"],
                    description=loop_data.get("description", ""),
                    procedure=loop_data["procedure"],
                    example_exercises=loop_data["exercises"],
                )

            stats = vector_store.get_collection_stats(course_code)
            console.print(f"   ✓ {stats.get('exercises_count', 0)} exercise embeddings")
            console.print(f"   ✓ {stats.get('procedures_count', 0)} procedure embeddings\n")

            # Show cache statistics
            cache_stats = llm.get_cache_stats()
            if cache_stats["total_requests"] > 0:
                console.print("📊 LLM Response Cache:")
                console.print(f"   Cache hits: {cache_stats['cache_hits']}")
                console.print(f"   Cache misses: {cache_stats['cache_misses']}")
                console.print(f"   Hit rate: {cache_stats['hit_rate_percent']}%")
                if cache_stats["cache_hits"] > 0:
                    console.print(
                        f"   [green]💰 Saved ~{cache_stats['cache_hits']} API calls![/green]\n"
                    )
                else:
                    console.print(f"   [dim]Run analyze again to see cache benefits[/dim]\n")

            # Show procedure cache statistics (Option 3)
            if analyzer.cache_stats and (
                analyzer.cache_stats["hits"] > 0 or analyzer.cache_stats["misses"] > 0
            ):
                total = analyzer.cache_stats["hits"] + analyzer.cache_stats["misses"]
                hit_rate = (analyzer.cache_stats["hits"] / total * 100) if total > 0 else 0
                console.print("📊 Procedure Pattern Cache:")
                console.print(f"   Hits: {analyzer.cache_stats['hits']}")
                console.print(f"   Misses: {analyzer.cache_stats['misses']}")
                console.print(f"   Hit rate: {hit_rate:.1f}%")
                if analyzer.cache_stats["hits"] > 0:
                    console.print(
                        f"   [green]💰 Skipped {analyzer.cache_stats['hits']} LLM analyses via pattern matching![/green]\n"
                    )

            # Summary
            console.print("[bold green]✨ Analysis complete![/bold green]\n")
            console.print(f"Topics: {len(topics)}")
            console.print(f"Core loops: {len(knowledge_items)}")
            console.print(f"Exercises: {discovery_result['merged_count']}\n")
            console.print(f"Next steps:")
            console.print(f"  • examina info --course {course} - View updated course info")
            console.print(f"  • examina learn --course {course} - Start learning (Phase 4)\n")

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}\n")
        import traceback

        traceback.print_exc()
        raise click.Abort()


def analyze_sync(course, limit, provider, profile, lang, force, parallel, batch_size, monolingual):
    """Synchronous analysis implementation."""
    console.print(f"\n[bold cyan]Analyzing exercises for {course}...[/bold cyan]\n")

    try:
        # Find course
        with Database() as db:
            all_courses = db.get_all_courses()
            found_course = None
            for c in all_courses:
                if c["code"] == course or c["acronym"] == course:
                    found_course = c
                    break

            if not found_course:
                console.print(f"[red]Course '{course}' not found.[/red]\n")
                return

            course_code = found_course["code"]
            console.print(f"Course: {found_course['name']} ({found_course['acronym']})\n")

            # Check for exercises and analyze resume capability
            all_exercises = db.get_exercises_by_course(course_code)
            if not all_exercises:
                console.print("[yellow]No exercises found. Run 'examina ingest' first.[/yellow]\n")
                return

            analyzed_exercises = db.get_exercises_by_course(course_code, analyzed_only=True)
            unanalyzed_exercises = db.get_exercises_by_course(course_code, unanalyzed_only=True)

            # Display progress summary
            total_count = len(all_exercises)
            analyzed_count = len(analyzed_exercises)
            remaining_count = len(unanalyzed_exercises)

            console.print(f"Found {total_count} exercise fragments")
            console.print(f"  [green]Already analyzed: {analyzed_count}[/green]")
            console.print(f"  [yellow]Remaining: {remaining_count}[/yellow]\n")

            # Determine which exercises to analyze
            if force:
                console.print("[yellow]--force flag: Re-analyzing all exercises[/yellow]\n")
                exercises = all_exercises
                # Reset analyzed flag for all exercises
                db.conn.execute(
                    "UPDATE exercises SET analyzed = 0 WHERE course_code = ?", (course_code,)
                )
                db.conn.commit()
            elif remaining_count == 0:
                console.print(
                    "[green]All exercises already analyzed! Use --force to re-analyze.[/green]\n"
                )
                return
            else:
                if analyzed_count > 0:
                    console.print(
                        f"[cyan]Resuming analysis from checkpoint ({remaining_count} exercises remaining)...[/cyan]\n"
                    )
                exercises = all_exercises  # Need all for proper merging context

        # Determine provider to use (provider flag overrides profile routing)
        effective_provider = provider
        if provider is None and profile is not None:
            # Use routing
            from core.provider_router import ProviderRouter
            from core.task_types import TaskType

            try:
                router = ProviderRouter()
                effective_provider = router.route(TaskType.BULK_ANALYSIS, profile)
                console.print(
                    f"[dim]Using profile '{profile}' → provider: {effective_provider}[/dim]\n"
                )
            except Exception as e:
                console.print(f"[yellow]Warning: Routing failed: {e}[/yellow]")
                console.print(
                    f"[dim]Falling back to default provider: {Config.LLM_PROVIDER}[/dim]\n"
                )
                effective_provider = Config.LLM_PROVIDER
        elif provider is None:
            # No provider or profile specified, use default
            effective_provider = Config.LLM_PROVIDER

        # Initialize components
        mode_str = f"language: {lang}, monolingual: {'ON' if monolingual else 'OFF'}"
        console.print(
            f"🤖 Initializing AI components (provider: {effective_provider}, {mode_str})..."
        )
        llm = LLMManager(provider=effective_provider)

        # Initialize procedure cache for faster analysis (Option 3 - Performance)
        procedure_cache = None
        if Config.PROCEDURE_CACHE_ENABLED:
            from core.procedure_cache import ProcedureCache

            try:
                with Database() as cache_db:
                    # SemanticMatcher removed - using LLM-based detect_synonyms() instead
                    procedure_cache = ProcedureCache(cache_db, semantic_matcher=None, user_id=None)
                    if Config.PROCEDURE_CACHE_PRELOAD:
                        procedure_cache.load_cache(course_code)
                    console.print(
                        f"   ✓ Procedure cache enabled ({len(procedure_cache._entries)} patterns)\n"
                    )
            except Exception as e:
                console.print(f"   ⚠ Procedure cache unavailable: {e}\n")
                procedure_cache = None

        analyzer = ExerciseAnalyzer(
            llm, language=lang, monolingual=monolingual, procedure_cache=procedure_cache
        )

        # Translation detector removed - names always extracted in English
        translation_detector = None

        # For embeddings, we still need Ollama (Groq/Anthropic don't provide embeddings)
        embed_llm = (
            LLMManager(provider="ollama") if effective_provider in ["groq", "anthropic"] else llm
        )
        vector_store = VectorStore(llm_manager=embed_llm)

        # Check if provider is ready
        if effective_provider == "ollama":
            console.print(f"   Checking {llm.primary_model}...")
            if not llm.check_model_available(llm.primary_model):
                console.print(f"[red]Model {llm.primary_model} not found![/red]")
                console.print(f"[yellow]Run: ollama pull {llm.primary_model}[/yellow]\n")
                return
            console.print(f"   ✓ {llm.primary_model} ready\n")
        elif effective_provider == "groq":
            console.print(f"   Using Groq API with {llm.primary_model}")
            if not Config.GROQ_API_KEY:
                console.print(f"[red]GROQ_API_KEY not set![/red]")
                console.print(
                    f"[yellow]Get your free API key at: https://console.groq.com[/yellow]"
                )
                console.print(f"[yellow]Then set it: export GROQ_API_KEY=your_key_here[/yellow]\n")
                return
            console.print(f"   ✓ API key found\n")
        elif provider == "anthropic":
            console.print(f"   Using Anthropic API with {llm.primary_model}")
            if not Config.ANTHROPIC_API_KEY:
                console.print(f"[red]ANTHROPIC_API_KEY not set![/red]")
                console.print(
                    f"[yellow]Get your API key at: https://console.anthropic.com[/yellow]"
                )
                console.print(
                    f"[yellow]Then set it: export ANTHROPIC_API_KEY=your_key_here[/yellow]\n"
                )
                return
            console.print(f"   ✓ API key found\n")

        # Limit for testing
        if limit:
            exercises = exercises[:limit]
            console.print(f"[dim]Limited to {limit} exercises for testing[/dim]\n")

        # Analyze and merge
        processing_mode = "parallel" if parallel else "sequential"
        console.print(f"🔍 Analyzing and merging exercise fragments ({processing_mode} mode)...")
        if parallel:
            batch_size_msg = batch_size if batch_size else Config.BATCH_SIZE
            console.print(f"[dim]Using batch size: {batch_size_msg}[/dim]")
        console.print("[dim]This may take a while...[/dim]\n")

        # Determine if we should skip analyzed exercises
        skip_analyzed = not force and analyzed_count > 0

        discovery_result = analyzer.discover_topics_and_knowledge_items(
            course_code,
            batch_size=batch_size or Config.BATCH_SIZE,
            skip_analyzed=skip_analyzed,
            use_parallel=parallel,
        )

        # Show progress summary
        if skip_analyzed:
            newly_analyzed = discovery_result["merged_count"]
            console.print(
                f"✓ Analyzed {newly_analyzed} new exercises (skipped {analyzed_count} already analyzed)"
            )
            console.print(
                f"  Total progress: {analyzed_count + newly_analyzed}/{total_count} exercises\n"
            )
        else:
            console.print(
                f"✓ Merged {discovery_result['original_count']} fragments → {discovery_result['merged_count']} exercises\n"
            )

        # Display results
        topics = discovery_result["topics"]
        knowledge_items = discovery_result["knowledge_items"]

        if topics:
            console.print("[bold]📚 Discovered Topics:[/bold]")
            for topic_name, topic_data in topics.items():
                console.print(
                    f"  • {topic_name} ({topic_data['exercise_count']} exercises, {len(topic_data['knowledge_items'])} core loops)"
                )

        if knowledge_items:
            console.print(f"\n[bold]🔄 Discovered Core Loops:[/bold]")
            for loop_id, loop_data in knowledge_items.items():
                console.print(f"  • {loop_data['name']} ({loop_data['exercise_count']} exercises)")
                if loop_data["procedure"]:
                    console.print(f"    [dim]Steps: {len(loop_data['procedure'])}[/dim]")

        # Store in database
        console.print(f"\n💾 Storing analysis results...")
        with Database() as db:
            # Get topic name mapping from analyzer (for deduplication)
            topic_name_mapping = getattr(analyzer, "topic_name_mapping", {})

            # Store topics (with language detection)
            for topic_name in topics.keys():
                # Detect language if translation detector is available
                topic_language = None
                if monolingual and analyzer.primary_language:
                    # In monolingual mode, use the detected primary language
                    topic_language = analyzer.primary_language
                elif translation_detector:
                    topic_language = translation_detector.detect_language(topic_name)
                    if topic_language == "unknown":
                        topic_language = None  # Store NULL instead of "unknown"

                topic_id = db.add_topic(course_code, topic_name, language=topic_language)

            # Store core loops (with language detection)
            for loop_id, loop_data in knowledge_items.items():
                # Get topic_id
                topic_name = loop_data.get("topic")
                if topic_name:
                    # Map topic name to canonical name (if deduplicated)
                    canonical_topic_name = topic_name
                    if canonical_topic_name in topic_name_mapping:
                        canonical_topic_name = topic_name_mapping[canonical_topic_name]

                    topic_rows = db.get_topics_by_course(course_code)
                    topic_id = next(
                        (t["id"] for t in topic_rows if t["name"] == canonical_topic_name), None
                    )

                    if topic_id:
                        # Detect language if translation detector is available
                        loop_language = None
                        if monolingual and analyzer.primary_language:
                            # In monolingual mode, use the detected primary language
                            loop_language = analyzer.primary_language
                        elif translation_detector:
                            loop_language = translation_detector.detect_language(loop_data["name"])
                            if loop_language == "unknown":
                                loop_language = None  # Store NULL instead of "unknown"

                        db.add_knowledge_item(
                            loop_id=loop_id,
                            topic_id=topic_id,
                            name=loop_data["name"],
                            procedure=loop_data["procedure"],
                            description=None,
                            language=loop_language,
                        )

            # Get core loop ID mapping from analyzer (for deduplication)
            knowledge_item_id_mapping = getattr(analyzer, "knowledge_item_id_mapping", {})

            # Update exercises with analysis
            for merged_ex in discovery_result["merged_exercises"]:
                # Update first exercise in merged group
                first_id = merged_ex["merged_from"][0]

                # Check if this exercise was skipped due to low confidence
                if merged_ex.get("low_confidence_skipped"):
                    db.conn.execute(
                        """
                        UPDATE exercises
                        SET analyzed = 1, low_confidence_skipped = 1
                        WHERE id = ?
                    """,
                        (first_id,),
                    )
                    continue

                analysis = merged_ex.get("analysis")
                if analysis and analysis.topic:
                    # Map topic name to canonical name (if deduplicated)
                    canonical_topic_name = analysis.topic
                    if canonical_topic_name in topic_name_mapping:
                        canonical_topic_name = topic_name_mapping[canonical_topic_name]

                    # Get topic_id
                    topic_rows = db.get_topics_by_course(course_code)
                    topic_id = next(
                        (t["id"] for t in topic_rows if t["name"] == canonical_topic_name), None
                    )

                    # Get primary knowledge_item_id (first procedure) for backward compatibility
                    primary_knowledge_item_id = analysis.knowledge_item_id
                    if (
                        primary_knowledge_item_id
                        and primary_knowledge_item_id in knowledge_item_id_mapping
                    ):
                        primary_knowledge_item_id = knowledge_item_id_mapping[
                            primary_knowledge_item_id
                        ]

                    # Only update if primary_knowledge_item_id exists in deduplicated knowledge_items OR database
                    if (
                        primary_knowledge_item_id
                        and primary_knowledge_item_id not in knowledge_items
                    ):
                        # Check if it exists in database (may have been deduplicated to existing DB entry)
                        if not db.get_knowledge_item(primary_knowledge_item_id):
                            print(
                                f"[DEBUG] Skipping exercise {first_id[:20]}... - knowledge_item_id '{primary_knowledge_item_id}' not found in deduplicated knowledge_items or database"
                            )
                            primary_knowledge_item_id = None

                    # Collect tags for flexible search
                    tags = []

                    # Process ALL procedures - link to junction table
                    if analysis.procedures:
                        for procedure_info in analysis.procedures:
                            proc_knowledge_item_id = AnalysisResult._normalize_knowledge_item_id(
                                procedure_info.name
                            )

                            # Map to canonical ID if deduplicated
                            if (
                                proc_knowledge_item_id
                                and proc_knowledge_item_id in knowledge_item_id_mapping
                            ):
                                proc_knowledge_item_id = knowledge_item_id_mapping[
                                    proc_knowledge_item_id
                                ]

                            # Link exercise to core loop via junction table (check both new loops and DB)
                            if proc_knowledge_item_id and (
                                proc_knowledge_item_id in knowledge_items
                                or db.get_knowledge_item(proc_knowledge_item_id)
                            ):
                                db.link_exercise_to_knowledge_item(
                                    exercise_id=first_id,
                                    knowledge_item_id=proc_knowledge_item_id,
                                    step_number=procedure_info.point_number,
                                )

                                # Collect tags
                                tags.append(procedure_info.type)
                                if procedure_info.transformation:
                                    src = (
                                        procedure_info.transformation.get("source_format", "")
                                        .lower()
                                        .replace(" ", "_")
                                    )
                                    tgt = (
                                        procedure_info.transformation.get("target_format", "")
                                        .lower()
                                        .replace(" ", "_")
                                    )
                                    tags.append(f"transform_{src}_to_{tgt}")

                    # Update exercise with primary core loop and metadata
                    db.update_exercise_analysis(
                        exercise_id=first_id,
                        topic_id=topic_id,
                        knowledge_item_id=primary_knowledge_item_id,
                        difficulty=analysis.difficulty,
                        variations=analysis.variations,
                        analyzed=True,
                    )

                    # Update tags
                    if tags:
                        db.update_exercise_tags(first_id, list(set(tags)))

                    # Phase 9.2: Update theory metadata if present
                    if analysis.exercise_type in ["theory", "proof", "hybrid"]:
                        db.update_exercise_theory_metadata(
                            exercise_id=first_id,
                            exercise_type=analysis.exercise_type,
                            theory_category=analysis.theory_category,
                            theorem_name=analysis.theorem_name,
                            concept_id=analysis.concept_id,
                            prerequisite_concepts=analysis.prerequisite_concepts,
                            theory_metadata=analysis.theory_metadata,
                        )

            db.conn.commit()
            console.print("   ✓ Stored in database\n")

        # Build vector store
        console.print("🧠 Building vector embeddings for RAG...")
        vector_store.add_exercises_batch(course_code, discovery_result["merged_exercises"])

        # Add core loops to vector store
        for loop_id, loop_data in knowledge_items.items():
            vector_store.add_knowledge_item(
                course_code=course_code,
                knowledge_item_id=loop_id,
                name=loop_data["name"],
                description=loop_data.get("description", ""),
                procedure=loop_data["procedure"],
                example_exercises=loop_data["exercises"],
            )

        stats = vector_store.get_collection_stats(course_code)
        console.print(f"   ✓ {stats.get('exercises_count', 0)} exercise embeddings")
        console.print(f"   ✓ {stats.get('procedures_count', 0)} procedure embeddings\n")

        # Show cache statistics
        cache_stats = llm.get_cache_stats()
        if cache_stats["total_requests"] > 0:
            console.print("📊 LLM Response Cache:")
            console.print(f"   Cache hits: {cache_stats['cache_hits']}")
            console.print(f"   Cache misses: {cache_stats['cache_misses']}")
            console.print(f"   Hit rate: {cache_stats['hit_rate_percent']}%")
            if cache_stats["cache_hits"] > 0:
                console.print(
                    f"   [green]💰 Saved ~{cache_stats['cache_hits']} API calls![/green]\n"
                )
            else:
                console.print(f"   [dim]Run analyze again to see cache benefits[/dim]\n")

        # Show procedure cache statistics (Option 3)
        if analyzer.cache_stats and (
            analyzer.cache_stats["hits"] > 0 or analyzer.cache_stats["misses"] > 0
        ):
            total = analyzer.cache_stats["hits"] + analyzer.cache_stats["misses"]
            hit_rate = (analyzer.cache_stats["hits"] / total * 100) if total > 0 else 0
            console.print("📊 Procedure Pattern Cache:")
            console.print(f"   Hits: {analyzer.cache_stats['hits']}")
            console.print(f"   Misses: {analyzer.cache_stats['misses']}")
            console.print(f"   Hit rate: {hit_rate:.1f}%")
            if analyzer.cache_stats["hits"] > 0:
                console.print(
                    f"   [green]💰 Skipped {analyzer.cache_stats['hits']} LLM analyses via pattern matching![/green]\n"
                )

        # Summary
        console.print("[bold green]✨ Analysis complete![/bold green]\n")
        console.print(f"Topics: {len(topics)}")
        console.print(f"Core loops: {len(knowledge_items)}")
        console.print(f"Exercises: {discovery_result['merged_count']}\n")
        console.print(f"Next steps:")
        console.print(f"  • examina info --course {course} - View updated course info")
        console.print(f"  • examina learn --course {course} - Start learning (Phase 4)\n")

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}\n")
        import traceback

        traceback.print_exc()
        raise click.Abort()


@cli.command(name="link-materials")
@click.option("--course", "-c", required=True, help="Course code (e.g., B006802 or ADE)")
@click.option(
    "--provider",
    "-p",
    type=click.Choice(["ollama", "groq", "anthropic", "openai", "deepseek"]),
    default="anthropic",
    help="LLM provider (default: anthropic)",
)
@click.option(
    "--lang",
    type=click.Choice(["en", "it"]),
    default="en",
    help="Output language for analysis (default: en)",
)
@click.option(
    "--link-exercises", is_flag=True, help="Also link worked examples to similar exercises"
)
def link_materials(course, provider, lang, link_exercises):
    """Link learning materials to topics and optionally to exercises.

    This command analyzes learning materials (theory, worked examples) and:
    1. Detects topics from material content
    2. Links materials to existing course topics
    3. (Optional) Links worked examples to similar practice exercises
    """
    console.print(f"\n[bold cyan]Linking learning materials for {course}...[/bold cyan]\n")

    try:
        # Find course
        with Database() as db:
            all_courses = db.get_all_courses()
            found_course = None
            for c in all_courses:
                if c["code"] == course or c["acronym"] == course:
                    found_course = c
                    break

            if not found_course:
                console.print(f"[red]Course '{course}' not found.[/red]\n")
                return

            course_code = found_course["code"]
            console.print(f"Course: {found_course['name']} ({found_course['acronym']})\n")

            # Check for materials
            materials = db.get_learning_materials_by_course(course_code)
            if not materials:
                console.print(
                    "[yellow]No learning materials found. Run 'examina ingest' first.[/yellow]\n"
                )
                return

            material_types = {}
            for mat in materials:
                mat_type = mat["material_type"]
                material_types[mat_type] = material_types.get(mat_type, 0) + 1

            console.print(f"Found {len(materials)} learning materials:")
            for mat_type, count in material_types.items():
                console.print(f"  • {mat_type}: {count}")
            console.print()

            # Check for topics
            topics = db.get_topics_by_course(course_code)
            if not topics:
                console.print("[yellow]No topics found. Run 'examina analyze' first.[/yellow]\n")
                return

            console.print(f"Found {len(topics)} topics in course\n")

        # Initialize components
        console.print(f"🤖 Initializing AI components (provider: {provider}, language: {lang})...")
        llm = LLMManager(provider=provider)
        analyzer = ExerciseAnalyzer(llm, language=lang)

        # Check if provider is ready
        if provider == "ollama":
            console.print(f"   Checking {llm.primary_model}...")
            if not llm.check_model_available(llm.primary_model):
                console.print(f"[red]Model {llm.primary_model} not found![/red]")
                console.print(f"[yellow]Run: ollama pull {llm.primary_model}[/yellow]\n")
                return
            console.print(f"   ✓ {llm.primary_model} ready\n")
        elif provider == "groq":
            console.print(f"   Using Groq API with {llm.primary_model}")
            if not Config.GROQ_API_KEY:
                console.print(f"[red]GROQ_API_KEY not set![/red]")
                console.print(
                    f"[yellow]Get your free API key at: https://console.groq.com[/yellow]"
                )
                console.print(f"[yellow]Then set it: export GROQ_API_KEY=your_key_here[/yellow]\n")
                return
            console.print(f"   ✓ API key found\n")
        elif provider == "anthropic":
            console.print(f"   Using Anthropic API with {llm.primary_model}")
            if not Config.ANTHROPIC_API_KEY:
                console.print(f"[red]ANTHROPIC_API_KEY not set![/red]")
                console.print(
                    f"[yellow]Get your API key at: https://console.anthropic.com[/yellow]"
                )
                console.print(
                    f"[yellow]Then set it: export ANTHROPIC_API_KEY=your_key_here[/yellow]\n"
                )
                return
            console.print(f"   ✓ API key found\n")

        # Step 1: Link materials to topics
        console.print("🔗 Linking materials to topics...")
        console.print("[dim]This may take a while...[/dim]\n")

        analyzer.link_materials_to_topics(course_code)

        # Step 2: Link worked examples to exercises (if requested)
        if link_exercises:
            console.print("\n🔗 Linking worked examples to exercises...")
            console.print("[dim]This may take a while...[/dim]\n")

            analyzer.link_worked_examples_to_exercises(course_code)

        # Summary
        console.print("\n[bold green]✨ Material linking complete![/bold green]\n")
        console.print(f"Next steps:")
        console.print(f"  • examina info --course {course} - View updated course info")
        if not link_exercises:
            console.print(
                f"  • examina link-materials --course {course} --link-exercises - Link worked examples to exercises"
            )
        console.print()

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}\n")
        import traceback

        traceback.print_exc()
        raise click.Abort()


@cli.command(name="split-topics")
@click.option("--course", "-c", required=True, help="Course code")
@click.option(
    "--provider",
    type=click.Choice(["anthropic", "openai", "groq", "ollama", "deepseek"]),
    default=Config.LLM_PROVIDER,
    help=f"LLM provider (default: {Config.LLM_PROVIDER})",
)
@click.option(
    "--lang",
    type=click.Choice(["en", "it"]),
    default=Config.DEFAULT_LANGUAGE,
    help=f"Output language (default: {Config.DEFAULT_LANGUAGE})",
)
@click.option("--dry-run", is_flag=True, help="Preview splits without applying changes")
@click.option("--force", is_flag=True, help="Skip confirmation prompts")
@click.option("--delete-old", is_flag=True, help="Delete old topic if empty after split")
def split_topics(course, provider, lang, dry_run, force, delete_old):
    """Automatically split generic topics into specific subtopics."""
    try:
        console.print(f"\n[bold cyan]Topic Splitting for {course}[/bold cyan]\n")

        # Initialize database and analyzer
        with Database() as db:
            # Check if course exists
            course_obj = db.get_course(course)
            if not course_obj:
                console.print(f"[bold red]Error:[/bold red] Course {course} not found\n")
                return

            # Initialize analyzer
            from models.llm_manager import LLMManager

            llm_manager = LLMManager(provider=provider)
            analyzer = ExerciseAnalyzer(llm_manager=llm_manager, language=lang)

            # Detect generic topics
            console.print(f"[bold]Detecting generic topics...[/bold]")
            generic_topics = analyzer.detect_generic_topics(course, db)

            if not generic_topics:
                console.print(
                    f"[green]✓ No generic topics found! All topics are sufficiently specific.[/green]\n"
                )
                return

            console.print(f"\n[yellow]Found {len(generic_topics)} generic topic(s):[/yellow]\n")
            for topic_info in generic_topics:
                console.print(f"  • {topic_info['name']}")
                console.print(f"    - Core loops: {topic_info['knowledge_item_count']}")
                console.print(f"    - Reason: {topic_info['reason']}\n")

            if dry_run:
                console.print(
                    "[yellow]Dry run mode: showing preview only, no changes will be made[/yellow]\n"
                )

            # Process each generic topic
            for topic_info in generic_topics:
                console.print(f"\n[bold]Processing topic: {topic_info['name']}[/bold]")

                # Get core loops for this topic
                knowledge_items = db.get_knowledge_items_by_topic(topic_info["id"])

                # Cluster core loops using LLM
                console.print(f"  Clustering {len(knowledge_items)} core loops...")
                clusters = analyzer.cluster_knowledge_items_for_topic(
                    topic_info["id"], topic_info["name"], knowledge_items
                )

                if not clusters:
                    console.print(f"  [red]✗ Clustering failed for this topic[/red]")
                    continue

                # Show preview
                console.print(f"\n  [green]✓ Generated {len(clusters)} new topics:[/green]\n")
                for i, cluster in enumerate(clusters, 1):
                    console.print(f"    {i}. [bold]{cluster['topic_name']}[/bold]")
                    console.print(f"       Core loops: {len(cluster['knowledge_item_ids'])}")

                    # Show core loop names
                    loop_names = [
                        cl["name"]
                        for cl in knowledge_items
                        if cl["id"] in cluster["knowledge_item_ids"]
                    ]
                    for loop_name in loop_names[:3]:  # Show first 3
                        console.print(f"         - {loop_name}")
                    if len(loop_names) > 3:
                        console.print(f"         ... and {len(loop_names) - 3} more")
                    console.print()

                if dry_run:
                    console.print("  [yellow](Dry run: skipping actual split)[/yellow]\n")
                    continue

                # Ask for confirmation unless --force
                if not force:
                    console.print(
                        f"  [yellow]Apply this split to topic '{topic_info['name']}'?[/yellow]"
                    )
                    confirm = click.confirm("  Proceed", default=True)
                    if not confirm:
                        console.print("  [yellow]Skipped[/yellow]\n")
                        continue

                # Apply the split
                try:
                    stats = db.split_topic(
                        old_topic_id=topic_info["id"],
                        clusters=clusters,
                        course_code=course,
                        delete_old=delete_old,
                    )

                    console.print(f"\n  [green]✓ Successfully split topic![/green]")
                    console.print(f"    - Old topic: {stats['old_topic_name']}")
                    console.print(f"    - New topics: {len(stats['new_topics'])}")
                    console.print(f"    - Core loops moved: {stats['knowledge_items_moved']}")

                    if delete_old and stats.get("old_topic_deleted"):
                        console.print(f"    - Old topic deleted: Yes")
                    elif delete_old:
                        console.print(
                            f"    - Old topic deleted: No ({stats.get('remaining_knowledge_items', 0)} core loops remain)"
                        )

                    if stats.get("errors"):
                        console.print(f"\n  [yellow]Warnings:[/yellow]")
                        for error in stats["errors"]:
                            console.print(f"    - {error}")

                except Exception as e:
                    console.print(f"\n  [red]✗ Split failed: {e}[/red]")
                    continue

            console.print(f"\n[green]✓ Topic splitting complete![/green]\n")

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}\n")
        import traceback

        traceback.print_exc()
        raise click.Abort()


@cli.command()
@click.option("--course", "-c", required=True, help="Course code")
@click.option("--loop", "-l", required=True, help="Core loop ID to learn")
@click.option(
    "--lang", type=click.Choice(["en", "it"]), default="en", help="Output language (default: en)"
)
@click.option(
    "--depth",
    "-d",
    type=click.Choice(["basic", "medium", "advanced"]),
    default="medium",
    help="Explanation depth: basic (concise), medium (balanced), advanced (comprehensive)",
)
@click.option("--no-concepts", is_flag=True, help="Skip prerequisite concept explanations")
@click.option(
    "--adaptive/--no-adaptive",
    default=True,
    help="Use adaptive teaching (auto-select depth based on mastery, default: enabled)",
)
@click.option("--strategy", is_flag=True, help="Include study strategy and metacognitive guidance")
@click.option(
    "--provider",
    "-p",
    type=click.Choice(["anthropic", "groq", "ollama", "openai", "deepseek"]),
    default=None,
    help="LLM provider (overrides profile routing)",
)
@click.option(
    "--profile",
    type=click.Choice(["free", "pro", "local"]),
    default=None,
    help="Provider profile for routing (free/pro/local). Uses EXAMINA_PROVIDER_PROFILE if not specified.",
)
@click.option(
    "--force", "-f", is_flag=True, help="Skip prerequisite mastery check and learn anyway"
)
def learn(course, loop, lang, depth, no_concepts, adaptive, strategy, provider, profile, force):
    """Learn core loops with AI tutor explanation (enhanced with WHY reasoning)."""
    from core.tutor import Tutor
    from models.llm_manager import LLMManager

    console.print(f"\n[bold cyan]Learning {loop}...[/bold cyan]")

    if adaptive:
        console.print(
            f"[dim]Mode: Adaptive teaching (depth and prerequisites auto-selected based on mastery)[/dim]\n"
        )
    elif not no_concepts:
        console.print(
            f"[dim]Mode: Enhanced learning with foundational concepts (depth: {depth})[/dim]\n"
        )
    else:
        console.print(
            f"[dim]Mode: Direct explanation without prerequisites (depth: {depth})[/dim]\n"
        )

    try:
        # Find course
        with Database() as db:
            all_courses = db.get_all_courses()
            found_course = None
            for c in all_courses:
                if c["code"] == course or c["acronym"] == course:
                    found_course = c
                    break

            if not found_course:
                console.print(f"[red]Course '{course}' not found.[/red]\n")
                return

            course_code = found_course["code"]

            # Look up core loop by name to get its ID
            knowledge_item_row = db.conn.execute(
                """
                SELECT cl.id FROM knowledge_items cl
                JOIN topics t ON cl.topic_id = t.id
                WHERE cl.name = ? AND t.course_code = ?
            """,
                (loop, course_code),
            ).fetchone()

            if not knowledge_item_row:
                console.print(f"[red]Core loop '{loop}' not found in course {course_code}.[/red]\n")
                console.print(
                    "[dim]Use 'examina info --course CODE' to see available core loops.[/dim]\n"
                )
                return

            knowledge_item_id = knowledge_item_row["id"]

            # Check prerequisite mastery (unless --force is used)
            if not force:
                from core.adaptive_teaching import AdaptiveTeachingManager

                with AdaptiveTeachingManager() as atm:
                    prereq_check = atm.check_prerequisite_mastery(course_code, loop)

                    if not prereq_check["ready"]:
                        console.print("[bold yellow]⚠️  Prerequisite Warning[/bold yellow]\n")
                        console.print("Some related concepts in this topic have low mastery:\n")

                        for prereq in prereq_check["weak_prerequisites"][:5]:
                            mastery_pct = prereq["mastery"] * 100
                            console.print(
                                f"  • [yellow]{prereq['name']}[/yellow] - {mastery_pct:.0f}% mastery"
                            )

                        console.print(f"\n[dim]{prereq_check['recommendation']}[/dim]")
                        console.print(
                            "\n[dim]Use --force to skip this check and learn anyway.[/dim]\n"
                        )

                        if not click.confirm("Continue anyway?", default=False):
                            return

        # Determine provider (adaptive uses PREMIUM, otherwise INTERACTIVE)
        task_type = "premium" if adaptive else "interactive"
        effective_provider = get_effective_provider(provider, profile, task_type)

        # Initialize tutor with enhanced learning
        llm = LLMManager(provider=effective_provider)
        tutor = Tutor(llm, language=lang)

        # Get enhanced explanation
        console.print("🤖 Generating deep explanation with reasoning...\n")
        result = tutor.learn(
            course_code=course_code,
            knowledge_item_id=knowledge_item_id,
            explain_concepts=not no_concepts,
            depth=depth,
            adaptive=adaptive,
            include_study_strategy=strategy,
        )

        if not result.success:
            console.print(f"[red]Error: {result.content}[/red]\n")
            return

        # Display explanation
        from rich.markdown import Markdown

        md = Markdown(result.content)
        console.print(md)

        # Display metadata
        includes_prereqs = result.metadata.get("includes_prerequisites", False)
        examples_count = result.metadata.get("examples_count", 0)
        actual_depth = result.metadata.get("depth", depth)
        prereq_status = "with prerequisites" if includes_prereqs else "without prerequisites"
        adaptive_status = "adaptive" if result.metadata.get("adaptive", False) else "manual"
        console.print(
            f"\n[dim]Core loop: {loop} | Depth: {actual_depth} | {prereq_status} | Examples: {examples_count} | Mode: {adaptive_status}[/dim]\n"
        )

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}\n")
        import traceback

        traceback.print_exc()
        raise click.Abort()


@cli.command()
@click.option("--course", "-c", required=True, help="Course code")
@click.option("--exercise", "-e", type=int, help="Specific exercise ID to prove")
@click.option(
    "--interactive",
    is_flag=True,
    default=True,
    help="Interactive mode with step-by-step guidance (default)",
)
@click.option(
    "--technique",
    "-t",
    type=click.Choice(["direct", "contradiction", "induction", "construction", "contrapositive"]),
    help="Force a specific proof technique",
)
@click.option(
    "--lang", type=click.Choice(["en", "it"]), default="en", help="Output language (default: en)"
)
@click.option(
    "--provider",
    "-p",
    type=click.Choice(["anthropic", "groq", "ollama", "openai", "deepseek"]),
    default=None,
    help="LLM provider (overrides profile routing)",
)
@click.option(
    "--profile",
    type=click.Choice(["free", "pro", "local"]),
    default=None,
    help="Provider profile for routing (free/pro/local). Uses EXAMINA_PROVIDER_PROFILE if not specified.",
)
def prove(course, exercise, interactive, technique, lang, provider, profile):
    """Practice mathematical proofs interactively with AI guidance."""
    from core.proof_tutor import ProofTutor
    from models.llm_manager import LLMManager
    from rich.prompt import Confirm

    console.print(f"\n[bold cyan]Proof Practice Mode[/bold cyan]\n")

    try:
        # Find course
        with Database() as db:
            all_courses = db.get_all_courses()
            found_course = None
            for c in all_courses:
                if c["code"] == course or c["acronym"] == course:
                    found_course = c
                    break

            if not found_course:
                console.print(f"[red]Course '{course}' not found.[/red]\n")
                return

            course_code = found_course["code"]

            # Get proof exercise
            if exercise:
                # Get specific exercise
                cursor = db.conn.execute(
                    """SELECT id, text, exercise_type FROM exercises
                       WHERE course_code = ? AND id = ? AND exercise_type IN ('proof', 'theory', 'hybrid')""",
                    (course_code, exercise),
                )
                ex = cursor.fetchone()
                if not ex:
                    console.print(f"[red]Proof exercise {exercise} not found.[/red]\n")
                    return
                ex_id, ex_text, ex_type = ex
            else:
                # Get random proof exercise
                cursor = db.conn.execute(
                    """SELECT id, text, exercise_type FROM exercises
                       WHERE course_code = ? AND exercise_type IN ('proof', 'theory')
                       ORDER BY RANDOM() LIMIT 1""",
                    (course_code,),
                )
                result = cursor.fetchone()
                if not result:
                    console.print(
                        f"[yellow]No proof exercises found for course {course}.[/yellow]\n"
                    )
                    console.print(
                        "[dim]Tip: Run analyze command first to detect exercise types[/dim]\n"
                    )
                    return
                ex_id, ex_text, ex_type = result

        # Determine provider (proof uses PREMIUM for step-by-step guidance)
        effective_provider = get_effective_provider(provider, profile, "premium")

        # Initialize proof tutor
        llm = LLMManager(provider=effective_provider)
        proof_tutor = ProofTutor(llm, language=lang)

        # Display exercise
        console.print(f"[bold]Exercise {ex_id} ({ex_type}):[/bold]")
        from rich.markdown import Markdown

        console.print(Markdown(ex_text))
        console.print()

        if interactive:
            # Interactive mode with step-by-step guidance
            console.print("[bold green]Interactive Proof Practice[/bold green]")
            console.print("[dim]I'll guide you through this proof step by step.[/dim]\n")

            # Get proof analysis
            console.print("🤖 Analyzing proof structure...\n")
            suggested_technique = proof_tutor.suggest_technique(ex_text)
            analysis_result = {"technique": suggested_technique}

            if analysis_result:
                # Show technique suggestion
                suggested_technique = analysis_result.get("technique", "direct")
                actual_technique = technique or suggested_technique

                console.print(f"[bold]Suggested proof technique:[/bold] {suggested_technique}")
                if technique:
                    console.print(f"[dim](You selected: {technique})[/dim]")
                console.print()

                # Show technique info
                technique_info = proof_tutor.PROOF_TECHNIQUES.get(actual_technique)
                if technique_info:
                    console.print(f"[bold cyan]{technique_info.name.title()} Proof:[/bold cyan]")
                    console.print(f"  {technique_info.description}")
                    console.print(f"\n[bold]When to use:[/bold] {technique_info.when_to_use}\n")

                # Get step-by-step guidance
                console.print("🤖 Generating step-by-step proof guidance...\n")
                guidance = proof_tutor.get_proof_guidance(ex_text, actual_technique)

                if guidance and guidance.get("success"):
                    steps = guidance.get("steps", [])
                    console.print(f"[bold]Proof steps ({len(steps)} steps):[/bold]\n")

                    for i, step in enumerate(steps, 1):
                        console.print(f"[cyan]Step {i}:[/cyan] {step}")

                        if Confirm.ask(f"\n[dim]Show hint for step {i}?[/dim]", default=False):
                            hint = proof_tutor.get_hint_for_step(ex_text, actual_technique, i)
                            if hint:
                                console.print(f"[yellow]Hint:[/yellow] {hint}\n")

                        console.print()

                    # Common mistakes
                    if technique_info:
                        console.print(f"\n[bold yellow]⚠️  Common mistakes to avoid:[/bold yellow]")
                        for mistake in technique_info.common_mistakes:
                            console.print(f"  • {mistake}")
                        console.print()

                # Offer full solution
                if Confirm.ask("\n[bold]Show complete solution?[/bold]", default=False):
                    console.print("\n🤖 Generating complete proof...\n")
                    solution = proof_tutor.get_full_proof(ex_text, actual_technique)
                    if solution:
                        console.print(Markdown(solution))
                    console.print()

        else:
            # Non-interactive mode: just show solution
            console.print("🤖 Generating proof solution...\n")
            suggested_technique = technique or "direct"
            solution = proof_tutor.get_full_proof(ex_text, suggested_technique)
            if solution:
                console.print(Markdown(solution))
            console.print()

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}\n")
        import traceback

        traceback.print_exc()
        raise click.Abort()


@cli.command()
@click.option("--course", "-c", required=True, help="Course code")
@click.option("--topic", "-t", help="Topic to practice")
@click.option(
    "--difficulty", "-d", type=click.Choice(["easy", "medium", "hard"]), help="Difficulty level"
)
@click.option(
    "--lang", type=click.Choice(["en", "it"]), default="en", help="Output language (default: en)"
)
@click.option(
    "--provider",
    "-p",
    type=click.Choice(["anthropic", "groq", "ollama", "openai", "deepseek"]),
    default=None,
    help="LLM provider (overrides profile routing)",
)
@click.option(
    "--profile",
    type=click.Choice(["free", "pro", "local"]),
    default=None,
    help="Provider profile for routing (free/pro/local). Uses EXAMINA_PROVIDER_PROFILE if not specified.",
)
def practice(course, topic, difficulty, lang, provider, profile):
    """Practice exercises interactively with AI feedback."""
    from core.tutor import Tutor
    from models.llm_manager import LLMManager

    console.print(f"\n[bold cyan]Practice mode for {course}...[/bold cyan]\n")

    try:
        # Find course
        with Database() as db:
            all_courses = db.get_all_courses()
            found_course = None
            for c in all_courses:
                if c["code"] == course or c["acronym"] == course:
                    found_course = c
                    break

            if not found_course:
                console.print(f"[red]Course '{course}' not found.[/red]\n")
                return

            course_code = found_course["code"]

        # Determine provider (practice uses INTERACTIVE)
        effective_provider = get_effective_provider(provider, profile, "interactive")

        # Initialize tutor
        llm = LLMManager(provider=effective_provider)
        tutor = Tutor(llm, language=lang)

        # Get practice exercise
        console.print("📝 Fetching practice exercise...\n")
        result = tutor.practice(course_code, topic=topic, difficulty=difficulty)

        if not result.success:
            console.print(f"[red]Error: {result.content}[/red]\n")
            return

        # Display exercise
        console.print("[bold]Exercise:[/bold]")
        console.print(result.content)
        console.print()

        # Get user answer
        console.print("[dim]Type your answer (press Enter twice to finish):[/dim]")
        answer_lines = []
        while True:
            line = input()
            if line == "" and answer_lines and answer_lines[-1] == "":
                break
            answer_lines.append(line)

        user_answer = "\n".join(answer_lines[:-1])  # Remove last empty line

        if not user_answer.strip():
            console.print("\n[yellow]No answer provided. Exiting practice mode.[/yellow]\n")
            return

        # Evaluate answer
        console.print("\n🤖 Evaluating your answer...\n")
        feedback = tutor.check_answer(
            result.metadata["exercise_id"], user_answer, provide_hints=True
        )

        if feedback.success:
            console.print(feedback.content)
        else:
            console.print(f"[red]Error: {feedback.content}[/red]")

        console.print()

    except KeyboardInterrupt:
        console.print("\n\n[yellow]Practice session cancelled.[/yellow]\n")
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}\n")
        import traceback

        traceback.print_exc()
        raise click.Abort()


@cli.command()
@click.option("--course", "-c", required=True, help="Course code")
@click.option("--loop", "-l", required=True, help="Core loop ID")
@click.option(
    "--difficulty",
    "-d",
    type=click.Choice(["easy", "medium", "hard"]),
    default="medium",
    help="Exercise difficulty",
)
@click.option(
    "--lang", type=click.Choice(["en", "it"]), default="en", help="Output language (default: en)"
)
@click.option(
    "--provider",
    "-p",
    type=click.Choice(["anthropic", "groq", "ollama", "openai", "deepseek"]),
    default=None,
    help="LLM provider (overrides profile routing)",
)
@click.option(
    "--profile",
    type=click.Choice(["free", "pro", "local"]),
    default=None,
    help="Provider profile for routing (free/pro/local). Uses EXAMINA_PROVIDER_PROFILE if not specified.",
)
def generate(course, loop, difficulty, lang, provider, profile):
    """Generate new practice exercises with AI."""
    from core.tutor import Tutor
    from models.llm_manager import LLMManager

    console.print(f"\n[bold cyan]Generating {difficulty} exercise for {loop}...[/bold cyan]\n")

    try:
        # Find course
        with Database() as db:
            all_courses = db.get_all_courses()
            found_course = None
            for c in all_courses:
                if c["code"] == course or c["acronym"] == course:
                    found_course = c
                    break

            if not found_course:
                console.print(f"[red]Course '{course}' not found.[/red]\n")
                return

            course_code = found_course["code"]

        # Determine provider (generate uses INTERACTIVE)
        effective_provider = get_effective_provider(provider, profile, "interactive")

        # Initialize tutor
        llm = LLMManager(provider=effective_provider)
        tutor = Tutor(llm, language=lang)

        # Generate exercise
        console.print("🤖 Generating new exercise...\n")
        result = tutor.generate(course_code, loop, difficulty=difficulty)

        if not result.success:
            console.print(f"[red]Error: {result.content}[/red]\n")
            return

        # Display generated exercise
        console.print("[bold]Generated Exercise:[/bold]\n")
        console.print(result.content)
        console.print(
            f"\n[dim]Core loop: {loop} | Difficulty: {difficulty} | Based on {result.metadata.get('based_on_examples', 0)} examples[/dim]\n"
        )

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}\n")
        import traceback

        traceback.print_exc()
        raise click.Abort()


@cli.command()
@click.option("--course", "-c", required=True, help="Course code")
@click.option("--interactive", "-i", is_flag=True, help="Interactive proof practice mode")
@click.option(
    "--lang", type=click.Choice(["en", "it"]), default="en", help="Output language (default: en)"
)
@click.option(
    "--provider",
    "-p",
    type=click.Choice(["anthropic", "groq", "ollama", "openai", "deepseek"]),
    default=None,
    help="LLM provider (overrides profile routing)",
)
@click.option(
    "--profile",
    type=click.Choice(["free", "pro", "local"]),
    default=None,
    help="Provider profile for routing (free/pro/local). Uses EXAMINA_PROVIDER_PROFILE if not specified.",
)
def prove(course, interactive, lang, provider, profile):
    """Practice proof exercises with specialized proof guidance."""
    from core.proof_tutor import ProofTutor
    from models.llm_manager import LLMManager
    from rich.panel import Panel

    console.print(f"\n[bold cyan]Proof Practice Mode for {course}...[/bold cyan]\n")

    try:
        # Find course
        with Database() as db:
            all_courses = db.get_all_courses()
            found_course = None
            for c in all_courses:
                if c["code"] == course or c["acronym"] == course:
                    found_course = c
                    break

            if not found_course:
                console.print(f"[red]Course '{course}' not found.[/red]\n")
                return

            course_code = found_course["code"]

            # Find proof exercises
            exercises = db.get_exercises_by_course(course_code)

        # Determine provider (proof uses PREMIUM for step-by-step guidance)
        effective_provider = get_effective_provider(provider, profile, "premium")

        # Initialize proof tutor
        llm = LLMManager(provider=effective_provider)
        proof_tutor = ProofTutor(llm, language=lang)

        # Filter proof exercises
        proof_exercises = [
            ex for ex in exercises if proof_tutor.is_proof_exercise(ex.get("text", ""))
        ]

        if not proof_exercises:
            console.print(
                f"[yellow]No proof exercises found for {found_course['name']}.[/yellow]\n"
            )
            console.print("Try a different course (AL or PC often have proofs).\n")
            return

        console.print(f"Found {len(proof_exercises)} proof exercise(s)\n")

        # Pick a random proof exercise
        import random

        exercise = random.choice(proof_exercises)

        # Display exercise
        console.print("[bold]Proof Exercise:[/bold]\n")
        console.print(Panel(exercise["text"], border_style="blue"))
        console.print()

        # Analyze the proof
        console.print("🔍 Analyzing proof structure...\n")
        analysis = proof_tutor.analyze_proof(course_code, exercise["text"])

        console.print(f"[bold]Proof Analysis:[/bold]")
        console.print(f"  Type: {analysis.proof_type}")
        console.print(f"  Suggested Technique: {analysis.technique_suggested}")
        console.print(f"  Difficulty: {analysis.difficulty}")
        console.print(f"\n  Given: {analysis.premise}")
        console.print(f"  To Prove: {analysis.goal}")
        if analysis.key_concepts:
            console.print(f"  Key Concepts: {', '.join(analysis.key_concepts)}")
        console.print()

        if interactive:
            # Get user's proof attempt
            console.print("[dim]Type your proof attempt (press Enter twice to submit):[/dim]\n")
            answer_lines = []
            empty_count = 0

            while True:
                try:
                    line = input()
                    if line == "":
                        empty_count += 1
                        if empty_count >= 2:
                            break
                    else:
                        empty_count = 0
                    answer_lines.append(line)
                except EOFError:
                    break

            user_proof = "\n".join(answer_lines[:-1]) if answer_lines else ""

            if not user_proof.strip():
                console.print("\n[yellow]No proof provided. Showing solution instead...[/yellow]\n")
                # Generate explanation
                console.print("🤖 Generating proof explanation...\n")
                explanation = proof_tutor.learn_proof(course_code, exercise["id"], exercise["text"])
                from rich.markdown import Markdown

                md = Markdown(explanation)
                console.print(md)
            else:
                # Evaluate proof attempt
                console.print("\n🤖 Evaluating your proof...\n")
                result = proof_tutor.practice_proof(
                    course_code, exercise["text"], user_proof, provide_hints=True
                )

                # Display feedback
                from rich.markdown import Markdown

                md = Markdown(result["feedback"])
                console.print(md)

                score_pct = int(result["score"] * 100)
                if result["is_correct"]:
                    console.print(f"\n[green]✅ Correct! Score: {score_pct}%[/green]\n")
                else:
                    console.print(f"\n[yellow]Score: {score_pct}%[/yellow]\n")
        else:
            # Just show explanation
            console.print("🤖 Generating proof explanation...\n")
            explanation = proof_tutor.learn_proof(course_code, exercise["id"], exercise["text"])
            from rich.markdown import Markdown

            md = Markdown(explanation)
            console.print(md)
            console.print(f"\n[dim]Exercise ID: {exercise['id'][:20]}...[/dim]\n")

    except KeyboardInterrupt:
        console.print("\n\n[yellow]Proof practice cancelled.[/yellow]\n")
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}\n")
        import traceback

        traceback.print_exc()
        raise click.Abort()


@cli.command()
@click.option("--course", "-c", required=True, help="Course code (e.g., B006802 or ADE)")
@click.option("--questions", "-n", type=int, default=10, help="Number of questions (default: 10)")
@click.option("--topic", "-t", help="Filter by topic")
@click.option("--loop", "-l", help="Filter by core loop ID or name pattern")
@click.option(
    "--difficulty", "-d", type=click.Choice(["easy", "medium", "hard"]), help="Filter by difficulty"
)
@click.option("--review-only", is_flag=True, help="Only exercises due for review")
@click.option(
    "--adaptive",
    "-a",
    is_flag=True,
    help="Adaptive selection based on mastery (40% weak, 40% learning, 20% strong)",
)
@click.option(
    "--procedure",
    "-p",
    type=click.Choice(
        ["design", "transformation", "verification", "minimization", "analysis", "implementation"]
    ),
    help="Filter by procedure type",
)
@click.option("--multi-only", is_flag=True, help="Only show multi-procedure exercises")
@click.option("--tags", help="Filter by tags (comma-separated)")
@click.option(
    "--type",
    "exercise_type",
    type=click.Choice(["procedural", "theory", "proof"]),
    help="Filter by exercise type (procedural=design/implementation, theory=analysis, proof=proofs)",
)
@click.option(
    "--lang",
    type=click.Choice(["en", "it"]),
    default="en",
    help="Language for feedback (default: en)",
)
@click.option(
    "--provider",
    type=click.Choice(["anthropic", "groq", "ollama", "openai", "deepseek"]),
    default=None,
    help="LLM provider (overrides profile routing)",
)
@click.option(
    "--profile",
    type=click.Choice(["free", "pro", "local"]),
    default=None,
    help="Provider profile for routing (free/pro/local). Uses EXAMINA_PROVIDER_PROFILE if not specified.",
)
def quiz(
    course,
    questions,
    topic,
    loop,
    difficulty,
    review_only,
    adaptive,
    procedure,
    multi_only,
    tags,
    exercise_type,
    lang,
    provider,
    profile,
):
    """Take an interactive quiz to test your knowledge."""
    from core.quiz_engine import QuizEngine
    from models.llm_manager import LLMManager
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    import time

    try:
        # Find course
        with Database() as db:
            all_courses = db.get_all_courses()
            found_course = None
            for c in all_courses:
                if c["code"] == course or c["acronym"] == course:
                    found_course = c
                    break

            if not found_course:
                console.print(f"\n[red]Course '{course}' not found.[/red]\n")
                console.print("Use 'examina courses' to see available courses.\n")
                return

            course_code = found_course["code"]

        # Determine provider (quiz uses INTERACTIVE)
        effective_provider = get_effective_provider(provider, profile, "interactive")

        # Initialize quiz engine
        llm = LLMManager(provider=effective_provider)
        quiz_engine = QuizEngine(llm_manager=llm, language=lang)

        # Create quiz session
        console.print(f"\n[bold cyan]Creating quiz for {found_course['name']}...[/bold cyan]\n")

        try:
            session = quiz_engine.create_quiz_session(
                course_code=course_code,
                num_questions=questions,
                topic=topic,
                knowledge_item=loop,
                difficulty=difficulty,
                review_only=review_only,
                procedure_type=procedure,
                multi_only=multi_only,
                tags=tags,
                exercise_type=exercise_type,
                adaptive=adaptive,
            )
        except ValueError as e:
            console.print(f"[red]Error: {e}[/red]\n")
            console.print("Try different filters or add more exercises.\n")
            return

        # Show prerequisite awareness for adaptive mode
        if adaptive:
            from core.adaptive_teaching import AdaptiveTeachingManager
            from core.mastery_aggregator import MasteryAggregator

            with Database() as db:
                aggregator = MasteryAggregator(db)
                weak_loops = aggregator.get_weak_knowledge_items(course_code, threshold=0.4)

                if weak_loops:
                    console.print("[dim]📊 Adaptive mode detected weak areas:[/dim]")
                    for wl in weak_loops[:3]:
                        console.print(
                            f"   [yellow]• {wl['knowledge_item_name']}[/yellow] ({wl['mastery_score']:.0%} mastery)"
                        )
                    console.print("[dim]   Quiz will prioritize these areas.\n[/dim]")

                # If filtering by specific core loop, check prerequisites
                if loop:
                    with AdaptiveTeachingManager() as atm:
                        prereq_check = atm.check_prerequisite_mastery(
                            course_code, loop, threshold=0.4
                        )
                        if not prereq_check["ready"]:
                            console.print(f"[dim]💡 Tip: {prereq_check['recommendation']}[/dim]\n")

        # Display quiz info
        quiz_info = f"📝 Quiz Session: {session.total_questions} questions"
        if adaptive:
            quiz_info += " | Mode: Adaptive (mastery-based)"
        if topic:
            quiz_info += f" | Topic: {topic}"
        if loop:
            quiz_info += f" | Core Loop: {loop}"
        if difficulty:
            quiz_info += f" | Difficulty: {difficulty}"
        if procedure:
            quiz_info += f" | Procedure: {procedure}"
        if exercise_type:
            quiz_info += f" | Type: {exercise_type}"
        if multi_only:
            quiz_info += " | Multi-Procedure Only"
        if review_only:
            quiz_info += " | Review Mode"

        console.print(Panel(quiz_info, style="cyan"))
        console.print()

        # Quiz loop
        for i, question in enumerate(session.questions, 1):
            question_start_time = time.time()

            # Display question
            console.print(f"[bold]Question {i}/{session.total_questions}[/bold]")

            # Display metadata - check if multi-procedure exercise
            if question.knowledge_items and len(question.knowledge_items) > 1:
                console.print(
                    f"[dim]Topic: {question.topic_name} | Difficulty: {question.difficulty}[/dim]"
                )
                console.print("[dim]Procedures:[/dim]")
                for idx, loop in enumerate(question.knowledge_items, 1):
                    loop_name = loop.get("name", "Unknown")
                    console.print(f"[dim]  {idx}. {loop_name}[/dim]")
                console.print()
            else:
                console.print(
                    f"[dim]Topic: {question.topic_name} | Core Loop: {question.knowledge_item_name} | Difficulty: {question.difficulty}[/dim]\n"
                )

            console.print(Panel(question.exercise_text, title="Exercise", border_style="blue"))
            console.print()

            # Get user answer
            console.print("[dim]Type your answer (press Enter twice to submit):[/dim]")
            answer_lines = []
            empty_count = 0

            while True:
                try:
                    line = input()
                    if line == "":
                        empty_count += 1
                        if empty_count >= 2:
                            break
                    else:
                        empty_count = 0
                    answer_lines.append(line)
                except EOFError:
                    break

            user_answer = "\n".join(answer_lines[:-1]) if answer_lines else ""

            if not user_answer.strip():
                console.print("\n[yellow]Skipped (no answer provided)[/yellow]\n")
                continue

            # Evaluate answer
            console.print("\n[dim]Evaluating your answer...[/dim]")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task(description="Checking...", total=None)
                evaluation = quiz_engine.evaluate_answer(
                    session=session, question=question, user_answer=user_answer, provide_hints=False
                )

            # Update question with results
            question.user_answer = user_answer
            question.is_correct = evaluation["is_correct"]
            question.score = evaluation["score"]
            question.feedback = evaluation["feedback"]
            question.time_spent = int(time.time() - question_start_time)

            # Display feedback
            console.print()
            if question.is_correct:
                console.print(
                    Panel(evaluation["feedback"], title="✅ Correct!", border_style="green")
                )
            else:
                console.print(
                    Panel(evaluation["feedback"], title="❌ Incorrect", border_style="red")
                )

            console.print(f"\n[dim]Score: {question.score:.1%}[/dim]")

            # Show real-time mastery update for this exercise
            if adaptive:
                from core.mastery_aggregator import MasteryAggregator

                with Database() as db:
                    agg = MasteryAggregator(db)
                    ex_mastery = agg.get_exercise_mastery(question.exercise_id)
                    if ex_mastery:
                        level = ex_mastery["mastery_level"]
                        level_color = {
                            "new": "red",
                            "learning": "yellow",
                            "reviewing": "cyan",
                            "mastered": "green",
                        }.get(level, "white")
                        console.print(
                            f"[dim]Mastery: [{level_color}]{level}[/{level_color}] (reps: {ex_mastery['repetition_number']}, next: {ex_mastery['interval_days']}d)[/dim]"
                        )

            # Show progress
            answered = i
            session.total_correct = sum(1 for q in session.questions[:answered] if q.is_correct)
            console.print(
                f"[dim]Progress: {answered}/{session.total_questions} | Correct: {session.total_correct}/{answered}[/dim]\n"
            )

            # Wait for user to continue (except on last question)
            if i < session.total_questions:
                input("[dim]Press Enter to continue...[/dim]\n")
                console.print()

        # Complete session
        quiz_engine.complete_session(session)

        # Display final results
        console.print("\n" + "=" * 60 + "\n")
        console.print("[bold cyan]📊 Quiz Complete![/bold cyan]\n")

        final_score = session.score * 100
        score_color = "green" if final_score >= 80 else "yellow" if final_score >= 60 else "red"

        console.print(
            f"[bold]Final Score: [{score_color}]{final_score:.1f}%[/{score_color}][/bold]"
        )
        console.print(f"Correct: {session.total_correct}/{session.total_questions}")

        if session.started_at and session.completed_at:
            duration = (session.completed_at - session.started_at).total_seconds()
            console.print(f"Time: {int(duration // 60)}m {int(duration % 60)}s")

        # Show mastery updates
        console.print("\n[bold]Mastery Updates:[/bold]")
        from core.analytics import ProgressAnalytics

        analytics = ProgressAnalytics()

        # Get unique core loops from quiz
        knowledge_items = set(q.knowledge_item_id for q in session.questions if q.knowledge_item_id)
        for loop_id in knowledge_items:
            progress = analytics.get_knowledge_item_progress(course_code, loop_id)
            loop_name = next(
                (
                    q.knowledge_item_name
                    for q in session.questions
                    if q.knowledge_item_id == loop_id
                ),
                loop_id,
            )

            mastery_pct = progress["mastery_score"] * 100
            mastery_color = (
                "green" if mastery_pct >= 80 else "yellow" if mastery_pct >= 50 else "red"
            )

            console.print(
                f"  • {loop_name}: [{mastery_color}]{mastery_pct:.0f}%[/{mastery_color}] mastery"
            )

        # Show personalized learning path recommendations (adaptive mode)
        if adaptive:
            from core.adaptive_teaching import AdaptiveTeachingManager

            atm = AdaptiveTeachingManager()
            learning_path = atm.get_personalized_learning_path(course_code, limit=3)

            if learning_path:
                console.print("\n[bold]📚 Recommended Next Steps:[/bold]")
                for item in learning_path:
                    action = item.get("action", "practice")
                    action_emoji = {
                        "review": "🔄",
                        "strengthen": "💪",
                        "learn": "📖",
                        "practice": "✏️",
                    }.get(action, "→")
                    name = item.get("knowledge_item_name") or item.get("topic_name", "Unknown")
                    reason = item.get("reason", "")
                    console.print(f"  {action_emoji} {action.capitalize()}: [cyan]{name}[/cyan]")
                    if reason:
                        console.print(f"     [dim]{reason}[/dim]")

        console.print(f"\n[dim]Session ID: {session.session_id}[/dim]\n")

    except KeyboardInterrupt:
        console.print("\n\n[yellow]Quiz cancelled.[/yellow]\n")
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}\n")
        import traceback

        traceback.print_exc()
        raise click.Abort()


@cli.command()
@click.option("--course", "-c", help="Course code (e.g., B006802 or ADE)")
def suggest(course):
    """Get personalized study suggestions based on spaced repetition."""
    from core.analytics import ProgressAnalytics

    try:
        # Find course
        course_code = None
        if course:
            with Database() as db:
                all_courses = db.get_all_courses()
                found_course = None
                for c in all_courses:
                    if c["code"] == course or c["acronym"] == course:
                        found_course = c
                        break

                if not found_course:
                    console.print(f"\n[red]Course '{course}' not found.[/red]\n")
                    console.print("Use 'examina courses' to see available courses.\n")
                    return

                course_code = found_course["code"]

        # Get suggestions
        analytics = ProgressAnalytics()
        suggestions = analytics.get_study_suggestions(course_code)

        # Display suggestions
        console.print("\n[bold cyan]📚 Study Suggestions[/bold cyan]\n")

        if course_code:
            with Database() as db:
                course_info = db.get_course(course_code)
                console.print(
                    f"[dim]Course: {course_info['name']} ({course_info['acronym']})[/dim]\n"
                )

        for suggestion in suggestions:
            console.print(f"  {suggestion}")

        console.print("\n[dim]Use 'examina quiz --course <CODE>' to start practicing![/dim]\n")

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}\n")
        import traceback

        traceback.print_exc()
        raise click.Abort()


@cli.command()
@click.option("--course", "-c", help="Course code (e.g., B006802 or ADE)")
@click.option("--topics", is_flag=True, help="Show topic breakdown")
@click.option("--detailed", is_flag=True, help="Show detailed statistics")
def progress(course, topics, detailed):
    """View your learning progress and mastery levels."""
    from core.analytics import ProgressAnalytics
    from rich.progress import Progress as RichProgress, BarColumn, TextColumn, TaskProgressColumn

    try:
        # Find course
        if not course:
            console.print("\n[yellow]Please specify a course with --course[/yellow]\n")
            console.print("Use 'examina courses' to see available courses.\n")
            return

        with Database() as db:
            all_courses = db.get_all_courses()
            found_course = None
            for c in all_courses:
                if c["code"] == course or c["acronym"] == course:
                    found_course = c
                    break

            if not found_course:
                console.print(f"\n[red]Course '{course}' not found.[/red]\n")
                console.print("Use 'examina courses' to see available courses.\n")
                return

            course_code = found_course["code"]

        # Get analytics
        analytics = ProgressAnalytics()
        summary = analytics.get_course_summary(course_code)

        # Display course header
        console.print(f"\n[bold cyan]{found_course['name']}[/bold cyan]")
        console.print(f"[dim]{found_course['code']} • {found_course['acronym']}[/dim]\n")

        # Overall progress
        console.print("[bold]📊 Overall Progress[/bold]\n")

        # Create progress bars
        if summary["total_exercises"] > 0:
            # Attempted progress
            attempted_pct = (summary["exercises_attempted"] / summary["total_exercises"]) * 100
            mastered_pct = (summary["exercises_mastered"] / summary["total_exercises"]) * 100

            console.print(
                f"Exercises Attempted: {summary['exercises_attempted']}/{summary['total_exercises']} ({attempted_pct:.1f}%)"
            )
            with RichProgress(
                TextColumn(""),
                BarColumn(complete_style="cyan", finished_style="cyan"),
                TaskProgressColumn(),
                console=console,
            ) as progress_bar:
                task = progress_bar.add_task("", total=summary["total_exercises"])
                progress_bar.update(task, completed=summary["exercises_attempted"])

            console.print(
                f"\nExercises Mastered: {summary['exercises_mastered']}/{summary['total_exercises']} ({mastered_pct:.1f}%)"
            )
            with RichProgress(
                TextColumn(""),
                BarColumn(complete_style="green", finished_style="green"),
                TaskProgressColumn(),
                console=console,
            ) as progress_bar:
                task = progress_bar.add_task("", total=summary["total_exercises"])
                progress_bar.update(task, completed=summary["exercises_mastered"])

            console.print(f"\nOverall Mastery: {summary['overall_mastery']:.1%}")
            mastery_color = (
                "green"
                if summary["overall_mastery"] >= 0.8
                else "yellow"
                if summary["overall_mastery"] >= 0.5
                else "red"
            )
            with RichProgress(
                TextColumn(""),
                BarColumn(complete_style=mastery_color, finished_style=mastery_color),
                TaskProgressColumn(),
                console=console,
            ) as progress_bar:
                task = progress_bar.add_task("", total=100)
                progress_bar.update(task, completed=int(summary["overall_mastery"] * 100))
        else:
            console.print(
                "[yellow]No exercises found. Run 'examina ingest' and 'examina analyze' first.[/yellow]"
            )

        console.print()

        # Quiz statistics
        if summary["quiz_sessions_completed"] > 0:
            console.print("[bold]🎯 Quiz Statistics[/bold]\n")
            console.print(f"Sessions Completed: {summary['quiz_sessions_completed']}")
            console.print(f"Average Score: {summary['avg_score']:.1%}")
            console.print(f"Total Time: {summary['total_time_spent']} minutes\n")

        # Core loops progress
        if summary["knowledge_items_discovered"] > 0:
            console.print("[bold]🔄 Core Loops[/bold]\n")
            console.print(f"Discovered: {summary['knowledge_items_discovered']}")
            console.print(f"Attempted: {summary['knowledge_items_attempted']}")

            if summary["knowledge_items_attempted"] > 0:
                progress_pct = (
                    summary["knowledge_items_attempted"] / summary["knowledge_items_discovered"]
                ) * 100
                console.print(f"Progress: {progress_pct:.1f}%")

            console.print()

        # Topic breakdown
        if topics or detailed:
            console.print("[bold]📚 Topic Breakdown[/bold]\n")
            breakdown = analytics.get_topic_breakdown(course_code)

            if breakdown:
                # Create table
                from rich.table import Table

                table = Table(show_header=True, header_style="bold")
                table.add_column("Topic", style="cyan")
                table.add_column("Status", justify="center")
                table.add_column("Mastery", justify="right")
                table.add_column("Exercises", justify="right")

                for topic_data in breakdown:
                    # Status icon
                    status_icons = {
                        "mastered": "✅",
                        "in_progress": "🔄",
                        "weak": "⚠️",
                        "not_started": "❌",
                    }
                    status = status_icons.get(topic_data["status"], "❓")

                    # Mastery color
                    mastery = topic_data["mastery_score"]
                    mastery_color = (
                        "green"
                        if mastery >= 0.8
                        else "yellow"
                        if mastery >= 0.5
                        else "red"
                        if mastery > 0
                        else "dim"
                    )
                    mastery_str = f"[{mastery_color}]{mastery:.1%}[/{mastery_color}]"

                    # Exercises
                    exercises_str = (
                        f"{topic_data['exercises_attempted']}/{topic_data['exercises_count']}"
                    )

                    table.add_row(topic_data["topic_name"], status, mastery_str, exercises_str)

                console.print(table)
                console.print()
            else:
                console.print("[dim]No topics discovered yet.[/dim]\n")

        # Detailed statistics
        if detailed:
            console.print("[bold]🔍 Detailed Statistics[/bold]\n")

            # Weak areas
            weak_areas = analytics.get_weak_areas(course_code)
            if weak_areas:
                console.print("[bold red]Weak Areas (< 50% mastery):[/bold red]")
                for area in weak_areas[:5]:  # Top 5
                    console.print(
                        f"  • {area['name']} ({area['topic_name']}): {area['mastery_score']:.1%} mastery"
                    )
                console.print()

            # Due reviews
            due_reviews = analytics.get_due_reviews(course_code)
            if due_reviews:
                overdue = [r for r in due_reviews if r["priority"] == "overdue"]
                due_today = [r for r in due_reviews if r["priority"] == "due_today"]

                if overdue:
                    console.print(f"[bold red]Overdue Reviews ({len(overdue)}):[/bold red]")
                    for review in overdue[:5]:
                        console.print(
                            f"  • {review['knowledge_item_name']}: {review['days_overdue']} days overdue"
                        )
                    console.print()

                if due_today:
                    console.print(f"[bold yellow]Due Today ({len(due_today)}):[/bold yellow]")
                    for review in due_today[:5]:
                        console.print(f"  • {review['knowledge_item_name']}")
                    console.print()

        # Next steps
        console.print(
            "[dim]Use 'examina suggest --course {0}' for study recommendations[/dim]".format(
                course_code
            )
        )
        console.print(
            "[dim]Use 'examina quiz --course {0}' to start practicing[/dim]\n".format(course_code)
        )

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}\n")
        import traceback

        traceback.print_exc()
        raise click.Abort()


@cli.command()
@click.option("--course", "-c", required=True, help="Course code")
@click.option("--loop", "-l", required=True, help="Core loop ID or name")
@click.option(
    "--difficulty",
    "-d",
    type=click.Choice(["easy", "medium", "hard"]),
    default="medium",
    help="Difficulty level for strategy adaptation (default: medium)",
)
@click.option(
    "--lang", type=click.Choice(["en", "it"]), default="en", help="Output language (default: en)"
)
def strategy(course, loop, difficulty, lang):
    """View study strategy and metacognitive guidance for a core loop."""
    from core.study_strategies import StudyStrategyManager
    from rich.markdown import Markdown

    console.print(f"\n[bold cyan]Study Strategy for {loop}...[/bold cyan]\n")

    try:
        # Find course
        with Database() as db:
            all_courses = db.get_all_courses()
            found_course = None
            for c in all_courses:
                if c["code"] == course or c["acronym"] == course:
                    found_course = c
                    break

            if not found_course:
                console.print(f"[red]Course '{course}' not found.[/red]\n")
                return

            course_code = found_course["code"]

            # Get core loop details
            knowledge_item = db.conn.execute(
                """
                SELECT cl.*, t.name as topic_name
                FROM knowledge_items cl
                JOIN topics t ON cl.topic_id = t.id
                WHERE cl.id = ? AND t.course_code = ?
            """,
                (loop, course_code),
            ).fetchone()

            if not knowledge_item:
                # Try searching by name pattern
                knowledge_item = db.conn.execute(
                    """
                    SELECT cl.*, t.name as topic_name
                    FROM knowledge_items cl
                    JOIN topics t ON cl.topic_id = t.id
                    WHERE cl.name LIKE ? AND t.course_code = ?
                    LIMIT 1
                """,
                    (f"%{loop}%", course_code),
                ).fetchone()

            if not knowledge_item:
                console.print(f"[red]Core loop '{loop}' not found for course {course}.[/red]\n")
                console.print(
                    "Use 'examina info --course {0}' to see available core loops.\n".format(course)
                )
                return

        knowledge_item_dict = dict(knowledge_item)
        knowledge_item_name = knowledge_item_dict.get("name", "")

        # Initialize strategy manager
        strategy_mgr = StudyStrategyManager(language=lang)

        # Get strategy
        strat = strategy_mgr.get_strategy_for_knowledge_item(
            knowledge_item_name, difficulty=difficulty
        )

        if not strat:
            console.print(
                f"[yellow]No specific strategy found for '{knowledge_item_name}'.[/yellow]\n"
            )
            console.print("This core loop may be new or not yet covered by the strategy system.\n")
            return

        # Format and display
        formatted_strategy = strategy_mgr.format_strategy_output(strat, knowledge_item_name)
        md = Markdown(formatted_strategy)
        console.print(md)

        console.print(
            f"\n[dim]Core loop: {knowledge_item_name} | Difficulty: {difficulty} | Language: {lang}[/dim]\n"
        )

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}\n")
        import traceback

        traceback.print_exc()
        raise click.Abort()


@cli.command()
@click.option("--course", "-c", required=True, help="Course code")
@click.option(
    "--limit", "-n", type=int, default=10, help="Number of items in learning path (default: 10)"
)
@click.option(
    "--lang", type=click.Choice(["en", "it"]), default="en", help="Output language (default: en)"
)
def path(course, limit, lang):
    """Show personalized learning path based on mastery and spaced repetition."""
    from core.adaptive_teaching import AdaptiveTeachingManager
    from rich.table import Table

    try:
        # Find course
        with Database() as db:
            all_courses = db.get_all_courses()
            found_course = None
            for c in all_courses:
                if c["code"] == course or c["acronym"] == course:
                    found_course = c
                    break

            if not found_course:
                console.print(f"\n[red]Course '{course}' not found.[/red]\n")
                console.print("Use 'examina courses' to see available courses.\n")
                return

            course_code = found_course["code"]

        # Get personalized learning path
        with AdaptiveTeachingManager() as atm:
            learning_path = atm.get_personalized_learning_path(course_code, limit=limit)

        if not learning_path:
            console.print(f"\n[yellow]No learning path available yet.[/yellow]")
            console.print(f"[dim]Start by taking quizzes to build your progress data.[/dim]\n")
            return

        # Display header
        console.print(f"\n[bold cyan]📚 Personalized Learning Path[/bold cyan]")
        console.print(f"[dim]{found_course['name']} ({found_course['acronym']})[/dim]\n")

        # Create table
        table = Table(show_header=True, header_style="bold")
        table.add_column("#", style="cyan", justify="center", width=3)
        table.add_column("Action", style="yellow", width=10)
        table.add_column("Core Loop", style="white")
        table.add_column("Topic", style="dim")
        table.add_column("Reason", style="green")
        table.add_column("Time", justify="right", style="magenta")

        for item in learning_path:
            # Format action with emoji
            action_icons = {"review": "🔄", "strengthen": "💪", "learn": "📖", "practice": "✍️"}
            action_display = f"{action_icons.get(item['action'], '•')} {item['action'].title()}"

            # Format urgency color
            urgency_colors = {"high": "red", "medium": "yellow", "low": "dim"}
            urgency_color = urgency_colors.get(item.get("urgency", "low"), "dim")

            table.add_row(
                str(item["priority"]),
                action_display,
                f"[{urgency_color}]{item['knowledge_item']}[/{urgency_color}]",
                item["topic"],
                item["reason"],
                f"{item['estimated_time']}m",
            )

        console.print(table)
        console.print(
            f"\n[dim]Total estimated time: {sum(item['estimated_time'] for item in learning_path)} minutes[/dim]\n"
        )

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}\n")
        import traceback

        traceback.print_exc()
        raise click.Abort()


@cli.command()
@click.option("--course", "-c", required=True, help="Course code")
@click.option("--loop", "-l", help="Filter by specific core loop")
@click.option(
    "--lang", type=click.Choice(["en", "it"]), default="en", help="Output language (default: en)"
)
def gaps(course, loop, lang):
    """Identify knowledge gaps and weak areas."""
    from core.adaptive_teaching import AdaptiveTeachingManager

    try:
        # Find course
        with Database() as db:
            all_courses = db.get_all_courses()
            found_course = None
            for c in all_courses:
                if c["code"] == course or c["acronym"] == course:
                    found_course = c
                    break

            if not found_course:
                console.print(f"\n[red]Course '{course}' not found.[/red]\n")
                console.print("Use 'examina courses' to see available courses.\n")
                return

            course_code = found_course["code"]

        # Detect knowledge gaps
        with AdaptiveTeachingManager() as atm:
            knowledge_gaps = atm.detect_knowledge_gaps(course_code, knowledge_item_name=loop)

        if not knowledge_gaps:
            console.print(f"\n[green]✅ No significant knowledge gaps detected![/green]")
            console.print(f"[dim]Your mastery levels look good across all topics.[/dim]\n")
            return

        # Display header
        console.print(f"\n[bold cyan]🔍 Knowledge Gaps Analysis[/bold cyan]")
        console.print(f"[dim]{found_course['name']} ({found_course['acronym']})[/dim]\n")

        # Group by severity
        high_gaps = [g for g in knowledge_gaps if g["severity"] == "high"]
        medium_gaps = [g for g in knowledge_gaps if g["severity"] == "medium"]
        low_gaps = [g for g in knowledge_gaps if g["severity"] == "low"]

        # Display high priority gaps
        if high_gaps:
            console.print("[bold red]⚠️  High Priority Gaps[/bold red]\n")
            for gap in high_gaps:
                mastery_pct = int(gap["mastery"] * 100)
                console.print(f"  [red]•[/red] [bold]{gap['gap']}[/bold] ({gap['topic']})")
                console.print(f"    Mastery: {mastery_pct}%")
                console.print(f"    💡 {gap['recommendation']}")
                if gap["impact"]:
                    console.print(f"    Affects: {', '.join(gap['impact'][:3])}")
                console.print()

        # Display medium priority gaps
        if medium_gaps:
            console.print("[bold yellow]⚡ Medium Priority Gaps[/bold yellow]\n")
            for gap in medium_gaps:
                mastery_pct = int(gap["mastery"] * 100)
                console.print(
                    f"  [yellow]•[/yellow] {gap['gap']} ({gap['topic']}) - {mastery_pct}% mastery"
                )
                console.print(f"    💡 {gap['recommendation']}\n")

        # Display low priority gaps (summarized)
        if low_gaps:
            console.print(
                f"[dim]ℹ️  {len(low_gaps)} additional area(s) for improvement (low priority)[/dim]\n"
            )

        # Summary
        console.print(f"[bold]Summary:[/bold]")
        console.print(f"  Total gaps found: {len(knowledge_gaps)}")
        console.print(f"  High priority: {len(high_gaps)}")
        console.print(f"  Medium priority: {len(medium_gaps)}")
        console.print(f"  Low priority: {len(low_gaps)}\n")

        console.print(
            f"[dim]Use 'examina path --course {course}' to see a personalized study plan[/dim]\n"
        )

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}\n")
        import traceback

        traceback.print_exc()
        raise click.Abort()


@cli.command()
@click.option("--course", "-c", required=True, help="Course code")
@click.option("--dry-run", is_flag=True, help="Show what would be merged without making changes")
@click.option(
    "--threshold",
    type=float,
    default=None,
    help="Similarity threshold (0.0-1.0, default: 0.85 for semantic, 0.85 for string)",
)
@click.option(
    "--bilingual",
    is_flag=True,
    help="Enable LLM-based translation matching (works for ANY language pair)",
)
@click.option("--clean-orphans", is_flag=True, help="Delete orphaned core loops with no exercises")
def deduplicate(course, dry_run, threshold, bilingual, clean_orphans):
    """Merge duplicate exercises, topics, and core loops using LLM-based synonym detection."""
    from difflib import SequenceMatcher
    import hashlib

    # Try to import LLM manager and detect_synonyms
    semantic_matcher = None  # Removed - using LLM-based detect_synonyms() instead
    llm_manager = None
    try:
        from models.llm_manager import LLMManager
        from core.merger import detect_synonyms

        # Create LLM manager for synonym detection
        llm_provider = Config.LLM_PROVIDER
        llm_manager = LLMManager(provider=llm_provider)
        console.print(f"[info]LLM provider: {llm_provider}[/info]")

        if Config.SEMANTIC_SIMILARITY_ENABLED:
            console.print(f"[info]Using: LLM-based synonym detection (anonymous approach)[/info]")
            use_semantic = True
            default_threshold = Config.SEMANTIC_SIMILARITY_THRESHOLD
        else:
            console.print("[info]Semantic matching disabled, using string similarity[/info]")
            use_semantic = False
            default_threshold = Config.KNOWLEDGE_ITEM_SIMILARITY_THRESHOLD
    except ImportError as e:
        console.print(
            f"[yellow]LLM-based synonym detection not available ({e}), using string similarity[/yellow]"
        )
        use_semantic = False
        default_threshold = Config.KNOWLEDGE_ITEM_SIMILARITY_THRESHOLD

    # Use provided threshold or default
    threshold = threshold if threshold is not None else default_threshold

    # REMOVED: Hardcoded bilingual_translations dictionary
    # Translation detection removed - names always extracted in English

    console.print(f"\n[bold cyan]Deduplicating {course}...[/bold cyan]")
    console.print(f"[info]Threshold: {threshold:.2f}[/info]")
    if bilingual:
        console.print(f"[info]Bilingual mode: ENABLED (works for ANY language pair)[/info]")
    console.print()

    if dry_run:
        console.print("[yellow]DRY RUN MODE - No changes will be made[/yellow]\n")

    try:
        with Database() as db:
            # Find course
            all_courses = db.get_all_courses()
            found_course = None
            for c in all_courses:
                if c["code"] == course or c["acronym"] == course:
                    found_course = c
                    break

            if not found_course:
                console.print(f"[red]Course '{course}' not found.[/red]\n")
                return

            course_code = found_course["code"]

            # NEW: Deduplicate exercises (by text hash)
            console.print("[bold]Deduplicating Exercises...[/bold]")
            cursor = db.conn.execute(
                """
                SELECT id, text, exercise_type, course_code
                FROM exercises
                WHERE course_code = ? AND analyzed = 1
                ORDER BY id
            """,
                (course_code,),
            )

            exercises = cursor.fetchall()
            exercise_hashes = {}
            exercise_duplicates = []

            for ex_id, text, ex_type, _ in exercises:
                # Create hash of normalized text (remove whitespace variations)
                normalized = " ".join(text.split())
                text_hash = hashlib.md5(normalized.encode()).hexdigest()

                if text_hash in exercise_hashes:
                    # Found duplicate
                    original_id = exercise_hashes[text_hash]
                    exercise_duplicates.append((original_id, ex_id, ex_type))
                else:
                    exercise_hashes[text_hash] = ex_id

            if exercise_duplicates:
                console.print(f"Found {len(exercise_duplicates)} duplicate exercise(s):\n")
                for orig_id, dup_id, ex_type in exercise_duplicates:
                    ex_type_display = ex_type or "procedural"
                    console.print(f"  • {orig_id[:30]}... ← {dup_id[:30]}... ({ex_type_display})")

                if not dry_run:
                    for orig_id, dup_id, ex_type in exercise_duplicates:
                        # Delete duplicate exercise (foreign keys will handle cleanup)
                        db.conn.execute("DELETE FROM exercises WHERE id = ?", (dup_id,))

                    db.conn.commit()
                    console.print(
                        f"\n[green]✓ Removed {len(exercise_duplicates)} duplicate exercises[/green]\n"
                    )
            else:
                console.print("  No duplicate exercises found\n")

            # Deduplicate topics
            console.print("[bold]Deduplicating Topics...[/bold]")
            topics = db.get_topics_by_course(course_code)
            topic_merges = []
            topic_skips = []

            def is_bilingual_match(name1, name2):
                """Check if two topic names are translations using LLM (ANY language pair)."""
                if not bilingual:
                    return False, None

                # Use SemanticMatcher's translation detection (LLM-based)
                if use_semantic and semantic_matcher and semantic_matcher.translation_detector:
                    try:
                        result = semantic_matcher.translation_detector.are_translations(
                            name1,
                            name2,
                            min_embedding_similarity=0.70,
                            use_language_detection=False,  # Skip for speed
                        )
                        if result.is_translation:
                            return (
                                True,
                                f"translation_detected (confidence: {result.confidence:.2f})",
                            )
                    except Exception as e:
                        console.print(f"[yellow]Translation detection failed: {e}[/yellow]")

                return False, None

            for i, topic1 in enumerate(topics):
                for topic2 in topics[i + 1 :]:
                    # First check bilingual translations
                    is_bilingual, bilingual_reason = is_bilingual_match(
                        topic1["name"], topic2["name"]
                    )
                    if is_bilingual:
                        topic_merges.append((topic1, topic2, 1.0, bilingual_reason))
                        continue

                    # Then check semantic/string similarity
                    if use_semantic and semantic_matcher:
                        result = semantic_matcher.should_merge(
                            topic1["name"], topic2["name"], threshold
                        )
                        if result.should_merge:
                            topic_merges.append(
                                (topic1, topic2, result.similarity_score, result.reason)
                            )
                        elif Config.SEMANTIC_LOG_NEAR_MISSES and result.similarity_score >= 0.80:
                            topic_skips.append(
                                (topic1, topic2, result.similarity_score, result.reason)
                            )
                    else:
                        similarity = SequenceMatcher(
                            None, topic1["name"].lower(), topic2["name"].lower()
                        ).ratio()
                        if similarity >= threshold:
                            topic_merges.append((topic1, topic2, similarity, "string_similarity"))

            if topic_merges:
                console.print(f"Found {len(topic_merges)} topic pairs to merge:\n")
                for t1, t2, sim, reason in topic_merges:
                    console.print(f"  • '{t1['name']}' ← '{t2['name']}'")
                    console.print(f"    Similarity: {sim:.2f}, Reason: {reason}")

                if not dry_run:
                    # Build merge mapping to resolve chains (e.g., A←B, B←C → both map to A)
                    merge_map = {}  # topic_id → canonical_topic_id

                    for t1, t2, sim, reason in topic_merges:
                        merge_map[t2["id"]] = t1["id"]

                    # Resolve chains: follow the chain to find ultimate target
                    def get_canonical(topic_id):
                        visited = set()
                        current = topic_id
                        while current in merge_map:
                            if current in visited:
                                # Circular reference, shouldn't happen but handle it
                                break
                            visited.add(current)
                            current = merge_map[current]
                        return current

                    # Update merge_map with canonical targets
                    for topic_id in list(merge_map.keys()):
                        merge_map[topic_id] = get_canonical(merge_map[topic_id])

                    # Apply merges: update all references to point to canonical topic
                    for source_id, target_id in merge_map.items():
                        # Update all exercises
                        db.conn.execute(
                            """
                            UPDATE exercises SET topic_id = ? WHERE topic_id = ?
                        """,
                            (target_id, source_id),
                        )

                        # Update all core loops
                        db.conn.execute(
                            """
                            UPDATE knowledge_items SET topic_id = ? WHERE topic_id = ?
                        """,
                            (target_id, source_id),
                        )

                        # Update topic_mastery
                        db.conn.execute(
                            """
                            UPDATE topic_mastery SET topic_id = ? WHERE topic_id = ?
                        """,
                            (target_id, source_id),
                        )

                        # Update quiz_sessions
                        db.conn.execute(
                            """
                            UPDATE quiz_sessions SET filter_topic_id = ? WHERE filter_topic_id = ?
                        """,
                            (target_id, source_id),
                        )

                        # Update theory_concepts
                        db.conn.execute(
                            """
                            UPDATE theory_concepts SET topic_id = ? WHERE topic_id = ?
                        """,
                            (target_id, source_id),
                        )

                    # Delete all merged topics (sources only, not targets)
                    for source_id in merge_map.keys():
                        db.conn.execute("DELETE FROM topics WHERE id = ?", (source_id,))

                    db.conn.commit()
                    console.print(
                        f"\n[green]✓ Merged {len(topic_merges)} duplicate topics[/green]\n"
                    )
            else:
                console.print("  No duplicate topics found\n")

            # Show near-misses if semantic matching is enabled
            if topic_skips:
                console.print(
                    f"\n[yellow]Skipped {len(topic_skips)} near-misses (high similarity but semantically different):[/yellow]"
                )
                for t1, t2, sim, reason in topic_skips:
                    console.print(f"  • '{t1['name']}' ≠ '{t2['name']}'")
                    console.print(f"    Similarity: {sim:.2f}, Reason: {reason}")
                console.print()

            # Deduplicate core loops
            console.print("[bold]Deduplicating Core Loops...[/bold]")
            knowledge_items = db.get_knowledge_items_by_course(course_code)
            loop_merges = []
            loop_skips = []

            for i, loop1 in enumerate(knowledge_items):
                for loop2 in knowledge_items[i + 1 :]:
                    # First check bilingual translations
                    is_bilingual, bilingual_reason = is_bilingual_match(
                        loop1["name"], loop2["name"]
                    )
                    if is_bilingual:
                        loop_merges.append((loop1, loop2, 1.0, bilingual_reason))
                        continue

                    # Then check semantic/string similarity
                    if use_semantic and semantic_matcher:
                        result = semantic_matcher.should_merge(
                            loop1["name"], loop2["name"], threshold
                        )
                        if result.should_merge:
                            loop_merges.append(
                                (loop1, loop2, result.similarity_score, result.reason)
                            )
                        elif Config.SEMANTIC_LOG_NEAR_MISSES and result.similarity_score >= 0.80:
                            loop_skips.append(
                                (loop1, loop2, result.similarity_score, result.reason)
                            )
                    else:
                        similarity = SequenceMatcher(
                            None, loop1["name"].lower(), loop2["name"].lower()
                        ).ratio()
                        if similarity >= threshold:
                            loop_merges.append((loop1, loop2, similarity, "string_similarity"))

            if loop_merges:
                console.print(f"Found {len(loop_merges)} core loop pairs to merge:\n")
                for l1, l2, sim, reason in loop_merges:
                    console.print(f"  • '{l1['name']}' ← '{l2['name']}'")
                    console.print(f"    Similarity: {sim:.2f}, Reason: {reason}")

                if not dry_run:
                    # Build merge mapping to resolve chains
                    loop_merge_map = {}  # loop_id → canonical_loop_id

                    for l1, l2, sim, reason in loop_merges:
                        loop_merge_map[l2["id"]] = l1["id"]

                    # Resolve chains
                    def get_canonical_loop(loop_id):
                        visited = set()
                        current = loop_id
                        while current in loop_merge_map:
                            if current in visited:
                                break
                            visited.add(current)
                            current = loop_merge_map[current]
                        return current

                    # Update merge_map with canonical targets
                    for loop_id in list(loop_merge_map.keys()):
                        loop_merge_map[loop_id] = get_canonical_loop(loop_merge_map[loop_id])

                    # Apply merges
                    for source_id, target_id in loop_merge_map.items():
                        # Delete duplicate entries from exercise_knowledge_items where exercise already has target
                        db.conn.execute(
                            """
                            DELETE FROM exercise_knowledge_items
                            WHERE knowledge_item_id = ?
                            AND exercise_id IN (
                                SELECT exercise_id FROM exercise_knowledge_items WHERE knowledge_item_id = ?
                            )
                        """,
                            (source_id, target_id),
                        )

                        # Update remaining exercise_knowledge_items entries
                        db.conn.execute(
                            """
                            UPDATE exercise_knowledge_items
                            SET knowledge_item_id = ?
                            WHERE knowledge_item_id = ?
                        """,
                            (target_id, source_id),
                        )

                        # Update legacy knowledge_item_id in exercises
                        db.conn.execute(
                            """
                            UPDATE exercises
                            SET knowledge_item_id = ?
                            WHERE knowledge_item_id = ?
                        """,
                            (target_id, source_id),
                        )

                    # Delete all merged core loops (sources only)
                    for source_id in loop_merge_map.keys():
                        db.conn.execute("DELETE FROM knowledge_items WHERE id = ?", (source_id,))

                    db.conn.commit()
                    console.print(
                        f"\n[green]✓ Merged {len(loop_merges)} duplicate core loops[/green]\n"
                    )
            else:
                console.print("  No duplicate core loops found\n")

            # Show near-misses if semantic matching is enabled
            if loop_skips:
                console.print(
                    f"\n[yellow]Skipped {len(loop_skips)} near-misses (high similarity but semantically different):[/yellow]"
                )
                for l1, l2, sim, reason in loop_skips:
                    console.print(f"  • '{l1['name']}' ≠ '{l2['name']}'")
                    console.print(f"    Similarity: {sim:.2f}, Reason: {reason}")
                console.print()

            # Clean up orphaned core loops if requested
            orphan_count = 0
            if clean_orphans:
                console.print("[bold]Cleaning Orphaned Core Loops...[/bold]")

                # Find core loops with no exercises
                cursor = db.conn.execute(
                    """
                    SELECT cl.id, cl.name, t.name as topic_name
                    FROM knowledge_items cl
                    JOIN topics t ON cl.topic_id = t.id
                    WHERE t.course_code = ?
                    AND cl.id NOT IN (
                        SELECT DISTINCT knowledge_item_id FROM exercises
                        WHERE knowledge_item_id IS NOT NULL
                    )
                    AND cl.id NOT IN (
                        SELECT DISTINCT knowledge_item_id FROM exercise_knowledge_items
                    )
                    ORDER BY cl.name
                """,
                    (course_code,),
                )

                orphans = cursor.fetchall()

                if orphans:
                    console.print(f"Found {len(orphans)} orphaned core loops:\n")
                    for loop_id, loop_name, topic_name in orphans:
                        console.print(f"  • {loop_name} (Topic: {topic_name[:40]}...)")

                    if not dry_run:
                        for loop_id, loop_name, topic_name in orphans:
                            db.conn.execute("DELETE FROM knowledge_items WHERE id = ?", (loop_id,))
                        db.conn.commit()
                        orphan_count = len(orphans)
                        console.print(
                            f"\n[green]✓ Deleted {orphan_count} orphaned core loops[/green]\n"
                        )
                    else:
                        console.print(f"\n[yellow](Dry run - orphans not deleted)[/yellow]\n")
                else:
                    console.print("  No orphaned core loops found\n")

            changes_made = topic_merges or loop_merges or (not dry_run and orphan_count > 0)
            if not dry_run and changes_made:
                console.print("[green]Deduplication complete![/green]\n")
            elif dry_run:
                console.print(
                    "[yellow]Dry run complete. Use without --dry-run to apply changes.[/yellow]\n"
                )
            else:
                console.print("[green]No changes needed![/green]\n")

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}\n")
        import traceback

        traceback.print_exc()
        raise click.Abort()


@cli.command(name="rate-limits")
@click.option("--provider", "-p", help="Show limits for specific provider (default: all)")
@click.option("--reset", is_flag=True, help="Reset rate limit tracking for provider")
def rate_limits(provider, reset):
    """View current rate limit usage for LLM API providers."""
    from models.llm_manager import LLMManager
    from rich.panel import Panel

    try:
        # Initialize LLM manager to access rate limiter
        llm = LLMManager(provider=Config.LLM_PROVIDER)

        console.print("\n[bold cyan]API Rate Limit Status[/bold cyan]\n")

        # Handle reset
        if reset:
            if provider:
                llm.rate_limiter.reset(provider)
                console.print(f"[green]✓ Reset rate limits for '{provider}'[/green]\n")
            else:
                llm.rate_limiter.reset_all()
                console.print(f"[green]✓ Reset rate limits for all providers[/green]\n")
            return

        # Get stats
        if provider:
            # Show single provider
            stats = llm.get_rate_limit_stats(provider)
            providers_to_show = {provider: stats}
        else:
            # Show all providers
            providers_to_show = llm.get_all_rate_limit_stats()

        # Display each provider
        for prov_name, stats in providers_to_show.items():
            if "error" in stats:
                console.print(f"[yellow]Provider '{prov_name}': {stats['error']}[/yellow]\n")
                continue

            if not stats.get("has_limits"):
                console.print(
                    f"[bold]{prov_name.title()}[/bold]: [green]No rate limits (local/unlimited)[/green]\n"
                )
                continue

            # Create provider panel
            req_stats = stats["requests"]
            token_stats = stats["tokens"]

            # Format request stats
            req_limit = req_stats["limit"]
            req_used = req_stats["used"]
            req_remaining = req_stats["remaining"]
            req_pct = req_stats["percentage"]

            # Format token stats
            token_limit = token_stats["limit"]
            token_used = token_stats["used"]
            token_remaining = token_stats["remaining"]
            token_pct = token_stats["percentage"]

            # Choose color based on usage
            def get_color(percentage):
                if percentage >= 90:
                    return "red"
                elif percentage >= 70:
                    return "yellow"
                else:
                    return "green"

            req_color = get_color(req_pct)
            token_color = get_color(token_pct)

            # Build panel content
            content_lines = []
            content_lines.append(f"[bold]Provider:[/bold] {prov_name}")
            content_lines.append(
                f"[bold]Current Provider:[/bold] {'✓ Active' if prov_name == Config.LLM_PROVIDER else '○ Inactive'}"
            )
            content_lines.append("")

            # Requests section
            if req_limit:
                content_lines.append(f"[bold]Requests (per minute):[/bold]")
                content_lines.append(
                    f"  Used: [{req_color}]{req_used}/{req_limit}[/{req_color}] ({req_pct}%)"
                )
                content_lines.append(f"  Remaining: {req_remaining}")

            # Tokens section
            if token_limit:
                content_lines.append("")
                content_lines.append(f"[bold]Tokens (per minute):[/bold]")
                content_lines.append(
                    f"  Used: [{token_color}]{token_used:,}/{token_limit:,}[/{token_color}] ({token_pct}%)"
                )
                content_lines.append(f"  Remaining: {token_remaining:,}")

            # Time info
            time_until_reset = stats.get("time_until_reset", 0)
            if time_until_reset > 0:
                content_lines.append("")
                content_lines.append(f"[dim]Time until reset: {time_until_reset:.1f}s[/dim]")

            content = "\n".join(content_lines)

            # Display panel
            border_color = (
                "red"
                if req_pct >= 90 or token_pct >= 90
                else "yellow"
                if req_pct >= 70 or token_pct >= 70
                else "green"
            )
            console.print(
                Panel(content, title=f"📊 {prov_name.title()}", border_style=border_color)
            )
            console.print()

        # Show helpful tips
        console.print("[dim]Tips:[/dim]")
        console.print(
            "[dim]  • Rate limiting is automatic - requests will wait if limits are exceeded[/dim]"
        )
        console.print("[dim]  • Use --provider <name> to see a specific provider[/dim]")
        console.print("[dim]  • Use --reset to clear rate limit tracking[/dim]")
        console.print(
            "[dim]  • Configure limits in .env: GROQ_RPM=30, ANTHROPIC_RPM=50, etc.[/dim]\n"
        )

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}\n")
        import traceback

        traceback.print_exc()
        raise click.Abort()


@cli.command(name="pattern-cache")
@click.option("--course", "-c", help="Course code (optional, defaults to all courses)")
@click.option(
    "--stats", "action", flag_value="stats", default=True, help="Show cache statistics (default)"
)
@click.option(
    "--build", "action", flag_value="build", help="Build cache from existing analyzed exercises"
)
@click.option("--clear", "action", flag_value="clear", help="Clear cache entries")
@click.option("--force", is_flag=True, default=False, help="Skip confirmation prompts")
def pattern_cache(course, action, force):
    """Manage procedure pattern cache for faster analysis.

    The procedure cache stores patterns from LLM analysis to avoid redundant
    API calls for similar exercises. This can significantly speed up analysis
    of courses with repetitive exercise patterns.

    Examples:
        examina pattern-cache              # Show cache stats (all courses)
        examina pattern-cache -c ARCH1     # Show stats for specific course
        examina pattern-cache --build      # Build cache from analyzed exercises
        examina pattern-cache --clear      # Clear all cache entries
        examina pattern-cache --clear -c ARCH1  # Clear cache for specific course
    """
    from rich.table import Table
    from core.procedure_cache import ProcedureCache

    try:
        with Database() as db:
            # Resolve course code if provided
            course_code = None
            if course:
                all_courses = db.get_all_courses()
                for c in all_courses:
                    if c["code"] == course or c["acronym"] == course:
                        course_code = c["code"]
                        break
                if not course_code:
                    console.print(f"[red]Course '{course}' not found.[/red]\n")
                    return

            if action == "stats":
                # Show cache statistics
                console.print("\n[bold cyan]Procedure Pattern Cache Statistics[/bold cyan]\n")

                # Get database-level stats
                db_stats = db.get_cache_stats(course_code=course_code, user_id=None)

                if course_code:
                    console.print(f"[bold]Course:[/bold] {course_code}\n")
                else:
                    console.print("[bold]Scope:[/bold] All courses\n")

                # Create stats table
                table = Table(title="Cache Overview")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="white")

                table.add_row("Total Patterns Cached", str(db_stats.get("total_entries", 0)))
                table.add_row("Total Cache Hits", str(db_stats.get("total_hits", 0)))

                # Add per-course breakdown if showing all
                if not course_code:
                    course_stats = db_stats.get("by_course", {})
                    if course_stats:
                        table.add_row("", "")  # Spacer
                        table.add_row("[bold]By Course[/bold]", "")
                        for cc, count in sorted(course_stats.items(), key=lambda x: -x[1]):
                            table.add_row(f"  {cc}", str(count))

                console.print(table)
                console.print()

                # Show configuration
                console.print("[bold]Configuration:[/bold]")
                console.print(
                    f"  • Embedding threshold: {Config.PROCEDURE_CACHE_EMBEDDING_THRESHOLD}"
                )
                console.print(
                    f"  • Text validation threshold: {Config.PROCEDURE_CACHE_TEXT_VALIDATION_THRESHOLD}"
                )
                console.print(f"  • Min confidence: {Config.PROCEDURE_CACHE_MIN_CONFIDENCE}")
                console.print(f"  • Cache enabled: {Config.PROCEDURE_CACHE_ENABLED}")
                console.print()

                # Tips
                console.print("[dim]Tips:[/dim]")
                console.print(
                    "[dim]  • Use --build to populate cache from existing analyzed exercises[/dim]"
                )
                console.print(
                    "[dim]  • Use --clear to reset cache (useful after major changes)[/dim]"
                )
                console.print(
                    "[dim]  • Configure thresholds via EXAMINA_PROCEDURE_CACHE_* env vars[/dim]"
                )
                console.print()

            elif action == "build":
                # Build cache from existing analyzed exercises
                console.print("\n[bold cyan]Building Procedure Pattern Cache[/bold cyan]\n")

                if course_code:
                    console.print(f"[bold]Course:[/bold] {course_code}\n")
                else:
                    console.print("[bold]Scope:[/bold] All courses\n")

                # SemanticMatcher removed - using LLM-based detect_synonyms() instead
                semantic_matcher = None
                console.print("   ✓ Using LLM-based synonym detection\n")

                # Initialize cache
                cache = ProcedureCache(db, semantic_matcher=None, user_id=None)
                cache.load_cache(course_code)

                # Get analyzed exercises with procedures
                console.print("📊 Scanning analyzed exercises...")

                # Query exercises with core loops (where procedures are stored)
                # Handle both schemas: junction table AND legacy knowledge_item_id column
                if course_code:
                    cursor = db.conn.execute(
                        """
                        SELECT e.id, e.text as exercise_text, e.source_pdf,
                               t.name as topic, e.difficulty, e.variations,
                               cl.procedure as procedures_json, cl.id as knowledge_item_id
                        FROM exercises e
                        JOIN exercise_knowledge_items ecl ON e.id = ecl.exercise_id
                        JOIN knowledge_items cl ON ecl.knowledge_item_id = cl.id
                        JOIN topics t ON cl.topic_id = t.id
                        WHERE e.course_code = ? AND cl.procedure IS NOT NULL
                        UNION
                        SELECT e.id, e.text as exercise_text, e.source_pdf,
                               t.name as topic, e.difficulty, e.variations,
                               cl.procedure as procedures_json, cl.id as knowledge_item_id
                        FROM exercises e
                        JOIN knowledge_items cl ON e.knowledge_item_id = cl.id
                        JOIN topics t ON cl.topic_id = t.id
                        WHERE e.course_code = ? AND e.knowledge_item_id IS NOT NULL AND cl.procedure IS NOT NULL
                    """,
                        (course_code, course_code),
                    )
                else:
                    cursor = db.conn.execute("""
                        SELECT e.id, e.text as exercise_text, e.source_pdf, e.course_code,
                               t.name as topic, e.difficulty, e.variations,
                               cl.procedure as procedures_json, cl.id as knowledge_item_id
                        FROM exercises e
                        JOIN exercise_knowledge_items ecl ON e.id = ecl.exercise_id
                        JOIN knowledge_items cl ON ecl.knowledge_item_id = cl.id
                        JOIN topics t ON cl.topic_id = t.id
                        WHERE cl.procedure IS NOT NULL
                        UNION
                        SELECT e.id, e.text as exercise_text, e.source_pdf, e.course_code,
                               t.name as topic, e.difficulty, e.variations,
                               cl.procedure as procedures_json, cl.id as knowledge_item_id
                        FROM exercises e
                        JOIN knowledge_items cl ON e.knowledge_item_id = cl.id
                        JOIN topics t ON cl.topic_id = t.id
                        WHERE e.knowledge_item_id IS NOT NULL AND cl.procedure IS NOT NULL
                    """)

                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]

                total = len(rows)
                added = 0
                skipped = 0

                if total == 0:
                    console.print("[yellow]No analyzed exercises found to cache.[/yellow]\n")
                    console.print("Run 'examina analyze' first to analyze exercises.\n")
                    return

                console.print(f"   Found {total} analyzed exercises\n")
                console.print("🔨 Building cache entries...")

                for row in rows:
                    row_dict = dict(zip(columns, row))
                    exercise_text = row_dict["exercise_text"]
                    topic = row_dict.get("topic")
                    difficulty = row_dict.get("difficulty")
                    ex_course_code = row_dict.get("course_code", course_code)

                    # Parse JSON fields (variations is stored as JSON string in exercises table)
                    try:
                        variations_raw = row_dict.get("variations")
                        variations = json.loads(variations_raw) if variations_raw else []

                        # Procedures are stored as JSON in knowledge_items.procedure
                        procedures_raw = row_dict.get("procedures_json")
                        procedures = json.loads(procedures_raw) if procedures_raw else []
                    except json.JSONDecodeError:
                        skipped += 1
                        continue

                    # Confidence defaults to 1.0 for existing analyzed exercises
                    confidence = 1.0

                    # Skip if no procedures
                    if not procedures:
                        skipped += 1
                        continue

                    # Try to add to cache (cache.add handles duplicates)
                    try:
                        cache.add(
                            exercise_text=exercise_text,
                            topic=topic,
                            difficulty=difficulty,
                            variations=variations,
                            procedures=procedures,
                            confidence=confidence,
                            course_code=ex_course_code,
                        )
                        added += 1
                    except Exception:
                        skipped += 1

                console.print(f"\n[green]✓ Cache build complete![/green]")
                console.print(f"  • Added: {added} new patterns")
                console.print(f"  • Skipped: {skipped} (duplicates/low confidence/no procedures)")
                console.print(f"  • Total in cache: {len(cache._entries)}\n")

            elif action == "clear":
                # Clear cache entries
                scope_msg = f"for course '{course_code}'" if course_code else "for ALL courses"

                if not force:
                    console.print(
                        f"\n[yellow]Warning:[/yellow] This will delete all cached procedure patterns {scope_msg}.\n"
                    )
                    if not click.confirm("Are you sure you want to continue?"):
                        console.print("[dim]Cancelled.[/dim]\n")
                        return

                console.print(f"\n[bold cyan]Clearing Procedure Pattern Cache[/bold cyan]\n")

                # Initialize cache and clear
                cache = ProcedureCache(db, semantic_matcher=None, user_id=None)
                cache.load_cache(course_code)

                entries_before = len(cache._entries)
                cache.clear(course_code)

                console.print(
                    f"[green]✓ Cleared {entries_before} cache entries {scope_msg}[/green]\n"
                )

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}\n")
        import traceback

        traceback.print_exc()
        raise click.Abort()


@cli.command()
@click.option("--course", "-c", required=True, help="Course code")
@click.option(
    "--dry-run",
    is_flag=True,
    default=True,
    help="Preview changes without updating database (default: true)",
)
@click.option(
    "--confidence-threshold",
    type=float,
    default=0.5,
    help="Minimum confidence to separate (0.0-1.0, default: 0.5)",
)
@click.option(
    "--provider",
    "-p",
    type=click.Choice(["anthropic", "groq", "ollama", "openai", "deepseek"]),
    help="LLM provider (default: from config)",
)
def separate_solutions(course, dry_run, confidence_threshold, provider):
    """Separate questions from solutions in exercises using LLM (works for any format/language)."""
    from core.solution_separator import process_course_solutions
    from models.llm_manager import LLMManager

    console.print(f"\n[bold cyan]Separating Solutions for {course}...[/bold cyan]\n")

    if dry_run:
        console.print("[yellow]DRY RUN MODE - No changes will be made[/yellow]\n")

    try:
        # Find course
        with Database() as db:
            all_courses = db.get_all_courses()
            found_course = None
            for c in all_courses:
                if c["code"] == course or c["acronym"] == course:
                    found_course = c
                    break

            if not found_course:
                console.print(f"[red]Course '{course}' not found.[/red]\n")
                return

            course_code = found_course["code"]

        # Initialize LLM
        console.print("🤖 Initializing AI solution separator...")
        console.print("[dim]Using LLM-based detection (works for any format/language)[/dim]\n")
        llm = LLMManager(provider=provider or Config.LLM_PROVIDER)

        # Process course
        console.print(f"📝 Analyzing exercises for {course_code}...\n")
        stats = process_course_solutions(course_code=course_code, llm_manager=llm, dry_run=dry_run)

        # Display results
        console.print("\n[bold]Results:[/bold]")
        console.print(f"  Total exercises: {stats['total_exercises']}")
        console.print(f"  Exercises with solutions: {stats['has_solution']}")
        console.print(f"  Successfully separated: {stats['separated']}")
        console.print(f"  High confidence (≥0.8): {stats['high_confidence']}")
        console.print(f"  Failed: {stats['failed']}")

        if stats["separated"] > 0:
            success_rate = (
                (stats["separated"] / stats["has_solution"] * 100)
                if stats["has_solution"] > 0
                else 0
            )
            console.print(f"\n[green]✓ Separation success rate: {success_rate:.1f}%[/green]")

        if dry_run and stats["separated"] > 0:
            console.print(f"\n[yellow]Run without --dry-run to apply changes[/yellow]")

        console.print()

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}\n")
        import traceback

        traceback.print_exc()
        raise click.Abort()


@cli.command()
@click.option("--course", "-c", required=True, help="Course code")
@click.option(
    "--dry-run", is_flag=True, default=False, help="Preview changes without updating database"
)
@click.option("--force", is_flag=True, default=False, help="Re-detect language even if already set")
def detect_languages(course, dry_run, force):
    """DEPRECATED: Language detection is no longer needed - names are always extracted in English."""
    console.print(f"\n[bold yellow]DEPRECATED[/bold yellow]\n")
    console.print("[yellow]Language detection is no longer needed.[/yellow]")
    console.print("[yellow]Names are now always extracted in English during analysis.[/yellow]\n")
    return

    if dry_run:
        console.print("[yellow]DRY RUN MODE - No changes will be made[/yellow]\n")

    try:
        # Find course
        with Database() as db:
            all_courses = db.get_all_courses()
            found_course = None
            for c in all_courses:
                if c["code"] == course or c["acronym"] == course:
                    found_course = c
                    break

            if not found_course:
                console.print(f"[red]Course '{course}' not found.[/red]\n")
                return

            course_code = found_course["code"]

        # Initialize LLM and detector
        console.print("🤖 Initializing language detector...")
        console.print("[dim]Using LLM-based detection (works for ANY language)[/dim]\n")
        llm = LLMManager(provider=Config.LLM_PROVIDER)
        detector = TranslationDetector(llm_manager=llm)

        with Database() as db:
            # Detect for core loops
            if force:
                query = """
                    SELECT id, name, language FROM knowledge_items
                    WHERE topic_id IN (
                        SELECT id FROM topics WHERE course_code = ?
                    )
                """
            else:
                query = """
                    SELECT id, name, language FROM knowledge_items
                    WHERE topic_id IN (
                        SELECT id FROM topics WHERE course_code = ?
                    ) AND language IS NULL
                """

            cursor = db.conn.execute(query, (course_code,))
            loops = cursor.fetchall()

            console.print(f"📝 Detecting languages for {len(loops)} core loops...\n")

            if loops:
                # Create results table
                table = Table(title="Core Loop Languages", show_header=True)
                table.add_column("Core Loop", style="cyan", width=50)
                table.add_column("Language", style="green", width=15)
                table.add_column("Action", style="yellow", width=15)

                updated_count = 0
                for loop_id, name, current_lang in loops:
                    # Detect language
                    lang_info = detector.detect_language_with_iso(name)

                    action = "detect"
                    if current_lang and not force:
                        action = "skip"
                    elif current_lang:
                        action = "re-detect"

                    display_lang = f"{lang_info.name}"
                    if lang_info.code:
                        display_lang += f" ({lang_info.code})"

                    # Truncate name for display
                    display_name = name[:47] + "..." if len(name) > 50 else name

                    table.add_row(display_name, display_lang, action)

                    # Update database
                    if not dry_run and (force or not current_lang):
                        db.conn.execute(
                            """
                            UPDATE knowledge_items SET language = ? WHERE id = ?
                        """,
                            (lang_info.name, loop_id),
                        )
                        updated_count += 1

                console.print(table)
                console.print()

                if not dry_run:
                    console.print(f"[green]✓ Updated {updated_count} core loops[/green]\n")
                else:
                    console.print(
                        f"[yellow]Would update {len(loops)} core loops (use without --dry-run to apply)[/yellow]\n"
                    )

            # Detect for topics
            if force:
                query = """
                    SELECT id, name, language FROM topics
                    WHERE course_code = ?
                """
            else:
                query = """
                    SELECT id, name, language FROM topics
                    WHERE course_code = ? AND language IS NULL
                """

            cursor = db.conn.execute(query, (course_code,))
            topics = cursor.fetchall()

            if topics:
                console.print(f"📚 Detecting languages for {len(topics)} topics...\n")

                # Create results table
                table = Table(title="Topic Languages", show_header=True)
                table.add_column("Topic", style="cyan", width=50)
                table.add_column("Language", style="green", width=15)
                table.add_column("Action", style="yellow", width=15)

                updated_count = 0
                for topic_id, name, current_lang in topics:
                    # Detect language
                    lang_info = detector.detect_language_with_iso(name)

                    action = "detect"
                    if current_lang and not force:
                        action = "skip"
                    elif current_lang:
                        action = "re-detect"

                    display_lang = f"{lang_info.name}"
                    if lang_info.code:
                        display_lang += f" ({lang_info.code})"

                    # Truncate name for display
                    display_name = name[:47] + "..." if len(name) > 50 else name

                    table.add_row(display_name, display_lang, action)

                    # Update database
                    if not dry_run and (force or not current_lang):
                        db.conn.execute(
                            """
                            UPDATE topics SET language = ? WHERE id = ?
                        """,
                            (lang_info.name, topic_id),
                        )
                        updated_count += 1

                console.print(table)
                console.print()

                if not dry_run:
                    console.print(f"[green]✓ Updated {updated_count} topics[/green]\n")
                else:
                    console.print(
                        f"[yellow]Would update {len(topics)} topics (use without --dry-run to apply)[/yellow]\n"
                    )

            # Get cache stats
            stats = detector.get_cache_stats()
            console.print(
                f"[dim]Cache: {stats['language_cache_size']} languages, {stats['translation_cache_size']} translations[/dim]"
            )

            if not dry_run:
                db.conn.commit()

        console.print("\n[bold green]✓ Language detection complete![/bold green]\n")

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}\n")
        import traceback

        traceback.print_exc()
        raise click.Abort()


@cli.command(name="concept-graph")
@click.option("--course", "-c", required=True, help="Course code")
@click.option(
    "--format",
    type=click.Choice(["ascii", "mermaid", "json"]),
    default="ascii",
    help="Output format (default: ascii)",
)
@click.option("--export", type=click.Path(), help="Export to file")
@click.option("--concept", help="Show learning path to specific concept")
@click.option(
    "--provider",
    type=click.Choice(["anthropic", "openai", "groq", "ollama", "deepseek"]),
    default=Config.LLM_PROVIDER,
    help=f"LLM provider (default: {Config.LLM_PROVIDER})",
)
def concept_graph(course, format, export, concept, provider):
    """Visualize theory concept dependencies and learning order."""
    try:
        from core.concept_graph import ConceptGraphBuilder
        from core.concept_visualizer import ConceptVisualizer
        from models.llm_manager import LLMManager

        console.print(f"\n[bold cyan]Building Concept Graph for {course}[/bold cyan]\n")

        # Initialize LLM and builder
        llm = LLMManager(provider=provider)
        builder = ConceptGraphBuilder(llm_manager=llm)

        # Build graph
        console.print("📊 Analyzing theory exercises...")
        graph = builder.build_from_course(course)

        if not graph.concepts:
            console.print("[yellow]No theory concepts found in this course.[/yellow]")
            console.print(
                "Theory concepts are extracted from exercises marked as 'theory', 'proof', or 'hybrid'."
            )
            console.print(
                "\nTip: Run 'examina analyze --course {} --reanalyze' to detect theory exercises.".format(
                    course
                )
            )
            return

        console.print(
            f"[green]✓ Found {len(graph.concepts)} concepts with {len(graph.edges)} dependencies[/green]\n"
        )

        # Check for cycles
        cycles = graph.detect_cycles()
        if cycles:
            console.print("[red]⚠ Warning: Cycles detected in dependency graph![/red]")
            for cycle in cycles:
                concept_names = [graph.concepts[cid].name for cid in cycle]
                console.print(f"  Cycle: {' → '.join(concept_names)}")
            console.print()

        # Visualize
        visualizer = ConceptVisualizer()

        if concept:
            # Show learning path to specific concept
            output = visualizer.render_learning_path(graph, concept)
        elif format == "ascii":
            output = visualizer.render_ascii(graph)
        elif format == "mermaid":
            output = visualizer.render_mermaid(graph)
        else:  # json
            output = visualizer.export_json(graph)

        if export:
            with open(export, "w") as f:
                f.write(output)
            console.print(f"[green]✓ Exported to {export}[/green]")
        else:
            console.print(output)

        # Show summary
        if not concept:
            console.print("\n[bold]Learning Order Summary:[/bold]")
            learning_order = graph.topological_sort()
            if learning_order:
                foundation_concepts = [
                    cid for cid in learning_order if not graph.get_prerequisites(cid)
                ]
                console.print(f"  • Foundation concepts (start here): {len(foundation_concepts)}")
                console.print(
                    f"  • Advanced concepts (require prerequisites): {len(learning_order) - len(foundation_concepts)}"
                )

                if foundation_concepts:
                    console.print("\n[bold]Recommended Starting Points:[/bold]")
                    for cid in foundation_concepts[:5]:  # Show first 5
                        c = graph.concepts[cid]
                        console.print(f"  • {c.name} ({c.exercise_count} exercises)")
                    if len(foundation_concepts) > 5:
                        console.print(f"  ... and {len(foundation_concepts) - 5} more")

        console.print()

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}\n")
        import traceback

        traceback.print_exc()
        raise click.Abort()


if __name__ == "__main__":
    cli()
