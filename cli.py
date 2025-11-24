#!/usr/bin/env python3
"""
Examina - AI-powered exam tutor system.
CLI interface for managing courses, ingesting exams, and studying.
"""

import click
import json
from rich.console import Console
from rich.table import Table
from rich import print as rprint
from pathlib import Path

from config import Config
from storage.database import Database
from storage.file_manager import FileManager
from storage.vector_store import VectorStore
from core.pdf_processor import PDFProcessor
from core.exercise_splitter import ExerciseSplitter
from core.analyzer import ExerciseAnalyzer, AnalysisResult
from models.llm_manager import LLMManager
from study_context import study_plan

console = Console()


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
                        degree_program=program
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
@click.option('--degree', type=click.Choice(['bachelor', 'master', 'all']), default='all',
              help='Filter by degree level')
def courses(degree):
    """List all available courses."""
    try:
        with Database() as db:
            all_courses = db.get_all_courses()

        if not all_courses:
            console.print("\n[yellow]No courses found. Run 'examina init' first.[/yellow]\n")
            return

        # Filter by degree if specified
        if degree != 'all':
            all_courses = [c for c in all_courses if c['degree_level'] == degree]

        # Create table
        table = Table(title=f"\n📚 Available Courses ({degree.title()})" if degree != 'all' else "\n📚 Available Courses")
        table.add_column("Code", style="cyan", no_wrap=True)
        table.add_column("Acronym", style="magenta")
        table.add_column("Name", style="white")
        table.add_column("Level", style="green")

        for course in all_courses:
            table.add_row(
                course['code'],
                course['acronym'] or '',
                course['name'],
                course['degree_level'].title()
            )

        console.print(table)
        console.print(f"\nTotal: {len(all_courses)} courses\n")

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}\n")
        raise click.Abort()


@cli.command()
@click.option('--course', '-c', required=True, help='Course code (e.g., B006802 or ADE)')
def info(course):
    """Show detailed information about a course."""
    try:
        with Database() as db:
            # Try to find course by code or acronym
            all_courses = db.get_all_courses()
            found_course = None

            for c in all_courses:
                if c['code'] == course or c['acronym'] == course:
                    found_course = c
                    break

            if not found_course:
                console.print(f"\n[red]Course '{course}' not found.[/red]\n")
                console.print("Use 'examina courses' to see available courses.\n")
                return

            # Get topics and stats
            topics = db.get_topics_by_course(found_course['code'])
            exercises = db.get_exercises_by_course(found_course['code'])

            # Display info
            console.print(f"\n[bold cyan]{found_course['name']}[/bold cyan]")
            if found_course['original_name']:
                console.print(f"[dim]{found_course['original_name']}[/dim]")

            console.print(f"\nCode: {found_course['code']}")
            console.print(f"Acronym: {found_course['acronym']}")
            console.print(f"Level: {found_course['degree_level'].title()} ({found_course['degree_program']})")

            console.print(f"\n[bold]Status:[/bold]")
            console.print(f"  Topics discovered: {len(topics)}")
            console.print(f"  Exercises ingested: {len(exercises)}")

            if topics:
                console.print(f"\n[bold]Topics:[/bold]")
                for topic in topics:
                    core_loops = db.get_core_loops_by_topic(topic['id'])
                    console.print(f"  • {topic['name']} ({len(core_loops)} core loops)")

            # Show multi-procedure statistics
            multi_proc_exercises = db.get_exercises_with_multiple_procedures(found_course['code'])
            if multi_proc_exercises:
                console.print(f"\n[bold]Multi-Procedure Exercises:[/bold]")
                console.print(f"  {len(multi_proc_exercises)}/{len(exercises)} exercises cover multiple procedures")

                # Show top 3 examples
                console.print(f"\n[bold]Top Examples:[/bold]")
                for ex in multi_proc_exercises[:3]:
                    # Get all core loops for this exercise
                    core_loops = db.get_exercise_core_loops(ex['id'])
                    console.print(f"  • Exercise {ex['exercise_number'] or ex['id'][:8]}: {ex['core_loop_count']} procedures")
                    for cl in core_loops[:3]:  # Show first 3 procedures
                        step_info = f" (point {cl['step_number']})" if cl['step_number'] else ""
                        console.print(f"    - {cl['name']}{step_info}")
                    if ex['core_loop_count'] > 3:
                        console.print(f"    ... and {ex['core_loop_count'] - 3} more")

            console.print()

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}\n")
        raise click.Abort()


@cli.command()
@click.option('--course', '-c', required=True, help='Course code (e.g., B006802 or ADE)')
@click.option('--tag', '-t', help='Search by procedure tag (design, transformation, etc.)')
@click.option('--text', help='Search by text in exercise content')
@click.option('--multi-only', is_flag=True, help='Only show multi-procedure exercises')
@click.option('--limit', '-l', type=int, default=20, help='Maximum number of results (default: 20)')
def search(course, tag, text, multi_only, limit):
    """Search exercises by tags or content."""
    try:
        with Database() as db:
            # Find course by code or acronym
            all_courses = db.get_all_courses()
            found_course = None
            for c in all_courses:
                if c['code'] == course or c['acronym'] == course:
                    found_course = c
                    break

            if not found_course:
                console.print(f"\n[red]Course '{course}' not found.[/red]\n")
                console.print("Use 'examina courses' to see available courses.\n")
                return

            course_code = found_course['code']

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
                multi_proc_ids = {ex['id'] for ex in multi_proc_exercises}
                exercises = [ex for ex in exercises if ex['id'] in multi_proc_ids]
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
                core_loops = db.get_exercise_core_loops(ex['id'])

                # Display exercise header
                exercise_id = ex.get('exercise_number') or ex.get('source_pdf', 'Unknown')
                console.print(f"[cyan]Exercise: {exercise_id}[/cyan]")

                # Display procedures
                if len(core_loops) > 1:
                    console.print(f"  [yellow]Procedures ({len(core_loops)}):[/yellow]")
                    for i, cl in enumerate(core_loops, 1):
                        step_info = f" (point {cl['step_number']})" if cl['step_number'] else ""
                        console.print(f"    {i}. {cl['name']}{step_info}")
                elif len(core_loops) == 1:
                    console.print(f"  Core Loop: {core_loops[0]['name']}")
                else:
                    console.print(f"  [dim]No core loops assigned[/dim]")

                # Display tags if present
                if ex.get('tags'):
                    tags = json.loads(ex['tags']) if isinstance(ex['tags'], str) else ex['tags']
                    console.print(f"  Tags: {', '.join(tags)}")

                # Display difficulty
                if ex.get('difficulty'):
                    console.print(f"  Difficulty: {ex['difficulty']}")

                # Display source
                if ex.get('source_pdf'):
                    page_info = f" (page {ex['page_number']})" if ex.get('page_number') else ""
                    console.print(f"  [dim]Source: {ex['source_pdf']}{page_info}[/dim]")

                console.print()

            if len(exercises) == limit:
                console.print(f"[dim]Showing first {limit} results. Use --limit to see more.[/dim]\n")

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}\n")
        import traceback
        traceback.print_exc()
        raise click.Abort()


@cli.command()
@click.option('--course', '-c', required=True, help='Course code (e.g., B006802 or ADE)')
@click.option('--zip', '-z', 'zip_file', required=True, type=click.Path(exists=True),
              help='Path to ZIP file containing exam PDFs')
def ingest(course, zip_file):
    """Ingest exam PDFs for a course."""
    from tqdm import tqdm

    console.print(f"\n[bold cyan]Ingesting exams for {course}...[/bold cyan]\n")

    try:
        # Find course by code or acronym
        with Database() as db:
            all_courses = db.get_all_courses()
            found_course = None
            for c in all_courses:
                if c['code'] == course or c['acronym'] == course:
                    found_course = c
                    break

            if not found_course:
                console.print(f"[red]Course '{course}' not found.[/red]\n")
                console.print("Use 'examina courses' to see available courses.\n")
                return

            course_code = found_course['code']
            console.print(f"Course: {found_course['name']} ({found_course['acronym']})\n")

        # Initialize components
        file_mgr = FileManager()
        pdf_processor = PDFProcessor()
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
        processed_pdfs = 0

        for pdf_path in pdf_files:
            console.print(f"📄 Processing {pdf_path.name}...")

            # Check if scanned PDF
            if pdf_processor.is_scanned_pdf(pdf_path):
                console.print("   [yellow]⚠️  Scanned PDF detected - OCR not yet implemented[/yellow]")
                console.print("   [dim]Skipping...[/dim]\n")
                continue

            try:
                # Extract content
                pdf_content = pdf_processor.process_pdf(pdf_path)
                console.print(f"   ✓ Extracted {pdf_content.total_pages} pages")

                # Split into exercises
                exercises = exercise_splitter.split_pdf_content(pdf_content, course_code)

                # Filter valid exercises
                valid_exercises = [ex for ex in exercises if exercise_splitter.validate_exercise(ex)]
                console.print(f"   ✓ Found {len(valid_exercises)} exercise(s)")

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
                            'id': exercise.id,
                            'course_code': course_code,
                            'topic_id': None,  # Will be filled in Phase 3 (AI analysis)
                            'core_loop_id': None,  # Will be filled in Phase 3
                            'source_pdf': pdf_path.name,
                            'page_number': exercise.page_number,
                            'exercise_number': exercise.exercise_number,
                            'text': cleaned_text,
                            'has_images': exercise.has_images,
                            'image_paths': image_paths if image_paths else None,
                            'latex_content': exercise.latex_content,
                            'difficulty': None,  # Will be analyzed in Phase 3
                            'variations': None,
                            'solution': None,
                            'analyzed': False,
                            'analysis_metadata': None
                        }

                        db.add_exercise(exercise_data)

                    db.conn.commit()

                total_exercises += len(valid_exercises)
                processed_pdfs += 1
                console.print(f"   ✓ Stored in database\n")

            except Exception as e:
                console.print(f"   [red]Error: {e}[/red]\n")
                continue

        # Summary
        console.print("[bold green]✨ Ingestion complete![/bold green]\n")
        console.print(f"Processed: {processed_pdfs} PDF(s)")
        console.print(f"Extracted: {total_exercises} exercise(s)")
        console.print(f"\nNext steps:")
        console.print(f"  • examina info --course {course} - View course status")
        console.print(f"  • Phase 3: AI analysis to discover topics and core loops\n")

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}\n")
        import traceback
        traceback.print_exc()
        raise click.Abort()


@cli.command()
@click.option('--course', '-c', required=True, help='Course code (e.g., B006802 or ADE)')
@click.option('--limit', '-l', type=int, help='Limit number of exercises to analyze (for testing)')
@click.option('--provider', '-p', type=click.Choice(['ollama', 'groq', 'anthropic']), default='anthropic',
              help='LLM provider (default: anthropic)')
@click.option('--lang', type=click.Choice(['en', 'it']), default='en',
              help='Output language for analysis (default: en)')
@click.option('--force', '-f', is_flag=True, help='Force re-analysis of all exercises (ignore existing analysis)')
@click.option('--parallel/--sequential', default=True,
              help='Use parallel batch processing for better performance (default: parallel)')
@click.option('--batch-size', '-b', type=int, default=None,
              help=f'Batch size for parallel processing (default: {Config.BATCH_SIZE})')
def analyze(course, limit, provider, lang, force, parallel, batch_size):
    """Analyze exercises with AI to discover topics and core loops."""
    console.print(f"\n[bold cyan]Analyzing exercises for {course}...[/bold cyan]\n")

    try:
        # Find course
        with Database() as db:
            all_courses = db.get_all_courses()
            found_course = None
            for c in all_courses:
                if c['code'] == course or c['acronym'] == course:
                    found_course = c
                    break

            if not found_course:
                console.print(f"[red]Course '{course}' not found.[/red]\n")
                return

            course_code = found_course['code']
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
                db.conn.execute("UPDATE exercises SET analyzed = 0 WHERE course_code = ?", (course_code,))
                db.conn.commit()
            elif remaining_count == 0:
                console.print("[green]All exercises already analyzed! Use --force to re-analyze.[/green]\n")
                return
            else:
                if analyzed_count > 0:
                    console.print(f"[cyan]Resuming analysis from checkpoint ({remaining_count} exercises remaining)...[/cyan]\n")
                exercises = all_exercises  # Need all for proper merging context

        # Initialize components
        console.print(f"🤖 Initializing AI components (provider: {provider}, language: {lang})...")
        llm = LLMManager(provider=provider)
        analyzer = ExerciseAnalyzer(llm, language=lang)

        # For embeddings, we still need Ollama (Groq/Anthropic don't provide embeddings)
        embed_llm = LLMManager(provider="ollama") if provider in ["groq", "anthropic"] else llm
        vector_store = VectorStore(llm_manager=embed_llm)

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
                console.print(f"[yellow]Get your free API key at: https://console.groq.com[/yellow]")
                console.print(f"[yellow]Then set it: export GROQ_API_KEY=your_key_here[/yellow]\n")
                return
            console.print(f"   ✓ API key found\n")
        elif provider == "anthropic":
            console.print(f"   Using Anthropic API with {llm.primary_model}")
            if not Config.ANTHROPIC_API_KEY:
                console.print(f"[red]ANTHROPIC_API_KEY not set![/red]")
                console.print(f"[yellow]Get your API key at: https://console.anthropic.com[/yellow]")
                console.print(f"[yellow]Then set it: export ANTHROPIC_API_KEY=your_key_here[/yellow]\n")
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

        discovery_result = analyzer.discover_topics_and_core_loops(
            course_code,
            batch_size=batch_size or Config.BATCH_SIZE,
            skip_analyzed=skip_analyzed,
            use_parallel=parallel
        )

        # Show progress summary
        if skip_analyzed:
            newly_analyzed = discovery_result['merged_count']
            console.print(f"✓ Analyzed {newly_analyzed} new exercises (skipped {analyzed_count} already analyzed)")
            console.print(f"  Total progress: {analyzed_count + newly_analyzed}/{total_count} exercises\n")
        else:
            console.print(f"✓ Merged {discovery_result['original_count']} fragments → {discovery_result['merged_count']} exercises\n")

        # Display results
        topics = discovery_result['topics']
        core_loops = discovery_result['core_loops']

        if topics:
            console.print("[bold]📚 Discovered Topics:[/bold]")
            for topic_name, topic_data in topics.items():
                console.print(f"  • {topic_name} ({topic_data['exercise_count']} exercises, {len(topic_data['core_loops'])} core loops)")

        if core_loops:
            console.print(f"\n[bold]🔄 Discovered Core Loops:[/bold]")
            for loop_id, loop_data in core_loops.items():
                console.print(f"  • {loop_data['name']} ({loop_data['exercise_count']} exercises)")
                if loop_data['procedure']:
                    console.print(f"    [dim]Steps: {len(loop_data['procedure'])}[/dim]")

        # Store in database
        console.print(f"\n💾 Storing analysis results...")
        with Database() as db:
            # Get topic name mapping from analyzer (for deduplication)
            topic_name_mapping = getattr(analyzer, 'topic_name_mapping', {})

            # Store topics
            for topic_name in topics.keys():
                topic_id = db.add_topic(course_code, topic_name)

            # Store core loops
            for loop_id, loop_data in core_loops.items():
                # Get topic_id
                topic_name = loop_data.get('topic')
                if topic_name:
                    # Map topic name to canonical name (if deduplicated)
                    canonical_topic_name = topic_name
                    if canonical_topic_name in topic_name_mapping:
                        canonical_topic_name = topic_name_mapping[canonical_topic_name]

                    topic_rows = db.get_topics_by_course(course_code)
                    topic_id = next((t['id'] for t in topic_rows if t['name'] == canonical_topic_name), None)

                    if topic_id:
                        db.add_core_loop(
                            loop_id=loop_id,
                            topic_id=topic_id,
                            name=loop_data['name'],
                            procedure=loop_data['procedure'],
                            description=None
                        )

            # Get core loop ID mapping from analyzer (for deduplication)
            core_loop_id_mapping = getattr(analyzer, 'core_loop_id_mapping', {})

            # Update exercises with analysis
            for merged_ex in discovery_result['merged_exercises']:
                # Update first exercise in merged group
                first_id = merged_ex['merged_from'][0]

                # Check if this exercise was skipped due to low confidence
                if merged_ex.get('low_confidence_skipped'):
                    db.conn.execute("""
                        UPDATE exercises
                        SET analyzed = 1, low_confidence_skipped = 1
                        WHERE id = ?
                    """, (first_id,))
                    continue

                analysis = merged_ex.get('analysis')
                if analysis and analysis.topic:
                    # Map topic name to canonical name (if deduplicated)
                    canonical_topic_name = analysis.topic
                    if canonical_topic_name in topic_name_mapping:
                        canonical_topic_name = topic_name_mapping[canonical_topic_name]

                    # Get topic_id
                    topic_rows = db.get_topics_by_course(course_code)
                    topic_id = next((t['id'] for t in topic_rows if t['name'] == canonical_topic_name), None)

                    # Get primary core_loop_id (first procedure) for backward compatibility
                    primary_core_loop_id = analysis.core_loop_id
                    if primary_core_loop_id and primary_core_loop_id in core_loop_id_mapping:
                        primary_core_loop_id = core_loop_id_mapping[primary_core_loop_id]

                    # Only update if primary_core_loop_id exists in deduplicated core_loops OR database
                    if primary_core_loop_id and primary_core_loop_id not in core_loops:
                        # Check if it exists in database (may have been deduplicated to existing DB entry)
                        if not db.get_core_loop(primary_core_loop_id):
                            print(f"[DEBUG] Skipping exercise {first_id[:20]}... - core_loop_id '{primary_core_loop_id}' not found in deduplicated core_loops or database")
                            primary_core_loop_id = None

                    # Collect tags for flexible search
                    tags = []

                    # Process ALL procedures - link to junction table
                    if analysis.procedures:
                        for procedure_info in analysis.procedures:
                            proc_core_loop_id = AnalysisResult._normalize_core_loop_id(procedure_info.name)

                            # Map to canonical ID if deduplicated
                            if proc_core_loop_id and proc_core_loop_id in core_loop_id_mapping:
                                proc_core_loop_id = core_loop_id_mapping[proc_core_loop_id]

                            # Link exercise to core loop via junction table (check both new loops and DB)
                            if proc_core_loop_id and (proc_core_loop_id in core_loops or db.get_core_loop(proc_core_loop_id)):
                                db.link_exercise_to_core_loop(
                                    exercise_id=first_id,
                                    core_loop_id=proc_core_loop_id,
                                    step_number=procedure_info.point_number
                                )

                                # Collect tags
                                tags.append(procedure_info.type)
                                if procedure_info.transformation:
                                    src = procedure_info.transformation.get('source_format', '').lower().replace(' ', '_')
                                    tgt = procedure_info.transformation.get('target_format', '').lower().replace(' ', '_')
                                    tags.append(f"transform_{src}_to_{tgt}")

                    # Update exercise with primary core loop and metadata
                    db.update_exercise_analysis(
                        exercise_id=first_id,
                        topic_id=topic_id,
                        core_loop_id=primary_core_loop_id,
                        difficulty=analysis.difficulty,
                        variations=analysis.variations,
                        analyzed=True
                    )

                    # Update tags
                    if tags:
                        db.update_exercise_tags(first_id, list(set(tags)))

            db.conn.commit()
            console.print("   ✓ Stored in database\n")

        # Build vector store
        console.print("🧠 Building vector embeddings for RAG...")
        vector_store.add_exercises_batch(course_code, discovery_result['merged_exercises'])

        # Add core loops to vector store
        for loop_id, loop_data in core_loops.items():
            vector_store.add_core_loop(
                course_code=course_code,
                core_loop_id=loop_id,
                name=loop_data['name'],
                description=loop_data.get('description', ''),
                procedure=loop_data['procedure'],
                example_exercises=loop_data['exercises']
            )

        stats = vector_store.get_collection_stats(course_code)
        console.print(f"   ✓ {stats.get('exercises_count', 0)} exercise embeddings")
        console.print(f"   ✓ {stats.get('procedures_count', 0)} procedure embeddings\n")

        # Show cache statistics
        cache_stats = llm.get_cache_stats()
        if cache_stats['total_requests'] > 0:
            console.print("📊 Cache Statistics:")
            console.print(f"   Cache hits: {cache_stats['cache_hits']}")
            console.print(f"   Cache misses: {cache_stats['cache_misses']}")
            console.print(f"   Hit rate: {cache_stats['hit_rate_percent']}%")
            if cache_stats['cache_hits'] > 0:
                console.print(f"   [green]💰 Saved ~{cache_stats['cache_hits']} API calls![/green]\n")
            else:
                console.print(f"   [dim]Run analyze again to see cache benefits[/dim]\n")

        # Summary
        console.print("[bold green]✨ Analysis complete![/bold green]\n")
        console.print(f"Topics: {len(topics)}")
        console.print(f"Core loops: {len(core_loops)}")
        console.print(f"Exercises: {discovery_result['merged_count']}\n")
        console.print(f"Next steps:")
        console.print(f"  • examina info --course {course} - View updated course info")
        console.print(f"  • examina learn --course {course} - Start learning (Phase 4)\n")

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}\n")
        import traceback
        traceback.print_exc()
        raise click.Abort()


@cli.command(name='split-topics')
@click.option('--course', '-c', required=True, help='Course code')
@click.option('--provider', type=click.Choice(['anthropic', 'openai', 'groq', 'ollama']),
              default=Config.LLM_PROVIDER, help=f'LLM provider (default: {Config.LLM_PROVIDER})')
@click.option('--lang', type=click.Choice(['en', 'it']), default=Config.DEFAULT_LANGUAGE,
              help=f'Output language (default: {Config.DEFAULT_LANGUAGE})')
@click.option('--dry-run', is_flag=True,
              help='Preview splits without applying changes')
@click.option('--force', is_flag=True,
              help='Skip confirmation prompts')
@click.option('--delete-old', is_flag=True,
              help='Delete old topic if empty after split')
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
                console.print(f"[green]✓ No generic topics found! All topics are sufficiently specific.[/green]\n")
                return

            console.print(f"\n[yellow]Found {len(generic_topics)} generic topic(s):[/yellow]\n")
            for topic_info in generic_topics:
                console.print(f"  • {topic_info['name']}")
                console.print(f"    - Core loops: {topic_info['core_loop_count']}")
                console.print(f"    - Reason: {topic_info['reason']}\n")

            if dry_run:
                console.print("[yellow]Dry run mode: showing preview only, no changes will be made[/yellow]\n")

            # Process each generic topic
            for topic_info in generic_topics:
                console.print(f"\n[bold]Processing topic: {topic_info['name']}[/bold]")

                # Get core loops for this topic
                core_loops = db.get_core_loops_by_topic(topic_info['id'])

                # Cluster core loops using LLM
                console.print(f"  Clustering {len(core_loops)} core loops...")
                clusters = analyzer.cluster_core_loops_for_topic(
                    topic_info['id'],
                    topic_info['name'],
                    core_loops
                )

                if not clusters:
                    console.print(f"  [red]✗ Clustering failed for this topic[/red]")
                    continue

                # Show preview
                console.print(f"\n  [green]✓ Generated {len(clusters)} new topics:[/green]\n")
                for i, cluster in enumerate(clusters, 1):
                    console.print(f"    {i}. [bold]{cluster['topic_name']}[/bold]")
                    console.print(f"       Core loops: {len(cluster['core_loop_ids'])}")

                    # Show core loop names
                    loop_names = [cl['name'] for cl in core_loops if cl['id'] in cluster['core_loop_ids']]
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
                    console.print(f"  [yellow]Apply this split to topic '{topic_info['name']}'?[/yellow]")
                    confirm = click.confirm("  Proceed", default=True)
                    if not confirm:
                        console.print("  [yellow]Skipped[/yellow]\n")
                        continue

                # Apply the split
                try:
                    stats = db.split_topic(
                        old_topic_id=topic_info['id'],
                        clusters=clusters,
                        course_code=course,
                        delete_old=delete_old
                    )

                    console.print(f"\n  [green]✓ Successfully split topic![/green]")
                    console.print(f"    - Old topic: {stats['old_topic_name']}")
                    console.print(f"    - New topics: {len(stats['new_topics'])}")
                    console.print(f"    - Core loops moved: {stats['core_loops_moved']}")

                    if delete_old and stats.get('old_topic_deleted'):
                        console.print(f"    - Old topic deleted: Yes")
                    elif delete_old:
                        console.print(f"    - Old topic deleted: No ({stats.get('remaining_core_loops', 0)} core loops remain)")

                    if stats.get('errors'):
                        console.print(f"\n  [yellow]Warnings:[/yellow]")
                        for error in stats['errors']:
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
@click.option('--course', '-c', required=True, help='Course code')
@click.option('--loop', '-l', required=True, help='Core loop ID to learn')
@click.option('--lang', type=click.Choice(['en', 'it']), default='en',
              help='Output language (default: en)')
@click.option('--depth', '-d', type=click.Choice(['basic', 'medium', 'advanced']), default='medium',
              help='Explanation depth: basic (concise), medium (balanced), advanced (comprehensive)')
@click.option('--no-concepts', is_flag=True,
              help='Skip prerequisite concept explanations')
@click.option('--adaptive/--no-adaptive', default=True,
              help='Use adaptive teaching (auto-select depth based on mastery, default: enabled)')
@click.option('--strategy', is_flag=True,
              help='Include study strategy and metacognitive guidance')
def learn(course, loop, lang, depth, no_concepts, adaptive, strategy):
    """Learn core loops with AI tutor explanation (enhanced with WHY reasoning)."""
    from core.tutor import Tutor
    from models.llm_manager import LLMManager

    console.print(f"\n[bold cyan]Learning {loop}...[/bold cyan]")

    if adaptive:
        console.print(f"[dim]Mode: Adaptive teaching (depth and prerequisites auto-selected based on mastery)[/dim]\n")
    elif not no_concepts:
        console.print(f"[dim]Mode: Enhanced learning with foundational concepts (depth: {depth})[/dim]\n")
    else:
        console.print(f"[dim]Mode: Direct explanation without prerequisites (depth: {depth})[/dim]\n")

    try:
        # Find course
        with Database() as db:
            all_courses = db.get_all_courses()
            found_course = None
            for c in all_courses:
                if c['code'] == course or c['acronym'] == course:
                    found_course = c
                    break

            if not found_course:
                console.print(f"[red]Course '{course}' not found.[/red]\n")
                return

            course_code = found_course['code']

        # Initialize tutor with enhanced learning
        llm = LLMManager(provider="anthropic")
        tutor = Tutor(llm, language=lang)

        # Get enhanced explanation
        console.print("🤖 Generating deep explanation with reasoning...\n")
        result = tutor.learn(
            course_code=course_code,
            core_loop_id=loop,
            explain_concepts=not no_concepts,
            depth=depth,
            adaptive=adaptive,
            include_study_strategy=strategy
        )

        if not result.success:
            console.print(f"[red]Error: {result.content}[/red]\n")
            return

        # Display explanation
        from rich.markdown import Markdown
        md = Markdown(result.content)
        console.print(md)

        # Display metadata
        includes_prereqs = result.metadata.get('includes_prerequisites', False)
        examples_count = result.metadata.get('examples_count', 0)
        actual_depth = result.metadata.get('depth', depth)
        prereq_status = "with prerequisites" if includes_prereqs else "without prerequisites"
        adaptive_status = "adaptive" if result.metadata.get('adaptive', False) else "manual"
        console.print(f"\n[dim]Core loop: {loop} | Depth: {actual_depth} | {prereq_status} | Examples: {examples_count} | Mode: {adaptive_status}[/dim]\n")

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}\n")
        import traceback
        traceback.print_exc()
        raise click.Abort()


@cli.command()
@click.option('--course', '-c', required=True, help='Course code')
@click.option('--topic', '-t', help='Topic to practice')
@click.option('--difficulty', '-d', type=click.Choice(['easy', 'medium', 'hard']),
              help='Difficulty level')
@click.option('--lang', type=click.Choice(['en', 'it']), default='en',
              help='Output language (default: en)')
def practice(course, topic, difficulty, lang):
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
                if c['code'] == course or c['acronym'] == course:
                    found_course = c
                    break

            if not found_course:
                console.print(f"[red]Course '{course}' not found.[/red]\n")
                return

            course_code = found_course['code']

        # Initialize tutor
        llm = LLMManager(provider="anthropic")
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
            result.metadata['exercise_id'],
            user_answer,
            provide_hints=True
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
@click.option('--course', '-c', required=True, help='Course code')
@click.option('--loop', '-l', required=True, help='Core loop ID')
@click.option('--difficulty', '-d', type=click.Choice(['easy', 'medium', 'hard']),
              default='medium', help='Exercise difficulty')
@click.option('--lang', type=click.Choice(['en', 'it']), default='en',
              help='Output language (default: en)')
def generate(course, loop, difficulty, lang):
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
                if c['code'] == course or c['acronym'] == course:
                    found_course = c
                    break

            if not found_course:
                console.print(f"[red]Course '{course}' not found.[/red]\n")
                return

            course_code = found_course['code']

        # Initialize tutor
        llm = LLMManager(provider="anthropic")
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
        console.print(f"\n[dim]Core loop: {loop} | Difficulty: {difficulty} | Based on {result.metadata.get('based_on_examples', 0)} examples[/dim]\n")

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}\n")
        import traceback
        traceback.print_exc()
        raise click.Abort()


@cli.command()
@click.option('--course', '-c', required=True, help='Course code (e.g., B006802 or ADE)')
@click.option('--questions', '-n', type=int, default=10, help='Number of questions (default: 10)')
@click.option('--topic', '-t', help='Filter by topic')
@click.option('--loop', '-l', help='Filter by core loop ID or name pattern')
@click.option('--difficulty', '-d', type=click.Choice(['easy', 'medium', 'hard']),
              help='Filter by difficulty')
@click.option('--review-only', is_flag=True, help='Only exercises due for review')
@click.option('--procedure', '-p',
              type=click.Choice(['design', 'transformation', 'verification', 'minimization', 'analysis', 'implementation']),
              help='Filter by procedure type')
@click.option('--multi-only', is_flag=True, help='Only show multi-procedure exercises')
@click.option('--tags', help='Filter by tags (comma-separated)')
@click.option('--lang', type=click.Choice(['en', 'it']), default='en',
              help='Language for feedback (default: en)')
def quiz(course, questions, topic, loop, difficulty, review_only, procedure, multi_only, tags, lang):
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
                if c['code'] == course or c['acronym'] == course:
                    found_course = c
                    break

            if not found_course:
                console.print(f"\n[red]Course '{course}' not found.[/red]\n")
                console.print("Use 'examina courses' to see available courses.\n")
                return

            course_code = found_course['code']

        # Initialize quiz engine
        llm = LLMManager(provider="anthropic")
        quiz_engine = QuizEngine(llm_manager=llm, language=lang)

        # Create quiz session
        console.print(f"\n[bold cyan]Creating quiz for {found_course['name']}...[/bold cyan]\n")

        try:
            session = quiz_engine.create_quiz_session(
                course_code=course_code,
                num_questions=questions,
                topic=topic,
                core_loop=loop,
                difficulty=difficulty,
                review_only=review_only,
                procedure_type=procedure,
                multi_only=multi_only,
                tags=tags
            )
        except ValueError as e:
            console.print(f"[red]Error: {e}[/red]\n")
            console.print("Try different filters or add more exercises.\n")
            return

        # Display quiz info
        quiz_info = f"📝 Quiz Session: {session.total_questions} questions"
        if topic:
            quiz_info += f" | Topic: {topic}"
        if loop:
            quiz_info += f" | Core Loop: {loop}"
        if difficulty:
            quiz_info += f" | Difficulty: {difficulty}"
        if procedure:
            quiz_info += f" | Procedure: {procedure}"
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
            if question.core_loops and len(question.core_loops) > 1:
                console.print(f"[dim]Topic: {question.topic_name} | Difficulty: {question.difficulty}[/dim]")
                console.print("[dim]Procedures:[/dim]")
                for idx, loop in enumerate(question.core_loops, 1):
                    loop_name = loop.get('name', 'Unknown')
                    console.print(f"[dim]  {idx}. {loop_name}[/dim]")
                console.print()
            else:
                console.print(f"[dim]Topic: {question.topic_name} | Core Loop: {question.core_loop_name} | Difficulty: {question.difficulty}[/dim]\n")

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
                console=console
            ) as progress:
                progress.add_task(description="Checking...", total=None)
                evaluation = quiz_engine.evaluate_answer(
                    session=session,
                    question=question,
                    user_answer=user_answer,
                    provide_hints=False
                )

            # Update question with results
            question.user_answer = user_answer
            question.is_correct = evaluation['is_correct']
            question.score = evaluation['score']
            question.feedback = evaluation['feedback']
            question.time_spent = int(time.time() - question_start_time)

            # Display feedback
            console.print()
            if question.is_correct:
                console.print(Panel(
                    evaluation['feedback'],
                    title="✅ Correct!",
                    border_style="green"
                ))
            else:
                console.print(Panel(
                    evaluation['feedback'],
                    title="❌ Incorrect",
                    border_style="red"
                ))

            console.print(f"\n[dim]Score: {question.score:.1%}[/dim]")

            # Show progress
            answered = i
            session.total_correct = sum(1 for q in session.questions[:answered] if q.is_correct)
            console.print(f"[dim]Progress: {answered}/{session.total_questions} | Correct: {session.total_correct}/{answered}[/dim]\n")

            # Wait for user to continue (except on last question)
            if i < session.total_questions:
                input("[dim]Press Enter to continue...[/dim]\n")
                console.print()

        # Complete session
        quiz_engine.complete_session(session)

        # Display final results
        console.print("\n" + "="*60 + "\n")
        console.print("[bold cyan]📊 Quiz Complete![/bold cyan]\n")

        final_score = session.score * 100
        score_color = "green" if final_score >= 80 else "yellow" if final_score >= 60 else "red"

        console.print(f"[bold]Final Score: [{score_color}]{final_score:.1f}%[/{score_color}][/bold]")
        console.print(f"Correct: {session.total_correct}/{session.total_questions}")

        if session.started_at and session.completed_at:
            duration = (session.completed_at - session.started_at).total_seconds()
            console.print(f"Time: {int(duration // 60)}m {int(duration % 60)}s")

        # Show mastery updates
        console.print("\n[bold]Mastery Updates:[/bold]")
        from core.analytics import ProgressAnalytics
        analytics = ProgressAnalytics()

        # Get unique core loops from quiz
        core_loops = set(q.core_loop_id for q in session.questions if q.core_loop_id)
        for loop_id in core_loops:
            progress = analytics.get_core_loop_progress(course_code, loop_id)
            loop_name = next((q.core_loop_name for q in session.questions if q.core_loop_id == loop_id), loop_id)

            mastery_pct = progress['mastery_score'] * 100
            mastery_color = "green" if mastery_pct >= 80 else "yellow" if mastery_pct >= 50 else "red"

            console.print(f"  • {loop_name}: [{mastery_color}]{mastery_pct:.0f}%[/{mastery_color}] mastery")

        console.print(f"\n[dim]Session ID: {session.session_id}[/dim]\n")

    except KeyboardInterrupt:
        console.print("\n\n[yellow]Quiz cancelled.[/yellow]\n")
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}\n")
        import traceback
        traceback.print_exc()
        raise click.Abort()


@cli.command()
@click.option('--course', '-c', help='Course code (e.g., B006802 or ADE)')
def suggest(course):
    """Get personalized study suggestions based on spaced repetition."""
    from core.analytics import ProgressAnalytics
    from rich.panel import Panel

    try:
        # Find course
        course_code = None
        if course:
            with Database() as db:
                all_courses = db.get_all_courses()
                found_course = None
                for c in all_courses:
                    if c['code'] == course or c['acronym'] == course:
                        found_course = c
                        break

                if not found_course:
                    console.print(f"\n[red]Course '{course}' not found.[/red]\n")
                    console.print("Use 'examina courses' to see available courses.\n")
                    return

                course_code = found_course['code']

        # Get suggestions
        analytics = ProgressAnalytics()
        suggestions = analytics.get_study_suggestions(course_code)

        # Display suggestions
        console.print("\n[bold cyan]📚 Study Suggestions[/bold cyan]\n")

        if course_code:
            with Database() as db:
                course_info = db.get_course(course_code)
                console.print(f"[dim]Course: {course_info['name']} ({course_info['acronym']})[/dim]\n")

        for suggestion in suggestions:
            console.print(f"  {suggestion}")

        console.print("\n[dim]Use 'examina quiz --course <CODE>' to start practicing![/dim]\n")

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}\n")
        import traceback
        traceback.print_exc()
        raise click.Abort()


@cli.command()
@click.option('--course', '-c', help='Course code (e.g., B006802 or ADE)')
@click.option('--topics', is_flag=True, help='Show topic breakdown')
@click.option('--detailed', is_flag=True, help='Show detailed statistics')
def progress(course, topics, detailed):
    """View your learning progress and mastery levels."""
    from core.analytics import ProgressAnalytics
    from rich.progress import Progress as RichProgress, BarColumn, TextColumn, TaskProgressColumn
    from rich.panel import Panel

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
                if c['code'] == course or c['acronym'] == course:
                    found_course = c
                    break

            if not found_course:
                console.print(f"\n[red]Course '{course}' not found.[/red]\n")
                console.print("Use 'examina courses' to see available courses.\n")
                return

            course_code = found_course['code']

        # Get analytics
        analytics = ProgressAnalytics()
        summary = analytics.get_course_summary(course_code)

        # Display course header
        console.print(f"\n[bold cyan]{found_course['name']}[/bold cyan]")
        console.print(f"[dim]{found_course['code']} • {found_course['acronym']}[/dim]\n")

        # Overall progress
        console.print("[bold]📊 Overall Progress[/bold]\n")

        # Create progress bars
        if summary['total_exercises'] > 0:
            # Attempted progress
            attempted_pct = (summary['exercises_attempted'] / summary['total_exercises']) * 100
            mastered_pct = (summary['exercises_mastered'] / summary['total_exercises']) * 100

            console.print(f"Exercises Attempted: {summary['exercises_attempted']}/{summary['total_exercises']} ({attempted_pct:.1f}%)")
            with RichProgress(
                TextColumn(""),
                BarColumn(complete_style="cyan", finished_style="cyan"),
                TaskProgressColumn(),
                console=console
            ) as progress_bar:
                task = progress_bar.add_task("", total=summary['total_exercises'])
                progress_bar.update(task, completed=summary['exercises_attempted'])

            console.print(f"\nExercises Mastered: {summary['exercises_mastered']}/{summary['total_exercises']} ({mastered_pct:.1f}%)")
            with RichProgress(
                TextColumn(""),
                BarColumn(complete_style="green", finished_style="green"),
                TaskProgressColumn(),
                console=console
            ) as progress_bar:
                task = progress_bar.add_task("", total=summary['total_exercises'])
                progress_bar.update(task, completed=summary['exercises_mastered'])

            console.print(f"\nOverall Mastery: {summary['overall_mastery']:.1%}")
            mastery_color = "green" if summary['overall_mastery'] >= 0.8 else "yellow" if summary['overall_mastery'] >= 0.5 else "red"
            with RichProgress(
                TextColumn(""),
                BarColumn(complete_style=mastery_color, finished_style=mastery_color),
                TaskProgressColumn(),
                console=console
            ) as progress_bar:
                task = progress_bar.add_task("", total=100)
                progress_bar.update(task, completed=int(summary['overall_mastery'] * 100))
        else:
            console.print("[yellow]No exercises found. Run 'examina ingest' and 'examina analyze' first.[/yellow]")

        console.print()

        # Quiz statistics
        if summary['quiz_sessions_completed'] > 0:
            console.print("[bold]🎯 Quiz Statistics[/bold]\n")
            console.print(f"Sessions Completed: {summary['quiz_sessions_completed']}")
            console.print(f"Average Score: {summary['avg_score']:.1%}")
            console.print(f"Total Time: {summary['total_time_spent']} minutes\n")

        # Core loops progress
        if summary['core_loops_discovered'] > 0:
            console.print("[bold]🔄 Core Loops[/bold]\n")
            console.print(f"Discovered: {summary['core_loops_discovered']}")
            console.print(f"Attempted: {summary['core_loops_attempted']}")

            if summary['core_loops_attempted'] > 0:
                progress_pct = (summary['core_loops_attempted'] / summary['core_loops_discovered']) * 100
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
                        'mastered': '✅',
                        'in_progress': '🔄',
                        'weak': '⚠️',
                        'not_started': '❌'
                    }
                    status = status_icons.get(topic_data['status'], '❓')

                    # Mastery color
                    mastery = topic_data['mastery_score']
                    mastery_color = "green" if mastery >= 0.8 else "yellow" if mastery >= 0.5 else "red" if mastery > 0 else "dim"
                    mastery_str = f"[{mastery_color}]{mastery:.1%}[/{mastery_color}]"

                    # Exercises
                    exercises_str = f"{topic_data['exercises_attempted']}/{topic_data['exercises_count']}"

                    table.add_row(
                        topic_data['topic_name'],
                        status,
                        mastery_str,
                        exercises_str
                    )

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
                    console.print(f"  • {area['name']} ({area['topic_name']}): {area['mastery_score']:.1%} mastery")
                console.print()

            # Due reviews
            due_reviews = analytics.get_due_reviews(course_code)
            if due_reviews:
                overdue = [r for r in due_reviews if r['priority'] == 'overdue']
                due_today = [r for r in due_reviews if r['priority'] == 'due_today']

                if overdue:
                    console.print(f"[bold red]Overdue Reviews ({len(overdue)}):[/bold red]")
                    for review in overdue[:5]:
                        console.print(f"  • {review['core_loop_name']}: {review['days_overdue']} days overdue")
                    console.print()

                if due_today:
                    console.print(f"[bold yellow]Due Today ({len(due_today)}):[/bold yellow]")
                    for review in due_today[:5]:
                        console.print(f"  • {review['core_loop_name']}")
                    console.print()

        # Next steps
        console.print("[dim]Use 'examina suggest --course {0}' for study recommendations[/dim]".format(course_code))
        console.print("[dim]Use 'examina quiz --course {0}' to start practicing[/dim]\n".format(course_code))

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}\n")
        import traceback
        traceback.print_exc()
        raise click.Abort()


@cli.command()
@click.option('--course', '-c', required=True, help='Course code')
@click.option('--loop', '-l', required=True, help='Core loop ID or name')
@click.option('--difficulty', '-d', type=click.Choice(['easy', 'medium', 'hard']), default='medium',
              help='Difficulty level for strategy adaptation (default: medium)')
@click.option('--lang', type=click.Choice(['en', 'it']), default='en',
              help='Output language (default: en)')
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
                if c['code'] == course or c['acronym'] == course:
                    found_course = c
                    break

            if not found_course:
                console.print(f"[red]Course '{course}' not found.[/red]\n")
                return

            course_code = found_course['code']

            # Get core loop details
            core_loop = db.conn.execute("""
                SELECT cl.*, t.name as topic_name
                FROM core_loops cl
                JOIN topics t ON cl.topic_id = t.id
                WHERE cl.id = ? AND t.course_code = ?
            """, (loop, course_code)).fetchone()

            if not core_loop:
                # Try searching by name pattern
                core_loop = db.conn.execute("""
                    SELECT cl.*, t.name as topic_name
                    FROM core_loops cl
                    JOIN topics t ON cl.topic_id = t.id
                    WHERE cl.name LIKE ? AND t.course_code = ?
                    LIMIT 1
                """, (f"%{loop}%", course_code)).fetchone()

            if not core_loop:
                console.print(f"[red]Core loop '{loop}' not found for course {course}.[/red]\n")
                console.print("Use 'examina info --course {0}' to see available core loops.\n".format(course))
                return

        core_loop_dict = dict(core_loop)
        core_loop_name = core_loop_dict.get('name', '')

        # Initialize strategy manager
        strategy_mgr = StudyStrategyManager(language=lang)

        # Get strategy
        strat = strategy_mgr.get_strategy_for_core_loop(core_loop_name, difficulty=difficulty)

        if not strat:
            console.print(f"[yellow]No specific strategy found for '{core_loop_name}'.[/yellow]\n")
            console.print("This core loop may be new or not yet covered by the strategy system.\n")
            return

        # Format and display
        formatted_strategy = strategy_mgr.format_strategy_output(strat, core_loop_name)
        md = Markdown(formatted_strategy)
        console.print(md)

        console.print(f"\n[dim]Core loop: {core_loop_name} | Difficulty: {difficulty} | Language: {lang}[/dim]\n")

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}\n")
        import traceback
        traceback.print_exc()
        raise click.Abort()


@cli.command()
@click.option('--course', '-c', required=True, help='Course code')
@click.option('--limit', '-n', type=int, default=10, help='Number of items in learning path (default: 10)')
@click.option('--lang', type=click.Choice(['en', 'it']), default='en',
              help='Output language (default: en)')
def path(course, limit, lang):
    """Show personalized learning path based on mastery and spaced repetition."""
    from core.adaptive_teaching import AdaptiveTeachingManager
    from rich.table import Table
    from rich.panel import Panel

    try:
        # Find course
        with Database() as db:
            all_courses = db.get_all_courses()
            found_course = None
            for c in all_courses:
                if c['code'] == course or c['acronym'] == course:
                    found_course = c
                    break

            if not found_course:
                console.print(f"\n[red]Course '{course}' not found.[/red]\n")
                console.print("Use 'examina courses' to see available courses.\n")
                return

            course_code = found_course['code']

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
            action_icons = {
                'review': '🔄',
                'strengthen': '💪',
                'learn': '📖',
                'practice': '✍️'
            }
            action_display = f"{action_icons.get(item['action'], '•')} {item['action'].title()}"

            # Format urgency color
            urgency_colors = {
                'high': 'red',
                'medium': 'yellow',
                'low': 'dim'
            }
            urgency_color = urgency_colors.get(item.get('urgency', 'low'), 'dim')

            table.add_row(
                str(item['priority']),
                action_display,
                f"[{urgency_color}]{item['core_loop']}[/{urgency_color}]",
                item['topic'],
                item['reason'],
                f"{item['estimated_time']}m"
            )

        console.print(table)
        console.print(f"\n[dim]Total estimated time: {sum(item['estimated_time'] for item in learning_path)} minutes[/dim]\n")

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}\n")
        import traceback
        traceback.print_exc()
        raise click.Abort()


@cli.command()
@click.option('--course', '-c', required=True, help='Course code')
@click.option('--loop', '-l', help='Filter by specific core loop')
@click.option('--lang', type=click.Choice(['en', 'it']), default='en',
              help='Output language (default: en)')
def gaps(course, loop, lang):
    """Identify knowledge gaps and weak areas."""
    from core.adaptive_teaching import AdaptiveTeachingManager
    from rich.table import Table
    from rich.panel import Panel

    try:
        # Find course
        with Database() as db:
            all_courses = db.get_all_courses()
            found_course = None
            for c in all_courses:
                if c['code'] == course or c['acronym'] == course:
                    found_course = c
                    break

            if not found_course:
                console.print(f"\n[red]Course '{course}' not found.[/red]\n")
                console.print("Use 'examina courses' to see available courses.\n")
                return

            course_code = found_course['code']

        # Detect knowledge gaps
        with AdaptiveTeachingManager() as atm:
            knowledge_gaps = atm.detect_knowledge_gaps(course_code, core_loop_name=loop)

        if not knowledge_gaps:
            console.print(f"\n[green]✅ No significant knowledge gaps detected![/green]")
            console.print(f"[dim]Your mastery levels look good across all topics.[/dim]\n")
            return

        # Display header
        console.print(f"\n[bold cyan]🔍 Knowledge Gaps Analysis[/bold cyan]")
        console.print(f"[dim]{found_course['name']} ({found_course['acronym']})[/dim]\n")

        # Group by severity
        high_gaps = [g for g in knowledge_gaps if g['severity'] == 'high']
        medium_gaps = [g for g in knowledge_gaps if g['severity'] == 'medium']
        low_gaps = [g for g in knowledge_gaps if g['severity'] == 'low']

        # Display high priority gaps
        if high_gaps:
            console.print("[bold red]⚠️  High Priority Gaps[/bold red]\n")
            for gap in high_gaps:
                mastery_pct = int(gap['mastery'] * 100)
                console.print(f"  [red]•[/red] [bold]{gap['gap']}[/bold] ({gap['topic']})")
                console.print(f"    Mastery: {mastery_pct}%")
                console.print(f"    💡 {gap['recommendation']}")
                if gap['impact']:
                    console.print(f"    Affects: {', '.join(gap['impact'][:3])}")
                console.print()

        # Display medium priority gaps
        if medium_gaps:
            console.print("[bold yellow]⚡ Medium Priority Gaps[/bold yellow]\n")
            for gap in medium_gaps:
                mastery_pct = int(gap['mastery'] * 100)
                console.print(f"  [yellow]•[/yellow] {gap['gap']} ({gap['topic']}) - {mastery_pct}% mastery")
                console.print(f"    💡 {gap['recommendation']}\n")

        # Display low priority gaps (summarized)
        if low_gaps:
            console.print(f"[dim]ℹ️  {len(low_gaps)} additional area(s) for improvement (low priority)[/dim]\n")

        # Summary
        console.print(f"[bold]Summary:[/bold]")
        console.print(f"  Total gaps found: {len(knowledge_gaps)}")
        console.print(f"  High priority: {len(high_gaps)}")
        console.print(f"  Medium priority: {len(medium_gaps)}")
        console.print(f"  Low priority: {len(low_gaps)}\n")

        console.print(f"[dim]Use 'examina path --course {course}' to see a personalized study plan[/dim]\n")

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}\n")
        import traceback
        traceback.print_exc()
        raise click.Abort()


@cli.command()
@click.option('--course', '-c', required=True, help='Course code')
@click.option('--dry-run', is_flag=True, help='Show what would be merged without making changes')
@click.option('--threshold', type=float, default=None, help='Similarity threshold (0.0-1.0, default: 0.85 for semantic, 0.85 for string)')
def deduplicate(course, dry_run, threshold):
    """Merge duplicate topics and core loops using semantic similarity."""
    from difflib import SequenceMatcher

    # Try to import semantic matcher
    try:
        from core.semantic_matcher import SemanticMatcher
        if Config.SEMANTIC_SIMILARITY_ENABLED:
            semantic_matcher = SemanticMatcher()
            use_semantic = semantic_matcher.enabled
            if use_semantic:
                console.print("[info]Using semantic similarity matching[/info]")
                default_threshold = Config.SEMANTIC_SIMILARITY_THRESHOLD
            else:
                console.print("[yellow]Semantic matcher unavailable, using string similarity[/yellow]")
                use_semantic = False
                default_threshold = Config.CORE_LOOP_SIMILARITY_THRESHOLD
        else:
            console.print("[info]Semantic matching disabled, using string similarity[/info]")
            use_semantic = False
            semantic_matcher = None
            default_threshold = Config.CORE_LOOP_SIMILARITY_THRESHOLD
    except ImportError:
        console.print("[yellow]SemanticMatcher not available, using string similarity[/yellow]")
        use_semantic = False
        semantic_matcher = None
        default_threshold = Config.CORE_LOOP_SIMILARITY_THRESHOLD

    # Use provided threshold or default
    threshold = threshold if threshold is not None else default_threshold

    console.print(f"\n[bold cyan]Deduplicating {course}...[/bold cyan]")
    console.print(f"[info]Threshold: {threshold:.2f}[/info]\n")

    if dry_run:
        console.print("[yellow]DRY RUN MODE - No changes will be made[/yellow]\n")

    try:
        with Database() as db:
            # Find course
            all_courses = db.get_all_courses()
            found_course = None
            for c in all_courses:
                if c['code'] == course or c['acronym'] == course:
                    found_course = c
                    break

            if not found_course:
                console.print(f"[red]Course '{course}' not found.[/red]\n")
                return

            course_code = found_course['code']

            # Deduplicate topics
            console.print("[bold]Deduplicating Topics...[/bold]")
            topics = db.get_topics_by_course(course_code)
            topic_merges = []
            topic_skips = []

            for i, topic1 in enumerate(topics):
                for topic2 in topics[i+1:]:
                    if use_semantic and semantic_matcher:
                        result = semantic_matcher.should_merge(
                            topic1['name'], topic2['name'], threshold
                        )
                        if result.should_merge:
                            topic_merges.append((topic1, topic2, result.similarity_score, result.reason))
                        elif Config.SEMANTIC_LOG_NEAR_MISSES and result.similarity_score >= 0.80:
                            topic_skips.append((topic1, topic2, result.similarity_score, result.reason))
                    else:
                        similarity = SequenceMatcher(None, topic1['name'].lower(), topic2['name'].lower()).ratio()
                        if similarity >= threshold:
                            topic_merges.append((topic1, topic2, similarity, "string_similarity"))

            if topic_merges:
                console.print(f"Found {len(topic_merges)} topic pairs to merge:\n")
                for t1, t2, sim, reason in topic_merges:
                    console.print(f"  • '{t1['name']}' ← '{t2['name']}'")
                    console.print(f"    Similarity: {sim:.2f}, Reason: {reason}")

                if not dry_run:
                    for t1, t2, sim, reason in topic_merges:
                        # Update all exercises to use canonical topic
                        db.conn.execute("""
                            UPDATE exercises SET topic_id = ? WHERE topic_id = ?
                        """, (t1['id'], t2['id']))

                        # Update all core loops to use canonical topic
                        db.conn.execute("""
                            UPDATE core_loops SET topic_id = ? WHERE topic_id = ?
                        """, (t1['id'], t2['id']))

                        # Delete duplicate topic
                        db.conn.execute("DELETE FROM topics WHERE id = ?", (t2['id'],))

                    db.conn.commit()
                    console.print(f"\n[green]✓ Merged {len(topic_merges)} duplicate topics[/green]\n")
            else:
                console.print("  No duplicate topics found\n")

            # Show near-misses if semantic matching is enabled
            if topic_skips:
                console.print(f"\n[yellow]Skipped {len(topic_skips)} near-misses (high similarity but semantically different):[/yellow]")
                for t1, t2, sim, reason in topic_skips:
                    console.print(f"  • '{t1['name']}' ≠ '{t2['name']}'")
                    console.print(f"    Similarity: {sim:.2f}, Reason: {reason}")
                console.print()

            # Deduplicate core loops
            console.print("[bold]Deduplicating Core Loops...[/bold]")
            core_loops = db.get_core_loops_by_course(course_code)
            loop_merges = []
            loop_skips = []

            for i, loop1 in enumerate(core_loops):
                for loop2 in core_loops[i+1:]:
                    if use_semantic and semantic_matcher:
                        result = semantic_matcher.should_merge(
                            loop1['name'], loop2['name'], threshold
                        )
                        if result.should_merge:
                            loop_merges.append((loop1, loop2, result.similarity_score, result.reason))
                        elif Config.SEMANTIC_LOG_NEAR_MISSES and result.similarity_score >= 0.80:
                            loop_skips.append((loop1, loop2, result.similarity_score, result.reason))
                    else:
                        similarity = SequenceMatcher(None, loop1['name'].lower(), loop2['name'].lower()).ratio()
                        if similarity >= threshold:
                            loop_merges.append((loop1, loop2, similarity, "string_similarity"))

            if loop_merges:
                console.print(f"Found {len(loop_merges)} core loop pairs to merge:\n")
                for l1, l2, sim, reason in loop_merges:
                    console.print(f"  • '{l1['name']}' ← '{l2['name']}'")
                    console.print(f"    Similarity: {sim:.2f}, Reason: {reason}")

                if not dry_run:
                    for l1, l2, sim, reason in loop_merges:
                        # Update exercise_core_loops junction table
                        db.conn.execute("""
                            UPDATE exercise_core_loops
                            SET core_loop_id = ?
                            WHERE core_loop_id = ?
                        """, (l1['id'], l2['id']))

                        # Update legacy core_loop_id in exercises
                        db.conn.execute("""
                            UPDATE exercises
                            SET core_loop_id = ?
                            WHERE core_loop_id = ?
                        """, (l1['id'], l2['id']))

                        # Delete duplicate core loop
                        db.conn.execute("DELETE FROM core_loops WHERE id = ?", (l2['id'],))

                    db.conn.commit()
                    console.print(f"\n[green]✓ Merged {len(loop_merges)} duplicate core loops[/green]\n")
            else:
                console.print("  No duplicate core loops found\n")

            # Show near-misses if semantic matching is enabled
            if loop_skips:
                console.print(f"\n[yellow]Skipped {len(loop_skips)} near-misses (high similarity but semantically different):[/yellow]")
                for l1, l2, sim, reason in loop_skips:
                    console.print(f"  • '{l1['name']}' ≠ '{l2['name']}'")
                    console.print(f"    Similarity: {sim:.2f}, Reason: {reason}")
                console.print()

            if not dry_run and (topic_merges or loop_merges):
                console.print("[green]Deduplication complete![/green]\n")
            elif dry_run:
                console.print("[yellow]Dry run complete. Use without --dry-run to apply changes.[/yellow]\n")
            else:
                console.print("[green]No duplicates found![/green]\n")

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}\n")
        import traceback
        traceback.print_exc()
        raise click.Abort()


if __name__ == '__main__':
    cli()
