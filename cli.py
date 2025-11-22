#!/usr/bin/env python3
"""
Examina - AI-powered exam tutor system.
CLI interface for managing courses, ingesting exams, and studying.
"""

import click
from rich.console import Console
from rich.table import Table
from rich import print as rprint
from pathlib import Path

from config import Config
from storage.database import Database
from storage.file_manager import FileManager
from core.pdf_processor import PDFProcessor
from core.exercise_splitter import ExerciseSplitter
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

            console.print()

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}\n")
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
@click.option('--course', '-c', required=True, help='Course code')
@click.option('--loop', '-l', help='Specific core loop to practice')
def learn(course, loop):
    """Learn core loops with guided examples."""
    console.print(f"\n[bold cyan]Learning mode for {course}...[/bold cyan]\n")
    console.print("[yellow]⚠️  This feature is not yet implemented.[/yellow]")
    console.print("Coming in Phase 4: AI tutor interaction.\n")


@cli.command()
@click.option('--course', '-c', required=True, help='Course code')
@click.option('--loop', '-l', help='Core loop to practice')
@click.option('--topic', '-t', help='Topic to practice')
def practice(course, loop, topic):
    """Practice exercises by topic or core loop."""
    console.print(f"\n[bold cyan]Practice mode for {course}...[/bold cyan]\n")
    console.print("[yellow]⚠️  This feature is not yet implemented.[/yellow]")
    console.print("Coming in Phase 4: AI tutor interaction.\n")


@cli.command()
@click.option('--course', '-c', required=True, help='Course code')
@click.option('--loop', '-l', help='Core loop')
@click.option('--difficulty', '-d', type=click.Choice(['easy', 'medium', 'hard']),
              default='medium', help='Exercise difficulty')
def generate(course, loop, difficulty):
    """Generate new practice exercises with AI."""
    console.print(f"\n[bold cyan]Generating {difficulty} exercise for {course}...[/bold cyan]\n")
    console.print("[yellow]⚠️  This feature is not yet implemented.[/yellow]")
    console.print("Coming in Phase 4: Exercise generation.\n")


@cli.command()
@click.option('--course', '-c', required=True, help='Course code')
@click.option('--questions', '-q', default=10, help='Number of questions')
@click.option('--loop', '-l', help='Specific core loop')
@click.option('--topic', '-t', help='Specific topic')
def quiz(course, questions, loop, topic):
    """Take a quiz to test your knowledge."""
    console.print(f"\n[bold cyan]Quiz mode for {course} ({questions} questions)...[/bold cyan]\n")
    console.print("[yellow]⚠️  This feature is not yet implemented.[/yellow]")
    console.print("Coming in Phase 5: Quiz system.\n")


@cli.command()
def suggest():
    """Get study suggestions based on spaced repetition."""
    console.print("\n[bold cyan]Study Suggestions[/bold cyan]\n")
    console.print("[yellow]⚠️  This feature is not yet implemented.[/yellow]")
    console.print("Coming in Phase 5: Progress tracking and recommendations.\n")


@cli.command()
@click.option('--course', '-c', help='Filter by course')
def progress(course):
    """View your learning progress."""
    console.print("\n[bold cyan]Learning Progress[/bold cyan]\n")
    console.print("[yellow]⚠️  This feature is not yet implemented.[/yellow]")
    console.print("Coming in Phase 5: Progress tracking.\n")


if __name__ == '__main__':
    cli()
