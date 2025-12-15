"""
Database management for Examina.
Handles SQLite operations and schema management.
"""

import sqlite3
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

from config import Config


class Database:
    """Manages SQLite database operations for Examina."""

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize database connection.

        Args:
            db_path: Path to SQLite database file. Uses Config.DB_PATH if not provided.
        """
        self.db_path = db_path or Config.DB_PATH
        self.conn: Optional[sqlite3.Connection] = None

    def connect(self):
        """Establish database connection."""
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row  # Enable dict-like access
        # Enable foreign keys
        self.conn.execute("PRAGMA foreign_keys = ON")

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.conn:
            if exc_type is None:
                self.conn.commit()
            else:
                self.conn.rollback()
            self.close()

    def initialize(self):
        """Create all tables and indexes."""
        if not self.conn:
            self.connect()

        self._create_tables()
        self._run_migrations()  # Run migrations BEFORE indexes (migrations may add columns needed for indexes)
        self._create_indexes()
        self.conn.commit()

    def _run_migrations(self):
        """Run database migrations for schema updates."""
        # Check if low_confidence_skipped column exists in exercises table
        cursor = self.conn.execute("PRAGMA table_info(exercises)")
        columns = [row[1] for row in cursor.fetchall()]

        if "low_confidence_skipped" not in columns:
            print(
                "[INFO] Running migration: Adding low_confidence_skipped column to exercises table"
            )
            self.conn.execute("""
                ALTER TABLE exercises
                ADD COLUMN low_confidence_skipped BOOLEAN DEFAULT 0
            """)
            print("[INFO] Migration completed successfully")

        # Phase 5: Create exercise_reviews table if it doesn't exist
        cursor = self.conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='exercise_reviews'
        """)
        if not cursor.fetchone():
            print("[INFO] Running migration: Creating exercise_reviews table for spaced repetition")
            self.conn.execute("""
                CREATE TABLE exercise_reviews (
                    exercise_id TEXT PRIMARY KEY,
                    course_code TEXT NOT NULL,
                    easiness_factor REAL DEFAULT 2.5,
                    repetition_number INTEGER DEFAULT 0,
                    interval_days INTEGER DEFAULT 0,
                    next_review_date DATE,
                    last_reviewed_at TIMESTAMP,
                    total_reviews INTEGER DEFAULT 0,
                    correct_reviews INTEGER DEFAULT 0,
                    mastery_level TEXT DEFAULT 'new',
                    FOREIGN KEY (exercise_id) REFERENCES exercises(id),
                    FOREIGN KEY (course_code) REFERENCES courses(code)
                )
            """)
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_exercise_reviews_course
                ON exercise_reviews(course_code)
            """)
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_exercise_reviews_next_review
                ON exercise_reviews(next_review_date)
            """)
            print("[INFO] Migration completed: exercise_reviews table created")

        # Phase 5: Create topic_mastery table if it doesn't exist
        cursor = self.conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='topic_mastery'
        """)
        if not cursor.fetchone():
            print("[INFO] Running migration: Creating topic_mastery table for progress tracking")
            self.conn.execute("""
                CREATE TABLE topic_mastery (
                    topic_id INTEGER PRIMARY KEY,
                    course_code TEXT NOT NULL,
                    exercises_total INTEGER DEFAULT 0,
                    exercises_mastered INTEGER DEFAULT 0,
                    mastery_percentage REAL DEFAULT 0.0,
                    last_practiced_at TIMESTAMP,
                    FOREIGN KEY (topic_id) REFERENCES topics(id),
                    FOREIGN KEY (course_code) REFERENCES courses(code)
                )
            """)
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_topic_mastery_course
                ON topic_mastery(course_code)
            """)
            print("[INFO] Migration completed: topic_mastery table created")

        # Phase 5: Migrate quiz_sessions table to new schema
        cursor = self.conn.execute("PRAGMA table_info(quiz_sessions)")
        quiz_sessions_columns = [row[1] for row in cursor.fetchall()]

        # Check if we need to migrate (old schema has 'time_limit' column)
        if "time_limit" in quiz_sessions_columns and "correct_answers" not in quiz_sessions_columns:
            print("[INFO] Running migration: Updating quiz_sessions table to Phase 5 schema")

            # Create new table with Phase 5 schema
            self.conn.execute("""
                CREATE TABLE quiz_sessions_new (
                    id TEXT PRIMARY KEY,
                    course_code TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    total_questions INTEGER,
                    correct_answers INTEGER,
                    score_percentage REAL,
                    quiz_type TEXT,
                    filter_topic_id INTEGER,
                    filter_knowledge_item_id TEXT,
                    filter_difficulty TEXT,
                    FOREIGN KEY (course_code) REFERENCES courses(code),
                    FOREIGN KEY (filter_topic_id) REFERENCES topics(id)
                )
            """)

            # Migrate existing data
            self.conn.execute("""
                INSERT INTO quiz_sessions_new
                (id, course_code, created_at, completed_at, total_questions,
                 correct_answers, score_percentage, quiz_type, filter_topic_id, filter_knowledge_item_id)
                SELECT
                    id, course_code, started_at, completed_at, total_questions,
                    total_correct, score, quiz_type, topic_id, knowledge_item_id
                FROM quiz_sessions
            """)

            # Drop old table and rename new one
            self.conn.execute("DROP TABLE quiz_sessions")
            self.conn.execute("ALTER TABLE quiz_sessions_new RENAME TO quiz_sessions")

            # Recreate index
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_quiz_sessions_course
                ON quiz_sessions(course_code)
            """)
            print("[INFO] Migration completed: quiz_sessions table updated")

        # Phase 5: Migrate quiz_answers table to quiz_attempts
        cursor = self.conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='quiz_answers'
        """)
        if cursor.fetchone():
            cursor = self.conn.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='quiz_attempts'
            """)
            if not cursor.fetchone():
                print("[INFO] Running migration: Migrating quiz_answers to quiz_attempts table")

                # Create new quiz_attempts table
                self.conn.execute("""
                    CREATE TABLE quiz_attempts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        exercise_id TEXT NOT NULL,
                        user_answer TEXT,
                        correct BOOLEAN,
                        time_taken_seconds INTEGER,
                        hint_used BOOLEAN DEFAULT 0,
                        feedback TEXT,
                        attempted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES quiz_sessions(id),
                        FOREIGN KEY (exercise_id) REFERENCES exercises(id)
                    )
                """)

                # Migrate existing data from quiz_answers
                self.conn.execute("""
                    INSERT INTO quiz_attempts
                    (session_id, exercise_id, user_answer, correct, time_taken_seconds,
                     hint_used, feedback, attempted_at)
                    SELECT
                        session_id, exercise_id, student_answer, is_correct, time_spent,
                        hint_used, mistakes, answered_at
                    FROM quiz_answers
                """)

                # Create index
                self.conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_quiz_attempts_session
                    ON quiz_attempts(session_id)
                """)
                self.conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_quiz_attempts_exercise
                    ON quiz_attempts(exercise_id)
                """)

                # Drop old table
                self.conn.execute("DROP TABLE quiz_answers")

                print("[INFO] Migration completed: quiz_answers migrated to quiz_attempts")

        # Migration: Mark exercises with knowledge_item_id as analyzed
        # (Fixes legacy data where exercises were analyzed but not marked)
        cursor = self.conn.execute("""
            SELECT COUNT(*) FROM exercises
            WHERE knowledge_item_id IS NOT NULL AND analyzed = 0
        """)
        unanalyzed_count = cursor.fetchone()[0]

        if unanalyzed_count > 0:
            print(
                f"[INFO] Running migration: Marking {unanalyzed_count} exercises with knowledge_item_id as analyzed=1"
            )

            self.conn.execute("""
                UPDATE exercises
                SET analyzed = 1
                WHERE knowledge_item_id IS NOT NULL AND analyzed = 0
            """)

            print(f"[INFO] Migration completed: {unanalyzed_count} exercises marked as analyzed")

        # Multi-core-loop migration: Create exercise_knowledge_items table
        cursor = self.conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='exercise_knowledge_items'
        """)
        if not cursor.fetchone():
            print("[INFO] Running migration: Creating exercise_knowledge_items junction table")
            self.conn.execute("""
                CREATE TABLE exercise_knowledge_items (
                    exercise_id TEXT NOT NULL,
                    knowledge_item_id TEXT NOT NULL,
                    step_number INTEGER,
                    PRIMARY KEY (exercise_id, knowledge_item_id),
                    FOREIGN KEY (exercise_id) REFERENCES exercises(id) ON DELETE CASCADE,
                    FOREIGN KEY (knowledge_item_id) REFERENCES knowledge_items(id) ON DELETE CASCADE
                )
            """)
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_exercise_knowledge_items_exercise
                ON exercise_knowledge_items(exercise_id)
            """)
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_exercise_knowledge_items_knowledge_item
                ON exercise_knowledge_items(knowledge_item_id)
            """)
            print("[INFO] Migration completed: exercise_knowledge_items table created")

        # Multi-core-loop migration: Add tags column to exercises
        cursor = self.conn.execute("PRAGMA table_info(exercises)")
        columns = [row[1] for row in cursor.fetchall()]

        if "tags" not in columns:
            print("[INFO] Running migration: Adding tags column to exercises table")
            self.conn.execute("""
                ALTER TABLE exercises
                ADD COLUMN tags TEXT
            """)
            print("[INFO] Migration completed: tags column added")

        # Multi-core-loop migration: Migrate existing knowledge_item_id data to exercise_knowledge_items
        # Check if migration is needed (exercise_knowledge_items exists but might be empty)
        cursor = self.conn.execute("""
            SELECT COUNT(*) FROM exercises
            WHERE knowledge_item_id IS NOT NULL
        """)
        exercises_with_knowledge_item = cursor.fetchone()[0]

        cursor = self.conn.execute("SELECT COUNT(*) FROM exercise_knowledge_items")
        junction_entries = cursor.fetchone()[0]

        if exercises_with_knowledge_item > 0 and junction_entries == 0:
            print(
                f"[INFO] Running migration: Migrating {exercises_with_knowledge_item} exercise knowledge_item_id references to exercise_knowledge_items table"
            )

            # Insert all existing knowledge_item_id relationships into junction table
            self.conn.execute("""
                INSERT INTO exercise_knowledge_items (exercise_id, knowledge_item_id, step_number)
                SELECT id, knowledge_item_id, NULL
                FROM exercises
                WHERE knowledge_item_id IS NOT NULL
            """)

            cursor = self.conn.execute("SELECT COUNT(*) FROM exercise_knowledge_items")
            migrated_count = cursor.fetchone()[0]

            print(
                f"[INFO] Migration completed: {migrated_count} relationships migrated to exercise_knowledge_items"
            )
            print(
                "[INFO] Note: exercises.knowledge_item_id column retained for backward compatibility"
            )

        # Phase 9.1: Add exercise_type and theory_metadata columns
        cursor = self.conn.execute("PRAGMA table_info(exercises)")
        columns = [row[1] for row in cursor.fetchall()]

        if "exercise_type" not in columns:
            print("[INFO] Running migration: Adding exercise_type column to exercises table")
            self.conn.execute("""
                ALTER TABLE exercises
                ADD COLUMN exercise_type TEXT DEFAULT 'procedural'
                    CHECK(exercise_type IN ('procedural', 'theory', 'proof', 'hybrid'))
            """)
            print("[INFO] Migration completed: exercise_type column added")

        if "theory_metadata" not in columns:
            print("[INFO] Running migration: Adding theory_metadata column to exercises table")
            self.conn.execute("""
                ALTER TABLE exercises
                ADD COLUMN theory_metadata TEXT
            """)
            print("[INFO] Migration completed: theory_metadata column added")

        # Phase 9.2: Add theory question categorization fields
        cursor = self.conn.execute("PRAGMA table_info(exercises)")
        columns = [row[1] for row in cursor.fetchall()]

        if "theory_category" not in columns:
            print("[INFO] Running migration: Adding theory_category column to exercises table")
            self.conn.execute("""
                ALTER TABLE exercises
                ADD COLUMN theory_category TEXT
            """)
            print("[INFO] Migration completed: theory_category column added")

        if "theorem_name" not in columns:
            self.conn.execute("""
                ALTER TABLE exercises
                ADD COLUMN theorem_name TEXT
            """)
            print("[INFO] Migration completed: theorem_name column added")

        if "concept_id" not in columns:
            self.conn.execute("""
                ALTER TABLE exercises
                ADD COLUMN concept_id TEXT
            """)
            print("[INFO] Migration completed: concept_id column added")

        if "prerequisite_concepts" not in columns:
            self.conn.execute("""
                ALTER TABLE exercises
                ADD COLUMN prerequisite_concepts TEXT
            """)
            print("[INFO] Migration completed: prerequisite_concepts column added")

        # Phase 9.2: Create theory_concepts table if it doesn't exist
        cursor = self.conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='theory_concepts'
        """)
        if not cursor.fetchone():
            print("[INFO] Running migration: Creating theory_concepts table")
            self.conn.execute("""
                CREATE TABLE theory_concepts (
                    id TEXT PRIMARY KEY,
                    course_code TEXT NOT NULL,
                    topic_id INTEGER,
                    name TEXT NOT NULL,
                    category TEXT,
                    description TEXT,
                    prerequisite_concept_ids TEXT,
                    related_concept_ids TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (course_code) REFERENCES courses(code),
                    FOREIGN KEY (topic_id) REFERENCES topics(id)
                )
            """)
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_theory_concepts_course
                ON theory_concepts(course_code)
            """)
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_theory_concepts_topic
                ON theory_concepts(topic_id)
            """)
            print("[INFO] Migration completed: theory_concepts table created")

        # Phase: Automatic Language Detection - Add language columns
        cursor = self.conn.execute("PRAGMA table_info(knowledge_items)")
        columns = [row[1] for row in cursor.fetchall()]

        if "language" not in columns:
            print("[INFO] Running migration: Adding language column to knowledge_items table")
            self.conn.execute("""
                ALTER TABLE knowledge_items
                ADD COLUMN language TEXT DEFAULT NULL
            """)
            print("[INFO] Migration completed: language column added to knowledge_items")

        cursor = self.conn.execute("PRAGMA table_info(topics)")
        columns = [row[1] for row in cursor.fetchall()]

        if "language" not in columns:
            print("[INFO] Running migration: Adding language column to topics table")
            self.conn.execute("""
                ALTER TABLE topics
                ADD COLUMN language TEXT DEFAULT NULL
            """)
            print("[INFO] Migration completed: language column added to topics")
            print(
                "[INFO] Note: Run 'examina detect-languages --course CODE' to detect languages for existing data"
            )

        # Phase: Learning Materials Support - Create tables for lecture notes, theory, worked examples
        cursor = self.conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='learning_materials'
        """)
        if not cursor.fetchone():
            print(
                "[INFO] Running migration: Creating learning_materials table for lecture notes and theory"
            )
            self.conn.execute("""
                CREATE TABLE learning_materials (
                    id TEXT PRIMARY KEY,
                    course_code TEXT NOT NULL,
                    material_type TEXT NOT NULL,
                    title TEXT,
                    content TEXT NOT NULL,
                    source_pdf TEXT NOT NULL,
                    page_number INTEGER,
                    has_images BOOLEAN DEFAULT 0,
                    image_paths TEXT,
                    latex_content TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (course_code) REFERENCES courses(code)
                )
            """)
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_learning_materials_course
                ON learning_materials(course_code)
            """)
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_learning_materials_type
                ON learning_materials(material_type)
            """)
            print("[INFO] Migration completed: learning_materials table created")

        # Phase: Learning Materials Support - Many-to-many: materials ↔ topics
        cursor = self.conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='material_topics'
        """)
        if not cursor.fetchone():
            print("[INFO] Running migration: Creating material_topics join table")
            self.conn.execute("""
                CREATE TABLE material_topics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    material_id TEXT NOT NULL,
                    topic_id INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (material_id) REFERENCES learning_materials(id),
                    FOREIGN KEY (topic_id) REFERENCES topics(id),
                    UNIQUE(material_id, topic_id)
                )
            """)
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_material_topics_material
                ON material_topics(material_id)
            """)
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_material_topics_topic
                ON material_topics(topic_id)
            """)
            print("[INFO] Migration completed: material_topics join table created")

        # Phase: Learning Materials Support - Create links between materials and exercises
        cursor = self.conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='material_exercise_links'
        """)
        if not cursor.fetchone():
            print("[INFO] Running migration: Creating material_exercise_links table")
            self.conn.execute("""
                CREATE TABLE material_exercise_links (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    material_id TEXT NOT NULL,
                    exercise_id TEXT NOT NULL,
                    link_type TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (material_id) REFERENCES learning_materials(id),
                    FOREIGN KEY (exercise_id) REFERENCES exercises(id),
                    UNIQUE(material_id, exercise_id, link_type)
                )
            """)
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_material_links_material
                ON material_exercise_links(material_id)
            """)
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_material_links_exercise
                ON material_exercise_links(exercise_id)
            """)
            print("[INFO] Migration completed: material_exercise_links table created")

        # Option 3: Add user_id column to procedure_cache_entries for web-ready multi-tenant support
        cursor = self.conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='procedure_cache_entries'
        """)
        if cursor.fetchone():
            # Table exists, check if user_id column exists
            cursor = self.conn.execute("PRAGMA table_info(procedure_cache_entries)")
            columns = [row[1] for row in cursor.fetchall()]
            if "user_id" not in columns:
                print(
                    "[INFO] Running migration: Adding user_id column to procedure_cache_entries for multi-tenant support"
                )
                self.conn.execute("""
                    ALTER TABLE procedure_cache_entries
                    ADD COLUMN user_id TEXT
                """)
                # Recreate unique constraint is complex in SQLite, skip for now
                # The INSERT will handle duplicates with ON CONFLICT
                print("[INFO] Migration completed: user_id column added to procedure_cache_entries")

    def _create_tables(self):
        """Create all database tables."""

        # Courses table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS courses (
                code TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                original_name TEXT,
                acronym TEXT,
                degree_level TEXT,
                degree_program TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Topics table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS topics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                course_code TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (course_code) REFERENCES courses(code),
                UNIQUE(course_code, name)
            )
        """)

        # Core loops table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_items (
                id TEXT PRIMARY KEY,
                topic_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                procedure TEXT NOT NULL,
                difficulty_avg REAL DEFAULT 0.0,
                exercise_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (topic_id) REFERENCES topics(id)
            )
        """)

        # Exercises table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS exercises (
                id TEXT PRIMARY KEY,
                course_code TEXT NOT NULL,
                topic_id INTEGER,
                knowledge_item_id TEXT,
                source_pdf TEXT,
                page_number INTEGER,
                exercise_number TEXT,
                text TEXT NOT NULL,
                has_images BOOLEAN DEFAULT 0,
                image_paths TEXT,
                latex_content TEXT,
                difficulty TEXT,
                variations TEXT,
                solution TEXT,
                analyzed BOOLEAN DEFAULT 0,
                analysis_metadata TEXT,
                low_confidence_skipped BOOLEAN DEFAULT 0,
                tags TEXT,
                exercise_type TEXT DEFAULT 'procedural'
                    CHECK(exercise_type IN ('procedural', 'theory', 'proof', 'hybrid')),
                theory_metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (course_code) REFERENCES courses(code),
                FOREIGN KEY (topic_id) REFERENCES topics(id),
                FOREIGN KEY (knowledge_item_id) REFERENCES knowledge_items(id)
            )
        """)

        # Student progress table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS student_progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                course_code TEXT NOT NULL,
                knowledge_item_id TEXT NOT NULL,
                total_attempts INTEGER DEFAULT 0,
                correct_attempts INTEGER DEFAULT 0,
                mastery_score REAL DEFAULT 0.0,
                last_practiced TIMESTAMP,
                next_review TIMESTAMP,
                review_interval INTEGER DEFAULT 1,
                common_mistakes TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (course_code) REFERENCES courses(code),
                FOREIGN KEY (knowledge_item_id) REFERENCES knowledge_items(id),
                UNIQUE(course_code, knowledge_item_id)
            )
        """)

        # Quiz sessions table (Phase 5 schema)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS quiz_sessions (
                id TEXT PRIMARY KEY,
                course_code TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                total_questions INTEGER,
                correct_answers INTEGER,
                score_percentage REAL,
                quiz_type TEXT,
                filter_topic_id INTEGER,
                filter_knowledge_item_id TEXT,
                filter_difficulty TEXT,
                FOREIGN KEY (course_code) REFERENCES courses(code),
                FOREIGN KEY (filter_topic_id) REFERENCES topics(id)
            )
        """)

        # Quiz attempts table (Phase 5 schema)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS quiz_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                exercise_id TEXT NOT NULL,
                user_answer TEXT,
                correct BOOLEAN,
                time_taken_seconds INTEGER,
                hint_used BOOLEAN DEFAULT 0,
                feedback TEXT,
                attempted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES quiz_sessions(id),
                FOREIGN KEY (exercise_id) REFERENCES exercises(id)
            )
        """)

        # Exercise reviews table (Phase 5 - Spaced Repetition)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS exercise_reviews (
                exercise_id TEXT PRIMARY KEY,
                course_code TEXT NOT NULL,
                easiness_factor REAL DEFAULT 2.5,
                repetition_number INTEGER DEFAULT 0,
                interval_days INTEGER DEFAULT 0,
                next_review_date DATE,
                last_reviewed_at TIMESTAMP,
                total_reviews INTEGER DEFAULT 0,
                correct_reviews INTEGER DEFAULT 0,
                mastery_level TEXT DEFAULT 'new',
                FOREIGN KEY (exercise_id) REFERENCES exercises(id),
                FOREIGN KEY (course_code) REFERENCES courses(code)
            )
        """)

        # Topic mastery table (Phase 5 - Progress Tracking)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS topic_mastery (
                topic_id INTEGER PRIMARY KEY,
                course_code TEXT NOT NULL,
                exercises_total INTEGER DEFAULT 0,
                exercises_mastered INTEGER DEFAULT 0,
                mastery_percentage REAL DEFAULT 0.0,
                last_practiced_at TIMESTAMP,
                FOREIGN KEY (topic_id) REFERENCES topics(id),
                FOREIGN KEY (course_code) REFERENCES courses(code)
            )
        """)

        # Generated exercises table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS generated_exercises (
                id TEXT PRIMARY KEY,
                course_code TEXT NOT NULL,
                knowledge_item_id TEXT NOT NULL,
                based_on_exercise_ids TEXT,
                difficulty TEXT,
                variations TEXT,
                text TEXT NOT NULL,
                solution_outline TEXT,
                common_mistakes TEXT,
                times_used INTEGER DEFAULT 0,
                avg_student_score REAL,
                flagged_for_review BOOLEAN DEFAULT 0,
                generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (course_code) REFERENCES courses(code),
                FOREIGN KEY (knowledge_item_id) REFERENCES knowledge_items(id)
            )
        """)

        # Exercise-KnowledgeItem junction table (many-to-many)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS exercise_knowledge_items (
                exercise_id TEXT NOT NULL,
                knowledge_item_id TEXT NOT NULL,
                step_number INTEGER,
                PRIMARY KEY (exercise_id, knowledge_item_id),
                FOREIGN KEY (exercise_id) REFERENCES exercises(id) ON DELETE CASCADE,
                FOREIGN KEY (knowledge_item_id) REFERENCES knowledge_items(id) ON DELETE CASCADE
            )
        """)

        # Procedure cache table (Option 3: Pattern Caching)
        # Web-ready: user_id nullable for CLI mode, included in unique constraint for multi-tenant isolation
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS procedure_cache_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,                        -- NULL for CLI/global, set for web multi-tenant
                course_code TEXT,
                pattern_hash TEXT NOT NULL,
                exercise_text_sample TEXT,
                topic TEXT,
                difficulty TEXT,
                variations_json TEXT,
                procedures_json TEXT NOT NULL,
                embedding BLOB,
                normalized_text TEXT,
                match_count INTEGER DEFAULT 0,
                confidence_avg REAL DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_matched_at TIMESTAMP,
                UNIQUE(user_id, course_code, pattern_hash)  -- Multi-tenant safe unique constraint
            )
        """)

    def _create_indexes(self):
        """Create database indexes for performance."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_exercises_course ON exercises(course_code)",
            "CREATE INDEX IF NOT EXISTS idx_exercises_knowledge_item ON exercises(knowledge_item_id)",
            "CREATE INDEX IF NOT EXISTS idx_exercises_topic ON exercises(topic_id)",
            "CREATE INDEX IF NOT EXISTS idx_topics_course ON topics(course_code)",
            "CREATE INDEX IF NOT EXISTS idx_knowledge_items_topic ON knowledge_items(topic_id)",
            "CREATE INDEX IF NOT EXISTS idx_progress_course ON student_progress(course_code)",
            "CREATE INDEX IF NOT EXISTS idx_progress_knowledge_item ON student_progress(knowledge_item_id)",
            "CREATE INDEX IF NOT EXISTS idx_quiz_sessions_course ON quiz_sessions(course_code)",
            "CREATE INDEX IF NOT EXISTS idx_quiz_attempts_session ON quiz_attempts(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_quiz_attempts_exercise ON quiz_attempts(exercise_id)",
            "CREATE INDEX IF NOT EXISTS idx_exercise_reviews_course ON exercise_reviews(course_code)",
            "CREATE INDEX IF NOT EXISTS idx_exercise_reviews_next_review ON exercise_reviews(next_review_date)",
            "CREATE INDEX IF NOT EXISTS idx_topic_mastery_course ON topic_mastery(course_code)",
            "CREATE INDEX IF NOT EXISTS idx_exercise_knowledge_items_exercise ON exercise_knowledge_items(exercise_id)",
            "CREATE INDEX IF NOT EXISTS idx_exercise_knowledge_items_knowledge_item ON exercise_knowledge_items(knowledge_item_id)",
            "CREATE INDEX IF NOT EXISTS idx_proc_cache_course ON procedure_cache_entries(course_code)",
            "CREATE INDEX IF NOT EXISTS idx_proc_cache_hash ON procedure_cache_entries(pattern_hash)",
            "CREATE INDEX IF NOT EXISTS idx_proc_cache_user ON procedure_cache_entries(user_id)",  # Web-ready
        ]

        for index_sql in indexes:
            self.conn.execute(index_sql)

    # Course operations
    def add_course(
        self,
        code: str,
        name: str,
        original_name: str = None,
        acronym: str = None,
        degree_level: str = None,
        degree_program: str = None,
    ):
        """Add a new course to the database.

        Args:
            code: Course code (e.g., "B006802")
            name: English course name
            original_name: Original language name (Italian)
            acronym: Course acronym
            degree_level: "bachelor" or "master"
            degree_program: "L-31" or "LM-18"
        """
        self.conn.execute(
            """
            INSERT OR IGNORE INTO courses
            (code, name, original_name, acronym, degree_level, degree_program)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (code, name, original_name, acronym, degree_level, degree_program),
        )

    def get_course(self, code: str) -> Optional[Dict[str, Any]]:
        """Get course information by code."""
        cursor = self.conn.execute("SELECT * FROM courses WHERE code = ?", (code,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_all_courses(self) -> List[Dict[str, Any]]:
        """Get all courses."""
        cursor = self.conn.execute("SELECT * FROM courses ORDER BY name")
        return [dict(row) for row in cursor.fetchall()]

    # Topic operations
    def add_topic(
        self, course_code: str, name: str, description: str = None, language: str = None
    ) -> int:
        """Add a new topic to a course.

        Args:
            course_code: Course code
            name: Topic name
            description: Topic description (optional)
            language: Language name (lowercase, e.g., "english", "italian") - optional

        Returns:
            Topic ID
        """
        cursor = self.conn.execute(
            """
            INSERT OR IGNORE INTO topics (course_code, name, description, language)
            VALUES (?, ?, ?, ?)
        """,
            (course_code, name, description, language),
        )

        if cursor.lastrowid == 0:
            # Topic already exists, fetch its ID
            cursor = self.conn.execute(
                "SELECT id FROM topics WHERE course_code = ? AND name = ?", (course_code, name)
            )
            return cursor.fetchone()[0]
        return cursor.lastrowid

    def get_topics_by_course(self, course_code: str) -> List[Dict[str, Any]]:
        """Get all topics for a course."""
        cursor = self.conn.execute(
            """
            SELECT * FROM topics
            WHERE course_code = ?
            ORDER BY name
        """,
            (course_code,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def split_topic(
        self,
        old_topic_id: int,
        clusters: List[Dict[str, Any]],
        course_code: str,
        delete_old: bool = False,
    ) -> Dict[str, Any]:
        """Split a generic topic into multiple specific topics.

        Args:
            old_topic_id: ID of topic to split
            clusters: List of dicts with 'topic_name' and 'knowledge_item_ids' keys
            course_code: Course code for the topic
            delete_old: Whether to delete old topic if empty (default: False)

        Returns:
            Dict with split statistics and new topic IDs

        Example clusters:
            [
                {
                    "topic_name": "Autovalori e Diagonalizzazione",
                    "knowledge_item_ids": ["loop1", "loop2", ...]
                },
                ...
            ]
        """
        try:
            # Get original topic info for logging
            cursor = self.conn.execute("SELECT name FROM topics WHERE id = ?", (old_topic_id,))
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Topic ID {old_topic_id} not found")

            old_topic_name = row[0]
            stats = {
                "old_topic_id": old_topic_id,
                "old_topic_name": old_topic_name,
                "new_topics": [],
                "knowledge_items_moved": 0,
                "errors": [],
            }

            # Process each cluster
            for cluster in clusters:
                topic_name = cluster.get("topic_name")
                knowledge_item_ids = cluster.get("knowledge_item_ids", [])

                if not topic_name or not knowledge_item_ids:
                    stats["errors"].append(f"Invalid cluster: {cluster}")
                    continue

                # Create new topic
                new_topic_id = self.add_topic(course_code, topic_name)

                # Move core loops to new topic
                moved_count = 0
                for loop_id in knowledge_item_ids:
                    try:
                        self.conn.execute(
                            """
                            UPDATE knowledge_items
                            SET topic_id = ?, updated_at = CURRENT_TIMESTAMP
                            WHERE id = ?
                        """,
                            (new_topic_id, loop_id),
                        )
                        moved_count += 1
                    except Exception as e:
                        stats["errors"].append(f"Failed to move {loop_id}: {e}")

                stats["new_topics"].append(
                    {"id": new_topic_id, "name": topic_name, "knowledge_items_moved": moved_count}
                )
                stats["knowledge_items_moved"] += moved_count

            # Optionally delete old topic if no core loops remain
            if delete_old:
                cursor = self.conn.execute(
                    """
                    SELECT COUNT(*) FROM knowledge_items WHERE topic_id = ?
                """,
                    (old_topic_id,),
                )
                remaining_loops = cursor.fetchone()[0]

                if remaining_loops == 0:
                    self.conn.execute("DELETE FROM topics WHERE id = ?", (old_topic_id,))
                    stats["old_topic_deleted"] = True
                else:
                    stats["old_topic_deleted"] = False
                    stats["remaining_knowledge_items"] = remaining_loops

            self.conn.commit()
            return stats

        except Exception as e:
            self.conn.rollback()
            raise RuntimeError(f"Topic split failed: {e}")

    # Core loop operations
    def add_knowledge_item(
        self,
        loop_id: str,
        topic_id: int,
        name: str,
        procedure: List[str],
        description: str = None,
        language: str = None,
    ) -> str:
        """Add a new core loop.

        Args:
            loop_id: Unique identifier for the core loop
            topic_id: Parent topic ID
            name: Name of the core loop
            procedure: List of procedure steps
            description: Optional description
            language: Language name (lowercase, e.g., "english", "italian") - optional

        Returns:
            Core loop ID
        """
        procedure_json = json.dumps(procedure)
        self.conn.execute(
            """
            INSERT OR REPLACE INTO knowledge_items
            (id, topic_id, name, description, procedure, language, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
            (loop_id, topic_id, name, description, procedure_json, language),
        )
        return loop_id

    def get_knowledge_item(self, loop_id: str) -> Optional[Dict[str, Any]]:
        """Get core loop by ID."""
        cursor = self.conn.execute("SELECT * FROM knowledge_items WHERE id = ?", (loop_id,))
        row = cursor.fetchone()
        if row:
            result = dict(row)
            result["procedure"] = json.loads(result["procedure"])
            return result
        return None

    def get_knowledge_items_by_topic(self, topic_id: int) -> List[Dict[str, Any]]:
        """Get all core loops for a topic."""
        cursor = self.conn.execute(
            """
            SELECT * FROM knowledge_items
            WHERE topic_id = ?
            ORDER BY name
        """,
            (topic_id,),
        )
        results = []
        for row in cursor.fetchall():
            result = dict(row)
            result["procedure"] = json.loads(result["procedure"])
            results.append(result)
        return results

    def get_knowledge_items_by_course(self, course_code: str) -> List[Dict[str, Any]]:
        """Get all core loops for a course."""
        cursor = self.conn.execute(
            """
            SELECT cl.* FROM knowledge_items cl
            JOIN topics t ON cl.topic_id = t.id
            WHERE t.course_code = ?
            ORDER BY cl.name
        """,
            (course_code,),
        )
        results = []
        for row in cursor.fetchall():
            result = dict(row)
            result["procedure"] = json.loads(result["procedure"])
            results.append(result)
        return results

    def update_knowledge_item_stats(self, loop_id: str):
        """Update exercise count and average difficulty for a core loop."""
        cursor = self.conn.execute(
            """
            SELECT COUNT(*) as count, AVG(
                CASE difficulty
                    WHEN 'easy' THEN 1
                    WHEN 'medium' THEN 2
                    WHEN 'hard' THEN 3
                    ELSE 2
                END
            ) as avg_diff
            FROM exercises
            WHERE knowledge_item_id = ?
        """,
            (loop_id,),
        )

        row = cursor.fetchone()
        if row:
            self.conn.execute(
                """
                UPDATE knowledge_items
                SET exercise_count = ?, difficulty_avg = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """,
                (row[0], row[1] or 0.0, loop_id),
            )

    # Exercise operations
    def add_exercise(self, exercise_data: Dict[str, Any]) -> str:
        """Add a new exercise.

        Args:
            exercise_data: Dictionary with exercise information

        Returns:
            Exercise ID
        """
        # Convert lists/dicts to JSON strings
        if "image_paths" in exercise_data and isinstance(exercise_data["image_paths"], list):
            exercise_data["image_paths"] = json.dumps(exercise_data["image_paths"])
        if "variations" in exercise_data and isinstance(exercise_data["variations"], list):
            exercise_data["variations"] = json.dumps(exercise_data["variations"])
        if "analysis_metadata" in exercise_data and isinstance(
            exercise_data["analysis_metadata"], dict
        ):
            exercise_data["analysis_metadata"] = json.dumps(exercise_data["analysis_metadata"])

        self.conn.execute(
            """
            INSERT INTO exercises
            (id, course_code, topic_id, knowledge_item_id, source_pdf, page_number,
             exercise_number, text, has_images, image_paths, latex_content,
             difficulty, variations, solution, analyzed, analysis_metadata)
            VALUES
            (:id, :course_code, :topic_id, :knowledge_item_id, :source_pdf, :page_number,
             :exercise_number, :text, :has_images, :image_paths, :latex_content,
             :difficulty, :variations, :solution, :analyzed, :analysis_metadata)
        """,
            exercise_data,
        )

        return exercise_data["id"]

    def get_exercise(self, exercise_id: str) -> Optional[Dict[str, Any]]:
        """Get exercise by ID."""
        cursor = self.conn.execute("SELECT * FROM exercises WHERE id = ?", (exercise_id,))
        row = cursor.fetchone()
        if row:
            result = dict(row)
            # Parse JSON fields
            if result.get("image_paths"):
                result["image_paths"] = json.loads(result["image_paths"])
            if result.get("variations"):
                result["variations"] = json.loads(result["variations"])
            if result.get("analysis_metadata"):
                result["analysis_metadata"] = json.loads(result["analysis_metadata"])
            return result
        return None

    def get_exercises_by_knowledge_item(self, knowledge_item_id: str) -> List[Dict[str, Any]]:
        """Get all exercises that include this core loop.

        Uses the junction table for many-to-many relationships.
        For backward compatibility, also checks the legacy knowledge_item_id column.
        """
        # Check if created_at column exists for ordering
        cursor = self.conn.execute("PRAGMA table_info(exercises)")
        columns = [row[1] for row in cursor.fetchall()]
        order_by = "e.created_at" if "created_at" in columns else "e.id"

        cursor = self.conn.execute(
            f"""
            SELECT DISTINCT e.*, ecl.step_number
            FROM exercises e
            LEFT JOIN exercise_knowledge_items ecl ON e.id = ecl.exercise_id
            WHERE ecl.knowledge_item_id = ? OR e.knowledge_item_id = ?
            ORDER BY {order_by}
        """,
            (knowledge_item_id, knowledge_item_id),
        )

        results = []
        for row in cursor.fetchall():
            result = dict(row)
            if result.get("image_paths"):
                result["image_paths"] = json.loads(result["image_paths"])
            if result.get("variations"):
                result["variations"] = json.loads(result["variations"])
            if result.get("analysis_metadata"):
                result["analysis_metadata"] = json.loads(result["analysis_metadata"])
            results.append(result)
        return results

    def get_exercises_by_course(
        self, course_code: str, analyzed_only: bool = False, unanalyzed_only: bool = False
    ) -> List[Dict[str, Any]]:
        """Get all exercises for a course.

        Args:
            course_code: Course code to filter by
            analyzed_only: If True, return only analyzed exercises
            unanalyzed_only: If True, return only unanalyzed exercises

        Returns:
            List of exercise dictionaries
        """
        query = """
            SELECT * FROM exercises
            WHERE course_code = ?
        """

        if analyzed_only:
            query += " AND analyzed = 1"
        elif unanalyzed_only:
            query += " AND analyzed = 0"

        query += " ORDER BY created_at"

        cursor = self.conn.execute(query, (course_code,))

        results = []
        for row in cursor.fetchall():
            result = dict(row)
            if result.get("image_paths"):
                result["image_paths"] = json.loads(result["image_paths"])
            if result.get("variations"):
                result["variations"] = json.loads(result["variations"])
            if result.get("analysis_metadata"):
                result["analysis_metadata"] = json.loads(result["analysis_metadata"])
            results.append(result)
        return results

    # Phase 5: Quiz Session operations
    def create_quiz_session(
        self,
        session_id: str,
        course_code: str,
        quiz_type: str,
        filter_topic_id: Optional[int] = None,
        filter_knowledge_item_id: Optional[str] = None,
        filter_difficulty: Optional[str] = None,
    ) -> str:
        """Create a new quiz session.

        Args:
            session_id: Unique identifier for the session
            course_code: Course code
            quiz_type: Type of quiz ('topic', 'knowledge_item', 'random', 'review')
            filter_topic_id: Optional topic filter
            filter_knowledge_item_id: Optional core loop filter
            filter_difficulty: Optional difficulty filter

        Returns:
            Session ID
        """
        self.conn.execute(
            """
            INSERT INTO quiz_sessions
            (id, course_code, quiz_type, filter_topic_id, filter_knowledge_item_id, filter_difficulty)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                session_id,
                course_code,
                quiz_type,
                filter_topic_id,
                filter_knowledge_item_id,
                filter_difficulty,
            ),
        )
        return session_id

    def get_quiz_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get quiz session by ID.

        Args:
            session_id: Session ID to retrieve

        Returns:
            Session dictionary or None if not found
        """
        cursor = self.conn.execute("SELECT * FROM quiz_sessions WHERE id = ?", (session_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def update_quiz_session(
        self,
        session_id: str,
        total_questions: Optional[int] = None,
        correct_answers: Optional[int] = None,
        score_percentage: Optional[float] = None,
        completed: bool = False,
    ):
        """Update quiz session with results.

        Args:
            session_id: Session ID to update
            total_questions: Total number of questions
            correct_answers: Number of correct answers
            score_percentage: Final score percentage
            completed: Whether the session is completed
        """
        updates = []
        params = []

        if total_questions is not None:
            updates.append("total_questions = ?")
            params.append(total_questions)

        if correct_answers is not None:
            updates.append("correct_answers = ?")
            params.append(correct_answers)

        if score_percentage is not None:
            updates.append("score_percentage = ?")
            params.append(score_percentage)

        if completed:
            updates.append("completed_at = CURRENT_TIMESTAMP")

        if updates:
            query = f"UPDATE quiz_sessions SET {', '.join(updates)} WHERE id = ?"
            params.append(session_id)
            self.conn.execute(query, params)

    def get_quiz_sessions_by_course(
        self, course_code: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get recent quiz sessions for a course.

        Args:
            course_code: Course code to filter by
            limit: Maximum number of sessions to return

        Returns:
            List of session dictionaries
        """
        cursor = self.conn.execute(
            """
            SELECT * FROM quiz_sessions
            WHERE course_code = ?
            ORDER BY created_at DESC
            LIMIT ?
        """,
            (course_code, limit),
        )
        return [dict(row) for row in cursor.fetchall()]

    # Phase 5: Quiz Attempt operations
    def add_quiz_attempt(
        self,
        session_id: str,
        exercise_id: str,
        user_answer: Optional[str] = None,
        correct: Optional[bool] = None,
        time_taken_seconds: Optional[int] = None,
        hint_used: bool = False,
        feedback: Optional[str] = None,
    ) -> int:
        """Record a quiz attempt.

        Args:
            session_id: Session ID
            exercise_id: Exercise ID
            user_answer: User's answer
            correct: Whether the answer was correct
            time_taken_seconds: Time taken in seconds
            hint_used: Whether a hint was used
            feedback: Feedback text

        Returns:
            Attempt ID
        """
        cursor = self.conn.execute(
            """
            INSERT INTO quiz_attempts
            (session_id, exercise_id, user_answer, correct, time_taken_seconds, hint_used, feedback)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                session_id,
                exercise_id,
                user_answer,
                correct,
                time_taken_seconds,
                hint_used,
                feedback,
            ),
        )
        return cursor.lastrowid

    def get_quiz_attempts(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all attempts for a quiz session.

        Args:
            session_id: Session ID

        Returns:
            List of attempt dictionaries
        """
        cursor = self.conn.execute(
            """
            SELECT * FROM quiz_attempts
            WHERE session_id = ?
            ORDER BY attempted_at
        """,
            (session_id,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_attempts_by_exercise(self, exercise_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent attempts for a specific exercise.

        Args:
            exercise_id: Exercise ID
            limit: Maximum number of attempts to return

        Returns:
            List of attempt dictionaries
        """
        cursor = self.conn.execute(
            """
            SELECT * FROM quiz_attempts
            WHERE exercise_id = ?
            ORDER BY attempted_at DESC
            LIMIT ?
        """,
            (exercise_id, limit),
        )
        return [dict(row) for row in cursor.fetchall()]

    # Phase 5: Exercise Review operations (Spaced Repetition)
    def get_exercise_review(self, exercise_id: str) -> Optional[Dict[str, Any]]:
        """Get spaced repetition data for an exercise.

        Args:
            exercise_id: Exercise ID

        Returns:
            Review data dictionary or None if not found
        """
        cursor = self.conn.execute(
            "SELECT * FROM exercise_reviews WHERE exercise_id = ?", (exercise_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def update_exercise_review(
        self,
        exercise_id: str,
        course_code: str,
        easiness_factor: float,
        repetition_number: int,
        interval_days: int,
        next_review_date: str,
        mastery_level: str,
        correct: bool,
    ):
        """Update spaced repetition data for an exercise.

        Args:
            exercise_id: Exercise ID
            course_code: Course code
            easiness_factor: SM-2 easiness factor
            repetition_number: Number of successful repetitions
            interval_days: Days until next review
            next_review_date: Date of next review (YYYY-MM-DD)
            mastery_level: Mastery level ('new', 'learning', 'mastered')
            correct: Whether the last review was correct
        """
        # Check if review record exists
        existing = self.get_exercise_review(exercise_id)

        if existing:
            # Update existing record
            self.conn.execute(
                """
                UPDATE exercise_reviews
                SET easiness_factor = ?,
                    repetition_number = ?,
                    interval_days = ?,
                    next_review_date = ?,
                    last_reviewed_at = CURRENT_TIMESTAMP,
                    total_reviews = total_reviews + 1,
                    correct_reviews = correct_reviews + ?,
                    mastery_level = ?
                WHERE exercise_id = ?
            """,
                (
                    easiness_factor,
                    repetition_number,
                    interval_days,
                    next_review_date,
                    1 if correct else 0,
                    mastery_level,
                    exercise_id,
                ),
            )
        else:
            # Create new record
            self.conn.execute(
                """
                INSERT INTO exercise_reviews
                (exercise_id, course_code, easiness_factor, repetition_number, interval_days,
                 next_review_date, last_reviewed_at, total_reviews, correct_reviews, mastery_level)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, 1, ?, ?)
            """,
                (
                    exercise_id,
                    course_code,
                    easiness_factor,
                    repetition_number,
                    interval_days,
                    next_review_date,
                    1 if correct else 0,
                    mastery_level,
                ),
            )

    def get_exercises_due_for_review(
        self, course_code: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get exercises that are due for review.

        Args:
            course_code: Course code to filter by
            limit: Maximum number of exercises to return

        Returns:
            List of exercise review dictionaries
        """
        cursor = self.conn.execute(
            """
            SELECT * FROM exercise_reviews
            WHERE course_code = ?
            AND next_review_date <= DATE('now')
            ORDER BY next_review_date
            LIMIT ?
        """,
            (course_code, limit),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_exercises_by_mastery(
        self, course_code: str, mastery_level: str
    ) -> List[Dict[str, Any]]:
        """Get exercises filtered by mastery level.

        Args:
            course_code: Course code to filter by
            mastery_level: Mastery level ('new', 'learning', 'mastered')

        Returns:
            List of exercise review dictionaries
        """
        cursor = self.conn.execute(
            """
            SELECT * FROM exercise_reviews
            WHERE course_code = ? AND mastery_level = ?
            ORDER BY last_reviewed_at DESC
        """,
            (course_code, mastery_level),
        )
        return [dict(row) for row in cursor.fetchall()]

    # Phase 5: Topic Mastery operations
    def get_topic_mastery(self, topic_id: int) -> Optional[Dict[str, Any]]:
        """Get mastery data for a topic.

        Args:
            topic_id: Topic ID

        Returns:
            Mastery data dictionary or None if not found
        """
        cursor = self.conn.execute("SELECT * FROM topic_mastery WHERE topic_id = ?", (topic_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def update_topic_mastery(
        self, topic_id: int, course_code: str, exercises_total: int, exercises_mastered: int
    ):
        """Update mastery data for a topic.

        Args:
            topic_id: Topic ID
            course_code: Course code
            exercises_total: Total number of exercises in topic
            exercises_mastered: Number of mastered exercises
        """
        mastery_percentage = (
            (exercises_mastered / exercises_total * 100.0) if exercises_total > 0 else 0.0
        )

        # Check if mastery record exists
        existing = self.get_topic_mastery(topic_id)

        if existing:
            # Update existing record
            self.conn.execute(
                """
                UPDATE topic_mastery
                SET exercises_total = ?,
                    exercises_mastered = ?,
                    mastery_percentage = ?,
                    last_practiced_at = CURRENT_TIMESTAMP
                WHERE topic_id = ?
            """,
                (exercises_total, exercises_mastered, mastery_percentage, topic_id),
            )
        else:
            # Create new record
            self.conn.execute(
                """
                INSERT INTO topic_mastery
                (topic_id, course_code, exercises_total, exercises_mastered,
                 mastery_percentage, last_practiced_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
                (topic_id, course_code, exercises_total, exercises_mastered, mastery_percentage),
            )

    def get_all_topic_mastery(self, course_code: str) -> List[Dict[str, Any]]:
        """Get mastery data for all topics in a course.

        Args:
            course_code: Course code to filter by

        Returns:
            List of topic mastery dictionaries with topic names
        """
        cursor = self.conn.execute(
            """
            SELECT tm.*, t.name as topic_name
            FROM topic_mastery tm
            JOIN topics t ON tm.topic_id = t.id
            WHERE tm.course_code = ?
            ORDER BY tm.mastery_percentage DESC
        """,
            (course_code,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def recalculate_topic_mastery(self, topic_id: int, course_code: str):
        """Recalculate mastery percentage for a topic based on exercise reviews.

        Args:
            topic_id: Topic ID
            course_code: Course code
        """
        # Count total exercises for this topic
        cursor = self.conn.execute(
            """
            SELECT COUNT(*) FROM exercises
            WHERE topic_id = ?
        """,
            (topic_id,),
        )
        exercises_total = cursor.fetchone()[0]

        # Count mastered exercises (those with mastery_level = 'mastered')
        cursor = self.conn.execute(
            """
            SELECT COUNT(*) FROM exercise_reviews er
            JOIN exercises e ON er.exercise_id = e.id
            WHERE e.topic_id = ? AND er.mastery_level = 'mastered'
        """,
            (topic_id,),
        )
        exercises_mastered = cursor.fetchone()[0]

        # Update topic mastery
        self.update_topic_mastery(topic_id, course_code, exercises_total, exercises_mastered)

    # Multi-core-loop operations
    def link_exercise_to_knowledge_item(
        self, exercise_id: str, knowledge_item_id: str, step_number: Optional[int] = None
    ) -> None:
        """Link exercise to a core loop (allows many-to-many).

        Args:
            exercise_id: Exercise ID
            knowledge_item_id: Core loop ID
            step_number: Optional step number indicating which point in exercise (1, 2, 3, etc)
        """
        self.conn.execute(
            """
            INSERT OR REPLACE INTO exercise_knowledge_items
            (exercise_id, knowledge_item_id, step_number)
            VALUES (?, ?, ?)
        """,
            (exercise_id, knowledge_item_id, step_number),
        )

    def get_exercise_knowledge_items(self, exercise_id: str) -> List[Dict]:
        """Get all core loops for an exercise.

        Args:
            exercise_id: Exercise ID

        Returns:
            List of core loop dictionaries with step_number included
        """
        cursor = self.conn.execute(
            """
            SELECT cl.*, ecl.step_number
            FROM knowledge_items cl
            JOIN exercise_knowledge_items ecl ON cl.id = ecl.knowledge_item_id
            WHERE ecl.exercise_id = ?
            ORDER BY ecl.step_number, cl.name
        """,
            (exercise_id,),
        )

        results = []
        for row in cursor.fetchall():
            result = dict(row)
            result["procedure"] = json.loads(result["procedure"])
            results.append(result)
        return results

    def get_exercises_with_multiple_procedures(self, course_code: str) -> List[Dict]:
        """Get exercises that cover multiple core loops.

        Args:
            course_code: Course code to filter by

        Returns:
            List of exercise dictionaries with a 'knowledge_item_count' field
        """
        # Check if created_at column exists for ordering
        cursor = self.conn.execute("PRAGMA table_info(exercises)")
        columns = [row[1] for row in cursor.fetchall()]
        order_by = "e.created_at" if "created_at" in columns else "e.id"

        cursor = self.conn.execute(
            f"""
            SELECT e.*, COUNT(ecl.knowledge_item_id) as knowledge_item_count
            FROM exercises e
            JOIN exercise_knowledge_items ecl ON e.id = ecl.exercise_id
            WHERE e.course_code = ?
            GROUP BY e.id
            HAVING knowledge_item_count > 1
            ORDER BY knowledge_item_count DESC, {order_by}
        """,
            (course_code,),
        )

        results = []
        for row in cursor.fetchall():
            result = dict(row)
            # Parse JSON fields
            if result.get("image_paths"):
                result["image_paths"] = json.loads(result["image_paths"])
            if result.get("variations"):
                result["variations"] = json.loads(result["variations"])
            if result.get("analysis_metadata"):
                result["analysis_metadata"] = json.loads(result["analysis_metadata"])
            results.append(result)
        return results

    def get_exercises_by_procedure_type(
        self, course_code: str, procedure_type: str
    ) -> List[Dict[str, Any]]:
        """Get exercises by procedure type (design, transformation, etc.).

        Args:
            course_code: Course code to filter by
            procedure_type: Type of procedure (design, transformation, verification, minimization, analysis, implementation)

        Returns:
            List of exercise dictionaries that include this procedure type in their tags
        """
        # Check if created_at column exists for ordering
        cursor = self.conn.execute("PRAGMA table_info(exercises)")
        columns = [row[1] for row in cursor.fetchall()]
        order_by = "e.created_at" if "created_at" in columns else "e.id"

        cursor = self.conn.execute(
            f"""
            SELECT DISTINCT e.*
            FROM exercises e
            WHERE e.course_code = ?
            AND e.tags LIKE ?
            ORDER BY {order_by} DESC
        """,
            (course_code, f'%"{procedure_type}"%'),
        )

        results = []
        for row in cursor.fetchall():
            result = dict(row)
            if result.get("image_paths"):
                result["image_paths"] = json.loads(result["image_paths"])
            if result.get("variations"):
                result["variations"] = json.loads(result["variations"])
            if result.get("analysis_metadata"):
                result["analysis_metadata"] = json.loads(result["analysis_metadata"])
            if result.get("tags"):
                result["tags"] = json.loads(result["tags"])
            results.append(result)
        return results

    def get_knowledge_items_for_exercise(self, exercise_id: str) -> List[Dict[str, Any]]:
        """Get all core loops associated with an exercise.

        Args:
            exercise_id: Exercise ID

        Returns:
            List of core loop dictionaries with step_number information
        """
        cursor = self.conn.execute(
            """
            SELECT cl.*, ecl.step_number
            FROM knowledge_items cl
            JOIN exercise_knowledge_items ecl ON cl.id = ecl.knowledge_item_id
            WHERE ecl.exercise_id = ?
            ORDER BY ecl.step_number
        """,
            (exercise_id,),
        )

        return [dict(row) for row in cursor.fetchall()]

    def update_exercise_tags(self, exercise_id: str, tags: List[str]):
        """Update tags for an exercise.

        Args:
            exercise_id: Exercise ID
            tags: List of tag strings
        """
        tags_json = json.dumps(tags) if tags else None
        self.conn.execute(
            """
            UPDATE exercises
            SET tags = ?
            WHERE id = ?
        """,
            (tags_json, exercise_id),
        )

    def update_exercise_analysis(
        self,
        exercise_id: str,
        topic_id: Optional[int] = None,
        knowledge_item_id: Optional[str] = None,
        difficulty: Optional[str] = None,
        variations: Optional[List[str]] = None,
        analysis_metadata: Optional[Dict[str, Any]] = None,
        analyzed: bool = True,
        low_confidence_skipped: bool = False,
    ):
        """Update exercise with analysis results.

        Args:
            exercise_id: Exercise ID
            topic_id: Topic ID (optional)
            knowledge_item_id: Primary core loop ID (optional, for backward compatibility)
            difficulty: Difficulty level (optional)
            variations: List of variations (optional)
            analysis_metadata: Additional metadata (optional)
            analyzed: Mark as analyzed
            low_confidence_skipped: Mark as skipped due to low confidence
        """
        updates = ["analyzed = ?"]
        params = [1 if analyzed else 0]

        if topic_id is not None:
            updates.append("topic_id = ?")
            params.append(topic_id)

        if knowledge_item_id is not None:
            updates.append("knowledge_item_id = ?")
            params.append(knowledge_item_id)

        if difficulty is not None:
            updates.append("difficulty = ?")
            params.append(difficulty)

        if variations is not None:
            updates.append("variations = ?")
            params.append(json.dumps(variations))

        if analysis_metadata is not None:
            updates.append("analysis_metadata = ?")
            params.append(json.dumps(analysis_metadata))

        if low_confidence_skipped:
            updates.append("low_confidence_skipped = ?")
            params.append(1)

        query = f"UPDATE exercises SET {', '.join(updates)} WHERE id = ?"
        params.append(exercise_id)
        self.conn.execute(query, params)

    # Search operations for Phase 6.5
    def get_exercises_by_tag(self, course_code: str, tag: str) -> List[Dict[str, Any]]:
        """Get exercises that have a specific tag.

        Args:
            course_code: Course code to filter by
            tag: Tag to search for (e.g., 'design', 'transformation', 'transform_mealy_to_moore')

        Returns:
            List of exercise dictionaries
        """
        cursor = self.conn.execute(
            """
            SELECT * FROM exercises
            WHERE course_code = ? AND tags LIKE ?
            ORDER BY created_at DESC
        """,
            (course_code, f"%{tag}%"),
        )

        results = []
        for row in cursor.fetchall():
            result = dict(row)
            if result.get("image_paths"):
                result["image_paths"] = json.loads(result["image_paths"])
            if result.get("variations"):
                result["variations"] = json.loads(result["variations"])
            if result.get("analysis_metadata"):
                result["analysis_metadata"] = json.loads(result["analysis_metadata"])
            if result.get("tags"):
                result["tags"] = json.loads(result["tags"])
            results.append(result)
        return results

    def search_exercises_by_text(self, course_code: str, search_text: str) -> List[Dict[str, Any]]:
        """Search exercises by text content.

        Args:
            course_code: Course code to filter by
            search_text: Text to search for in exercise content

        Returns:
            List of exercise dictionaries
        """
        cursor = self.conn.execute(
            """
            SELECT * FROM exercises
            WHERE course_code = ? AND (
                text LIKE ? OR
                exercise_number LIKE ? OR
                source_pdf LIKE ?
            )
            ORDER BY created_at DESC
        """,
            (course_code, f"%{search_text}%", f"%{search_text}%", f"%{search_text}%"),
        )

        results = []
        for row in cursor.fetchall():
            result = dict(row)
            if result.get("image_paths"):
                result["image_paths"] = json.loads(result["image_paths"])
            if result.get("variations"):
                result["variations"] = json.loads(result["variations"])
            if result.get("analysis_metadata"):
                result["analysis_metadata"] = json.loads(result["analysis_metadata"])
            if result.get("tags"):
                result["tags"] = json.loads(result["tags"])
            results.append(result)
        return results

    # Phase 9.2: Theory Concept operations
    def add_theory_concept(
        self,
        concept_id: str,
        course_code: str,
        name: str,
        category: Optional[str] = None,
        topic_id: Optional[int] = None,
        description: Optional[str] = None,
        prerequisite_concept_ids: Optional[List[str]] = None,
        related_concept_ids: Optional[List[str]] = None,
    ) -> str:
        """Add a new theory concept.

        Args:
            concept_id: Unique identifier for the concept
            course_code: Course code
            name: Concept name
            category: Category (definition, theorem, axiom, property, etc.)
            topic_id: Optional topic ID
            description: Optional description
            prerequisite_concept_ids: List of prerequisite concept IDs
            related_concept_ids: List of related concept IDs

        Returns:
            Concept ID
        """
        prereq_json = json.dumps(prerequisite_concept_ids) if prerequisite_concept_ids else None
        related_json = json.dumps(related_concept_ids) if related_concept_ids else None

        self.conn.execute(
            """
            INSERT OR REPLACE INTO theory_concepts
            (id, course_code, topic_id, name, category, description,
             prerequisite_concept_ids, related_concept_ids)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                concept_id,
                course_code,
                topic_id,
                name,
                category,
                description,
                prereq_json,
                related_json,
            ),
        )
        return concept_id

    def get_theory_concept(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """Get theory concept by ID."""
        cursor = self.conn.execute("SELECT * FROM theory_concepts WHERE id = ?", (concept_id,))
        row = cursor.fetchone()
        if row:
            result = dict(row)
            if result.get("prerequisite_concept_ids"):
                result["prerequisite_concept_ids"] = json.loads(result["prerequisite_concept_ids"])
            if result.get("related_concept_ids"):
                result["related_concept_ids"] = json.loads(result["related_concept_ids"])
            return result
        return None

    def get_theory_concepts_by_course(self, course_code: str) -> List[Dict[str, Any]]:
        """Get all theory concepts for a course."""
        cursor = self.conn.execute(
            """
            SELECT * FROM theory_concepts
            WHERE course_code = ?
            ORDER BY name
        """,
            (course_code,),
        )
        results = []
        for row in cursor.fetchall():
            result = dict(row)
            if result.get("prerequisite_concept_ids"):
                result["prerequisite_concept_ids"] = json.loads(result["prerequisite_concept_ids"])
            if result.get("related_concept_ids"):
                result["related_concept_ids"] = json.loads(result["related_concept_ids"])
            results.append(result)
        return results

    def get_theory_concepts_by_category(
        self, course_code: str, category: str
    ) -> List[Dict[str, Any]]:
        """Get theory concepts filtered by category."""
        cursor = self.conn.execute(
            """
            SELECT * FROM theory_concepts
            WHERE course_code = ? AND category = ?
            ORDER BY name
        """,
            (course_code, category),
        )
        results = []
        for row in cursor.fetchall():
            result = dict(row)
            if result.get("prerequisite_concept_ids"):
                result["prerequisite_concept_ids"] = json.loads(result["prerequisite_concept_ids"])
            if result.get("related_concept_ids"):
                result["related_concept_ids"] = json.loads(result["related_concept_ids"])
            results.append(result)
        return results

    def update_exercise_theory_metadata(
        self,
        exercise_id: str,
        exercise_type: Optional[str] = None,
        theory_category: Optional[str] = None,
        theorem_name: Optional[str] = None,
        concept_id: Optional[str] = None,
        prerequisite_concepts: Optional[List[str]] = None,
        theory_metadata: Optional[Dict[str, Any]] = None,
    ):
        """Update exercise with theory metadata.

        Args:
            exercise_id: Exercise ID
            exercise_type: Exercise type (procedural, theory, proof, hybrid)
            theory_category: Theory category (definition, theorem, proof, explanation, etc.)
            theorem_name: Name of theorem (if applicable)
            concept_id: Main concept ID
            prerequisite_concepts: List of prerequisite concept IDs
            theory_metadata: Additional theory metadata as dict
        """
        updates = []
        params = []

        if exercise_type is not None:
            updates.append("exercise_type = ?")
            params.append(exercise_type)

        if theory_category is not None:
            updates.append("theory_category = ?")
            params.append(theory_category)

        if theorem_name is not None:
            updates.append("theorem_name = ?")
            params.append(theorem_name)

        if concept_id is not None:
            updates.append("concept_id = ?")
            params.append(concept_id)

        if prerequisite_concepts is not None:
            updates.append("prerequisite_concepts = ?")
            params.append(json.dumps(prerequisite_concepts))

        if theory_metadata is not None:
            updates.append("theory_metadata = ?")
            params.append(json.dumps(theory_metadata))

        if updates:
            query = f"UPDATE exercises SET {', '.join(updates)} WHERE id = ?"
            params.append(exercise_id)
            self.conn.execute(query, params)

    def get_exercises_by_theory_category(
        self, course_code: str, theory_category: str
    ) -> List[Dict[str, Any]]:
        """Get exercises filtered by theory category.

        Args:
            course_code: Course code to filter by
            theory_category: Theory category (definition, theorem, proof, etc.)

        Returns:
            List of exercise dictionaries
        """
        cursor = self.conn.execute(
            """
            SELECT * FROM exercises
            WHERE course_code = ? AND theory_category = ?
            ORDER BY created_at DESC
        """,
            (course_code, theory_category),
        )

        results = []
        for row in cursor.fetchall():
            result = dict(row)
            if result.get("image_paths"):
                result["image_paths"] = json.loads(result["image_paths"])
            if result.get("variations"):
                result["variations"] = json.loads(result["variations"])
            if result.get("analysis_metadata"):
                result["analysis_metadata"] = json.loads(result["analysis_metadata"])
            if result.get("theory_metadata"):
                result["theory_metadata"] = json.loads(result["theory_metadata"])
            if result.get("prerequisite_concepts"):
                result["prerequisite_concepts"] = json.loads(result["prerequisite_concepts"])
            if result.get("tags"):
                result["tags"] = json.loads(result["tags"])
            results.append(result)
        return results

    # Learning Materials Methods (Phase: Learning Materials Support)

    def store_learning_material(
        self,
        material_id: str,
        course_code: str,
        material_type: str,
        content: str,
        source_pdf: str,
        page_number: int,
        title: Optional[str] = None,
        has_images: bool = False,
        image_paths: Optional[List[str]] = None,
        latex_content: Optional[str] = None,
    ):
        """Store a learning material (theory, worked example, reference).

        Note: Use link_material_to_topic() to associate with topics (many-to-many).

        Args:
            material_id: Unique ID for the material
            course_code: Course code
            material_type: Type of material ('theory', 'worked_example', 'reference')
            content: Material content
            source_pdf: Source PDF filename
            page_number: Page number
            title: Optional title
            has_images: Whether material has images
            image_paths: List of image paths
            latex_content: LaTeX content if present
        """
        image_paths_json = json.dumps(image_paths) if image_paths else None

        self.conn.execute(
            """
            INSERT OR REPLACE INTO learning_materials
            (id, course_code, material_type, title, content,
             source_pdf, page_number, has_images, image_paths, latex_content)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                material_id,
                course_code,
                material_type,
                title,
                content,
                source_pdf,
                page_number,
                has_images,
                image_paths_json,
                latex_content,
            ),
        )

    def link_material_to_topic(self, material_id: str, topic_id: int):
        """Link a learning material to a topic (many-to-many).

        Args:
            material_id: Learning material ID
            topic_id: Topic ID
        """
        self.conn.execute(
            """
            INSERT OR IGNORE INTO material_topics
            (material_id, topic_id)
            VALUES (?, ?)
        """,
            (material_id, topic_id),
        )

    def get_learning_materials_by_course(
        self, course_code: str, material_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get learning materials for a course.

        Args:
            course_code: Course code
            material_type: Optional filter by material type

        Returns:
            List of learning material dictionaries
        """
        if material_type:
            cursor = self.conn.execute(
                """
                SELECT * FROM learning_materials
                WHERE course_code = ? AND material_type = ?
                ORDER BY created_at DESC
            """,
                (course_code, material_type),
            )
        else:
            cursor = self.conn.execute(
                """
                SELECT * FROM learning_materials
                WHERE course_code = ?
                ORDER BY created_at DESC
            """,
                (course_code,),
            )

        results = []
        for row in cursor.fetchall():
            result = dict(row)
            if result.get("image_paths"):
                result["image_paths"] = json.loads(result["image_paths"])
            results.append(result)
        return results

    def get_learning_materials_by_topic(
        self, topic_id: int, material_type: Optional[str] = None, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get learning materials linked to a topic (via many-to-many join).

        Args:
            topic_id: Topic ID
            material_type: Optional filter by material type
            limit: Optional limit on number of results

        Returns:
            List of learning material dictionaries
        """
        if material_type:
            query = """
                SELECT lm.*
                FROM learning_materials lm
                JOIN material_topics mt ON lm.id = mt.material_id
                WHERE mt.topic_id = ? AND lm.material_type = ?
                ORDER BY lm.created_at DESC
            """
            if limit:
                query += f" LIMIT {limit}"
            cursor = self.conn.execute(query, (topic_id, material_type))
        else:
            query = """
                SELECT lm.*
                FROM learning_materials lm
                JOIN material_topics mt ON lm.id = mt.material_id
                WHERE mt.topic_id = ?
                ORDER BY lm.created_at DESC
            """
            if limit:
                query += f" LIMIT {limit}"
            cursor = self.conn.execute(query, (topic_id,))

        results = []
        for row in cursor.fetchall():
            result = dict(row)
            if result.get("image_paths"):
                result["image_paths"] = json.loads(result["image_paths"])
            results.append(result)
        return results

    def get_topics_for_material(self, material_id: str) -> List[Dict[str, Any]]:
        """Get all topics linked to a learning material.

        Args:
            material_id: Learning material ID

        Returns:
            List of topic dictionaries
        """
        cursor = self.conn.execute(
            """
            SELECT t.*
            FROM topics t
            JOIN material_topics mt ON t.id = mt.topic_id
            WHERE mt.material_id = ?
            ORDER BY t.name
        """,
            (material_id,),
        )

        return [dict(row) for row in cursor.fetchall()]

    def link_material_to_exercise(self, material_id: str, exercise_id: str, link_type: str):
        """Create a link between a learning material and an exercise.

        Args:
            material_id: Learning material ID
            exercise_id: Exercise ID
            link_type: Type of link ('worked_example', 'theory_reference', 'prerequisite')
        """
        self.conn.execute(
            """
            INSERT OR IGNORE INTO material_exercise_links
            (material_id, exercise_id, link_type)
            VALUES (?, ?, ?)
        """,
            (material_id, exercise_id, link_type),
        )

    def get_materials_for_exercise(self, exercise_id: str) -> List[Dict[str, Any]]:
        """Get all learning materials linked to an exercise.

        Args:
            exercise_id: Exercise ID

        Returns:
            List of (material, link_type) dictionaries
        """
        cursor = self.conn.execute(
            """
            SELECT lm.*, mel.link_type
            FROM learning_materials lm
            JOIN material_exercise_links mel ON lm.id = mel.material_id
            WHERE mel.exercise_id = ?
            ORDER BY mel.created_at
        """,
            (exercise_id,),
        )

        results = []
        for row in cursor.fetchall():
            result = dict(row)
            if result.get("image_paths"):
                result["image_paths"] = json.loads(result["image_paths"])
            results.append(result)
        return results

    def get_exercises_for_material(self, material_id: str) -> List[Dict[str, Any]]:
        """Get all exercises linked to a learning material.

        Args:
            material_id: Learning material ID

        Returns:
            List of (exercise, link_type) dictionaries
        """
        cursor = self.conn.execute(
            """
            SELECT e.*, mel.link_type
            FROM exercises e
            JOIN material_exercise_links mel ON e.id = mel.exercise_id
            WHERE mel.material_id = ?
            ORDER BY mel.created_at
        """,
            (material_id,),
        )

        results = []
        for row in cursor.fetchall():
            result = dict(row)
            if result.get("image_paths"):
                result["image_paths"] = json.loads(result["image_paths"])
            if result.get("variations"):
                result["variations"] = json.loads(result["variations"])
            if result.get("analysis_metadata"):
                result["analysis_metadata"] = json.loads(result["analysis_metadata"])
            results.append(result)
        return results

    # Procedure Cache operations (Option 3: Pattern Caching)

    def store_procedure_cache_entry(self, entry_data: Dict[str, Any]) -> int:
        """Store a new procedure cache entry.

        Args:
            entry_data: Dictionary containing cache entry fields:
                - user_id: Optional user ID for multi-tenant isolation (None = CLI mode)
                - course_code: Optional course code (None = global cache)
                - pattern_hash: Hash of normalized exercise text
                - exercise_text_sample: First 500 chars for inspection
                - topic: Cached topic name
                - difficulty: Cached difficulty
                - variations_json: JSON array of variations
                - procedures_json: JSON array of ProcedureInfo dicts
                - embedding: Vector embedding as numpy bytes (BLOB)
                - normalized_text: Normalized exercise text for matching
                - match_count: How many times this was matched (default: 0)
                - confidence_avg: Average confidence when matched (default: 1.0)

        Returns:
            Entry ID
        """
        # Convert lists/dicts to JSON strings if needed
        if "variations_json" in entry_data and isinstance(entry_data["variations_json"], list):
            entry_data["variations_json"] = json.dumps(entry_data["variations_json"])
        if "procedures_json" in entry_data and isinstance(entry_data["procedures_json"], list):
            entry_data["procedures_json"] = json.dumps(entry_data["procedures_json"])

        cursor = self.conn.execute(
            """
            INSERT INTO procedure_cache_entries
            (user_id, course_code, pattern_hash, exercise_text_sample, topic, difficulty,
             variations_json, procedures_json, embedding, normalized_text,
             match_count, confidence_avg)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                entry_data.get("user_id"),  # Web-ready: NULL for CLI mode
                entry_data.get("course_code"),
                entry_data["pattern_hash"],
                entry_data.get("exercise_text_sample"),
                entry_data.get("topic"),
                entry_data.get("difficulty"),
                entry_data.get("variations_json"),
                entry_data["procedures_json"],
                entry_data.get("embedding"),
                entry_data.get("normalized_text"),
                entry_data.get("match_count", 0),
                entry_data.get("confidence_avg", 1.0),
            ),
        )
        return cursor.lastrowid

    def get_procedure_cache_entries(
        self, course_code: Optional[str] = None, user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get procedure cache entries.

        Args:
            course_code: Optional course code to filter by.
                        If None, returns all entries (including global cache).
            user_id: Optional user ID for multi-tenant isolation.
                    If None (CLI mode), returns entries with user_id IS NULL.

        Returns:
            List of cache entry dictionaries
        """
        # Build query based on filters (web-ready with user isolation)
        if course_code is None and user_id is None:
            # CLI mode: return all entries with NULL user_id
            cursor = self.conn.execute("""
                SELECT * FROM procedure_cache_entries
                WHERE user_id IS NULL
                ORDER BY created_at DESC
            """)
        elif course_code is None and user_id is not None:
            # Web mode: return all entries for specific user
            cursor = self.conn.execute(
                """
                SELECT * FROM procedure_cache_entries
                WHERE user_id = ?
                ORDER BY created_at DESC
            """,
                (user_id,),
            )
        elif course_code is not None and user_id is None:
            # CLI mode: return entries for specific course + global entries
            cursor = self.conn.execute(
                """
                SELECT * FROM procedure_cache_entries
                WHERE (course_code = ? OR course_code IS NULL) AND user_id IS NULL
                ORDER BY created_at DESC
            """,
                (course_code,),
            )
        else:
            # Web mode: return entries for specific course + global entries for user
            cursor = self.conn.execute(
                """
                SELECT * FROM procedure_cache_entries
                WHERE (course_code = ? OR course_code IS NULL) AND user_id = ?
                ORDER BY created_at DESC
            """,
                (course_code, user_id),
            )

        results = []
        for row in cursor.fetchall():
            result = dict(row)
            # Parse JSON fields
            if result.get("variations_json"):
                result["variations_json"] = json.loads(result["variations_json"])
            if result.get("procedures_json"):
                result["procedures_json"] = json.loads(result["procedures_json"])
            results.append(result)
        return results

    def update_cache_entry_stats(self, entry_id: int, confidence: float):
        """Update match statistics for a cache entry.

        Updates match_count (increments by 1) and confidence_avg (running average).
        Also updates last_matched_at timestamp.

        Args:
            entry_id: Cache entry ID
            confidence: Confidence score of this match (0.0-1.0)
        """
        # Get current stats
        cursor = self.conn.execute(
            """
            SELECT match_count, confidence_avg
            FROM procedure_cache_entries
            WHERE id = ?
        """,
            (entry_id,),
        )
        row = cursor.fetchone()

        if row:
            current_count = row[0]
            current_avg = row[1]

            # Calculate new running average
            new_count = current_count + 1
            new_avg = ((current_avg * current_count) + confidence) / new_count

            # Update entry
            self.conn.execute(
                """
                UPDATE procedure_cache_entries
                SET match_count = ?,
                    confidence_avg = ?,
                    last_matched_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """,
                (new_count, new_avg, entry_id),
            )

    def delete_procedure_cache(
        self, course_code: Optional[str] = None, user_id: Optional[str] = None
    ):
        """Delete procedure cache entries.

        Args:
            course_code: Optional course code. If provided, deletes entries for that course.
                        If None, deletes ALL cache entries (including global).
            user_id: Optional user ID for multi-tenant isolation.
                    If None (CLI mode), deletes entries with user_id IS NULL.
        """
        # Web-ready: always scope by user_id for isolation
        if course_code is None and user_id is None:
            # CLI mode: delete all entries with NULL user_id
            self.conn.execute("DELETE FROM procedure_cache_entries WHERE user_id IS NULL")
        elif course_code is None and user_id is not None:
            # Web mode: delete all entries for specific user
            self.conn.execute("DELETE FROM procedure_cache_entries WHERE user_id = ?", (user_id,))
        elif course_code is not None and user_id is None:
            # CLI mode: delete entries for specific course (keeps global cache)
            self.conn.execute(
                """
                DELETE FROM procedure_cache_entries
                WHERE course_code = ? AND user_id IS NULL
            """,
                (course_code,),
            )
        else:
            # Web mode: delete entries for specific course for specific user
            self.conn.execute(
                """
                DELETE FROM procedure_cache_entries
                WHERE course_code = ? AND user_id = ?
            """,
                (course_code, user_id),
            )

    def get_cache_stats(
        self, course_code: Optional[str] = None, user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get statistics about the procedure cache.

        Args:
            course_code: Optional course code to filter by.
                        If None, returns stats for all entries.
            user_id: Optional user ID for multi-tenant isolation.
                    If None (CLI mode), returns stats for entries with user_id IS NULL.

        Returns:
            Dictionary with cache statistics:
                - total_entries: Total number of cache entries
                - total_matches: Sum of all match_count values
                - avg_confidence: Average of all confidence_avg values
                - course_entries: Number of course-specific entries (if course_code provided)
                - global_entries: Number of global entries
        """
        # Build user filter clause (web-ready)
        user_filter = "user_id IS NULL" if user_id is None else f"user_id = ?"
        user_params = () if user_id is None else (user_id,)

        if course_code is None:
            # Stats for all entries (scoped by user)
            cursor = self.conn.execute(
                f"""
                SELECT
                    COUNT(*) as total_entries,
                    SUM(match_count) as total_matches,
                    AVG(confidence_avg) as avg_confidence
                FROM procedure_cache_entries
                WHERE {user_filter}
            """,
                user_params,
            )
            row = cursor.fetchone()

            # Count global vs course-specific (within user scope)
            cursor = self.conn.execute(
                f"""
                SELECT COUNT(*) FROM procedure_cache_entries
                WHERE course_code IS NULL AND {user_filter}
            """,
                user_params,
            )
            global_count = cursor.fetchone()[0]

            return {
                "total_entries": row[0] or 0,
                "total_matches": row[1] or 0,
                "avg_confidence": row[2] or 0.0,
                "global_entries": global_count,
                "course_entries": (row[0] or 0) - global_count,
            }
        else:
            # Stats for specific course + global (scoped by user)
            params = (course_code,) + user_params
            cursor = self.conn.execute(
                f"""
                SELECT
                    COUNT(*) as total_entries,
                    SUM(match_count) as total_matches,
                    AVG(confidence_avg) as avg_confidence
                FROM procedure_cache_entries
                WHERE (course_code = ? OR course_code IS NULL) AND {user_filter}
            """,
                params,
            )
            row = cursor.fetchone()

            # Count course-specific (within user scope)
            cursor = self.conn.execute(
                f"""
                SELECT COUNT(*) FROM procedure_cache_entries
                WHERE course_code = ? AND {user_filter}
            """,
                params,
            )
            course_count = cursor.fetchone()[0]

            # Count global (within user scope)
            cursor = self.conn.execute(
                f"""
                SELECT COUNT(*) FROM procedure_cache_entries
                WHERE course_code IS NULL AND {user_filter}
            """,
                user_params,
            )
            global_count = cursor.fetchone()[0]

            return {
                "total_entries": row[0] or 0,
                "total_matches": row[1] or 0,
                "avg_confidence": row[2] or 0.0,
                "course_entries": course_count,
                "global_entries": global_count,
            }
