"""
Database management for Examina.
Handles SQLite operations and schema management.
"""

import sqlite3
import json
from datetime import datetime
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
        self._create_indexes()
        self._run_migrations()
        self.conn.commit()

    def _run_migrations(self):
        """Run database migrations for schema updates."""
        # Check if low_confidence_skipped column exists in exercises table
        cursor = self.conn.execute("PRAGMA table_info(exercises)")
        columns = [row[1] for row in cursor.fetchall()]

        if 'low_confidence_skipped' not in columns:
            print("[INFO] Running migration: Adding low_confidence_skipped column to exercises table")
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
        if 'time_limit' in quiz_sessions_columns and 'correct_answers' not in quiz_sessions_columns:
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
                    filter_core_loop_id TEXT,
                    filter_difficulty TEXT,
                    FOREIGN KEY (course_code) REFERENCES courses(code),
                    FOREIGN KEY (filter_topic_id) REFERENCES topics(id)
                )
            """)

            # Migrate existing data
            self.conn.execute("""
                INSERT INTO quiz_sessions_new
                (id, course_code, created_at, completed_at, total_questions,
                 correct_answers, score_percentage, quiz_type, filter_topic_id, filter_core_loop_id)
                SELECT
                    id, course_code, started_at, completed_at, total_questions,
                    total_correct, score, quiz_type, topic_id, core_loop_id
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

        # Migration: Mark exercises with core_loop_id as analyzed
        # (Fixes legacy data where exercises were analyzed but not marked)
        cursor = self.conn.execute("""
            SELECT COUNT(*) FROM exercises
            WHERE core_loop_id IS NOT NULL AND analyzed = 0
        """)
        unanalyzed_count = cursor.fetchone()[0]

        if unanalyzed_count > 0:
            print(f"[INFO] Running migration: Marking {unanalyzed_count} exercises with core_loop_id as analyzed=1")

            self.conn.execute("""
                UPDATE exercises
                SET analyzed = 1
                WHERE core_loop_id IS NOT NULL AND analyzed = 0
            """)

            print(f"[INFO] Migration completed: {unanalyzed_count} exercises marked as analyzed")

        # Multi-core-loop migration: Create exercise_core_loops table
        cursor = self.conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='exercise_core_loops'
        """)
        if not cursor.fetchone():
            print("[INFO] Running migration: Creating exercise_core_loops junction table")
            self.conn.execute("""
                CREATE TABLE exercise_core_loops (
                    exercise_id TEXT NOT NULL,
                    core_loop_id TEXT NOT NULL,
                    step_number INTEGER,
                    PRIMARY KEY (exercise_id, core_loop_id),
                    FOREIGN KEY (exercise_id) REFERENCES exercises(id) ON DELETE CASCADE,
                    FOREIGN KEY (core_loop_id) REFERENCES core_loops(id) ON DELETE CASCADE
                )
            """)
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_exercise_core_loops_exercise
                ON exercise_core_loops(exercise_id)
            """)
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_exercise_core_loops_core_loop
                ON exercise_core_loops(core_loop_id)
            """)
            print("[INFO] Migration completed: exercise_core_loops table created")

        # Multi-core-loop migration: Add tags column to exercises
        cursor = self.conn.execute("PRAGMA table_info(exercises)")
        columns = [row[1] for row in cursor.fetchall()]

        if 'tags' not in columns:
            print("[INFO] Running migration: Adding tags column to exercises table")
            self.conn.execute("""
                ALTER TABLE exercises
                ADD COLUMN tags TEXT
            """)
            print("[INFO] Migration completed: tags column added")

        # Multi-core-loop migration: Migrate existing core_loop_id data to exercise_core_loops
        # Check if migration is needed (exercise_core_loops exists but might be empty)
        cursor = self.conn.execute("""
            SELECT COUNT(*) FROM exercises
            WHERE core_loop_id IS NOT NULL
        """)
        exercises_with_core_loop = cursor.fetchone()[0]

        cursor = self.conn.execute("SELECT COUNT(*) FROM exercise_core_loops")
        junction_entries = cursor.fetchone()[0]

        if exercises_with_core_loop > 0 and junction_entries == 0:
            print(f"[INFO] Running migration: Migrating {exercises_with_core_loop} exercise core_loop_id references to exercise_core_loops table")

            # Insert all existing core_loop_id relationships into junction table
            self.conn.execute("""
                INSERT INTO exercise_core_loops (exercise_id, core_loop_id, step_number)
                SELECT id, core_loop_id, NULL
                FROM exercises
                WHERE core_loop_id IS NOT NULL
            """)

            cursor = self.conn.execute("SELECT COUNT(*) FROM exercise_core_loops")
            migrated_count = cursor.fetchone()[0]

            print(f"[INFO] Migration completed: {migrated_count} relationships migrated to exercise_core_loops")
            print("[INFO] Note: exercises.core_loop_id column retained for backward compatibility")

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
            CREATE TABLE IF NOT EXISTS core_loops (
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
                core_loop_id TEXT,
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
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (course_code) REFERENCES courses(code),
                FOREIGN KEY (topic_id) REFERENCES topics(id),
                FOREIGN KEY (core_loop_id) REFERENCES core_loops(id)
            )
        """)

        # Student progress table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS student_progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                course_code TEXT NOT NULL,
                core_loop_id TEXT NOT NULL,
                total_attempts INTEGER DEFAULT 0,
                correct_attempts INTEGER DEFAULT 0,
                mastery_score REAL DEFAULT 0.0,
                last_practiced TIMESTAMP,
                next_review TIMESTAMP,
                review_interval INTEGER DEFAULT 1,
                common_mistakes TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (course_code) REFERENCES courses(code),
                FOREIGN KEY (core_loop_id) REFERENCES core_loops(id),
                UNIQUE(course_code, core_loop_id)
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
                filter_core_loop_id TEXT,
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
                core_loop_id TEXT NOT NULL,
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
                FOREIGN KEY (core_loop_id) REFERENCES core_loops(id)
            )
        """)

        # Exercise-CoreLoop junction table (many-to-many)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS exercise_core_loops (
                exercise_id TEXT NOT NULL,
                core_loop_id TEXT NOT NULL,
                step_number INTEGER,
                PRIMARY KEY (exercise_id, core_loop_id),
                FOREIGN KEY (exercise_id) REFERENCES exercises(id) ON DELETE CASCADE,
                FOREIGN KEY (core_loop_id) REFERENCES core_loops(id) ON DELETE CASCADE
            )
        """)

    def _create_indexes(self):
        """Create database indexes for performance."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_exercises_course ON exercises(course_code)",
            "CREATE INDEX IF NOT EXISTS idx_exercises_core_loop ON exercises(core_loop_id)",
            "CREATE INDEX IF NOT EXISTS idx_exercises_topic ON exercises(topic_id)",
            "CREATE INDEX IF NOT EXISTS idx_topics_course ON topics(course_code)",
            "CREATE INDEX IF NOT EXISTS idx_core_loops_topic ON core_loops(topic_id)",
            "CREATE INDEX IF NOT EXISTS idx_progress_course ON student_progress(course_code)",
            "CREATE INDEX IF NOT EXISTS idx_progress_core_loop ON student_progress(core_loop_id)",
            "CREATE INDEX IF NOT EXISTS idx_quiz_sessions_course ON quiz_sessions(course_code)",
            "CREATE INDEX IF NOT EXISTS idx_quiz_attempts_session ON quiz_attempts(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_quiz_attempts_exercise ON quiz_attempts(exercise_id)",
            "CREATE INDEX IF NOT EXISTS idx_exercise_reviews_course ON exercise_reviews(course_code)",
            "CREATE INDEX IF NOT EXISTS idx_exercise_reviews_next_review ON exercise_reviews(next_review_date)",
            "CREATE INDEX IF NOT EXISTS idx_topic_mastery_course ON topic_mastery(course_code)",
            "CREATE INDEX IF NOT EXISTS idx_exercise_core_loops_exercise ON exercise_core_loops(exercise_id)",
            "CREATE INDEX IF NOT EXISTS idx_exercise_core_loops_core_loop ON exercise_core_loops(core_loop_id)",
        ]

        for index_sql in indexes:
            self.conn.execute(index_sql)

    # Course operations
    def add_course(self, code: str, name: str, original_name: str = None,
                   acronym: str = None, degree_level: str = None,
                   degree_program: str = None):
        """Add a new course to the database.

        Args:
            code: Course code (e.g., "B006802")
            name: English course name
            original_name: Original language name (Italian)
            acronym: Course acronym
            degree_level: "bachelor" or "master"
            degree_program: "L-31" or "LM-18"
        """
        self.conn.execute("""
            INSERT OR IGNORE INTO courses
            (code, name, original_name, acronym, degree_level, degree_program)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (code, name, original_name, acronym, degree_level, degree_program))

    def get_course(self, code: str) -> Optional[Dict[str, Any]]:
        """Get course information by code."""
        cursor = self.conn.execute(
            "SELECT * FROM courses WHERE code = ?", (code,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_all_courses(self) -> List[Dict[str, Any]]:
        """Get all courses."""
        cursor = self.conn.execute("SELECT * FROM courses ORDER BY name")
        return [dict(row) for row in cursor.fetchall()]

    # Topic operations
    def add_topic(self, course_code: str, name: str, description: str = None) -> int:
        """Add a new topic to a course.

        Returns:
            Topic ID
        """
        cursor = self.conn.execute("""
            INSERT OR IGNORE INTO topics (course_code, name, description)
            VALUES (?, ?, ?)
        """, (course_code, name, description))

        if cursor.lastrowid == 0:
            # Topic already exists, fetch its ID
            cursor = self.conn.execute(
                "SELECT id FROM topics WHERE course_code = ? AND name = ?",
                (course_code, name)
            )
            return cursor.fetchone()[0]
        return cursor.lastrowid

    def get_topics_by_course(self, course_code: str) -> List[Dict[str, Any]]:
        """Get all topics for a course."""
        cursor = self.conn.execute("""
            SELECT * FROM topics
            WHERE course_code = ?
            ORDER BY name
        """, (course_code,))
        return [dict(row) for row in cursor.fetchall()]

    def split_topic(self, old_topic_id: int, clusters: List[Dict[str, Any]],
                    course_code: str, delete_old: bool = False) -> Dict[str, Any]:
        """Split a generic topic into multiple specific topics.

        Args:
            old_topic_id: ID of topic to split
            clusters: List of dicts with 'topic_name' and 'core_loop_ids' keys
            course_code: Course code for the topic
            delete_old: Whether to delete old topic if empty (default: False)

        Returns:
            Dict with split statistics and new topic IDs

        Example clusters:
            [
                {
                    "topic_name": "Autovalori e Diagonalizzazione",
                    "core_loop_ids": ["loop1", "loop2", ...]
                },
                ...
            ]
        """
        try:
            # Get original topic info for logging
            cursor = self.conn.execute(
                "SELECT name FROM topics WHERE id = ?", (old_topic_id,)
            )
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Topic ID {old_topic_id} not found")

            old_topic_name = row[0]
            stats = {
                "old_topic_id": old_topic_id,
                "old_topic_name": old_topic_name,
                "new_topics": [],
                "core_loops_moved": 0,
                "errors": []
            }

            # Process each cluster
            for cluster in clusters:
                topic_name = cluster.get("topic_name")
                core_loop_ids = cluster.get("core_loop_ids", [])

                if not topic_name or not core_loop_ids:
                    stats["errors"].append(f"Invalid cluster: {cluster}")
                    continue

                # Create new topic
                new_topic_id = self.add_topic(course_code, topic_name)

                # Move core loops to new topic
                moved_count = 0
                for loop_id in core_loop_ids:
                    try:
                        self.conn.execute("""
                            UPDATE core_loops
                            SET topic_id = ?, updated_at = CURRENT_TIMESTAMP
                            WHERE id = ?
                        """, (new_topic_id, loop_id))
                        moved_count += 1
                    except Exception as e:
                        stats["errors"].append(f"Failed to move {loop_id}: {e}")

                stats["new_topics"].append({
                    "id": new_topic_id,
                    "name": topic_name,
                    "core_loops_moved": moved_count
                })
                stats["core_loops_moved"] += moved_count

            # Optionally delete old topic if no core loops remain
            if delete_old:
                cursor = self.conn.execute("""
                    SELECT COUNT(*) FROM core_loops WHERE topic_id = ?
                """, (old_topic_id,))
                remaining_loops = cursor.fetchone()[0]

                if remaining_loops == 0:
                    self.conn.execute("DELETE FROM topics WHERE id = ?", (old_topic_id,))
                    stats["old_topic_deleted"] = True
                else:
                    stats["old_topic_deleted"] = False
                    stats["remaining_core_loops"] = remaining_loops

            self.conn.commit()
            return stats

        except Exception as e:
            self.conn.rollback()
            raise RuntimeError(f"Topic split failed: {e}")

    # Core loop operations
    def add_core_loop(self, loop_id: str, topic_id: int, name: str,
                      procedure: List[str], description: str = None) -> str:
        """Add a new core loop.

        Args:
            loop_id: Unique identifier for the core loop
            topic_id: Parent topic ID
            name: Name of the core loop
            procedure: List of procedure steps
            description: Optional description

        Returns:
            Core loop ID
        """
        procedure_json = json.dumps(procedure)
        self.conn.execute("""
            INSERT OR REPLACE INTO core_loops
            (id, topic_id, name, description, procedure, updated_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (loop_id, topic_id, name, description, procedure_json))
        return loop_id

    def get_core_loop(self, loop_id: str) -> Optional[Dict[str, Any]]:
        """Get core loop by ID."""
        cursor = self.conn.execute(
            "SELECT * FROM core_loops WHERE id = ?", (loop_id,)
        )
        row = cursor.fetchone()
        if row:
            result = dict(row)
            result['procedure'] = json.loads(result['procedure'])
            return result
        return None

    def get_core_loops_by_topic(self, topic_id: int) -> List[Dict[str, Any]]:
        """Get all core loops for a topic."""
        cursor = self.conn.execute("""
            SELECT * FROM core_loops
            WHERE topic_id = ?
            ORDER BY name
        """, (topic_id,))
        results = []
        for row in cursor.fetchall():
            result = dict(row)
            result['procedure'] = json.loads(result['procedure'])
            results.append(result)
        return results

    def get_core_loops_by_course(self, course_code: str) -> List[Dict[str, Any]]:
        """Get all core loops for a course."""
        cursor = self.conn.execute("""
            SELECT cl.* FROM core_loops cl
            JOIN topics t ON cl.topic_id = t.id
            WHERE t.course_code = ?
            ORDER BY cl.name
        """, (course_code,))
        results = []
        for row in cursor.fetchall():
            result = dict(row)
            result['procedure'] = json.loads(result['procedure'])
            results.append(result)
        return results

    def update_core_loop_stats(self, loop_id: str):
        """Update exercise count and average difficulty for a core loop."""
        cursor = self.conn.execute("""
            SELECT COUNT(*) as count, AVG(
                CASE difficulty
                    WHEN 'easy' THEN 1
                    WHEN 'medium' THEN 2
                    WHEN 'hard' THEN 3
                    ELSE 2
                END
            ) as avg_diff
            FROM exercises
            WHERE core_loop_id = ?
        """, (loop_id,))

        row = cursor.fetchone()
        if row:
            self.conn.execute("""
                UPDATE core_loops
                SET exercise_count = ?, difficulty_avg = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (row[0], row[1] or 0.0, loop_id))

    # Exercise operations
    def add_exercise(self, exercise_data: Dict[str, Any]) -> str:
        """Add a new exercise.

        Args:
            exercise_data: Dictionary with exercise information

        Returns:
            Exercise ID
        """
        # Convert lists/dicts to JSON strings
        if 'image_paths' in exercise_data and isinstance(exercise_data['image_paths'], list):
            exercise_data['image_paths'] = json.dumps(exercise_data['image_paths'])
        if 'variations' in exercise_data and isinstance(exercise_data['variations'], list):
            exercise_data['variations'] = json.dumps(exercise_data['variations'])
        if 'analysis_metadata' in exercise_data and isinstance(exercise_data['analysis_metadata'], dict):
            exercise_data['analysis_metadata'] = json.dumps(exercise_data['analysis_metadata'])

        self.conn.execute("""
            INSERT INTO exercises
            (id, course_code, topic_id, core_loop_id, source_pdf, page_number,
             exercise_number, text, has_images, image_paths, latex_content,
             difficulty, variations, solution, analyzed, analysis_metadata)
            VALUES
            (:id, :course_code, :topic_id, :core_loop_id, :source_pdf, :page_number,
             :exercise_number, :text, :has_images, :image_paths, :latex_content,
             :difficulty, :variations, :solution, :analyzed, :analysis_metadata)
        """, exercise_data)

        return exercise_data['id']

    def get_exercise(self, exercise_id: str) -> Optional[Dict[str, Any]]:
        """Get exercise by ID."""
        cursor = self.conn.execute(
            "SELECT * FROM exercises WHERE id = ?", (exercise_id,)
        )
        row = cursor.fetchone()
        if row:
            result = dict(row)
            # Parse JSON fields
            if result.get('image_paths'):
                result['image_paths'] = json.loads(result['image_paths'])
            if result.get('variations'):
                result['variations'] = json.loads(result['variations'])
            if result.get('analysis_metadata'):
                result['analysis_metadata'] = json.loads(result['analysis_metadata'])
            return result
        return None

    def get_exercises_by_core_loop(self, core_loop_id: str) -> List[Dict[str, Any]]:
        """Get all exercises that include this core loop.

        Uses the junction table for many-to-many relationships.
        For backward compatibility, also checks the legacy core_loop_id column.
        """
        # Check if created_at column exists for ordering
        cursor = self.conn.execute("PRAGMA table_info(exercises)")
        columns = [row[1] for row in cursor.fetchall()]
        order_by = "e.created_at" if 'created_at' in columns else "e.id"

        cursor = self.conn.execute(f"""
            SELECT DISTINCT e.*, ecl.step_number
            FROM exercises e
            LEFT JOIN exercise_core_loops ecl ON e.id = ecl.exercise_id
            WHERE ecl.core_loop_id = ? OR e.core_loop_id = ?
            ORDER BY {order_by}
        """, (core_loop_id, core_loop_id))

        results = []
        for row in cursor.fetchall():
            result = dict(row)
            if result.get('image_paths'):
                result['image_paths'] = json.loads(result['image_paths'])
            if result.get('variations'):
                result['variations'] = json.loads(result['variations'])
            if result.get('analysis_metadata'):
                result['analysis_metadata'] = json.loads(result['analysis_metadata'])
            results.append(result)
        return results

    def get_exercises_by_course(self, course_code: str, analyzed_only: bool = False,
                                 unanalyzed_only: bool = False) -> List[Dict[str, Any]]:
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
            if result.get('image_paths'):
                result['image_paths'] = json.loads(result['image_paths'])
            if result.get('variations'):
                result['variations'] = json.loads(result['variations'])
            if result.get('analysis_metadata'):
                result['analysis_metadata'] = json.loads(result['analysis_metadata'])
            results.append(result)
        return results

    # Phase 5: Quiz Session operations
    def create_quiz_session(self, session_id: str, course_code: str, quiz_type: str,
                           filter_topic_id: Optional[int] = None,
                           filter_core_loop_id: Optional[str] = None,
                           filter_difficulty: Optional[str] = None) -> str:
        """Create a new quiz session.

        Args:
            session_id: Unique identifier for the session
            course_code: Course code
            quiz_type: Type of quiz ('topic', 'core_loop', 'random', 'review')
            filter_topic_id: Optional topic filter
            filter_core_loop_id: Optional core loop filter
            filter_difficulty: Optional difficulty filter

        Returns:
            Session ID
        """
        self.conn.execute("""
            INSERT INTO quiz_sessions
            (id, course_code, quiz_type, filter_topic_id, filter_core_loop_id, filter_difficulty)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (session_id, course_code, quiz_type, filter_topic_id, filter_core_loop_id, filter_difficulty))
        return session_id

    def get_quiz_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get quiz session by ID.

        Args:
            session_id: Session ID to retrieve

        Returns:
            Session dictionary or None if not found
        """
        cursor = self.conn.execute(
            "SELECT * FROM quiz_sessions WHERE id = ?", (session_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def update_quiz_session(self, session_id: str, total_questions: Optional[int] = None,
                           correct_answers: Optional[int] = None,
                           score_percentage: Optional[float] = None,
                           completed: bool = False):
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

    def get_quiz_sessions_by_course(self, course_code: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent quiz sessions for a course.

        Args:
            course_code: Course code to filter by
            limit: Maximum number of sessions to return

        Returns:
            List of session dictionaries
        """
        cursor = self.conn.execute("""
            SELECT * FROM quiz_sessions
            WHERE course_code = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (course_code, limit))
        return [dict(row) for row in cursor.fetchall()]

    # Phase 5: Quiz Attempt operations
    def add_quiz_attempt(self, session_id: str, exercise_id: str, user_answer: Optional[str] = None,
                        correct: Optional[bool] = None, time_taken_seconds: Optional[int] = None,
                        hint_used: bool = False, feedback: Optional[str] = None) -> int:
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
        cursor = self.conn.execute("""
            INSERT INTO quiz_attempts
            (session_id, exercise_id, user_answer, correct, time_taken_seconds, hint_used, feedback)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (session_id, exercise_id, user_answer, correct, time_taken_seconds, hint_used, feedback))
        return cursor.lastrowid

    def get_quiz_attempts(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all attempts for a quiz session.

        Args:
            session_id: Session ID

        Returns:
            List of attempt dictionaries
        """
        cursor = self.conn.execute("""
            SELECT * FROM quiz_attempts
            WHERE session_id = ?
            ORDER BY attempted_at
        """, (session_id,))
        return [dict(row) for row in cursor.fetchall()]

    def get_attempts_by_exercise(self, exercise_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent attempts for a specific exercise.

        Args:
            exercise_id: Exercise ID
            limit: Maximum number of attempts to return

        Returns:
            List of attempt dictionaries
        """
        cursor = self.conn.execute("""
            SELECT * FROM quiz_attempts
            WHERE exercise_id = ?
            ORDER BY attempted_at DESC
            LIMIT ?
        """, (exercise_id, limit))
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

    def update_exercise_review(self, exercise_id: str, course_code: str,
                               easiness_factor: float, repetition_number: int,
                               interval_days: int, next_review_date: str,
                               mastery_level: str, correct: bool):
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
            self.conn.execute("""
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
            """, (easiness_factor, repetition_number, interval_days, next_review_date,
                  1 if correct else 0, mastery_level, exercise_id))
        else:
            # Create new record
            self.conn.execute("""
                INSERT INTO exercise_reviews
                (exercise_id, course_code, easiness_factor, repetition_number, interval_days,
                 next_review_date, last_reviewed_at, total_reviews, correct_reviews, mastery_level)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, 1, ?, ?)
            """, (exercise_id, course_code, easiness_factor, repetition_number, interval_days,
                  next_review_date, 1 if correct else 0, mastery_level))

    def get_exercises_due_for_review(self, course_code: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get exercises that are due for review.

        Args:
            course_code: Course code to filter by
            limit: Maximum number of exercises to return

        Returns:
            List of exercise review dictionaries
        """
        cursor = self.conn.execute("""
            SELECT * FROM exercise_reviews
            WHERE course_code = ?
            AND next_review_date <= DATE('now')
            ORDER BY next_review_date
            LIMIT ?
        """, (course_code, limit))
        return [dict(row) for row in cursor.fetchall()]

    def get_exercises_by_mastery(self, course_code: str, mastery_level: str) -> List[Dict[str, Any]]:
        """Get exercises filtered by mastery level.

        Args:
            course_code: Course code to filter by
            mastery_level: Mastery level ('new', 'learning', 'mastered')

        Returns:
            List of exercise review dictionaries
        """
        cursor = self.conn.execute("""
            SELECT * FROM exercise_reviews
            WHERE course_code = ? AND mastery_level = ?
            ORDER BY last_reviewed_at DESC
        """, (course_code, mastery_level))
        return [dict(row) for row in cursor.fetchall()]

    # Phase 5: Topic Mastery operations
    def get_topic_mastery(self, topic_id: int) -> Optional[Dict[str, Any]]:
        """Get mastery data for a topic.

        Args:
            topic_id: Topic ID

        Returns:
            Mastery data dictionary or None if not found
        """
        cursor = self.conn.execute(
            "SELECT * FROM topic_mastery WHERE topic_id = ?", (topic_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def update_topic_mastery(self, topic_id: int, course_code: str,
                            exercises_total: int, exercises_mastered: int):
        """Update mastery data for a topic.

        Args:
            topic_id: Topic ID
            course_code: Course code
            exercises_total: Total number of exercises in topic
            exercises_mastered: Number of mastered exercises
        """
        mastery_percentage = (exercises_mastered / exercises_total * 100.0) if exercises_total > 0 else 0.0

        # Check if mastery record exists
        existing = self.get_topic_mastery(topic_id)

        if existing:
            # Update existing record
            self.conn.execute("""
                UPDATE topic_mastery
                SET exercises_total = ?,
                    exercises_mastered = ?,
                    mastery_percentage = ?,
                    last_practiced_at = CURRENT_TIMESTAMP
                WHERE topic_id = ?
            """, (exercises_total, exercises_mastered, mastery_percentage, topic_id))
        else:
            # Create new record
            self.conn.execute("""
                INSERT INTO topic_mastery
                (topic_id, course_code, exercises_total, exercises_mastered,
                 mastery_percentage, last_practiced_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (topic_id, course_code, exercises_total, exercises_mastered, mastery_percentage))

    def get_all_topic_mastery(self, course_code: str) -> List[Dict[str, Any]]:
        """Get mastery data for all topics in a course.

        Args:
            course_code: Course code to filter by

        Returns:
            List of topic mastery dictionaries with topic names
        """
        cursor = self.conn.execute("""
            SELECT tm.*, t.name as topic_name
            FROM topic_mastery tm
            JOIN topics t ON tm.topic_id = t.id
            WHERE tm.course_code = ?
            ORDER BY tm.mastery_percentage DESC
        """, (course_code,))
        return [dict(row) for row in cursor.fetchall()]

    def recalculate_topic_mastery(self, topic_id: int, course_code: str):
        """Recalculate mastery percentage for a topic based on exercise reviews.

        Args:
            topic_id: Topic ID
            course_code: Course code
        """
        # Count total exercises for this topic
        cursor = self.conn.execute("""
            SELECT COUNT(*) FROM exercises
            WHERE topic_id = ?
        """, (topic_id,))
        exercises_total = cursor.fetchone()[0]

        # Count mastered exercises (those with mastery_level = 'mastered')
        cursor = self.conn.execute("""
            SELECT COUNT(*) FROM exercise_reviews er
            JOIN exercises e ON er.exercise_id = e.id
            WHERE e.topic_id = ? AND er.mastery_level = 'mastered'
        """, (topic_id,))
        exercises_mastered = cursor.fetchone()[0]

        # Update topic mastery
        self.update_topic_mastery(topic_id, course_code, exercises_total, exercises_mastered)

    # Multi-core-loop operations
    def link_exercise_to_core_loop(self, exercise_id: str, core_loop_id: str, step_number: Optional[int] = None) -> None:
        """Link exercise to a core loop (allows many-to-many).

        Args:
            exercise_id: Exercise ID
            core_loop_id: Core loop ID
            step_number: Optional step number indicating which point in exercise (1, 2, 3, etc)
        """
        self.conn.execute("""
            INSERT OR REPLACE INTO exercise_core_loops
            (exercise_id, core_loop_id, step_number)
            VALUES (?, ?, ?)
        """, (exercise_id, core_loop_id, step_number))

    def get_exercise_core_loops(self, exercise_id: str) -> List[Dict]:
        """Get all core loops for an exercise.

        Args:
            exercise_id: Exercise ID

        Returns:
            List of core loop dictionaries with step_number included
        """
        cursor = self.conn.execute("""
            SELECT cl.*, ecl.step_number
            FROM core_loops cl
            JOIN exercise_core_loops ecl ON cl.id = ecl.core_loop_id
            WHERE ecl.exercise_id = ?
            ORDER BY ecl.step_number, cl.name
        """, (exercise_id,))

        results = []
        for row in cursor.fetchall():
            result = dict(row)
            result['procedure'] = json.loads(result['procedure'])
            results.append(result)
        return results

    def get_exercises_with_multiple_procedures(self, course_code: str) -> List[Dict]:
        """Get exercises that cover multiple core loops.

        Args:
            course_code: Course code to filter by

        Returns:
            List of exercise dictionaries with a 'core_loop_count' field
        """
        # Check if created_at column exists for ordering
        cursor = self.conn.execute("PRAGMA table_info(exercises)")
        columns = [row[1] for row in cursor.fetchall()]
        order_by = "e.created_at" if 'created_at' in columns else "e.id"

        cursor = self.conn.execute(f"""
            SELECT e.*, COUNT(ecl.core_loop_id) as core_loop_count
            FROM exercises e
            JOIN exercise_core_loops ecl ON e.id = ecl.exercise_id
            WHERE e.course_code = ?
            GROUP BY e.id
            HAVING core_loop_count > 1
            ORDER BY core_loop_count DESC, {order_by}
        """, (course_code,))

        results = []
        for row in cursor.fetchall():
            result = dict(row)
            # Parse JSON fields
            if result.get('image_paths'):
                result['image_paths'] = json.loads(result['image_paths'])
            if result.get('variations'):
                result['variations'] = json.loads(result['variations'])
            if result.get('analysis_metadata'):
                result['analysis_metadata'] = json.loads(result['analysis_metadata'])
            results.append(result)
        return results

    def get_exercises_by_procedure_type(self, course_code: str, procedure_type: str) -> List[Dict[str, Any]]:
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
        order_by = "e.created_at" if 'created_at' in columns else "e.id"

        cursor = self.conn.execute(f"""
            SELECT DISTINCT e.*
            FROM exercises e
            WHERE e.course_code = ?
            AND e.tags LIKE ?
            ORDER BY {order_by} DESC
        """, (course_code, f'%"{procedure_type}"%'))

        results = []
        for row in cursor.fetchall():
            result = dict(row)
            if result.get('image_paths'):
                result['image_paths'] = json.loads(result['image_paths'])
            if result.get('variations'):
                result['variations'] = json.loads(result['variations'])
            if result.get('analysis_metadata'):
                result['analysis_metadata'] = json.loads(result['analysis_metadata'])
            if result.get('tags'):
                result['tags'] = json.loads(result['tags'])
            results.append(result)
        return results

    def get_core_loops_for_exercise(self, exercise_id: str) -> List[Dict[str, Any]]:
        """Get all core loops associated with an exercise.

        Args:
            exercise_id: Exercise ID

        Returns:
            List of core loop dictionaries with step_number information
        """
        cursor = self.conn.execute("""
            SELECT cl.*, ecl.step_number
            FROM core_loops cl
            JOIN exercise_core_loops ecl ON cl.id = ecl.core_loop_id
            WHERE ecl.exercise_id = ?
            ORDER BY ecl.step_number
        """, (exercise_id,))

        return [dict(row) for row in cursor.fetchall()]

    def update_exercise_tags(self, exercise_id: str, tags: List[str]):
        """Update tags for an exercise.

        Args:
            exercise_id: Exercise ID
            tags: List of tag strings
        """
        tags_json = json.dumps(tags) if tags else None
        self.conn.execute("""
            UPDATE exercises
            SET tags = ?
            WHERE id = ?
        """, (tags_json, exercise_id))

    def update_exercise_analysis(self, exercise_id: str, topic_id: Optional[int] = None,
                                 core_loop_id: Optional[str] = None,
                                 difficulty: Optional[str] = None,
                                 variations: Optional[List[str]] = None,
                                 analysis_metadata: Optional[Dict[str, Any]] = None,
                                 analyzed: bool = True,
                                 low_confidence_skipped: bool = False):
        """Update exercise with analysis results.

        Args:
            exercise_id: Exercise ID
            topic_id: Topic ID (optional)
            core_loop_id: Primary core loop ID (optional, for backward compatibility)
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

        if core_loop_id is not None:
            updates.append("core_loop_id = ?")
            params.append(core_loop_id)

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
        cursor = self.conn.execute("""
            SELECT * FROM exercises
            WHERE course_code = ? AND tags LIKE ?
            ORDER BY created_at DESC
        """, (course_code, f'%{tag}%'))

        results = []
        for row in cursor.fetchall():
            result = dict(row)
            if result.get('image_paths'):
                result['image_paths'] = json.loads(result['image_paths'])
            if result.get('variations'):
                result['variations'] = json.loads(result['variations'])
            if result.get('analysis_metadata'):
                result['analysis_metadata'] = json.loads(result['analysis_metadata'])
            if result.get('tags'):
                result['tags'] = json.loads(result['tags'])
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
        cursor = self.conn.execute("""
            SELECT * FROM exercises
            WHERE course_code = ? AND (
                text LIKE ? OR
                exercise_number LIKE ? OR
                source_pdf LIKE ?
            )
            ORDER BY created_at DESC
        """, (course_code, f'%{search_text}%', f'%{search_text}%', f'%{search_text}%'))

        results = []
        for row in cursor.fetchall():
            result = dict(row)
            if result.get('image_paths'):
                result['image_paths'] = json.loads(result['image_paths'])
            if result.get('variations'):
                result['variations'] = json.loads(result['variations'])
            if result.get('analysis_metadata'):
                result['analysis_metadata'] = json.loads(result['analysis_metadata'])
            if result.get('tags'):
                result['tags'] = json.loads(result['tags'])
            results.append(result)
        return results

