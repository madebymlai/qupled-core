"""
Procedure Pattern Cache for Examina.
Caches procedure patterns to avoid redundant LLM calls during analysis.

This module implements Option 3: Procedure Pattern Caching from the performance optimization roadmap.

Key Features:
- Embedding-based similarity matching using semantic embeddings
- Fallback to text-based matching when embeddings unavailable
- Hybrid matching combining embedding and text similarity
- Pattern normalization to match exercises with different specific values
- Persistent storage in SQLite database
- In-memory cache with batch similarity computation for speed

Workflow:
1. On lookup: Check if exercise matches cached patterns (embedding + text validation)
2. On miss: After LLM analysis, add new pattern to cache
3. On hit: Return cached procedures (skip LLM call)

Performance Impact:
- Reduces redundant LLM calls for similar exercises
- Expected speedup: 2-3x on courses with repetitive patterns
- Minimal overhead: Fast numpy-based batch similarity computation

Usage Example:
    from storage.database import Database
    from core.procedure_cache import ProcedureCache

    db = Database()
    cache = ProcedureCache(db)

    # Lookup exercise in cache
    result = cache.lookup("Design Mealy machine for pattern 110")
    if result:
        procedures = result.procedures  # Use cached procedures
    else:
        # Perform LLM analysis...
        procedures = analyze_with_llm(exercise)
        # Add to cache for future use
        cache.add(exercise, topic, difficulty, variations, procedures, confidence)
"""

import re
import json
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from difflib import SequenceMatcher
from datetime import datetime

from config import Config


@dataclass
class CacheHit:
    """Result of a cache lookup."""

    procedures: List[Dict[str, Any]]  # List of procedure dicts
    topic: Optional[str]
    difficulty: Optional[str]
    variations: Optional[List[str]]
    confidence: float  # 0.0-1.0
    match_type: str  # 'exact', 'embedding', 'hybrid'
    source_entry_id: int
    embedding_similarity: float = 0.0
    text_similarity: float = 0.0


@dataclass
class CacheStats:
    """Statistics for procedure cache."""

    hits: int = 0
    misses: int = 0

    @property
    def total_lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total_lookups if self.total_lookups > 0 else 0.0


class ProcedureCache:
    """
    Cache for procedure patterns using embedding-based matching.

    Workflow:
    1. On lookup: Check if exercise matches cached patterns
    2. On miss: After LLM analysis, add new pattern to cache
    3. On hit: Return cached procedures (skip LLM call)
    """

    def __init__(self, db, semantic_matcher=None, user_id: Optional[str] = None):
        """
        Initialize procedure cache.

        Args:
            db: Database instance for persistence
            semantic_matcher: SemanticMatcher instance for embeddings (optional)
            user_id: Optional user ID for multi-tenant isolation (None = CLI mode)
        """
        self.db = db
        self.semantic_matcher = semantic_matcher
        self.user_id = user_id  # Web-ready: None for CLI, set for web multi-tenant
        self.stats = CacheStats()

        # In-memory cache (loaded from DB)
        self._entries: List[Dict[str, Any]] = []
        self._embeddings: Optional[np.ndarray] = None  # Matrix for batch similarity
        self._loaded = False

        # Configuration thresholds (from Config for flexibility)
        self._embedding_threshold = Config.PROCEDURE_CACHE_EMBEDDING_THRESHOLD
        self._text_threshold = Config.PROCEDURE_CACHE_TEXT_VALIDATION_THRESHOLD
        self._min_confidence = Config.PROCEDURE_CACHE_MIN_CONFIDENCE
        self._hybrid_threshold = 0.80  # Hybrid score threshold (0.7*embedding + 0.3*text)

    def load_cache(self, course_code: Optional[str] = None):
        """
        Load cache entries from database into memory.

        Args:
            course_code: Optional course code to filter entries. None for all courses.
        """
        if not self.db.conn:
            self.db.connect()

        # Use database method with user_id isolation (web-ready)
        rows = self.db.get_procedure_cache_entries(course_code=course_code, user_id=self.user_id)

        # Load entries into memory
        # Note: rows from get_procedure_cache_entries have JSON already parsed
        self._entries = []
        embeddings_list = []

        for row in rows:
            entry = {
                "id": row["id"],
                "exercise_text": row.get("exercise_text_sample"),
                "normalized_text": row.get("normalized_text"),
                "pattern_hash": row.get("pattern_hash"),
                "topic": row.get("topic"),
                "difficulty": row.get("difficulty"),
                "variations": row.get("variations_json") or [],  # Already parsed by DB method
                "procedures": row.get("procedures_json") or [],  # Already parsed by DB method
                "confidence": row.get("confidence_avg", 1.0),
                "course_code": row.get("course_code"),
                "hit_count": row.get("match_count", 0),
                "created_at": row.get("created_at"),
            }

            # Parse embedding if available
            if row.get("embedding"):
                embedding = np.frombuffer(row["embedding"], dtype=np.float32)
                entry["embedding"] = embedding
                embeddings_list.append(embedding)
            else:
                entry["embedding"] = None

            self._entries.append(entry)

        # Build embedding matrix for batch similarity computation
        if embeddings_list and self.semantic_matcher:
            self._embeddings = np.vstack(embeddings_list)
        else:
            self._embeddings = None

        self._loaded = True
        print(f"[INFO] Loaded {len(self._entries)} procedure cache entries")

    def lookup(self, exercise_text: str, course_code: Optional[str] = None) -> Optional[CacheHit]:
        """
        Look up exercise in cache using two-stage matching:
        1. Embedding similarity (if semantic_matcher available)
        2. Text validation (normalized text comparison)

        Args:
            exercise_text: Exercise text to look up
            course_code: Optional course code filter

        Returns:
            CacheHit if match found above threshold, None otherwise
        """
        # Ensure cache is loaded
        if not self._loaded:
            self.load_cache(course_code)

        # Empty cache check
        if not self._entries:
            self.stats.misses += 1
            return None

        # Normalize exercise text
        normalized = self.normalize_exercise_text(exercise_text)

        # Stage 1: Embedding-based similarity (if available)
        best_match = None
        best_score = 0.0
        embedding_sim = 0.0
        text_sim = 0.0

        if self.semantic_matcher and self.semantic_matcher.enabled and self._embeddings is not None:
            # Get embedding for query exercise
            query_embedding = self.semantic_matcher.get_embedding(exercise_text)

            if query_embedding is not None:
                # Batch compute similarities with all cached embeddings
                similarities = self._batch_cosine_similarity(query_embedding, self._embeddings)

                # Find best match above threshold
                max_idx = np.argmax(similarities)
                max_sim = similarities[max_idx]

                if max_sim >= self._embedding_threshold:
                    candidate = self._entries[max_idx]

                    # Stage 2: Text validation
                    text_sim = SequenceMatcher(
                        None, normalized, candidate["normalized_text"]
                    ).ratio()

                    # Hybrid confidence score
                    hybrid_score = (max_sim * 0.7) + (text_sim * 0.3)

                    if hybrid_score >= self._hybrid_threshold:
                        best_match = candidate
                        best_score = hybrid_score
                        embedding_sim = max_sim

        # Fallback: Text-only matching (if no embeddings or embedding match failed)
        if best_match is None:
            for entry in self._entries:
                # Skip if course filter specified and doesn't match
                if course_code and entry["course_code"] and entry["course_code"] != course_code:
                    continue

                # Compute text similarity
                text_sim = SequenceMatcher(None, normalized, entry["normalized_text"]).ratio()

                if text_sim >= self._text_threshold and text_sim > best_score:
                    best_match = entry
                    best_score = text_sim
                    embedding_sim = 0.0

        # Return result
        if best_match:
            self.stats.hits += 1

            # Update hit count in database
            self._update_hit_count(best_match["id"])

            # Determine match type
            if embedding_sim > 0 and text_sim > 0:
                match_type = "hybrid"
            elif embedding_sim > 0:
                match_type = "embedding"
            else:
                match_type = "exact"

            return CacheHit(
                procedures=best_match["procedures"],
                topic=best_match["topic"],
                difficulty=best_match["difficulty"],
                variations=best_match["variations"],
                confidence=best_score,
                match_type=match_type,
                source_entry_id=best_match["id"],
                embedding_similarity=embedding_sim,
                text_similarity=text_sim,
            )
        else:
            self.stats.misses += 1
            return None

    def add(
        self,
        exercise_text: str,
        topic: str,
        difficulty: str,
        variations: List[str],
        procedures: List[Dict[str, Any]],
        confidence: float,
        course_code: Optional[str] = None,
    ):
        """
        Add new pattern to cache after LLM analysis.
        Thread-safe: creates its own db connection.

        Args:
            exercise_text: Original exercise text
            topic: Detected topic
            difficulty: Detected difficulty
            variations: Detected variations
            procedures: List of procedure dicts from analysis
            confidence: Analysis confidence
            course_code: Course code (None for global)
        """
        # Normalize text and compute hash
        normalized_text = self.normalize_exercise_text(exercise_text)
        pattern_hash = self.compute_pattern_hash(normalized_text)

        # Get embedding if semantic matcher available (before db operations)
        embedding_bytes = None
        embedding_array = None
        if self.semantic_matcher and self.semantic_matcher.enabled:
            embedding_array = self.semantic_matcher.get_embedding(exercise_text)
            if embedding_array is not None:
                embedding_bytes = embedding_array.astype(np.float32).tobytes()

        # Thread-safe: create new connection for db operations
        from storage.database import Database

        try:
            with Database() as thread_db:
                # Check if pattern already exists (avoid duplicates) - web-ready: scope by user_id
                if self.user_id is None:
                    cursor = thread_db.conn.execute(
                        "SELECT id FROM procedure_cache_entries WHERE pattern_hash = ? AND user_id IS NULL",
                        (pattern_hash,),
                    )
                else:
                    cursor = thread_db.conn.execute(
                        "SELECT id FROM procedure_cache_entries WHERE pattern_hash = ? AND user_id = ?",
                        (pattern_hash, self.user_id),
                    )
                if cursor.fetchone():
                    # Pattern already cached - skip
                    return

                # Insert into database (web-ready: include user_id)
                thread_db.conn.execute(
                    """
                    INSERT INTO procedure_cache_entries
                    (user_id, exercise_text_sample, normalized_text, pattern_hash, topic, difficulty,
                     variations_json, procedures_json, confidence_avg, course_code, embedding, match_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                    (
                        self.user_id,  # Web-ready: NULL for CLI, set for web
                        exercise_text,
                        normalized_text,
                        pattern_hash,
                        topic,
                        difficulty,
                        json.dumps(variations),
                        json.dumps(procedures),
                        confidence,
                        course_code,
                        embedding_bytes,
                    ),
                )
                thread_db.conn.commit()

                # Get the new entry ID
                cursor = thread_db.conn.execute("SELECT last_insert_rowid()")
                entry_id = cursor.fetchone()[0]
        except Exception as e:
            # Log but don't fail analysis for cache errors
            print(f"  [CACHE] Failed to add pattern: {e}")
            return

        # Add to in-memory cache
        new_entry = {
            "id": entry_id,
            "exercise_text": exercise_text,
            "normalized_text": normalized_text,
            "pattern_hash": pattern_hash,
            "topic": topic,
            "difficulty": difficulty,
            "variations": variations,
            "procedures": procedures,
            "confidence": confidence,
            "course_code": course_code,
            "embedding": embedding_array,
            "hit_count": 0,
            "created_at": datetime.now().isoformat(),
        }
        self._entries.append(new_entry)

        # Rebuild embedding matrix if using embeddings
        if self._embeddings is not None and embedding_array is not None:
            self._embeddings = np.vstack([self._embeddings, embedding_array])

    def clear(self, course_code: Optional[str] = None):
        """
        Clear cache entries.

        Args:
            course_code: Optional course code to clear. None clears all entries.
        """
        if not self.db.conn:
            self.db.connect()

        # Use database method with user_id isolation (web-ready)
        self.db.delete_procedure_cache(course_code=course_code, user_id=self.user_id)

        # Update in-memory cache
        if course_code:
            # Remove matching entries from memory
            self._entries = [e for e in self._entries if e["course_code"] != course_code]
        else:
            self._entries = []
            self._embeddings = None

        # Rebuild embedding matrix if needed
        if self._entries and self.semantic_matcher:
            embeddings_list = [e["embedding"] for e in self._entries if e["embedding"] is not None]
            if embeddings_list:
                self._embeddings = np.vstack(embeddings_list)
            else:
                self._embeddings = None

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "hit_rate": self.stats.hit_rate,
            "total_entries": len(self._entries),
            "loaded": self._loaded,
            "embeddings_enabled": self._embeddings is not None,
        }

    # Helper methods

    @staticmethod
    def normalize_exercise_text(text: str) -> str:
        """
        Normalize exercise text for consistent matching.

        Strips specific values while preserving structure:
        - "Design Mealy for 110" → "design mealy for [PATTERN]"
        - "Convert 0xFF to binary" → "convert [HEX] to binary"

        Args:
            text: Raw exercise text

        Returns:
            Normalized text with placeholders
        """
        normalized = text.lower().strip()

        # Replace specific patterns with placeholders
        normalized = re.sub(r"\b[01]{3,}\b", "[PATTERN]", normalized)  # Binary patterns
        normalized = re.sub(r"\b0x[0-9a-fA-F]+\b", "[HEX]", normalized)  # Hex values
        normalized = re.sub(r"\b\d{2,}\b", "[NUM]", normalized)  # Multi-digit numbers
        normalized = re.sub(r"\s+", " ", normalized)  # Normalize whitespace

        return normalized

    @staticmethod
    def compute_pattern_hash(normalized_text: str) -> str:
        """
        Compute hash for normalized text.

        Args:
            normalized_text: Normalized exercise text

        Returns:
            32-character hash string
        """
        return hashlib.sha256(normalized_text.encode()).hexdigest()[:32]

    def _batch_cosine_similarity(self, query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between query vector and all rows in matrix.

        Args:
            query_vec: Query embedding vector
            matrix: Matrix of cached embeddings (rows = entries)

        Returns:
            Array of similarity scores (0.0-1.0)
        """
        # Normalize query vector
        query_norm = query_vec / np.linalg.norm(query_vec)

        # Normalize matrix rows
        matrix_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix_norm = matrix / matrix_norms

        # Compute dot products (cosine similarities)
        similarities = np.dot(matrix_norm, query_norm)

        # Convert from [-1, 1] to [0, 1]
        similarities = (similarities + 1) / 2

        return similarities

    def _update_hit_count(self, entry_id: int):
        """
        Update hit count for cache entry.
        Thread-safe: creates its own db connection.

        Args:
            entry_id: Entry ID to update
        """
        try:
            # Create new connection for thread safety
            from storage.database import Database

            with Database() as thread_db:
                thread_db.conn.execute(
                    "UPDATE procedure_cache_entries SET match_count = match_count + 1, last_matched_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (entry_id,),
                )
                thread_db.conn.commit()
        except Exception:
            # Silently ignore hit count update failures (non-critical)
            pass

        # Update in-memory entry
        for entry in self._entries:
            if entry["id"] == entry_id:
                entry["hit_count"] = entry.get("hit_count", 0) + 1
                break
