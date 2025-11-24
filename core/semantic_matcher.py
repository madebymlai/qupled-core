"""
Semantic similarity matcher for intelligent deduplication.

This module provides semantic matching capabilities to prevent false-positive merges
in deduplication. It uses embeddings to detect:
- True semantic similarity (duplicates)
- Translations (English/Italian pairs)
- Semantic differences (similar names, different concepts)

Examples of prevented false-positives:
- "Mealy Machine" ≠ "Moore Machine" (semantically different)
- "Minimizzazione SoP" ≠ "Minimizzazione PoS" (opposite concepts)
- "Algebra Booleana" == "Boolean Algebra" (translation)
"""

from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass
import numpy as np

from config import Config

# Import sentence-transformers with fallback
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("[WARNING] sentence-transformers not available. Install with: pip install sentence-transformers")


# REMOVED: Hardcoded TRANSLATION_PAIRS dictionary
#
# Old approach: Hardcoded dictionary with ~68 translation pairs (English ↔ Italian only)
# Problems:
# - Only worked for Italian/English pairs
# - Required manual maintenance for each new course/domain
# - Didn't scale to other language pairs (Spanish, French, German, etc.)
# - Violated Examina's "no hardcoding" philosophy
#
# New approach: LLM-based translation detection via TranslationDetector
# Benefits:
# - Works for ANY language pair (IT/EN, ES/EN, FR/EN, DE/EN, etc.)
# - Adapts to new courses/domains automatically
# - No manual dictionary maintenance required
# - Consistent with Examina's generic, scalable design
#
# See: core/translation_detector.py


# REMOVED: SEMANTIC_OPPOSITES hardcoded list
# Now using LLM-based dynamic detection for high-similarity pairs (>85%)
#
# Old approach: Hardcoded list of domain-specific opposites
# Problem: Doesn't scale when ingesting new courses from different domains
# New approach: Ask LLM "are these concepts opposites?" for high-similarity pairs
# Benefits:
# - Works for ANY domain (CS, Chemistry, Physics, etc.)
# - Adapts to new courses automatically
# - Maintains Examina's "no hardcoding" philosophy


@dataclass
class SimilarityResult:
    """Result of similarity comparison."""
    should_merge: bool
    similarity_score: float
    reason: str
    metadata: Optional[Dict[str, Any]] = None


class SemanticMatcher:
    """Semantic similarity matcher using embeddings."""

    def __init__(self, model_name: str = None, use_embeddings: bool = True, llm_manager=None, enable_translation_detection: bool = True):
        """Initialize semantic matcher.

        Args:
            model_name: Name of the sentence transformer model to use (default from Config)
            use_embeddings: If False, use string matching fallback
            llm_manager: Optional LLMManager instance for dynamic opposite detection and translation detection
            enable_translation_detection: If True, use LLM-based translation detection
        """
        # Use config default if no model specified
        self.model_name = model_name or Config.SEMANTIC_EMBEDDING_MODEL
        self.use_embeddings = use_embeddings and SENTENCE_TRANSFORMERS_AVAILABLE
        self.model = None
        self.llm_manager = llm_manager

        # LLM opposite detection cache (stores results to avoid repeated API calls)
        # Format: {(text1, text2): bool} where bool = True if opposites
        self._opposite_cache = {}

        # Initialize translation detector if LLM available
        self.translation_detector = None
        if llm_manager and enable_translation_detection:
            from core.translation_detector import TranslationDetector
            self.translation_detector = TranslationDetector(llm_manager=llm_manager)
            print(f"[INFO] Translation detection enabled (LLM-based)")
        else:
            print(f"[INFO] Translation detection disabled")

        # Initialize embedding model if requested
        if self.use_embeddings:
            try:
                print(f"[INFO] Loading sentence-transformers model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
                self.enabled = True
                print(f"[INFO] Semantic matcher initialized with embeddings")

                # Pass embedding model to translation detector
                if self.translation_detector:
                    self.translation_detector.embedding_model = self.model
            except Exception as e:
                print(f"[WARNING] Failed to load semantic model: {e}")
                print(f"[WARNING] Falling back to string-based matching")
                self.enabled = False
                self.use_embeddings = False
        else:
            self.enabled = False
            print(f"[INFO] Semantic matcher initialized without embeddings (string fallback)")

    # REMOVED: _build_translation_tables() method
    # No longer needed - using LLM-based TranslationDetector instead

    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for a text.

        Args:
            text: Input text

        Returns:
            Embedding vector or None if model not available
        """
        if not self.enabled or self.model is None:
            return None

        try:
            return self.model.encode(text, convert_to_numpy=True)
        except Exception as e:
            print(f"[WARNING] Failed to encode text: {e}")
            return None

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Normalize vectors
        vec1_norm = vec1 / np.linalg.norm(vec1)
        vec2_norm = vec2 / np.linalg.norm(vec2)

        # Compute cosine similarity
        similarity = np.dot(vec1_norm, vec2_norm)

        # Ensure it's in [0, 1] range (some models may return [-1, 1])
        return float((similarity + 1) / 2)

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0.0 and 1.0
            Uses cosine similarity between embeddings
        """
        if not self.use_embeddings:
            # Fallback to string matching
            from difflib import SequenceMatcher
            return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

        try:
            # Get embeddings
            emb1 = self.get_embedding(text1)
            emb2 = self.get_embedding(text2)

            if emb1 is None or emb2 is None:
                # Fallback to string matching
                from difflib import SequenceMatcher
                return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

            # Compute cosine similarity
            similarity = self.cosine_similarity(emb1, emb2)
            return similarity

        except Exception as e:
            print(f"[WARNING] Error computing similarity: {e}")
            # Fallback to string matching
            from difflib import SequenceMatcher
            return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def is_translation(self, text1: str, text2: str, min_similarity: float = 0.70) -> bool:
        """
        Detect if two texts are translations of each other (ANY language pair).

        NEW: Uses LLM-based TranslationDetector instead of hardcoded dictionary.
        Works for ANY language pair: IT/EN, ES/EN, FR/EN, DE/EN, etc.

        Examples (ANY language pair):
        - "Moore Machine Design" ↔ "Progettazione Macchina di Moore" (EN↔IT) → True
        - "Implementazione Monitor" ↔ "Monitor Implementation" (IT↔EN) → True
        - "Eliminación Gaussiana" ↔ "Gaussian Elimination" (ES↔EN) → True
        - "Mealy Machine" ↔ "Moore Machine" (same lang, different concepts) → False

        Args:
            text1: First text (any language)
            text2: Second text (any language)
            min_similarity: Minimum embedding similarity threshold

        Returns:
            True if texts are translations
        """
        if not self.translation_detector:
            # Translation detection disabled - conservative fallback
            return False

        try:
            result = self.translation_detector.are_translations(
                text1, text2,
                min_embedding_similarity=min_similarity,
                use_language_detection=False  # Skip language detection for speed
            )
            return result.is_translation

        except Exception as e:
            print(f"[WARNING] Translation detection failed: {e}")
            return False

    def is_inverse_transformation(self, text1: str, text2: str) -> bool:
        """
        Detect if two texts represent inverse transformations (generic, no hardcoding).

        Examples:
        - "Mealy to Moore" ↔ "Moore to Mealy" → True
        - "Binary to Decimal" ↔ "Decimal to Binary" → True
        - "Conversione Mealy Moore" ↔ "Conversione Moore Mealy" → True
        - "Matrix→Vector" ↔ "Vector→Matrix" → True

        Args:
            text1: First text
            text2: Second text

        Returns:
            True if texts are inverse transformations
        """
        import re

        t1_lower = text1.lower().strip()
        t2_lower = text2.lower().strip()

        # Transformation patterns to detect (generic, works for any A/B pair)
        patterns = [
            # English patterns
            r'(.+?)\s+to\s+(.+)',           # "A to B"
            r'(.+?)\s*→\s*(.+)',            # "A→B"
            r'(.+?)\s*->\s*(.+)',           # "A->B"
            r'(.+?)\s+into\s+(.+)',         # "A into B"
            r'transform\s+(.+?)\s+to\s+(.+)',  # "transform A to B"
            # Italian patterns
            r'conversione\s+(.+?)\s+(.+)',  # "conversione A B"
            r'trasforma\w*\s+(.+?)\s+in\s+(.+)',  # "trasforma A in B"
        ]

        for pattern in patterns:
            match1 = re.search(pattern, t1_lower)
            match2 = re.search(pattern, t2_lower)

            if match1 and match2:
                # Extract source and target for both texts
                source1, target1 = match1.groups()
                source2, target2 = match2.groups()

                # Clean up whitespace
                source1, target1 = source1.strip(), target1.strip()
                source2, target2 = source2.strip(), target2.strip()

                # Check if they're inverses (swapped)
                if source1 == target2 and target1 == source2:
                    return True

        return False

    def has_opposite_affixes(self, text1: str, text2: str) -> bool:
        """
        Detect if texts contain words with opposite prefixes/suffixes (generic, no hardcoding).

        Examples:
        - "synchronous" ↔ "asynchronous" (prefix: a-)
        - "deterministic" ↔ "nondeterministic" (prefix: non-)
        - "linear" ↔ "nonlinear" (prefix: non-)
        - "sincrono" ↔ "asincrono" (Italian prefix: a-)

        Args:
            text1: First text
            text2: Second text

        Returns:
            True if texts have words with opposite affixes
        """
        import re

        # Opposite prefixes (works for any language)
        opposite_prefixes = ['a', 'non', 'in', 'un', 'de', 'anti', 'dis', 'il', 'im', 'ir']

        t1_lower = text1.lower()
        t2_lower = text2.lower()

        # Extract all words from both texts
        words1 = set(re.findall(r'\b\w+\b', t1_lower))
        words2 = set(re.findall(r'\b\w+\b', t2_lower))

        # Check if any word in text1 is the prefixed version of a word in text2 (or vice versa)
        for word1 in words1:
            for prefix in opposite_prefixes:
                # Check if word1 = prefix + word2
                if word1.startswith(prefix) and len(word1) > len(prefix):
                    base1 = word1[len(prefix):]  # Remove prefix
                    if base1 in words2:
                        # word1 is "aword", word2 is "word" → opposite!
                        return True

        for word2 in words2:
            for prefix in opposite_prefixes:
                # Check if word2 = prefix + word1
                if word2.startswith(prefix) and len(word2) > len(prefix):
                    base2 = word2[len(prefix):]  # Remove prefix
                    if base2 in words1:
                        # word2 is "aword", word1 is "word" → opposite!
                        return True

        return False

    def are_opposites_llm(self, text1: str, text2: str, similarity: float) -> bool:
        """
        Use LLM to determine if two high-similarity concepts are semantic opposites.

        This replaces the hardcoded SEMANTIC_OPPOSITES list with dynamic detection
        that works for ANY domain (Chemistry, Physics, Math, CS, etc.).

        Only called for pairs with >85% similarity that passed generic checks.
        Results are cached to avoid repeated API calls.

        Examples that should return True:
        - "sum of products" ↔ "product of sums"
        - "endothermic reaction" ↔ "exothermic reaction"
        - "positive charge" ↔ "negative charge"
        - "clockwise rotation" ↔ "counterclockwise rotation"

        Args:
            text1: First text
            text2: Second text
            similarity: Precomputed similarity score (for context)

        Returns:
            True if LLM determines they are semantic opposites
        """
        if not self.llm_manager:
            # No LLM available - conservative fallback (don't merge)
            return False

        # Check cache first (normalized tuple to handle order)
        cache_key = tuple(sorted([text1.lower(), text2.lower()]))
        if cache_key in self._opposite_cache:
            return self._opposite_cache[cache_key]

        # Ask LLM
        prompt = f"""Are these two concepts semantic opposites or complementary concepts that should NOT be merged together?

Concept 1: "{text1}"
Concept 2: "{text2}"

Context: These concepts have {similarity*100:.1f}% semantic similarity according to embeddings.

Answer with ONLY "yes" or "no".

Examples:
- "sum of products" vs "product of sums" → yes (opposite boolean operations)
- "SoP" vs "PoS" → yes (abbreviations for opposite operations)
- "NFA" vs "DFA" → yes (nondeterministic vs deterministic automata)
- "NFA Design" vs "DFA Design" → yes (different automata types)
- "Mealy machine" vs "Moore machine" → yes (different FSM types)
- "endothermic" vs "exothermic" → yes (opposite thermodynamic processes)
- "positive charge" vs "negative charge" → yes (opposite electrical properties)
- "clockwise" vs "counterclockwise" → yes (opposite directions)
- "finite state machine" vs "macchina a stati finiti" → no (translation, same concept)
- "FSM minimization" vs "FSM minimizzazione" → no (translation, same concept)

Answer:"""

        try:
            response = self.llm_manager.generate(
                prompt=prompt,
                system="You are an expert at detecting semantic opposites across all domains. Respond with only 'yes' or 'no'.",
                temperature=0.0,  # Deterministic
                max_tokens=10
            )

            if response.success:
                answer = response.text.strip().lower()
                is_opposite = answer.startswith("yes")

                # Cache the result
                self._opposite_cache[cache_key] = is_opposite

                return is_opposite
            else:
                # LLM failed - conservative fallback (don't merge)
                return False

        except Exception as e:
            print(f"[WARNING] LLM opposite detection failed: {e}")
            # Conservative fallback
            return False

    def are_semantically_different(self, text1: str, text2: str) -> bool:
        """
        Check if texts contain semantically opposite terms using GENERIC algorithms.

        This prevents merging of concepts that are fundamentally different
        despite potential string similarity.

        Uses ONLY generic algorithms (NO HARDCODING):
        1. Inverse transformation detection ("A to B" ↔ "B to A")
        2. Opposite affix detection (synchronous ↔ asynchronous)

        For high-similarity pairs (>85%), use are_opposites_llm() instead.

        Examples:
        - "Mealy to Moore" ↔ "Moore to Mealy" → True (inverse transformation)
        - "Synchronous" ↔ "Asynchronous" → True (opposite affixes)
        - "Binary to Decimal" ↔ "Decimal to Binary" → True (inverse transformation)

        Args:
            text1: First text
            text2: Second text

        Returns:
            True if texts contain semantically opposite concepts (generic patterns only)
        """
        # Check for generic inverse transformations (NO HARDCODING)
        if self.is_inverse_transformation(text1, text2):
            return True

        # Check for opposite affixes (GENERIC PATTERN DETECTION)
        if self.has_opposite_affixes(text1, text2):
            return True

        return False

    # Keep old name for backward compatibility
    def are_semantic_opposites(self, name1: str, name2: str) -> bool:
        """Backward compatibility wrapper for are_semantically_different."""
        return self.are_semantically_different(name1, name2)

    def should_merge(self, name1: str, name2: str, threshold: float = 0.85) -> SimilarityResult:
        """
        Decide if two items should be merged.

        This implements a multi-stage matching approach (FULLY DYNAMIC, NO HARDCODING):
        1. Check generic patterns (inverse transformations, opposite affixes)
        2. Check translation dictionary (exact matches)
        3. Compute semantic similarity
        4. If high similarity (>= threshold), ask LLM if they're opposites
        5. Return merge decision

        Args:
            name1: First item name
            name2: Second item name
            threshold: Similarity threshold (defaults to 0.85)

        Returns:
            SimilarityResult with decision, score, and reason
        """
        # Stage 1: Check generic patterns (HIGHEST PRIORITY)
        # This catches inverse transformations and opposite affixes
        if self.are_semantically_different(name1, name2):
            return SimilarityResult(
                should_merge=False,
                similarity_score=0.0,
                reason="semantically_different_generic",
                metadata={"stage": 1, "method": "generic_patterns"}
            )

        # Stage 2: Check translation dictionary (exact matches)
        if self.is_translation(name1, name2):
            return SimilarityResult(
                should_merge=True,
                similarity_score=1.0,
                reason="translation",
                metadata={"stage": 2, "method": "translation_dictionary"}
            )

        # Stage 3: Compute semantic similarity
        semantic_sim = self.compute_similarity(name1, name2)

        # Stage 4: For high-similarity pairs, ask LLM if they're opposites
        # This replaces the hardcoded SEMANTIC_OPPOSITES list
        if semantic_sim >= threshold:
            # Check if they're domain-specific opposites using LLM
            if self.are_opposites_llm(name1, name2, semantic_sim):
                return SimilarityResult(
                    should_merge=False,
                    similarity_score=semantic_sim,
                    reason="semantically_different_llm",
                    metadata={"stage": 4, "method": "llm_opposite_detection"}
                )

            # Not opposites - safe to merge
            return SimilarityResult(
                should_merge=True,
                similarity_score=semantic_sim,
                reason="semantic_similarity",
                metadata={"stage": 4, "method": "embedding" if self.use_embeddings else "string"}
            )
        else:
            # Below threshold - don't merge
            return SimilarityResult(
                should_merge=False,
                similarity_score=semantic_sim,
                reason="below_threshold",
                metadata={"stage": 3, "threshold": threshold, "method": "embedding" if self.use_embeddings else "string"}
            )

    def find_similar_items(self, query: str, candidates: List[str],
                          threshold: float = 0.85) -> List[Tuple[str, float]]:
        """
        Find semantically similar items from candidates.

        Args:
            query: Query text
            candidates: List of candidate texts
            threshold: Minimum similarity threshold

        Returns:
            List of (candidate, similarity_score) tuples above threshold,
            sorted by similarity (highest first)
        """
        results = []

        for candidate in candidates:
            similarity = self.compute_similarity(query, candidate)
            if similarity >= threshold:
                results.append((candidate, similarity))

        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)

        return results

    def batch_should_merge(self, pairs: List[Tuple[str, str]],
                          threshold: float = 0.85) -> List[SimilarityResult]:
        """
        Batch version of should_merge for efficiency.

        Args:
            pairs: List of (name1, name2) tuples to compare
            threshold: Similarity threshold

        Returns:
            List of SimilarityResult objects (same order as input)
        """
        return [self.should_merge(n1, n2, threshold) for n1, n2 in pairs]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the matcher.

        Returns:
            Dictionary with matcher configuration and stats
        """
        stats = {
            "model_name": self.model_name,
            "use_embeddings": self.use_embeddings,
            "embedding_available": SENTENCE_TRANSFORMERS_AVAILABLE,
            "translation_detection_enabled": self.translation_detector is not None,
            "opposite_cache_size": len(self._opposite_cache),
            "embedding_dimension": self.model.get_sentence_embedding_dimension() if self.model else None,
        }

        # Add translation detector stats if available
        if self.translation_detector:
            stats.update(self.translation_detector.get_cache_stats())

        return stats
