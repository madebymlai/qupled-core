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

# Import sentence-transformers with fallback
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("[WARNING] sentence-transformers not available. Install with: pip install sentence-transformers")


# Translation pairs for common Computer Science terms (English ↔ Italian)
# Format: (english_term, italian_term)
TRANSLATION_PAIRS = {
    # FSM/Automata
    ("finite state machine", "macchina a stati finiti"),
    ("finite state machines", "macchine a stati finiti"),  # Plural form
    ("moore machine", "macchina di moore"),
    ("mealy machine", "macchina di mealy"),
    ("minimization", "minimizzazione"),
    ("state diagram", "diagramma degli stati"),
    ("state table", "tabella degli stati"),
    ("transition", "transizione"),
    ("automaton", "automa"),
    ("automata", "automi"),

    # Boolean Algebra
    ("boolean algebra", "algebra booleana"),
    ("karnaugh map", "mappa di karnaugh"),
    ("sum of products", "somma di prodotti"),
    ("product of sums", "prodotto di somme"),
    ("sop", "sop"),  # Same in both languages
    ("pos", "pos"),  # Same in both languages
    ("logic gate", "porta logica"),
    ("truth table", "tavola di verità"),

    # Circuits
    ("sequential circuit", "circuito sequenziale"),
    ("combinational circuit", "circuito combinatorio"),
    ("flip-flop", "flip-flop"),
    ("latch", "latch"),
    ("counter", "contatore"),

    # Design and Analysis
    ("design", "progettazione"),
    ("design", "disegno"),
    ("verification", "verifica"),
    ("implementation", "implementazione"),
    ("transformation", "trasformazione"),
    ("conversion", "conversione"),

    # Performance
    ("speedup", "accelerazione"),
    ("throughput", "throughput"),
    ("latency", "latenza"),
    ("bandwidth", "banda"),

    # Concurrent Programming
    ("monitor", "monitor"),
    ("semaphore", "semaforo"),
    ("mutex", "mutex"),
    ("synchronization", "sincronizzazione"),
    ("deadlock", "deadlock"),
    ("race condition", "race condition"),

    # Linear Algebra
    ("gaussian elimination", "eliminazione di gauss"),
    ("eigenvalue", "autovalore"),
    ("eigenvector", "autovettore"),
    ("diagonalization", "diagonalizzazione"),
    ("matrix", "matrice"),
    ("vector", "vettore"),

    # General terms
    ("procedure", "procedura"),
    ("algorithm", "algoritmo"),
    ("optimization", "ottimizzazione"),
    ("analysis", "analisi"),
}


# Known semantically different pairs (should NEVER merge)
# These are concepts that may have high string similarity but are semantically different
SEMANTIC_OPPOSITES = [
    ("mealy", "moore"),  # Different FSM types
    ("mealy to moore", "moore to mealy"),  # Inverse transformations
    ("mealy→moore", "moore→mealy"),  # Inverse transformations (arrow notation)
    ("conversione mealy moore", "conversione moore mealy"),  # Italian inverse transformations
    ("sum of products", "product of sums"),  # Full English names
    ("sop", "pos"),  # Sum of Products vs Product of Sums (abbreviations)
    ("somma di prodotti", "prodotto di somme"),  # Italian equivalents
    ("sequential", "combinational"),  # Different circuit types
    ("sequenziale", "combinatorio"),  # Italian equivalents
    ("combinational", "combinatory"),  # Similar strings, different meanings
    ("nfa", "dfa"),  # Nondeterministic vs Deterministic Finite Automaton
    ("synchronous", "asynchronous"),  # Timing types
    ("sincrono", "asincrono"),  # Italian timing types
    ("static", "dynamic"),
    ("statico", "dinamico"),
]


@dataclass
class SimilarityResult:
    """Result of similarity comparison."""
    should_merge: bool
    similarity_score: float
    reason: str
    metadata: Optional[Dict[str, Any]] = None


class SemanticMatcher:
    """Semantic similarity matcher using embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_embeddings: bool = True):
        """Initialize semantic matcher.

        Args:
            model_name: Name of the sentence transformer model to use
            use_embeddings: If False, use string matching fallback
        """
        self.model_name = model_name
        self.use_embeddings = use_embeddings and SENTENCE_TRANSFORMERS_AVAILABLE
        self.model = None

        # Build translation lookup tables for fast access
        self._build_translation_tables()

        # Initialize embedding model if requested
        if self.use_embeddings:
            try:
                print(f"[INFO] Loading sentence-transformers model: {model_name}")
                self.model = SentenceTransformer(model_name)
                self.enabled = True
                print(f"[INFO] Semantic matcher initialized with embeddings")
            except Exception as e:
                print(f"[WARNING] Failed to load semantic model: {e}")
                print(f"[WARNING] Falling back to string-based matching")
                self.enabled = False
                self.use_embeddings = False
        else:
            self.enabled = False
            print(f"[INFO] Semantic matcher initialized without embeddings (string fallback)")

    def _build_translation_tables(self):
        """Build bidirectional translation lookup tables."""
        self.en_to_it = {}
        self.it_to_en = {}

        for en_term, it_term in TRANSLATION_PAIRS:
            # Normalize to lowercase
            en_lower = en_term.lower()
            it_lower = it_term.lower()

            # Build bidirectional mapping
            self.en_to_it[en_lower] = it_lower
            self.it_to_en[it_lower] = en_lower

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

    def is_translation(self, text1: str, text2: str) -> bool:
        """
        Detect if two texts are translations of each other.

        This checks against known translation pairs for computer science terms.
        Uses a conservative approach: requires multiple translation pairs AND
        similar text structure to avoid false positives.

        Examples:
        - "Finite State Machines" ↔ "Macchine a Stati Finiti" → True
        - "Moore Machine" ↔ "Macchina di Moore" → True
        - "Mealy Machine" ↔ "Moore Machine" → False
        - "Implementazione Monitor" ↔ "Progettazione Monitor" → False (different verbs)

        Args:
            text1: First text
            text2: Second text

        Returns:
            True if texts are known translations
        """
        # Normalize texts
        t1_lower = text1.lower().strip()
        t2_lower = text2.lower().strip()

        # Check exact match (same text)
        if t1_lower == t2_lower:
            return True

        # Count translation pairs found
        translation_pairs_found = 0
        total_unique_words = len(set(t1_lower.split()) | set(t2_lower.split()))

        # Check if either text contains known translation pairs
        for en_term, it_term in TRANSLATION_PAIRS:
            # Check if both terms appear (one in each text)
            has_en_in_t1 = en_term in t1_lower
            has_it_in_t1 = it_term in t1_lower
            has_en_in_t2 = en_term in t2_lower
            has_it_in_t2 = it_term in t2_lower

            # Count translation pair if one has English term and other has Italian term
            if (has_en_in_t1 and has_it_in_t2) or (has_it_in_t1 and has_en_in_t2):
                translation_pairs_found += 1

        # Require at least 2 translation pairs to avoid false positives
        # (e.g., "Implementazione Monitor" vs "Progettazione Monitor" has only 1 pair: monitor)
        if translation_pairs_found >= 2:
            return True

        # Single translation pair is OK only if the texts are very short (2-4 words)
        # and the word counts are similar
        if translation_pairs_found == 1:
            t1_words = len(t1_lower.split())
            t2_words = len(t2_lower.split())
            if 2 <= t1_words <= 4 and 2 <= t2_words <= 4 and abs(t1_words - t2_words) <= 1:
                return True

        return False

    def are_semantically_different(self, text1: str, text2: str) -> bool:
        """
        Check if texts contain semantically opposite terms.

        This prevents merging of concepts that are fundamentally different
        despite potential string similarity.

        Examples:
        - "Mealy Machine Design" vs "Moore Machine Design" → True (different)
        - "Minimizzazione SoP" vs "Minimizzazione PoS" → True (different)
        - "Sequential Circuit" vs "Combinational Circuit" → True (different)

        Args:
            text1: First text
            text2: Second text

        Returns:
            True if texts contain semantically opposite concepts
        """
        import re

        t1_lower = text1.lower()
        t2_lower = text2.lower()

        for term1, term2 in SEMANTIC_OPPOSITES:
            # Use word boundaries to match whole words only (prevents "asynchronous" matching "synchronous")
            # For multi-word terms, use simple containment
            if ' ' in term1 or ' ' in term2:
                # Multi-word terms - use containment
                has_term1_in_t1 = term1 in t1_lower
                has_term2_in_t1 = term2 in t1_lower
                has_term1_in_t2 = term1 in t2_lower
                has_term2_in_t2 = term2 in t2_lower
            else:
                # Single-word terms - use word boundaries
                has_term1_in_t1 = bool(re.search(r'\b' + re.escape(term1) + r'\b', t1_lower))
                has_term2_in_t1 = bool(re.search(r'\b' + re.escape(term2) + r'\b', t1_lower))
                has_term1_in_t2 = bool(re.search(r'\b' + re.escape(term1) + r'\b', t2_lower))
                has_term2_in_t2 = bool(re.search(r'\b' + re.escape(term2) + r'\b', t2_lower))

            # Opposite if one has term1 and other has term2 (but not both in same text)
            if (has_term1_in_t1 and not has_term2_in_t1 and
                has_term2_in_t2 and not has_term1_in_t2):
                return True
            if (has_term2_in_t1 and not has_term1_in_t1 and
                has_term1_in_t2 and not has_term2_in_t2):
                return True

        return False

    # Keep old name for backward compatibility
    def are_semantic_opposites(self, name1: str, name2: str) -> bool:
        """Backward compatibility wrapper for are_semantically_different."""
        return self.are_semantically_different(name1, name2)

    def should_merge(self, name1: str, name2: str, threshold: float = 0.85) -> SimilarityResult:
        """
        Decide if two items should be merged.

        This implements a multi-stage matching approach:
        1. Check if semantically different (HIGHEST PRIORITY - prevents false merges)
        2. Check translation dictionary (exact matches)
        3. Compute semantic similarity

        Args:
            name1: First item name
            name2: Second item name
            threshold: Similarity threshold (defaults to 0.85)

        Returns:
            SimilarityResult with decision, score, and reason
        """
        # Stage 1: Check if semantically different (HIGHEST PRIORITY)
        # This prevents merging of Mealy/Moore, SoP/PoS, etc. even if they look similar
        if self.are_semantically_different(name1, name2):
            return SimilarityResult(
                should_merge=False,
                similarity_score=0.0,
                reason="semantically_different",
                metadata={"stage": 1, "method": "semantic_opposites"}
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

        if semantic_sim >= threshold:
            return SimilarityResult(
                should_merge=True,
                similarity_score=semantic_sim,
                reason="semantic_similarity",
                metadata={"stage": 3, "method": "embedding" if self.use_embeddings else "string"}
            )
        else:
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
        return {
            "model_name": self.model_name,
            "use_embeddings": self.use_embeddings,
            "embedding_available": SENTENCE_TRANSFORMERS_AVAILABLE,
            "translation_pairs_count": len(TRANSLATION_PAIRS),
            "semantic_opposites_count": len(SEMANTIC_OPPOSITES),
            "embedding_dimension": self.model.get_sentence_embedding_dimension() if self.model else None,
        }
