"""
AI-powered exercise analyzer for Examina.
Handles exercise merging, topic discovery, and core loop identification.
"""

import json
import re
import time
import asyncio
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor, as_completed

from models.llm_manager import LLMManager, LLMResponse
from storage.database import Database
from config import Config

# Type checking imports (avoid circular dependencies)
if TYPE_CHECKING:
    from core.procedure_cache import ProcedureCache, CacheHit

# Try to import semantic matcher, fallback to string similarity if not available
try:
    from core.semantic_matcher import SemanticMatcher
    SEMANTIC_MATCHING_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] SemanticMatcher not available: {e}")
    print("  Falling back to string-based similarity matching")
    SEMANTIC_MATCHING_AVAILABLE = False


@dataclass
class ProcedureInfo:
    """Information about a single procedure/algorithm in an exercise."""
    name: str
    type: str  # design, transformation, verification, minimization, analysis, other
    steps: List[str]
    point_number: Optional[int] = None
    transformation: Optional[Dict[str, str]] = None  # {"source_format": "X", "target_format": "Y"}


@dataclass
class AnalysisResult:
    """Result of exercise analysis."""
    is_valid_exercise: bool
    is_fragment: bool
    should_merge_with_previous: bool
    topic: Optional[str]
    difficulty: Optional[str]
    variations: Optional[List[str]]
    confidence: float
    procedures: List[ProcedureInfo]  # NEW: Multiple procedures support

    # Phase 9.1: Exercise type detection
    exercise_type: Optional[str] = 'procedural'  # 'procedural', 'theory', 'proof', 'hybrid'
    type_confidence: float = 0.0  # Confidence in type classification
    proof_keywords: Optional[List[str]] = None  # Detected proof keywords if any
    theory_metadata: Optional[Dict[str, Any]] = None  # Theory-specific metadata

    # Phase 9.2: Theory question categorization
    theory_category: Optional[str] = None  # 'definition', 'theorem', 'proof', 'explanation', 'derivation', 'concept'
    theorem_name: Optional[str] = None  # Name of theorem if applicable
    concept_id: Optional[str] = None  # ID of main concept
    prerequisite_concepts: Optional[List[str]] = None  # List of prerequisite concept IDs

    # Backward compatibility fields (derived from first procedure)
    @property
    def core_loop_id(self) -> Optional[str]:
        """Primary core loop ID (first procedure)."""
        if self.procedures:
            return self._normalize_core_loop_id(self.procedures[0].name)
        return None

    @property
    def core_loop_name(self) -> Optional[str]:
        """Primary core loop name (first procedure)."""
        if self.procedures:
            return self.procedures[0].name
        return None

    @property
    def procedure(self) -> Optional[List[str]]:
        """Primary procedure steps (first procedure)."""
        if self.procedures:
            return self.procedures[0].steps
        return None

    @staticmethod
    def _normalize_core_loop_id(core_loop_name: Optional[str]) -> Optional[str]:
        """Normalize core loop name to ID."""
        if not core_loop_name:
            return None
        core_loop_id = core_loop_name.lower()
        core_loop_id = re.sub(r'[^\w\s-]', '', core_loop_id)
        core_loop_id = re.sub(r'[\s-]+', '_', core_loop_id)
        return core_loop_id


class ExerciseAnalyzer:
    """Analyzes exercises using LLM to discover topics and core loops."""

    def __init__(self, llm_manager: Optional[LLMManager] = None, language: str = "en",
                 monolingual: bool = False, procedure_cache: Optional['ProcedureCache'] = None):
        """Initialize analyzer.

        Args:
            llm_manager: LLM manager instance
            language: Output language for analysis (any ISO 639-1 code, e.g., "en", "de", "zh")
            monolingual: Enable strictly monolingual mode (all procedures in single language)
            procedure_cache: Optional ProcedureCache instance for pattern caching (Option 3 optimization)
        """
        self.llm = llm_manager or LLMManager()
        self.language = language
        self.monolingual = monolingual
        self.primary_language = None  # Will be detected from course metadata/exercises

        # Option 3: Procedure Pattern Caching
        self.procedure_cache = procedure_cache
        self.cache_stats = {'hits': 0, 'misses': 0}

        # Initialize translation detector for monolingual mode
        self.translation_detector = None
        if self.monolingual:
            try:
                from core.translation_detector import TranslationDetector
                self.translation_detector = TranslationDetector(llm_manager=self.llm)
                print("[INFO] Translation detector initialized for monolingual mode")
            except Exception as e:
                print(f"[WARNING] Failed to initialize TranslationDetector for monolingual mode: {e}")
                print("  Monolingual mode will be disabled")
                self.monolingual = False  # Disable if can't initialize

        # Initialize semantic matcher if available
        if SEMANTIC_MATCHING_AVAILABLE and Config.SEMANTIC_SIMILARITY_ENABLED:
            try:
                self.semantic_matcher = SemanticMatcher()
                self.use_semantic = self.semantic_matcher.enabled
                if self.use_semantic:
                    print("[INFO] Semantic similarity matching enabled")
                else:
                    print("[INFO] Semantic matcher loaded but model unavailable, using string similarity")
            except Exception as e:
                print(f"[WARNING] Failed to initialize SemanticMatcher: {e}")
                self.semantic_matcher = None
                self.use_semantic = False
        else:
            self.semantic_matcher = None
            self.use_semantic = False
            if not SEMANTIC_MATCHING_AVAILABLE:
                print("[INFO] Semantic matching not available, using string similarity")
            else:
                print("[INFO] Semantic matching disabled in config, using string similarity")

    def _language_instruction(self, action: str = "Respond") -> str:
        """Generate dynamic language instruction for any language."""
        return f"{action} in {self.language.upper()} language."

    def _lang_instruction(self) -> str:
        """Generate language instruction phrase for any language."""
        return f"in {self.language.upper()} language"

    def _language_name(self) -> str:
        """Get full language name for prompts."""
        return self.language.upper()

    def analyze_exercise(self, exercise_text: str, course_name: str,
                        previous_exercise: Optional[str] = None,
                        existing_context: Optional[Dict[str, Any]] = None) -> AnalysisResult:
        """Analyze a single exercise.

        Args:
            exercise_text: Exercise text
            course_name: Course name for context
            previous_exercise: Previous exercise text (for merge detection)
            existing_context: Optional dict with existing entities for context-aware analysis:
                - topics: List[dict] with {"name": str, "procedure_count": int}
                - procedures: List[dict] with {"name": str, "type": str}
                - concepts: List[dict] with {"name": str, "type": str, "topic": str}

        Returns:
            AnalysisResult with classification
        """
        # Build prompt
        prompt = self._build_analysis_prompt(
            exercise_text, course_name, previous_exercise, existing_context
        )

        # Call LLM
        response = self.llm.generate(
            prompt=prompt,
            model=self.llm.primary_model,
            temperature=0.3,  # Lower temp for more consistent analysis
            json_mode=True
        )

        if not response.success:
            # Log error and return default
            print(f"[ERROR] LLM failed for exercise: {response.error}")
            print(f"  Text preview: {exercise_text[:100]}...")
            return self._default_analysis_result()

        # Parse JSON response
        data = self.llm.parse_json_response(response)
        if not data:
            return self._default_analysis_result()

        # Parse procedures (new format) or fallback to old format
        procedures = []
        if "procedures" in data and data["procedures"]:
            # New format: multiple procedures
            for proc_data in data["procedures"]:
                procedures.append(ProcedureInfo(
                    name=proc_data.get("name", "Unknown Procedure"),
                    type=proc_data.get("type", "other"),
                    steps=proc_data.get("steps", []),
                    point_number=proc_data.get("point_number"),
                    transformation=proc_data.get("transformation")
                ))
        elif "core_loop_name" in data and data["core_loop_name"]:
            # Old format: single procedure - convert to new format
            procedures.append(ProcedureInfo(
                name=data["core_loop_name"],
                type="other",  # Unknown type in old format
                steps=data.get("procedure", []),
                point_number=None,
                transformation=None
            ))

        # Normalize procedures to primary language if monolingual mode enabled
        if self.monolingual and procedures:
            procedures = self._normalize_procedures_to_primary_language(procedures)

        # Phase 9.1: Extract exercise type information
        exercise_type = data.get("exercise_type", "procedural")
        type_confidence = data.get("type_confidence", 0.5)
        proof_keywords = data.get("proof_keywords", [])
        theory_metadata = data.get("theory_metadata")

        # Phase 9.2: Extract theory categorization information
        theory_category = data.get("theory_category")
        theorem_name = data.get("theorem_name")
        concept_id = data.get("concept_id")
        prerequisite_concepts = data.get("prerequisite_concepts")

        # Extract fields
        return AnalysisResult(
            is_valid_exercise=data.get("is_valid_exercise", True),
            is_fragment=data.get("is_fragment", False),
            should_merge_with_previous=data.get("should_merge_with_previous", False),
            topic=data.get("topic"),
            difficulty=data.get("difficulty"),
            variations=data.get("variations", []),
            confidence=data.get("confidence", 0.5),
            procedures=procedures,
            exercise_type=exercise_type,
            type_confidence=type_confidence,
            proof_keywords=proof_keywords if proof_keywords else None,
            theory_metadata=theory_metadata,
            theory_category=theory_category,
            theorem_name=theorem_name,
            concept_id=concept_id,
            prerequisite_concepts=prerequisite_concepts
        )

    def _build_analysis_prompt(self, exercise_text: str, course_name: str,
                               previous_exercise: Optional[str],
                               existing_context: Optional[Dict[str, Any]] = None) -> str:
        """Build prompt for exercise analysis.

        Args:
            exercise_text: Exercise text
            course_name: Course name
            previous_exercise: Previous exercise (for merge detection)
            existing_context: Optional dict with existing topics/procedures/concepts

        Returns:
            Prompt string
        """
        base_prompt = f"""You are analyzing exam exercises for the course: {course_name}.

Your task is to analyze this text and determine:
1. Is it a valid, complete exercise? Or just exam instructions/headers?
2. Is it a fragment that should be merged with other parts?
3. What topic does it cover?
4. What is the core solving procedure (core loop)?

EXERCISE TEXT:
```
{exercise_text[:2000]}
```
"""

        if previous_exercise:
            base_prompt += f"""
PREVIOUS EXERCISE:
```
{previous_exercise[:1000]}
```

Does this exercise appear to be a continuation or sub-part of the previous one?
"""

        # Add existing context if provided (Phase 3: Context Injection)
        if existing_context:
            base_prompt += self._build_context_section(existing_context)

        # Add language instruction (supports any ISO 639-1 language)
        base_prompt += f"""
IMPORTANT: {self._language_instruction("Respond")} All topic names, procedures and steps must be in {self._language_name()} language.

Respond in JSON format with:
{{
  "is_valid_exercise": true/false,  // false if it's just exam instructions or headers
  "is_fragment": true/false,  // true if incomplete or part of larger exercise
  "should_merge_with_previous": true/false,  // true if continuation of previous
  "topic": "SPECIFIC topic name",  // MUST be specific, NOT generic course name!
  "difficulty": "easy|medium|hard",
  "variations": ["variation1", ...],  // specific variants used
  "confidence": 0.0-1.0,  // your confidence in this analysis
  "procedures": [  // ALL distinct procedures/algorithms required (NEW: can be multiple!)
    {{
      "name": "procedure name",  // e.g., "[Entity] Design", "[Entity A] to [Entity B] Conversion"
      "type": "design|transformation|verification|minimization|analysis|other",
      "steps": ["step 1", "step 2", ...],  // solving steps for this procedure
      "point_number": 1,  // which numbered point (1, 2, 3, etc.) - null if not applicable
      "transformation": {{  // ONLY if type=transformation
        "source_format": "format name",  // source entity/format from exercise
        "target_format": "format name"   // target entity/format from exercise
      }}
    }}
  ],
  "exercise_type": "procedural|theory|proof|hybrid",  // Type of exercise (Phase 9.1)
  "type_confidence": 0.0-1.0,  // Confidence in exercise type classification
  "proof_keywords": ["keyword1", ...],  // Detected proof keywords if any
  "theory_metadata": {{  // Theory-specific metadata (optional)
    "theorem_name": "name if applicable",
    "requires_definition": true/false,
    "requires_explanation": true/false
  }},
  "theory_category": "definition|theorem|axiom|property|explanation|derivation|concept|null",  // Phase 9.2: Theory category
  "theorem_name": "specific theorem name if asking about a theorem",  // Phase 9.2: exact name from exercise
  "concept_id": "normalized_concept_id",  // Phase 9.2: snake_case version of concept name
  "prerequisite_concepts": ["concept_id_1", "concept_id_2"]  // Phase 9.2: concepts needed to understand this
}}

IMPORTANT ANALYSIS GUIDELINES:
- If text contains only exam rules (like "NON si può usare la calcolatrice"), mark as NOT valid exercise
- If text is clearly a sub-question (starts with "1.", "2.") right after numbered list, it's a fragment
- Core loop/procedure is the ALGORITHM/PROCEDURE to solve, not just the topic

TOPIC NAMING RULES (CRITICAL):
- NEVER use the course name "{course_name}" as the topic - it's too generic!
- Topics should be at CHAPTER/UNIT level - like a section in a textbook or syllabus
- Topic = PROBLEM DOMAIN (what kind of problems), Procedure = SOLUTION METHOD (how to solve)

TOPIC GRANULARITY TEST:
Ask: "Would this topic have 3-10 related procedures/methods?"
- If YES → Good topic level (chapter-level)
- If only 1 procedure possible → Too specific! Find the broader problem domain
- If 50+ procedures → Too broad! Split into subtopics

TOPIC vs PROCEDURE PATTERN:
A TOPIC is a problem domain (broad category).
A PROCEDURE is a specific method/algorithm used to solve problems in that domain.

Pattern: One topic can have MULTIPLE related procedures:
- Topic: [Broad Problem Domain]
  - Procedure: [Method A for this domain]
  - Procedure: [Method B for this domain]
  - Procedure: [Conversion between formats in this domain]
  - Procedure: [Verification/checking in this domain]

BAD TOPIC NAMES (too granular - these should be procedures!):
- A specific technique name → Should be a procedure under broader topic
- A single formula/law → Should be a procedure under analysis topic
- A single conversion method → Should be a procedure under broader topic

GOOD TOPIC NAMES (chapter-level):
- Broad categories that appear in course syllabus
- Topics that have multiple related methods/procedures
- Problem domains, not individual solution techniques

MULTI-PROCEDURE DETECTION:
- If exercise has numbered points (1., 2., 3.), analyze EACH point separately
- Each distinct procedure should have its own entry in "procedures" array
- Set "point_number" to indicate which numbered point it belongs to
- If exercise requires multiple procedures (e.g., "design AND verify"), list ALL of them
- For transformations/conversions (A→B, X→Y, etc.), set type="transformation" and fill "transformation" object

PROCEDURE NAMING (CRITICAL - be specific, not generic!):
Procedure names must include the SPECIFIC ENTITY being acted upon, not category/topic names.

Pattern: [Action Type] + [Specific Entity from Exercise Text]

How to identify the entity:
1. Find the concrete noun in the exercise that the action operates on
2. It should be a specific name that appears in the exercise text
3. NOT a broad category that could contain multiple different things

Test: "Does this exact name appear in the exercise as the thing being worked on?"
- YES → Use it in the procedure name
- NO → You're using a category name, look for the actual entity

PROCEDURE IDENTITY RULES (CRITICAL - when to keep separate vs merge):
Two procedures are THE SAME if they:
- Solve the EXACT same problem type with the SAME algorithm
- Only differ in language or minor phrasing

Two procedures are DIFFERENT (keep separate!) if they:
- Have different procedure types (design ≠ verify ≠ minimize)
- Transform in different directions (A→B ≠ B→A)
- Use fundamentally different algorithms for same goal

MUST STAY SEPARATE (even if in same topic):
- "Design X" vs "Verify X" vs "Minimize X" (different procedure types)
- "A to B Conversion" vs "B to A Conversion" (different transformation directions)
- "Method 1 for X" vs "Method 2 for X" (different algorithms for same goal)

CAN BE MERGED (same procedure, name variations):
- Same algorithm with different word order: "X Design" = "Design X"
- Same algorithm in different languages: English name = translated name
- Same algorithm with abbreviation: Full name = common abbreviation

EXERCISE TYPE CLASSIFICATION (Phase 9.1):
- **procedural**: Exercise requires applying an algorithm/procedure to solve a problem
  * Pattern: "Design...", "Calculate...", "Solve...", "Convert...", "Implement..."
  * Has clear input → process → output structure
  * Focuses on HOW to solve (execution of steps)

- **theory**: Exercise asks for definitions, explanations, or conceptual understanding
  * Pattern: "Define...", "What is...", "Explain...", "Describe..."
  * Asks for understanding, not computation
  * No procedural work required
  * NOTE: Even ONE theory keyword (define, explain, what is, etc.) is sufficient to classify as theory

- **proof**: Exercise requires proving a theorem, property, or statement
  * KEYWORDS to detect (Italian): "dimostra", "dimostrare", "dimostrazione", "provare", "prova che"
  * KEYWORDS to detect (English): "prove", "proof", "show that", "demonstrate that", "verify that"
  * Requires logical reasoning and mathematical proof structure
  * May ask to prove theorems, properties, or general statements

- **hybrid**: Exercise combines multiple types (e.g., prove something AND apply it to compute)
  * Has both procedural and theory/proof components
  * Pattern: "Prove... and then use it to..."

PROOF KEYWORD DETECTION:
- Scan the exercise text for proof keywords in BOTH Italian and English
- If found, classify as "proof" or "hybrid" (if also has procedural component)
- Store detected keywords in "proof_keywords" array
- Set type_confidence higher (0.9+) when proof keywords are explicitly present

THEORY QUESTION CATEGORIZATION (Phase 9.2):
For exercises with exercise_type="theory" or "hybrid", categorize into specific theory categories.
NOTE: Presence of even ONE theory keyword from any category below is sufficient for classification.

**definition**: Asks for a formal definition of a concept, term, or object
  * Keywords (IT): "definisci", "definizione", "cos'è", "cosa si intende per"
  * Keywords (EN): "define", "definition", "what is", "what does ... mean"
  * Pattern: "Give the definition of X"
  * Set concept_id to normalized name of the concept

**theorem**: Asks to state, explain, or apply a specific theorem
  * Keywords (IT): "enunciare", "teorema", "enunciato"
  * Keywords (EN): "state the theorem", "theorem", "proposition"
  * Pattern: "State theorem X", "Explain theorem Y"
  * Set theorem_name to specific theorem name
  * Set concept_id to theorem identifier

**axiom**: Asks about axioms, postulates, or fundamental properties
  * Keywords (IT): "assioma", "assiomi", "proprietà fondamentale", "postulato"
  * Keywords (EN): "axiom", "axioms", "postulate", "fundamental property"
  * Pattern: "List the axioms of X"
  * Set concept_id to axiom system

**property**: Asks about properties, characteristics, or conditions
  * Keywords (IT): "proprietà", "caratteristica", "condizione"
  * Keywords (EN): "property", "characteristic", "condition"
  * Pattern: "What properties does X have?"
  * Set concept_id to property being discussed

**explanation**: Asks to explain HOW or WHY something works
  * Keywords (IT): "spiega", "spiegare", "come funziona", "perché", "illustra"
  * Keywords (EN): "explain", "how does", "why", "illustrate", "describe how"
  * Pattern: "Explain how X works", "Why does Y happen?"
  * Set concept_id to concept being explained

**derivation**: Asks to derive or show how to obtain a result
  * Keywords (IT): "deriva", "derivare", "come si ottiene", "ricava"
  * Keywords (EN): "derive", "obtain", "show how to get"
  * Pattern: "Derive formula X", "Show how to obtain Y"
  * Set concept_id to formula/result being derived

**concept**: General conceptual question not fitting other categories
  * Use for understanding checks, conceptual comparisons, relationships
  * Pattern: "What is the relationship between X and Y?"
  * Set concept_id to main concept discussed

PREREQUISITE CONCEPT DETECTION (CRITICAL - for ALL exercise types):
- Identify theoretical concepts, methods, or techniques used in this exercise
- Set prerequisite_concepts for BOTH procedural AND theory exercises
- For procedural exercises: extract the underlying concepts the procedure applies
- For theory exercises: extract foundational concepts needed to understand this one
- List 3-7 most important concepts as normalized IDs
- ALWAYS fill this field - every exercise uses theoretical concepts!

CONCEPT ID NORMALIZATION:
- Convert concept names to lowercase IDs with underscores
- Pattern: "Concept Name" → "concept_name"
- Remove articles, prepositions when possible
- Keep acronyms/numbers as-is: "X 123" → "x_123"

IMPORTANT:
- theory_category is ONLY for theory/hybrid exercises (null for procedural)
- For hybrid exercises, fill BOTH procedures array AND theory fields
- If exercise asks definition AND computation, mark as hybrid with both

BACKWARD COMPATIBILITY:
- Even if exercise has only ONE procedure, still return it in "procedures" array
- Extract actual solving steps if you can identify them
- The first procedure in the array is considered the PRIMARY procedure

Respond ONLY with valid JSON, no other text.
"""

        return base_prompt

    def _build_context_section(self, existing_context: Dict[str, Any]) -> str:
        """Build the context section for the prompt with existing entities.

        Args:
            existing_context: Dict with topics, procedures, concepts lists

        Returns:
            Formatted context string for the prompt
        """
        sections = []

        # Add existing topics
        topics = existing_context.get("topics", [])
        if topics:
            topic_lines = []
            for t in topics[:20]:  # Limit to avoid prompt bloat
                name = t.get("name", t) if isinstance(t, dict) else t
                count = t.get("procedure_count", 0) if isinstance(t, dict) else 0
                if count:
                    topic_lines.append(f"  - {name} ({count} procedures)")
                else:
                    topic_lines.append(f"  - {name}")
            sections.append(f"""
EXISTING TOPICS in this course (PREFER these if exercise fits):
{chr(10).join(topic_lines)}""")

        # Add existing procedures
        procedures = existing_context.get("procedures", [])
        if procedures:
            proc_lines = []
            for p in procedures[:30]:  # Limit to avoid prompt bloat
                name = p.get("name", p) if isinstance(p, dict) else p
                ptype = p.get("type", "unknown") if isinstance(p, dict) else "unknown"
                proc_lines.append(f"  - {name} (type: {ptype})")
            sections.append(f"""
EXISTING PROCEDURES (PREFER these if exercise uses same method):
{chr(10).join(proc_lines)}""")

        # Add existing concepts
        concepts = existing_context.get("concepts", [])
        if concepts:
            concept_lines = []
            for c in concepts[:20]:  # Limit to avoid prompt bloat
                name = c.get("name", c) if isinstance(c, dict) else c
                ctype = c.get("type", "unknown") if isinstance(c, dict) else "unknown"
                topic = c.get("topic", "") if isinstance(c, dict) else ""
                if topic:
                    concept_lines.append(f"  - {name} (type: {ctype}, topic: {topic})")
                else:
                    concept_lines.append(f"  - {name} (type: {ctype})")
            sections.append(f"""
EXISTING CONCEPTS (PREFER these for theory questions):
{chr(10).join(concept_lines)}""")

        if sections:
            context_rules = """
CONTEXT RULES (CRITICAL):
1. PREFER existing topic if exercise fits semantically - do NOT create similar/duplicate topics
2. PREFER existing procedure if solving same problem with same method
3. PREFER existing concept if asking about same theoretical entity
4. Create NEW topic only if exercise doesn't fit ANY existing topic
5. Create NEW procedure only if method is genuinely different
6. When in doubt, USE EXISTING names over creating new ones
7. Topics should be CHAPTER-LEVEL (containing 3-10 related procedures), not individual techniques"""
            return "".join(sections) + context_rules

        return ""

    def _normalize_core_loop_id(self, core_loop_name: Optional[str]) -> Optional[str]:
        """Normalize core loop name to ID.

        Args:
            core_loop_name: Human-readable name

        Returns:
            Normalized ID (lowercase, underscores)
        """
        if not core_loop_name:
            return None

        # Convert to lowercase, replace spaces with underscores
        core_loop_id = core_loop_name.lower()
        core_loop_id = re.sub(r'[^\w\s-]', '', core_loop_id)
        core_loop_id = re.sub(r'[\s-]+', '_', core_loop_id)

        return core_loop_id

    def _default_analysis_result(self) -> AnalysisResult:
        """Return default analysis result on error."""
        return AnalysisResult(
            is_valid_exercise=True,
            is_fragment=False,
            should_merge_with_previous=False,
            topic=None,
            difficulty=None,
            variations=None,
            confidence=0.0,
            procedures=[],  # Empty procedures list
            exercise_type='procedural',  # Default type
            type_confidence=0.0,
            proof_keywords=None,
            theory_metadata=None
        )

    def _build_result_from_cache(self, cache_hit: 'CacheHit', exercise_text: str) -> AnalysisResult:
        """Build AnalysisResult from cache hit (Option 3: Procedure Pattern Caching).

        Args:
            cache_hit: Cache hit result containing cached procedures and metadata
            exercise_text: Original exercise text

        Returns:
            AnalysisResult built from cached data
        """
        from core.procedure_cache import CacheHit

        # Convert cached procedures to ProcedureInfo objects
        # Handle two formats:
        # 1. List of dicts (from analyzer): [{'name': '...', 'type': '...', 'steps': [...]}]
        # 2. List of strings (from build command): ['Step 1', 'Step 2', 'Step 3']
        procedures = []

        if not cache_hit.procedures:
            # Empty procedures list
            pass
        elif isinstance(cache_hit.procedures[0], str):
            # Format 2: List of step strings - wrap in single ProcedureInfo
            procedures.append(ProcedureInfo(
                name=cache_hit.topic or 'Procedure',
                type='other',
                steps=cache_hit.procedures,  # Use the string list as steps
                point_number=None,
                transformation=None
            ))
        else:
            # Format 1: List of ProcedureInfo dicts
            for p in cache_hit.procedures:
                procedures.append(ProcedureInfo(
                    name=p.get('name', 'Unknown'),
                    type=p.get('type', 'other'),
                    steps=p.get('steps', []),
                    point_number=p.get('point_number'),
                    transformation=p.get('transformation')
                ))

        return AnalysisResult(
            is_valid_exercise=True,
            is_fragment=False,
            should_merge_with_previous=False,
            topic=cache_hit.topic,
            difficulty=cache_hit.difficulty,
            variations=cache_hit.variations,
            confidence=cache_hit.confidence,
            procedures=procedures,
            exercise_type='procedural'  # Default - cache doesn't store this
        )

    def _detect_primary_language(self, exercises: List[Dict[str, Any]], course_name: str) -> str:
        """Detect the primary language of the course.

        Args:
            exercises: List of exercise dicts
            course_name: Course name for additional context

        Returns:
            Primary language code (e.g., "english", "italian")
        """
        if not self.translation_detector:
            # Fallback to analysis language if no detector (supports any language)
            return self.language

        # Sample first few exercises to detect language
        sample_size = min(5, len(exercises))
        language_counts = {}

        for exercise in exercises[:sample_size]:
            text = exercise.get('text', '')
            if not text:
                continue

            detected_lang = self.translation_detector.detect_language(text)
            if detected_lang and detected_lang != "unknown":
                language_counts[detected_lang] = language_counts.get(detected_lang, 0) + 1

        # Get most common language
        if language_counts:
            primary_lang = max(language_counts, key=language_counts.get)
            print(f"[INFO] Detected primary course language: {primary_lang} (from {sample_size} exercises)")
            return primary_lang
        else:
            # Fallback to analysis language (supports any language)
            print(f"[INFO] Could not detect language, using fallback: {self.language}")
            return self.language

    def _translate_procedure(self, procedure_info: ProcedureInfo, target_language: str) -> ProcedureInfo:
        """Translate a procedure to target language.

        Args:
            procedure_info: Procedure to translate
            target_language: Target language (e.g., "english", "italian")

        Returns:
            Translated ProcedureInfo
        """
        if not self.llm:
            return procedure_info

        # Build translation prompt
        prompt = f"""Translate this procedure name and steps to {target_language}.
Maintain technical accuracy and preserve the exact meaning.

Original procedure:
Name: {procedure_info.name}
Type: {procedure_info.type}
Steps:
{chr(10).join([f"{i+1}. {step}" for i, step in enumerate(procedure_info.steps)])}

Return ONLY valid JSON in this format:
{{
  "name": "translated procedure name",
  "steps": ["translated step 1", "translated step 2", ...]
}}

No markdown code blocks, just JSON."""

        try:
            response = self.llm.generate(
                prompt=prompt,
                temperature=0.3,
                json_mode=True
            )

            if response.success:
                data = self.llm.parse_json_response(response)
                if data and 'name' in data and 'steps' in data:
                    return ProcedureInfo(
                        name=data['name'],
                        type=procedure_info.type,
                        steps=data['steps'],
                        point_number=procedure_info.point_number,
                        transformation=procedure_info.transformation
                    )
        except Exception as e:
            print(f"[WARNING] Failed to translate procedure '{procedure_info.name}': {e}")

        # Fallback: return original
        return procedure_info

    def _normalize_procedures_to_primary_language(self, procedures: List[ProcedureInfo]) -> List[ProcedureInfo]:
        """Normalize all procedures to primary language in monolingual mode.

        Args:
            procedures: List of procedures to normalize

        Returns:
            List of procedures in primary language
        """
        if not self.monolingual or not self.primary_language or not self.translation_detector:
            return procedures

        normalized_procedures = []

        for proc in procedures:
            # Detect language of procedure
            proc_language = self.translation_detector.detect_language(proc.name)

            if proc_language == "unknown":
                # Can't detect, keep as is
                normalized_procedures.append(proc)
                continue

            # Check if procedure is in primary language
            if proc_language.lower() == self.primary_language.lower():
                # Already in primary language
                normalized_procedures.append(proc)
            else:
                # Translate to primary language
                print(f"[MONOLINGUAL] Translating procedure '{proc.name}' from {proc_language} to {self.primary_language}")
                translated_proc = self._translate_procedure(proc, self.primary_language)
                print(f"  → '{translated_proc.name}'")
                normalized_procedures.append(translated_proc)

        return normalized_procedures

    def merge_exercises(self, exercises: List[Dict[str, Any]], skip_analyzed: bool = False) -> List[Dict[str, Any]]:
        """Merge exercise fragments into complete exercises.

        Args:
            exercises: List of exercise dicts from database
            skip_analyzed: If True, skip exercises already marked as analyzed

        Returns:
            List of merged exercises
        """
        if not exercises:
            return []

        merged = []
        current_merge = None

        for i, exercise in enumerate(exercises):
            # Skip already analyzed exercises if requested
            if skip_analyzed and exercise.get('analyzed'):
                # If we have a current merge, save it before skipping
                if current_merge:
                    merged.append(current_merge)
                    current_merge = None
                print(f"[DEBUG] Skipping already analyzed exercise: {exercise['id'][:40]}...")
                continue

            # Analyze exercise
            previous_text = current_merge["text"] if current_merge else None
            if i > 0 and not current_merge:
                previous_text = exercises[i-1].get("text")

            analysis = self.analyze_exercise(
                exercise["text"],
                "Computer Architecture",  # TODO: get from exercise
                previous_text
            )

            # Skip invalid exercises (exam instructions, etc.)
            if not analysis.is_valid_exercise:
                print(f"[DEBUG] Skipping invalid exercise: {exercise['id'][:20]}... ({exercise['text'][:60]}...)")
                continue

            # Should merge with previous?
            if analysis.should_merge_with_previous and current_merge:
                # Merge into current
                current_merge["text"] += "\n\n" + exercise["text"]
                current_merge["merged_from"].append(exercise["id"])
                if exercise.get("image_paths"):
                    if not current_merge.get("image_paths"):
                        current_merge["image_paths"] = []
                    current_merge["image_paths"].extend(exercise["image_paths"])
            else:
                # Save previous merge if exists
                if current_merge:
                    merged.append(current_merge)

                # Start new exercise (or merge)
                current_merge = {
                    **exercise,
                    "merged_from": [exercise["id"]],
                    "analysis": analysis
                }

        # Don't forget last one
        if current_merge:
            merged.append(current_merge)

        return merged

    def _analyze_exercise_with_retry(self, exercise_text: str, course_name: str,
                                     previous_exercise: Optional[str] = None,
                                     max_retries: int = 2) -> AnalysisResult:
        """Analyze exercise with retry logic for failed API calls.

        Args:
            exercise_text: Exercise text
            course_name: Course name for context
            previous_exercise: Previous exercise text (for merge detection)
            max_retries: Maximum number of retries on failure

        Returns:
            AnalysisResult with classification
        """
        # Check procedure cache first (Option 3: Performance Optimization)
        if self.procedure_cache and Config.PROCEDURE_CACHE_ENABLED:
            cache_hit = self.procedure_cache.lookup(exercise_text, course_code=course_name)
            if cache_hit and cache_hit.confidence >= Config.PROCEDURE_CACHE_MIN_CONFIDENCE:
                self.cache_stats['hits'] += 1
                # Build result from cache
                return self._build_result_from_cache(cache_hit, exercise_text)
            else:
                # No cache hit or low confidence - track as miss
                self.cache_stats['misses'] += 1

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                result = self.analyze_exercise(exercise_text, course_name, previous_exercise)
                # Check if analysis was successful
                if result.confidence > 0.0 or result.topic is not None:
                    # Add to procedure cache for future use (Option 3)
                    if self.procedure_cache and Config.PROCEDURE_CACHE_ENABLED and result.procedures:
                        self.procedure_cache.add(
                            exercise_text=exercise_text,
                            topic=result.topic or '',
                            difficulty=result.difficulty or 'medium',
                            variations=result.variations or [],
                            procedures=[p.__dict__ if hasattr(p, '__dict__') else p for p in result.procedures],
                            confidence=result.confidence,
                            course_code=course_name
                        )
                    return result
                # If we got default result, retry
                if attempt < max_retries:
                    print(f"  Retry {attempt + 1}/{max_retries} for exercise...")
                    time.sleep(1 * (attempt + 1))  # Exponential backoff
                    continue
                return result
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    print(f"  Error on attempt {attempt + 1}: {str(e)}, retrying...")
                    time.sleep(1 * (attempt + 1))
                    continue

        # All retries failed, return default
        print(f"  All retries failed: {last_error}")
        return self._default_analysis_result()

    async def _analyze_exercise_with_retry_async(self, exercise_text: str, course_name: str,
                                                  previous_exercise: Optional[str] = None,
                                                  max_retries: int = 2) -> AnalysisResult:
        """Analyze exercise asynchronously with retry logic for failed API calls.

        Args:
            exercise_text: Exercise text
            course_name: Course name for context
            previous_exercise: Previous exercise text (for merge detection)
            max_retries: Maximum number of retries on failure

        Returns:
            AnalysisResult with classification
        """
        # Check procedure cache first (Option 3: Performance Optimization)
        # Note: Cache lookup is synchronous, but fast (in-memory)
        if self.procedure_cache and Config.PROCEDURE_CACHE_ENABLED:
            cache_hit = self.procedure_cache.lookup(exercise_text, course_code=course_name)
            if cache_hit and cache_hit.confidence >= Config.PROCEDURE_CACHE_MIN_CONFIDENCE:
                self.cache_stats['hits'] += 1
                # Build result from cache
                return self._build_result_from_cache(cache_hit, exercise_text)
            else:
                # No cache hit or low confidence - track as miss
                self.cache_stats['misses'] += 1

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                # Build prompt
                prompt = self._build_analysis_prompt(
                    exercise_text, course_name, previous_exercise
                )

                # Call LLM asynchronously
                response = await self.llm.generate_async(
                    prompt=prompt,
                    model=self.llm.primary_model,
                    temperature=0.3,
                    json_mode=True
                )

                if not response.success:
                    # Log error and retry or return default
                    if attempt < max_retries:
                        print(f"  Retry {attempt + 1}/{max_retries} for exercise (error: {response.error})...")
                        await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff
                        continue
                    print(f"[ERROR] LLM failed for exercise: {response.error}")
                    print(f"  Text preview: {exercise_text[:100]}...")
                    return self._default_analysis_result()

                # Parse JSON response
                data = self.llm.parse_json_response(response)
                if not data:
                    if attempt < max_retries:
                        print(f"  Retry {attempt + 1}/{max_retries} for exercise (invalid JSON)...")
                        await asyncio.sleep(1 * (attempt + 1))
                        continue
                    return self._default_analysis_result()

                # Parse procedures (new format) or fallback to old format
                procedures = []
                if "procedures" in data and data["procedures"]:
                    # New format: multiple procedures
                    for proc_data in data["procedures"]:
                        procedures.append(ProcedureInfo(
                            name=proc_data.get("name", "Unknown Procedure"),
                            type=proc_data.get("type", "other"),
                            steps=proc_data.get("steps", []),
                            point_number=proc_data.get("point_number"),
                            transformation=proc_data.get("transformation")
                        ))
                elif "core_loop_name" in data and data["core_loop_name"]:
                    # Old format: single procedure - convert to new format
                    procedures.append(ProcedureInfo(
                        name=data["core_loop_name"],
                        type="other",
                        steps=data.get("procedure", []),
                        point_number=None,
                        transformation=None
                    ))

                # Normalize procedures to primary language if monolingual mode enabled
                if self.monolingual and procedures:
                    procedures = self._normalize_procedures_to_primary_language(procedures)

                # Phase 9.1: Extract exercise type information
                exercise_type = data.get("exercise_type", "procedural")
                type_confidence = data.get("type_confidence", 0.5)
                proof_keywords = data.get("proof_keywords", [])
                theory_metadata = data.get("theory_metadata")

                # Phase 9.2: Extract theory categorization information
                theory_category = data.get("theory_category")
                theorem_name = data.get("theorem_name")
                concept_id = data.get("concept_id")
                prerequisite_concepts = data.get("prerequisite_concepts")

                # Build result
                result = AnalysisResult(
                    is_valid_exercise=data.get("is_valid_exercise", True),
                    is_fragment=data.get("is_fragment", False),
                    should_merge_with_previous=data.get("should_merge_with_previous", False),
                    topic=data.get("topic"),
                    difficulty=data.get("difficulty"),
                    variations=data.get("variations", []),
                    confidence=data.get("confidence", 0.5),
                    procedures=procedures,
                    exercise_type=exercise_type,
                    type_confidence=type_confidence,
                    proof_keywords=proof_keywords if proof_keywords else None,
                    theory_metadata=theory_metadata,
                    theory_category=theory_category,
                    theorem_name=theorem_name,
                    concept_id=concept_id,
                    prerequisite_concepts=prerequisite_concepts
                )

                # Check if analysis was successful
                if result.confidence > 0.0 or result.topic is not None:
                    # Add to procedure cache for future use (Option 3)
                    if self.procedure_cache and Config.PROCEDURE_CACHE_ENABLED and result.procedures:
                        self.procedure_cache.add(
                            exercise_text=exercise_text,
                            topic=result.topic or '',
                            difficulty=result.difficulty or 'medium',
                            variations=result.variations or [],
                            procedures=[p.__dict__ if hasattr(p, '__dict__') else p for p in result.procedures],
                            confidence=result.confidence,
                            course_code=course_name
                        )
                    return result

                # If we got default result, retry
                if attempt < max_retries:
                    print(f"  Retry {attempt + 1}/{max_retries} for exercise (low confidence)...")
                    await asyncio.sleep(1 * (attempt + 1))
                    continue

                return result

            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    print(f"  Error on attempt {attempt + 1}: {str(e)}, retrying...")
                    await asyncio.sleep(1 * (attempt + 1))
                    continue

        # All retries failed, return default
        print(f"  All retries failed: {last_error}")
        return self._default_analysis_result()

    def merge_exercises_parallel(self, exercises: List[Dict[str, Any]],
                                 batch_size: Optional[int] = None,
                                 show_progress: bool = True,
                                 skip_analyzed: bool = False) -> List[Dict[str, Any]]:
        """Merge exercise fragments using parallel batch processing.

        This method analyzes exercises in parallel batches to improve performance.
        Each batch is processed concurrently, with retry logic for failed exercises.

        Args:
            exercises: List of exercise dicts from database
            batch_size: Number of exercises to process in parallel (defaults to Config.BATCH_SIZE)
            show_progress: Show progress bar during processing
            skip_analyzed: If True, skip exercises already marked as analyzed

        Returns:
            List of merged exercises
        """
        if not exercises:
            return []

        batch_size = batch_size or Config.BATCH_SIZE
        total = len(exercises)

        print(f"[INFO] Starting parallel batch analysis of {total} exercises (batch_size={batch_size})...")
        start_time = time.time()

        # Store analysis results indexed by exercise position
        analysis_results = {}

        # Process in batches
        def analyze_single(index: int, exercise: Dict[str, Any], prev_text: Optional[str]) -> tuple:
            """Analyze a single exercise and return (index, analysis, error)."""
            try:
                # Skip already analyzed if requested
                if skip_analyzed and exercise.get('analyzed'):
                    return (index, None, None)  # Signal to skip

                analysis = self._analyze_exercise_with_retry(
                    exercise["text"],
                    "Computer Architecture",  # TODO: get from exercise
                    prev_text
                )
                return (index, analysis, None)
            except Exception as e:
                print(f"  [ERROR] Failed to analyze exercise {index}: {str(e)}")
                return (index, self._default_analysis_result(), str(e))

        # Process exercises in batches with ThreadPoolExecutor
        processed = 0
        failed_count = 0
        skipped_count = 0

        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch = exercises[batch_start:batch_end]
            batch_indices = list(range(batch_start, batch_end))

            if show_progress:
                print(f"  Processing batch {batch_start//batch_size + 1}/{(total + batch_size - 1)//batch_size} (exercises {batch_start+1}-{batch_end}/{total})...")

            # Prepare analysis tasks for this batch
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = {}

                for i, (idx, exercise) in enumerate(zip(batch_indices, batch)):
                    # Determine previous exercise text for merge detection
                    prev_text = None
                    if idx > 0:
                        prev_text = exercises[idx - 1].get("text")

                    future = executor.submit(analyze_single, idx, exercise, prev_text)
                    futures[future] = idx

                # Collect results as they complete
                for future in as_completed(futures):
                    idx, analysis, error = future.result()

                    if analysis is None:  # Skipped exercise
                        skipped_count += 1
                    else:
                        analysis_results[idx] = analysis

                    processed += 1

                    if error:
                        failed_count += 1

                    if show_progress and processed % 5 == 0:
                        elapsed = time.time() - start_time
                        rate = processed / elapsed if elapsed > 0 else 0
                        eta = (total - processed) / rate if rate > 0 else 0
                        print(f"    Progress: {processed}/{total} ({100*processed/total:.1f}%) | {rate:.1f} ex/s | ETA: {eta:.0f}s")

        elapsed_time = time.time() - start_time

        print(f"[INFO] Batch analysis complete in {elapsed_time:.1f}s")
        print(f"  Processed: {processed}/{total} exercises")
        if skipped_count > 0:
            print(f"  Skipped (already analyzed): {skipped_count} exercises")
        print(f"  Failed: {failed_count} exercises")
        print(f"  Rate: {processed/elapsed_time:.1f} exercises/second")

        # Now merge exercises sequentially based on analysis results
        print(f"[INFO] Merging exercise fragments...")
        merged = []
        current_merge = None

        for i, exercise in enumerate(exercises):
            # Check if skipped
            if i not in analysis_results:
                if current_merge:
                    merged.append(current_merge)
                    current_merge = None
                continue

            analysis = analysis_results[i]

            # Skip invalid exercises
            if not analysis.is_valid_exercise:
                print(f"[DEBUG] Skipping invalid exercise: {exercise['id'][:20]}... ({exercise['text'][:60]}...)")
                continue

            # Should merge with previous?
            if analysis.should_merge_with_previous and current_merge:
                # Merge into current
                current_merge["text"] += "\n\n" + exercise["text"]
                current_merge["merged_from"].append(exercise["id"])
                if exercise.get("image_paths"):
                    if not current_merge.get("image_paths"):
                        current_merge["image_paths"] = []
                    current_merge["image_paths"].extend(exercise["image_paths"])
            else:
                # Save previous merge if exists
                if current_merge:
                    merged.append(current_merge)

                # Start new exercise
                current_merge = {
                    **exercise,
                    "merged_from": [exercise["id"]],
                    "analysis": analysis
                }

        # Don't forget last one
        if current_merge:
            merged.append(current_merge)

        print(f"[INFO] Merged {len(exercises)} fragments → {len(merged)} complete exercises")

        return merged

    async def merge_exercises_async(self, exercises: List[Dict[str, Any]],
                                    batch_size: Optional[int] = None,
                                    show_progress: bool = True,
                                    skip_analyzed: bool = False) -> List[Dict[str, Any]]:
        """Merge exercise fragments using async batch processing.

        This method analyzes exercises in async batches to improve performance.
        Each batch is processed concurrently using asyncio.gather(), with retry logic for failed exercises.

        Args:
            exercises: List of exercise dicts from database
            batch_size: Number of exercises to process concurrently (defaults to Config.BATCH_SIZE)
            show_progress: Show progress bar during processing
            skip_analyzed: If True, skip exercises already marked as analyzed

        Returns:
            List of merged exercises
        """
        if not exercises:
            return []

        batch_size = batch_size or Config.BATCH_SIZE
        total = len(exercises)

        print(f"[INFO] Starting async batch analysis of {total} exercises (batch_size={batch_size})...")
        start_time = time.time()

        # Store analysis results indexed by exercise position
        analysis_results = {}

        # Process in batches
        async def analyze_single(index: int, exercise: Dict[str, Any], prev_text: Optional[str]) -> tuple:
            """Analyze a single exercise and return (index, analysis, error)."""
            try:
                # Skip already analyzed if requested
                if skip_analyzed and exercise.get('analyzed'):
                    return (index, None, None)  # Signal to skip

                analysis = await self._analyze_exercise_with_retry_async(
                    exercise["text"],
                    "Computer Architecture",  # TODO: get from exercise
                    prev_text
                )
                return (index, analysis, None)
            except Exception as e:
                print(f"  [ERROR] Failed to analyze exercise {index}: {str(e)}")
                return (index, self._default_analysis_result(), str(e))

        # Process exercises in batches with asyncio.gather()
        processed = 0
        failed_count = 0
        skipped_count = 0

        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch = exercises[batch_start:batch_end]
            batch_indices = list(range(batch_start, batch_end))

            if show_progress:
                print(f"  Processing batch {batch_start//batch_size + 1}/{(total + batch_size - 1)//batch_size} (exercises {batch_start+1}-{batch_end}/{total})...")

            # Prepare analysis tasks for this batch
            tasks = []
            for idx, exercise in zip(batch_indices, batch):
                # Determine previous exercise text for merge detection
                prev_text = None
                if idx > 0:
                    prev_text = exercises[idx - 1].get("text")

                task = analyze_single(idx, exercise, prev_text)
                tasks.append(task)

            # Run all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for result in results:
                if isinstance(result, Exception):
                    # Task raised an exception
                    print(f"  [ERROR] Task failed with exception: {result}")
                    failed_count += 1
                    processed += 1
                    continue

                idx, analysis, error = result

                if analysis is None:  # Skipped exercise
                    skipped_count += 1
                else:
                    analysis_results[idx] = analysis

                processed += 1

                if error:
                    failed_count += 1

                if show_progress and processed % 5 == 0:
                    elapsed = time.time() - start_time
                    rate = processed / elapsed if elapsed > 0 else 0
                    eta = (total - processed) / rate if rate > 0 else 0
                    print(f"    Progress: {processed}/{total} ({100*processed/total:.1f}%) | {rate:.1f} ex/s | ETA: {eta:.0f}s")

        elapsed_time = time.time() - start_time

        print(f"[INFO] Async batch analysis complete in {elapsed_time:.1f}s")
        print(f"  Processed: {processed}/{total} exercises")
        if skipped_count > 0:
            print(f"  Skipped (already analyzed): {skipped_count} exercises")
        print(f"  Failed: {failed_count} exercises")
        print(f"  Rate: {processed/elapsed_time:.1f} exercises/second")

        # Now merge exercises sequentially based on analysis results
        print(f"[INFO] Merging exercise fragments...")
        merged = []
        current_merge = None

        for i, exercise in enumerate(exercises):
            # Check if skipped
            if i not in analysis_results:
                if current_merge:
                    merged.append(current_merge)
                    current_merge = None
                continue

            analysis = analysis_results[i]

            # Skip invalid exercises
            if not analysis.is_valid_exercise:
                print(f"[DEBUG] Skipping invalid exercise: {exercise['id'][:20]}... ({exercise['text'][:60]}...)")
                continue

            # Should merge with previous?
            if analysis.should_merge_with_previous and current_merge:
                # Merge into current
                current_merge["text"] += "\n\n" + exercise["text"]
                current_merge["merged_from"].append(exercise["id"])
                if exercise.get("image_paths"):
                    if not current_merge.get("image_paths"):
                        current_merge["image_paths"] = []
                    current_merge["image_paths"].extend(exercise["image_paths"])
            else:
                # Save previous merge if exists
                if current_merge:
                    merged.append(current_merge)

                # Start new exercise
                current_merge = {
                    **exercise,
                    "merged_from": [exercise["id"]],
                    "analysis": analysis
                }

        # Don't forget last one
        if current_merge:
            merged.append(current_merge)

        print(f"[INFO] Merged {len(exercises)} fragments → {len(merged)} complete exercises")

        return merged

    def discover_topics_and_core_loops(self, course_code: str,
                                      batch_size: int = 10,
                                      skip_analyzed: bool = False,
                                      use_parallel: bool = True) -> Dict[str, Any]:
        """Discover topics and core loops for a course.

        Args:
            course_code: Course code
            batch_size: Number of exercises to analyze at once
            skip_analyzed: If True, skip already analyzed exercises
            use_parallel: If True, use parallel batch processing (default: True)

        Returns:
            Dict with topics and core loops discovered
        """
        with Database() as db:
            # Get all exercises for course
            exercises = db.get_exercises_by_course(course_code)

            if not exercises:
                return {"topics": {}, "core_loops": {}}

            # Detect primary language if monolingual mode enabled
            if self.monolingual and not self.primary_language:
                course = db.get_course(course_code)
                course_name = course['name'] if course else course_code
                self.primary_language = self._detect_primary_language(exercises, course_name)
                print(f"[MONOLINGUAL MODE] Primary language set to: {self.primary_language}")
                print(f"  All procedures will be normalized to {self.primary_language}\n")

            # Merge fragments first - use parallel or sequential mode
            if use_parallel:
                merged_exercises = self.merge_exercises_parallel(
                    exercises,
                    batch_size=batch_size,
                    skip_analyzed=skip_analyzed
                )
            else:
                merged_exercises = self.merge_exercises(exercises, skip_analyzed=skip_analyzed)

            # Collect all analyses
            topics = {}
            core_loops = {}
            low_confidence_count = 0

            for merged_ex in merged_exercises:
                analysis = merged_ex.get("analysis")
                if not analysis:
                    continue

                # Skip low-confidence analyses
                if analysis.confidence < Config.MIN_ANALYSIS_CONFIDENCE:
                    low_confidence_count += 1
                    print(f"[INFO] Skipping exercise due to low confidence ({analysis.confidence:.2f} < {Config.MIN_ANALYSIS_CONFIDENCE}): {merged_ex['id'][:40]}...")
                    # Mark exercise as skipped in metadata
                    merged_ex["low_confidence_skipped"] = True
                    continue

                # Track topic
                if analysis.topic:
                    if analysis.topic not in topics:
                        topics[analysis.topic] = {
                            "name": analysis.topic,
                            "exercise_count": 0,
                            "core_loops": set()
                        }
                    topics[analysis.topic]["exercise_count"] += 1

                # Process ALL procedures (new multi-procedure support)
                if analysis.procedures:
                    # Log if multiple procedures detected
                    if len(analysis.procedures) > 1:
                        print(f"[INFO] Multiple procedures detected ({len(analysis.procedures)}) in exercise {merged_ex['id'][:40]}:")
                        for i, proc in enumerate(analysis.procedures, 1):
                            print(f"  {i}. {proc.name} (type: {proc.type}, point: {proc.point_number})")

                    # Process each procedure
                    for procedure_info in analysis.procedures:
                        core_loop_id = self._normalize_core_loop_id(procedure_info.name)

                        # Track core loop under topic
                        if analysis.topic and core_loop_id:
                            topics[analysis.topic]["core_loops"].add(core_loop_id)

                        # Track core loop
                        if core_loop_id and procedure_info.name:
                            if core_loop_id not in core_loops:
                                core_loops[core_loop_id] = {
                                    "id": core_loop_id,
                                    "name": procedure_info.name,
                                    "topic": analysis.topic,
                                    "procedure": procedure_info.steps or [],
                                    "type": procedure_info.type,
                                    "transformation": procedure_info.transformation,
                                    "exercise_count": 0,
                                    "exercises": []
                                }
                            core_loops[core_loop_id]["exercise_count"] += 1
                            if merged_ex["id"] not in core_loops[core_loop_id]["exercises"]:
                                core_loops[core_loop_id]["exercises"].append(merged_ex["id"])

            # Convert sets to lists for JSON serialization
            for topic_data in topics.values():
                topic_data["core_loops"] = list(topic_data["core_loops"])

            # Deduplicate against existing database entries first, then within batch
            topics = self._deduplicate_topics_with_database(topics, course_code, db)
            core_loops = self._deduplicate_core_loops_with_database(core_loops, course_code, db)

            # Log summary statistics
            accepted_count = len(merged_exercises) - low_confidence_count
            if low_confidence_count > 0:
                print(f"\n[SUMMARY] Confidence Filtering Results:")
                print(f"  Total merged exercises: {len(merged_exercises)}")
                print(f"  Accepted (>= {Config.MIN_ANALYSIS_CONFIDENCE} confidence): {accepted_count}")
                print(f"  Skipped (low confidence): {low_confidence_count}")
                print(f"  Skip rate: {(low_confidence_count / len(merged_exercises) * 100):.1f}%\n")

            return {
                "topics": topics,
                "core_loops": core_loops,
                "merged_exercises": merged_exercises,
                "original_count": len(exercises),
                "merged_count": len(merged_exercises),
                "low_confidence_skipped": low_confidence_count,
                "accepted_count": accepted_count
            }

    async def discover_topics_and_core_loops_async(self, course_code: str,
                                                   batch_size: int = 10,
                                                   skip_analyzed: bool = False) -> Dict[str, Any]:
        """Discover topics and core loops for a course using async processing.

        Args:
            course_code: Course code
            batch_size: Number of exercises to analyze at once
            skip_analyzed: If True, skip already analyzed exercises

        Returns:
            Dict with topics and core loops discovered
        """
        with Database() as db:
            # Get all exercises for course
            exercises = db.get_exercises_by_course(course_code)

            if not exercises:
                return {"topics": {}, "core_loops": {}}

            # Detect primary language if monolingual mode enabled
            if self.monolingual and not self.primary_language:
                course = db.get_course(course_code)
                course_name = course['name'] if course else course_code
                self.primary_language = self._detect_primary_language(exercises, course_name)
                print(f"[MONOLINGUAL MODE] Primary language set to: {self.primary_language}")
                print(f"  All procedures will be normalized to {self.primary_language}\n")

            # Merge fragments using async processing
            merged_exercises = await self.merge_exercises_async(
                exercises,
                batch_size=batch_size,
                skip_analyzed=skip_analyzed
            )

            # Collect all analyses (sync processing of results)
            topics = {}
            core_loops = {}
            low_confidence_count = 0

            for merged_ex in merged_exercises:
                analysis = merged_ex.get("analysis")
                if not analysis:
                    continue

                # Skip low-confidence analyses
                if analysis.confidence < Config.MIN_ANALYSIS_CONFIDENCE:
                    low_confidence_count += 1
                    print(f"[INFO] Skipping exercise due to low confidence ({analysis.confidence:.2f} < {Config.MIN_ANALYSIS_CONFIDENCE}): {merged_ex['id'][:40]}...")
                    merged_ex["low_confidence_skipped"] = True
                    continue

                # Track topic (same logic as sync version)
                if analysis.topic:
                    if analysis.topic not in topics:
                        topics[analysis.topic] = {
                            "name": analysis.topic,
                            "exercise_count": 0,
                            "core_loops": set()
                        }
                    topics[analysis.topic]["exercise_count"] += 1

                # Process ALL procedures
                if analysis.procedures:
                    if len(analysis.procedures) > 1:
                        print(f"[INFO] Multiple procedures detected ({len(analysis.procedures)}) in exercise {merged_ex['id'][:40]}:")
                        for i, proc in enumerate(analysis.procedures, 1):
                            print(f"  {i}. {proc.name} (type: {proc.type}, point: {proc.point_number})")

                    for procedure_info in analysis.procedures:
                        core_loop_id = self._normalize_core_loop_id(procedure_info.name)

                        # Track core loop under topic
                        if analysis.topic and core_loop_id:
                            topics[analysis.topic]["core_loops"].add(core_loop_id)

                        # Track core loop
                        if core_loop_id and procedure_info.name:
                            if core_loop_id not in core_loops:
                                core_loops[core_loop_id] = {
                                    "id": core_loop_id,
                                    "name": procedure_info.name,
                                    "topic": analysis.topic,
                                    "procedure": procedure_info.steps or [],
                                    "type": procedure_info.type,
                                    "transformation": procedure_info.transformation,
                                    "exercise_count": 0,
                                    "exercises": []
                                }
                            core_loops[core_loop_id]["exercise_count"] += 1
                            core_loops[core_loop_id]["exercises"].append(merged_ex["id"])

            # Convert sets to lists for JSON serialization
            for topic_data in topics.values():
                topic_data["core_loops"] = list(topic_data["core_loops"])

            # Deduplicate against existing database entries
            topics = self._deduplicate_topics_with_database(topics, course_code, db)
            core_loops = self._deduplicate_core_loops_with_database(core_loops, course_code, db)

            # Log summary statistics
            accepted_count = len(merged_exercises) - low_confidence_count
            if low_confidence_count > 0:
                print(f"\n[SUMMARY] Confidence Filtering Results:")
                print(f"  Total merged exercises: {len(merged_exercises)}")
                print(f"  Accepted (>= {Config.MIN_ANALYSIS_CONFIDENCE} confidence): {accepted_count}")
                print(f"  Skipped (low confidence): {low_confidence_count}")
                print(f"  Skip rate: {(low_confidence_count / len(merged_exercises) * 100):.1f}%\n")

            return {
                "topics": topics,
                "core_loops": core_loops,
                "merged_exercises": merged_exercises,
                "original_count": len(exercises),
                "merged_count": len(merged_exercises),
                "low_confidence_skipped": low_confidence_count,
                "accepted_count": accepted_count
            }

    def _similarity(self, str1: str, str2: str) -> Tuple[float, str]:
        """Calculate similarity between two strings (0.0 to 1.0).

        Args:
            str1: First string
            str2: Second string

        Returns:
            Tuple of (similarity score, reason)
            - similarity: 0.0 = completely different, 1.0 = identical
            - reason: "semantic_similarity", "translation", "string_similarity", etc.
        """
        if self.use_semantic and self.semantic_matcher:
            # Use semantic similarity
            result = self.semantic_matcher.should_merge(str1, str2, threshold=0.0)
            return result.similarity_score, result.reason
        else:
            # Fallback to string similarity
            similarity = SequenceMatcher(None, str1.lower(), str2.lower()).ratio()
            return similarity, "string_similarity"

    def _deduplicate_topics(self, topics: Dict[str, Any]) -> Dict[str, Any]:
        """Deduplicate similar topics using semantic similarity.

        Args:
            topics: Dictionary of topics

        Returns:
            Deduplicated topics dictionary
        """
        if len(topics) <= 1:
            return topics

        threshold = Config.SEMANTIC_SIMILARITY_THRESHOLD if self.use_semantic else Config.CORE_LOOP_SIMILARITY_THRESHOLD
        topic_names = list(topics.keys())
        merged_topics = {}
        skip_topics = set()

        # Track mapping from old topic names to canonical names
        self.topic_name_mapping = {}

        for i, topic1 in enumerate(topic_names):
            if topic1 in skip_topics:
                continue

            # Start with this topic
            canonical_topic = topic1
            canonical_data = topics[topic1].copy()

            # Map canonical topic to itself
            self.topic_name_mapping[canonical_topic] = canonical_topic

            # Check for similar topics
            for topic2 in topic_names[i+1:]:
                if topic2 in skip_topics:
                    continue

                # Use semantic matching if available
                if self.use_semantic and self.semantic_matcher:
                    result = self.semantic_matcher.should_merge(topic1, topic2, threshold)
                    if result.should_merge:
                        print(f"[DEDUP] Topic '{topic1}' → '{topic2}' (similarity: {result.similarity_score:.2f}, reason: {result.reason})")
                        # Merge topic2 into canonical
                        canonical_data["exercise_count"] += topics[topic2]["exercise_count"]
                        canonical_data["core_loops"] = list(set(canonical_data["core_loops"]) | set(topics[topic2]["core_loops"]))
                        skip_topics.add(topic2)
                        # Map merged topic to canonical
                        self.topic_name_mapping[topic2] = canonical_topic
                    elif Config.SEMANTIC_LOG_NEAR_MISSES and result.similarity_score >= 0.80:
                        print(f"[SKIP] Topic '{topic1}' ≠ '{topic2}' (similarity: {result.similarity_score:.2f}, reason: {result.reason})")
                else:
                    # Fallback to string similarity
                    similarity, reason = self._similarity(topic1, topic2)
                    if similarity >= threshold:
                        print(f"[DEBUG] Merging similar topics: '{topic1}' ≈ '{topic2}' (similarity: {similarity:.2f})")
                        # Merge topic2 into canonical
                        canonical_data["exercise_count"] += topics[topic2]["exercise_count"]
                        canonical_data["core_loops"] = list(set(canonical_data["core_loops"]) | set(topics[topic2]["core_loops"]))
                        skip_topics.add(topic2)
                        # Map merged topic to canonical
                        self.topic_name_mapping[topic2] = canonical_topic

            merged_topics[canonical_topic] = canonical_data

        return merged_topics

    def _deduplicate_core_loops(self, core_loops: Dict[str, Any]) -> Dict[str, Any]:
        """Deduplicate similar core loops using semantic similarity.

        Args:
            core_loops: Dictionary of core loops

        Returns:
            Tuple of (deduplicated core loops dictionary, ID mapping dict)
        """
        if len(core_loops) <= 1:
            return core_loops

        threshold = Config.SEMANTIC_SIMILARITY_THRESHOLD if self.use_semantic else Config.CORE_LOOP_SIMILARITY_THRESHOLD
        loop_ids = list(core_loops.keys())
        merged_loops = {}
        skip_loops = set()

        # Track mapping from old IDs to canonical IDs
        self.core_loop_id_mapping = {}

        for i, loop1_id in enumerate(loop_ids):
            if loop1_id in skip_loops:
                continue

            loop1 = core_loops[loop1_id]
            canonical_id = loop1_id
            canonical_data = loop1.copy()

            # Map canonical ID to itself
            self.core_loop_id_mapping[canonical_id] = canonical_id

            # Check for similar core loops (compare names, not IDs)
            for loop2_id in loop_ids[i+1:]:
                if loop2_id in skip_loops:
                    continue

                loop2 = core_loops[loop2_id]

                # Use semantic matching if available
                if self.use_semantic and self.semantic_matcher:
                    result = self.semantic_matcher.should_merge(
                        loop1["name"], loop2["name"], threshold
                    )
                    if result.should_merge:
                        print(f"[DEDUP] Core loop '{loop1['name']}' → '{loop2['name']}' (similarity: {result.similarity_score:.2f}, reason: {result.reason})")
                        # Merge loop2 into canonical
                        canonical_data["exercise_count"] += loop2["exercise_count"]
                        canonical_data["exercises"] = list(set(canonical_data["exercises"]) | set(loop2["exercises"]))
                        # Merge procedures (prefer longer/more detailed one)
                        if len(loop2.get("procedure", [])) > len(canonical_data.get("procedure", [])):
                            canonical_data["procedure"] = loop2["procedure"]
                        skip_loops.add(loop2_id)
                        # Map merged ID to canonical ID
                        self.core_loop_id_mapping[loop2_id] = canonical_id
                    elif Config.SEMANTIC_LOG_NEAR_MISSES and result.similarity_score >= 0.80:
                        print(f"[SKIP] Core loop '{loop1['name']}' ≠ '{loop2['name']}' (similarity: {result.similarity_score:.2f}, reason: {result.reason})")
                else:
                    # Fallback to string similarity
                    similarity, reason = self._similarity(loop1["name"], loop2["name"])
                    if similarity >= threshold:
                        print(f"[DEBUG] Merging similar core loops: '{loop1['name']}' ≈ '{loop2['name']}' (similarity: {similarity:.2f})")
                        # Merge loop2 into canonical
                        canonical_data["exercise_count"] += loop2["exercise_count"]
                        canonical_data["exercises"] = list(set(canonical_data["exercises"]) | set(loop2["exercises"]))
                        # Merge procedures (prefer longer/more detailed one)
                        if len(loop2.get("procedure", [])) > len(canonical_data.get("procedure", [])):
                            canonical_data["procedure"] = loop2["procedure"]
                        skip_loops.add(loop2_id)
                        # Map merged ID to canonical ID
                        self.core_loop_id_mapping[loop2_id] = canonical_id

            merged_loops[canonical_id] = canonical_data

        return merged_loops

    def _deduplicate_topics_with_database(self, topics: Dict[str, Any],
                                          course_code: str,
                                          db) -> Dict[str, Any]:
        """Deduplicate topics against existing database entries, then within batch.

        Args:
            topics: Dictionary of new topics from current analysis
            course_code: Course code
            db: Database instance

        Returns:
            Deduplicated topics dictionary with mappings to existing DB topics
        """
        threshold = Config.SEMANTIC_SIMILARITY_THRESHOLD if self.use_semantic else Config.CORE_LOOP_SIMILARITY_THRESHOLD

        # Load existing topics from database
        existing_topics = db.get_topics_by_course(course_code)
        existing_topic_map = {t['name']: t for t in existing_topics}

        # Track mappings from new topic names to canonical (db or batch) names
        topic_mapping = {}
        deduplicated_topics = {}

        for new_topic_name, new_topic_data in topics.items():
            matched_existing = None
            best_similarity = 0.0
            best_reason = ""

            # Check against existing database topics first
            for existing_name in existing_topic_map.keys():
                if self.use_semantic and self.semantic_matcher:
                    result = self.semantic_matcher.should_merge(
                        new_topic_name, existing_name, threshold
                    )
                    if result.should_merge and result.similarity_score > best_similarity:
                        best_similarity = result.similarity_score
                        matched_existing = existing_name
                        best_reason = result.reason
                else:
                    similarity, reason = self._similarity(new_topic_name, existing_name)
                    if similarity >= threshold and similarity > best_similarity:
                        best_similarity = similarity
                        matched_existing = existing_name
                        best_reason = reason

            if matched_existing:
                # Reuse existing topic
                print(f"[DEDUP] Topic '{new_topic_name}' → existing '{matched_existing}' (similarity: {best_similarity:.2f}, reason: {best_reason})")
                topic_mapping[new_topic_name] = matched_existing
                # Don't add to deduplicated_topics, we'll use DB entry
            else:
                # New topic, add to batch
                deduplicated_topics[new_topic_name] = new_topic_data
                topic_mapping[new_topic_name] = new_topic_name

        # Now deduplicate within the new batch
        deduplicated_topics = self._deduplicate_topics(deduplicated_topics)

        # Update topic mapping with any batch deduplication
        if hasattr(self, 'topic_name_mapping'):
            for old_name, canonical_name in self.topic_name_mapping.items():
                if old_name in topic_mapping:
                    topic_mapping[old_name] = canonical_name

        # Store the mapping for later use
        self.topic_name_mapping = topic_mapping

        return deduplicated_topics

    def _deduplicate_core_loops_with_database(self, core_loops: Dict[str, Any],
                                              course_code: str,
                                              db) -> Dict[str, Any]:
        """Deduplicate core loops against existing database entries, then within batch.

        Args:
            core_loops: Dictionary of new core loops from current analysis
            course_code: Course code
            db: Database instance

        Returns:
            Deduplicated core loops dictionary with mappings to existing DB loops
        """
        threshold = Config.SEMANTIC_SIMILARITY_THRESHOLD if self.use_semantic else Config.CORE_LOOP_SIMILARITY_THRESHOLD

        # Load existing core loops from database
        existing_loops = db.get_core_loops_by_course(course_code)
        existing_loop_map = {loop['id']: loop for loop in existing_loops}

        # Track mappings from new loop IDs to canonical (db or batch) IDs
        loop_id_mapping = {}
        deduplicated_loops = {}

        for new_loop_id, new_loop_data in core_loops.items():
            matched_existing = None
            best_similarity = 0.0
            best_reason = ""

            # Check against existing database loops first
            for existing_loop in existing_loops:
                if self.use_semantic and self.semantic_matcher:
                    result = self.semantic_matcher.should_merge(
                        new_loop_data['name'], existing_loop['name'], threshold
                    )
                    if result.should_merge and result.similarity_score > best_similarity:
                        best_similarity = result.similarity_score
                        matched_existing = existing_loop['id']
                        best_reason = result.reason
                else:
                    similarity, reason = self._similarity(new_loop_data['name'], existing_loop['name'])
                    if similarity >= threshold and similarity > best_similarity:
                        best_similarity = similarity
                        matched_existing = existing_loop['id']
                        best_reason = reason

            if matched_existing:
                # Reuse existing loop
                print(f"[DEDUP] Core loop '{new_loop_data['name']}' → existing '{existing_loop_map[matched_existing]['name']}' (similarity: {best_similarity:.2f}, reason: {best_reason})")
                loop_id_mapping[new_loop_id] = matched_existing
                # Don't add to deduplicated_loops, we'll use DB entry
            else:
                # New loop, add to batch
                deduplicated_loops[new_loop_id] = new_loop_data
                loop_id_mapping[new_loop_id] = new_loop_id

        # Now deduplicate within the new batch
        deduplicated_loops = self._deduplicate_core_loops(deduplicated_loops)

        # Update loop mapping with any batch deduplication
        if hasattr(self, 'core_loop_id_mapping'):
            for old_id, canonical_id in self.core_loop_id_mapping.items():
                if old_id in loop_id_mapping:
                    loop_id_mapping[old_id] = canonical_id

        # Store the mapping for later use
        self.core_loop_id_mapping = loop_id_mapping

        return deduplicated_loops

    # ========================================================================
    # Learning Material Analysis Methods (Phase 10)
    # ========================================================================

    def analyze_learning_material(self, material_text: str, course_name: str) -> AnalysisResult:
        """Analyze a learning material to detect topics.

        Similar to analyze_exercise() but for theory/worked examples.
        Returns topic names that this material relates to.

        Args:
            material_text: Material content (theory or worked example)
            course_name: Course name for context

        Returns:
            AnalysisResult with topic and metadata
        """
        # Build prompt for material analysis
        prompt = self._build_material_analysis_prompt(material_text, course_name)

        # Call LLM
        response = self.llm.generate(
            prompt=prompt,
            model=self.llm.primary_model,
            temperature=0.3,
            json_mode=True
        )

        if not response.success:
            print(f"[ERROR] LLM failed for material: {response.error}")
            print(f"  Text preview: {material_text[:100]}...")
            return self._default_analysis_result()

        # Parse JSON response
        data = self.llm.parse_json_response(response)
        if not data:
            return self._default_analysis_result()

        # Extract topics (material may relate to multiple topics)
        topics = data.get("topics", [])
        primary_topic = topics[0] if topics else None

        # Extract procedures if any (for worked examples)
        procedures = []
        if "procedures" in data and data["procedures"]:
            for proc_data in data["procedures"]:
                procedures.append(ProcedureInfo(
                    name=proc_data.get("name", "Unknown Procedure"),
                    type=proc_data.get("type", "other"),
                    steps=proc_data.get("steps", []),
                    point_number=proc_data.get("point_number"),
                    transformation=proc_data.get("transformation")
                ))

        # Return analysis result
        return AnalysisResult(
            is_valid_exercise=True,  # Materials are always valid
            is_fragment=False,
            should_merge_with_previous=False,
            topic=primary_topic,
            difficulty=data.get("difficulty"),
            variations=topics[1:] if len(topics) > 1 else [],  # Additional topics
            confidence=data.get("confidence", 0.5),
            procedures=procedures,
            exercise_type=data.get("material_type", "theory"),
            type_confidence=data.get("type_confidence", 0.8)
        )

    def _build_material_analysis_prompt(self, material_text: str, course_name: str) -> str:
        """Build prompt for learning material analysis.

        Args:
            material_text: Material content
            course_name: Course name

        Returns:
            Prompt string
        """
        prompt = f"""You are analyzing learning materials (theory or worked examples) for the course: {course_name}.

Your task is to analyze this material and determine:
1. What topic(s) does it cover?
2. What concepts or procedures does it explain?
3. Is it theory (definitions, explanations) or a worked example (step-by-step solution)?

MATERIAL TEXT:
```
{material_text[:3000]}
```

IMPORTANT: {self._language_instruction("Respond")} All topic names must be in {self._language_name()} language.

Respond in JSON format with:
{{
  "topics": ["primary topic", "secondary topic", ...],  // 1-3 specific topics this material covers
  "material_type": "theory|worked_example|reference",  // Type of material
  "difficulty": "easy|medium|hard",  // Complexity level
  "confidence": 0.0-1.0,  // Your confidence in this analysis
  "procedures": [  // ONLY for worked examples: procedures demonstrated
    {{
      "name": "procedure name",
      "type": "design|transformation|verification|minimization|analysis|other",
      "steps": ["step 1", "step 2", ...],
      "point_number": null,
      "transformation": null
    }}
  ],
  "key_concepts": ["concept1", "concept2", ...],  // Main concepts covered
  "type_confidence": 0.0-1.0  // Confidence in material type
}}

TOPIC NAMING RULES (CRITICAL):
- NEVER use the course name "{course_name}" as the topic - it's too generic!
- Topics MUST be SPECIFIC subtopics within the course
- Be as specific as possible - narrow topics are better than broad ones
- For worked examples, identify the procedure/algorithm being demonstrated
- Materials may relate to multiple topics - list the most relevant ones (1-3)

MATERIAL TYPE CLASSIFICATION:
- **theory**: Definitions, theorems, explanations, conceptual content
  * Contains definitions, properties, explanations without computations
  * No step-by-step computations

- **worked_example**: Step-by-step solution showing how to solve a problem
  * Shows the execution of a procedure/algorithm with steps
  * Should identify the procedure and extract steps

- **reference**: Tables, formulas, reference material without explanations
  * Summary tables, formula sheets, quick reference content

Respond ONLY with valid JSON, no other text.
"""
        return prompt

    def link_materials_to_topics(self, course_code: str):
        """Analyze learning materials and link them to topics.

        For each unlinked learning material:
        1. Analyze content to detect topics
        2. Match detected topics to existing course topics (by name/semantic similarity)
        3. Create links via db.link_material_to_topic()

        Args:
            course_code: Course code
        """
        with Database() as db:
            # Get all materials for this course
            materials = db.get_learning_materials_by_course(course_code)

            if not materials:
                print(f"[INFO] No learning materials found for course {course_code}")
                return

            # Get existing topics for this course
            course_topics = db.get_topics_by_course(course_code)
            topic_map = {t['name']: t for t in course_topics}

            if not course_topics:
                print(f"[WARNING] No topics found for course {course_code}. Run 'analyze' first.")
                return

            # Get course name for context
            course = db.get_course(course_code)
            course_name = course['name'] if course else course_code

            print(f"[INFO] Linking {len(materials)} materials to {len(course_topics)} topics...")

            linked_count = 0
            skipped_count = 0

            for material in materials:
                # Check if already linked
                existing_topics = db.get_topics_for_material(material['id'])
                if existing_topics:
                    print(f"[DEBUG] Material {material['id'][:40]} already linked to {len(existing_topics)} topic(s), skipping")
                    skipped_count += 1
                    continue

                # Analyze material to detect topics
                print(f"[DEBUG] Analyzing material: {material['id'][:40]}... (type: {material['material_type']})")
                analysis = self.analyze_learning_material(
                    material['content'],
                    course_name
                )

                if analysis.confidence < Config.MIN_ANALYSIS_CONFIDENCE:
                    print(f"[WARNING] Low confidence analysis ({analysis.confidence:.2f}), skipping material")
                    continue

                # Collect all detected topics (primary + variations)
                detected_topics = []
                if analysis.topic:
                    detected_topics.append(analysis.topic)
                if analysis.variations:
                    detected_topics.extend(analysis.variations)

                if not detected_topics:
                    print(f"[WARNING] No topics detected for material {material['id'][:40]}")
                    continue

                # Match each detected topic to existing course topics
                for detected_topic in detected_topics:
                    matched_topic_id = self._match_topic_to_existing(
                        detected_topic,
                        course_topics,
                        db
                    )

                    if matched_topic_id:
                        db.link_material_to_topic(material['id'], matched_topic_id)
                        matched_topic = next(t for t in course_topics if t['id'] == matched_topic_id)
                        print(f"[INFO] Linked material to topic: '{matched_topic['name']}'")
                        linked_count += 1
                    else:
                        print(f"[WARNING] Could not match detected topic '{detected_topic}' to existing topics")

            print(f"\n[SUMMARY] Material linking complete:")
            print(f"  Linked: {linked_count} material-topic links created")
            print(f"  Skipped (already linked): {skipped_count} materials")

    def _match_topic_to_existing(self, detected_topic: str, course_topics: List[Dict[str, Any]],
                                  db) -> Optional[int]:
        """Match a detected topic name to an existing course topic.

        Uses semantic similarity to find the best match.

        Args:
            detected_topic: Topic name detected from material
            course_topics: List of existing course topics
            db: Database instance

        Returns:
            Topic ID if match found, None otherwise
        """
        threshold = Config.SEMANTIC_SIMILARITY_THRESHOLD if self.use_semantic else Config.CORE_LOOP_SIMILARITY_THRESHOLD

        best_match_id = None
        best_similarity = 0.0
        best_reason = ""

        for topic in course_topics:
            # Try semantic matching if available
            if self.use_semantic and self.semantic_matcher:
                result = self.semantic_matcher.should_merge(
                    detected_topic, topic['name'], threshold
                )
                if result.should_merge and result.similarity_score > best_similarity:
                    best_similarity = result.similarity_score
                    best_match_id = topic['id']
                    best_reason = result.reason
            else:
                # Fallback to string similarity
                similarity, reason = self._similarity(detected_topic, topic['name'])
                if similarity >= threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = topic['id']
                    best_reason = reason

        if best_match_id:
            matched_topic = next(t for t in course_topics if t['id'] == best_match_id)
            print(f"[MATCH] '{detected_topic}' → '{matched_topic['name']}' (similarity: {best_similarity:.2f}, reason: {best_reason})")

        return best_match_id

    def link_worked_examples_to_exercises(self, course_code: str, max_links_per_example: int = 5):
        """Link worked examples to similar practice exercises.

        For each worked_example material:
        1. Get its topics
        2. Find exercises with same topics
        3. Use semantic similarity to find most related exercises
        4. Create links via db.link_material_to_exercise(type='worked_example')

        Args:
            course_code: Course code
            max_links_per_example: Maximum number of exercises to link per example
        """
        with Database() as db:
            # Get all worked example materials
            worked_examples = db.get_learning_materials_by_course(
                course_code,
                material_type='worked_example'
            )

            if not worked_examples:
                print(f"[INFO] No worked examples found for course {course_code}")
                return

            print(f"[INFO] Linking {len(worked_examples)} worked examples to exercises...")

            total_links = 0

            for example in worked_examples:
                # Get topics for this worked example
                example_topics = db.get_topics_for_material(example['id'])

                if not example_topics:
                    print(f"[WARNING] Worked example {example['id'][:40]} has no topics, skipping")
                    continue

                print(f"[DEBUG] Processing example {example['id'][:40]} with {len(example_topics)} topic(s)")

                # Find exercises with same topics
                candidate_exercises = []
                for topic in example_topics:
                    # Get core loops for this topic
                    core_loops = db.get_core_loops_by_topic(topic['id'])

                    # Get exercises for each core loop
                    for core_loop in core_loops:
                        exercises = db.get_exercises_by_core_loop(core_loop['id'])
                        candidate_exercises.extend(exercises)

                # Remove duplicates
                candidate_exercises = {ex['id']: ex for ex in candidate_exercises}.values()
                candidate_exercises = list(candidate_exercises)

                if not candidate_exercises:
                    print(f"[WARNING] No candidate exercises found for topics: {[t['name'] for t in example_topics]}")
                    continue

                print(f"[DEBUG] Found {len(candidate_exercises)} candidate exercises")

                # Rank exercises by similarity to worked example
                similarities = []
                for exercise in candidate_exercises:
                    similarity = self._calculate_text_similarity(
                        example['content'],
                        exercise['text']
                    )
                    similarities.append((exercise['id'], similarity))

                # Sort by similarity (highest first) and take top N
                similarities.sort(key=lambda x: x[1], reverse=True)
                top_matches = similarities[:max_links_per_example]

                # Create links for top matches
                for exercise_id, similarity in top_matches:
                    if similarity >= Config.WORKED_EXAMPLE_EXERCISE_SIMILARITY_THRESHOLD:
                        db.link_material_to_exercise(
                            example['id'],
                            exercise_id,
                            link_type='worked_example'
                        )
                        total_links += 1
                        print(f"[LINK] Example → Exercise {exercise_id[:40]} (similarity: {similarity:.2f})")

            print(f"\n[SUMMARY] Worked example linking complete:")
            print(f"  Created {total_links} worked_example links")

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings.

        Uses semantic embeddings if available, otherwise falls back to string matching.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0.0 to 1.0)
        """
        if self.use_semantic and self.semantic_matcher:
            # Use semantic matcher's embedding-based similarity
            result = self.semantic_matcher.should_merge(text1, text2, threshold=0.0)
            return result.similarity_score
        else:
            # Fallback to string similarity
            similarity, _ = self._similarity(text1, text2)
            return similarity

    # ========================================================================
    # Topic Splitting Methods (Phase 6)
    # ========================================================================

    def detect_generic_topics(self, course_code: str, db: 'Database') -> List[Dict[str, Any]]:
        """Detect generic topics that should be split.

        Args:
            course_code: Course code to check
            db: Database instance

        Returns:
            List of dicts with topic info that should be split:
            [{"id": topic_id, "name": topic_name, "core_loop_count": N}, ...]
        """
        generic_topics = []

        # Get all topics for this course
        topics = db.get_topics_by_course(course_code)

        # Get course info for name comparison
        course = db.get_course(course_code)
        course_name = course['name'] if course else ""

        for topic in topics:
            # Get core loops for this topic
            core_loops = db.get_core_loops_by_topic(topic['id'])
            loop_count = len(core_loops)

            # Detection criteria
            is_generic = False
            reason = []

            # Criterion 1: Topic has too many core loops
            if loop_count >= Config.GENERIC_TOPIC_THRESHOLD:
                is_generic = True
                reason.append(f"{loop_count} core loops (threshold: {Config.GENERIC_TOPIC_THRESHOLD})")

            # Criterion 2: Topic name matches or is very similar to course name
            if course_name and self._is_topic_name_generic(topic['name'], course_name):
                is_generic = True
                reason.append(f"topic name '{topic['name']}' is generic/matches course")

            if is_generic:
                generic_topics.append({
                    "id": topic['id'],
                    "name": topic['name'],
                    "core_loop_count": loop_count,
                    "core_loops": [cl['id'] for cl in core_loops],
                    "reason": "; ".join(reason)
                })

        return generic_topics

    def _is_topic_name_generic(self, topic_name: str, course_name: str) -> bool:
        """Check if topic name is too generic compared to course name."""
        # Normalize both names for comparison
        topic_norm = topic_name.lower().strip()
        course_norm = course_name.lower().strip()

        # Check if topic is exactly the course name
        if topic_norm == course_norm:
            return True

        # Check if topic is main words from course name
        # e.g., "Algebra Lineare" contains "Algebra"
        course_words = set(course_norm.split())
        topic_words = set(topic_norm.split())

        # If topic is subset of course words and has < 3 unique words, it's generic
        if topic_words.issubset(course_words) and len(topic_words) < 3:
            return True

        return False

    def cluster_core_loops_for_topic(self, topic_id: int, topic_name: str,
                                     core_loops: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        """Cluster core loops into semantic groups using LLM.

        Args:
            topic_id: ID of generic topic
            topic_name: Name of generic topic
            core_loops: List of core loop dicts (with 'id' and 'name' keys)

        Returns:
            List of clusters or None if clustering fails:
            [
                {
                    "topic_name": "Specific Topic Name",
                    "core_loop_ids": ["loop1", "loop2", ...]
                },
                ...
            ]
        """
        try:
            print(f"[INFO] Clustering {len(core_loops)} core loops for topic '{topic_name}'...")

            # Build core loop list for prompt
            core_loop_list = "\n".join([
                f"{i+1}. {cl['name']} (ID: {cl['id']})"
                for i, cl in enumerate(core_loops)
            ])

            # Build clustering prompt
            prompt = f"""You are analyzing core loops (procedural problem-solving patterns) from the topic "{topic_name}".

These {len(core_loops)} core loops are currently grouped together but are too diverse.
Cluster them into {Config.TOPIC_CLUSTER_MIN}-{Config.TOPIC_CLUSTER_MAX} specific subtopics based on semantic similarity.

Core loops to cluster:
{core_loop_list}

Requirements:
- Create {Config.TOPIC_CLUSTER_MIN}-{Config.TOPIC_CLUSTER_MAX} clusters
- Each core loop must appear in exactly ONE cluster
- Give each cluster a specific, descriptive topic name {self._lang_instruction()}
- Topic names should reflect the mathematical/algorithmic concept, NOT be generic
- Group by semantic similarity (what concepts/techniques are being practiced)

Return ONLY valid JSON in this format:
{{
  "clusters": [
    {{
      "topic_name": "Specific Topic Name Here",
      "core_loop_ids": ["loop_id_1", "loop_id_2", ...]
    }},
    ...
  ]
}}

No markdown code blocks, just JSON."""

            # Call LLM
            response = self.llm.generate(
                prompt=prompt,
                temperature=0.5,  # Some creativity for grouping
                json_mode=True
            )

            if not response.success:
                print(f"[ERROR] LLM clustering failed: {response.error}")
                return None

            # Parse response
            data = self.llm.parse_json_response(response)
            if not data or 'clusters' not in data:
                print(f"[ERROR] Invalid clustering response format")
                return None

            clusters = data['clusters']

            # Validate clustering
            all_assigned_ids = set()
            for cluster in clusters:
                if 'topic_name' not in cluster or 'core_loop_ids' not in cluster:
                    print(f"[ERROR] Invalid cluster format: {cluster}")
                    return None
                all_assigned_ids.update(cluster['core_loop_ids'])

            original_ids = set(cl['id'] for cl in core_loops)

            # Check if all core loops were assigned
            if all_assigned_ids != original_ids:
                missing = original_ids - all_assigned_ids
                extra = all_assigned_ids - original_ids
                print(f"[WARNING] Clustering validation failed:")
                if missing:
                    print(f"  Missing IDs: {missing}")
                if extra:
                    print(f"  Extra IDs: {extra}")
                return None

            # Check cluster count
            if not (Config.TOPIC_CLUSTER_MIN <= len(clusters) <= Config.TOPIC_CLUSTER_MAX):
                print(f"[WARNING] Cluster count {len(clusters)} outside range {Config.TOPIC_CLUSTER_MIN}-{Config.TOPIC_CLUSTER_MAX}")

            print(f"[INFO] Successfully created {len(clusters)} clusters")
            for i, cluster in enumerate(clusters, 1):
                print(f"  {i}. {cluster['topic_name']}: {len(cluster['core_loop_ids'])} core loops")

            return clusters

        except Exception as e:
            print(f"[ERROR] Clustering failed: {e}")
            return None
