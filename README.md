# Examina

AI-powered exam tutor system for mastering university courses through automated analysis of past exams.

## Overview

Examina analyzes past exam PDFs to:
- Auto-discover topics and "core loops" (resolutive algorithms/procedures)
- Build a knowledge base of exercise types and solving methods
- Provide an AI tutor for guided learning and practice
- Generate new exercises based on discovered patterns
- Track progress with spaced repetition

## Project Status

**Phase 1: Setup & Database** ✅ COMPLETED
- Project structure created
- Database schema implemented
- Basic CLI with course management

**Phase 2: PDF Processing** ✅ COMPLETED
- Extract text, images, and LaTeX from PDFs
- Split PDFs into individual exercises
- Store extracted content with intelligent merging

**Phase 3: AI Analysis** ✅ COMPLETED
- Auto-discover topics and core loops with LLM
- Extract solving procedures (step-by-step algorithms)
- Build knowledge base with RAG (vector embeddings)
- **NEW**: Confidence threshold filtering
- **NEW**: LLM response caching (100% hit rate on re-runs)
- **NEW**: Resume capability for interrupted analysis
- **NEW**: Parallel batch processing (7-8x speedup)
- **NEW**: Topic/core loop deduplication (similarity-based)

**Phase 4: AI Tutor** ✅ COMPLETED
- Learn mode with theory, procedure walkthrough, and examples
- Practice mode with interactive feedback and hints
- Exercise generation with difficulty control
- Multi-language support (Italian/English)

**Phase 5: Quiz System** ✅ COMPLETED
- Interactive quiz system with AI feedback
- SM-2 spaced repetition algorithm for optimal learning
- Progress tracking with mastery levels (new → learning → reviewing → mastered)
- Study suggestions based on weak areas and review schedule
- Analytics dashboard with topic breakdown
- Multi-language support (Italian/English)

**Phase 6: Multi-Core-Loop Support** 🚧 IN PROGRESS
- Extract ALL procedures from multi-step exercises (design + transformation + minimization)
- Many-to-many exercise-to-core-loop relationships
- Intelligent detection of numbered points and transformations
- Tag-based search (e.g., find all "Mealy→Moore" exercises)
- Backward compatible with existing analyses

**Phase 7: Enhanced Learning System** 🚧 IN PROGRESS
- ✅ **Deep theory explanations** - Prerequisite concepts with examples and analogies
- ✅ **Step-by-step reasoning** - WHY behind each algorithm step
- **NEW**: 3 depth levels (basic, medium, advanced)
- **NEW**: --no-concepts flag to skip prerequisites
- 🔜 Metacognitive learning strategies (study tips and frameworks)
- 🔜 Adaptive teaching based on mastery level

**Phase 8: Automatic Topic Splitting** ✅ COMPLETED
- ✅ **Generic topic detection** - Automatically finds topics with too many core loops
- ✅ **LLM-based clustering** - Semantically groups core loops into focused subtopics
- ✅ **Smart splitting** - Converts generic topics into 4-6 specific, manageable topics
- **NEW**: `split-topics` command with dry-run mode
- **NEW**: Transaction-safe database updates with rollback
- **NEW**: Fully LLM-driven (no hardcoding, works for any subject)

## Installation

### Prerequisites

- Python 3.10+
- Anthropic API key (recommended) or Groq API key or Ollama (local)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/madebymlai/Examina.git
cd Examina
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your LLM provider:

**Option A: Anthropic Claude (Recommended - Best Quality)**
```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

**Option B: Groq (Fast, Free Tier Available)**
```bash
export GROQ_API_KEY="your-api-key-here"
```

**Option C: Ollama (Local, Free)**
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull embedding model (required for all providers)
ollama pull nomic-embed-text
```

5. Initialize Examina:
```bash
python3 cli.py init
```

## Quick Start

### 1. Ingest Exam PDFs

```bash
# Ingest exams from a ZIP file
python3 cli.py ingest --course ADE --zip ADE-ESAMI.zip

# View ingested exercises
python3 cli.py info --course ADE
```

### 2. Analyze with AI

```bash
# Analyze exercises to discover topics and core loops
python3 cli.py analyze --course ADE --provider anthropic --lang en

# With custom settings
python3 cli.py analyze --course ADE \
  --provider anthropic \
  --lang it \
  --parallel \
  --batch-size 10

# Resume interrupted analysis
python3 cli.py analyze --course ADE  # Automatically resumes

# Force re-analysis
python3 cli.py analyze --course ADE --force
```

**Analysis Features:**
- **Parallel Processing**: 7-8x faster (20s → 3s for 27 exercises)
- **Caching**: Zero cost on re-runs (100% cache hit rate)
- **Resume**: Automatic checkpoint recovery
- **Database-Aware Deduplication**: Prevents duplicates across analysis runs (0.85 similarity threshold)
- **Confidence Filtering**: Filters low-quality analyses (default 0.5)

### 3. Learn with AI Tutor (Enhanced with Deep Explanations)

```bash
# Enhanced learning with prerequisite concepts (default)
python3 cli.py learn --course ADE --loop moore_machine_design --lang en

# Control explanation depth
python3 cli.py learn --course ADE --loop conversione_mealy_moore --depth basic --lang en    # Concise
python3 cli.py learn --course ADE --loop minimizzazione --depth advanced --lang en           # Comprehensive

# Skip prerequisites for faster response
python3 cli.py learn --course ADE --loop moore_machine_design --no-concepts --lang en

# Italian language with custom depth
python3 cli.py learn --course AL --loop gauss_elimination --depth medium --lang it
```

**New Features:**
- **Prerequisite Concepts**: Foundational explanations with examples and analogies
- **WHY Reasoning**: Deep explanations of why each step works
- **5-Section Structure**: Big Picture → Step-by-Step → Pitfalls → Decision-Making → Practice Strategy
- **Depth Control**: basic (quick), medium (balanced), advanced (comprehensive)

### 4. Practice Exercises

```bash
# Practice with interactive feedback
python3 cli.py practice --course ADE --difficulty medium --lang en

# Filter by topic
python3 cli.py practice --course PC --topic "Sincronizzazione" --lang it
```

### 5. Generate New Exercises

```bash
# Generate exercise variations
python3 cli.py generate --course ADE --loop moore_machine_design --difficulty hard --lang en
```

### 6. Deduplicate Topics and Core Loops

```bash
# Preview what would be merged (dry run)
python3 cli.py deduplicate --course ADE --dry-run

# Merge duplicate topics and core loops
python3 cli.py deduplicate --course ADE

# Custom similarity threshold (default: 0.85)
python3 cli.py deduplicate --course ADE --threshold 0.90
```

**Deduplication Features:**
- Merges similar topics/core loops using string similarity
- Updates all foreign key references automatically
- Dry-run mode to preview changes before applying
- Configurable similarity threshold (0.0-1.0)
- Prevents data loss by preserving all exercise associations

### 7. Automatic Topic Splitting

```bash
# Preview topic splits without applying changes (dry run)
python3 cli.py split-topics --course AL --lang it --dry-run

# Automatically split generic topics into focused subtopics
python3 cli.py split-topics --course AL --lang it --force

# Delete empty original topic after split
python3 cli.py split-topics --course AL --lang it --force --delete-old
```

**Topic Splitting Features:**
- **Automatic Detection**: Finds generic topics (>10 core loops or matching course name)
- **LLM-Based Clustering**: Semantically groups core loops into 4-6 focused subtopics
- **Smart Organization**: Creates specific, descriptive topic names (Italian/English)
- **Safe Execution**: Transaction-based with rollback on failure
- **Preview Mode**: Dry-run shows proposed splits before applying
- **Data Integrity**: All core loops preserved and validated
- **No Hardcoding**: Fully LLM-driven, works for any subject

**Example Result:**
```
"Algebra Lineare" (30 core loops) → Split into 6 topics:
  - Sottospazi Vettoriali e Basi (10 loops)
  - Applicazioni Lineari e Trasformazioni (6 loops)
  - Diagonalizzazione e Autovalori (5 loops)
  - Cambi di Base e Basi Ortonormali (3 loops)
  - Matrici Parametriche e Determinanti (3 loops)
  - Teoria e Problemi Integrati (3 loops)
```

## Usage Examples

### View Available Courses

```bash
python3 cli.py courses
python3 cli.py courses --degree bachelor
python3 cli.py courses --degree master
```

### Get Course Information

```bash
python3 cli.py info --course ADE
python3 cli.py info --course B006802
```

### Advanced Analysis Options

```bash
# Test with limited exercises
python3 cli.py analyze --course ADE --limit 10

# Sequential mode for debugging
python3 cli.py analyze --course ADE --sequential

# Custom batch size for rate limits
python3 cli.py analyze --course ADE --batch-size 5
```

## Academic Context

Examina is designed for UNIFI Computer Science programs:

### Bachelor's Degree (L-31)
17 courses including:
- Linear Algebra (AL)
- Computer Architecture (ADE)
- Operating Systems (SO)
- Databases (BDSI)
- Computer Networks (RC)
- Concurrent Programming (PC)
- And more...

### Master's Degree (LM-18) - Software: Science and Technology
13 courses including:
- Distributed Programming (DP)
- Software Architectures (SAM)
- Penetration Testing (PT)
- Computer and Network Security (CNS)
- And more...

## Architecture

```
examina/
├── cli.py                  # CLI interface
├── config.py              # Configuration
├── study_context.py       # Course metadata
├── core/
│   ├── analyzer.py        # AI exercise analysis + parallel processing
│   ├── tutor.py           # AI tutor (learn, practice, generate)
│   └── splitter.py        # Exercise splitting logic
├── models/
│   └── llm_manager.py     # Multi-provider LLM + caching
├── storage/
│   ├── database.py        # SQLite operations + migrations
│   ├── vector_store.py    # ChromaDB for RAG
│   └── file_manager.py    # File operations
└── data/                  # Data directory (git-ignored)
    ├── examina.db         # SQLite database
    ├── chroma/            # Vector embeddings
    ├── cache/             # LLM response cache
    └── files/             # PDFs and images
```

## Technology Stack

- **CLI**: Click + Rich
- **Database**: SQLite + ChromaDB (vector store)
- **PDF Processing**: PyMuPDF, pdfplumber, pytesseract
- **LLM**: Anthropic Claude Sonnet 4.5 (primary), Groq, Ollama
- **Embeddings**: sentence-transformers, nomic-embed-text
- **Math**: SymPy, latex2sympy2
- **Concurrency**: ThreadPoolExecutor for parallel analysis

## Configuration

Edit `config.py` or use environment variables:

```bash
# LLM Provider
export EXAMINA_LLM_PROVIDER=anthropic  # or groq, ollama

# Models
export ANTHROPIC_MODEL=claude-sonnet-4-20250514
export GROQ_MODEL=llama-3.3-70b-versatile

# Analysis Settings
export EXAMINA_LANGUAGE=en  # or it
export EXAMINA_MIN_CONFIDENCE=0.5  # Confidence threshold (0.0-1.0)

# Cache Settings
export EXAMINA_CACHE_ENABLED=true
export EXAMINA_CACHE_TTL=3600  # seconds
```

## Performance Benchmarks

**Analysis Speed (27 exercises):**
- Sequential: ~20 seconds (1.7 ex/s)
- Parallel (batch=10): ~3 seconds (13.2 ex/s)
- **Speedup: 7.76x**

**Caching Benefits:**
- First run: ~26s (cache cold)
- Second run: ~0.01s (cache warm)
- **Speedup: 5000x on re-runs**

**Cost Savings:**
- Cached re-analysis: $0 (zero API calls)
- Resume on failure: Saves partial progress

## Tested Courses

✅ Computer Architecture (ADE) - 27 exercises, 11 topics, 21 core loops
✅ Linear Algebra (AL) - 38 exercises, 2 topics, 4 core loops
✅ Concurrent Programming (PC) - 26 exercises, 6 topics, 14 core loops

## Advanced Features (Phase 6)

### Multi-Procedure Extraction

Examina now extracts **ALL procedures** from multi-step exercises:

**Example:**
```
Exercise: "1. Design Mealy machine, 2. Transform to Moore, 3. Minimize"

Old behavior: Extracts only "Mealy Machine Design" ❌
New behavior: Extracts all 3 procedures ✅
  - Mealy Machine Design (step 1)
  - Mealy to Moore Transformation (step 2)
  - State Minimization (step 3)
```

**Features:**
- **Intelligent Detection**: Recognizes numbered points, transformations, conversions
- **Tag System**: Search exercises by procedure type (e.g., all "transformation" exercises)
- **Many-to-Many**: Exercises can be linked to multiple core loops
- **Backward Compatible**: Existing analyses continue to work

**Supported Patterns:**
- Numeric: "1.", "2.", "3."
- Letters: "a)", "b)", "c)"
- Italian: "Punto 1", "Esercizio 2.a"
- Roman: "I.", "II.", "III."

**Transformation Detection:**
- Mealy ↔ Moore conversions
- DFA ↔ NFA conversions
- Logic form transformations (SOP, POS, etc.)
- 15+ keyword patterns in English and Italian

## Contributing

This is currently a personal learning project. Contributions and suggestions are welcome!

## License

TBD

## Acknowledgments

Built with Claude Code for studying at Università degli Studi di Firenze (UNIFI).

---

**Current Work**:
- Phase 6: Multi-Core-Loop Support ✅ COMPLETE (extracting ALL procedures from multi-step exercises)
- Phase 7: Enhanced Learning System 🚧 IN PROGRESS
  - ✅ Phase 7.1: Deep theory explanations with prerequisite concepts
  - ✅ Phase 7.2: Step-by-step WHY reasoning (partially complete)
  - 🔜 Phase 7.3: Metacognitive learning strategies
  - 🔜 Phase 7.4: Adaptive teaching based on mastery
- Phase 8: Automatic Topic Splitting ✅ COMPLETE (LLM-driven clustering of generic topics)
