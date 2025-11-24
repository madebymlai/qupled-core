# Examina

AI-powered exam preparation system that learns from your course materials to help you master university courses.

## What It Does

Examina analyzes your course materials (past exams, homework, problem sets, lecture notes) to automatically:
- **Discover topics & procedures** - Identifies recurring problem-solving patterns ("core loops")
- **Build a knowledge base** - Extracts exercises, procedures, and solving strategies
- **Teach interactively** - Provides AI tutoring with theory, examples, and feedback
- **Track progress** - Uses spaced repetition (SM-2) to optimize learning
- **Generate practice** - Creates new exercises based on learned patterns

## Quick Start

### Installation

```bash
# Clone and setup
git clone https://github.com/madebymlai/Examina.git
cd Examina
python -m venv venv
source venv/bin/activate  # Linux/Mac: source venv/bin/activate | Windows: venv\Scripts\activate
pip install -r requirements.txt

# Configure LLM (choose one)
export ANTHROPIC_API_KEY="your-key"  # Recommended - best quality
export GROQ_API_KEY="your-key"       # Alternative - fast & free tier

# Or use local Ollama
ollama pull nomic-embed-text

# Initialize database
python3 cli.py init
```

### Basic Usage

```bash
# 1. Add a course
python3 cli.py add-course --code ADE --name "Computer Architecture"

# 2. Ingest course materials (past exams, homework, problem sets, etc.)
python3 cli.py ingest --course ADE --zip course_materials.zip

# 3. Analyze with AI (discovers topics & procedures)
python3 cli.py analyze --course ADE --provider anthropic

# 4. View what was learned
python3 cli.py info --course ADE

# 5. Start learning
python3 cli.py learn --course ADE --loop "Mealy Machine Design"

# 6. Take a quiz
python3 cli.py quiz --course ADE --questions 5

# 7. Check progress
python3 cli.py progress --course ADE
```

## What Can You Upload?

Examina works with **any course material containing problems and exercises**:

- ✅ **Past Exams** - With or without solutions (best for discovering exam patterns)
- ✅ **Homework Assignments** - Problem sets from professors or TAs
- ✅ **Practice Exams** - Mock exams or practice problems
- ✅ **Exercise Collections** - PDFs from course websites or textbooks
- ✅ **Lecture Notes** - Notes with worked examples and practice problems
- ✅ **Problem Set PDFs** - Any structured problem collection

**The engine learns from structure, not source.** Past exams are ideal but not required.

## Key Features

### 🎯 Smart Analysis
- Automatically discovers topics and core loops (solving procedures)
- Extracts multi-step procedures from complex exercises
- Supports theory questions, proofs, and procedural exercises
- Works across any subject (Computer Science, Math, Engineering, etc.)

### 🧠 AI Tutoring
- **Learn mode**: Theory explanations with prerequisites, examples, and analogies
- **Practice mode**: Interactive problem-solving with hints and feedback
- **Quiz mode**: AI-evaluated answers with detailed explanations

### 📊 Progress Tracking
- SM-2 spaced repetition algorithm
- Mastery levels: new → learning → reviewing → mastered
- Analytics dashboard with weak areas identification
- Personalized study suggestions

### 🌐 Multi-Language
- Full Italian/English support
- Bilingual deduplication (merges "Finite State Machine" ↔ "Macchina a Stati Finiti")

## Commands Reference

### Course Management
```bash
python3 cli.py add-course --code B006802 --name "Architettura degli Elaboratori"
python3 cli.py list-courses
python3 cli.py info --course B006802
```

### Content Ingestion
```bash
# From ZIP archive (past exams, homework, problem sets, lecture notes, etc.)
python3 cli.py ingest --course B006802 --zip course_materials.zip

# From directory
python3 cli.py ingest --course B006802 --dir ./course_pdfs/

# Examples of what you can ingest:
# - Past exams with or without solutions
# - Homework assignments and problem sets
# - Practice exams from the professor
# - Exercise collections from course sites
# - Lecture notes with worked examples
# - Textbook problem PDFs
```

### Analysis
```bash
# Analyze all exercises
python3 cli.py analyze --course B006802 --provider anthropic --lang it

# Resume interrupted analysis
python3 cli.py analyze --course B006802 --resume

# Force re-analysis
python3 cli.py analyze --course B006802 --force
```

### Learning
```bash
# Learn a specific procedure
python3 cli.py learn --course B006802 --loop "Moore Machine Design"

# With depth control
python3 cli.py learn --course B006802 --loop "Mealy Machine Design" --depth advanced

# Skip prerequisites
python3 cli.py learn --course B006802 --loop "FSM Minimization" --no-concepts
```

### Quizzes
```bash
# Random quiz
python3 cli.py quiz --course B006802 --questions 10

# Filtered quiz
python3 cli.py quiz --course B006802 --topic "Automi a Stati Finiti" --difficulty medium

# Review mode (spaced repetition)
python3 cli.py quiz --course B006802 --review-only

# Filter by exercise type
python3 cli.py quiz --course B006802 --type theory
python3 cli.py quiz --course B006802 --type proof
```

### Progress & Analytics
```bash
# Overall progress
python3 cli.py progress --course B006802

# Study suggestions
python3 cli.py suggest --course B006802
```

### Maintenance
```bash
# Deduplicate topics/core loops
python3 cli.py deduplicate --course B006802 --dry-run

# Split generic topics
python3 cli.py split-topics --course B006802 --dry-run
```

## Configuration

### LLM Providers

**Anthropic Claude Sonnet 4.5** (Recommended)
- Best quality and reasoning
- Higher rate limits
- `--provider anthropic`

**Groq** (Free tier available)
- Fast inference
- 30 requests/minute free tier
- `--provider groq`

**Ollama** (Local)
- Free and private
- Requires local GPU
- `--provider ollama`

### Environment Variables

```bash
# LLM Provider
export EXAMINA_LLM_PROVIDER=anthropic  # or groq, ollama
export ANTHROPIC_API_KEY="your-key"
export GROQ_API_KEY="your-key"

# Analysis Settings
export EXAMINA_MIN_CONFIDENCE=0.5      # Filter low-confidence analyses
export EXAMINA_PARALLEL_WORKERS=4      # Parallel analysis workers

# Topic Splitting
export EXAMINA_GENERIC_TOPIC_THRESHOLD=10  # Min core loops to trigger split
export EXAMINA_TOPIC_SPLITTING_ENABLED=1

# Deduplication
export EXAMINA_SIMILARITY_THRESHOLD=0.85
export EXAMINA_SEMANTIC_MATCHING=1
```

## Project Status

**Production Ready:**
- ✅ PDF ingestion & extraction
- ✅ AI analysis & knowledge discovery
- ✅ Interactive AI tutor
- ✅ Quiz system with spaced repetition
- ✅ Multi-procedure extraction
- ✅ Automatic topic splitting
- ✅ Theory & proof support
- ✅ Bilingual deduplication

**In Progress:**
- 🚧 Enhanced learning system (metacognitive strategies)

**Planned:**
- 📋 Exam files with solutions parsing
- 📋 Orphaned core loops cleanup
- 📋 Adaptive teaching based on mastery

See [TODO.md](TODO.md) for detailed task list and [CHANGELOG.md](CHANGELOG.md) for version history.

## Architecture

```
Examina/
├── cli.py              # Main CLI interface
├── config.py           # Configuration management
├── core/               # Core modules
│   ├── analyzer.py     # Exercise analysis
│   ├── tutor.py        # AI teaching
│   ├── quiz_engine.py  # Quiz system
│   ├── sm2.py          # Spaced repetition
│   └── semantic_matcher.py  # Deduplication
├── models/             # LLM integrations
│   └── llm_manager.py  # Provider abstraction
├── storage/            # Data layer
│   └── database.py     # SQLite + migrations
└── utils/              # Utilities
    ├── pdf_extractor.py
    └── splitter.py
```

## Contributing

Issues and pull requests welcome! See [TODO.md](TODO.md) for areas needing work.

## Privacy & Data

**Your course materials stay yours.** Examina analyzes the materials you upload (exams, homework, problem sets, notes) to build a private knowledge base for you only. We don't share your materials or generated questions with other users.

- 📄 Your PDFs are stored locally in your account
- 🔒 Content sent to LLM providers only for generating explanations/quizzes
- 🚫 We don't sell data or train models on your course materials
- 🗑️ Delete your uploads and data anytime

See [PRIVACY.md](PRIVACY.md) for full details.

## License

MIT License - see LICENSE file for details.

## Credits

Built with Claude Code by Anthropic.
