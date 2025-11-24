# ExaminaService - Service Layer for SaaS Readiness

## Overview

The `ExaminaService` class provides a stateless service interface for Examina's core operations, designed to enable easy integration with web frameworks (FastAPI/Flask) while maintaining all business logic in the core layer.

## Architecture

### Design Principles

1. **Stateless Design**: No global state, each service instance is independent
2. **Provider-Configurable**: Supports multiple LLM providers (anthropic, groq, ollama, openai)
3. **Explicit Dependencies**: Uses dependency injection for LLM and database access
4. **Clean Separation**: Business logic stays in core/, CLI is just a thin wrapper

### Service Interface Pattern

```python
from core.service import ExaminaService, ServiceResult

# Create service with user's preferred provider
service = ExaminaService(provider="anthropic", language="en")

# Call service methods - all return ServiceResult
result = service.learn_core_loop(
    course_code="CS101",
    core_loop_id="binary_search"
)

if result.success:
    print(result.data['content'])
else:
    print(f"Error: {result.error}")
```

## CLI Integration

All CLI commands that use LLM now support the `--provider` flag:

```bash
# Use default provider from config
examina learn -c CS101 -l binary_search

# Override with specific provider
examina learn -c CS101 -l binary_search --provider groq

# Works with all commands
examina practice -c CS101 --provider anthropic
examina prove -c MATH101 --provider ollama
examina quiz -c CS101 --provider openai
examina generate -c CS101 -l merge_sort --provider groq
```

### Commands with --provider Support

- `ingest`: Ingest course materials
- `analyze`: Analyze exercises to discover topics
- `learn`: Learn core loops with AI tutor
- `practice`: Practice exercises interactively
- `prove`: Practice mathematical proofs
- `generate`: Generate new exercises
- `quiz`: Take interactive quizzes
- `separate-solutions`: Separate questions from solutions

## Future Web API Integration

### Example with FastAPI

```python
from fastapi import FastAPI, Depends, HTTPException
from core.service import ExaminaService, ServiceResult
from typing import Optional

app = FastAPI()

# Dependency injection for service
def get_service(
    provider: Optional[str] = None,
    language: str = "en"
) -> ExaminaService:
    """Create service instance with user preferences."""
    return ExaminaService(provider=provider, language=language)

# Endpoint example
@app.post("/api/courses/{course_code}/learn/{loop_id}")
async def learn_endpoint(
    course_code: str,
    loop_id: str,
    provider: Optional[str] = None,
    service: ExaminaService = Depends(get_service)
) -> ServiceResult:
    """Learn a core loop with AI tutor."""
    return service.learn_core_loop(course_code, loop_id)

@app.post("/api/courses/{course_code}/practice")
async def practice_endpoint(
    course_code: str,
    topic: Optional[str] = None,
    difficulty: Optional[str] = None,
    service: ExaminaService = Depends(get_service)
) -> ServiceResult:
    """Get practice exercise."""
    return service.practice_exercise(course_code, topic, difficulty)

@app.post("/api/courses/{course_code}/quiz")
async def start_quiz(
    course_code: str,
    length: int = 10,
    service: ExaminaService = Depends(get_service)
) -> ServiceResult:
    """Start a new quiz session."""
    return service.start_quiz(course_code, length=length)
```

### Multi-Tenant Support

The service layer makes it easy to support per-user provider preferences:

```python
@app.get("/api/learn/{course}/{loop}")
async def learn(
    course: str,
    loop: str,
    current_user: User = Depends(get_current_user)
):
    # Each user can have their own provider preference
    service = ExaminaService(
        provider=current_user.preferred_llm_provider,
        language=current_user.preferred_language
    )
    return service.learn_core_loop(course, loop)
```

## ServiceResult Format

All service methods return a `ServiceResult` object:

```python
@dataclass
class ServiceResult:
    success: bool          # Operation success/failure
    message: str           # Human-readable message
    data: Optional[Dict]   # Result data (if successful)
    error: Optional[str]   # Error message (if failed)
    metadata: Optional[Dict]  # Additional metadata
```

This format is:
- Easy to serialize to JSON for web APIs
- Consistent across all operations
- Contains all necessary information for clients

## Available Service Methods

### Ingestion & Analysis
- `ingest_notes()`: Ingest PDF notes/exercises
- `analyze_exercises()`: Analyze exercises to discover topics
- `link_learning_materials()`: Link theory and examples

### Learning
- `learn_core_loop()`: Get AI tutor explanation for a core loop

### Practice
- `practice_exercise()`: Get practice exercise with guidance
- `check_answer()`: Check user's answer to an exercise

### Proof Practice
- `practice_proof()`: Get proof practice with step-by-step guidance

### Quiz
- `start_quiz()`: Start a new quiz session
- `submit_quiz_answer()`: Submit answer to quiz question

### Generation
- `generate_exercise()`: Generate new practice exercises

### Utilities
- `get_course_stats()`: Get course statistics
- `get_study_recommendations()`: Get personalized study recommendations

## Benefits for SaaS

1. **Clean API Layer**: Web framework only handles HTTP/auth/routing
2. **Business Logic Isolation**: All domain logic stays in core/
3. **Easy Testing**: Service methods are easy to unit test
4. **Provider Flexibility**: Support per-user provider preferences
5. **No Breaking Changes**: Existing CLI works unchanged
6. **Future-Proof**: Easy to add new endpoints without touching core logic

## Migration Path

### Phase 1: Service Layer (✓ Complete)
- Created `ExaminaService` class in `core/service.py`
- Added `--provider` flag to all LLM-using commands
- Updated commands to use provider parameter

### Phase 2: Web API (Future)
- Create FastAPI application
- Add authentication/authorization
- Map HTTP endpoints to service methods
- Add rate limiting and monitoring

### Phase 3: Multi-Tenancy (Future)
- User account management
- Per-user provider preferences
- Usage tracking and quotas
- Data isolation

## Design Decisions

### Why Stateless?
- Enables horizontal scaling in web environment
- Simpler to reason about and test
- No shared state between requests
- Each request is independent

### Why Dependency Injection?
- Easy to mock for testing
- Supports different configurations per request
- Clear dependencies and responsibilities
- Follows SOLID principles

### Why ServiceResult?
- Consistent error handling
- Easy to serialize for APIs
- Contains all necessary information
- Type-safe with dataclasses

### Why Keep Business Logic in Core?
- Single source of truth
- CLI and API share same logic
- Easier to maintain and test
- Clear separation of concerns

## Testing

Service methods can be easily tested:

```python
def test_learn_core_loop():
    # Create service with test provider
    service = ExaminaService(provider="ollama")

    # Call service method
    result = service.learn_core_loop("CS101", "binary_search")

    # Assert results
    assert result.success
    assert "content" in result.data
    assert "binary search" in result.data['content'].lower()
```

## Configuration

Provider selection follows this priority:

1. Explicit `--provider` flag (highest priority)
2. User preference in web API
3. Environment variable `EXAMINA_LLM_PROVIDER`
4. Config file default (lowest priority)

This allows flexibility while maintaining sensible defaults.

## Conclusion

The service layer makes Examina ready for SaaS deployment while maintaining backward compatibility with the CLI. All business logic remains in the core layer, and the service provides a clean, stateless interface that's easy to integrate with any web framework.
