"""
Examina Cloud - FastAPI Backend
"""

from app.core.config import settings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Examina Cloud API",
    description="AI-powered exam preparation platform",
    version="0.1.0",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "0.1.0"}


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Examina Cloud API",
        "docs": "/docs",
        "health": "/health",
    }


# Import and include routers
# from app.api.v1 import courses, exercises, quiz, learn, auth
# app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
# app.include_router(courses.router, prefix="/api/v1/courses", tags=["courses"])
# app.include_router(exercises.router, prefix="/api/v1/exercises", tags=["exercises"])
# app.include_router(quiz.router, prefix="/api/v1/quiz", tags=["quiz"])
# app.include_router(learn.router, prefix="/api/v1/learn", tags=["learn"])
