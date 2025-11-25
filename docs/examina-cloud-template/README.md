# Examina Cloud

Private SaaS platform for Examina - AI-powered exam preparation.

## Repository Structure

```
examina-cloud/
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── api/            # API routes
│   │   │   ├── v1/
│   │   │   │   ├── courses.py
│   │   │   │   ├── exercises.py
│   │   │   │   ├── quiz.py
│   │   │   │   ├── learn.py
│   │   │   │   └── auth.py
│   │   │   └── deps.py     # Dependencies (auth, db)
│   │   ├── core/           # App config
│   │   │   ├── config.py
│   │   │   └── security.py
│   │   ├── models/         # Pydantic models
│   │   ├── services/       # Business logic (imports examina-core)
│   │   └── main.py
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/               # React/Vue frontend
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── services/       # API client
│   │   └── App.tsx
│   ├── package.json
│   └── Dockerfile
├── worker/                 # Background job workers
│   ├── tasks/
│   │   ├── analyze.py      # PDF analysis jobs
│   │   └── ingest.py       # Ingestion jobs
│   ├── celery_app.py
│   └── Dockerfile
├── infrastructure/         # Deployment configs
│   ├── docker-compose.yml
│   ├── docker-compose.prod.yml
│   ├── nginx.conf
│   └── kubernetes/         # K8s manifests (optional)
├── migrations/             # Database migrations
│   └── alembic/
├── scripts/                # Dev/deploy scripts
│   ├── setup-dev.sh
│   └── deploy.sh
├── .github/
│   └── workflows/
│       ├── ci.yml          # Tests on PR
│       └── deploy.yml      # Deploy on merge
├── .env.example
├── .gitignore
├── docker-compose.yml      # Local dev
└── README.md
```

## Quick Start (Local Development)

```bash
# Clone
git clone git@github.com:madebymlai/examina-cloud.git
cd examina-cloud

# Copy env file
cp .env.example .env
# Edit .env with your API keys

# Start services
docker-compose up -d

# Backend at http://localhost:8000
# Frontend at http://localhost:3000
# API docs at http://localhost:8000/docs
```

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Frontend  │────▶│   Backend   │────▶│  PostgreSQL │
│  (React)    │     │  (FastAPI)  │     │             │
└─────────────┘     └──────┬──────┘     └─────────────┘
                           │
                    ┌──────▼──────┐     ┌─────────────┐
                    │   Worker    │────▶│    Redis    │
                    │  (Celery)   │     │   (Queue)   │
                    └──────┬──────┘     └─────────────┘
                           │
                    ┌──────▼──────┐
                    │ examina-core│  (pip install from public repo)
                    └─────────────┘
```

## Key Design Decisions

1. **examina-core as dependency** - Import from public repo, don't duplicate code
2. **Multi-tenant from day 1** - All queries scoped by user_id
3. **Background jobs for analysis** - Long-running tasks via Celery
4. **JWT authentication** - Stateless, scalable auth
5. **PostgreSQL** - User data, progress, subscriptions

## Environment Variables

See `.env.example` for full list. Key ones:

```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/examina

# Redis (job queue)
REDIS_URL=redis://localhost:6379

# JWT
JWT_SECRET_KEY=your-secret-key
JWT_ALGORITHM=HS256

# LLM Providers (from examina-core)
DEEPSEEK_API_KEY=...
GROQ_API_KEY=...
ANTHROPIC_API_KEY=...

# Stripe (payments)
STRIPE_SECRET_KEY=...
STRIPE_WEBHOOK_SECRET=...
```

## Deployment

See `infrastructure/` for Docker and Kubernetes configs.

## License

Proprietary - All rights reserved.
