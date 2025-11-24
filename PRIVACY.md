# Privacy Policy

## Your Course Materials Stay Yours

Examina analyzes the materials you upload (past exams, homework, problem sets, lecture notes) to build a private knowledge base and tutor for you only. We don't share your materials or generated questions with other users.

## What Happens When You Upload Course Materials

1. **Storage**: Materials are stored in your account only and used to extract exercises, topics, and procedures.

2. **LLM Processing**: The content may be sent to our LLM providers (e.g., Anthropic, Groq, or your own local Ollama) only to generate explanations, quizzes, and summaries.

3. **No Training**: We don't sell your data or use your course materials to train our own models.

4. **Your Control**: You can delete your uploads and associated data at any time.

## LLM Provider Privacy

When using external LLM providers (Anthropic, Groq, OpenAI), your exam content is sent to their APIs for processing:

- **Anthropic**: Subject to [Anthropic's Commercial Terms](https://www.anthropic.com/legal/commercial-terms) - they do not train on customer data
- **Groq**: Subject to [Groq's Terms of Service](https://groq.com/terms-of-service/)
- **Ollama (Local)**: All processing happens on your machine - no data leaves your computer

## Data Retention

- **Course Material PDFs**: Stored in your local database (`data/examina.db`) until you delete them
- **Analysis Cache**: LLM responses are cached locally to save API costs and improve performance
- **Quiz History**: Your quiz attempts and progress are stored locally
- **Rate Limit Tracking**: API usage is tracked locally in cache files

## Your Rights

You have the right to:

1. **Access**: View all your stored data in the SQLite database
2. **Delete**: Remove any course, exercise, or analysis data
3. **Export**: Extract your data from the database
4. **Use Locally**: Run Examina entirely offline with Ollama (no cloud API)

## Data Security

- **Local Storage**: All data is stored in SQLite database on your machine
- **API Keys**: Your API keys are stored in environment variables, not in code
- **No Cloud Storage**: Examina is currently a CLI tool - all data stays on your computer

## Future Web Application

When Examina becomes a web application:

- This privacy policy will be updated to reflect web-based storage
- User accounts will be separate and isolated
- All existing privacy principles will remain (your exams stay yours)
- You will be notified of any changes before migration

## Contact

For privacy-related questions or concerns:
- Open an issue: https://github.com/madebymlai/Examina/issues
- Email: mikhail@laimk.dev

---

*Last Updated: 2025-11-24*
