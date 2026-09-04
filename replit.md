# PyGuide FAQ Chatbot

PyGuide is a Flask web app that answers Python and Flask questions using TF-IDF vectorization and cosine similarity.

## Run & Operate

- `python main.py` — run the Flask chatbot (port 5000, or the `PORT` environment variable)
- `pnpm --filter @workspace/api-server run dev` — run the API server (port 5000)
- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from the OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- No environment variables are required. `PORT` and `FLASK_DEBUG=1` are optional.

## Stack

- pnpm workspaces, Node.js 24, TypeScript 5.9
- API: Express 5
- DB: PostgreSQL + Drizzle ORM
- Validation: Zod (`zod/v4`), `drizzle-zod`
- API codegen: Orval (from OpenAPI spec)
- Build: esbuild (CJS bundle)
- Chatbot: Python, Flask, scikit-learn

## Where things live

- `main.py` — FAQ data, TF-IDF index, matching logic, Flask routes
- `templates/index.html` — chat interface and client-side submit behavior
- `static/styles.css` — responsive visual styling
- `requirements.txt` — Python runtime dependencies

## Architecture decisions

- FAQ questions are vectorized once at startup; each user question is transformed and compared with cosine similarity.
- A similarity threshold of `0.22` keeps unrelated questions from returning a misleading FAQ answer.
- The UI calls a small JSON endpoint so the chat can update without a full-page reload.

## Product

- Ask natural-language questions about Python and Flask.
- See the closest matched answer in a chat-style interface.
- Receive the requested fallback response when no FAQ matches confidently.

## User preferences

_No project-specific preferences recorded._

## Gotchas

- Update `FAQS` in `main.py` when changing the subject; the vector index is rebuilt automatically on restart.
- Keep the Flask and scikit-learn versions in `requirements.txt` aligned with the runtime environment.

## Pointers

- See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details
