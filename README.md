# AIntern (MIST 5400, Group 3, Track 2)

Personalised skill-building curriculum for interns. The student enters company,
industry, job description and skills they want. An agent workflow then:

1. **Research** (`POST /api/research`): Claude analyses the role and returns a structured brief.
2. **Generate** (`POST /api/curriculum`): Claude turns the brief and goals into 8 staged tasks
   (objective, deliverable, hours), then each task is scored against the skill goals.
3. **Re-rank** (`POST /api/rerank`): when the student finishes a task, the open tasks are re-scored.
   Skill goals already covered by finished work get less weight, so uncovered goals rise.

Ranking uses Voyage AI embeddings (cosine similarity) when `VOYAGE_API_KEY` is set, otherwise a
TF-IDF keyword fallback. The UI shows which one ran. The fallback is weaker, so use Voyage for the demo.

Nothing is sent to an employer. Progress is stored in the student's browser (localStorage).
This version has no web access: research is based on the student's input and general knowledge,
and the prompt forbids inventing company-specific facts.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env        # then fill in keys, and load them: export $(grep -v '^#' .env | xargs)
uvicorn app:app --reload
# open http://127.0.0.1:8000
python -m pytest            # tests use a mocked Claude, no key needed
```

## Deploy on Render

1. Put this folder in a GitHub repo (the `.env` file is git-ignored; never commit keys).
2. Render dashboard: **New > Blueprint** (reads `render.yaml`) or **New > Web Service**:
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - Health check path: `/api/health`
3. In **Environment**, add `ANTHROPIC_API_KEY` (required) and `VOYAGE_API_KEY` (recommended).
4. Deploy. Open `https://<your-service>.onrender.com/api/health`. It should show `"claude_key_set": true`.
5. Put the service URL as a hyperlink on your Google Slides deck.

Free-tier services sleep after ~15 minutes idle, so the first load can take 30 to 60 seconds.
Open the link a few minutes before presenting.

## Cost and abuse controls (public link)

- Set a monthly spend limit on the API key in the Anthropic Console.
- Per-IP limit: 12 generations and 90 re-ranks per 10 minutes.
- Global cap: `MAX_DAILY_CLAUDE_CALLS` (default 300 Claude calls per UTC day; each plan uses 2).
- Input length caps on every field.
- Each plan costs two Claude calls. Change `ANTHROPIC_MODEL` (for example to `claude-haiku-5-5`) to cut cost and latency.
- Rate limits are in memory, so they reset when the free instance restarts.

## Safety notes

- Job descriptions are wrapped in `<student_input>` tags and treated as data by the prompts.
- Prompts require tasks that use no confidential data or production systems and contact no one automatically.
- If the AI call fails, the intake screen offers a clearly labelled saved example plan (`static/sample.json`).
  The UI shows a banner when a saved plan is displayed.
