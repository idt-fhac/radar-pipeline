# radar-pipeline

Data pipeline of the RADAR project: reproducible, LLM-based extraction of decision criteria and their salience weights from online discussions (YouTube, Reddit) as a basis for multi-criteria decision analysis (MCDA).

Developed at the Institute of Data-Driven Technologies (IDT), FH Aachen.

> **Status:** early development. Collection (YouTube), sampling and the LLM client are implemented; LLM annotation runs, mapping and analysis are in progress. There is no command-line interface yet.

## Purpose

The pipeline is a research tool. It collects public discussions on a topic, extracts the criteria people use to justify their stance, unifies them into a shared vocabulary and derives salience weights from them. Every result is traceable to its inputs, so that runs can be repeated and compared (e.g. different models, prompts or sources).

**Non-goals:** no web interface, no real-time processing, no analysis of individual persons.

## Pipeline stages

| Stage | What it does | Main tables |
|---|---|---|
| Collection | Fetches videos/threads and their comments for a search query (YouTube implemented, Reddit planned) | `search_query`, `collection_run`, `finding`, `container`, `post`, `snapshot` |
| Sampling | Freezes a cleaned set of posts as input for annotation | `sample`, `sample_post` |
| Annotation | Extracts criteria per post, by an LLM or a human annotator | `annotation_run`, `llm_run`, `human_run`, `batch`, `annotation`, `criterion_mention` |
| Mapping | Embeds criterion labels and clusters them into canonical criteria | `embedding_config`, `embedding`, `mapping_run`, `criterion`, `assignment` |
| Analysis | Salience weights and time series, computed as SQL views | — |

The full data model with diagrams is documented in [docs/data-model.md](docs/data-model.md).

## Design principles

1. **Raw data is append-only.** Collected data is never overwritten; repeated collection adds new snapshots.
2. **Every derived result knows its origin.** Runs store model, prompt version, parameters and the Git commit (`code_version`) they were produced with.
3. **Humans and LLMs annotate in the same structure**, so agreement between runs and between human and model is computed the same way.
4. **Stages only communicate through the database.** Source adapters (YouTube, Reddit) translate into a common model.
5. **Store what is expensive, non-deterministic or a research decision; compute everything else.**

## Tech stack

- Python 3.12, managed with [uv](https://docs.astral.sh/uv/)
- SQLite as database (single file under `data/`)
- SQLAlchemy 2.x (ORM) for models and writes; analyses as SQL views
- Alembic for schema migrations
- OpenAI Python SDK for LLM access via OpenAI-compatible APIs (KI:connect.NRW, IDT)

## Setup

```bash
git clone https://github.com/idt-fhac/radar-pipeline.git
cd radar-pipeline
uv sync
```

Create a `.env` file in the repository root (never commit it):

```
YT_API_KEY=...      # YouTube Data API v3
IDT_API_KEY=...     # LLM API of the IDT (default provider)
KI_API_KEY=...      # KI:connect.NRW (alternative provider)
SALT_HEX=...        # key for pseudonymisation, e.g. python -c "import secrets; print(secrets.token_hex(32))"
```

`SALT_HEX` must never change once real data has been collected, otherwise the same person gets a new pseudonym.

Create the database schema:

```bash
uv run alembic upgrade head
```

After changing a model:

```bash
uv run alembic revision --autogenerate -m "short description"
# review the generated script in migrations/versions/
uv run alembic upgrade head
```

All paths (database `data/radar.db`, `prompts/`) are built from the repository root, so commands work from any directory.

Run the tests:

```bash
uv run pytest
```

## Usage

No command-line interface yet; the stages are called from Python.

```python
from radar.db.engine import Session
from radar.collection.collect import collect_youtube
from radar.sampling.sample import create_sample, create_subset

# 1. Collect: search YouTube and store videos and comments (refuses a dirty Git tree)
collect_youtube('E-Auto', 'Verbrenner-Aus', max_videos=25)

# 2. Sample: load, filter (empty, short, duplicate posts) and freeze; the caller commits
with Session() as session:
    create_sample(session, 'e-auto-main', 'E-Auto')
    create_subset(session, 'e-auto-300', 'e-auto-main', 300, seed=42)
    session.commit()
```

LLM access (no database involved):

```python
from radar import ROOT
from radar.annotation.llm_client import annotate_batch, list_models, render_prompt

template = (ROOT / 'prompts' / 'extraction_v1.txt').read_text(encoding='utf-8')
result = annotate_batch(render_prompt(template, 'E-Auto'), [{'id': 1, 'text': '...'}])
result['parsed'], result['error']
list_models('idt')  # model ids of a provider
```

Prompts live in `prompts/` and are stored as prompt versions under their file name. A prompt that has been used in a run must not be edited; save changes as a new file (e.g. `extraction_v2.txt`).

## Project structure

```
radar-pipeline/
├── src/radar/
│   ├── __init__.py          # ROOT path, loads .env
│   ├── privacy.py           # pseudonymisation of authors and @-mentions
│   ├── provenance.py        # code_version: Git commit of the running code
│   ├── db/
│   │   ├── base.py          # declarative base, naming convention, readable repr
│   │   ├── engine.py        # engine, foreign key pragma, sessions
│   │   └── models/          # collection.py, annotation.py, mapping.py
│   ├── sources/
│   │   └── youtube.py       # YouTube Data API client (no database access)
│   ├── collection/          # collect.py (store a collection run), raw.py (clean raw JSON)
│   ├── sampling/            # filters.py (content filters), sample.py (load, freeze, subsets)
│   └── annotation/          # llm_client.py (LLM APIs), prompts.py (prompt versions)
├── prompts/                 # prompt templates, versioned by file name
├── docs/                    # documentation, data model diagrams
├── migrations/              # Alembic migrations
├── tests/
└── data/                    # local database (not in the repository)
```

## Data protection

- Author names are pseudonymised **on ingestion** with a keyed hash (HMAC) before anything is stored, including raw API responses. The key (`SALT_HEX` in `.env`) is not part of the repository. @-mentions in texts are pseudonymised the same way.
- Collected data (database, exports, CSV files) and credentials are excluded via `.gitignore` and must never be committed.
- Data is processed for scientific text and data mining under § 60d UrhG.

## Conventions

- Code, identifiers and database names are in English.
- All timestamps are stored in UTC.
- JSON columns are serialised with sorted keys so that equal parameters compare equal.

## License

MIT, see [LICENSE](LICENSE).
