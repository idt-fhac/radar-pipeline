# radar-pipeline

Data pipeline of the RADAR project: reproducible, LLM-based extraction of decision criteria and their salience weights from online discussions (YouTube, Reddit) as a basis for multi-criteria decision analysis (MCDA).

Developed at the Institute of Data-Driven Technologies (IDT), FH Aachen.

> **Status:** early development. The database layer is being built; the pipeline stages below are planned, not yet implemented.

## Purpose

The pipeline is a research tool. It collects public discussions on a topic, extracts the criteria people use to justify their stance, unifies them into a shared vocabulary and derives salience weights from them. Every result is traceable to its inputs, so that runs can be repeated and compared (e.g. different models, prompts or sources).

**Non-goals:** no web interface, no real-time processing, no analysis of individual persons.

## Pipeline stages

| Stage | What it does | Main tables |
|---|---|---|
| Collection | Fetches videos/threads and their comments for a search query | `search_query`, `collection_run`, `finding`, `container`, `post`, `snapshot` |
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

## Setup

```bash
git clone https://github.com/idt-fhac/radar-pipeline.git
cd radar-pipeline
uv sync
```

Create the database schema (once migrations exist):

```bash
uv run alembic upgrade head
```

After changing a model:

```bash
uv run alembic revision --autogenerate -m "short description"
# review the generated script in migrations/versions/
uv run alembic upgrade head
```

Run all commands from the repository root; the database path `data/radar.db` is relative to it.

## Project structure

```
radar-pipeline/
├── src/radar/
│   └── db/
│       ├── base.py          # declarative base, constraint naming convention
│       ├── engine.py        # engine and sessions
│       └── models/
│           ├── collection.py   # collection stage
│           ├── annotation.py   # sampling and annotation stage
│           └── mapping.py      # mapping stage
├── docs/                    # documentation, data model diagrams
├── migrations/              # Alembic migrations
└── data/                    # local database (not in the repository)
```

## Data protection

- Author names are pseudonymised **on ingestion** with a keyed hash (HMAC) before anything is stored, including raw API responses. The key (`salt.bin`) is not part of the repository.
- Collected data (database, exports, CSV files) and credentials are excluded via `.gitignore` and must never be committed.
- Data is processed for scientific text and data mining under § 60d UrhG.

## Conventions

- Code, identifiers and database names are in English.
- All timestamps are stored in UTC.
- JSON columns are serialised with sorted keys so that equal parameters compare equal.

## License

MIT, see [LICENSE](LICENSE).
