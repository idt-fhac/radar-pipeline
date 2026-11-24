# radar-pipeline

Data pipeline of the RADAR project: reproducible, LLM-based extraction of decision criteria and their salience weights from online discussions (YouTube, Reddit) as a basis for multi-criteria decision analysis (MCDA).

Developed at the Institute of Data-Driven Technologies (IDT), FH Aachen.

> **Status:** early development. Collection (YouTube), sampling and LLM annotation runs are implemented and available via the `radar` command. Mapping (embeddings and clustering) is implemented as a library function without a CLI command yet; analysis is in progress.

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
- sentence-transformers (F2LLM) for embeddings, scikit-learn and HDBSCAN for clustering, UMAP for visualisation

## Setup

```bash
git clone https://github.com/idt-fhac/radar-pipeline.git
cd radar-pipeline
uv sync
```

LLM providers (OpenAI-compatible APIs) are configured in `providers.example.toml`. To use other providers, copy it to `providers.toml` (ignored by Git, takes precedence) and add an entry with `base_url`, `key_env` (name of the environment variable holding the key) and `default_model`. The provider name and its `base_url` are stored with every run.

Create a `.env` file in the repository root (never commit it):

```
YT_API_KEY=...      # YouTube Data API v3
IDT_API_KEY=...     # LLM API of the IDT (default provider)
KI_API_KEY=...      # KI:connect.NRW (alternative provider)
SALT_HEX=...        # key for pseudonymisation, e.g. python -c "import secrets; print(secrets.token_hex(32))"
```

There are two databases: `data/radar-test.db` (default, for development and test runs) and `data/radar.db` (production). Production has to be chosen explicitly:

```powershell
$env:RADAR_DB = "prod"          # PowerShell; stays set for this terminal
Remove-Item Env:RADAR_DB        # back to the test database
```

`SALT_HEX` must never change once real data has been collected, otherwise the same person gets a new pseudonym.

Create the database schema (once per database):

```bash
uv run alembic upgrade head
```

After changing a model:

```bash
uv run alembic revision --autogenerate -m "short description"
# review the generated script in migrations/versions/
uv run alembic upgrade head
```

All paths (databases under `data/`, `prompts/`) are built from the repository root, so commands work from any directory.

Run the tests:

```bash
uv run pytest
```

## Usage

All stages are run with the `radar` command (`uv run radar --help`, `uv run radar <command> --help`).
`--db prod` selects the production database (default: test); `--sql` logs every SQL statement.

```bash
uv run radar collect "E-Auto" "Verbrenner-Aus" --max-videos 25   # search YouTube, store videos and comments
uv run radar sample e-auto-main "E-Auto"                          # load, filter, freeze (prints the attrition)
uv run radar subset e-auto-300 e-auto-main 300 --seed 42          # reproducible random subset
uv run radar run e-auto-300 extraction_v1 --context text+video_title   # LLM annotation run
uv run radar resume 7                                             # continue an aborted run
uv run radar runs                                                 # list runs
uv run radar show 7                                               # results: stance, criteria, quotes
uv run radar models kiconnect                                     # model ids of a provider
```

`run` options: `--provider idt|kiconnect`, `--model`, `--temperature`, `--batch-size`, `--context text|text+video_title`, `--thinking/--no-thinking`, `--max-rounds`, `--workers`. Unset options use the defaults in the code. Every command that stores something refuses to run with uncommitted code changes unless `--allow-dirty` is given.

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
│   ├── annotation/          # llm_client.py (LLM APIs), prompts.py (prompt versions),
│   │                        # parse.py (validate LLM output), run_llm.py (annotation runs)
│   ├── mapping/             # f2llm.py (embedding model), map.py (embed, cluster, store criteria)
│   └── cli.py               # `radar` command
├── prompts/                 # prompt templates, versioned by file name
├── providers.example.toml   # LLM providers (copy to providers.toml to change locally)
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
