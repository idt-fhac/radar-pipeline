# radar-pipeline

Data pipeline of the RADAR project: reproducible, LLM-based extraction of decision criteria and their salience weights from online discussions (YouTube, Reddit) as a basis for multi-criteria decision analysis (MCDA).

Developed at the Institute of Data-Driven Technologies (IDT), FH Aachen.

> **Status:** early development. Collection (YouTube), sampling, LLM annotation runs and a local tool for human annotation are implemented and available via the `radar` command. Mapping (embeddings and clustering) is implemented as a library function without a CLI command yet; analysis is in progress.

## Purpose

The pipeline is a research tool. It collects public discussions on a topic, extracts the criteria people use to justify their stance, unifies them into a shared vocabulary and derives salience weights from them. Every result is traceable to its inputs, so that runs can be repeated and compared (e.g. different models, prompts or sources).

**Non-goals:** no web interface for results, no real-time processing, no analysis of individual persons. (A local annotation tool for human annotators is part of the project.)

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
- Flask for the local annotation tool (one static page, no build step, nothing loaded from the internet)
- sentence-transformers (F2LLM) for embeddings, scikit-learn and HDBSCAN for clustering, UMAP for visualisation

## Setup

```bash
git clone https://github.com/idt-fhac/radar-pipeline.git
cd radar-pipeline
uv sync
```

All settings live in `config.example.toml` with their defaults: collection (`[youtube]`), sampling (`[sampling]`), annotation (`[annotation]`), the annotation tool (`[annotation_tool]`), models and mapping. The models are configured there as well: `[llm]` for the annotation (OpenAI-compatible chat APIs with `base_url`, `key_env` – name of the environment variable holding the key – and `default_model`) and `[embedding]` for the mapping (`backend = "local"` for a sentence-transformers model run on this computer, `backend = "openai"` for an OpenAI-compatible `/embeddings` endpoint). To change settings locally, create `config.toml` (ignored by Git) with only the values to change, under the same sections; everything else comes from `config.example.toml`. The LLM provider and its `base_url` are stored with every run; the embedding provider and model are part of `embedding_config.model`, so different deployments never share cached embeddings. What is embedded per criterion is a named pattern in `[embedding.templates]` (placeholders `{topic}`, `{label}`); `[mapping]` holds the defaults of the mapping: UMAP projection before clustering, the method (`hdbscan` or `agglomerative`) with its options and the naming of criteria (`centroid` or `frequent`). Every value a mapping run uses is stored in `mapping_run.params`.

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

### Human annotation

Humans annotate the same samples as the LLMs, in a local tool that writes directly to the database:

```bash
uv run radar --db prod annotate e-auto-300 extraction_v2 --annotator Anna   # opens http://127.0.0.1:8765/
uv run radar --db prod export-human 12 exports/run-12.json                  # write a human run to a file
uv run radar --db prod import-human exports/run-12.json                     # read it into another database
```

`annotate` creates the human run for (sample, prompt, annotator) or continues it: the tool shows the comments that have no annotation in this run yet, one after the other. Stop it with Ctrl+C at any time and start it again with the same arguments to go on. Options: `--port`, `--no-browser`, `--allow-dirty`.

In the tool, mark a passage of the comment with the mouse (it becomes the quote), name the criterion, choose its sentiment and add it; a criterion without quote is possible. Choose the stance of the comment and save. "Keine Kriterien" stores an annotation without criteria, which is a result of its own and different from a comment not annotated yet. "Zurück" goes to the previous comment to correct it. Keys: `1`–`4` set the stance, `Enter` saves and moves on (inside the criterion field it adds the criterion).

- The prompt version is shown as guideline, with the topic inserted: humans read the very text the LLM gets. The title of the video is always shown, which corresponds to the LLM context `text+video_title`.
- Annotation is blind: results of other runs (LLM or human) are never shown.
- Values are stored as in LLM runs: stance `pro`/`contra`/`mixed`/`unclear`, sentiment `positive`/`negative`/`neutral`; each criterion once per comment; a quote must occur word for word in the comment.
- Once a run is input of a mapping run it is locked and only shown, because the mapping refers to its criteria.
- The server listens on `127.0.0.1` only and the page loads nothing from the internet.

**A second annotator** works on their own computer:

1. They get a copy of the production database `data/radar.db` (inside the research group only, never via Git) and the repository at the same commit, so `prompts/` is identical.
2. They annotate with `uv run radar --db prod annotate <sample> <prompt> --annotator <name>`.
3. They export their run (`uv run radar --db prod runs` lists the run ids) with `export-human` and send the file back. `exports/` is ignored by Git; the file contains quotes from the comments.
4. `import-human` adds the run to the main database. Posts are matched by the external ids of video and comment, not by `post.id`. The import checks that the sample exists under its name, the prompt has the same text (hash), every post belongs to the sample, all values are valid and the run was not imported before; if anything fails, nothing is written.

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
│   │                        # parse.py (validate LLM output), run_llm.py (annotation runs),
│   │                        # human/ (annotation tool: app.py server, store.py database access,
│   │                        # transfer.py export/import, static/index.html the page)
│   ├── mapping/             # embed.py (embedding models, templates), cluster.py (UMAP, clustering, naming), map.py (mapping run)
│   └── cli.py               # `radar` command
├── prompts/                 # prompt templates, versioned by file name
├── config.example.toml      # LLM and embedding models (copy to config.toml to change locally)
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
