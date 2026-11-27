# Data model

The database is split into three blocks that follow the pipeline stages. Each block is implemented in its own module under `src/radar/db/models/`.

Notation: `||--o{` means "exactly one to zero or more". Several `UK` markers in one table form **one** composite unique key, unless a column is noted as unique on its own.

## Block 1: Collection (`collection.py`)

A **search query** describes *what* is searched; a **collection run** describes *when* it was executed and how deeply comments were fetched. Running the same query again appends data instead of overwriting it. A **finding** records which run found which container (video or thread) at which rank. A **container** holds **posts** as a tree: every container has exactly one root post (the video with its description, or the thread post), and comments point to their parent. Values that change between fetches (likes, reply count, last edit) live in **snapshots**.

```mermaid
erDiagram
    topic ||--o{ topic_search_query : ""
    search_query ||--o{ topic_search_query : ""
    search_query ||--o{ collection_run : "is executed in"
    collection_run ||--o{ finding : yields
    container ||--o{ finding : "appears in"
    container ||--o{ post : contains
    author |o--o{ post : writes
    post |o--o{ post : "replies to"
    post ||--o{ snapshot : "has state"
    collection_run ||--o{ snapshot : captures

    topic { int id PK
            string name UK }
    topic_search_query { int topic_id PK, FK
                         int search_query_id PK, FK }
    search_query { int id PK
                   string source UK
                   string search_term UK
                   json search_params UK "serialised with sorted keys" }
    collection_run { int id PK
                     int search_query_id FK
                     datetime started_at
                     datetime finished_at "NULL = aborted or running"
                     json fetch_params
                     string code_version "Git commit" }
    finding { int collection_run_id PK, FK
              int container_id PK, FK
              int rank }
    container { int id PK
                string source UK
                string external_id UK
                string title "as first fetched, mentions pseudonymised" }
    author { string pseudonym PK "HMAC of the account ID" }
    post { int id PK
           int container_id FK, UK
           string external_id UK
           int parent_id FK "NULL = root post"
           string author_pseudonym FK "NULL = unknown or deleted"
           string text
           datetime published_at
           json raw "first fetch only, pseudonymised"
           }
    snapshot { int post_id PK, FK
               int collection_run_id PK, FK
               int likes "NULL = hidden or not reported"
               int reply_count "NULL = not reported"
               datetime updated_at "NULL = edit time unknown" }
```

## Block 2: Sampling and annotation (`annotation.py`)

A **sample** is a frozen set of posts for one topic; its `definition` records the filters (and seed) that produced it, so it can be reproduced. Samples can be subsets of other samples. An **annotation run** extracts criteria from a sample with one **prompt version**, either by an LLM or by a human; both share the same structure, so runs and annotators are compared the same way. **Batches** record what was sent to the LLM, **annotations** what came back, so skipped posts can be detected and resent. A post without criteria still gets an (empty) annotation.

```mermaid
erDiagram
    topic ||--o{ sample : "belongs to"
    sample ||--o{ sample_post : includes
    post ||--o{ sample_post : ""
    sample |o--o{ sample : "is subset of"
    sample ||--o{ annotation_run : "is annotated in"
    prompt_version ||--o{ annotation_run : instructs
    annotation_run ||--o| llm_run : "is a"
    annotation_run ||--o| human_run : "is a"
    llm_run ||--o{ batch : sends
    batch ||--|{ batch_post : contains
    post ||--o{ batch_post : ""
    annotation_run ||--o{ annotation : produces
    post ||--o{ annotation : ""
    annotation ||--o{ criterion_mention : contains

    sample { int id PK
             string name UK
             int topic_id FK
             int base_sample_id FK "NULL = not a subset"
             json definition "filters, parameters, seed"
             string code_version
             datetime created_at }
    sample_post { int sample_id PK, FK
                  int post_id PK, FK }
    prompt_version { int id PK
                     string name UK "unique on its own (file name in prompts/)"
                     string text
                     string hash UK "unique on its own (SHA-256 of text)" }
    annotation_run { int id PK
                     int sample_id FK
                     int prompt_version_id FK
                     string code_version
                     datetime started_at }
    llm_run { int run_id PK, FK
              string model "as requested; the name the API reports is in batch.raw_response"
              float temperature
              int batch_size
              string context_strategy "what the LLM gets besides the comment"
              json params "provider, base_url, thinking, json_mode" }
    human_run { int run_id PK, FK
                string annotator }
    batch { int id PK
            int run_id FK, UK
            int number UK
            string status "sent, done, partial, invalid or error"
            json raw_response "NULL if the request failed" }
    batch_post { int batch_id PK, FK
                 int post_id PK, FK
                 int position }
    annotation { int run_id PK, FK
                 int post_id PK, FK
                 string stance "NULL if not asked for" }
    criterion_mention { int id PK
                        int run_id FK "composite FK to annotation"
                        int post_id FK "composite FK to annotation"
                        string label
                        string sentiment "nullable"
                        string quote "nullable" }
```

## Block 3: Mapping (`mapping.py`)

Criterion labels from different runs name the same thing differently ("CO2", "climate protection"). An **embedding config** defines the model and a template for the text that is embedded (first version: topic prefix plus label; later possibly plus the quote). **Embeddings** are cached by the hash of that text. A **mapping run** clusters the embeddings of the mentions from its input runs; each cluster becomes a **criterion**, outliers become criteria of their own. **Assignments** link every mention to its criterion. Criteria belong to a mapping run, so different thresholds or methods yield comparable but separate vocabularies.

```mermaid
erDiagram
    embedding_config ||--o{ embedding : produces
    embedding_config |o--o{ mapping_run : "is used by"
    mapping_run ||--o{ mapping_input : ""
    annotation_run ||--o{ mapping_input : "provides mentions"
    mapping_run ||--o{ criterion : defines
    criterion ||--o{ assignment : ""
    criterion_mention ||--o{ assignment : ""
    mapping_run ||--o{ assignment : ""

    embedding_config { int id PK
                       string model UK
                       string template UK }
    embedding { int id PK
                int config_id FK, UK
                string input_hash UK
                string input "rendered text sent to the model"
                blob vector }
    mapping_run { int id PK
                  int embedding_config_id FK "NULL for llm or manual mapping"
                  string method
                  float threshold "NULL if the method has none"
                  json params
                  string code_version
                  datetime created_at }
    mapping_input { int mapping_run_id PK, FK
                    int annotation_run_id PK, FK }
    criterion { int id PK
                int mapping_run_id FK
                string name "label closest to the cluster mean" }
    assignment { int mapping_run_id PK, FK
                 int mention_id PK, FK
                 int criterion_id FK }
```

## From mentions to weights

Salience weights are not stored; they are computed as SQL views along the chain post → annotation → criterion mention → assignment (for one mapping run) → criterion, e.g. distinct authors per criterion and month.

## Computed, not stored

Everything that follows from stored data is computed on request, so it cannot disagree with the data and needs no table or column:

- **Quote fidelity** of a run (`radar.analysis.quotes`, `radar quotes`): the share of `criterion_mention.quote` values that occur word for word in `post.text` of their post.
- **Root post or comment**: `post.parent_id IS NULL` marks the root post; a reply is a post whose parent has a parent itself.
- **Kind of a run**: an annotation run is an LLM run or a human run by the subtype row it has.
