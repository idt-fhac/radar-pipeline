# Collection: sequence diagram

This diagram shows how `collect_youtube()` in `src/radar/collection/collect.py` fills **Block 1: Collection** of the [data model](data-model.md). It covers one call: one topic, one search term, one collection run.

Participants:

| Participant | Module | Role |
|---|---|---|
| `collect` | `radar.collection.collect` | Orchestrates the run, commits it step by step |
| `DB` | session of the caller + `radar.db.models.collection` | SQLAlchemy session and ORM tables |
| `provenance` | `radar.provenance` | Git commit as `code_version` |
| `youtube` | `radar.sources.youtube` | Wraps the YouTube Data API v3 (paging, batching) |
| `API` | YouTube Data API v3 | External service |
| `raw` | `radar.collection.raw` | Strips personal data from raw API items |
| `privacy` | `radar.privacy` | HMAC pseudonyms, replaces @-mentions |

```mermaid
sequenceDiagram
    autonumber
    actor Caller
    participant collect as collect.py
    participant DB as DB Session
    participant provenance as provenance.py
    participant youtube as sources/youtube.py
    participant API as YouTube Data API
    participant raw as collection/raw.py
    participant privacy as privacy.py

    Caller->>DB: open Session()
    Caller->>collect: collect_youtube(session, topic_name, search_term, max_videos, ...)

    rect rgba(127,127,127,0.08)
    Note over collect,DB: Set up the run (topic, search_query, collection_run)
    collect->>DB: get_or_create_topic(name)
    DB-->>collect: topic (new or existing, with search_queries)
    collect->>DB: get_or_create_search_query('youtube', search_term, search_params)
    DB-->>collect: search_query
    opt search_query not linked to topic
        collect->>DB: topic.search_queries.append(search_query)<br/>(row in topic_search_query)
    end
    collect->>provenance: get_code_version(allow_dirty)
    provenance-->>collect: git commit hash
    collect->>DB: create_collection_run(search_query, fetch_params)<br/>started_at = now, finished_at = NULL
    collect->>DB: commit()
    end

    rect rgba(127,127,127,0.08)
    Note over collect,API: Search and video details
    collect->>youtube: get_vids(search_term, max_videos, search_params)
    loop until max_videos or no nextPageToken
        youtube->>API: search.list(q, type=video, pageToken)
        API-->>youtube: page of search results
    end
    youtube-->>collect: vids (ranked)
    collect->>youtube: get_video_details(video_ids)
    loop batches of up to 50 ids
        youtube->>API: videos.list(part=snippet,statistics)
        API-->>youtube: items
    end
    youtube-->>collect: vid_details {video_id: item}
    end

    loop for each vid (rank = 1..n)
        alt video_id missing from vid_details (deleted / private)
            collect->>collect: skip video
        else details available
            collect->>DB: get_or_create_container('youtube', video_id, title)
            DB-->>collect: container
            collect->>DB: create_finding(collection_run, container, rank)

            Note over collect,API: Comments are fetched first: their handles are<br/>needed to pseudonymise mentions in the video text
            collect->>youtube: get_comments(video_id, max_threads, order, replies)
            loop comment threads (paged, up to max_threads)
                youtube->>API: commentThreads.list(videoId, order)
                API-->>youtube: threads (or 403 commentsDisabled -> [])
            end
            opt replies == 'all' and thread has more replies than included
                youtube->>API: comments.list(parentId) (paged)
                API-->>youtube: all replies
            end
            youtube-->>collect: flat comment list<br/>(top-level comment before its replies)

            collect->>raw: build_handle_map(comments)
            raw->>privacy: pseudonym('youtube', channel_id) per author
            privacy-->>raw: HMAC pseudonym
            raw-->>collect: handles {"@name": pseudonym}

            Note over collect,DB: Root post = the video itself
            collect->>raw: clean_video(details, handles)
            raw->>privacy: pseudonym(channelId), pseudonymize_mentions(title, description)
            privacy-->>raw: pseudonymised values
            raw-->>collect: video_raw (no channelTitle)
            collect->>DB: get_or_create_author(pseudonymised channelId)
            DB-->>collect: channel author
            collect->>DB: get_or_create_post(container, video_id, description,<br/>author=channel, raw=video_raw, parent=NULL)
            DB-->>collect: root post (existing post is not overwritten)
            collect->>DB: create_snapshot(root, collection_run,<br/>likes, reply_count=commentCount, updated_at=publishedAt)

            loop for each comment
                collect->>raw: clean_comment(comment, handles)
                raw->>privacy: pseudonym(authorChannelId), pseudonymize_mentions(text)
                privacy-->>raw: pseudonymised values
                raw-->>collect: clean_comment (no name, image, channel URL)
                opt author pseudonym present (not a deleted user)
                    collect->>DB: get_or_create_author(pseudonym)
                    DB-->>collect: author
                end
                alt top-level comment (no parentId)
                    collect->>collect: parent = root, reply_count = totalReplyCount
                else reply
                    collect->>collect: parent = posts_by_external_id[parentId]<br/>reply_count = 0
                end
                collect->>DB: get_or_create_post(container, comment_id, text,<br/>author, raw=clean_comment, parent)
                DB-->>collect: post
                collect->>collect: posts_by_external_id[comment_id] = post
                collect->>DB: create_snapshot(post, collection_run,<br/>likeCount, reply_count, updatedAt)
            end
            collect->>DB: commit()
        end
    end

    collect->>DB: collection_run.finished_at = now
    collect->>DB: commit()
    collect-->>Caller: collection_run
    Caller->>DB: close Session
```

## Notes

- **Session.** The caller (the CLI, a test) opens the session and passes it in; `collect_youtube` never opens one itself, so it can run against any database. It commits itself, because a run is stored step by step.
- **Commit points.** The run is committed before any API call, and then once per video. If the process aborts, the videos stored so far are kept and `collection_run.finished_at` stays `NULL`, which the data model defines as "aborted or running".
- **Append, don't overwrite.** `topic`, `search_query`, `container`, `author` and `post` are *get-or-create*. A post that already exists from an earlier run keeps its text and `raw` ("first fetch only"). Only `finding` and `snapshot` are always created new, so every run records its own ranks, likes, reply counts and edit times.
- **Reply tree.** The YouTube API only references the top-level comment of a reply, so every reply gets that comment as its parent. Because `get_comments` returns a top-level comment before its replies, the parent is already in `posts_by_external_id` when a reply is processed.
- **Pseudonymisation happens before storage.** Author IDs (channels and commenters) are replaced by HMAC pseudonyms, and @-mentions of known handles in titles, descriptions and comment texts are replaced as well. Only cleaned copies reach `post.raw` and `author.pseudonym`.
