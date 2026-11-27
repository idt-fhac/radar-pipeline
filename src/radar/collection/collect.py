# collect.py — writes data from the sources into the database
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from radar import privacy
from radar.db.models.collection import (
    Author, CollectionRun, Container, Finding, Post, SearchQuery, Snapshot, Topic,
)
from radar.collection import raw
from radar.sources import youtube
from radar.provenance import get_code_version

import logging

logger = logging.getLogger(__name__)

def collect_youtube(session, topic_name, search_term, max_videos = youtube.MAX_VIDEOS, max_threads_per_video = youtube.MAX_THREADS_PER_VIDEO, comment_order = youtube.COMMENT_ORDER, replies = youtube.REPLIES, search_params = None, allow_dirty = False):
    """Search YouTube and store the videos found with their comments as one collection run.

    Commits itself: once after creating the run and once per video, so an aborted run keeps
    every video it finished (collection_run.finished_at stays NULL).
    :return: the CollectionRun
    """
    search_params = youtube.SEARCH_PARAMS if search_params is None else search_params
    fetch_params = {'max_videos': max_videos, 'max_threads_per_video': max_threads_per_video, 'comment_order': comment_order, 'replies': replies}

    topic = get_or_create_topic(session, topic_name)
    search_query = get_or_create_search_query(session, 'youtube', search_term, search_params)
    if search_query not in topic.search_queries:
        topic.search_queries.append(search_query)
    collection_run = create_collection_run(session, search_query=search_query, fetch_params=fetch_params, allow_dirty=allow_dirty)
    session.commit()
    vids = youtube.get_vids(search_term = search_term, max_videos = max_videos, search_params = search_params)
    video_ids = [vid['id']['videoId'] for vid in vids]
    vid_details = youtube.get_video_details(video_ids = video_ids)
    for rank, vid in enumerate(vids, start = 1):
        video_id = vid['id']['videoId']
        if video_id not in vid_details:
            continue
        published_at = datetime.fromisoformat(vid_details[video_id]['snippet']['publishedAt'])
        likes = int(vid_details[video_id]['statistics'].get('likeCount')) if vid_details[video_id]['statistics'].get('likeCount') is not None else None
        comment_count = int(vid_details[video_id]['statistics'].get('commentCount')) if vid_details[video_id]['statistics'].get('commentCount') is not None else None
        # comments first: their authors' handles are needed to pseudonymise mentions
        comments = youtube.get_comments(video_id, max_threads = max_threads_per_video, order = comment_order, replies = replies)
        handles = raw.build_handle_map(comments)

        video_raw = raw.clean_video(vid_details[video_id], handles)
        # title from the cleaned video details, not from the search result: search.list
        # escapes HTML ('&amp;') and its title would be stored with plain @-mentions
        container = get_or_create_container(session, source = search_query.source, external_id = video_id, title = video_raw['snippet']['title'])
        create_finding(session, collection_run = collection_run, container = container, rank = rank)
        channel = get_or_create_author(session, video_raw['snippet']['channelId'])
        root = get_or_create_post(session, container = container, external_id = video_id, text = video_raw['snippet']['description'], published_at = published_at, author = channel, raw = video_raw)
        create_snapshot(session, post = root, collection_run = collection_run, likes = likes, reply_count = comment_count, updated_at = published_at)
        # YouTube ids -> Post objects of this video; a top-level comment always comes
        # before its replies, so the parent of a reply is already in here
        posts_by_external_id = {}
        for comment in comments:
            clean_comment = raw.clean_comment(comment, handles)
            snippet = clean_comment['snippet']
            external_id = clean_comment['id']
            pseudonym = snippet.get('authorChannelId', {}).get('value')  # None for deleted users
            author = get_or_create_author(session, pseudonym = pseudonym) if pseudonym else None
            text = snippet.get('textOriginal', snippet['textDisplay'])

            parent_external_id = snippet.get('parentId')
            if parent_external_id is None:
                parent = root
                reply_count = clean_comment['totalReplyCount']
            else: # the api only references the top level comment, not the actual comment the reply is referencing
                parent = posts_by_external_id[parent_external_id]
                reply_count = 0
            post = get_or_create_post(session, container = container, external_id = external_id, text = text, published_at = datetime.fromisoformat(snippet['publishedAt']), author = author, raw = clean_comment, parent = parent)
            posts_by_external_id[external_id] = post
            create_snapshot(session, post = post, collection_run = collection_run, likes = snippet.get('likeCount'), reply_count = reply_count, updated_at = datetime.fromisoformat(snippet['updatedAt']))
        session.commit()
    collection_run.finished_at = datetime.now(timezone.utc)
    session.commit()
    return collection_run


def get_or_create_topic(session, name):
    topic = session.scalar(select(Topic).where(Topic.name == name).options(selectinload(Topic.search_queries)))
    if topic is None:
        # empty list = collection counts as loaded; otherwise the autoflush before the next
        # query turns it into an unloaded collection and lazy='raise' refuses to load it
        topic = Topic(name=name, search_queries=[])
        session.add(topic)
    return topic

def get_or_create_search_query(session, source, search_term, search_params):
    search_query = session.scalar(select(SearchQuery).where(SearchQuery.source == source, SearchQuery.search_term == search_term, SearchQuery.search_params == search_params))
    if search_query is None:
        search_query = SearchQuery(source = source, search_term = search_term, search_params = search_params)
        session.add(search_query)
    return search_query

def get_or_create_author(session, pseudonym):
    author = session.scalar(select(Author).where(Author.pseudonym == pseudonym))
    if author is None:
        author = Author(pseudonym = pseudonym)
        session.add(author)
    return author

def get_or_create_container(session, source, external_id, title):
    container = session.scalar(select(Container).where(Container.source == source, Container.external_id == external_id))
    if container is None:
        container = Container(source = source, external_id = external_id, title=title)
        session.add(container)
    return container

def create_collection_run(session, search_query, fetch_params, allow_dirty = False):
    #started_at = datetime.now(timezone.utc)
    collection_run = CollectionRun(search_query = search_query, started_at = datetime.now(timezone.utc), fetch_params = fetch_params, code_version = get_code_version(allow_dirty=allow_dirty))
    session.add(collection_run)
    return collection_run

def create_finding(session, collection_run, container, rank):
    finding = Finding(collection_run = collection_run, container = container, rank = rank)
    session.add(finding)
    return finding

def get_or_create_post(session, container, external_id, author, text, published_at, raw, parent = None):
    post = session.scalar(select(Post).where(Post.container_id == container.id, Post.external_id == external_id))
    if post is None:
        post = Post(container = container, external_id = external_id, author = author, text = text, published_at = published_at, raw = raw, parent = parent)
        session.add(post)
    return post

def create_snapshot(session, post, collection_run, likes, reply_count, updated_at):
    snapshot = Snapshot(post = post, collection_run = collection_run, likes = likes, reply_count = reply_count, updated_at = updated_at)
    session.add(snapshot)
    return snapshot