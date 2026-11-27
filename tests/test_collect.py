# collect_youtube against a faked YouTube client and a temporary database.

import json
from datetime import datetime

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from radar.collection import collect
from radar.collection.collect import collect_youtube
from radar.db.models.collection import (Author, CollectionRun, Container, Finding, Post, SearchQuery,
                                        Snapshot, Topic)
from radar.privacy import pseudonym
from radar.sources import youtube

ANNA = pseudonym('youtube', 'UC-anna')
BEN = pseudonym('youtube', 'UC-ben')
CHANNEL = pseudonym('youtube', 'UC-kanal')


def video(video_id, title='E-Auto im Test', description='Beschreibung', likes='120', comment_count='3'):
    """A videos.list item; the API sends 64-bit numbers as strings."""
    statistics = {'viewCount': '9999'}
    if likes is not None:
        statistics['likeCount'] = likes
    if comment_count is not None:
        statistics['commentCount'] = comment_count
    return {'kind': 'youtube#video', 'id': video_id, 'statistics': statistics, 'snippet': {
        'channelId': 'UC-kanal', 'channelTitle': 'Kanalname', 'title': title, 'description': description,
        'publishedAt': '2026-10-01T08:00:00Z'}}


def comment(comment_id, channel_id, name, text, parent_id=None, likes=0, total_replies=0,
            published='2026-10-02T09:30:00Z', updated=None):
    """A comment as youtube.get_comments returns it (top-level comments carry totalReplyCount)."""
    snippet = {'authorDisplayName': name, 'authorProfileImageUrl': f'https://yt3.ggpht.com/{name}.jpg',
               'authorChannelUrl': f'http://www.youtube.com/{name}', 'channelId': 'UC-kanal',
               'textDisplay': text, 'textOriginal': text, 'likeCount': likes,
               'publishedAt': published, 'updatedAt': updated or published}
    if channel_id is not None:
        snippet['authorChannelId'] = {'value': channel_id}
    item = {'kind': 'youtube#comment', 'id': comment_id, 'snippet': snippet}
    if parent_id is None:
        item['totalReplyCount'] = total_replies
    else:
        snippet['parentId'] = parent_id
    return item


class FakeYouTube:
    """What the YouTube client would return: the search result, the video details and the comments per video."""

    def __init__(self):
        self.search = []      # video ids in ranked order
        self.videos = {}      # video id -> videos.list item (a missing id = deleted or private video)
        self.comments = {}    # video id -> comments
        self.calls = []

    def add(self, item, comments=(), in_details=True):
        self.search.append(item['id'])
        if in_details:
            self.videos[item['id']] = item
        self.comments[item['id']] = list(comments)

    def get_vids(self, search_term, max_videos, search_params):
        self.calls.append(('get_vids', search_term, max_videos, search_params))
        # like search.list: the title here is HTML-escaped and no @-mention in it is pseudonymised
        return [{'id': {'videoId': video_id}, 'snippet': {'title': f'Suchtitel {video_id}'}}
                for video_id in self.search[:max_videos]]

    def get_video_details(self, video_ids):
        return {video_id: self.videos[video_id] for video_id in video_ids if video_id in self.videos}

    def get_comments(self, video_id, max_threads, order, replies):
        self.calls.append(('get_comments', video_id, max_threads, order, replies))
        if isinstance(self.comments[video_id], Exception):
            raise self.comments[video_id]
        return self.comments[video_id]


@pytest.fixture
def api(monkeypatch):
    """A FakeYouTube installed in place of the client functions; the code version is fixed."""
    fake = FakeYouTube()
    for name in ['get_vids', 'get_video_details', 'get_comments']:
        monkeypatch.setattr(youtube, name, getattr(fake, name))
    monkeypatch.setattr(collect, 'get_code_version',
                        lambda allow_dirty=False: 'test-version-dirty' if allow_dirty else 'test-version')
    return fake


@pytest.fixture
def one_video(api):
    """One video with a thread of Anna, a reply of Ben and a comment of a deleted user."""
    api.add(video('v1', description='Heute mit @anna.'), [
        comment('c1', 'UC-anna', '@anna', 'Die Reichweite reicht mir.', likes=5, total_replies=1),
        comment('r1', 'UC-ben', '@ben', '@anna im Winter nicht, frag @fremd', parent_id='c1',
                updated='2026-10-03T10:00:00Z'),
        comment('c2', None, '@geloescht', 'Verbrenner bleibt.'),
    ])
    return api


def count(session, model):
    return session.scalar(select(func.count()).select_from(model))


def post(session, external_id):
    return session.scalar(select(Post).where(Post.external_id == external_id)
                          .options(selectinload(Post.snapshots)))


# --- one run ---------------------------------------------------------------------------------

def test_run_stores_topic_search_query_and_the_run_itself(database, one_video):
    with database() as session:
        returned = collect_youtube(session, 'E-Auto', 'Verbrenner-Aus', max_videos=7, max_threads_per_video=50,
                                   comment_order='time', replies='included', search_params={'regionCode': 'AT'})
        run_id = returned.id
    with database() as session:
        topic = session.scalar(select(Topic).options(selectinload(Topic.search_queries)))
        search_query = session.scalar(select(SearchQuery))
        assert topic.name == 'E-Auto' and topic.search_queries == [search_query]
        assert (search_query.source, search_query.search_term) == ('youtube', 'Verbrenner-Aus')
        assert search_query.search_params == {'regionCode': 'AT'}

        collection_run = session.get(CollectionRun, run_id)
        assert collection_run.search_query_id == search_query.id
        assert collection_run.fetch_params == {'max_videos': 7, 'max_threads_per_video': 50,
                                               'comment_order': 'time', 'replies': 'included'}
        assert collection_run.code_version == 'test-version'
        assert collection_run.started_at <= collection_run.finished_at
    assert ('get_vids', 'Verbrenner-Aus', 7, {'regionCode': 'AT'}) in one_video.calls
    assert ('get_comments', 'v1', 50, 'time', 'included') in one_video.calls


def test_defaults_come_from_the_config(database, one_video):
    with database() as session:
        collect_youtube(session, 'E-Auto', 'Verbrenner-Aus')
        assert session.scalar(select(SearchQuery.search_params)) == youtube.SEARCH_PARAMS
        assert session.scalar(select(CollectionRun.fetch_params)) == {
            'max_videos': youtube.MAX_VIDEOS, 'max_threads_per_video': youtube.MAX_THREADS_PER_VIDEO,
            'comment_order': youtube.COMMENT_ORDER, 'replies': youtube.REPLIES}


def test_allow_dirty_reaches_the_code_version(database, one_video):
    with database() as session:
        collect_youtube(session, 'E-Auto', 'Verbrenner-Aus', allow_dirty=True)
        assert session.scalar(select(CollectionRun.code_version)) == 'test-version-dirty'


def test_video_becomes_container_finding_and_root_post(database, one_video):
    with database() as session:
        run_id = collect_youtube(session, 'E-Auto', 'Verbrenner-Aus').id
    with database() as session:
        container = session.scalar(select(Container))
        assert (container.source, container.external_id) == ('youtube', 'v1')
        finding = session.scalar(select(Finding))
        assert (finding.collection_run_id, finding.container_id, finding.rank) == (run_id, container.id, 1)

        root = post(session, 'v1')
        assert root.container_id == container.id and root.parent_id is None
        assert root.text == f'Heute mit @{ANNA}.'  # the description, mentions pseudonymised
        assert root.author_pseudonym == CHANNEL
        assert root.published_at == datetime(2026, 10, 1, 8, 0)  # UTC, stored without time zone
        assert 'channelTitle' not in root.raw['snippet'] and root.raw['snippet']['channelId'] == CHANNEL

        snapshot = root.snapshots[0]
        # "120" and "3" arrive as strings; commentCount is the reply count of the video
        assert (snapshot.collection_run_id, snapshot.likes, snapshot.reply_count) == (run_id, 120, 3)
        assert snapshot.updated_at == root.published_at


def test_container_title_comes_from_the_video_details_not_from_the_search(database, api):
    api.add(video('v1', title='Reichweite & Verbrauch | @kanalname'))
    with database() as session:
        collect_youtube(session, 'E-Auto', 'Verbrenner-Aus')
        title = session.scalar(select(Container.title))
        # unescaped and with the mention pseudonymised, exactly as in the raw JSON of the root post
        assert title == f'Reichweite & Verbrauch | @{pseudonym("youtube", "@kanalname")}'
        assert title == post(session, 'v1').raw['snippet']['title']


def test_container_title_stays_as_first_fetched(database, api):
    api.add(video('v1', title='erster Titel'))
    with database() as session:
        collect_youtube(session, 'E-Auto', 'Verbrenner-Aus')
    api.videos['v1'] = video('v1', title='umbenannt')
    with database() as session:
        collect_youtube(session, 'E-Auto', 'Verbrenner-Aus')
        assert session.scalar(select(Container.title)) == 'erster Titel'


def test_video_with_hidden_likes_and_no_comment_count_gets_nulls(database, api):
    api.add(video('v1', likes=None, comment_count=None))
    with database() as session:
        collect_youtube(session, 'E-Auto', 'Verbrenner-Aus')
        snapshot = post(session, 'v1').snapshots[0]
        assert snapshot.likes is None and snapshot.reply_count is None


def test_comments_form_a_tree_below_the_video(database, one_video):
    with database() as session:
        collect_youtube(session, 'E-Auto', 'Verbrenner-Aus')
    with database() as session:
        root, top, reply, other = (post(session, external_id) for external_id in ['v1', 'c1', 'r1', 'c2'])
        assert top.parent_id == root.id and other.parent_id == root.id
        assert reply.parent_id == top.id  # replies hang below the top-level comment
        assert {top.container_id, reply.container_id, other.container_id} == {root.container_id}

        assert top.text == 'Die Reichweite reicht mir.'
        assert top.published_at == datetime(2026, 10, 2, 9, 30)
        assert (top.snapshots[0].likes, top.snapshots[0].reply_count) == (5, 1)  # totalReplyCount
        assert (reply.snapshots[0].likes, reply.snapshots[0].reply_count) == (0, 0)
        assert reply.snapshots[0].updated_at == datetime(2026, 10, 3, 10, 0)  # edited later
        assert top.snapshots[0].updated_at == top.published_at


def test_authors_are_pseudonyms_and_a_deleted_user_has_none(database, one_video):
    with database() as session:
        collect_youtube(session, 'E-Auto', 'Verbrenner-Aus')
    with database() as session:
        assert post(session, 'c1').author_pseudonym == ANNA
        assert post(session, 'r1').author_pseudonym == BEN
        assert post(session, 'c2').author_pseudonym is None  # not one shared pseudonym for all deleted users
        assert set(session.scalars(select(Author.pseudonym))) == {ANNA, BEN, CHANNEL}


def test_mentions_get_the_pseudonym_of_the_author_they_name(database, one_video):
    with database() as session:
        collect_youtube(session, 'E-Auto', 'Verbrenner-Aus')
        reply = post(session, 'r1')
        # @anna wrote in the same video: her author pseudonym. @fremd did not: hashed from the handle.
        assert reply.text == f'@{ANNA} im Winter nicht, frag @{pseudonym("youtube", "@fremd")}'
        assert reply.raw['snippet']['textOriginal'] == reply.text


def test_nothing_that_names_a_person_reaches_the_posts(database, one_video):
    with database() as session:
        collect_youtube(session, 'E-Auto', 'Verbrenner-Aus')
    with database() as session:
        stored = ' '.join(json.dumps([row.text, row.raw, row.author_pseudonym], ensure_ascii=False)
                          for row in session.scalars(select(Post)))
        stored += ' '.join(session.scalars(select(Container.title)))
        for plain in ['anna', 'UC-anna', 'UC-ben', 'UC-kanal', 'Kanalname', '@ben', 'fremd', 'geloescht',
                      'yt3.ggpht.com', 'Suchtitel']:
            assert plain not in stored


def test_videos_keep_their_rank_and_a_video_without_details_is_skipped(database, api):
    api.add(video('v1'))
    api.add(video('v2'), in_details=False)  # deleted or private since the search
    api.add(video('v3'))
    with database() as session:
        collect_youtube(session, 'E-Auto', 'Verbrenner-Aus')
        ranks = dict(session.execute(select(Container.external_id, Finding.rank)
                                     .join(Finding, Finding.container_id == Container.id)).all())
        assert ranks == {'v1': 1, 'v3': 3}
        assert count(session, Post) == 2


def test_video_without_comments_is_stored_with_its_root_post_only(database, api):
    api.add(video('v1', comment_count='0'))
    with database() as session:
        collect_youtube(session, 'E-Auto', 'Verbrenner-Aus')
        assert count(session, Post) == 1 and count(session, Snapshot) == 1


# --- repeated runs: append, never overwrite --------------------------------------------------

def test_second_run_adds_findings_and_snapshots_but_no_duplicates(database, one_video):
    with database() as session:
        first = collect_youtube(session, 'E-Auto', 'Verbrenner-Aus').id
    with database() as session:
        second = collect_youtube(session, 'E-Auto', 'Verbrenner-Aus').id
    with database() as session:
        assert first != second and count(session, CollectionRun) == 2
        for model, rows in [(Topic, 1), (SearchQuery, 1), (Container, 1), (Post, 4), (Author, 3),
                            (Finding, 2), (Snapshot, 8)]:
            assert count(session, model) == rows, model.__name__
        assert [snapshot.collection_run_id for snapshot in
                sorted(post(session, 'c1').snapshots, key=lambda snapshot: snapshot.collection_run_id)] == [
            first, second]


def test_second_run_keeps_the_first_text_and_records_the_new_numbers(database, api):
    api.add(video('v1'), [comment('c1', 'UC-anna', '@anna', 'erste Fassung', likes=1)])
    with database() as session:
        first = collect_youtube(session, 'E-Auto', 'Verbrenner-Aus').id
    api.comments['v1'] = [comment('c1', 'UC-anna', '@anna', 'später bearbeitet', likes=9, total_replies=2,
                                  updated='2026-11-05T12:00:00Z')]
    with database() as session:
        second = collect_youtube(session, 'E-Auto', 'Verbrenner-Aus').id
    with database() as session:
        stored = post(session, 'c1')
        assert stored.text == 'erste Fassung'  # post holds the first fetch only
        assert stored.raw['snippet']['textOriginal'] == 'erste Fassung'
        by_run = {snapshot.collection_run_id: snapshot for snapshot in stored.snapshots}
        assert (by_run[first].likes, by_run[first].reply_count) == (1, 0)
        assert (by_run[second].likes, by_run[second].reply_count) == (9, 2)
        assert by_run[second].updated_at == datetime(2026, 11, 5, 12, 0)  # the edit is visible here


def test_new_comments_of_a_later_run_are_added(database, api):
    api.add(video('v1'), [comment('c1', 'UC-anna', '@anna', 'erster', total_replies=0)])
    with database() as session:
        collect_youtube(session, 'E-Auto', 'Verbrenner-Aus')
    api.comments['v1'].append(comment('r1', 'UC-ben', '@ben', 'Antwort', parent_id='c1'))
    with database() as session:
        collect_youtube(session, 'E-Auto', 'Verbrenner-Aus')
        assert post(session, 'r1').parent_id == post(session, 'c1').id
        assert count(session, Post) == 3


def test_second_search_term_of_a_topic_shares_the_video(database, one_video):
    with database() as session:
        collect_youtube(session, 'E-Auto', 'Verbrenner-Aus')
    with database() as session:
        collect_youtube(session, 'E-Auto', 'Elektroautos im Test')
    with database() as session:
        topic = session.scalar(select(Topic).options(selectinload(Topic.search_queries)))
        assert sorted(query.search_term for query in topic.search_queries) == [
            'Elektroautos im Test', 'Verbrenner-Aus']
        assert count(session, Topic) == 1 and count(session, Container) == 1 and count(session, Finding) == 2


def test_same_search_term_with_other_params_is_another_search_query(database, one_video):
    with database() as session:
        collect_youtube(session, 'E-Auto', 'Verbrenner-Aus', search_params={'regionCode': 'DE'})
    with database() as session:
        collect_youtube(session, 'E-Auto', 'Verbrenner-Aus', search_params={'regionCode': 'AT'})
        assert count(session, SearchQuery) == 2


def test_a_search_query_can_belong_to_two_topics(database, one_video):
    with database() as session:
        collect_youtube(session, 'E-Auto', 'Verbrenner-Aus')
    with database() as session:
        collect_youtube(session, 'Verkehrswende', 'Verbrenner-Aus')
        assert count(session, Topic) == 2 and count(session, SearchQuery) == 1


# --- an aborted run --------------------------------------------------------------------------

def test_aborted_run_keeps_the_finished_videos_and_stays_unfinished(database, api):
    api.add(video('v1'), [comment('c1', 'UC-anna', '@anna', 'gespeichert')])
    api.add(video('v2'))
    api.comments['v2'] = RuntimeError('quota exceeded')
    with database() as session:
        with pytest.raises(RuntimeError, match='quota exceeded'):
            collect_youtube(session, 'E-Auto', 'Verbrenner-Aus')
    with database() as session:
        collection_run = session.scalar(select(CollectionRun))
        assert collection_run.finished_at is None  # the mark of an aborted run
        assert post(session, 'c1').text == 'gespeichert'  # committed per video
        assert post(session, 'v2') is None
        assert count(session, Container) == 1 and count(session, Finding) == 1
