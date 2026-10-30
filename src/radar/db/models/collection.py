# collection.py — Block 1

from radar.db.base import Base
from sqlalchemy import String, JSON, UniqueConstraint, Column, ForeignKey, Table
from sqlalchemy.orm import Mapped, mapped_column, relationship
from datetime import datetime

topic_search_query = Table(
    'topic_search_query',
    Base.metadata,
    Column('topic_id', ForeignKey('topic.id'), primary_key=True),
    Column('search_query_id', ForeignKey('search_query.id'), primary_key=True, index=True),
)


class Topic(Base):
    __tablename__ = "topic"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True)

    search_queries: Mapped[list['SearchQuery']] = relationship(
        secondary = topic_search_query, back_populates = 'topics', lazy = 'raise'
    )


class SearchQuery(Base):
    __tablename__ = "search_query"
    __table_args__ = (UniqueConstraint("source", "search_term", "search_params"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    source: Mapped[str] = mapped_column(String)
    search_term: Mapped[str] = mapped_column(String)
    search_params: Mapped[dict] = mapped_column(JSON)

    topics: Mapped[list['Topic']] = relationship(
        secondary = topic_search_query, back_populates = 'search_queries', lazy = 'raise'
    )
    collection_runs: Mapped[list['CollectionRun']] = relationship(
        back_populates = 'search_query', lazy = 'raise'
    )


class CollectionRun(Base):
    __tablename__ = 'collection_run'

    id: Mapped[int] = mapped_column(primary_key=True)
    search_query_id: Mapped[int] = mapped_column(ForeignKey('search_query.id'), index=True)
    started_at: Mapped[datetime] = mapped_column()
    finished_at: Mapped[datetime | None] = mapped_column()
    fetch_params: Mapped[dict] = mapped_column(JSON)
    code_version: Mapped[str] = mapped_column()

    search_query: Mapped['SearchQuery'] = relationship(
        back_populates = 'collection_runs', lazy = 'raise'
    )
    findings: Mapped[list['Finding']] = relationship(
        back_populates = 'collection_run', lazy = 'raise'
    )
    snapshots: Mapped[list['Snapshot']] = relationship(
        back_populates = 'collection_run', lazy = 'raise'
    )



class Container(Base):
    __tablename__ = 'container'
    __table_args__ = (UniqueConstraint('source', 'external_id'),)

    id: Mapped[int] = mapped_column(primary_key=True)
    source: Mapped[str] = mapped_column(String)
    external_id: Mapped[str] = mapped_column(String)
    title: Mapped[str] = mapped_column(String)

    findings: Mapped[list['Finding']] = relationship(
        back_populates = 'container', lazy = 'raise'
    )
    posts: Mapped[list['Post']] = relationship(
        back_populates = 'container', lazy = 'raise'
    )


class Finding(Base):
    __tablename__ = 'finding'

    collection_run_id: Mapped[int] = mapped_column(ForeignKey('collection_run.id'), primary_key=True)
    container_id: Mapped[int] = mapped_column(ForeignKey('container.id'), primary_key=True, index=True)
    rank: Mapped[int] = mapped_column()

    collection_run: Mapped['CollectionRun'] = relationship(
        back_populates = 'findings', lazy = 'raise'
    )
    container: Mapped['Container'] = relationship(
        back_populates = 'findings', lazy = 'raise'
    )


# [?] could be an attribute but will remain a class
class Author(Base):
    __tablename__ = 'author'

    pseudonym: Mapped[str] = mapped_column(String, primary_key=True)

    posts: Mapped[list['Post']] = relationship(
        back_populates = 'author', lazy = 'raise'
    )


class Post(Base):
    __tablename__ = 'post'
    __table_args__ = (UniqueConstraint('container_id', 'external_id'),)

    id: Mapped[int] = mapped_column(primary_key=True)
    container_id: Mapped[int] = mapped_column(ForeignKey('container.id'), index=True)
    external_id: Mapped[str] = mapped_column()
    parent_id: Mapped[int | None] = mapped_column(ForeignKey('post.id'), index=True) # None for the root post (video or thread)
    author_pseudonym: Mapped[str | None] = mapped_column(ForeignKey('author.pseudonym'), index=True) # None for unknown or deleted users
    text: Mapped[str] = mapped_column()
    published_at: Mapped[datetime] = mapped_column()
    raw: Mapped[dict] = mapped_column(JSON)

    container: Mapped['Container'] = relationship(
        back_populates = 'posts', lazy = 'raise'
    )
    author: Mapped['Author | None'] = relationship(
        back_populates = 'posts', lazy = 'raise'
    )
    snapshots: Mapped[list['Snapshot']] = relationship(
        back_populates = 'post', lazy = 'raise'
    )
    parent: Mapped['Post | None'] = relationship(
        back_populates = 'children', remote_side = id, lazy = 'raise'
    )
    children: Mapped[list['Post']] = relationship(
        back_populates = 'parent', lazy = 'raise'
    )

class Snapshot(Base):
    __tablename__ = 'snapshot'

    post_id: Mapped[int] = mapped_column(ForeignKey('post.id'), primary_key=True)
    collection_run_id: Mapped[int] = mapped_column(ForeignKey('collection_run.id'), primary_key=True, index=True)
    likes: Mapped[int] = mapped_column()
    reply_count: Mapped[int] = mapped_column()
    updated_at: Mapped[datetime | None] = mapped_column()

    post: Mapped['Post'] = relationship(
        back_populates = 'snapshots', lazy = 'raise'
    )
    collection_run: Mapped['CollectionRun'] = relationship(
        back_populates= 'snapshots', lazy = 'raise'
    )
