# annotation.py — Block 2

from radar.db.base import Base

from sqlalchemy import JSON, Column, ForeignKey, ForeignKeyConstraint, Index, Table, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from datetime import datetime


sample_post = Table(
    'sample_post',
    Base.metadata,
    Column('sample_id', ForeignKey('sample.id'), primary_key = True),
    Column('post_id', ForeignKey('post.id'), primary_key = True, index = True),
)


class Sample(Base):
    __tablename__ = 'sample'

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(unique=True)
    topic_id: Mapped[int] = mapped_column(ForeignKey('topic.id'), index=True)
    base_sample_id: Mapped[int | None] = mapped_column(ForeignKey('sample.id'), index=True)
    definition: Mapped[dict] = mapped_column(JSON)
    code_version: Mapped[str] = mapped_column()
    created_at: Mapped[datetime] = mapped_column()

    topic: Mapped['Topic'] = relationship(
        back_populates = 'samples', lazy = 'raise'
    )
    base: Mapped['Sample | None'] = relationship(
        back_populates = 'subsets', remote_side = id, lazy = 'raise'
    )
    subsets: Mapped[list['Sample']] = relationship(
        back_populates = 'base', lazy = 'raise'
    )
    annotation_runs: Mapped[list['AnnotationRun']] = relationship(
        back_populates = 'sample', lazy = 'raise'
    )
    posts: Mapped[list['Post']] = relationship(
        secondary = sample_post, back_populates = 'samples', lazy = 'raise'
    )


class PromptVersion(Base):
    __tablename__ = 'prompt_version'

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(unique=True)
    text: Mapped[str] = mapped_column()
    hash: Mapped[str] = mapped_column(unique=True)

    annotation_runs: Mapped[list['AnnotationRun']] = relationship(
        back_populates = 'prompt_version', lazy = 'raise'
    )


class AnnotationRun(Base):
    __tablename__ = 'annotation_run'

    id: Mapped[int] = mapped_column(primary_key= True)
    sample_id: Mapped[int] = mapped_column(ForeignKey('sample.id'), index=True)
    prompt_version_id: Mapped[int] = mapped_column(ForeignKey('prompt_version.id'), index=True)
    code_version: Mapped[str] = mapped_column()
    started_at: Mapped[datetime] = mapped_column()

    sample: Mapped['Sample'] = relationship(
        back_populates = 'annotation_runs', lazy = 'raise'
    )
    prompt_version: Mapped['PromptVersion'] = relationship(
        back_populates = 'annotation_runs', lazy = 'raise'
    )
    # 1:1 subtypes: exactly one of the two exists for each run
    llm_run: Mapped['LlmRun | None'] = relationship(
        back_populates = 'annotation_run', lazy = 'raise'
    )
    human_run: Mapped['HumanRun | None'] = relationship(
        back_populates = 'annotation_run', lazy = 'raise'
    )
    annotations: Mapped[list['Annotation']] = relationship(
        back_populates = 'annotation_run', lazy = 'raise'
    )
    mapping_runs: Mapped[list['MappingRun']] = relationship(
        secondary = 'mapping_input', back_populates = 'annotation_runs', lazy = 'raise'
    )


class LlmRun(Base):
    __tablename__ = 'llm_run'

    # primary key and foreign key at once: one llm_run per annotation_run
    run_id: Mapped[int] = mapped_column(ForeignKey('annotation_run.id'), primary_key=True)
    model: Mapped[str] = mapped_column()
    temperature: Mapped[float] = mapped_column()
    batch_size: Mapped[int] = mapped_column()
    context_strategy: Mapped[str] = mapped_column()
    params: Mapped[dict] = mapped_column(JSON)

    annotation_run: Mapped['AnnotationRun'] = relationship(
        back_populates = 'llm_run', lazy = 'raise'
    )
    batches: Mapped[list['Batch']] = relationship(
        back_populates = 'llm_run', lazy = 'raise'
    )


class HumanRun(Base):
    __tablename__ = 'human_run'

    run_id: Mapped[int] = mapped_column(ForeignKey('annotation_run.id'), primary_key=True)
    annotator: Mapped[str] = mapped_column()

    annotation_run: Mapped['AnnotationRun'] = relationship(
        back_populates = 'human_run', lazy = 'raise'
    )


class Batch(Base):
    __tablename__ = 'batch'
    __table_args__ = (UniqueConstraint('run_id', 'number'),)

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey('llm_run.run_id'), index=True)
    number: Mapped[int] = mapped_column()
    status: Mapped[str] = mapped_column()
    raw_response: Mapped[dict | None] = mapped_column(JSON(none_as_null=True))  # SQL NULL if the request failed

    llm_run: Mapped['LlmRun'] = relationship(
        back_populates = 'batches', lazy = 'raise'
    )
    batch_posts: Mapped[list['BatchPost']] = relationship(
        back_populates = 'batch', lazy = 'raise'
    )


class BatchPost(Base):
    __tablename__ = 'batch_post'

    batch_id: Mapped[int] = mapped_column(ForeignKey('batch.id'), primary_key=True)
    post_id: Mapped[int] = mapped_column(ForeignKey('post.id'), primary_key=True, index=True)
    position: Mapped[int] = mapped_column()

    batch: Mapped['Batch'] = relationship(
        back_populates = 'batch_posts', lazy = 'raise'
    )
    post: Mapped['Post'] = relationship(
        back_populates = 'batch_posts', lazy = 'raise'
    )


class Annotation(Base):
    __tablename__ = 'annotation'

    run_id: Mapped[int] = mapped_column(ForeignKey('annotation_run.id'), primary_key=True)
    post_id: Mapped[int] = mapped_column(ForeignKey('post.id'), primary_key=True, index=True)
    stance: Mapped[str | None] = mapped_column()  # None if the prompt version does not ask for a stance

    annotation_run: Mapped['AnnotationRun'] = relationship(
        back_populates = 'annotations', lazy = 'raise'
    )
    post: Mapped['Post'] = relationship(
        back_populates = 'annotations', lazy = 'raise'
    )
    criterion_mentions: Mapped[list['CriterionMention']] = relationship(
        back_populates = 'annotation', lazy = 'raise'
    )


class CriterionMention(Base):
    __tablename__ = 'criterion_mention'
    # composite foreign key: (run_id, post_id) together identify one annotation
    __table_args__ = (
        ForeignKeyConstraint(['run_id', 'post_id'], ['annotation.run_id', 'annotation.post_id']),
        Index('ix_criterion_mention_run_id_post_id', 'run_id', 'post_id'),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column()
    post_id: Mapped[int] = mapped_column()
    label: Mapped[str] = mapped_column()
    sentiment: Mapped[str | None] = mapped_column()
    quote: Mapped[str | None] = mapped_column()

    annotation: Mapped['Annotation'] = relationship(
        back_populates = 'criterion_mentions', lazy = 'raise'
    )
    assignments: Mapped[list['Assignment']] = relationship(
        back_populates = 'mention', lazy = 'raise'
    )
