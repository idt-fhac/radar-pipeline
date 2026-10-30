# mapping.py — Block 3

from radar.db.base import Base

from sqlalchemy import JSON, Column, ForeignKey, LargeBinary, Table, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from datetime import datetime


mapping_input = Table(
    'mapping_input',
    Base.metadata,
    Column('mapping_run_id', ForeignKey('mapping_run.id'), primary_key = True),
    Column('annotation_run_id', ForeignKey('annotation_run.id'), primary_key = True, index = True),
)


class EmbeddingConfig(Base):
    __tablename__ = 'embedding_config'
    __table_args__ = (UniqueConstraint('model', 'template'),)

    id: Mapped[int] = mapped_column(primary_key=True)
    model: Mapped[str] = mapped_column()
    template: Mapped[str] = mapped_column()  # e.g. 'Topic: {topic}. Criterion: {label}'

    embeddings: Mapped[list['Embedding']] = relationship(
        back_populates = 'config', lazy = 'raise'
    )
    mapping_runs: Mapped[list['MappingRun']] = relationship(
        back_populates = 'embedding_config', lazy = 'raise'
    )


class Embedding(Base):
    __tablename__ = 'embedding'
    __table_args__ = (UniqueConstraint('config_id', 'input_hash'),)

    id: Mapped[int] = mapped_column(primary_key=True)
    config_id: Mapped[int] = mapped_column(ForeignKey('embedding_config.id'), index=True)
    input_hash: Mapped[str] = mapped_column()
    input: Mapped[str] = mapped_column()  # rendered text that was sent to the model
    vector: Mapped[bytes] = mapped_column(LargeBinary)  # numpy array via tobytes()/frombuffer()

    config: Mapped['EmbeddingConfig'] = relationship(
        back_populates = 'embeddings', lazy = 'raise'
    )


class MappingRun(Base):
    __tablename__ = 'mapping_run'

    id: Mapped[int] = mapped_column(primary_key=True)
    embedding_config_id: Mapped[int | None] = mapped_column(ForeignKey('embedding_config.id'), index=True)  # None for methods without embeddings (llm, manual)
    method: Mapped[str] = mapped_column()
    threshold: Mapped[float | None] = mapped_column()  # None for methods without a threshold
    params: Mapped[dict] = mapped_column(JSON)
    code_version: Mapped[str] = mapped_column()
    created_at: Mapped[datetime] = mapped_column()

    embedding_config: Mapped['EmbeddingConfig | None'] = relationship(
        back_populates = 'mapping_runs', lazy = 'raise'
    )
    annotation_runs: Mapped[list['AnnotationRun']] = relationship(
        secondary = mapping_input, back_populates = 'mapping_runs', lazy = 'raise'
    )
    criteria: Mapped[list['Criterion']] = relationship(
        back_populates = 'mapping_run', lazy = 'raise'
    )
    assignments: Mapped[list['Assignment']] = relationship(
        back_populates = 'mapping_run', lazy = 'raise'
    )


class Criterion(Base):
    __tablename__ = 'criterion'

    id: Mapped[int] = mapped_column(primary_key=True)
    mapping_run_id: Mapped[int] = mapped_column(ForeignKey('mapping_run.id'), index=True)
    name: Mapped[str] = mapped_column()

    mapping_run: Mapped['MappingRun'] = relationship(
        back_populates = 'criteria', lazy = 'raise'
    )
    assignments: Mapped[list['Assignment']] = relationship(
        back_populates = 'criterion', lazy = 'raise'
    )


class Assignment(Base):
    __tablename__ = 'assignment'

    mapping_run_id: Mapped[int] = mapped_column(ForeignKey('mapping_run.id'), primary_key=True)
    mention_id: Mapped[int] = mapped_column(ForeignKey('criterion_mention.id'), primary_key=True, index=True)
    criterion_id: Mapped[int] = mapped_column(ForeignKey('criterion.id'), index=True)

    mapping_run: Mapped['MappingRun'] = relationship(
        back_populates = 'assignments', lazy = 'raise'
    )
    mention: Mapped['CriterionMention'] = relationship(
        back_populates = 'assignments', lazy = 'raise'
    )
    criterion: Mapped['Criterion'] = relationship(
        back_populates = 'assignments', lazy = 'raise'
    )
