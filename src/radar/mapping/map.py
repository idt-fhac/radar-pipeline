# map.py — maps the criterion mentions of annotation runs onto shared criteria
import hashlib
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from radar.db.engine import Session
from radar.db.models.annotation import AnnotationRun, CriterionMention, Sample
from radar.db.models.mapping import Assignment, Criterion, Embedding, EmbeddingConfig, MappingRun
from radar.provenance import get_code_version
from radar.config import CONFIG
from radar.mapping import embed
from radar.mapping.cluster import cluster, name_cluster, reduce

import logging

logger = logging.getLogger(__name__)

EMBED_BATCH_SIZE = 256  # inputs per call to the model; the new embeddings are committed after each batch
VECTOR_DTYPE = np.float32  # dtype of Embedding.vector (bytes)


def map_criteria(annotation_run_ids, method = None, threshold = None, params = None, allow_dirty = False,
                 embedding_provider = None, template = None, reduction = None, naming = None):
    """Map the criterion mentions of the given annotation runs onto shared criteria.

    Embeds every distinct rendered text once with the embedding provider (cached in `embedding`,
    per provider, model and template), projects the vectors with UMAP, clusters them and stores
    one mapping_run with its criteria and one assignment per mention. Returns its id.

    Every argument left out comes from [mapping] / [embedding] in the config:
    method 'hdbscan' or 'agglomerative'; threshold (agglomerative only); params = options of
    the clustering method; reduction = UMAP options; naming 'centroid' or 'frequent';
    template = name of an [embedding.templates] entry. Everything used is stored in
    mapping_run.params, so a run can be reproduced from the database alone.
    """
    defaults = CONFIG.get('mapping', {})
    method = method or defaults.get('method', 'hdbscan')
    options = dict(defaults.get(method, {}))
    if method == 'agglomerative':
        threshold = options.pop('threshold', None) if threshold is None else threshold
        options.pop('threshold', None)
    options.update(params or {})
    reduction = {**defaults.get('reduction', {}), **(reduction or {})}
    naming = naming or defaults.get('naming', 'centroid')
    embedding_provider = embedding_provider or embed.DEFAULT_PROVIDER
    template = template or embed.DEFAULT_TEMPLATE
    pattern = embed.template(template)
    code_version = get_code_version(allow_dirty = allow_dirty)  # before the expensive part: fail fast

    with Session() as session:
        # load and check the input
        annotation_runs = get_annotation_runs(session, annotation_run_ids)
        topic_name = annotation_runs[0].sample.topic.name
        mentions = get_mentions(session, annotation_runs)
        if not mentions:
            raise ValueError(f'annotation runs {sorted(annotation_run_ids)} contain no criterion mentions')

        # embeddings: one per distinct rendered text, cached by input hash
        embedding_config = get_or_create_embedding_config(session, embed.model_id(embedding_provider), pattern)
        mentions_by_hash = defaultdict(list)
        input_by_hash = {}
        label_by_hash = {}
        for mention in mentions:
            input = embed.render(pattern, topic_name, mention.label)
            input_hash = hash_input(input)
            mentions_by_hash[input_hash].append(mention)
            input_by_hash[input_hash] = input
            label_by_hash[input_hash] = mention.label
        # vectors are kept as numpy arrays: the commits below expire the ORM objects
        vectors_by_hash = {input_hash: np.frombuffer(embedding.vector, dtype = VECTOR_DTYPE) for input_hash, embedding in get_embeddings(session, embedding_config, list(mentions_by_hash)).items()}
        missing = [input_hash for input_hash in mentions_by_hash if input_hash not in vectors_by_hash]
        logger.info('%d mentions, %d distinct inputs, %d cached, %d to embed', len(mentions), len(mentions_by_hash), len(vectors_by_hash), len(missing))
        for start in range(0, len(missing), EMBED_BATCH_SIZE):
            batch = missing[start:start + EMBED_BATCH_SIZE]
            # the rendered texts are embedded exactly as hashed, so cache key and model input match
            vectors = embed.embed([input_by_hash[input_hash] for input_hash in batch], embedding_provider).astype(VECTOR_DTYPE)
            for input_hash, vector in zip(batch, vectors):
                create_embedding(session, embedding_config, input_by_hash[input_hash], vector.tobytes())
                vectors_by_hash[input_hash] = vector
            session.commit()

        # reduction and clustering: one point per distinct text, in a fixed order (reproducible)
        hashes = sorted(mentions_by_hash)
        vectors = np.stack([vectors_by_hash[input_hash] for input_hash in hashes])
        labels = cluster(reduce(vectors, **reduction), method, threshold, options)
        points_by_cluster = defaultdict(list)
        for point, label in enumerate(labels):
            points_by_cluster[label].append(point)
        counts = [len(mentions_by_hash[input_hash]) for input_hash in hashes]
        texts = [label_by_hash[input_hash] for input_hash in hashes]

        # store the run: mapping_run, mapping_input, criterion and assignment in one transaction
        run_params = {'template': template, 'reduction': reduction, 'clustering': options, 'naming': naming}
        mapping_run = create_mapping_run(session, method, run_params, annotation_runs, embedding_config = embedding_config, threshold = threshold, code_version = code_version)
        for points in points_by_cluster.values():
            criterion = create_criterion(session, mapping_run, name = name_cluster(points, vectors, counts, texts, naming))
            for point in points:
                for mention in mentions_by_hash[hashes[point]]:
                    create_assignment(session, mapping_run, mention, criterion)
        session.commit()
        logger.info('mapping run %d: %d criteria for %d mentions', mapping_run.id, len(points_by_cluster), len(mentions))
        return mapping_run.id


def hash_input(input):
    """Cache key of an embedding: SHA-256 of the rendered text sent to the model."""
    return hashlib.sha256(input.encode('utf-8')).hexdigest()

def get_annotation_runs(session, annotation_run_ids):
    """The annotation runs (with sample.topic); raises if an id is missing or the topics differ."""
    annotation_run_ids = set(annotation_run_ids)
    annotation_runs = list(session.scalars(select(AnnotationRun).where(AnnotationRun.id.in_(annotation_run_ids)).options(selectinload(AnnotationRun.sample).selectinload(Sample.topic)).order_by(AnnotationRun.id)))
    missing = annotation_run_ids - {annotation_run.id for annotation_run in annotation_runs}
    if not annotation_runs or missing:
        raise ValueError(f'annotation runs not found: {sorted(missing) or "none given"}')
    topics = {annotation_run.sample.topic.name for annotation_run in annotation_runs}
    if len(topics) > 1:
        raise ValueError(f'annotation runs belong to different topics: {sorted(topics)}')
    return annotation_runs

def get_or_create_embedding_config(session, model, template):
    embedding_config = session.scalar(select(EmbeddingConfig).where(EmbeddingConfig.model == model, EmbeddingConfig.template == template))
    if embedding_config is None:
        embedding_config = EmbeddingConfig(model = model, template = template)
        session.add(embedding_config)
    return embedding_config

# no get_or_create: the vector would have to be computed before knowing whether it is cached
def get_embedding(session, embedding_config, input):
    return session.scalar(select(Embedding).where(Embedding.config == embedding_config, Embedding.input_hash == hash_input(input)))

def get_embeddings(session, embedding_config, input_hashes):
    """Cached embeddings of this config for the given input hashes, as {input_hash: embedding}."""
    embeddings = session.scalars(select(Embedding).where(Embedding.config == embedding_config, Embedding.input_hash.in_(input_hashes)))
    return {embedding.input_hash: embedding for embedding in embeddings}

def create_embedding(session, embedding_config, input, vector):
    # vector: bytes, e.g. numpy_array.tobytes()
    embedding = Embedding(config = embedding_config, input_hash = hash_input(input), input = input, vector = vector)
    session.add(embedding)
    return embedding

def create_mapping_run(session, method, params, annotation_runs, embedding_config = None, threshold = None, code_version = None):
    # annotation_runs fill mapping_input; embedding_config and threshold stay None for methods without them
    mapping_run = MappingRun(embedding_config = embedding_config, method = method, threshold = threshold, params = params, annotation_runs = list(annotation_runs), code_version = code_version, created_at = datetime.now(timezone.utc))
    session.add(mapping_run)
    return mapping_run

def get_mentions(session, annotation_runs):
    """All criterion mentions of the given annotation runs (the input of a mapping run)."""
    run_ids = [annotation_run.id for annotation_run in annotation_runs]
    return list(session.scalars(select(CriterionMention).where(CriterionMention.run_id.in_(run_ids)).order_by(CriterionMention.id)))

def create_criterion(session, mapping_run, name):
    criterion = Criterion(mapping_run = mapping_run, name = name)
    session.add(criterion)
    return criterion

def create_assignment(session, mapping_run, mention, criterion):
    assignment = Assignment(mapping_run = mapping_run, mention = mention, criterion = criterion)
    session.add(assignment)
    return assignment
