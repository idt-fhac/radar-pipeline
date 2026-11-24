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
from radar.mapping import embed

import logging

logger = logging.getLogger(__name__)

EMBED_BATCH_SIZE = 256  # inputs per call to the model; the new embeddings are committed after each batch
VECTOR_DTYPE = np.float32  # dtype of Embedding.vector (bytes)


def map_criteria(annotation_run_ids, method, threshold = None, params = None, allow_dirty = False,
                 embedding_provider = embed.DEFAULT_PROVIDER):
    """Map the criterion mentions of the given annotation runs onto shared criteria.

    Embeds every distinct rendered text once with the embedding provider from the config
    (cached in `embedding`, per provider and model), clusters
    the vectors and stores one mapping_run with its criteria and one assignment per mention.
    Returns the id of the new mapping_run.
    """
    params = {} if params is None else params

    with Session() as session:
        # load and check the input
        annotation_runs = get_annotation_runs(session, annotation_run_ids)
        topic = annotation_runs[0].sample.topic
        mentions = get_mentions(session, annotation_runs)
        if not mentions:
            raise ValueError(f'annotation runs {sorted(annotation_run_ids)} contain no criterion mentions')

        # embeddings: one per distinct rendered text, cached by input hash
        embedding_config = get_or_create_embedding_config(session, embed.model_id(embedding_provider), embed.TEMPLATE)
        mentions_by_hash = defaultdict(list)
        input_by_hash = {}
        label_by_hash = {}
        for mention in mentions:
            input = render(embedding_config.template, topic, mention)
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

        # clustering: one point per distinct text
        hashes = list(mentions_by_hash)
        vectors = np.stack([vectors_by_hash[input_hash] for input_hash in hashes])
        labels = cluster(vectors, method, threshold, params)
        points_by_cluster = defaultdict(list)
        for point, label in enumerate(labels):
            points_by_cluster[label].append(point)

        # store the run: mapping_run, mapping_input, criterion and assignment in one transaction
        mapping_run = create_mapping_run(session, method, params, annotation_runs, embedding_config = embedding_config, threshold = threshold, allow_dirty = allow_dirty)
        for points in points_by_cluster.values():
            mean = vectors[points].mean(axis = 0)
            closest = points[int(np.argmin(np.linalg.norm(vectors[points] - mean, axis = 1)))]
            criterion = create_criterion(session, mapping_run, name = label_by_hash[hashes[closest]])
            for point in points:
                for mention in mentions_by_hash[hashes[point]]:
                    create_assignment(session, mapping_run, mention, criterion)
        session.commit()
        logger.info('mapping run %d: %d criteria for %d mentions', mapping_run.id, len(points_by_cluster), len(mentions))
        return mapping_run.id


def cluster(vectors, method, threshold = None, params = None):
    """Cluster id per point (row of vectors). Outliers become a cluster of their own.

    method 'agglomerative': threshold is the cosine distance at which clusters stop merging,
        params are passed to sklearn's AgglomerativeClustering (e.g. {'linkage': 'average'}).
    method 'hdbscan': threshold is cluster_selection_epsilon (optional), params are passed
        to hdbscan.HDBSCAN (e.g. {'min_cluster_size': 3}); its outliers (-1) are split up.
    """
    params = {} if params is None else params
    if len(vectors) < 2:
        return [0] * len(vectors)

    if method == 'agglomerative':
        from sklearn.cluster import AgglomerativeClustering
        if threshold is None:
            raise ValueError("method 'agglomerative' needs a threshold")
        options = {'metric': 'cosine', 'linkage': 'average', **params}
        labels = AgglomerativeClustering(n_clusters = None, distance_threshold = threshold, **options).fit_predict(vectors)
    elif method == 'hdbscan':
        import hdbscan
        options = dict(params)
        if threshold is not None:
            options['cluster_selection_epsilon'] = threshold
        labels = hdbscan.HDBSCAN(**options).fit_predict(vectors)
    else:
        raise ValueError(f'unknown clustering method: {method!r}')

    # every outlier (-1) becomes a cluster of its own
    labels = [int(label) for label in labels]
    next_label = max(labels) + 1
    for point, label in enumerate(labels):
        if label == -1:
            labels[point] = next_label
            next_label += 1
    return labels


def render(template, topic, mention):
    """Text that is sent to the embedding model for one mention."""
    return template.format(topic = topic.name, label = mention.label)

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

def create_mapping_run(session, method, params, annotation_runs, embedding_config = None, threshold = None, allow_dirty = False):
    # annotation_runs fill mapping_input; embedding_config and threshold stay None for methods without them
    mapping_run = MappingRun(embedding_config = embedding_config, method = method, threshold = threshold, params = params, annotation_runs = list(annotation_runs), code_version = get_code_version(allow_dirty=allow_dirty), created_at = datetime.now(timezone.utc))
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
