# map.py — maps the criterion mentions of annotation runs onto shared criteria
import hashlib
from datetime import datetime, timezone

from sqlalchemy import select

from radar.db.models.annotation import CriterionMention
from radar.db.models.mapping import Assignment, Criterion, Embedding, EmbeddingConfig, MappingRun
from radar.provenance import get_code_version

from radar.sources import 

import logging

logger = logging.getLogger(__name__)

def mapping(model, method, params, annotation_runs, embedding_config = None, threshold = None, allow_dirty = False):

    with Session() as session:
        embedding_config = get_or_create_embedding_config(session, model, template)
        mapping_run = create_mapping_run(session, method, params, annotation_runs, embedding_config = None, threshold = None, allow_dirty = False)




def hash_input(input):
    """Cache key of an embedding: SHA-256 of the rendered text sent to the model."""
    return hashlib.sha256(input.encode('utf-8')).hexdigest()

def get_or_create_embedding_config(session, model, template):
    embedding_config = session.scalar(select(EmbeddingConfig).where(EmbeddingConfig.model == model, EmbeddingConfig.template == template))
    if embedding_config is None:
        embedding_config = EmbeddingConfig(model = model, template = template)
        session.add(embedding_config)
    return embedding_config

# no get_or_create: the vector would have to be computed before knowing whether it is cached
def get_embedding(session, embedding_config, input):
    return session.scalar(select(Embedding).where(Embedding.config == embedding_config, Embedding.input_hash == hash_input(input)))

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
