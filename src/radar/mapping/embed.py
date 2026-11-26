# embed.py — turns texts into embedding vectors. Knows no database: texts in, vectors out.
# The model comes from the [embedding] section of config.toml / config.example.toml:
# either a local sentence-transformers model or an OpenAI-compatible /embeddings endpoint.

import os

import numpy as np

from radar.config import CONFIG, SOURCE, providers, setting

PROVIDERS, DEFAULT_PROVIDER = providers('embedding')

# What is embedded per criterion: named patterns from [embedding.templates].
TEMPLATES = CONFIG['embedding'].get('templates', {})
DEFAULT_TEMPLATE = CONFIG['embedding'].get('template')
if DEFAULT_TEMPLATE not in TEMPLATES:
    raise ValueError(f'default embedding template {DEFAULT_TEMPLATE!r} is not defined in {SOURCE}')

TIMEOUT = setting('embedding', 'timeout')
MAX_RETRIES = setting('embedding', 'max_retries')

_models = {}  # one loaded model or API client per provider, created on first use


def model_id(provider=DEFAULT_PROVIDER):
    """Identifier stored in embedding_config.model, e.g. 'f2llm:codefuse-ai/F2LLM-v2-1.7B'.

    The provider is part of it: the same model name at another provider can be a different
    deployment (e.g. quantised) and must not share the embedding cache.
    """
    return f'{provider}:{config(provider)["model"]}'


def template(name=DEFAULT_TEMPLATE):
    """The pattern of a named template, e.g. 'Instruct: ...\nQuery: {label}'."""
    if name not in TEMPLATES:
        raise ValueError(f'unknown embedding template {name!r}, known: {", ".join(TEMPLATES)}')
    return TEMPLATES[name]


def render(pattern, topic, label):
    """The text sent to the model for one label; its hash is the cache key.

    str.replace instead of str.format: labels come from an LLM and may contain braces.
    """
    return pattern.replace('{topic}', topic).replace('{label}', label)


def config(provider):
    if provider not in PROVIDERS:
        raise ValueError(f'unknown embedding provider {provider!r}, known: {", ".join(PROVIDERS)}')
    return PROVIDERS[provider]


def embed(texts, provider=DEFAULT_PROVIDER):
    """Embed fully rendered texts; returns an array (len(texts), dimension) of float32, length 1 per row."""
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    backend = config(provider)['backend']
    if backend == 'local':
        vectors = embed_local(texts, provider)
    elif backend == 'openai':
        vectors = embed_openai(texts, provider)
    else:
        raise ValueError(f'unknown backend {backend!r} for embedding provider {provider!r}')
    vectors = np.asarray(vectors, dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def embed_local(texts, provider):
    if provider not in _models:
        from sentence_transformers import SentenceTransformer  # heavy import, only when needed
        settings = config(provider)
        _models[provider] = SentenceTransformer(
            settings['model'], device=settings.get('device', 'cpu'),
            model_kwargs={'torch_dtype': settings.get('dtype', 'float32')})
    return _models[provider].encode(texts, batch_size=32)


def embed_openai(texts, provider):
    settings = config(provider)
    if provider not in _models:
        from openai import OpenAI
        _models[provider] = OpenAI(base_url=settings['base_url'], api_key=os.environ[settings['key_env']],
                                   timeout=TIMEOUT, max_retries=MAX_RETRIES)
    response = _models[provider].embeddings.create(model=settings['model'], input=texts)
    return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]


def main():
    """Demo: distances and a UMAP plot for a few terms (needs umap-learn and matplotlib)."""
    import matplotlib.pyplot as plt
    import umap

    terms = ["Sicherheit", "Gefährlichkeit", "Preis", "teuer", "Preis-Leistung", "Geschwindigkeit", "Fahrspaß", "Reichweite"]
    context = "E-Autos"

    emb = embed([render(template(), context, term) for term in terms])
    print(emb.shape)

    dist = {}
    for i in range(len(terms)):
        for j in range(i + 1, len(terms)):
            dist.update({f"{terms[i]} : {terms[j]}": round(float(100 * np.linalg.norm(emb[i] - emb[j])), 1)})
    print(sorted(dist.items(), key=lambda item: item[1]))

    reducer = umap.UMAP(n_neighbors=4, min_dist=0.1, metric="cosine", random_state=42)
    emb_down = reducer.fit_transform(emb)

    plt.figure(figsize=(8, 8))
    plt.scatter(emb_down[:, 0], emb_down[:, 1], s=3, alpha=0.7)
    for (x, y), term in zip(emb_down, terms):
        plt.annotate(term, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=11)
    plt.gca().set_aspect("equal", "datalim")
    plt.title(f"UMAP: Begriffe im Kontext '{context}'")
    plt.show()


if __name__ == "__main__":
    main()
