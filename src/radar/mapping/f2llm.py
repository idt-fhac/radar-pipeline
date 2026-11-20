from sentence_transformers import SentenceTransformer
import numpy as np

import umap

MODEL = "codefuse-ai/F2LLM-v2-1.7B"
PROMPT = ("Instruct: Die folgenden Begriffe stammen aus dem Bereich "
          "{topic}. Identifiziere die semantische Kategorie des Terms\nQuery: ")
# rendered text the model actually sees: the prompt is prepended to each criterion
TEMPLATE = PROMPT + "{label}"

model = SentenceTransformer(MODEL, device="cpu", model_kwargs={"torch_dtype": "bfloat16"})

def embed(topic, criteria):

    context = PROMPT.format(topic=topic)

    return model.encode(criteria, prompt=context, batch_size=32, normalize_embeddings=True)


def main():
    import matplotlib.pyplot as plt
    terms = ["Sicherheit", "Gefährlichkeit", "Preis", "teuer", "Preis-Leistung", "Geschwindigkeit", "Fahrspaß", "Reichweite"]
    context = "E-Autos"

    emb = embed(context, terms)
    print(emb.shape)

    dist = {}

    for i in range(len(terms)):
        for j in range(i+1, len(terms)):
            dist.update({f"{terms[i]} : {terms[j]}" : round(float(100 * np.linalg.norm(emb[i] - emb[j])), 1)})

    print(sorted(dist.items(), key=lambda item: item[1]))

    reducer = umap.UMAP(
        n_neighbors=4,
        min_dist=0.1,
        metric="cosine",
        random_state=42
    )

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