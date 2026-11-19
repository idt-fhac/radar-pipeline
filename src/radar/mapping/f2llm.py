from sentence_transformers import SentenceTransformer
import numpy as np
import scipy as sp

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
    list = ["Sicherheit", "Gefährlichkeit", "Preis", "teuer", "Preis-Leistung", "Geschwindigkeit", "Fahrspaß", "Reichweite"]
    context = "E-Autos"

    emb = embed(context, list)
    print(emb.shape)

    dist = {}

    for i in range(len(list)):
        for j in range(i+1, len(list)):
            dist.update({f"{list[i]} : {list[j]}" : round(float(100 * np.linalg.norm(emb[i] - emb[j])), 1)})

    print(sorted(dist.items(), key=lambda item: item[1]))
    

if __name__ == "__main__":
    main()