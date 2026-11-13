from sentence_transformers import SentenceTransformer


model = SentenceTransformer("codefuse-ai/F2LLM-v2-1.7B", device="cpu", model_kwargs={"torch_dtype": "bfloat16"})

def embed(topic, criteria):

    context = ("Instruct: Die folgenden Begriffe stammen aus dem Bereich "
           f"{topic}. Identify the semantic category of the term\nQuery: ")

    return model.encode(criteria, prompt=context, batch_size=32, normalize_embeddings=True)


def main():
    list = ["Sicherheit", "Gefährlichkeit", "Preis", "teuer", "Preis-Leistung", "Geschwindigkeit", "Fahrspaß", "Reichweite"]
    context = "E-Autos"

    embedding = embed(context, list)
    print(embedding.shape)

if __name__ == "__main__":
    main()