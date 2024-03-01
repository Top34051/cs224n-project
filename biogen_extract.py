import wikipedia
import wikipediaapi
import re
from tqdm import tqdm
import torch
import os
import json

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


with open("biogen/entities.txt", "r") as f:
    entities = [entity.strip() for entity in f.readlines()]


def clean_wiki_text(text):
    text = re.sub(r"\[(\d+)\]", "", text)
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


model_name_or_path = "sentence-transformers/all-mpnet-base-v2"
model = SentenceTransformer(model_name_or_path, device="cuda" if torch.cuda.is_available() else "cpu")

chunk_size = 500
overlap = 100

database = {
    "metadata": {
        "dataset": "biogen",
        "embedding": model_name_or_path,
        "chunk_size": chunk_size,
        "overlap": overlap,
    }
}

wiki_wiki = wikipediaapi.Wikipedia("MyProjectName (merlin@example.com)", "en")

id = 0
for entity in tqdm(entities):

    try:
        page = wiki_wiki.page(entity)
        if not page.exists():
            raise wikipedia.exceptions.PageError

        sections = page.text.split("\n\n")
        docs = []
        for section in sections:
            doc = clean_wiki_text(section)
            docs.extend([doc[i : i + chunk_size] for i in range(0, len(doc), chunk_size - overlap)])

        question = "Question: Tell me a bio of {entity}.".format(entity=entity)

        embeddings = model.encode([question] + docs)
        similarities = cosine_similarity([embeddings[0]], embeddings[1:]).tolist()[0]
        contexts = sorted(list(zip(similarities, docs)), key=lambda x: x[0], reverse=True)

        database[id] = {
            "question": question,
            "contexts": [context[1] for context in contexts],
            "similarities": [context[0] for context in contexts],
        }
        id += 1

        os.makedirs("biogen", exist_ok=True)
        json.dump(database, open("biogen/entities.json", "w"), indent=2)

    except wikipedia.exceptions.DisambiguationError as e:
        print("Multiple search results found. Here are the options:")
        for option in e.options:
            print(option)
    except wikipedia.exceptions.PageError:
        print(f"No Wikipedia page found for '{entity}'.")
