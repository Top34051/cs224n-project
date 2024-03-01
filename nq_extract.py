from datasets import load_dataset
from sentence_transformers import SentenceTransformer

import torch
import json
import os
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


dataset = load_dataset("lighteval/natural_questions_clean", split="validation")
dataset = dataset.shuffle(seed=42).select(range(1000))

model_name_or_path = "sentence-transformers/all-mpnet-base-v2"
model = SentenceTransformer(model_name_or_path, device="cuda" if torch.cuda.is_available() else "cpu")

chunk_size = 500
overlap = 100

database = {
    "metadata": {
        "dataset": "natural_questions_clean",
        "split": "validation",
        "embedding": model_name_or_path,
        "chunk_size": chunk_size,
        "overlap": overlap,
    }
}

for id, sample in enumerate(tqdm(dataset)):
    question = sample["question"]
    docs = []
    for doc in [sample["document"]]:
        docs.extend([doc[i : i + chunk_size] for i in range(0, len(doc), chunk_size - overlap)])

    embeddings = model.encode([question] + docs)
    similarities = cosine_similarity([embeddings[0]], embeddings[1:]).tolist()[0]
    contexts = sorted(list(zip(similarities, docs)), key=lambda x: x[0], reverse=True)

    if id % 100 == 0:
        print(question)
        print(sample["short_answers"])
        for context in contexts[:5]:
            print(context)
        print()

    database[id] = {
        "question": question,
        "answers": sample["short_answers"],
        "contexts": [context[1] for context in contexts],
        "similarities": [context[0] for context in contexts],
    }

os.makedirs("nq", exist_ok=True)
json.dump(database, open("nq/validation.json", "w"), indent=2)
