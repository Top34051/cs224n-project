import numpy as np
from sentence_transformers import SentenceTransformer


class SimilarityRewardMax:
    def __init__(self, contexts, prefix=""):
        self.contexts = contexts
        self.prefix = prefix
        self.model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        self.embeddings = self.model.encode(contexts)

    def __call__(self, texts):
        texts = [text.removeprefix(self.prefix) for text in texts]
        text_embeddings = self.model.encode(texts)
        rewards = np.max(np.dot(text_embeddings, self.embeddings.T), axis=1)
        return rewards
