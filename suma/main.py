# Suma extracts key sentences from a text document and synthesizes an
# abstractive summary using llm.

import sys
from typing import List

import frontmatter
import kmedoids
import markdown
import nltk
import numpy as np
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from suma.lexrank import degree_centrality_scores

extractive_model = SentenceTransformer("BAAI/bge-small-en")
abstractive_model = "xxx"


def extract_sentences(file: str):
    if file.endswith(".md"):
        post = frontmatter.load(file)
        html = markdown.markdown(post.content)
        html = html.replace("\n", " ")
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text()
    else:
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()

    return nltk.tokenize.sent_tokenize(text)


def get_embeddings(phrases: List[str]):
    return extractive_model.encode(phrases)


def get_key_sentences_lexrank(phrases: List[str], num_sentences: int) -> List[str]:
    embeddings = get_embeddings(phrases)
    sim_matrix = cosine_similarity(embeddings)

    centrality_scores = degree_centrality_scores(sim_matrix, threshold=None)
    most_central_indices = np.argsort(-centrality_scores)[:num_sentences]
    return [phrases[i] for i in most_central_indices]


def get_key_sentences_kmedoid(phrases: List[str], num_sentences: int) -> List[str]:
    embeddings = get_embeddings(phrases)
    sim_matrix = cosine_similarity(embeddings)

    km = kmedoids.KMedoids(num_sentences, method="fasterpam")
    km.fit(sim_matrix)
    return [phrases[i] for i in sorted(km.medoid_indices_)]


def summarize(args: List[str]):
    if len(args) < 1:
        print("Usage: suma <text file>")
        return

    if not nltk.data.find("tokenizers/punkt"):
        nltk.download("punkt")

    sentences = extract_sentences(args[0])
    num_clusters = int(len(sentences) / 5)
    key_sentences_kmedoid = get_key_sentences_kmedoid(sentences, num_clusters)
    key_sentences_lexrank = get_key_sentences_lexrank(sentences, num_clusters)

    print(key_sentences_kmedoid)
    print("----------")
    print(key_sentences_lexrank)

    pass


def main():
    summarize(sys.argv[1:])


if __name__ == "__main__":
    summarize(sys.argv[1:])
