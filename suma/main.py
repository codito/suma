# Suma extracts key sentences from a text document and synthesizes an
# abstractive summary using llm.

import sys
from typing import List

import frontmatter
import kmedoids
import markdown
import nltk
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

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


def summarize(args: List[str]):
    if len(args) < 1:
        print("Usage: suma <text file>")
        return

    if not nltk.data.find("tokenizers/punkt"):
        nltk.download("punkt")

    sentences = extract_sentences(args[0])
    embeddings = get_embeddings(sentences)
    sim_matrix = cosine_similarity(embeddings)

    num_clusters = int(len(sentences) / 5)
    km = kmedoids.KMedoids(num_clusters, method="fasterpam")
    km.fit(sim_matrix)

    for index in sorted(km.medoid_indices_):
        print(sentences[index])

    pass


def main():
    summarize(sys.argv[1:])


if __name__ == "__main__":
    summarize(sys.argv[1:])
