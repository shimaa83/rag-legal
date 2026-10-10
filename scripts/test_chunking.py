import json
from pathlib import Path

from legal_rag.chunking import chunk_articles

INPUT_PATH = Path("data/processed/civil_code.json")


def main() -> None:
    with INPUT_PATH.open("r", encoding="utf-8") as file:
        articles = json.load(file)

    chunks = chunk_articles(articles)

    lengths = [
        len(chunk["text_ar_normalized"])
        for chunk in chunks
        if chunk["text_ar_normalized"]
    ]

    multi_chunk_articles = {}

    for chunk in chunks:
        article_number = chunk["article_number"]
        multi_chunk_articles.setdefault(article_number, 0)
        multi_chunk_articles[article_number] += 1

    split_articles = {
        article: count
        for article, count in multi_chunk_articles.items()
        if count > 1
    }

    print(f"Articles: {len(articles)}")
    print(f"Chunks: {len(chunks)}")

    if lengths:
        print(f"Average characters: {sum(lengths) / len(lengths):.1f}")
        print(f"Maximum characters: {max(lengths)}")
        print(f"Minimum characters: {min(lengths)}")

    print(f"Articles split into multiple chunks: {len(split_articles)}")

    if split_articles:
        print("\nSplit articles:")
        for article, count in sorted(split_articles.items()):
            print(f"  Article {article}: {count} chunks")


if __name__ == "__main__":
    main()