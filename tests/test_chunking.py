from legal_rag.chunking import chunk_articles


def test_one_article_creates_one_chunk() -> None:
    articles = [
        {
            "article_number": 147,
            "text_ar": "النص الأصلي للمادة.",
            "text_ar_normalized": "النص المنظم للمادة.",
            "text_en": "English text.",
            "book": "Obligations",
            "chapter": "Contracts",
            "section": "Effects",
            "topic": "Contract",
            "is_repealed": False,
            "source_page": 20,
            "citation": "Egyptian Civil Code, Article 147",
        }
    ]

    chunks = chunk_articles(articles)

    assert len(chunks) == 1
    assert chunks[0]["chunk_id"] == "article_147_0"
    assert chunks[0]["article_number"] == 147
    assert chunks[0]["citation"] == "Egyptian Civil Code, Article 147"
    assert chunks[0]["text_ar_normalized"] == "النص المنظم للمادة."