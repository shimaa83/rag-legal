from legal_rag.embeddings import ArabicEmbedder


def test_arabic_embedding_dimension() -> None:
    embedder = ArabicEmbedder()

    texts = [
        "يجب تنفيذ العقد طبقاً لما اشتمل عليه وبطريقة تتفق مع ما يوجبه حسن النية.",
        "العقد شريعة المتعاقدين.",
        "إذا تم العقد بطريقة صحيحة فلا يجوز نقضه أو تعديله إلا باتفاق الطرفين.",
    ]

    embeddings = embedder.embed_documents(texts)

    assert len(embeddings) == 3
    assert all(len(embedding) == 384 for embedding in embeddings)


def test_query_embedding_dimension() -> None:
    embedder = ArabicEmbedder()

    embedding = embedder.embed_query("ما هي آثار العقد؟")

    assert len(embedding) == 384