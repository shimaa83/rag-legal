from legal_rag.retrieval import hybrid_search

question = "ما هي شروط العقد؟"

results = hybrid_search(
    question=question,
    top_k=5,
    candidate_k=10,
)

print(f"\nQuestion: {question}\n")
print("=" * 80)

for index, result in enumerate(results, start=1):
    print(f"\n#{index}")
    print(f"Article: {result['article_number']}")
    print(f"Citation: {result['citation']}")
    print(f"RRF Score: {result['rrf_score']:.6f}")
    print(f"Vector Rank: {result['vector_rank']}")
    print(f"Keyword Rank: {result['keyword_rank']}")
    print(f"Text: {result['text_ar_normalized'][:500]}")