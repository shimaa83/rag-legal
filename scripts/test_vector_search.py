from legal_rag.retrieval import keyword_search

question = "ما هي شروط العقد؟"

results = keyword_search(
    question=question,
    top_k=5,
)

print(f"\nQuestion: {question}\n")
print("=" * 80)

for index, result in enumerate(results, start=1):
    print(f"\n#{index}")
    print(f"Article: {result['article_number']}")
    print(f"Citation: {result['citation']}")
    print(f"Score: {result['score']:.4f}")
    print(f"Text: {result['text_ar_normalized'][:500]}")