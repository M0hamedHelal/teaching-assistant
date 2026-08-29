from backend.rag import chunk_text, create_embeddings, build_vector_store, search_similar_chunks

# نص تجريبي فيه معلومات مختلفة عشان نتأكد إن البحث فعلاً بيميز بينها
sample_text = """
القاهرة هي عاصمة مصر ويبلغ عدد سكانها حوالي 10 مليون نسمة.
الأهرامات من أشهر المعالم السياحية في مصر وتقع في الجيزة.
نهر النيل هو أطول نهر في العالم ويمر عبر عدة دول أفريقية.
اللغة العربية هي اللغة الرسمية في مصر ويتحدث بها معظم السكان.
"""

chunks = chunk_text(sample_text, chunk_size=15, overlap=3)
embeddings = create_embeddings(chunks)
index = build_vector_store(embeddings)

question = "ايه عاصمة مصر؟"
results = search_similar_chunks(question, index, chunks, top_k=2)

print(f"السؤال: {question}\n")
print("أقرب النتائج:")
for i, r in enumerate(results, 1):
    print(f"{i}. {r}\n")