from backend.rag import chunk_text, create_embeddings, build_vector_store, search_similar_chunks

sample_text = """
القاهرة هي عاصمة مصر ويبلغ عدد سكانها حوالي 10 مليون نسمة.
الأهرامات من أشهر المعالم السياحية في مصر وتقع في الجيزة.
نهر النيل هو أطول نهر في العالم ويمر عبر عدة دول أفريقية.
اللغة العربية هي اللغة الرسمية في مصر ويتحدث بها معظم السكان.
"""


def test_retrieval_finds_relevant_chunk():
    chunks = chunk_text(sample_text, chunk_size=15, overlap=3)
    embeddings = create_embeddings(chunks)
    index = build_vector_store(embeddings)

    results = search_similar_chunks("ايه عاصمة مصر؟", index, chunks, top_k=2)

    assert len(results) == 2
    assert any("القاهرة" in r for r in results)