from backend.rag import chunk_text


def test_chunk_text_produces_multiple_chunks():
    sample_text = "i'm mohamed helal. i'm enginner. " * 100
    chunks = chunk_text(sample_text, chunk_size=20, overlap=5)

    assert len(chunks) > 1
    assert all(isinstance(c, str) and len(c) > 0 for c in chunks)


def test_chunk_overlap_shares_words():
    sample_text = "word " * 200
    chunks = chunk_text(sample_text, chunk_size=20, overlap=5)

    first_words = chunks[0].split()
    second_words = chunks[1].split()
    # آخر 5 كلمات من القطعة الأولى المفروض تتكرر في أول القطعة الثانية
    assert first_words[-5:] == second_words[:5]