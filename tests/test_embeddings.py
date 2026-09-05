from backend.rag import chunk_text, create_embeddings


def test_embeddings_shape_matches_chunks():
    sample_text = "هذا نص تجريبي طويل. " * 100
    chunks = chunk_text(sample_text, chunk_size=20, overlap=5)
    embeddings = create_embeddings(chunks)

    assert embeddings.shape[0] == len(chunks)
    assert embeddings.shape[1] > 0  