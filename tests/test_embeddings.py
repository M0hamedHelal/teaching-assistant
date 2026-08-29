from backend.rag import chunk_text, create_embeddings

sample_text = "هذا نص تجريبي طويل. " * 100

chunks = chunk_text(sample_text, chunk_size=20, overlap=5)
embeddings = create_embeddings(chunks)

print(f"عدد القطع: {len(chunks)}")
print(f"شكل المتجهات (embeddings): {embeddings.shape}")
print(f"أول قيم من متجه القطعة الأولى: {embeddings[0][:5]}")