from backend.rag import chunk_text

sample_text = "i'm mohamed helal. i'm enginner." * 100  # نص وهمي للاختبار

chunks = chunk_text(sample_text, chunk_size=20, overlap=5)

print(f"عدد القطع: {len(chunks)}")
print(f"أول قطعة:\n{chunks[0]}")
print(f"\nتاني قطعة:\n{chunks[1]}")