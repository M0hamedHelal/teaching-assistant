import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.rag import chunk_text, create_embeddings, build_vector_store, ask_with_rag
from backend.main import ask_llm

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
answer = ask_with_rag(question, index, chunks, ask_llm)

print(f"السؤال: {question}")
print(f"الإجابة: {answer}")