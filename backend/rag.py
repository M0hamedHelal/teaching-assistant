from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:

    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  

    return chunks


embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def create_embeddings(chunks: list[str]):
 
    embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
    return embeddings


def build_vector_store(embeddings: np.ndarray):
    """
    يبني فهرس FAISS من المتجهات، عشان نقدر نبحث فيه بسرعة
    """
    dimension = embeddings.shape[1]  
    index = faiss.IndexFlatL2(dimension)  
    index.add(embeddings)
    return index


def search_similar_chunks(query: str, index, chunks: list[str], top_k: int = 3):
    """
    يبحث عن أقرب القطع لسؤال معين
    """
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)

    results = [chunks[i] for i in indices[0]]
    return results

def ask_with_rag(question: str, index, chunks: list[str], ask_llm_func, top_k: int = 3) -> str:
    relevant_chunks = search_similar_chunks(question, index, chunks, top_k=top_k)

    print("=" * 50)
    print(f"Question: {question}")
    print("Retrieved Chunks:")
    
    for i, chunk in enumerate(relevant_chunks, 1):
        print(f"\n[{i}] {chunk[:200]}...")
    print("=" * 50)

    context = "\n\n".join(relevant_chunks)

    
    prompt = f"""Use ONLY the provided context below to answer the user's question clearly and directly.
Do NOT invent, infer, or assume any information that is not explicitly mentioned in the context.
If the context does not contain enough information to answer the question, explicitly state: "هذه المعلومة غير متوفرة في الملف المرفوع."
Always respond in the same language as the question.

Context:
\"\"\"
{context}
\"\"\"

Question: {question}

Answer:"""

    answer = ask_llm_func(prompt)
    return answer
