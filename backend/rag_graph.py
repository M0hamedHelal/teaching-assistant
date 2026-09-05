from typing import TypedDict
from langgraph.graph import StateGraph, END



class RAGState(TypedDict):
    question: str
    is_relevant: bool
    retrieved_chunks: list[str]
    answer: str


def classify_node(state: RAGState, ask_llm_func) -> RAGState:
    question = state["question"]

    prompt = f"""Is the following question asking for information, facts, or details that could be found in a document? 
If it is a general casual chat like "hello", "who are you", or completely unrelated (like sports/weather), answer "NO".
Otherwise, answer "YES".

Question: {question}

Answer with ONLY "YES" or "NO":"""

    result = ask_llm_func(prompt).strip().upper()

   
    is_relevant = "YES" in result and "NO" not in result
    
    
    if "YES" not in result and "NO" not in result:
        is_relevant = True

    state["is_relevant"] = is_relevant
    return state



def retrieve_node(state: RAGState, search_func, index, chunks) -> RAGState:
    question = state["question"]

   
    total_chunks = len(chunks)
    if total_chunks <= 5:
        dynamic_top_k = total_chunks
    elif total_chunks <= 15:
        dynamic_top_k = 5
    else:
        dynamic_top_k = 10

    retrieved = search_func(question, index, chunks, top_k=dynamic_top_k)

    print("=" * 50)
    print(f"Question: {question}")
    print(f"Retrieved {len(retrieved)} chunks (top_k={dynamic_top_k}):")
    for i, chunk in enumerate(retrieved, 1):
        print(f"\n[{i}] {chunk[:150]}...")
    print("=" * 50)

    state["retrieved_chunks"] = retrieved
    return state



def answer_node(state: RAGState, ask_llm_func) -> RAGState:
    context = "\n\n".join(state["retrieved_chunks"])
    question = state["question"]
    
    prompt = f"""Use ONLY the provided context below to answer the user's question accurately and directly.
Do NOT invent, infer, or assume any information that is not explicitly mentioned in the context.
If the context does not contain enough information to answer the question, respond with: "هذه المعلومة غير متوفرة في الملف المرفوع."
Always respond in the same language as the user's question (e.g., respond in Arabic if asked in Arabic, respond in English if asked in English).

Context:
\"\"\"
{context}
\"\"\"

Question: {question}

Answer:"""

    answer = ask_llm_func(prompt)
    state["answer"] = answer
    return state



def rejected_node(state: RAGState) -> RAGState:
    state["answer"] = "هذا السؤال لا يبدو متعلقًا بمحتوى الملف المرفوع. يرجى طرح سؤال عن محتوى المستند."
    return state



def decide_after_classify(state: RAGState) -> str:
    if state["is_relevant"]:
        return "retrieve"
    else:
        return "reject"


def build_rag_graph(ask_llm_func, search_func, index, chunks):
    graph = StateGraph(RAGState)

   
    graph.add_node("classify", lambda state: classify_node(state, ask_llm_func))
    graph.add_node("retrieve", lambda state: retrieve_node(state, search_func, index, chunks))
    graph.add_node("answer", lambda state: answer_node(state, ask_llm_func))
    graph.add_node("reject", rejected_node)

  
    graph.set_entry_point("classify")

   
    graph.add_conditional_edges(
        "classify",
        decide_after_classify,
        {
            "retrieve": "retrieve",
            "reject": "reject"
        }
    )

   
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)
    graph.add_edge("reject", END)

    return graph.compile()