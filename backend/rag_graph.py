# from typing import TypedDict
# from langgraph.graph import StateGraph, END


# # 1. تعريف الـ State: الصندوق اللي هيحمل المعلومات بين الخطوات
# class RAGState(TypedDict):
#     question: str
#     is_relevant: bool
#     retrieved_chunks: list[str]
#     answer: str


# # 2. Node رقم 1: Classify - يحدد هل السؤال متعلق بالموضوع ولا لأ
# def classify_node(state: RAGState, ask_llm_func) -> RAGState:
#     question = state["question"]

#     prompt = f"""Is the following question asking for information, facts, or details that could be found in a document? 
# If it is a general casual chat like "hello", "who are you", or completely unrelated (like sports/weather), answer "NO".
# Otherwise, answer "YES".

# Question: {question}

# Answer with ONLY "YES" or "NO":"""

#     result = ask_llm_func(prompt).strip().upper()

#     # فحص مرن لتجنب أخطاء التصنيف
#     is_relevant = "YES" in result or "NO" not in result
#     state["is_relevant"] = is_relevant
#     return state


# # 3. Node رقم 2: Retrieve - يجيب القطع المناسبة من الـ vector store
# def retrieve_node(state: RAGState, search_func, index, chunks) -> RAGState:
#     question = state["question"]

#     # احتساب top_k ديناميكياً بحسب عدد القطع المتاحة
#     total_chunks = len(chunks)
#     if total_chunks <= 5:
#         dynamic_top_k = total_chunks
#     elif total_chunks <= 15:
#         dynamic_top_k = 5
#     else:
#         dynamic_top_k = 10

#     retrieved = search_func(question, index, chunks, top_k=dynamic_top_k)

#     print("=" * 50)
#     print(f"Question: {question}")
#     print(f"Retrieved {len(retrieved)} chunks (top_k={dynamic_top_k}):")
#     for i, chunk in enumerate(retrieved, 1):
#         print(f"\n[{i}] {chunk[:150]}...")
#     print("=" * 50)

#     state["retrieved_chunks"] = retrieved
#     return state


# # 4. Node رقم 3: Answer - يولد الإجابة النهائية
# def answer_node(state: RAGState, ask_llm_func) -> RAGState:
#     context = "\n\n".join(state["retrieved_chunks"])
#     question = state["question"]
    
#     prompt = f"""استخدم المعلومات التالية فقط للإجابة على السؤال، ولا تخترع أي معلومة غير موجودة فيها. إذا لم تجد إجابة واضحة، قل بوضوح: "هذه المعلومة غير متوفرة في الملف المرفوع." أجب بنفس لغة السؤال.

# المعلومات:
# \"\"\"
# {context}
# \"\"\"

# السؤال: {question}

# الإجابة (بنفس لغة السؤال):"""

#     answer = ask_llm_func(prompt)
#     state["answer"] = answer
#     return state


# # 5. Node بديل: رد مباشر لو السؤال مش متعلق بالموضوع
# def rejected_node(state: RAGState) -> RAGState:
#     state["answer"] = "هذا السؤال لا يبدو متعلقًا بمحتوى الملف المرفوع. يرجى طرح سؤال عن محتوى المستند."
#     return state


# # 6. دالة القرار: تحدد أي مسار ناخد بعد الـ classify
# def decide_after_classify(state: RAGState) -> str:
#     if state["is_relevant"]:
#         return "retrieve"
#     else:
#         return "reject"
# def build_rag_graph(ask_llm_func, search_func, index, chunks):
#     graph = StateGraph(RAGState)

#     # نضيف الـ nodes، مع "تغليفها" (wrapping) عشان تقدر تاخد المتغيرات الإضافية
#     graph.add_node("classify", lambda state: classify_node(state, ask_llm_func))
#     graph.add_node("retrieve", lambda state: retrieve_node(state, search_func, index, chunks))
#     graph.add_node("answer", lambda state: answer_node(state, ask_llm_func))
#     graph.add_node("reject", rejected_node)

#     # نحدد نقطة البداية
#     graph.set_entry_point("classify")

#     # نحدد التفرع الشرطي بعد الـ classify
#     graph.add_conditional_edges(
#         "classify",
#         decide_after_classify,
#         {
#             "retrieve": "retrieve",
#             "reject": "reject"
#         }
#     )

#     # بعد retrieve، نروح على طول لـ answer
#     graph.add_edge("retrieve", "answer")

#     # answer و reject الاتنين نهاية المسار
#     graph.add_edge("answer", END)
#     graph.add_edge("reject", END)

#     return graph.compile()    
from typing import TypedDict
from langgraph.graph import StateGraph, END


# 1. تعريف الـ State: الصندوق الذي يحمل المعلومات بين الخطوات
class RAGState(TypedDict):
    question: str
    is_relevant: bool
    retrieved_chunks: list[str]
    answer: str


# 2. Node رقم 1: Classify - يحدد هل السؤال متعلق بالموضوع أم لا
def classify_node(state: RAGState, ask_llm_func) -> RAGState:
    question = state["question"]

    prompt = f"""Is the following question asking for information, facts, or details that could be found in a document? 
If it is a general casual chat like "hello", "who are you", or completely unrelated (like sports/weather), answer "NO".
Otherwise, answer "YES".

Question: {question}

Answer with ONLY "YES" or "NO":"""

    result = ask_llm_func(prompt).strip().upper()

    # فحص دقيق: يعتبر السؤال متعلقاً فقط إذا كانت الإجابة صريحة بـ YES أو لا تحتوي على NO
    is_relevant = "YES" in result and "NO" not in result
    
    # في حال استجابة النموذج بأسلوب غير غير المتوقع، نعتبر السؤال مرتبطاً كافتراض آمن
    if "YES" not in result and "NO" not in result:
        is_relevant = True

    state["is_relevant"] = is_relevant
    return state


# 3. Node رقم 2: Retrieve - يسترجع القطع المناسبة من الـ vector store
def retrieve_node(state: RAGState, search_func, index, chunks) -> RAGState:
    question = state["question"]

    # احتساب top_k ديناميكياً بحسب عدد القطع المتاحة
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


# 4. Node رقم 3: Answer - يولد الإجابة النهائية بالبرومبت الإنجليزي المحسّن
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


# 5. Node بديل: رد مباشر إذا كان السؤال خارج نطاق الملف
def rejected_node(state: RAGState) -> RAGState:
    state["answer"] = "هذا السؤال لا يبدو متعلقًا بمحتوى الملف المرفوع. يرجى طرح سؤال عن محتوى المستند."
    return state


# 6. دالة القرار: تحدد المسار بعد عقدة التصنيف
def decide_after_classify(state: RAGState) -> str:
    if state["is_relevant"]:
        return "retrieve"
    else:
        return "reject"


def build_rag_graph(ask_llm_func, search_func, index, chunks):
    graph = StateGraph(RAGState)

    # إضافة الـ nodes مع تغليفها بالتمرير المناسب
    graph.add_node("classify", lambda state: classify_node(state, ask_llm_func))
    graph.add_node("retrieve", lambda state: retrieve_node(state, search_func, index, chunks))
    graph.add_node("answer", lambda state: answer_node(state, ask_llm_func))
    graph.add_node("reject", rejected_node)

    # تحديد نقطة البداية
    graph.set_entry_point("classify")

    # المسار الشرطي
    graph.add_conditional_edges(
        "classify",
        decide_after_classify,
        {
            "retrieve": "retrieve",
            "reject": "reject"
        }
    )

    # الانتقالات المباشرة إلى END
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)
    graph.add_edge("reject", END)

    return graph.compile()