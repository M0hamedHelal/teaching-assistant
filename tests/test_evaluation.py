import sys
import os
import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_URL = "http://localhost:8000"

# Test cases format: (Question, Is it relevant to document?, Expected keywords)
test_cases = [
    # Relevant questions
    ("What are the three workflows supported by the assistant?", True, ["Question Generation", "Summarization", "Q&A"]),
    ("What technologies are used in the project?", True, ["FastAPI", "Gradio", "LangChain", "LangGraph"]),
    ("Does the system support video upload?", True, ["نعم", "فيديو", "video"]),

    # Irrelevant questions (Expected to be rejected)
    ("Which football team do you support?", False, []),
    ("What is the best Egyptian dish?", False, []),
    ("What is the weather temperature today?", False, []),
    ("Who invented the light bulb?", False, []),
    ("Give me a chocolate cake recipe", False, []),
]


def run_evaluation():
    correct_count = 0

    print("======================================================================")
    print("Starting RAG System Evaluation (Classification + Answer Accuracy)")
    print("======================================================================")

    for question, expected_relevant, expected_keywords in test_cases:
        try:
            response = requests.post(
                f"{BASE_URL}/ask_rag",
                params={"question": question}
            )
            data = response.json()
            answer = data.get("answer", data.get("error", "No response"))
        except Exception as e:
            print(f"Error connecting to server: {e}")
            continue

        rejection_keywords = [
            "غير متوفرة", 
            "غير متعلق", 
            "لا أجد", 
            "not available", 
            "not related", 
            "cannot find", 
            "irrelevant",
            "up load file"
        ]
        
        was_rejected = any(keyword in answer.lower() for keyword in rejection_keywords)
        actual_relevant = not was_rejected

        # 2. Accuracy Evaluation
        if expected_relevant:
            has_keywords = any(kw.lower() in answer.lower() for kw in expected_keywords) if expected_keywords else True
            is_correct = actual_relevant and has_keywords
        else:
            is_correct = was_rejected

        if is_correct:
            correct_count += 1

        status = "[SUCCESS]" if is_correct else "[FAILED]"
        print(f"\n{status} Question: {question}")
        print(f"   Expected: {'Relevant' if expected_relevant else 'Irrelevant'} | Actual: {'Relevant' if actual_relevant else 'Irrelevant'}")
        print(f"   Response: {answer[:120]}...")

    total = len(test_cases)
    accuracy = (correct_count / total) * 100

    print("\n======================================================================")
    print(f"Final RAG Evaluation Result: {correct_count}/{total} Correct ({accuracy:.1f}%)")
    print("======================================================================")

    return accuracy


if __name__ == "__main__":
    run_evaluation()