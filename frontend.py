import gradio as gr
import requests

def ask_backend(question):
    response = requests.post(
        "http://localhost:8000/ask",
        params={"question": question}
    )
    return response.json()["answer"]

def upload_pdf_ui(file):
    with open(file.name, "rb") as f:
        response = requests.post(
            "http://localhost:8000/upload",
            files={"file": f}
        )
    data = response.json()
    return f"File: {data['filename']}\n Text Length: {data['full_length']}\n\nPreview:\n{data['text_preview']}"

def upload_media_ui(file):
    with open(file.name, "rb") as f:
        response = requests.post(
            "http://localhost:8000/upload_media",
            files={"file": f}
        )
    data = response.json()
    if "error" in data:
        return data["error"]
    return f"File: {data['filename']}\nText Length: {data['full_length']}\n\nPreview:\n{data['text_preview']}"

def ask_rag_ui(question):
    response = requests.post(
        "http://localhost:8000/ask_rag",
        params={"question": question}
    )
    data = response.json()
    if "error" in data:
        return data["error"]
    return data["answer"]

with gr.Blocks() as demo:
    with gr.Tab("ASK"):
        gr.Interface(fn=ask_backend, inputs="text", outputs="text")
    with gr.Tab("Upload PDF File"):
        gr.Interface(fn=upload_pdf_ui, inputs=gr.File(), outputs="text")
    with gr.Tab("Upload Audio or Video"):
        gr.Interface(fn=upload_media_ui, inputs=gr.File(), outputs="text")
    with gr.Tab("Ask Document"):
        gr.Interface(fn=ask_rag_ui, inputs="text", outputs="text")
demo.launch()