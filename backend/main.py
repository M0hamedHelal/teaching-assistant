import imageio_ffmpeg
import os

# imageio_ffmpeg بيجيب مسار ffmpeg جاهز وشغال على أي نظام تشغيل (ويندوز/لينكس/ماك)
# فمش محتاجين ننسخه في مكان تاني، بس نضيف الفولدر بتاعه للـ PATH
ffmpeg_dir = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]
from fastapi import FastAPI, UploadFile, File
import requests
import fitz
import whisper
from moviepy import VideoFileClip

from backend.rag import chunk_text, create_embeddings, build_vector_store, ask_with_rag
from backend.rag_graph import build_rag_graph
from backend.rag import search_similar_chunks
app = FastAPI()

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2"

whisper_model = whisper.load_model("base")

# حالة الجلسة الحالية (ملف واحد بس دلوقتي)
current_chunks = []
current_index = None


# def ask_llm(prompt: str) -> str:
#     response = requests.post(
#         OLLAMA_URL,
#         json={"model": MODEL_NAME, "prompt": prompt, "stream": False}
#     )
#     return response.json()["response"]


def ask_llm(prompt: str) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1
            }
        }
    )
    return response.json()["response"]

def process_text_for_rag(text: str):
    """Chunks the extracted text, creates vector embeddings, and updates the global vector store."""
    global current_chunks, current_index
    current_chunks = chunk_text(text, chunk_size=500, overlap=50)
    embeddings = create_embeddings(current_chunks)
    current_index = build_vector_store(embeddings)


@app.get("/")
def home():
    return {"status": "backend Successful"}


@app.post("/ask")
def ask(question: str):
    answer = ask_llm(question)
    return {"answer": answer}


@app.post("/ask_rag")
def ask_rag(question: str):
    if current_index is None or len(current_chunks) == 0:
        return {"error": "You Should Up Load File Befor Ask ."}
    
    rag_graph = build_rag_graph(ask_llm, search_similar_chunks, current_index, current_chunks)
    
    result = rag_graph.invoke({
        "question": question,
        "is_relevant": False,
        "retrieved_chunks": [],
        "answer": ""
    })
    
    return {"answer": result["answer"]}



def extract_text_from_pdf(file_path: str) -> str:
    doc = fitz.open(file_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    return full_text


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    extracted_text = extract_text_from_pdf(temp_path)
    os.remove(temp_path)

    process_text_for_rag(extracted_text)

    return {
        "filename": file.filename,
        "text_preview": extracted_text[:500],
        "full_length": len(extracted_text)
    }


def extract_audio_from_video(video_path: str) -> str:
    audio_path = video_path.rsplit(".", 1)[0] + ".wav"
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path, logger=None)
    video.close()
    return audio_path


def transcribe_audio(audio_path: str) -> str:
    result = whisper_model.transcribe(audio_path, language="ar")
    return result["text"]


@app.post("/upload_media")
async def upload_media(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    file_extension = file.filename.rsplit(".", 1)[-1].lower()
    video_extensions = ["mp4", "avi", "mov", "mkv", "flv"]
    audio_extensions = ["mp3", "wav"]

    if file_extension in video_extensions:
        audio_path = extract_audio_from_video(temp_path)
        extracted_text = transcribe_audio(audio_path)
        os.remove(audio_path)
    elif file_extension in audio_extensions:
        extracted_text = transcribe_audio(temp_path)
    else:
        os.remove(temp_path)
        return {"error": "Unsupported file format"}

    os.remove(temp_path)

    process_text_for_rag(extracted_text)

    return {
        "filename": file.filename,
        "text_preview": extracted_text[:500],
        "full_length": len(extracted_text)
    }