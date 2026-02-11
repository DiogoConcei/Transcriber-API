import os
import yt_dlp
from fastapi import FastAPI
from pydantic import BaseModel
from faster_whisper import WhisperModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["http://localhost:5173"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class VideoModel(BaseModel):
    url: str

print("Carregando Faster-Whisper (tiny.en)...")
model = WhisperModel(
    "tiny.en",
    device="cpu",
    compute_type="int8"
)
print("Modelo carregado!")

@app.post("/api/video")
def transcribe_faster(video: VideoModel):

    if not os.path.exists("downloads"):
        os.makedirs("downloads")

    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "outtmpl": "downloads/%(id)s.%(ext)s",
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video.url, download=True)
        filename = f"downloads/{info['id']}.mp3"

    print("Transcrevendo com Faster-Whisper...")
    segments, info = model.transcribe(
        filename,
        language="en",
        beam_size=1
    )

    segments = list(segments)

    formatted_segments = [
        {
            "text": seg.text.strip(),
            "start": seg.start,
            "end": seg.end
        }

        for seg in segments
        if seg.text.strip()
    ]

    return {
        "statusCode": 200,
        "segments": formatted_segments,
        "engine": "faster-whisper",
    }
