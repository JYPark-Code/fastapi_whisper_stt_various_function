import shutil
import os
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from fastapi.responses import FileResponse
# import requests
import whisper
# youtube subtitle
from whisper.utils import WriteSRT
from fastapi.responses import JSONResponse
import chardet
import json
from pathlib import Path

from pydub import AudioSegment
# for video to audio and extract youtube links to audio
from moviepy.editor import VideoFileClip
import pytube

app = FastAPI()

@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.post("/stt_only_uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    with open(file.filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # audio = AudioSegment.from_file(file.filename)
    # return {"filename": file.filename, "duration": len(audio)}
    # whisper_detect_lang(file.filename)

    return {"transcription": whisper_transcribe(file.filename)}


# Route to upload a video file and extract audio
# Didn't try yet
@app.post("/stt_from_video/")
async def extract_audio(file: UploadFile = File(...)):
    # Save the uploaded video file to disk
    with open(file.filename, "wb") as buffer:
        buffer.write(await file.read())

    # Extract audio from the video file
    video = VideoFileClip(file.filename)
    audio = video.audio
    audio_filename = f"{file.filename.split('.')[0]}.mp3"
    audio.write_audiofile(audio_filename)

    # Return the audio file
    # return FileResponse(audio_filename)
    return {"transcription": whisper_transcribe(audio_filename)}


# Route to extract audio from a YouTube link
@app.post("/stt_youtube_video")
async def extract_youtube_audio(link: str):
    try:
       # Download video using pytube
        yt = pytube.YouTube(link)
        video = yt.streams.filter(only_audio=True).first()
        video_file = video.download(output_path="audio_files", filename=str(uuid.uuid4()))
        
        # Convert video to audio using pydub
        audio = AudioSegment.from_file(video_file)
        audio_file = os.path.splitext(video_file)[0] + ".mp3"
        audio.export(audio_file, format="mp3")

        # return {"transcription": whisper_transcribe(audio_file)}

        # Generate SRT file with timestamped transcript
        result = whisper_transcribe_result(audio_file)

        p = Path(audio_file)

        srt_path = p.with_suffix('.srt').with_name(p.stem)
        writer = WriteSRT(p.parent)
        writer(result, srt_path)
        
        # return {"Subtitles": read_srt_pysubs(srt_path)}
        return srt_to_json(srt_path.with_suffix('.srt'))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# detect languages and decode to lower-level access model
def whisper_detect_lang(input_audio):
    model = whisper.load_model("base")

    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(input_audio)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    # decode the audio
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)

    # print the recognized text
    # print(result.text)
    return (result.text)


def whisper_transcribe(input_audio):
    model = whisper.load_model("base")
    result = model.transcribe(input_audio)
    return (result["text"])


def whisper_transcribe_result(input_audio):
    model = whisper.load_model("base")
    result = model.transcribe(input_audio)
    return result


def srt_to_json(path: str):
    # Open the .srt file and detect its character encoding
    with open(path, "rb") as f:
        srt_content = f.read()
    encoding = chardet.detect(srt_content)["encoding"]

    # Read the .srt file with the correct encoding
    with open(path, "r", encoding=encoding) as f:
        srt_content = f.read()

    # Split the .srt content into lines and process each line
    srt_lines = srt_content.split("\n")
    srt_json = []
    i = 0
    while i < len(srt_lines):
        if srt_lines[i] == '':
            i += 1
            continue
        if i + 2 >= len(srt_lines):
            break
        # Get the start and end timestamps
        timestamps = srt_lines[i+1].split(" --> ")
        start_time = timestamps[0]
        end_time = timestamps[1]

        # Get the subtitle text
        subtitle_text = srt_lines[i+2]

        # Create a dictionary object for the subtitle
        subtitle = {
            "start_time": start_time,
            "end_time": end_time,
            "subtitle_text": subtitle_text
        }

        # Add the subtitle to the JSON array
        srt_json.append(subtitle)

        i += 4

    # Convert the JSON array to a JSON object
    srt_json_obj = {"subtitles": srt_json}

    # Return the JSON object as the response
    return Response(content=json.dumps(srt_json_obj, ensure_ascii=False), media_type="application/json")
