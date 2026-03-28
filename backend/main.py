from utilities import process_video
from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from typing import Annotated
import json

app = FastAPI()

#Set uploads folder path to save the uploaded video to.
UPLOAD_DIR = Path() / 'uploads'

#Set origin for the frontend port
origins = [
    "http://localhost:5173"
]

#Enable frontend access to the server port 8000
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

#Set up processed videos directory to serve the annotated videos to the frontend from this location
app.mount("/processed_videos", StaticFiles(directory="processed_videos"), name="processed_videos")

@app.get("/")
def root():
    return{"Hello":"World"}

#Post to accept input data
@app.post("/upload/")
async def upload_video(file: Annotated[UploadFile, File()], points: Annotated[str, Form()]):
    #Destringify the json points into json object
    points = json.loads(points)
    #Read the video file
    data = await file.read()
    #Save and write the video file to the backend/uploads directory
    save_to = UPLOAD_DIR / file.filename
    with open(save_to, 'wb') as f:
        f.write(data)
    #Convert the source points objects into [X,Y] format
    source_points = []
    for p in points:
        x, y = p['x'], p['y']
        source_points.append((x, y))
    #Process the video
    #Return the output path so it is viewable in the browser
    #Return risk results for the table
    output_path, risk_results = process_video(save_to, source_points)
    video_url = f"http://localhost:8000/{output_path}"
    return{"output_path": video_url,
           "risk_results": risk_results}