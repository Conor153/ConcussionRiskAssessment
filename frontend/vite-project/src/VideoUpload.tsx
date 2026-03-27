import axios from "axios";
import { ChangeEvent, useEffect, useRef, useState } from "react";
import { Player } from "./App";

//Pass in setPlayer and setVideo functions from App.tsx
interface Props {
  setPlayer: (players: Player[]) => void;
  setVideo: (video: string) => void;
}

//Create a Points interface to store X and Y co-ordinates for source matrix
interface Points {
  x: number;
  y: number;
}

function VideoUpload(props: Props) {
  //State to store uploaded file
  const [file, setFile] = useState<File | null>();
  //State to store canvas points
  const [points, setPoints] = useState<Points[]>([]);
  //State to set canvas to visible
  const [hidden, setHidden] = useState(true);
  const [error, setError] = useState("");
  //References to video, input and canvas objects
  const idFile = useRef<HTMLInputElement>(null);
  const idCanvas = useRef<HTMLCanvasElement>(null);
  const idVideo = useRef<HTMLVideoElement>(null);

  //Function to retrieve file information as it is uploaded
  const handleVideo = (event: ChangeEvent<HTMLInputElement>) => {
    const target = event.target as HTMLInputElement & { files: FileList };
    //Check to ensure file is of type video
    if (target.files[0].type.startsWith("video/")) {
      console.log(target.files[0]);
      setFile(target.files[0]);

      if (idVideo.current != null && idCanvas.current != null) {
        const canvas = idCanvas.current;
        const video = idVideo.current;
        const videoURL = URL.createObjectURL(target.files[0]);
        video.src = videoURL;
        idVideo.current.onloadedmetadata = () => {
          video.currentTime = 0;
        };
        idVideo.current.onseeked = () => {
          setHidden(false);
          console.log(video.videoWidth, video.videoHeight);
          const ctx = canvas.getContext("2d");
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          if (ctx != null) {
            ctx.drawImage(video, 0, 0);
          }
        };
      }
    }
  };
  //Post video to backend for processing
  const handleVideoUpload = (event: React.MouseEvent) => {
    if (!file || points.length!=4) return;
    const formData = new FormData();
    formData.append("file", file);
    formData.append("points", JSON.stringify(points));

    try {
      axios
        .post(`http://localhost:8000/`, formData)
        .then((response) => {
          console.log(response.data);
        })
        .catch((error) => {
          console.log(error.message);
        });
    } catch {}
  };

  //Function to handle selected co-ordinates
  const handlePoints = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const newPoints = [...points];
    const ctx = idCanvas.current?.getContext("2d");
    const video = idVideo.current;
    if (ctx != null && video != null) {
      ctx.fillStyle = "#FF5F1F";
      ctx.fillRect(e.nativeEvent.offsetX, e.nativeEvent.offsetY, 5, 5);
      const co_ordinate = {
        x: e.nativeEvent.offsetX,
        y: e.nativeEvent.offsetY,
      };
      newPoints.push(co_ordinate);
      if (newPoints.length > 4) {
        newPoints.shift();
        ctx.drawImage(video, 0, 0)
        setPoints(newPoints);
        newPoints.forEach((p) => {
          ctx.fillRect(p.x, p.y, 5, 5);
        });
      } else {
        setPoints(newPoints);
      }
      console.log(newPoints);
    }
  };

  //Return component
  return (
    <div>
      <form>
        <input
          ref={idFile}
          id="videoUpload"
          type="file"
          className="form-control"
          placeholder="Upload American Football Video"
          onChange={handleVideo}
        />
        <button onClick={handleVideoUpload} type="button" className="btn">
          Upload
        </button>
        <video ref={idVideo} hidden src=""></video>
        <canvas ref={idCanvas} hidden={hidden} onClick={handlePoints}></canvas>
      </form>
    </div>
  );
}

export default VideoUpload;
