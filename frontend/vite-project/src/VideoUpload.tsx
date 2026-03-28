import axios from "axios";
import { ChangeEvent, useRef, useState } from "react";
import { Results } from "./App";

//Pass in setPlayer and setVideo functions from App.tsx
interface Props {
  setResults: (results: Results[]) => void;
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
    if (!file || points.length != 4) return;
    setHidden(true);
    const formData = new FormData();
    formData.append("file", file);
    formData.append("points", JSON.stringify(points));
    axios
      .post(`http://localhost:8000/upload/`, formData)
      .then((response) => {
        console.log(response.data.output_path);
        props.setVideo(response.data.output_path);
        console.log(response.data.risk_results);
        props.setResults(response.data.risk_results);
      })
      .catch((error) => {
        console.log(error.message);
      });
  };

  //Function to handle selected co-ordinates
  const handlePoints = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const newPoints = [...points];
    const ctx = idCanvas.current?.getContext("2d");
    const video = idVideo.current;
    if (ctx != null && video != null) {
      ctx.fillStyle = "#FF5F1F";
      ctx.fillRect(e.nativeEvent.offsetX - 4, e.nativeEvent.offsetY - 4, 8, 8);
      const co_ordinate = {
        x: e.nativeEvent.offsetX,
        y: e.nativeEvent.offsetY,
      };
      newPoints.push(co_ordinate);
      if (newPoints.length > 4) {
        newPoints.shift();
        ctx.drawImage(video, 0, 0);
        setPoints(newPoints);
        newPoints.forEach((p) => {
          ctx.fillRect(p.x - 4, p.y - 4, 8, 8);
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
