import axios from "axios";
import { ChangeEvent, useRef, useState } from "react";
import { Results } from "./App";

//Pass in setPlayer and setVideo functions from App.tsx
interface Props {
  setResults: (results: Results[]) => void;
  setVideo: (video: string) => void;
  setTableVisibility: (table: boolean) => void;
  setvideoPlayerVisibility: (videoPlayer: boolean) => void;
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

  //States to chnage visibility of components
  const [hidden, setHidden] = useState(true);
  const [disable, setDisable] = useState(true);
  const [upload, setUploadVisibility] = useState(false);
  const [gif, setGifVisability] = useState(true);

  //References to video, input and canvas objects
  const idFile = useRef<HTMLInputElement>(null);
  const idCanvas = useRef<HTMLCanvasElement>(null);
  const idVideo = useRef<HTMLVideoElement>(null);

  //Error state
  const [error, setError] = useState("");

  //Function to retrieve file information as it is uploaded
  const handleVideo = (event: ChangeEvent<HTMLInputElement>) => {
    const target = event.target as HTMLInputElement & { files: FileList };

    //Check to ensure file is of type video
    if (target.files[0].type.startsWith("video/")) {
      setFile(target.files[0]);

      //Set component visibility
      props.setvideoPlayerVisibility(true);
      props.setTableVisibility(true);
      setUploadVisibility(true);

      //Clear Source co-ordinates and video player
      clearPoints();
      props.setVideo("");

      //Set up Canvas with frame 1 of uploaded video
      if (idVideo.current != null && idCanvas.current != null) {
        const canvas = idCanvas.current;
        const video = idVideo.current;
        const videoURL = URL.createObjectURL(target.files[0]);
        video.src = videoURL;
        idVideo.current.onloadedmetadata = () => {
          video.currentTime = 0;
        };
        idVideo.current.onseeked = () => {
          //Unhide canvas and enable drawing
          setHidden(false);
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
    //Ensure files exists and 4 co-ordinates selected
    if (!file || points.length != 4) return;
    //Hide canvas and display loading gif
    setHidden(true);
    setGifVisability(false);

    //Construct post form to store video and source-points
    const formData = new FormData();
    formData.append("file", file);
    formData.append("points", JSON.stringify(points));
    axios
      .post(`http://localhost:8000/upload/`, formData)
      .then((response) => {
        //Set visablity of table, video player and gif
        //Set response values to the video player and the table
        setGifVisability(true);
        props.setVideo(response.data.output_path);
        props.setvideoPlayerVisibility(false);
        props.setResults(response.data.risk_results);
        props.setTableVisibility(false);
        setUploadVisibility(false);

        // console.log(response.data.output_path);
        // console.log(response.data.risk_results);
      })
      .catch((error) => {
        console.log(error.message);
      });
  };

  //Function to handle selected co-ordinates
  const handlePoints = (e: React.MouseEvent<HTMLCanvasElement>) => {
    //Assign existing co-ordinate list
    const newPoints = [...points];
    const canvas = idCanvas.current;
    const video = idVideo.current;

    //Check to ensure canvas is not null
    if (canvas != null && video != null) {
      const ctx = canvas.getContext("2d");
      const rect = canvas.getBoundingClientRect();

      //Scale co-ordinates on the canvas to align with video width
      //to ensure accurate co-ordinate selection
      const scalewidth = canvas.width / rect.width;
      const scaleheight = canvas.height / rect.height;
      const co_ordinate = {
        x: Math.round(e.nativeEvent.offsetX * scalewidth),
        y: Math.round(e.nativeEvent.offsetY * scaleheight),
      };

      //Display points on canvas and add to the points list
      if (ctx != null) {
        ctx.fillStyle = "orange";
        ctx.fillRect(co_ordinate.x, co_ordinate.y, 8, 8);
        newPoints.push(co_ordinate);
        //If greater than 4 remove the first co-ordnate
        if (newPoints.length > 4) {
          newPoints.shift();
          ctx.drawImage(video, 0, 0);
          setPoints(newPoints);
          newPoints.forEach((p) => {
            ctx.fillRect(p.x - 4, p.y - 4, 8, 8);
          });
        } else {
          setPoints(newPoints);
          //If 4 points are selected undisable button
          if (newPoints.length == 4) setDisable(false);
        }
      }
    }
  };

  //Change the selected canvas video
  //Empty the points
  const changeVideo = (event: React.MouseEvent) => {
    const newPoints = [...points];
    for (let i = 0; i < points.length; i++) {
      newPoints.pop();
    }
    setPoints(newPoints);
    setHidden(true);
    setDisable(true);
    clearPoints();
  };

  //Empty all selected points from canvas
  const clearPoints = () => {
    const ctx = idCanvas.current?.getContext("2d");
    const video = idVideo.current;
    const newPoints = [...points];
    for (let i = 0; i < points.length; i++) {
      newPoints.pop();
    }
    setPoints(newPoints);
    setDisable(true);
    if (ctx != null && video != null) {
      ctx.drawImage(video, 0, 0);
    }
  };

  //Return Video Upload
  return (
    <div
      className={`bg-secondary mx-auto ${!hidden ? "max-w-fit px-4" : "max-w-md"} rounded-xl`}
    >
      <div
        className={`${upload ? "hidden" : "max-w-md w-full mt-auto mx-auto text-center rounded-md flex flex-col justify-center p-4"} `}
      >
        <label className="bg-white w-auto p-3 hover:bg-tertiary rounded-md flex flex-col border-dashed border-4 border-orange">
          <input
            ref={idFile}
            id="videoUpload"
            type="file"
            placeholder="Upload American Football Video"
            onChange={handleVideo}
            hidden
          />
          <img className="h-32 w-auto p-4" src="src/assets/video-icon.svg" />
          <div>
            <p className="text-lg">
              Click here to add your American football video
            </p>
          </div>
        </label>
      </div>
      <div
        className={`${gif ? "hidden" : "max-w-md w-full mt-auto mx-auto text-center rounded-md flex flex-col justify-center p-4"} `}
      >
        <h3 className="text-4xl font-bold text-white text-center p-2">
          {" "}
          Video Processing
        </h3>
        <img
          hidden={gif}
          className="w-1/2 mx-auto p-4"
          src="src/assets/loading.gif"
        />
        <p className="text-lg font-bold text-white text-center pt-2">
          Processing hang on for a couple of seconds
        </p>
      </div>

      <video ref={idVideo} hidden src=""></video>
      <div className={`${hidden ? "hidden" : "flex flex-col justify-center"} `}>
        <h2 className="text-4xl font-bold text-white text-center pt-2">
          Select 4 source co-ordinates | Points Clicked: {points.length} /4
        </h2>
        <canvas
          ref={idCanvas}
          hidden={hidden}
          onClick={handlePoints}
          className="border-solid border-4 mt-4 border-white rounded-xl cursor-crosshair"
        ></canvas>
        <div className="w-full justify-between flex">
          <button
            hidden={hidden}
            onClick={handleVideoUpload}
            type="button"
            disabled={disable}
            className={`${disable ? "bg-grey" : "bg-orange hover:bg-hover"} w-full m-3 font-semibold rounded-md text-lg`}
          >
            Upload
          </button>
          <button
            hidden={hidden}
            onClick={changeVideo}
            type="button"
            className="bg-orange hover:bg-hover w-full m-3 p-3 font-semibold rounded-md text-lg"
          >
            Change Video
          </button>
          <button
            hidden={hidden}
            onClick={clearPoints}
            type="button"
            className="bg-orange hover:bg-hover w-full m-3 p-3 font-semibold rounded-md text-lg"
          >
            Clear Points
          </button>
        </div>
      </div>
    </div>
  );
}

export default VideoUpload;
