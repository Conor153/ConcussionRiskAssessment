import { useState } from "react";
import "./index.css";
import Footer from "./Footer";
import Header from "./Header";
import VideoUpload from "./VideoUpload";
import VideoPlayer from "./VideoPlayer";
import Table from "./Table";

//Interfac eto store risk Results
export interface Results {
  track_id: number;
  g_force: number;
  angular_acceleration: number;
  risk: string;
}

function App() {
  //States to store video and results so that they are accessible in other components
  const [video, setVideo] = useState<string>("");
  const [results, setResults] = useState<Results[]>([]);
  const [videoPlayer, setvideoPlayerVisibility] = useState(true);
  const [table, setTableVisibility] = useState(true);

  return (
    <div className="min-h-screen bg-tertiary flex flex-col">
      <Header />
      <div className="max-w-l w-full mt-[1%] mx-auto p-4">
        <div className="VideoUpload py-4">
          <VideoUpload setVideo={setVideo} setResults={setResults} setTableVisibility={setTableVisibility} setvideoPlayerVisibility={setvideoPlayerVisibility}/>
        </div>

        <div className="VideoPlayer py-4">
          <VideoPlayer video={video} videoPlayer={videoPlayer}/>
        </div>

        <div className="Results py-4">
          <Table results={results} table={table}/>
        </div>
      </div>
      <Footer />
    </div>
  );
}

export default App;
