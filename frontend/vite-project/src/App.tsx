import { useRef, useState } from "react";
import "./output.css";
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

  return (
    <div className="App">
      <Header />
      <div className="VideoUpload">
        <VideoUpload setVideo={setVideo} setResults={setResults} />
      </div>

      <div className="VideoPlayer">
        <VideoPlayer video={video} />
      </div>

      <div className="Results">
        <Table results={results} />
      </div>
      <Footer />
    </div>
  );
}

export default App;
