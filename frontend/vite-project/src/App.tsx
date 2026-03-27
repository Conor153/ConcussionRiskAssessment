import { useRef, useState } from 'react'
import './output.css'
import Footer from "./Footer";
import Header from "./Header";
import VideoUpload from "./VideoUpload"
import VideoPlayer from "./VideoPlayer"
import Table from './Table';

export interface Player {
  id: number;
  g_force: number;
  angular_acceleration: number;
  risk: string;
}

function App() {
  const [video, setVideo] = useState<string>("");
  const [players, setPlayer] = useState<Player[]>([]);

  return (
    <div className="App">
      <Header/>
      <div className="VideoUpload">
      <VideoUpload setVideo={setVideo} setPlayer={setPlayer} />
      </div>

      <div className="VideoPlayer">
        <VideoPlayer video={video}/>
      </div>

      <div className="Results">
        <Table players={players}/>
      </div>
      <Footer/>
    </div>
    
  )
}

export default App
