// Props  to store video URL and video player visibility
interface Props {
  video: string;
  videoPlayer: boolean;
}

function VideoPlayer(props: Props) {
  return (
    <div
      className={`bg-secondary mx-auto ${!props.videoPlayer ? "w-full p-4" : "max-w-md"} rounded-xl`}
    >
      <h2
        hidden={props.videoPlayer}
        className="text-4xl font-bold text-white text-center p-2"
      >
        Concussion Risk Assessment Footage
      </h2>
      <video
        className="border-4 w-full rounded-xl"
        hidden={props.videoPlayer}
        src={props.video}
        width="800"
        height="400"
        controls
      ></video>
    </div>
  );
}

export default VideoPlayer;
