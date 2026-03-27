interface Props {
  video: string;
}

function VideoPlayer(props: Props) {
    return (
        <video width="320" height="240" controls>
        <source
          className="videoPlayer"
          src={props.video}
          type="video/mp4"
        ></source>
      </video>
    )
}

export default VideoPlayer;