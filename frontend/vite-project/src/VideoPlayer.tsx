interface Props {
  video: string;
  //hidden: true
}

function VideoPlayer(props: Props) {
  return <video src={props.video} width="320" height="240" controls></video>;
}

export default VideoPlayer;
