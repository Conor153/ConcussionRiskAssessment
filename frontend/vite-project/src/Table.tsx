import { Player } from "./App";

interface Props {
  players: Player[];
}

function Table(props: Props) {

  const greenRisk = props.players.filter((player) => player.risk === "GREEN");
  const yellowRisk = props.players.filter((player) => player.risk === "YELLOW");
  const redRisk = props.players.filter((player) => player.risk === "RED");

  return (
    <table>
      <thead>
        <tr>
          <th>Green</th>
          <th>Yellow</th>
          <th>Red</th>
        </tr>
        <tr>
          <th>G-Force 49G | Angular Acceleration 3512 Rad/s^2</th>
          <th>G-Force 49G and 80G | Angular Acceleration 3512 & 5875 Rad/s^2</th>
          <th>G-Force 80G | Angular Acceleration 5875 Rad/s^2</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>
            <div>
              <table>
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>G-Force</th>
                    <th>Angular Acceleration</th>
                  </tr>
                </thead>
                <tbody>
                  {greenRisk.map((green, index) => (
                    <tr key={index}>
                      <td>{green.id}</td>
                      <td>{green.g_force}</td>
                      <td>{green.angular_acceleration}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </td>
          <td>
            <div>
              <table>
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>G-Force</th>
                    <th>Angular Acceleration</th>
                  </tr>
                </thead>
                <tbody>
                  {yellowRisk.map((yellow, index) => (
                    <tr key={index}>
                      <td>{yellow.id}</td>
                      <td>{yellow.g_force}</td>
                      <td>{yellow.angular_acceleration}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </td>
          <td>
            <div>
              <table>
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>G-Force</th>
                    <th>Angular Acceleration</th>
                  </tr>
                </thead>
                <tbody>
                  {redRisk.map((red, index) => (
                    <tr key={index}>
                      <td>{red.id}</td>
                      <td>{red.g_force}</td>
                      <td>{red.angular_acceleration}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </td>
        </tr>
      </tbody>
    </table>
  );
}
export default Table;
