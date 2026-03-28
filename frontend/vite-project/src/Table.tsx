import { Results } from "./App";

interface Props {
  results: Results[];
}

function Table(props: Props) {
  //Filter the results into 3 arrays absed on returned risk colour
  const greenRisk = props.results.filter((result) => result.risk === "GREEN");
  const yellowRisk = props.results.filter((result) => result.risk === "YELLOW");
  const redRisk = props.results.filter((result) => result.risk === "RED");

  return (
    <table>
      <thead>
        <tr>
          <th>Green</th>
          <th>Yellow</th>
          <th>Red</th>
        </tr>
        <tr>
          <th>G-Force 49G</th>
          <th>G-Force 49G and 80G</th>
          <th>G-Force 80G</th>
        </tr>
        <tr>
          <th>Angular Acceleration 3512 Rad/s^2</th>
          <th>Angular Acceleration 3512 & 5875 Rad/s^2</th>
          <th>Angular Acceleration 5875 Rad/s^2</th>
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
                  {/* Loop through the green risk results and store them into the table */}
                  {greenRisk.map((green, index) => (
                    <tr key={index}>
                      <td>{green.track_id}</td>
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
                  {/* Loop through the yellow risk results and store them into the table */}
                  {yellowRisk.map((yellow, index) => (
                    <tr key={index}>
                      <td>{yellow.track_id}</td>
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
                  {/* Loop through the red risk results and store them into the table */}
                  {redRisk.map((red, index) => (
                    <tr key={index}>
                      <td>{red.track_id}</td>
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
