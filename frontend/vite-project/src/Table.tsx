//Import results interface
import { Results } from "./App";

// Props  to store risk results and table visibility
interface Props {
  results: Results[];
  table: boolean;
}

function Table(props: Props) {
  //Filter the results into 3 arrays based on returned risk colour
  const greenRisk = props.results.filter((result) => result.risk === "GREEN");
  const yellowRisk = props.results.filter((result) => result.risk === "YELLOW");
  const redRisk = props.results.filter((result) => result.risk === "RED");

  return (
    <div
      hidden={props.table}
      className="max-w-7xl w-full mx-auto p-5 justify-center bg-secondary rounded-xl"
    >
      <h2
        hidden={props.table}
        className="text-4xl font-bold text-white text-center p-2"
      >
        Concussion Risk Assessment Results
      </h2>
      <table
        hidden={props.table}
        className="bg-grey  w-full table-auto border-collapse border-solid border-4 border-primary"
      >
        <thead>
          <tr>
            <th className="bg-green border border-primary p-3"></th>
            <th className="bg-yellow border border-primary p-3"></th>
            <th className="bg-red border border-primary p-3"></th>
          </tr>
          <tr>
            <th className="text-xl font-bold border border-primary p-3">
              Green
            </th>
            <th className="text-xl font-bold border border-primary p-3">
              Yellow
            </th>
            <th className="text-xl font-bold border border-primary p-3">Red</th>
          </tr>
          <tr>
            <th className="p-3 text-lg border border-primary font-bold">
              G-Force &lt;49G
            </th>
            <th className="p-3 text-lg border border-primary font-bold ">
              G-Force &gt;49G and &lt;80G
            </th>
            <th className="p-3 text-lg border border-primary font-bold">
              G-Force &gt;80G
            </th>
          </tr>
          <tr>
            <th className="p-3 text-lg font-bold border border-primary border-solid">
              Angular Acceleration &lt;3512 Rad/s&sup2;
            </th>
            <th className="p-3 text-lg font-bold border border-primary border-solid">
              Angular Acceleration &gt;3512 & &lt;5875 Rad/s&sup2;
            </th>
            <th className="p-3 text-lg font-bold border border-primary border-solid">
              Angular Acceleration &gt;5875 Rad/s&sup2;
            </th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="w-1/3 align-top border border-primary p-0">
              <table className="w-full table-auto border-collapse">
                <thead>
                  <tr>
                    <th className="p-3 text-lg border border-primary">ID</th>
                    <th className="p-3 text-lg border border-primary">
                      G-Force
                    </th>
                    <th className="p-3 text-lg border border-primary">
                      Angular Acceleration
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {greenRisk.map((green, index) => (
                    <tr key={index}>
                      <td className="p-3 text-lg border border-primary">
                        {green.track_id}
                      </td>
                      <td className="p-3 text-lg border border-primary">
                        {green.g_force}
                      </td>
                      <td className="p-3 text-lg border border-primary">
                        {green.angular_acceleration}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </td>
            <td className="w-1/3 align-top border border-primary p-0">
              <table className="w-full table-auto border-collapse">
                <thead>
                  <tr>
                    <th className="p-3 text-lg border border-primary">ID</th>
                    <th className="p-3 text-lg border border-primary">
                      G-Force
                    </th>
                    <th className="p-3 text-lg border border-primary">
                      Angular Acceleration
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {yellowRisk.map((yellow, index) => (
                    <tr key={index}>
                      <td className="p-3 text-lg border border-primary">
                        {yellow.track_id}
                      </td>
                      <td className="p-3 text-lg border border-primary">
                        {yellow.g_force}
                      </td>
                      <td className="p-3 text-lg border border-primary">
                        {yellow.angular_acceleration}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </td>
            <td className="w-1/3 align-top border border-primary p-0">
              <table className="w-full table-auto border-collapse">
                <thead>
                  <tr>
                    <th className="p-3 text-lg border border-primary">ID</th>
                    <th className="p-3 text-lg border border-primary">
                      G-Force
                    </th>
                    <th className="p-3 text-lg border border-primary">
                      Angular Acceleration
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {redRisk.map((red, index) => (
                    <tr key={index}>
                      <td className="p-3 text-lg border border-primary">
                        {red.track_id}
                      </td>
                      <td className="p-3 text-lg border border-primary">
                        {red.g_force}
                      </td>
                      <td className="p-3 text-lg border border-primary">
                        {red.angular_acceleration}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}
export default Table;
