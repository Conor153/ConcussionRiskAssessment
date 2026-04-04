/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        primary: "#16425B",
        secondary: "#2F6690",
        tertiary: "#E5F3FD",
        cyan: "#82C0CC",
        grey: "#EDE7E3",
        orange: "#FFA62B",
        green: "#4CBB17",
        yellow: "#FFBF00",
        red: "#EE2400",
        hover: "#E08300"
      },
    },
  },
  plugins: [],
};
