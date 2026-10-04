/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        canvas: "#E6EBF0",
        paper: "#FBFCFD",
        ink: "#14213A",
        muted: "#5B6B82",
        line: "#C9D2DD",
        teal: { DEFAULT: "#0B7F86", dark: "#086369", soft: "#D8EEEF" },
        amber: { DEFAULT: "#D9822B", soft: "#F8E6D0" },
        steel: "#6B7A90",
        cobalt: "#3D5A9B",
        danger: "#B3392F",
      },
      fontFamily: { sans: ['"IBM Plex Sans"', "system-ui", "sans-serif"] },
    },
  },
  plugins: [],
};
