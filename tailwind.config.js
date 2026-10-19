/** Tailwind scans these files for class names; dynamic classes live as literals in the templates. */
module.exports = {
  content: ["./app/templates/**/*.html", "./app/static/*.js"],
  theme: { extend: {} },
  plugins: [],
};
