#!/bin/sh
# Rebuild app/static/app.css from the templates (run after changing Tailwind classes).
# Needs Node (npx). The built CSS is committed, so running the app does NOT need Node.
cd "$(dirname "$0")/.." || exit 1
npx --yes tailwindcss@3.4.16 -c tailwind.config.js -i app/static/src/input.css -o app/static/app.css --minify
