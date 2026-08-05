#!/usr/bin/env bash
#
# Serve the studio and the 3D viewer.
#
#   bash start.sh                    newest run under runs/ (or ./run), port 8090
#   bash start.sh runs/JhasiRoad     a specific run
#   bash start.sh runs/JhasiRoad 9000
#   bash start.sh --dev              viewer with hot reload, API forced to :8000
#
# Run setup.sh once first.
#
# WHY IT PICKS A RUN AT ALL
# `pos serve` needs a run directory to show at / and that directory must exist --
# but the studio can browse every run under runs/ via ?run=<name> regardless of
# which one is named here. So the argument only chooses the DEFAULT view, and a
# missing ./run must not be a startup failure on a fresh clone.

set -euo pipefail
cd "$(dirname "$0")"

DEV=0
ARGS=()
for a in "$@"; do
  case "$a" in
    --dev) DEV=1 ;;
    *) ARGS+=("$a") ;;
  esac
done

RUN="${ARGS[0]:-}"
PORT="${ARGS[1]:-8090}"

# Vite's proxy target is hardcoded to 127.0.0.1:8000 in viewer/vite.config.ts, so in
# dev mode the API MUST listen there or every /api call from :5173 404s.
if [ "$DEV" -eq 1 ]; then
  PORT=8000
fi

# ------------------------------------------------------------------- which run?
if [ -z "$RUN" ]; then
  if [ -f run/manifest.json ]; then
    RUN=run
  else
    # Newest run that actually has a manifest. A half-written directory would make
    # `pos serve` exit on a missing file instead of serving the others.
    RUN="$(find runs -maxdepth 2 -name manifest.json -printf '%T@ %h\n' 2>/dev/null \
           | sort -rn | head -1 | cut -d' ' -f2-)"
  fi
fi

if [ -z "$RUN" ] || [ ! -d "$RUN" ]; then
  mkdir -p runs uploads
  cat <<'EOF'

  No processed run yet, so there is nothing to show at / -- but the STUDIO works:
  start it and upload a video + GPX (+ an optional findings CSV) there.

  Or build the committed synthetic sample first:
    uv run python scripts/make_sample.py
    uv run pos run --video samples/road/road.mp4 --gpx samples/road/track.gpx \
        --truth samples/road/truth.json --out run

EOF
  # Serving `runs` itself: it has no manifest, so / shows the viewer's own "no run"
  # screen while /studio stays fully usable. Better than refusing to start.
  RUN=runs
fi

# --------------------------------------------------------------- viewer bundle
# Rebuild only when a source file is newer than the bundle. Vite takes ~10 s, and the
# bundle filename is content-hashed, so rebuilding on every start would also mean a
# hard-refresh on every start.
if [ ! -f viewer/dist/index.html ]; then
  echo "  viewer not built yet -- building (first run only)"
  ( cd viewer && npm install && npm run build )
elif [ -n "$(find viewer/src viewer/index.html -newer viewer/dist/index.html 2>/dev/null | head -1)" ]; then
  echo "  viewer sources changed since the last build -- rebuilding"
  ( cd viewer && npm run build )
  echo "  bundle name changed: hard-refresh the browser (Ctrl+Shift+R)"
fi

# ----------------------------------------------------------------------- serve
if [ "$DEV" -eq 1 ]; then
  echo
  echo "  DEV MODE"
  echo "    viewer  http://127.0.0.1:5173/          hot reload, proxies to :8000"
  echo "    studio  http://127.0.0.1:8000/studio"
  echo
  # API in the background, Vite in the foreground: Ctrl-C stops Vite and the trap
  # takes the API down with it, rather than leaving :8000 held by an orphan.
  uv run pos serve --run "$RUN" --studio --port "$PORT" &
  API_PID=$!
  trap 'kill "$API_PID" 2>/dev/null || true' EXIT INT TERM
  ( cd viewer && npm run dev )
  exit 0
fi

echo
echo "  run     $RUN"
echo "  studio  http://127.0.0.1:$PORT/studio     <- upload here"
echo "  viewer  http://127.0.0.1:$PORT/"
echo

exec uv run pos serve --run "$RUN" --studio --port "$PORT"
