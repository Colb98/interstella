#!/usr/bin/env bash
# Headless render of one scene, for checking work without a browser window.
#
#   tools/snap.sh <sceneId> [out.png] [extra query]
#
# extra query examples:  "cam=3,80,40"  (distance,polar°,azimuth° around the scene target)
#                        "t=120"        (simulation time)
#                        "params=sunAngle:120,craters:0.8"
#                        "ui=0"         (hide the HUD)   "q=low|med|high"
#
# Prints the page's status JSON: {"ok":bool,"scene":id,"errors":[...],"hud":[...]}.
# Shader compile errors include the offending source lines. Writes the PNG to out.png.
set -u
SCENE=${1:?scene id}
OUT=${2:-snap-$SCENE.png}
EXTRA=${3:-}
PORT=${SNAP_PORT:-5317}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
EDGE="/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
[ -x "$EDGE" ] || EDGE="/c/Program Files/Google/Chrome/Application/chrome.exe"

if ! curl -s -o /dev/null "http://127.0.0.1:$PORT/index.html"; then
  (cd "$ROOT" && nohup python -m http.server "$PORT" --bind 127.0.0.1 </dev/null >/dev/null 2>&1 &)
  for _ in $(seq 1 40); do curl -s -o /dev/null "http://127.0.0.1:$PORT/index.html" && break; python -c "import time; time.sleep(0.25)"; done
fi

PROFILE=$(mktemp -d)
URL="http://127.0.0.1:$PORT/index.html?snap=1&scene=$SCENE&$EXTRA"
FLAGS=(--headless=new --user-data-dir="$(cygpath -w "$PROFILE")" --no-first-run --disable-extensions
       --use-angle=d3d11 --ignore-gpu-blocklist --enable-webgl --window-size=1280,720 --hide-scrollbars
       --virtual-time-budget=45000 --run-all-compositor-stages-before-draw)

# retry: with several authors snapping at once the page occasionally misses its time budget
for attempt in 1 2 3; do
  STATUS=$(timeout 150 "$EDGE" "${FLAGS[@]}" --dump-dom "$URL" 2>/dev/null | tr -d '\r' | sed -n 's:.*<pre id="snap"[^>]*>\(.*\)</pre>.*:\1:p' | sed 's/&quot;/"/g; s/&lt;/</g; s/&gt;/>/g; s/&amp;/\&/g')
  if [ -n "$STATUS" ] && ! echo "$STATUS" | grep -q 'not found (loaded:'; then break; fi
done
timeout 150 "$EDGE" "${FLAGS[@]}" --screenshot="$(cygpath -w "$(realpath -m "$OUT")")" "$URL" >/dev/null 2>&1
rm -rf "$PROFILE"
if [ -z "$STATUS" ]; then
  echo '{"ok":false,"errors":["no status: the page did not finish within the time budget (module syntax error? check the browser console or run: node --check js/scenes/<file>.js)"]}'
else
  echo "$STATUS"
fi
echo "screenshot: $OUT"
