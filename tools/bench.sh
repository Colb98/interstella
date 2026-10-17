#!/usr/bin/env bash
# Real-GPU frame time per scene (headless Edge, no virtual time). Usage: tools/bench.sh [q=med] [id ...]
Q=${1:-med}; shift || true
PORT=${SNAP_PORT:-5317}
EDGE="/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
IDS=${@:-blackhole wormhole pulsar magnetar sun red-dwarf red-supergiant blue-supergiant white-dwarf wolf-rayet brown-dwarf venus earth mars moon jupiter saturn uranus neptune lava-world comet asteroid interstellar-object spiral-galaxy galaxy-collision emission-nebula planetary-nebula supernova-remnant dark-nebula}
for id in $IDS; do
  P=$(mktemp -d)
  R=$("$EDGE" --headless=new --user-data-dir="$(cygpath -w "$P")" --use-angle=d3d11 --ignore-gpu-blocklist --window-size=1920,1080 --virtual-time-budget=45000 --dump-dom "http://127.0.0.1:$PORT/index.html?snap=1&ui=0&q=$Q&bench=40&scene=$id" 2>/dev/null | grep -o '"benchMs":[0-9.null]*,"res":\[[0-9,]*\]')
  rm -rf "$P"
  echo "$id $R"
done
