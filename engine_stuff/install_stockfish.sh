#!/usr/bin/env bash
# Run from any folder: bash engine_stuff/install_stockfish.sh
# Linux counterpart of install_stockfish.ps1. Installs only inside this project
# (no sudo). Keep the official license/source files. Alternative: apt install stockfish.
set -euo pipefail
case "$(uname -m)" in
    x86_64) asset=stockfish-linux-x86-64-universal.tar.gz ;;
    aarch64|arm64) asset=stockfish-linux-arm64-universal.tar.gz ;;
    *) echo "No official Stockfish 19 Linux build for $(uname -m)" >&2; exit 1 ;;
esac
destination="$(cd "$(dirname "$0")" && pwd)/bin"
mkdir -p "$destination"
release=$(curl -fsSL https://api.github.com/repos/official-stockfish/Stockfish/releases/tags/sf_19)
read -r url digest < <(python3 -c '
import json, sys
asset = [a for a in json.load(sys.stdin)["assets"] if a["name"] == sys.argv[1]]
if not asset: sys.exit("Official release has no " + sys.argv[1])
print(asset[0]["browser_download_url"], asset[0].get("digest") or "")' "$asset" <<< "$release")
download="$destination/$asset"
curl -fsSL "$url" -o "$download"
hash=$(sha256sum "$download" | cut -d' ' -f1)
if [ "$digest" != "sha256:$hash" ]; then
    rm -f "$download"
    echo "Archive does not match the official SHA256 digest; installation stopped" >&2
    exit 1
fi
tar -xzf "$download" -C "$destination"
# Delete just the verified download, not the installed engine/license/source.
rm "$download"
echo "Installed sf_19 in $destination; SHA256 $hash"
