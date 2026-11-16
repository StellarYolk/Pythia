#!/usr/bin/env bash
# Robust model download: HF downloads stall on this network, so fetch with curl
# (resume + retry) into a local model directory.
set -uo pipefail

REPO_ID="${1:-BAAI/bge-base-en-v1.5}"
DEST="${2:-/home/parth/pythia-data/models/bge-base-en-v1.5}"
FILES="${3:-config.json model.safetensors tokenizer.json tokenizer_config.json vocab.txt special_tokens_map.json}"

mkdir -p "$DEST"
for f in $FILES; do
    if [ -s "$DEST/$f" ]; then
        echo "  [skip] $f"
        continue
    fi
    URL="https://huggingface.co/$REPO_ID/resolve/main/$f"
    echo "  [get ] $f"
    for attempt in 1 2 3 4 5 6 7 8; do
        curl -sL --connect-timeout 20 --max-time 900 --retry 0 -C - -o "$DEST/$f" "$URL" && \
            [ -s "$DEST/$f" ] && break
        echo "    retry $attempt"
        sleep 5
    done
    ls -la "$DEST/$f" | awk '{print "    ", $5, $9}'
done
echo "done -> $DEST"
