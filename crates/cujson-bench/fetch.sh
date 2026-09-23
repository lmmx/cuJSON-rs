#!/usr/bin/env sh
# Download the benchmark input: one file of philippesaade/wikidata, pinned to the
# revision the wikidata pipeline processed, and check its SHA-256.
set -eu
REV=064f404b6bf7d9ed45d0b1c7bde29dc8b0b02bef
FILE=chunk_0-00283-of-00546.parquet
SHA=4c21bb311a44c8fc1fdb0b44bae3ef3ffa6450c32a268e08e97dac33499246e4
DEST="$(dirname "$0")/data/$FILE"
mkdir -p "$(dirname "$DEST")"
curl -fSL -o "$DEST" "https://huggingface.co/datasets/philippesaade/wikidata/resolve/$REV/data/$FILE"
echo "$SHA  $DEST" | sha256sum -c -
