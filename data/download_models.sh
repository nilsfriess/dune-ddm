#!/bin/bash
# Downloads the velocity models of the Helmholtz examples into the directory of this script and checks them. Files that
# are already there are only checked. The .ini files that describe the models are in the repository.
#
#   marmousi_vp.bin  the original Marmousi P-wave velocity model, see marmousi_vp.ini
#
# Usage: ./download_models.sh
set -euo pipefail
cd "$(dirname "$0")"

# file, URL, SHA-256 of the file
download() {
  local file=$1 url=$2 sha256=$3
  if [ ! -f "$file" ]; then
    echo "Downloading $file"
    curl --fail --location --output "$file.part" "$url"
    mv "$file.part" "$file"
  fi
  echo "$sha256  $file" | sha256sum --check --quiet || { echo "$file has the wrong checksum, delete it and run this script again" >&2; exit 1; }
  echo "$file is OK"
}

download marmousi_vp.bin "https://zenodo.org/records/16114161/files/marmousi_vp.bin?download=1" f4302792e84bb7ddbdc9d4d0f963b9df32d9f648960084884127e186119b99dd
