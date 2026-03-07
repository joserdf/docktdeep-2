#!/usr/bin/env bash
set -euo pipefail

echo "Building Docker image..."
docker build -t docktdeep:latest "$(dirname "$0")"

echo "Installing docktdeep to /usr/local/bin/..."
sudo install -m 755 "$(dirname "$0")/scripts/docktdeep" /usr/local/bin/docktdeep

echo "Done. Run 'docktdeep --help' to get started."
