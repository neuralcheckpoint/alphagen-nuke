#!/usr/bin/env sh
# Usage: ./medratools.sh install | models | backend | check
set -eu
uv_bin=$(command -v uv || true)
if [ -z "$uv_bin" ]; then
    uv_bin="$HOME/.local/bin/uv"
    [ -x "$uv_bin" ] || curl -LsSf https://astral.sh/uv/install.sh | sh
fi
exec "$uv_bin" run --no-project --python 3.12 "$(dirname "$0")/medratools.py" "$@"
