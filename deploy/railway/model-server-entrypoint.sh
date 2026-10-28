#!/bin/sh
# Fetches the configured bundle into the volume, then serves it.
set -eu
# A command that is not a server flag runs instead of the server, so the
# same image serves as the drift-check cron job: `python -m ml.monitoring.drift`.
if [ $# -gt 0 ] && [ "${1#--}" = "$1" ]; then
  exec "$@"
fi
if [ -n "${PODFIND_BUNDLE_ID:-}" ]; then
  python -m ml.serving.bundlestore fetch --bundle-id "$PODFIND_BUNDLE_ID" --dest "${PODFIND_BUNDLE_DIR:-/data/bundles}"
  export PODFIND_BUNDLE="${PODFIND_BUNDLE_DIR:-/data/bundles}/$PODFIND_BUNDLE_ID"
fi
exec python -m ml.serving.server --host :: --port "${PORT:-8081}" "$@"
