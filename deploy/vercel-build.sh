#!/usr/bin/env bash
#
# Vercel build. Produces public/ from the one real copy of the front end, so
# there is no second index.html to drift out of sync with templates/index.html.
#
# The only transformation is injecting the backend origin as a global, ahead of
# the app's own <script>. index.html reads it as:
#
#     const API_BASE = (window.XAI_API_BASE || "").replace(/\/+$/, "");
#
# so with nothing injected API_BASE is "" and every call is same-origin — which
# is exactly how the app behaves when FastAPI serves the page itself. Local runs
# are therefore untouched by any of this.

set -euo pipefail

if [[ -z "${XAI_API_BASE:-}" ]]; then
  cat >&2 <<'MSG'
XAI_API_BASE is not set.

It must be the origin of the running backend container, with no trailing path,
e.g. https://your-name-xai-decision-attribution.hf.space

Set it in Vercel under Settings -> Environment Variables (all environments),
then redeploy. Without it the deployed page would call its own static host for
/api/summarize and every request would 404.
MSG
  exit 1
fi

# Trailing slashes would produce //api/summarize once concatenated.
API_BASE="${XAI_API_BASE%/}"

case "$API_BASE" in
  https://*|http://localhost*|http://127.0.0.1*) ;;
  *)
    echo "XAI_API_BASE must start with https:// (got: $API_BASE)" >&2
    echo "A browser on an https:// page will refuse to call an http:// backend." >&2
    exit 1
    ;;
esac

echo "Building static front end -> public/  (API_BASE=$API_BASE)"

rm -rf public
mkdir -p public/static

cp static/logo.png public/static/logo.png

# Insert the global immediately before </head>, i.e. before the app's script at
# the bottom of <body>. awk rather than sed: the value is a URL full of slashes,
# and this avoids escaping it into an unreadable one-liner.
awk -v base="$API_BASE" '
  !injected && /<\/head>/ {
    printf "  <script>window.XAI_API_BASE=%c%s%c;</script>\n", 34, base, 34
    injected = 1
  }
  { print }
  END {
    if (!injected) {
      print "ERROR: no </head> found in templates/index.html; nothing injected." > "/dev/stderr"
      exit 1
    }
  }
' templates/index.html > public/index.html

# Fail loudly rather than shipping a page that silently calls the wrong origin.
if ! grep -q "window.XAI_API_BASE" public/index.html; then
  echo "ERROR: injection produced no XAI_API_BASE in public/index.html" >&2
  exit 1
fi

echo "Built public/index.html ($(wc -c < public/index.html) bytes) and public/static/logo.png"
