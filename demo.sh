#!/usr/bin/env bash
# Walks through the environment. Run inside an activated environment:
#   flox activate -s -- ./demo.sh
# The -s starts the services declared in the manifest; activation alone is
# deliberately inert.
set -euo pipefail

PORT="${EMBED_PORT:-8000}"
BASE="localhost:$PORT"

# The strings we embed. Two are about the same thing, one is not — so the
# cosine similarities at the end have an obvious right answer.
read -r -d '' PAYLOAD <<'JSON' || true
{"input":["the ferry from Southworth","a boat to Seattle","tire rotation"]}
JSON

rule() { printf '\n── %s %s\n' "$1" "$(printf '─%.0s' $(seq 1 $((66 - ${#1}))))"; }

wait_for_api() {
  # Cold start has to load the model into memory; on a slow machine with an
  # empty cache that includes downloading it first.
  for _ in $(seq 1 150); do
    curl -sf "$BASE/healthz" >/dev/null 2>&1 && return 0
    sleep 2
  done
  echo "api never came up on $BASE — service log follows:" >&2
  flox services logs api 2>&1 | tail -20 >&2
  exit 1
}

# ---------------------------------------------------------------------------
rule "Part 1 — the service is already up"
cat <<EOF
Nothing was built and no container was started. 'flox activate -s' brought up
the API as a supervised service, so this GET should answer immediately. The
'dimensions' field proves the model is loaded in memory, not lazy.

  GET http://$BASE/healthz
EOF
echo
wait_for_api
curl -s "$BASE/healthz" | jq

# ---------------------------------------------------------------------------
rule "Part 1a — a real embedding, not a stub"
cat <<EOF
POSTing three strings to the model. Two are about crossing Puget Sound and one
is about a car, which is what makes the vectors worth looking at.

  POST http://$BASE/embed
  content-type: application/json
EOF
echo "$PAYLOAD" | jq
echo
echo "  input strings:"
echo "$PAYLOAD" | jq -r '.input[] | "    • \(.)"'
echo
echo "  response (384 dims per string, first 5 floats of vector 0):"
RESP=$(curl -s "$BASE/embed" -H 'content-type: application/json' -d "$PAYLOAD")
echo "$RESP" | jq '{model, dimensions, count: (.embeddings | length)}'
echo "$RESP" | jq -c '.embeddings[0][:5]'

echo
echo "  cosine similarity — the model actually understands the strings:"
# Passed via the environment, not a pipe: the heredoc below already owns stdin.
RESP="$RESP" python - <<'PY'
import json, os
d = json.loads(os.environ["RESP"])
v = d["embeddings"]
labels = ["ferry from Southworth", "boat to Seattle", "tire rotation"]


def cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb)


for i in range(len(v)):
    for j in range(i + 1, len(v)):
        print(f"    {labels[i]:24s} ↔ {labels[j]:24s}  {cos(v[i], v[j]):.3f}")
PY

# ---------------------------------------------------------------------------
rule "Part 2 — one lockfile, one dependency system"
cat <<'EOF'
There is no requirements.txt here, no venv, and no pip install. The
interpreter, the Python packages, and the native libraries underneath them are
all resolved by Flox and pinned by hash in the same manifest.lock.

EOF
echo "  declared in the manifest, pinned by hash in manifest.lock:"
flox list | sed 's/^/    /'
echo
echo "  and what actually got imported:"
python - <<'PY'
import sys
import fastapi, pydantic, sentence_transformers, torch
print(f"    python                {sys.version.split()[0]}")
print(f"    sentence-transformers {sentence_transformers.__version__}")
print(f"    torch                 {torch.__version__}")
print(f"    fastapi               {fastapi.__version__}")
print(f"    pydantic              {pydantic.VERSION}")
PY


# ---------------------------------------------------------------------------
rule "Part 3 — supervision comes with the manifest"
cat <<'EOF'
Declared in [services.api], eight words of TOML. No Compose file, no
supervisord. This process exits when the last activation does, so there is no
orphaned uvicorn holding port 8000 after you close the terminal.

EOF
flox services status | sed 's/^/    /'
echo
