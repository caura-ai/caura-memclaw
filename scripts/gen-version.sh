#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VERSION=$(cat "$ROOT/VERSION" | tr -d '[:space:]')
cat > "$ROOT/plugin/src/version.ts" <<EOF
// Auto-generated from /VERSION — do not edit
export const PLUGIN_VERSION = "$VERSION";
EOF
