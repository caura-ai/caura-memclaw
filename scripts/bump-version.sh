#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NEW_VERSION="${1:?Usage: bump-version.sh <version>}"

# 1. Write the source of truth
printf '%s\n' "$NEW_VERSION" > "$ROOT/VERSION"

# 2. Update plugin/package.json (also updates package-lock.json)
cd "$ROOT/plugin"
npm version "$NEW_VERSION" --no-git-tag-version --allow-same-version

# 3. Update plugin/openclaw.plugin.json
python3 -c "
import json, pathlib
p = pathlib.Path('$ROOT/plugin/openclaw.plugin.json')
d = json.loads(p.read_text())
d['version'] = '$NEW_VERSION'
p.write_text(json.dumps(d, indent=2) + '\n')
"

# 4. Regenerate plugin/src/version.ts
bash "$ROOT/scripts/gen-version.sh"

echo "Bumped to $NEW_VERSION"
