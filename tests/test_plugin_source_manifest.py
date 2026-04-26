"""Guards the two hardcoded plugin-file lists in core_api.routes.plugin.

A 2026-04-16 refactor added plugin/src/paths.ts + logger.ts but forgot to
register them in either the Python allow-list (`_plugin_files`) or the
bash `for srcfile in …` loop inside the install-script template. Every
fresh `curl … | bash` install broke with `TS2307: Cannot find module
'./paths.js'` until both lists were fixed on 2026-04-19.

This test keeps them in lockstep with `plugin/src/*.ts`.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from core_api.routes import plugin as plugin_mod


REPO_ROOT = Path(__file__).resolve().parent.parent
PLUGIN_SRC = REPO_ROOT / "plugin" / "src"


def _expected_source_files() -> set[str]:
    """Every .ts file the install script needs to list.

    Excludes only test files. `version.ts` is present on disk, served by
    /api/plugin-source, AND listed in the bash loop (the install script
    then overwrites it inline from the request's ``version`` parameter,
    but the fetch still happens for parity with the manifest).
    """
    return {
        p.name
        for p in PLUGIN_SRC.glob("*.ts")
        if not p.name.endswith(".test.ts")
    }


def test_python_allow_list_matches_plugin_src():
    """`_plugin_files` (serves `/api/plugin-source?file=…`) must cover plugin/src."""
    actual = set(plugin_mod._plugin_files)
    expected = _expected_source_files()
    missing = expected - actual
    extra = actual - expected
    assert not missing and not extra, (
        f"_plugin_files drift — missing={sorted(missing)}, extra={sorted(extra)}. "
        "Add the new file to core_api/routes/plugin.py _plugin_files (and to "
        "the bash srcfile loop) so fresh plugin installs can download it."
    )


def test_install_script_srcfile_loop_matches_plugin_src():
    """The bash `for srcfile in …` loop in the install script template must match."""
    src = Path(plugin_mod.__file__).read_text(encoding="utf-8")
    match = re.search(r"for srcfile in\s+([^;]+);", src)
    assert match, (
        "Could not find the `for srcfile in …` loop in plugin.py — if you "
        "renamed the install-script template, update this test's regex too."
    )
    loop_files = set(match.group(1).split())
    expected = _expected_source_files()
    missing = expected - loop_files
    extra = loop_files - expected
    assert not missing and not extra, (
        f"install-script srcfile loop drift — missing={sorted(missing)}, "
        f"extra={sorted(extra)}. Any .ts file in plugin/src/ must appear here "
        "or `npm run build` on the target VM will fail with TS2307."
    )


def test_python_and_bash_lists_agree():
    """Keep the two hardcoded lists in lockstep with each other."""
    src = Path(plugin_mod.__file__).read_text(encoding="utf-8")
    match = re.search(r"for srcfile in\s+([^;]+);", src)
    assert match
    loop_files = set(match.group(1).split())
    python_files = set(plugin_mod._plugin_files)
    assert loop_files == python_files, (
        f"_plugin_files and install-script loop disagree — "
        f"only-in-python={sorted(python_files - loop_files)}, "
        f"only-in-bash={sorted(loop_files - python_files)}."
    )
