#!/usr/bin/env python3
"""Increment the patch digit of ``__version__`` and report the new value.

Driven by ``.github/workflows/release.yml``. Every push to master releases a new
``x.y.PATCH`` so that ``pip install git+...@master`` always sees a *different*
version and actually reinstalls -- pip skips the reinstall when the resolved
version matches what is already in site-packages, which is how a branch install
silently goes stale.

Only the last digit ever moves. Major and minor are edited by hand.

    python scripts/bump_version.py --dry-run    # print, change nothing
    python scripts/bump_version.py              # rewrite the version file
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VERSION_FILE = REPO_ROOT / "magic_llm" / "__init__.py"

# Captures only the quoted value, so any trailing comment on the line survives.
VERSION_RE = re.compile(
    r"""(?P<prefix>^__version__\s*=\s*)(?P<q>['"])(?P<version>[^'"]+)(?P=q)""",
    re.MULTILINE,
)


def fail(message: str) -> "typing.NoReturn":  # noqa: F821 - runtime-only annotation
    sys.exit(f"bump_version: {message}")


def parse_patch_version(version: str) -> tuple[int, int, int]:
    parts = version.split(".")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        fail(f"__version__ = {version!r} is not a numeric x.y.z; fix it by hand")
    major, minor, patch = (int(part) for part in parts)
    return major, minor, patch


def tag_exists(tag: str) -> bool:
    completed = subprocess.run(
        ["git", "rev-parse", "-q", "--verify", f"refs/tags/{tag}"],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    return completed.returncode == 0


def next_free_version(major: int, minor: int, patch: int, skip_tag_check: bool) -> str:
    """First x.y.PATCH above the current one whose ``v`` tag is not taken.

    The tag check keeps a re-run, a manual tag, or a raced release from trying to
    push a tag that already exists (which would fail the whole job).
    """
    candidate = patch + 1
    while True:
        version = f"{major}.{minor}.{candidate}"
        if skip_tag_check or not tag_exists(f"v{version}"):
            return version
        candidate += 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the current and next version without touching the file",
    )
    parser.add_argument(
        "--no-tag-check",
        action="store_true",
        help="do not skip versions whose git tag already exists",
    )
    args = parser.parse_args()

    if not VERSION_FILE.exists():
        fail(f"{VERSION_FILE} does not exist")

    text = VERSION_FILE.read_text(encoding="utf-8")
    match = VERSION_RE.search(text)
    if match is None:
        fail(f"no top-level __version__ assignment in {VERSION_FILE}")

    current = match.group("version")
    new_version = next_free_version(*parse_patch_version(current), args.no_tag_check)

    if not args.dry_run:
        start, end = match.span()
        replacement = f"{match.group('prefix')}{match.group('q')}{new_version}{match.group('q')}"
        VERSION_FILE.write_text(text[:start] + replacement + text[end:], encoding="utf-8")

    print(f"current={current}")
    print(f"version={new_version}")

    # Hand both values to the workflow; `version` is what the tag and release use.
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as handle:
            handle.write(f"current={current}\n")
            handle.write(f"version={new_version}\n")
            handle.write(f"tag=v{new_version}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
