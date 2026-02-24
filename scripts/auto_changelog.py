#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import subprocess
from datetime import date
from pathlib import Path

CHANGELOG_PATH = Path("CHANGELOG.md")
UNRELEASED_HEADER = "## [Unreleased]"
CHANGED_HEADER = "### Changed"


def _run_git(args: list[str]) -> str:
    proc = subprocess.run(["git", *args], check=True, capture_output=True, text=True)
    return proc.stdout.strip()


def _get_staged_files() -> list[str]:
    out = _run_git(["diff", "--cached", "--name-only", "--diff-filter=ACMR"])
    files = [line.strip() for line in out.splitlines() if line.strip()]
    return files


def _ensure_changelog_skeleton() -> str:
    if CHANGELOG_PATH.exists():
        return CHANGELOG_PATH.read_text(encoding="utf-8")

    today = date.today().isoformat()
    content = (
        "# CHANGELOG\n\n"
        f"{UNRELEASED_HEADER} - {today}\n\n"
        "### Added\n"
        "- N/A\n\n"
        "### Changed\n"
        "- N/A\n\n"
        "### Fixed\n"
        "- N/A\n\n"
        "### Docs\n"
        "- N/A\n"
    )
    CHANGELOG_PATH.write_text(content, encoding="utf-8")
    return content


def _find_section_bounds(lines: list[str], section_header: str) -> tuple[int, int]:
    start = -1
    end = len(lines)
    for idx, line in enumerate(lines):
        if line.strip().startswith(section_header):
            start = idx
            break
    if start == -1:
        return -1, -1
    for idx in range(start + 1, len(lines)):
        if lines[idx].startswith("## "):
            end = idx
            break
    return start, end


def _insert_changed_entry(text: str, entry: str, fingerprint: str) -> str:
    lines = text.splitlines()

    unreleased_start, unreleased_end = _find_section_bounds(lines, UNRELEASED_HEADER)
    if unreleased_start == -1:
        today = date.today().isoformat()
        prepend = [
            f"{UNRELEASED_HEADER} - {today}",
            "",
            "### Added",
            "- N/A",
            "",
            "### Changed",
            "- N/A",
            "",
            "### Fixed",
            "- N/A",
            "",
            "### Docs",
            "- N/A",
            "",
        ]
        if lines and lines[0].startswith("# "):
            lines = [lines[0], ""] + prepend + lines[1:]
        else:
            lines = ["# CHANGELOG", ""] + prepend + lines
        unreleased_start, unreleased_end = _find_section_bounds(lines, UNRELEASED_HEADER)

    unreleased_block = "\n".join(lines[unreleased_start:unreleased_end])
    if f"[id:{fingerprint}]" in unreleased_block:
        return text

    changed_start = -1
    changed_end = unreleased_end
    for idx in range(unreleased_start + 1, unreleased_end):
        if lines[idx].strip() == CHANGED_HEADER:
            changed_start = idx
            break
    if changed_start == -1:
        insert_at = unreleased_end
        for idx in range(unreleased_start + 1, unreleased_end):
            if lines[idx].strip() == "### Added":
                insert_at = idx + 1
                while insert_at < unreleased_end and lines[insert_at].strip().startswith("-"):
                    insert_at += 1
                break
        lines = lines[:insert_at] + ["", CHANGED_HEADER, "- N/A"] + lines[insert_at:]
        unreleased_start, unreleased_end = _find_section_bounds(lines, UNRELEASED_HEADER)
        for idx in range(unreleased_start + 1, unreleased_end):
            if lines[idx].strip() == CHANGED_HEADER:
                changed_start = idx
                break
    if changed_start == -1:
        return "\n".join(lines).rstrip() + "\n"

    for idx in range(changed_start + 1, unreleased_end):
        if lines[idx].startswith("### "):
            changed_end = idx
            break

    block_lines = lines[changed_start + 1 : changed_end]
    block_lines = [line for line in block_lines if line.strip() != "- N/A"]

    block_lines = [entry] + block_lines
    lines = lines[: changed_start + 1] + block_lines + lines[changed_end:]
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    staged = _get_staged_files()
    meaningful = [path for path in staged if path != "CHANGELOG.md"]
    if not meaningful:
        return 0

    digest_input = "\n".join(sorted(meaningful)).encode("utf-8")
    fingerprint = hashlib.sha1(digest_input).hexdigest()[:10]
    files_short = ", ".join(sorted(meaningful)[:6])
    if len(meaningful) > 6:
        files_short += f", ... (+{len(meaningful) - 6})"
    entry = f"- Auto: updated {files_short} [id:{fingerprint}]"

    current = _ensure_changelog_skeleton()
    updated = _insert_changed_entry(current, entry, fingerprint)
    if updated != current:
        CHANGELOG_PATH.write_text(updated, encoding="utf-8")
        _run_git(["add", str(CHANGELOG_PATH)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
