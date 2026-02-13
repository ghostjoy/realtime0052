#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path


def _agents_template(project_name: str) -> str:
    return f"""# AGENTS.md

This file defines how coding agents should work in this repository.

## Project
- Name: `{project_name}`
- Primary goal: fill in this section with product and business context.

## Required Reading Order
1. `README.md`
2. `PROJECT_CONTEXT.md`
3. `CHANGELOG.md`
4. Main app entrypoints and tests

## Working Rules
- Keep changes minimal and focused.
- Prefer root-cause fixes.
- Keep docs in sync with behavior changes.
- Validate with tests or targeted checks before handoff.

## Validation
- Run the narrowest tests first.
- Expand to broader checks once local changes look correct.
"""


def _project_context_template(project_name: str) -> str:
    return f"""# PROJECT_CONTEXT.md

This file is a quick context handoff for humans and LLMs.

## Project Summary
- Project: `{project_name}`
- What it does:
  - TODO
- Who uses it:
  - TODO

## Architecture
- Runtime / framework:
  - TODO
- Core modules:
  - TODO
- Data storage:
  - TODO

## Key Decisions
- TODO: record important tradeoffs and why they were chosen.

## Known Risks / Constraints
- TODO

## Operational Notes
- Environment variables:
  - TODO
- Startup command:
  - TODO
- Test command:
  - TODO
"""


def _changelog_template() -> str:
    today = date.today().isoformat()
    return f"""# CHANGELOG

## [Unreleased] - {today}

### Added
- Initial changelog scaffold.

### Changed
- N/A

### Fixed
- N/A

### Docs
- N/A
"""


def _prompt_templates() -> str:
    return """# PROMPT_TEMPLATES.md

Reusable prompt templates for this repository.

## 1) Implement a feature
```
Read AGENTS.md and PROJECT_CONTEXT.md first.
Implement: <feature>.
Constraints: <constraints>.
Validation: run relevant tests.
Update CHANGELOG.md with Unreleased entry.
```

## 2) Fix a bug
```
Read AGENTS.md and PROJECT_CONTEXT.md first.
Bug: <symptom>.
Expected: <expected behavior>.
Please identify root cause, implement a minimal fix, and validate with tests.
Update CHANGELOG.md.
```

## 3) Improve docs
```
Please update README.md / PROJECT_CONTEXT.md / CHANGELOG.md to match current behavior.
Keep descriptions concise and accurate.
```

## 4) New-machine quick start
```
Read README.md, AGENTS.md, PROJECT_CONTEXT.md, and CHANGELOG.md.
Summarize:
1) what this repo does
2) current priorities
3) potential risks
4) suggested next steps
```
"""


def _write_if_needed(path: Path, content: str, force: bool) -> str:
    if path.exists() and not force:
        return "skipped"
    path.write_text(content, encoding="utf-8")
    return "updated" if path.exists() else "created"


def _ensure_readme_links(readme_path: Path) -> str:
    if not readme_path.exists():
        return "missing"

    bullets = [
        "- Agent rules: `AGENTS.md`",
        "- Project context: `PROJECT_CONTEXT.md`",
        "- Prompt templates: `PROMPT_TEMPLATES.md`",
        "- Changelog: `CHANGELOG.md`",
    ]

    text = readme_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    try:
        header_idx = next(i for i, line in enumerate(lines) if line.strip() == "## 相關文件")
    except StopIteration:
        block = ["", "## 相關文件", ""] + bullets + [""]
        readme_path.write_text(text.rstrip() + "\n" + "\n".join(block) + "\n", encoding="utf-8")
        return "section_added"

    section_end = len(lines)
    for i in range(header_idx + 1, len(lines)):
        if lines[i].startswith("## "):
            section_end = i
            break

    section_text = "\n".join(lines[header_idx:section_end])
    missing = [item for item in bullets if item not in section_text]
    if not missing:
        return "unchanged"

    new_lines = lines[:section_end] + missing + lines[section_end:]
    readme_path.write_text("\n".join(new_lines).rstrip() + "\n", encoding="utf-8")
    return "updated"


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap repo docs for faster LLM onboarding.")
    parser.add_argument("--target", default=".", help="Target repository directory. Default: current directory.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing template files.")
    parser.add_argument("--skip-readme-links", action="store_true", help="Do not edit README.md links section.")
    args = parser.parse_args()

    target = Path(args.target).resolve()
    target.mkdir(parents=True, exist_ok=True)
    project_name = target.name

    file_states: dict[str, str] = {}
    file_states["AGENTS.md"] = _write_if_needed(target / "AGENTS.md", _agents_template(project_name), args.force)
    file_states["PROJECT_CONTEXT.md"] = _write_if_needed(
        target / "PROJECT_CONTEXT.md", _project_context_template(project_name), args.force
    )
    file_states["CHANGELOG.md"] = _write_if_needed(target / "CHANGELOG.md", _changelog_template(), args.force)
    file_states["PROMPT_TEMPLATES.md"] = _write_if_needed(
        target / "PROMPT_TEMPLATES.md", _prompt_templates(), args.force
    )

    readme_state = "skipped"
    if not args.skip_readme_links:
        readme_state = _ensure_readme_links(target / "README.md")

    print(f"Bootstrapped repo docs in: {target}")
    for name, state in file_states.items():
        print(f"- {name}: {state}")
    print(f"- README.md links: {readme_state}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
