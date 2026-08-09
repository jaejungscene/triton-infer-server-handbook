"""Repository-local Markdown structure and link validation."""

import re
from pathlib import Path
from urllib.parse import unquote


MARKDOWN_LINK = re.compile(r"!?\[[^]]*\]\(([^)]+)\)")
EXTERNAL_SCHEMES = ("http://", "https://", "mailto:")


def _markdown_files(project_root):
    root = Path(project_root)
    return sorted(
        path
        for path in root.rglob("*.md")
        if ".git" not in path.parts and ".venv" not in path.parts
    )


def test_local_markdown_links_resolve(project_root):
    failures = []
    for markdown_path in _markdown_files(project_root):
        content = markdown_path.read_text(encoding="utf-8")
        for raw_target in MARKDOWN_LINK.findall(content):
            target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
            if not target or target.startswith(("#", *EXTERNAL_SCHEMES)):
                continue
            relative_path = unquote(target.split("#", 1)[0])
            resolved = (markdown_path.parent / relative_path).resolve()
            if not resolved.exists():
                failures.append(
                    f"{markdown_path.relative_to(project_root)} -> {target}"
                )

    assert not failures, "Broken local Markdown links:\n" + "\n".join(failures)


def test_markdown_code_fences_are_closed(project_root):
    failures = []
    for markdown_path in _markdown_files(project_root):
        opened = None
        for line_number, line in enumerate(
            markdown_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            stripped = line.lstrip()
            match = re.match(r"(`{3,}|~{3,})", stripped)
            if match is None:
                continue
            fence = match.group(1)
            if opened is None:
                opened = (fence[0], len(fence), line_number)
            elif fence[0] == opened[0] and len(fence) >= opened[1]:
                opened = None
        if opened is not None:
            failures.append(
                f"{markdown_path.relative_to(project_root)}:{opened[2]}"
            )

    assert not failures, "Unclosed Markdown fences:\n" + "\n".join(failures)
