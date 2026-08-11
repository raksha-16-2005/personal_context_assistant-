#!/usr/bin/env python
"""Refuse to publish anything derived from private mail.

    python scripts/check_privacy.py            # check the working tree
    python scripts/check_privacy.py --staged   # check what is about to be committed

Run before the first push, and as a pre-push hook after that:

    ln -s ../../scripts/check_privacy.py .git/hooks/pre-push

`.gitignore` already covers `data/`, and that is not enough. It is a convention:
`git add -f` defeats it, a moved path escapes it, and a Gmail index written to a
directory nobody thought to ignore is tracked by default. The failure mode is
unrecoverable - once private mail is in a public commit or on a Hugging Face Space,
rewriting history does not un-publish it - so this checks for the actual artifacts
rather than trusting the ignore rules.

What it looks for:

  * tracked files that are corpora, indices, vectors, tokens or LLM caches
  * Gmail-derived artifacts anywhere in the tree, tracked or not, since those
    should never exist inside the repo at all
  * OAuth tokens and API keys in tracked content
  * a Space payload (spaces/) carrying a non-Enron index

Exit code 1 means do not push.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# Artifacts that must never be tracked, whatever the ignore rules say.
FORBIDDEN_TRACKED = [
    (re.compile(r"(^|/)\.env$"), "API keys"),
    (re.compile(r"\.npy$"), "embedding vectors"),
    (re.compile(r"\.parquet$"), "corpus rows"),
    (re.compile(r"\.onnx$"), "exported model weights"),
    (re.compile(r"(^|/)llm_cache/"), "cached LLM prompts, which embed message bodies"),
    (re.compile(r"gmail_token|token\.json$"), "OAuth tokens"),
    (re.compile(r"(^|/)maildir/"), "raw corpus"),
    (re.compile(r"chunk_texts\.jsonl\.gz$"), "chunk text (a second copy of the corpus)"),
]

# Paths that indicate Gmail-derived data exists in the repo at all. Retrieval over
# a private mailbox is meant to happen with the index outside the repo.
GMAIL_ARTIFACTS = [
    "data/gmail",
    "data/interim/gmail.parquet",
    "data/index/gmail",
    "data/commitments/gmail",
    # The multi-tenant web app's default USER_INDEX_ROOT (webapp/app/config.py)
    # when no override is set - one directory per real user, each holding a
    # real mailbox's messages.parquet and index. Production points
    # USER_INDEX_ROOT outside the repo entirely; this only catches a local dev
    # run that forgot to.
    "data/index/users",
]

# Secrets in tracked content. Narrow on purpose: a broad "looks like a token" regex
# fires on test fixtures and trains people to pass `--force`.
SECRET_PATTERNS = [
    (re.compile(rb"1//[0-9A-Za-z_\-]{30,}"), "Google OAuth refresh token"),
    (re.compile(rb"ya29\.[0-9A-Za-z_\-]{20,}"), "Google OAuth access token"),
    (re.compile(rb"AIza[0-9A-Za-z_\-]{35}"), "Google API key"),
    (re.compile(rb"sk-ant-[0-9A-Za-z_\-]{20,}"), "Anthropic API key"),
    (re.compile(rb"gsk_[0-9A-Za-z]{40,}"), "Groq API key"),
]

TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".jsonl", ".yaml", ".yml", ".html",
                 ".ipynb", ".cfg", ".toml", ".env", ".example", ".sh", ""}


def tracked_files(staged: bool) -> list[str]:
    cmd = (["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"]
           if staged else ["git", "ls-files"])
    out = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return [line for line in out.stdout.splitlines() if line.strip()]


def check_tracked(files: list[str]) -> list[str]:
    problems = []
    for path in files:
        for pattern, why in FORBIDDEN_TRACKED:
            if pattern.search(path):
                problems.append(f"tracked {path}  ({why})")
    return problems


def check_gmail_artifacts(root: Path) -> list[str]:
    return [f"{p} exists - Gmail-derived data must live outside the repo"
            for p in GMAIL_ARTIFACTS if (root / p).exists()]


def check_secrets(root: Path, files: list[str]) -> list[str]:
    problems = []
    for name in files:
        path = root / name
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
            continue
        try:
            blob = path.read_bytes()
        except OSError:
            continue
        if len(blob) > 2_000_000:
            continue
        for pattern, why in SECRET_PATTERNS:
            if pattern.search(blob):
                problems.append(f"{name} contains what looks like a {why}")
    return problems


def check_space_payload(root: Path) -> list[str]:
    """A Space directory carrying an index must be carrying the Enron one."""
    problems = []
    space_data = root / "spaces" / "data"
    if not space_data.exists():
        return problems
    sample = space_data / "interim" / "sample.parquet"
    if sample.exists():
        try:
            import pyarrow.parquet as pq
            senders = pq.read_table(sample, columns=["sender"]).column("sender").to_pylist()
            if not any("@enron.com" in (s or "") for s in senders[:5000]):
                problems.append(
                    f"{sample} has no Enron senders - the public Space must be "
                    f"Enron-only")
        except Exception as exc:                       # noqa: BLE001
            problems.append(f"could not verify {sample} is the public corpus: {exc}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--staged", action="store_true",
                    help="check what is staged rather than the whole tree")
    ap.add_argument("--root", type=Path, default=Path("."))
    args = ap.parse_args()

    root = args.root.resolve()
    files = tracked_files(args.staged)
    scope = "staged" if args.staged else "tracked"
    print(f"checking {len(files)} {scope} file(s) in {root}")

    problems = (check_tracked(files)
                + check_gmail_artifacts(root)
                + check_secrets(root, files)
                + check_space_payload(root))

    if problems:
        print(f"\n{len(problems)} problem(s) - DO NOT PUSH:\n", file=sys.stderr)
        for p in problems:
            print(f"  ✗ {p}", file=sys.stderr)
        print("\nOnce private mail is in a public commit, rewriting history does "
              "not un-publish it.", file=sys.stderr)
        return 1

    print("\nclean: no corpora, vectors, tokens, caches or Gmail artifacts are "
          "tracked, and no secrets found in tracked content.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
