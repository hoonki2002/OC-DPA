#!/usr/bin/env python3
"""Export the public DPA-MoE tree from this private working repository.

    .venv/bin/python tools/export_public.py OUT_DIR [--commit] [--version 1.0.0]

Copies an allowlist of tracked files into OUT_DIR, rewrites absolute
workspace paths in result tables to repo-relative ones, points the
llama.cpp submodule at the public fork, and fails if anything private
(local paths, names, tokens, old project names) remains. With --commit,
OUT_DIR becomes a fresh git repository with a single release commit.

Nothing is pushed; publish OUT_DIR by hand after reviewing it.
"""
from __future__ import annotations

import argparse
import fnmatch
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
SUBMODULE = "third_party/llama.cpp"
PUBLIC_FORK = "https://github.com/kimGreg/llama.cpp.git"
PUBLIC_FORK_BRANCH = "dpa-moe"

# Tracked paths copied to the public tree (fnmatch patterns, matched in order;
# the first matching rule wins).
RULES: list[tuple[str, bool]] = [
    ("tools/*", False),                         # maintainer tooling stays private
    ("experiments/results/*.log", False),       # raw stdout dumps
    ("experiments/results/*.bin", False),       # raw llama-perplexity outputs
    # Per-task run records and generated HTML reports record the original
    # (pre-rename) binary paths and env var names; the same numbers are in
    # the per-config summary.csv files, which are exported.
    ("experiments/results/*/bench/*/llama_mc/*.json", False),
    ("experiments/results/*.html", False),
    ("README.md", True), ("LICENSE", True), ("NOTICE", True),
    ("pyproject.toml", True), ("uv.lock", True), (".python-version", True),
    (".gitignore", True),
    ("src/*", True),
    ("tests/*", True),
    ("experiments/*", True),
    ("third_party/anybcq/*", True),
]

# Absolute workspace prefixes in result files → repo-relative paths.
PATH_REWRITE = re.compile(r"/home/[^/\s\"',]+/workspace/[^/\s\"',]+/")

LEAK_PATTERNS = {
    "absolute home path": re.compile(r"/home/[A-Za-z0-9_.-]+"),
    "personal name/handle": re.compile(r"jaeyun|kimjy|kimgreg/(?!llama\.cpp|DPA-MoE\b)", re.I),
    "organization": re.compile(r"makinarocks|kgtis|add\.re\.kr", re.I),
    "review-period wording": re.compile(r"emnlp|anonymi[sz]", re.I),
    "old project name": re.compile(r"streamllm|\bDP_MOE|\bdp_moe\b|\bDP-MoE\b|\bDPMoE\b", re.I),
    "token-like secret": re.compile(r"hf_[A-Za-z0-9]{30,}|ghp_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{30,}"),
}
# Files allowed to mention the original author (license/citation metadata).
LEAK_ALLOWLIST = {"README.md": {"personal name/handle"}}


def git(*args: str, cwd: pathlib.Path = REPO) -> str:
    return subprocess.run(["git", "-C", str(cwd), *args], check=True,
                          capture_output=True, text=True).stdout


def selected(path: str) -> bool:
    for pattern, keep in RULES:
        if fnmatch.fnmatch(path, pattern):
            return keep
    return False


def is_text(data: bytes) -> bool:
    if b"\0" in data[:8192]:
        return False
    try:
        data.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def export(out: pathlib.Path) -> list[str]:
    files = [f for f in git("ls-files", "-z").split("\0") if f and f != SUBMODULE]
    copied = []
    for f in files:
        if not selected(f):
            continue
        src, dst = REPO / f, out / f
        data = src.read_bytes()
        if f.startswith("experiments/results/") and is_text(data):
            data = PATH_REWRITE.sub("", data.decode("utf-8")).encode("utf-8")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        if src.stat().st_mode & 0o111:
            dst.chmod(0o755)
        copied.append(f)
    (out / ".gitmodules").write_text(
        f'[submodule "{SUBMODULE}"]\n\tpath = {SUBMODULE}\n'
        f"\turl = {PUBLIC_FORK}\n\tbranch = {PUBLIC_FORK_BRANCH}\n")
    return copied


def scan(out: pathlib.Path) -> list[str]:
    problems = []
    for p in sorted(out.rglob("*")):
        if not p.is_file() or ".git" in p.relative_to(out).parts:
            continue
        rel = str(p.relative_to(out))
        data = p.read_bytes()
        if not is_text(data):
            continue
        text = data.decode("utf-8")
        for label, pattern in LEAK_PATTERNS.items():
            if label in LEAK_ALLOWLIST.get(rel, set()):
                continue
            for m in pattern.finditer(text):
                line = text.count("\n", 0, m.start()) + 1
                problems.append(f"{rel}:{line}: {label}: {m.group(0)!r}")
    return problems


def submodule_commit() -> str:
    return git("rev-parse", "HEAD", cwd=REPO / SUBMODULE).strip()


def check_public_fork(sha: str) -> str | None:
    heads = git("ls-remote", PUBLIC_FORK, f"refs/heads/{PUBLIC_FORK_BRANCH}").split()
    if not heads:
        return f"{PUBLIC_FORK} has no branch {PUBLIC_FORK_BRANCH}"
    if heads[0] != sha:
        return (f"{PUBLIC_FORK}@{PUBLIC_FORK_BRANCH} is {heads[0][:9]}, but the submodule "
                f"pins {sha[:9]}; push the fork branch before publishing")
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=pathlib.Path)
    ap.add_argument("--commit", action="store_true",
                    help="initialise OUT as a git repo with one release commit")
    ap.add_argument("--version", default="1.0.0")
    args = ap.parse_args()

    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} exists and is not empty")
    out.mkdir(parents=True, exist_ok=True)

    copied = export(out)
    print(f"[export] {len(copied)} files -> {out}")

    problems = scan(out)
    if problems:
        print(f"[export] {len(problems)} leak-check findings:")
        print("\n".join("  " + p for p in problems[:200]))
        return 1
    print("[export] leak check: clean")

    sha = submodule_commit()
    fork_issue = check_public_fork(sha)
    print(f"[export] submodule {SUBMODULE} -> {PUBLIC_FORK}@{sha[:9]}")
    if fork_issue:
        print(f"[export] WARNING: {fork_issue}")

    if args.commit:
        git("init", "-q", "-b", "main", cwd=out)
        git("add", "-A", cwd=out)
        git("update-index", "--add", "--cacheinfo", f"160000,{sha},{SUBMODULE}", cwd=out)
        git("commit", "-q", "-m", f"DPA-MoE v{args.version}", cwd=out)
        print(f"[export] committed: {git('log', '--oneline', '-1', cwd=out).strip()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
