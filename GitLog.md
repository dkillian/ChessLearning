# Git Log — Chess Learning

A running record of git operations, commands, and decisions for this project.
Append a new entry each session that involves meaningful git activity.

---

## Session 10 — October 7, 2026

### Context

First serious push attempt after many sessions of local-only commits.
The database file (`data/chess_learning.db`, 168 MB) had been committed since
the beginning of the project and was blocking GitHub's 100 MB file size limit.

### What was in .gitignore before

```
.Rproj.user
.Rhistory
.RData
.Ruserdata
.positai
```

### Problem: database too large to push

```
remote: error: File data/chess_learning.db is 168.91 MB;
this exceeds GitHub's file size limit of 100.00 MB
```

### Resolution

**Step 1: Update `.gitignore`**

Added the following entries:

```
# Database (too large for GitHub)
data/chess_learning.db

# Python cache
scripts/python/__pycache__/

# Rendered HTML (built from .qmd source)
scripts/r/*.html
scripts/r/*_files/

# Scratch files
nul
```

**Step 2: Untrack the files from the current index**

```bash
git rm --cached data/chess_learning.db scripts/python/__pycache__/engine.cpython-312.pyc
```

**Step 3: Stage and commit everything**

```bash
git add -A
git commit -m "Session 10: softmax notebook, Stockfish integration, game viewer analysis panel, 6 new R scripts"
```

**Step 4: Push failed** — database was still in commit history (committed in prior sessions under generic "d" commit messages).

Git history containing the DB:
```
b3ea5ce  Session 10 commit
9080ed4  d
75fcf96  d
4096f8f  d
3113037  d
2de660c  d
fcb7c76  d
0eb90bd  d          ← first commit with DB
b3d9470  Initial commit
```

**Step 5: Purge DB from all history using `filter-branch`**

`git filter-repo` was not installed, so used the legacy `git filter-branch`:

```bash
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch data/chess_learning.db scripts/python/__pycache__/engine.cpython-312.pyc" \
  --prune-empty --tag-name-filter cat -- --all
```

Output confirmed file removed from all 9 commits. History was rewritten.

**Step 6: Force push rewritten history**

```bash
git push --force origin main
```

Succeeded. Remote is now clean.

### Outcome

- Database removed from all commits — git history is clean
- `.gitignore` updated to prevent future commits of DB, cache, rendered HTML
- Local database remains intact on disk (OneDrive) — only excluded from git
- Future pushes will be fast (no large files)

### Notes

- `git filter-repo` is the preferred modern tool for history rewriting — consider installing via `pip install git-filter-repo` for future use
- Force-pushing rewrites remote history — safe for a personal single-developer repo, but would require coordination on a shared repo
- The database is the source of truth for training runs; it should be backed up via OneDrive, not git

---

## Commit History

| Hash | Date | Message |
|---|---|---|
| `ce5e426` | 2026-10-07 | Session 10: softmax notebook, Stockfish integration, game viewer analysis panel, 6 new R scripts (post-filter-branch SHA) |
| `b3d9470` | earlier | Initial commit |

*(Intermediate "d" commits were rewritten by filter-branch and their SHAs changed.)*
