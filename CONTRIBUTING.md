# Contributing to Kynema-SGF

Welcome. This guide covers everything you need to start contributing to
Kynema-SGF as a new member of the group.

---

## Prerequisites

Make sure you have the following installed on your Linux machine:

- `git`
- A GitHub account (ask Prof. Sato to add you as a collaborator)

Configure Git with your name and email before doing anything else:

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

---

## 1. Clone the repository and initialize submodules

```bash
git clone https://github.com/yohei-sato-psi/psi-boil-kynema.git
cd psi-boil-kynema
git submodule update --init --recursive
```

The `--recursive` flag is required because Kynema-SGF depends on several
submodules (AMReX, AMReX-Hydro, Waves2AMR, googletest, doxygen-awesome-css).
This will take several minutes. Always run this step after cloning, and
again after pulling changes that update a submodule.

Verify the setup:

```bash
git submodule status   # all lines should start with a space, not + or -
git remote -v          # should show the GitHub URL
```

---

## 2. Create your personal branch

Before writing any code, create a personal branch named after yourself and
the current year-month:

```bash
git checkout main
git pull
git checkout -b firstname-YYYY-MM
```

For example, if your name is Mashu and you start in June 2026:

```bash
git checkout -b mashu-2026-06
```

Push your branch to the remote immediately so it is backed up from the start:

```bash
git push -u origin mashu-2026-06
```

---

## 3. Daily git workflow

### At the start of each day — sync with the latest changes

```bash
git fetch origin
git rebase origin/main
```

If there are conflicts, resolve them file by file, then:

```bash
git add [resolved files]
git rebase --continue
```

### After making changes — stage, commit, and push

```bash
git status            # overview of changed files
git diff              # review unstaged changes line by line
git diff --staged     # review staged changes before committing

git add src/path/to/changed_file.cpp
git commit -m "Short, clear description of what and why"
git push
```

Commit often — small, focused commits are easier to review and to revert
if something goes wrong.

---

## 4. Rules

### Never commit to `main`

All work happens on personal or feature branches. Direct commits to `main`
are not allowed. Changes reach `main` only through a pull request reviewed
by Prof. Sato.

### Write clear commit messages

A good commit message answers *what changed* and *why*, not just *how*.

```
# Good
Add latent heat source term to temperature equation for phase change

# Bad
fix
update code
changes
```

- Keep the first line under 72 characters.
- If more explanation is needed, leave a blank line after the first line
  and add a short paragraph below it.

### Update NOTES.md

After any meaningful change — a new feature, a bug fix, a test result, a
design decision — add a short entry to `NOTES.md` with the date and your
name. This keeps a human-readable log of what was done and why, separate
from the git history.

### Ask Prof. Sato before large changes

If you are considering a change that affects more than a single module,
alters a numerical scheme, changes the build system, or touches files
others are actively working on, discuss it with Prof. Sato first.
A short conversation beforehand saves a lot of rework.

---

## 5. Building the code

See `wrk/release/README.TXT` for the exact CMake flags used to build the
MPI-only and MPI+CUDA (GPU) executables. The release binaries are kept in
`wrk/release/`:

```
wrk/release/kynema_sgf-MPI    # MPI-only build
wrk/release/kynema_sgf-GPU    # MPI + CUDA build
```

---

## 6. Running a test case

Test input files are in `test/test_files/`. Each run is carried out in its
own subdirectory under `wrk/`. Record the exact run command in a
`README.TXT` inside the run directory, for example:

```
wrk/
  release/
    kynema_sgf-MPI
    kynema_sgf-GPU
  mashu_bubble_growth-MPI/
    README.TXT          <- record the run command here
    log-MPI.txt
    plt00000/
    plt00100/
```

---

## Getting help

- Read `NOTES.md` in the repository root for project overview and current status
- For code questions, use Claude Code: `cd psi-boil-kynema && claude`
