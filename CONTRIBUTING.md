# Contributing to psi-boil-kynema

## Prerequisites
Make sure you have the following installed on your Linux machine:
- `git`
- A GitHub account (ask Prof. Sato to add you as a collaborator)

---

## Step 1: Configure Git
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

---

## Step 2: Clone the repository
```bash
git clone https://github.com/yohei-sato-psi/psi-boil-kynema.git
cd psi-boil-kynema
```

---

## Step 3: Initialize submodules
This is essential — Kynema-SGF depends on AMReX and other libraries:
```bash
git submodule update --init --recursive
```
This will take several minutes as it downloads AMReX, AMReX-Hydro,
and other submodules.

---

## Step 4: Create your personal branch
Name your branch using the format: `firstname-year-month`
For example:
```bash
git checkout -b mashu-2026-06
git push -u origin mashu-2026-06
```

---

## Step 5: Verify your setup
```bash
# Check your branch
git branch

# Check remotes
git remote -v
```

---

## Daily workflow

### Before starting work — sync with latest changes
```bash
git fetch origin
git rebase origin/feature/phase-change-model
```

### After making changes — commit and push
```bash
git add [files you changed]
git commit -m "Brief description of what you did"
git push origin mashu-2026-06
```

---

## Important rules
1. **Never commit directly to `main`** — always work on your branch
2. **Always write a clear commit message** describing what you changed and why
3. **Update NOTES.md** at the end of each session with your progress
4. **Ask Prof. Sato** before making large structural changes to the code

---

## Getting help
- Read `NOTES.md` in the repository root for project overview and current status
- For code questions, use Claude Code: `cd psi-boil-kynema && claude`
