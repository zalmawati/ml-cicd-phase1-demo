# Phase 1 — CI/CD Fundamentals for ML (hands-on project)

This is a complete, working mini-project: a "churn prediction" training
script that gets containerized, linted, tested, and automatically deployed
via CI/CD — with the final trigger landing in **your existing Airflow**
instead of a human running `python train.py` by hand.

All code in this repo has already been run and verified (tests pass,
linters pass, the CLI produces a real model). Your job is to walk it
through git/GitHub and watch the automation work.

---

## 0. What you're building

```
 you push code
      │
      ▼
 ┌─────────────┐   PR opened    ┌────────────────────┐
 │  git branch │ ─────────────► │   GitHub Actions:   │
 └─────────────┘                │   lint + test (CI)  │
                                 └─────────┬───────────┘
                                           │ merge to main
                                           ▼
                                 ┌────────────────────────┐
                                 │  GitHub Actions:        │
                                 │  build Docker image     │
                                 │  push to ghcr.io        │
                                 └───────────┬─────────────┘
                                             │
                              you tag a release (v1.0.0)
                                             │
                                             ▼
                                 ┌────────────────────────┐
                                 │  GitHub Actions:        │
                                 │  call Airflow REST API  │
                                 └───────────┬─────────────┘
                                             │
                                             ▼
                                 ┌────────────────────────┐
                                 │  Airflow DAG run:       │
                                 │  pulls image, trains    │
                                 └────────────────────────┘
```

**Why this shape**: your team's problem is "ML code runs by someone typing
a command." This project fixes that in two stages — (1) every code change
is automatically linted, tested, and packaged, so nothing broken ever
reaches a runnable artifact, and (2) triggering a training run becomes a
git action (a tag push) instead of a person SSHing somewhere.

---

## 1. Prerequisites

- Git installed locally (`git --version`)
- A GitHub account (you have one — good)
- Docker installed locally, if you want to test image builds (optional for
  this phase, not required to follow along)
- Python 3.11 available locally
- Your existing Airflow instance, reachable over HTTP (for the last step)

---

## 2. Git & GitHub primer (since you're rusty)

You said you rarely use GitHub day-to-day — here's the exact command flow
we'll use, explained, not just listed.

| Command | What it actually does |
|---|---|
| `git init` | Turns the current folder into a git repo (only needed if you're not cloning) |
| `git clone <url>` | Downloads a repo you created on GitHub's website |
| `git checkout -b feature/my-change` | Creates and switches to a new branch — **always work on a branch, never commit straight to `main`** |
| `git add .` | Stages your changes for commit |
| `git commit -m "message"` | Saves a snapshot of staged changes with a message |
| `git push -u origin feature/my-change` | Uploads your branch to GitHub the first time (`-u` remembers the link for future plain `git push`) |
| Pull Request (on GitHub's website) | A request to merge your branch into `main` — this is what triggers CI on `pull_request` |
| `git tag v1.0.0` | Marks the current commit with a version label |
| `git push origin v1.0.0` | Pushes that tag to GitHub — this is what triggers the Airflow-trigger workflow |

The whole point of a PR-based workflow: CI runs on every PR *before* code
is allowed into `main`, so `main` is always in a known-good state.

---

## 3. Step-by-step

### Step 1 — Create the GitHub repo
1. Go to github.com → **New repository**.
2. Name it e.g. `ml-cicd-phase1-demo`. Keep it **private** if this is a
   learning sandbox tied to your work identity. Don't initialize with a
   README (you already have one).
3. Copy the remote URL it gives you (e.g. `git@github.com:you/ml-cicd-phase1-demo.git`).

### Step 2 — Push this project
From inside this folder on your laptop:
```bash
git init
git add .
git commit -m "Initial commit: phase 1 CI/CD demo project"
git branch -M main
git remote add origin <the-url-you-copied>
git push -u origin main
```
This first push goes straight to `main` since the repo is currently empty
— that's fine for the very first commit only. Everything after this goes
through a branch + PR.

### Step 3 — Set up your local dev environment
```bash
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -v                        # confirm all 8 tests pass locally, just like they did for me
```

### Step 4 — Install the pre-commit hooks
This catches formatting/lint issues *before* you even push, so CI failures
become rare instead of routine.
```bash
pre-commit install
pre-commit run --all-files       # first run reformats everything once
```

### Step 5 — Make a change on a branch and open a PR
```bash
git checkout -b feature/tweak-model-params
# edit src/train.py, e.g. change n_estimators to 300
git add .
git commit -m "Increase n_estimators to 300"
git push -u origin feature/tweak-model-params
```
Now go to GitHub → you'll see a banner offering to open a PR from that
branch. Open it. Within seconds you should see the **CI** workflow start
running under the PR's "Checks" tab — lint, then test.

### Step 6 — Watch CI, then merge
- If lint or test fails, the PR shows a red ❌ and you can click through to
  see exactly which line failed and why — fix locally, push again, CI
  re-runs automatically.
- Once both are green, merge the PR into `main`.
- Merging triggers the `push: branches: [main]` condition, which runs
  lint + test again on `main` itself, then **build-and-push** — this
  publishes a real Docker image to `ghcr.io/<your-username>/<repo-name>`.

### Step 7 — Check the published package
Go to your GitHub profile → **Packages** tab. You should see the container
image, tagged both `latest` and with the commit SHA. By default GHCR
packages are private to your account/org — that's fine for this exercise.

### Step 8 — Wire up the Airflow secrets
In your GitHub repo: **Settings → Secrets and variables → Actions → New
repository secret**. Add:
- `AIRFLOW_BASE_URL` — e.g. `https://your-airflow-host`
- `AIRFLOW_USERNAME`
- `AIRFLOW_PASSWORD`

(These never appear in logs — GitHub Actions masks secret values automatically.)

### Step 9 — Copy the DAG into your Airflow instance
Copy `dags/training_dag.py` into your Airflow DAGs folder, and edit the
`IMAGE_REPO` constant at the top to your actual `ghcr.io/<org>/<repo>` path.
Confirm `apache-airflow-providers-docker` is installed on your Airflow
workers, and that the worker has access to a Docker socket (or swap in
`KubernetesPodOperator` if your Airflow runs on K8s — same pattern).

### Step 10 — Tag a release and watch it fire
```bash
git checkout main
git pull
git tag v1.0.0
git push origin v1.0.0
```
This triggers `trigger-airflow.yml`, which calls your Airflow's REST API
to start a `churn_training_dag` run with `image_tag=v1.0.0` in its conf.
Check the Airflow UI — you should see a new DAG run appear, pulling the
exact image built from that tag and running training inside it.

---

## 4. What each file does

| File | Purpose |
|---|---|
| `src/data.py` | Generates synthetic data + validates it (a taste of the data-quality gates you'll build properly in Phase 4) |
| `src/train.py` | The actual training logic + CLI entrypoint |
| `tests/` | Unit tests — these are what CI runs on every PR |
| `Dockerfile` | Packages the training script into a runnable image |
| `.pre-commit-config.yaml` | Local hooks so lint issues get caught before you even push |
| `.github/workflows/ci.yml` | Lint → test → build & push image, gated so a broken PR never reaches an image |
| `.github/workflows/trigger-airflow.yml` | Tag push → calls your Airflow REST API |
| `dags/training_dag.py` | The Airflow-side DAG that receives the trigger and runs the container |

---

## 5. Troubleshooting

- **CI fails on `black --check`**: run `black src tests` locally, commit
  the reformatted files, push again.
- **`build-and-push` job doesn't run**: it only runs on a push to `main`
  (not on PRs) — check you actually merged, not just opened the PR.
- **GHCR push fails with permission error**: check repo **Settings →
  Actions → General → Workflow permissions** is set to "Read and write
  permissions."
- **Airflow REST API call returns 401**: double check the three secrets
  are set exactly as your Airflow auth expects (some setups use API keys
  instead of basic auth — adjust the `curl` command accordingly).

---

## 6. Exercises (stretch goals)

1. Add a **branch protection rule** on `main` requiring the CI checks to
   pass before merge is even allowed (Settings → Branches).
2. Add a `CODEOWNERS` file so PRs auto-request a review.
3. Make the version tag drive semantic versioning automatically instead of
   typing `v1.0.0` by hand (look into `python-semantic-release` or similar).
4. Modify `trigger-airflow.yml` to also post a Slack message when the DAG
   is triggered.

Once you're comfortable with this flow, Phase 2 (experiment tracking with
MLflow) plugs directly into `src/train.py` — you'll just add
`mlflow.log_metric(...)` calls where the metrics dict is built.
