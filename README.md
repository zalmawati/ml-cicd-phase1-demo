# Phase 2 — Experiment Tracking & Versioning (MLflow + DVC)

Phase 1 made your training code *shippable*: tested, containerized, triggered by a
git tag. Phase 2 makes every result *explainable*. After this phase you can pick any
model and answer, with evidence instead of memory:

> **Which code, which data and which hyperparameters produced this model — and was it better than the last one?**

Everything here was run and verified against a real MLflow server and a real DVC
repo (19 unit tests pass, the sweep/registry/prediction flow works end to end,
DVC time-travel restores byte-identical data). The two things I could **not** run
for you are the Docker Compose stack and the Airflow DAG, since they need your machine;
Step 1 and Step 7 have explicit checks so you find out immediately if something is off.

---

## 1. The mental model (read this first)

Four tools, four different questions. Confusing them is the #1 beginner mistake.

| Tool | Answers the question | Stores |
|---|---|---|
| **Git** | *Which code?* | Source files, `params.yaml`, `dvc.yaml`, `dvc.lock` |
| **DVC** | *Which data? Can I rebuild it?* | Big files (data, model) in a cache/remote; tiny pointer files in git |
| **MLflow Tracking** | *What happened in each run?* | Params, metrics, tags, artifacts, the model |
| **MLflow Registry** | *Which model is live right now?* | Named models, numbered versions, movable aliases |

They are stitched together by **lineage tags** that every run records:

```
 registry alias  ──►  model version 3  ──►  MLflow run  ──►  tags: git_sha, image_tag,
 "champion"                                                        data_sha256, git_dirty
                                                                       │
                                     git checkout <git_sha>  ◄─────────┤
                                     dvc checkout  (restores exact data)◄┘
```

**Key MLflow vocabulary**

- **Experiment**: a folder of related runs (`churn-model`).
- **Run**: one execution of training. Holds *params* (inputs, e.g. `max_depth`), *metrics* (outputs, e.g. `roc_auc`), *tags* (free labels), *artifacts* (files).
- **Nested run**: child runs under a parent. We use them for hyperparameter sweeps.
- **Registered model / version / alias**: `churn-model` → `v1, v2, v3` → labels like `champion` that you can move. Consumers load `models:/churn-model@champion` and never hard-code a version. (Older tutorials use "stages" like Staging/Production; those are deprecated. Aliases replace them.)

**Key DVC vocabulary**

- **Pointer file / `dvc.lock`**: a small text file, committed to git, that records the hash of each data file. Git versions the *pointer*; DVC stores the *bytes* elsewhere.
- **Remote**: where the bytes live (a folder, S3, GCS...). `dvc push` / `dvc pull` sync it.
- **Pipeline (`dvc.yaml`)**: stages with declared inputs and outputs. `dvc repro` only reruns what changed.
- **`params.yaml`**: the single place where hyperparameters and dataset settings live. Both DVC and MLflow read it.

---

## 2. What changed compared to Phase 1

| File | Status | What it does |
|---|---|---|
| `src/tracking.py` | **new** | Connects to MLflow (with a fast health check) and computes lineage tags |
| `src/params.py`, `params.yaml` | **new** | One source of truth for settings |
| `src/make_dataset.py` | **new** | Writes `data/churn.csv` so DVC can version it |
| `src/sweep.py` | **new** | Grid search; every combination is a tracked nested run |
| `src/registry.py` | **new** | Register a `candidate`; promote to `champion` only if it beats the current one |
| `src/predict.py` | **new** | What a *consumer* of the model does (loads by alias) |
| `src/train.py` | **changed** | Now logs params, metrics, model, dataset, tags to MLflow |
| `src/data.py` | changed | Adds `label_noise` and `load_csv` |
| `dvc.yaml` | **new** | The two-stage data pipeline |
| `infra/mlflow/` | **new** | Docker Compose for the MLflow server (Postgres + artifact store) |
| `Dockerfile` | changed | Bakes the commit SHA in; no longer installs dev tools |
| `requirements.txt` / `requirements-dev.txt` | **split** | Runtime vs. dev/CI dependencies |
| `.github/workflows/ci.yml` | changed | Uses dev requirements, passes `GIT_SHA` to the build |
| `dags/training_dag.py` | changed | Passes MLflow URL + lineage env vars, registers a candidate |
| `tests/` | changed/new | Tracking, registry and training tests (no server needed) |

Your own edits from Phase 1 (e.g. `n_estimators`) now live in `params.yaml`
instead of being hard-coded in `train.py`.

---

## 3. Step-by-step

All commands are **PowerShell** from the root of your project folder.

### Step 0 — Work on a branch (Phase 1 habit)

```powershell
git checkout main
git pull
git checkout -b feature/phase2-tracking-versioning
```

Copy the files from this zip into your project. **Do not blindly overwrite** these two:

- `.gitignore`: merge in the new lines (`mlflow.db`, `mlruns/`, `mlartifacts/`) and keep your `mlopsvenv/` / `*venv*/` rules.
- `dags/training_dag.py`: this repo's copy is for reference; you'll install it into your *Airflow* project in Step 7.

Also delete `.github/workflows/trigger-airflow.yml` if it still exists in your project; its job now lives in `ci.yml`.

### Step 1 — Start the MLflow server

```powershell
cd infra\mlflow
docker compose up -d --build
docker compose ps
```

Wait until both services show **healthy** (about 30–60 s the first time). Then verify:

```powershell
Invoke-RestMethod http://127.0.0.1:5000/health     # expect: OK
```

Open **http://localhost:5000** in a browser. You should see an empty MLflow UI.

- *Port 5000 already used?* Change `"5000:5000"` to e.g. `"5001:5000"` in `docker-compose.yml` and use that port everywhere below.
- *What is this stack?* Postgres stores metadata (runs, params, registry). Model files go into a Docker volume, uploaded **over HTTP through the server** (`--serve-artifacts`). That's why a training container needs no shared folders and no cloud keys.

### Step 2 — Install dependencies and confirm everything is green

```powershell
cd ..\..                                # back to project root
.\mlopsvenv\Scripts\Activate.ps1        # your venv
pip install -r requirements-dev.txt
pip install dvc
pre-commit run --all-files
pytest -v
```

`pytest` builds a throw-away MLflow database per test, so it needs no server. Expect it
to take around a minute: creating a fresh MLflow database is not free. **All 19 tests should pass.**

### Step 3 — Your first tracked run

```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
python -m src.train
```

Refresh the UI → experiment **churn-model** → click the run. Explore:

- **Parameters** (`train.n_estimators`, ...) and **Metrics** (`roc_auc`, `f1`, `accuracy`).
- **Tags**: `git_sha`, `git_dirty`, `image_tag`, `data_path`, `triggered_by`. Make an uncommitted edit, train again, and watch `git_dirty` flip to `true`. **A `git_sha` with `git_dirty=true` cannot be used to reproduce that run.**
- **Artifacts**: `feature_importances.json`. Click it.
- **The model**: in MLflow 3, models are first-class objects stored separately from run artifacts; find them under the experiment's **Models** tab. Note the *signature* (expected input columns and types) and the input example.

`$env:MLFLOW_TRACKING_URI` only lasts for this PowerShell window. Without it, the code falls back to a local `mlflow.db` file, which is handy for quick experiments but invisible to the server.

### Step 4 — Hyperparameter sweep and run comparison

```powershell
python -m src.sweep
```

This runs 12 combinations (4 × 3), each a nested run under one parent `sweep`.
In the UI: open the experiment → tick several child runs → **Compare**. Try the
**parallel coordinates** plot: which hyperparameter actually moves `roc_auc`? Sort the run
table by `roc_auc` to find the winner, and open the parent run's `best_params.json`.

Query runs from code instead of clicking:

```powershell
python -c "import mlflow; mlflow.set_tracking_uri('http://127.0.0.1:5000'); print(mlflow.search_runs(experiment_names=['churn-model'], order_by=['metrics.roc_auc DESC'])[['run_id','params.train.max_depth','metrics.roc_auc']].head())"
```

*Why the sweep matters for your team:* "I tried a few things and this seemed best" becomes a table anyone can re-open.

### Step 5 — The model registry: candidate → champion

```powershell
python -m src.train --register-as churn-model     # creates a new version, alias: candidate
python -m src.registry promote --name churn-model # moves alias "champion" ONLY if it's better
python -m src.predict                             # loads models:/churn-model@champion
```

Look at **Models → churn-model** in the UI: versions, and the `candidate`/`champion` aliases.

**Do the experiment that teaches the lesson:** edit `params.yaml` to `n_estimators: 2` and
`max_depth: 1`, run `--register-as` again, then `promote`. The log says the candidate
**did not** beat the champion, and `predict` still uses the old good model. Restore your params afterwards.

The important design point: `predict.py` contains no version number. Promotion is a
one-line alias move, and every consumer picks it up instantly. Rollback is the same move in reverse.

### Step 6 — Data versioning with DVC

```powershell
dvc init
dvc remote add -d localremote D:\dvc-remote      # any folder; S3/GCS in real life
git add .dvc .dvcignore
git commit -m "Initialise DVC"
```

Build data v1 and record it:

```powershell
dvc repro                                         # make_dataset -> train
git add dvc.lock data\.gitignore outputs\.gitignore outputs\metrics.json params.yaml
git commit -m "Data v1: label_noise 0.05"
dvc push                                          # copy the actual bytes to the remote
```

`dvc repro` created `data\churn.csv` and trained on it. Note what git stores: the small
`dvc.lock` with hashes. The 2000-row CSV is **not** in git (DVC added it to `.gitignore`).

Now create **data v2**, a messier dataset. Edit `params.yaml` and change `label_noise` from `0.05` to `0.20`:

```powershell
dvc repro                                         # reruns BOTH stages: params changed
git add -A
git commit -m "Data v2: label_noise 0.20"
dvc push

dvc params  diff <commit-of-v1>                   # what changed in the settings
dvc metrics diff <commit-of-v1>                   # what that did to the results
```

(Find the v1 commit with `git log --oneline`.) You'll see `roc_auc` drop, and it's
now attributable to a specific, recorded change in the data.

**Time travel** to the exact v1 dataset:

```powershell
git checkout <commit-of-v1>
dvc checkout                                      # data\churn.csv is now byte-identical to v1
# ... inspect, retrain, compare ...
git checkout feature/phase2-tracking-versioning   # back to your branch
dvc checkout
```

**Simulate a new teammate's laptop** (no data, empty cache):

```powershell
Remove-Item -Recurse -Force .dvc\cache, data\churn.csv, outputs\model.joblib
dvc pull                                          # everything comes back from the remote
```

Finally, connect the tools. Every DVC-driven training run tags MLflow with `data_sha256`.
Open a run trained by `dvc repro` and compare that tag with:

```powershell
Get-FileHash data\churn.csv -Algorithm SHA256
```

They match (compare case-insensitively; PowerShell prints uppercase). The run *proves* which bytes it saw.

### Step 7 — Ship it through CI/CD (Phase 1 pipeline, now with tracking)

1. **Airflow Variable:** in the Airflow UI → *Admin → Variables* → add `mlflow_tracking_uri` = `http://host.docker.internal:5000`.
2. **DAG:** copy `dags/training_dag.py` into your Airflow project's `dags/` folder (its `IMAGE_REPO` is already set to your repo). Requires `apache-airflow-providers-docker` in your Airflow image, as before.
3. **Docker pull rights:** on the machine running Airflow, `docker login ghcr.io -u zalmawati` (token with `read:packages`), as in Phase 1.
4. **Keep running:** the MLflow stack (Step 1) and your self-hosted GitHub runner (`.\run.cmd`).
5. Commit, push the branch, open a PR, watch CI go green, merge.
6. Tag and watch the whole chain fire:

```powershell
git checkout main
git pull
git tag v1.1.0
git push origin v1.1.0
```

Expected: CI builds `ghcr.io/zalmawati/ml-cicd-phase1-demo:v1.1.0` → the runner triggers
Airflow → the DAG starts the container → **a new run appears in MLflow** with
`image_tag = v1.1.0`, `triggered_by = airflow` and the commit SHA, plus a new `candidate`
model version. No human ran `python`. Promote it with `python -m src.registry promote --name churn-model`.
(Automatic promotion gates are Phase 4.)

### Step 8 — The payoff: answer the lineage question

Pick the `champion` version in the UI and, using only what MLflow recorded, reproduce its inputs:

1. Models → `churn-model` → the champion version → open its **source run**.
2. Read the tags: `git_sha`, `git_dirty` (want `false`), `data_sha256`, `image_tag`.
3. `git checkout <git_sha>` then `dvc checkout`: you now have that exact code and data.
4. `Get-FileHash data\churn.csv` equals the `data_sha256` tag.

Note that a run started by Airflow trains on the *synthetic data generated inside the container* (there is no
DVC data in the image yet), so it has no `data_sha256`; its data is pinned by `params.yaml` in the image instead. Runs from
`dvc repro` carry the hash. Feeding DVC data into the container is Phase 4 territory.

If you can do this for a `dvc repro` run, Phase 2 is done.

---

## 4. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Cannot reach MLflow at http://127.0.0.1:...` | Server not running or wrong URL. Inside a Docker container `127.0.0.1` is the container: use `http://host.docker.internal:5000`. |
| HTTP **403 / "Invalid Host header"** from the container | MLflow 3.x rejects Host headers it doesn't know. The compose file already lists `host.docker.internal:*` via `--allowed-hosts`; if you use another hostname or IP, add it there and `docker compose up -d`. (I verified: without the flag, `host.docker.internal` gets a 403 on API calls.) |
| `untrusted types ... sklearn.tree._tree.Tree` | MLflow 3.x saves sklearn models with `skops` and refuses unknown types. `train.py` passes `skops_trusted_types` for exactly the one type a RandomForest needs. Using a different model type may need a different list; read the error, it names the types. |
| `Can not safely convert int64 to float64` from `predict` | Working as intended: the model **signature** rejects wrongly-typed input. Send floats for `tenure_months` / `monthly_spend`. |
| `dvc repro`: "output ... is git-ignored" | Don't put DVC outputs under an ignored folder. That's why the stage writes to `outputs/`, not `artifacts/`. |
| `dvc repro` says stage is up to date but you expected a rerun | DVC reruns only when a dependency, a watched param or the command changed. Edit `params.yaml` or `dvc repro -f`. |
| Airflow task can't pull the image | `docker login ghcr.io` on the Airflow host; image path must be lowercase. |
| Container run fails right at the end | Check the container log for MLflow 403/connect errors; tracking happens first, model registration last. |
| Tests take about a minute | Each MLflow test creates a fresh DB with migrations. Deliberate: it guarantees isolation. |

---

## 5. Exercises (harder, in increasing difficulty)

1. **Log a plot.** Add a confusion-matrix or ROC figure via `mlflow.log_figure` (needs `matplotlib`, already installed with MLflow). Find it in the run's artifacts.
2. **Stricter promotion.** Run `promote --min-gain 0.01`; then write a test proving a +0.005 candidate is rejected.
3. **A model that isn't a RandomForest.** Swap in `GradientBoostingClassifier`, run it, and hit the `skops` trust error yourself. Fix it by reading the message.
4. **Track the data shape.** Log row count, class balance and a hash of the column list as metrics/tags; use them to notice when data v2 differs from v1 *before* looking at accuracy.
5. **Real object storage.** Replace the artifact Docker volume with **MinIO** (S3-compatible) in the compose file; configure `--default-artifact-root s3://...`. This is how production MLflow is usually deployed.
6. **Authentication.** MLflow 3 supports basic auth (`--app-name basic-auth`). Enable it, then give the Airflow container credentials via `MLFLOW_TRACKING_USERNAME/PASSWORD`.
7. **Sweep in Airflow.** Add a second DAG that runs `src.sweep` on a schedule and one that runs `promote` afterwards. That is most of Phase 4's continuous training, and you will already understand every piece.

**Next: Phase 3**: serving the `champion` model behind an API, packaging, and Kubernetes basics.
